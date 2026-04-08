use crate::GlyphInstance;
use crate::glyph_cache::{
    ColorGlyphEntry, ColorGlyphLayer, ColorV1GlyphEntry, GlyphKey,
    COLOR_V1_VECTOR_GLYPH, COLOR_VECTOR_GLYPH, NON_VECTOR_GLYPH,
};
use crate::outline::{ColorGlyphInfo, extract_color_info, extract_outline};
use crate::prepare::apply_italic_shear;
use crate::raster_text::{NonVectorGlyph, RasterVertex};
use crate::text_atlas::TextAtlas;
use crate::types::{PrepareError, RenderError, TextArea};
use crate::viewport::Viewport;

use rustc_hash::FxHashMap;
use skrifa::setting::VariationSetting;

use std::sync::Arc;
use wgpu::{
    Buffer, BufferDescriptor, BufferUsages, COPY_BUFFER_ALIGNMENT, CommandEncoder,
    DepthStencilState, Device, MultisampleState, Queue, RenderPass, RenderPipeline,
};

use crate::types::TextBounds;

/// Cached per-font data to avoid re-parsing font tables on every glyph miss.
struct CachedFont {
    font: Arc<cosmic_text::Font>,
    face_index: u32,
    units_per_em: f32,
    has_colr: bool,
}

/// Cached prepared output for a single TextArea. Reusable when the text
/// content, styling, and atlas state haven't changed.
struct CachedTextArea {
    left: f32,
    top: f32,
    scale: f32,
    bounds: TextBounds,
    default_color: cosmic_text::Color,
    atlas_generation: u32,
    instances: Vec<GlyphInstance>,
    distinct_keys: Vec<GlyphKey>,
    non_vector_glyphs: Vec<NonVectorGlyph>,
}

/// A text renderer that uses the Slug algorithm to render text into an
/// existing render pass.
pub struct TextRenderer {
    vertex_buffer: Buffer,
    vertex_buffer_size: u64,
    pipeline: RenderPipeline,
    instances: Vec<GlyphInstance>,
    glyphs_to_render: u32,
    /// Per-font cache: avoids db().face(), get_font(), and FontRef parsing per miss.
    font_cache: FxHashMap<(cosmic_text::fontdb::ID, cosmic_text::Weight), CachedFont>,
    /// Per-TextArea retained cache, keyed by buffer pointer.
    text_area_cache: FxHashMap<*const cosmic_text::Buffer, CachedTextArea>,
    /// Resolution from last frame, for cache invalidation.
    cached_resolution: crate::types::Resolution,
    /// Atlas generation at last prepare() — detects trim(reset) between prepare and render.
    prepared_atlas_generation: u32,
    // Raster fallback: per-frame instances drawn using TextAtlas's shared raster resources
    raster_instances: Vec<RasterVertex>,
    raster_vertex_buffer: Buffer,
    raster_vertex_buffer_size: u64,
    raster_glyphs_to_render: u32,
}

impl TextRenderer {
    pub fn new(
        atlas: &mut TextAtlas,
        device: &Device,
        multisample: MultisampleState,
        depth_stencil: Option<DepthStencilState>,
    ) -> Self {
        let vertex_buffer_size = next_copy_buffer_size(4096);
        let vertex_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("sluggrs vertices"),
            size: vertex_buffer_size,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline = atlas.get_or_create_pipeline(device, multisample, depth_stencil.clone());

        atlas.init_raster(device, depth_stencil, multisample);

        let raster_vertex_buffer_size = 4096u64;
        let raster_vertex_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("sluggrs raster vertices"),
            size: raster_vertex_buffer_size,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            vertex_buffer,
            vertex_buffer_size,
            pipeline,
            instances: Vec::new(),
            glyphs_to_render: 0,
            font_cache: FxHashMap::default(),
            text_area_cache: FxHashMap::default(),
            cached_resolution: crate::types::Resolution {
                width: 0,
                height: 0,
            },
            prepared_atlas_generation: 0,
            raster_instances: Vec::new(),
            raster_vertex_buffer,
            raster_vertex_buffer_size,
            raster_glyphs_to_render: 0,
        }
    }

    /// Prepare text areas for rendering, with per-glyph depth mapping.
    ///
    /// `encoder` and `cache` are unused — they exist for cryoglyph API
    /// compatibility. sluggrs uses `queue.write_texture` (no encoder needed)
    /// and extracts outlines via skrifa (no swash rasterization).
    #[allow(clippy::too_many_arguments)]
    #[hotpath::measure]
    pub fn prepare_with_depth<'a>(
        &mut self,
        device: &Device,
        queue: &Queue,
        _encoder: &CommandEncoder,
        font_system: &mut cosmic_text::FontSystem,
        atlas: &mut TextAtlas,
        viewport: &Viewport,
        text_areas: impl IntoIterator<Item = TextArea<'a>>,
        _cache: &mut cosmic_text::SwashCache,
        mut metadata_to_depth: impl FnMut(usize) -> f32,
    ) -> Result<(), PrepareError> {
        self.instances.clear();
        let mut non_vector_collector: Vec<NonVectorGlyph> = Vec::new();

        let resolution = viewport.resolution();
        let scroll = viewport.scroll_offset();
        let atlas_gen = atlas.generation();

        // Invalidate all cached entries if resolution changed
        if resolution != self.cached_resolution {
            self.text_area_cache.clear();
            self.cached_resolution = resolution;
        }

        let mut all_hit = true;
        let mut any_position_changed = false;
        // Track which cache entries were used this frame for cleanup
        let mut used_ptrs: Vec<*const cosmic_text::Buffer> = Vec::new();

        for text_area in text_areas {
            let buffer_ptr: *const cosmic_text::Buffer = text_area.buffer;
            used_ptrs.push(buffer_ptr);

            // Try cache hit
            if let Some(cached) = self.text_area_cache.get(&buffer_ptr) {
                if !text_area.buffer.redraw()
                    && cached.scale == text_area.scale
                    && cached.bounds == text_area.bounds
                    && cached.default_color == text_area.default_color
                    && cached.atlas_generation == atlas_gen
                {
                    // Validate all distinct glyphs still in atlas
                    let glyphs_valid = cached
                        .distinct_keys
                        .iter()
                        .all(|k| atlas.glyphs.get_and_mark_used(k).is_some());

                    if glyphs_valid {
                        let dx = text_area.left - cached.left;
                        let dy = text_area.top - cached.top;

                        if dx == 0.0 && dy == 0.0 {
                            // Exact position match — extend from cache directly
                            self.instances.extend_from_slice(&cached.instances);
                            non_vector_collector.extend_from_slice(&cached.non_vector_glyphs);
                        } else {
                            // Position shifted — adjust screen_rect and re-cull
                            any_position_changed = true;
                            let bounds_min_x = text_area.bounds.left.max(0) as f32;
                            let bounds_min_y = text_area.bounds.top.max(0) as f32;
                            let bounds_max_x =
                                text_area.bounds.right.min(resolution.width as i32) as f32;
                            let bounds_max_y =
                                text_area.bounds.bottom.min(resolution.height as i32) as f32;

                            for inst in &cached.instances {
                                let sx = inst.screen_rect[0] + dx;
                                let sy = inst.screen_rect[1] + dy;
                                let sw = inst.screen_rect[2];
                                let sh = inst.screen_rect[3];

                                // Shader adds scroll_offset, so visible position
                                // is (sx + scroll[0], sy + scroll[1]).
                                let vx = sx + scroll[0];
                                let vy = sy + scroll[1];
                                if vx + sw + 1.0 < bounds_min_x
                                    || vx - 1.0 > bounds_max_x
                                    || vy + sh + 1.0 < bounds_min_y
                                    || vy - 1.0 > bounds_max_y
                                {
                                    continue;
                                }

                                let mut adjusted = *inst;
                                adjusted.screen_rect[0] = sx;
                                adjusted.screen_rect[1] = sy;
                                self.instances.push(adjusted);
                            }

                            // Replay non-vector glyphs with adjusted positions
                            let dx_i = dx.round() as i32;
                            let dy_i = dy.round() as i32;
                            for nv in &cached.non_vector_glyphs {
                                let mut adjusted = nv.clone();
                                adjusted.physical.x += dx_i;
                                adjusted.physical.y += dy_i;
                                adjusted.clip_bounds = [
                                    bounds_min_x as i32,
                                    bounds_min_y as i32,
                                    bounds_max_x as i32,
                                    bounds_max_y as i32,
                                ];
                                non_vector_collector.push(adjusted);
                            }
                        }
                        continue;
                    }
                }
            }

            // Cache miss — full glyph loop for this TextArea
            all_hit = false;
            let instance_start = self.instances.len();
            let mut area_keys: Vec<GlyphKey> = Vec::new();
            let mut area_non_vector: Vec<NonVectorGlyph> = Vec::new();

            let bounds_min_x = text_area.bounds.left.max(0);
            let bounds_min_y = text_area.bounds.top.max(0);
            let bounds_max_x = text_area.bounds.right.min(resolution.width as i32);
            let bounds_max_y = text_area.bounds.bottom.min(resolution.height as i32);

            let is_run_visible = |run: &cosmic_text::LayoutRun| {
                let start_y = (text_area.top + run.line_top * text_area.scale) as i32;
                let end_y = start_y + (run.line_height * text_area.scale) as i32;
                start_y <= bounds_max_y && bounds_min_y <= end_y
            };

            let layout_runs = text_area
                .buffer
                .layout_runs()
                .skip_while(|run| !is_run_visible(run))
                .take_while(is_run_visible);

            let default_color = color_to_f32(text_area.default_color);

            for run in layout_runs {
                for glyph in run.glyphs {
                    let key = GlyphKey::from_layout_glyph(glyph);
                    let entry = match atlas.glyphs.get_and_mark_used(&key) {
                        Some(e) => e,
                        None => self.resolve_glyph_miss(device, font_system, atlas, glyph, key)?,
                    };

                    area_keys.push(key);

                    if entry.is_non_vector() {
                        let physical =
                            glyph.physical((text_area.left, text_area.top), text_area.scale);
                        let color = match glyph.color_opt {
                            Some(c) => color_to_f32(c),
                            None => default_color,
                        };
                        area_non_vector.push(NonVectorGlyph {
                            physical,
                            color,
                            depth: metadata_to_depth(glyph.metadata),
                            line_y_scaled_rounded: (run.line_y * text_area.scale).round(),
                            clip_bounds: [bounds_min_x, bounds_min_y, bounds_max_x, bounds_max_y],
                        });
                        continue;
                    }

                    if entry.is_color_v1_vector() {
                        // COLRv1: single instance, shader interprets command sequence.
                        if let Some(v1_entry) = atlas.color_v1_glyphs.get(&key) {
                            let scale =
                                glyph.font_size * text_area.scale / v1_entry.units_per_em;
                            let glyph_x =
                                text_area.left + (glyph.x + glyph.x_offset) * text_area.scale;
                            let glyph_y =
                                text_area.top + (run.line_y + glyph.y_offset) * text_area.scale;
                            let [min_x, min_y, max_x, max_y] = v1_entry.bounds;
                            let screen_x = glyph_x + min_x * scale;
                            let screen_y = glyph_y - max_y * scale;
                            let screen_w = (max_x - min_x) * scale;
                            let screen_h = (max_y - min_y) * scale;

                            let vis_x = screen_x + scroll[0];
                            let vis_y = screen_y + scroll[1];
                            if vis_x + screen_w + 1.0 >= bounds_min_x as f32
                                && vis_x - 1.0 <= bounds_max_x as f32
                                && vis_y + screen_h + 1.0 >= bounds_min_y as f32
                                && vis_y - 1.0 <= bounds_max_y as f32
                            {
                                self.instances.push(GlyphInstance {
                                    screen_rect: [screen_x, screen_y, screen_w, screen_h],
                                    em_rect: [min_x, min_y, max_x, max_y],
                                    band_transform: [0.0; 4], // unused for V1
                                    glyph_data: [
                                        v1_entry.blob_offset,
                                        0,
                                        0,
                                        v1_entry.cmd_count,
                                    ],
                                    color: match glyph.color_opt {
                                        Some(c) => color_to_f32(c),
                                        None => default_color,
                                    },
                                    depth: metadata_to_depth(glyph.metadata),
                                    ppem: glyph.font_size * text_area.scale,
                                    _pad: [0.0; 2],
                                });
                            }
                        }
                        continue;
                    }

                    if entry.is_color_vector() {
                        // COLRv0: emit one instance per layer, back-to-front.
                        if let Some(color_entry) = atlas.color_glyphs.get(&key) {
                            let foreground_color = match glyph.color_opt {
                                Some(c) => color_to_f32(c),
                                None => default_color,
                            };
                            let scale =
                                glyph.font_size * text_area.scale / color_entry.units_per_em;
                            let glyph_x =
                                text_area.left + (glyph.x + glyph.x_offset) * text_area.scale;
                            let glyph_y =
                                text_area.top + (run.line_y + glyph.y_offset) * text_area.scale;
                            let depth = metadata_to_depth(glyph.metadata);
                            let ppem = glyph.font_size * text_area.scale;

                            for layer in &color_entry.layers {
                                let [min_x, min_y, max_x, max_y] = layer.entry.bounds;
                                let screen_x = glyph_x + min_x * scale;
                                let screen_y = glyph_y - max_y * scale;
                                let screen_w = (max_x - min_x) * scale;
                                let screen_h = (max_y - min_y) * scale;

                                let vis_x = screen_x + scroll[0];
                                let vis_y = screen_y + scroll[1];
                                if vis_x + screen_w + 1.0 < bounds_min_x as f32
                                    || vis_x - 1.0 > bounds_max_x as f32
                                    || vis_y + screen_h + 1.0 < bounds_min_y as f32
                                    || vis_y - 1.0 > bounds_max_y as f32
                                {
                                    continue;
                                }

                                let color = if layer.use_foreground {
                                    foreground_color
                                } else {
                                    layer.color
                                };

                                self.instances.push(GlyphInstance {
                                    screen_rect: [screen_x, screen_y, screen_w, screen_h],
                                    em_rect: [min_x, min_y, max_x, max_y],
                                    band_transform: layer.entry.band_transform,
                                    glyph_data: [
                                        layer.entry.band_offset,
                                        layer.entry.band_max_x,
                                        layer.entry.band_max_y,
                                        0,
                                    ],
                                    color,
                                    depth,
                                    ppem,
                                    _pad: [0.0; 2],
                                });
                            }
                        }
                        continue;
                    }

                    let scale = glyph.font_size * text_area.scale / entry.units_per_em;
                    let [min_x, min_y, max_x, max_y] = entry.bounds;

                    let glyph_x = text_area.left + (glyph.x + glyph.x_offset) * text_area.scale;
                    let glyph_y = text_area.top + (run.line_y + glyph.y_offset) * text_area.scale;

                    let screen_x = glyph_x + min_x * scale;
                    let screen_y = glyph_y - max_y * scale;
                    let screen_w = (max_x - min_x) * scale;
                    let screen_h = (max_y - min_y) * scale;

                    let vis_x = screen_x + scroll[0];
                    let vis_y = screen_y + scroll[1];
                    if vis_x + screen_w + 1.0 < bounds_min_x as f32
                        || vis_x - 1.0 > bounds_max_x as f32
                        || vis_y + screen_h + 1.0 < bounds_min_y as f32
                        || vis_y - 1.0 > bounds_max_y as f32
                    {
                        continue;
                    }

                    let color = match glyph.color_opt {
                        Some(c) => color_to_f32(c),
                        None => default_color,
                    };

                    self.instances.push(GlyphInstance {
                        screen_rect: [screen_x, screen_y, screen_w, screen_h],
                        em_rect: [min_x, min_y, max_x, max_y],
                        band_transform: entry.band_transform,
                        glyph_data: [entry.band_offset, entry.band_max_x, entry.band_max_y, 0],
                        color,
                        depth: metadata_to_depth(glyph.metadata),
                        ppem: glyph.font_size * text_area.scale,
                        _pad: [0.0; 2],
                    });
                }
            }

            // Deduplicate keys for efficient mark-used on future hits
            area_keys.sort_unstable();
            area_keys.dedup();

            let area_instances = self.instances[instance_start..].to_vec();
            non_vector_collector.extend_from_slice(&area_non_vector);
            self.text_area_cache.insert(
                buffer_ptr,
                CachedTextArea {
                    left: text_area.left,
                    top: text_area.top,
                    scale: text_area.scale,
                    bounds: text_area.bounds,
                    default_color: text_area.default_color,
                    atlas_generation: atlas_gen,
                    instances: area_instances,
                    distinct_keys: area_keys,
                    non_vector_glyphs: area_non_vector,
                },
            );
        }

        // Remove stale cache entries (buffers no longer in the text_areas set)
        self.text_area_cache
            .retain(|ptr, _| used_ptrs.contains(ptr));

        atlas.flush_uploads(queue);

        // Rasterize non-vector glyphs via the shared atlas
        self.raster_instances = atlas.rasterize_glyphs(queue, font_system, &non_vector_collector);
        self.raster_glyphs_to_render = self.raster_instances.len() as u32;

        // Whole-frame fast path: if all areas hit cache with no position changes,
        // the GPU vertex buffer already contains the correct data and raster
        // instances haven't changed.
        if all_hit
            && !any_position_changed
            && self.instances.len() == self.glyphs_to_render as usize
        {
            // Still need to upload raster vertex buffer (raster atlas may have changed)
            self.upload_raster_vertices(device, queue);
            self.prepared_atlas_generation = atlas_gen;
            return Ok(());
        }

        self.upload_vertices(device, queue);
        self.upload_raster_vertices(device, queue);
        self.prepared_atlas_generation = atlas_gen;
        Ok(())
    }

    /// Resolve a glyph: return cached entry or extract + upload on miss.
    /// Cold path: called only on cache miss. Extracts outline, builds bands,
    /// uploads glyph blob, and inserts into cache.
    fn resolve_glyph_miss(
        &mut self,
        device: &Device,
        font_system: &mut cosmic_text::FontSystem,
        atlas: &mut TextAtlas,
        glyph: &cosmic_text::LayoutGlyph,
        key: GlyphKey,
    ) -> Result<crate::glyph_cache::GlyphEntry, PrepareError> {
        // Cache miss — look up font from cache or populate
        let cache_key = (glyph.font_id, glyph.font_weight);
        if !self.font_cache.contains_key(&cache_key) {
            let face_index = font_system
                .db()
                .face(glyph.font_id)
                .map(|info| info.index)
                .unwrap_or(0);
            let font = match font_system.get_font(glyph.font_id, glyph.font_weight) {
                Some(f) => f,
                None => {
                    log::warn!("Font not found for glyph {key:?}");
                    return Ok(atlas.glyphs.insert_and_mark_used(key, NON_VECTOR_GLYPH));
                }
            };
            let skrifa_font = skrifa::FontRef::from_index(font.data(), face_index).ok();
            let units_per_em = skrifa_font
                .as_ref()
                .and_then(|f| {
                    use skrifa::raw::TableProvider;
                    f.head().map(|h| h.units_per_em() as f32).ok()
                })
                .unwrap_or(1000.0);
            let has_colr = skrifa_font
                .as_ref()
                .map(|f| {
                    use skrifa::raw::TableProvider;
                    f.colr().is_ok()
                })
                .unwrap_or(false);
            self.font_cache.insert(
                cache_key,
                CachedFont {
                    font,
                    face_index,
                    units_per_em,
                    has_colr,
                },
            );
        }
        let cached = &self.font_cache[&cache_key];
        let (font_data, face_index, units_per_em, has_colr) =
            (cached.font.data(), cached.face_index, cached.units_per_em, cached.has_colr);

        let wght_tag = skrifa::Tag::new(b"wght");
        let location = [VariationSetting::new(wght_tag, glyph.font_weight.0 as f32)];

        // Check for COLR color glyph first — COLRv0 fonts often have fallback
        // monochrome outlines, so extract_outline would succeed but miss the color.
        // Skip the COLR check entirely for fonts without a COLR table.
        let color_info = if has_colr {
            extract_color_info(font_data, face_index, glyph.glyph_id, &location)
        } else {
            None
        };
        let entry = match color_info {
            Some(ColorGlyphInfo::V0Layers(layers)) => {
                let fake_italic = glyph
                    .cache_key_flags
                    .contains(cosmic_text::CacheKeyFlags::FAKE_ITALIC);
                match self.upload_colr_v0_layers(
                    device, atlas, font_data, face_index, units_per_em,
                    &location, &layers, fake_italic, key,
                ) {
                    Ok(entry) => entry,
                    Err(_) => NON_VECTOR_GLYPH,
                }
            }
            Some(ColorGlyphInfo::V1(mut v1_data)) => {
                match atlas.upload_color_v1(device, &mut v1_data, units_per_em) {
                    Ok(v1_entry) => {
                        atlas.color_v1_glyphs.insert(key, v1_entry);
                        COLOR_V1_VECTOR_GLYPH
                    }
                    Err(_) => NON_VECTOR_GLYPH,
                }
            }
            None => {
                // No color data — try regular outline, else raster fallback.
                match extract_outline(font_data, face_index, glyph.glyph_id, &location) {
                    Some(mut outline) => {
                        if glyph
                            .cache_key_flags
                            .contains(cosmic_text::CacheKeyFlags::FAKE_ITALIC)
                        {
                            apply_italic_shear(&mut outline);
                        }
                        let band_count = band_count_for_curves(outline.curves.len());
                        atlas.upload_glyph(
                            device, &outline, band_count, band_count, units_per_em,
                        )?
                    }
                    None => NON_VECTOR_GLYPH,
                }
            }
        };

        Ok(atlas.glyphs.insert_and_mark_used(key, entry))
    }

    /// Upload all sub-glyph outlines for a COLRv0 color glyph and store the
    /// ColorGlyphEntry. Returns COLOR_VECTOR_GLYPH sentinel for the main cache.
    #[allow(clippy::too_many_arguments)]
    fn upload_colr_v0_layers(
        &self,
        device: &Device,
        atlas: &mut TextAtlas,
        font_data: &[u8],
        face_index: u32,
        units_per_em: f32,
        location: &[VariationSetting],
        layers: &[crate::outline::ColorLayer],
        fake_italic: bool,
        key: GlyphKey,
    ) -> Result<crate::glyph_cache::GlyphEntry, PrepareError> {
        let mut entries = Vec::with_capacity(layers.len());

        for layer in layers {
            let outline = extract_outline(font_data, face_index, layer.glyph_id, location);
            let mut outline = match outline {
                Some(o) => o,
                None => continue, // Skip layers with no outline (e.g. empty glyphs)
            };

            if fake_italic {
                apply_italic_shear(&mut outline);
            }

            let band_count = band_count_for_curves(outline.curves.len());
            let entry =
                atlas.upload_glyph(device, &outline, band_count, band_count, units_per_em)?;
            entries.push(ColorGlyphLayer {
                entry,
                color: layer.color,
                use_foreground: layer.use_foreground,
            });
        }

        if entries.is_empty() {
            return Ok(NON_VECTOR_GLYPH);
        }

        atlas.color_glyphs.insert(
            key,
            ColorGlyphEntry {
                layers: entries,
                units_per_em,
            },
        );

        Ok(COLOR_VECTOR_GLYPH)
    }

    /// Upload the instance buffer to the GPU.
    fn upload_vertices(&mut self, device: &Device, queue: &Queue) {
        self.glyphs_to_render = self.instances.len() as u32;

        if self.instances.is_empty() {
            return;
        }

        let vertices_raw = bytemuck::cast_slice(&self.instances);

        if self.vertex_buffer_size >= vertices_raw.len() as u64 {
            queue.write_buffer(&self.vertex_buffer, 0, vertices_raw);
        } else {
            self.vertex_buffer.destroy();

            let new_size = next_copy_buffer_size(vertices_raw.len() as u64);
            self.vertex_buffer = device.create_buffer(&BufferDescriptor {
                label: Some("sluggrs vertices"),
                size: new_size,
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                mapped_at_creation: true,
            });
            self.vertex_buffer.slice(..).get_mapped_range_mut()[..vertices_raw.len()]
                .copy_from_slice(vertices_raw);
            self.vertex_buffer.unmap();
            self.vertex_buffer_size = new_size;
        }
    }

    fn upload_raster_vertices(&mut self, device: &Device, queue: &Queue) {
        if self.raster_instances.is_empty() {
            self.raster_glyphs_to_render = 0;
            return;
        }

        let data = bytemuck::cast_slice(&self.raster_instances);

        if self.raster_vertex_buffer_size >= data.len() as u64 {
            queue.write_buffer(&self.raster_vertex_buffer, 0, data);
        } else {
            self.raster_vertex_buffer.destroy();
            let new_size = (data.len() as u64).next_power_of_two().max(4096);
            self.raster_vertex_buffer = device.create_buffer(&BufferDescriptor {
                label: Some("sluggrs raster vertices"),
                size: new_size,
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&self.raster_vertex_buffer, 0, data);
            self.raster_vertex_buffer_size = new_size;
        }
    }

    /// Prepares all of the provided text areas for rendering.
    #[allow(clippy::too_many_arguments)] // matches cryoglyph's API
    pub fn prepare<'a>(
        &mut self,
        device: &Device,
        queue: &Queue,
        encoder: &CommandEncoder,
        font_system: &mut cosmic_text::FontSystem,
        atlas: &mut TextAtlas,
        viewport: &Viewport,
        text_areas: impl IntoIterator<Item = TextArea<'a>>,
        cache: &mut cosmic_text::SwashCache,
    ) -> Result<(), PrepareError> {
        self.prepare_with_depth(
            device,
            queue,
            encoder,
            font_system,
            atlas,
            viewport,
            text_areas,
            cache,
            zero_depth,
        )
    }

    /// Renders all layouts that were previously provided to `prepare`.
    pub fn render<'a>(
        &'a self,
        atlas: &'a TextAtlas,
        viewport: &'a Viewport,
        pass: &mut RenderPass<'a>,
    ) -> Result<(), RenderError> {
        // Detect trim(reset) between prepare() and render(): the atlas was
        // recreated so our instance buffer references stale glyph offsets.
        if atlas.generation() != self.prepared_atlas_generation {
            return Err(RenderError::RemovedFromAtlas);
        }

        if self.glyphs_to_render > 0 {
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &viewport.bind_group, &[]);
            pass.set_bind_group(1, &atlas.bind_group, &[]);
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            pass.draw(0..4, 0..self.glyphs_to_render);
        }

        // Raster fallback (emoji, bitmap fonts)
        if self.raster_glyphs_to_render > 0 {
            atlas.render_raster_pass(
                viewport,
                pass,
                &self.raster_vertex_buffer,
                self.raster_glyphs_to_render,
            );
        }

        Ok(())
    }

    pub fn trim(&mut self) {
        // Raster trim is handled by TextAtlas::trim()
    }
}

/// Determine the band count for a glyph based on its curve complexity.
/// Matches harfbuzz: 1:1 up to a cap of 16 bands.
fn band_count_for_curves(num_curves: usize) -> u32 {
    (num_curves as u32).clamp(1, 16)
}

/// Convert a cosmic_text Color to normalized [f32; 4].
fn color_to_f32(c: cosmic_text::Color) -> [f32; 4] {
    [
        c.r() as f32 / 255.0,
        c.g() as f32 / 255.0,
        c.b() as f32 / 255.0,
        c.a() as f32 / 255.0,
    ]
}

fn next_copy_buffer_size(size: u64) -> u64 {
    let align_mask = COPY_BUFFER_ALIGNMENT - 1;
    ((size.next_power_of_two() + align_mask) & !align_mask).max(COPY_BUFFER_ALIGNMENT)
}

fn zero_depth(_: usize) -> f32 {
    0.0
}
