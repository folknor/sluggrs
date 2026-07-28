use crate::GlyphInstance;
use crate::glyph_cache::{COLOR_V1_VECTOR_GLYPH, COLOR_VECTOR_GLYPH, GlyphKey, NON_VECTOR_GLYPH};
use crate::outline::{ColorGlyphInfo, extract_color_info, extract_outline};
use crate::prep::{PrepScratch, prepare_mono};
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

type BufferPtr = *const cosmic_text::Buffer;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct TextAreaCacheKey {
    buffer_ptr: BufferPtr,
    occurrence: usize,
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PrepareStats {
    pub direct_hits: usize,
    pub reculls: usize,
    pub misses: usize,
}

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
    scroll: [f32; 2],
    scale: f32,
    bounds: TextBounds,
    default_color: cosmic_text::Color,
    atlas_generation: u32,
    instances: Vec<GlyphInstance>,
    distinct_keys: Vec<GlyphKey>,
    non_vector_glyphs: Vec<NonVectorGlyph>,
    /// Whether the cached candidates cover the entire area, so a later
    /// placement-only change can re-cull them without re-walking the buffer.
    complete: bool,
}

/// One glyph queued for instance packing in pass 3 (cache-miss areas only).
struct WorkItem<'a> {
    glyph: &'a cosmic_text::LayoutGlyph,
    line_y: f32,
    key: GlyphKey,
}

/// Per-area context for a cache-miss area. Built in pass 1, consumed in pass 3.
struct MissArea<'a> {
    cache_key: TextAreaCacheKey,
    text_area: TextArea<'a>,
    work_start: usize,
    work_end: usize,
    bounds_min_x: i32,
    bounds_min_y: i32,
    bounds_max_x: i32,
    bounds_max_y: i32,
    default_color: [f32; 4],
    all_runs_included: bool,
}

/// Plan record per text area produced by pass 1 and consumed by pass 3 in input order.
enum AreaPlan<'a> {
    HitDirect {
        cache_key: TextAreaCacheKey,
    },
    ReCull {
        cache_key: TextAreaCacheKey,
        dx: f32,
        dy: f32,
        left: f32,
        top: f32,
        bounds: [i32; 4],
        scroll: [f32; 2],
    },
    Miss(MissArea<'a>),
}

/// How a cached area relates to the requested placement (`left`, `top`,
/// viewport scroll). See `classify_placement`.
enum PlacementClass {
    Direct,
    ReCull,
    Miss,
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
    /// Per-TextArea retained cache, keyed by buffer pointer and occurrence.
    text_area_cache: FxHashMap<TextAreaCacheKey, CachedTextArea>,
    /// Per-frame occurrence counters for shared buffers.
    text_area_occurrences: FxHashMap<BufferPtr, usize>,
    last_prepare_stats: PrepareStats,
    /// Resolution from last frame, for cache invalidation.
    cached_resolution: crate::types::Resolution,
    /// Atlas generation at last prepare() - detects trim(reset) between prepare and render.
    prepared_atlas_generation: u32,
    // Raster fallback: per-frame instances drawn using TextAtlas's shared raster resources
    raster_instances: Vec<RasterVertex>,
    raster_vertex_buffer: Buffer,
    raster_vertex_buffer_size: u64,
    raster_glyphs_to_render: u32,
    /// Reusable scratch for `prep::prepare_mono` calls. One buffer set total
    /// while pass 2 is serial; future rayon work will hold one per worker.
    prep_scratch: PrepScratch,
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
            text_area_occurrences: FxHashMap::default(),
            last_prepare_stats: PrepareStats::default(),
            cached_resolution: crate::types::Resolution {
                width: 0,
                height: 0,
            },
            prepared_atlas_generation: 0,
            raster_instances: Vec::new(),
            raster_vertex_buffer,
            raster_vertex_buffer_size,
            raster_glyphs_to_render: 0,
            prep_scratch: PrepScratch::default(),
        }
    }

    /// Prepare text areas for rendering, with per-glyph depth mapping.
    ///
    /// `encoder` and `cache` are unused - they exist for cryoglyph API
    /// compatibility. sluggrs uses `queue.write_texture` (no encoder needed)
    /// and extracts outlines via skrifa (no swash rasterization).
    ///
    /// Three-pass structure:
    /// 1. Classify each text area as cache-hit (direct/shifted) or miss;
    ///    for misses, walk visible runs and collect work items + distinct keys.
    /// 2. Resolve distinct missing glyph keys (extract+upload). Future-parallel.
    /// 3. Walk plans in input order; emit instances per area; populate cache.
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
        self.text_area_occurrences.clear();
        let mut non_vector_collector: Vec<NonVectorGlyph> = Vec::new();

        let resolution = viewport.resolution();
        let scroll = viewport.scroll_offset();
        let atlas_gen = atlas.generation();

        if resolution != self.cached_resolution {
            self.text_area_cache.clear();
            self.cached_resolution = resolution;
        }

        // Only an all-direct-hit frame may reuse the vector vertex buffer.
        let mut all_direct_hits = true;
        let mut prepare_stats = PrepareStats::default();

        let mut plans: Vec<AreaPlan<'a>> = Vec::new();
        let mut work: Vec<WorkItem<'a>> = Vec::new();
        let mut distinct_misses: Vec<GlyphKey> = Vec::new();

        // ===== Pass 1: classify areas, collect work items =====
        for text_area in text_areas {
            let buffer_ptr = text_area.buffer as BufferPtr;
            let count = self.text_area_occurrences.entry(buffer_ptr).or_default();
            let cache_key = TextAreaCacheKey {
                buffer_ptr,
                occurrence: *count,
            };
            *count += 1;

            if let Some(cached) = self.text_area_cache.get(&cache_key)
                && !text_area.buffer.redraw()
                && cached.scale == text_area.scale
                && cached.bounds == text_area.bounds
                && cached.default_color == text_area.default_color
                && cached.atlas_generation == atlas_gen
            {
                let glyphs_valid = cached
                    .distinct_keys
                    .iter()
                    .all(|k| atlas.glyphs.get_and_mark_used(k).is_some());

                if glyphs_valid {
                    let dx = text_area.left - cached.left;
                    let dy = text_area.top - cached.top;
                    match classify_placement(
                        dx,
                        dy,
                        cached.scroll == scroll,
                        cached.complete,
                        !cached.non_vector_glyphs.is_empty(),
                    ) {
                        PlacementClass::Direct => {
                            prepare_stats.direct_hits += 1;
                            plans.push(AreaPlan::HitDirect { cache_key });
                            continue;
                        }
                        PlacementClass::ReCull => {
                            all_direct_hits = false;
                            prepare_stats.reculls += 1;
                            plans.push(AreaPlan::ReCull {
                                cache_key,
                                dx,
                                dy,
                                left: text_area.left,
                                top: text_area.top,
                                bounds: clipped_bounds(text_area.bounds, resolution),
                                scroll,
                            });
                            continue;
                        }
                        // Fall through to the miss path below.
                        PlacementClass::Miss => {}
                    }
                }
            }

            all_direct_hits = false;
            prepare_stats.misses += 1;
            let [bounds_min_x, bounds_min_y, bounds_max_x, bounds_max_y] =
                clipped_bounds(text_area.bounds, resolution);

            let default_color = color_to_f32(text_area.default_color);
            let work_start = work.len();

            let mut all_runs_included = true;
            let mut started_visible_range = false;
            for run in text_area.buffer.layout_runs() {
                if !run_is_visible(
                    text_area.top,
                    text_area.scale,
                    scroll[1],
                    &run,
                    bounds_min_y,
                    bounds_max_y,
                ) {
                    all_runs_included = false;
                    if started_visible_range {
                        break;
                    }
                    continue;
                }
                started_visible_range = true;
                let line_y = run.line_y;
                for glyph in run.glyphs {
                    let key = GlyphKey::from_layout_glyph(glyph);
                    if atlas.glyphs.get_and_mark_used(&key).is_none() {
                        distinct_misses.push(key);
                    }
                    work.push(WorkItem { glyph, line_y, key });
                }
            }
            let work_end = work.len();

            plans.push(AreaPlan::Miss(MissArea {
                cache_key,
                text_area,
                work_start,
                work_end,
                bounds_min_x,
                bounds_min_y,
                bounds_max_x,
                bounds_max_y,
                default_color,
                all_runs_included,
            }));
        }

        // ===== Pass 2: resolve distinct misses =====
        // Sort+dedup so each glyph is extracted+uploaded once even if it
        // appears across many text areas. Future: parallelize the extract
        // step with serial atlas commit.
        distinct_misses.sort_unstable();
        distinct_misses.dedup();
        for key in &distinct_misses {
            self.resolve_glyph_miss(font_system, atlas, *key)?;
        }

        // ===== Pass 3: emit instances per area in input order =====
        for plan in &plans {
            match plan {
                AreaPlan::HitDirect { cache_key } => {
                    let cached = &self.text_area_cache[cache_key];
                    self.instances.extend_from_slice(&cached.instances);
                    non_vector_collector.extend_from_slice(&cached.non_vector_glyphs);
                }
                AreaPlan::ReCull {
                    cache_key,
                    dx,
                    dy,
                    left,
                    top,
                    bounds,
                    scroll,
                } => {
                    let cached = self
                        .text_area_cache
                        .get_mut(cache_key)
                        .expect("re-cull cache entry exists");
                    let (instances, complete) = re_cull_vector_instances(
                        &cached.instances,
                        cached.complete,
                        *dx,
                        *dy,
                        *scroll,
                        *bounds,
                    );
                    non_vector_collector.extend_from_slice(&cached.non_vector_glyphs);
                    self.instances.extend_from_slice(&instances);
                    cached.left = *left;
                    cached.top = *top;
                    cached.scroll = *scroll;
                    cached.instances = instances;
                    cached.complete = complete;
                }
                AreaPlan::Miss(area) => {
                    let mut area_instances: Vec<GlyphInstance> = Vec::new();
                    let mut area_non_vector: Vec<NonVectorGlyph> = Vec::new();
                    let mut area_keys: Vec<GlyphKey> = Vec::new();
                    let mut complete = area.all_runs_included;

                    let text_area = &area.text_area;
                    let bounds = [
                        area.bounds_min_x,
                        area.bounds_min_y,
                        area.bounds_max_x,
                        area.bounds_max_y,
                    ];

                    for wi in &work[area.work_start..area.work_end] {
                        let glyph = wi.glyph;
                        let entry = atlas.glyphs.get(&wi.key).expect("miss resolved in pass 2");
                        area_keys.push(wi.key);

                        if entry.is_non_vector() {
                            let physical =
                                glyph.physical((text_area.left, text_area.top), text_area.scale);
                            let color = match glyph.color_opt {
                                Some(c) => color_to_f32(c),
                                None => area.default_color,
                            };
                            area_non_vector.push(NonVectorGlyph {
                                physical,
                                color,
                                depth: metadata_to_depth(glyph.metadata),
                                line_y_scaled_rounded: (wi.line_y * text_area.scale).round(),
                                clip_bounds: [
                                    area.bounds_min_x,
                                    area.bounds_min_y,
                                    area.bounds_max_x,
                                    area.bounds_max_y,
                                ],
                            });
                            continue;
                        }

                        if entry.is_color_v1_vector() {
                            if let Some(v1_entry) = atlas.color_v1_glyphs.get(&wi.key) {
                                let scale =
                                    glyph.font_size * text_area.scale / v1_entry.units_per_em;
                                let glyph_x =
                                    text_area.left + (glyph.x + glyph.x_offset) * text_area.scale;
                                let glyph_y =
                                    text_area.top + (wi.line_y + glyph.y_offset) * text_area.scale;
                                let [min_x, min_y, max_x, max_y] = v1_entry.bounds;
                                let screen_x = glyph_x + min_x * scale;
                                let screen_y = glyph_y - max_y * scale;
                                let screen_w = (max_x - min_x) * scale;
                                let screen_h = (max_y - min_y) * scale;

                                let screen_rect = [screen_x, screen_y, screen_w, screen_h];
                                if vector_rect_visible(screen_rect, scroll, bounds) {
                                    area_instances.push(GlyphInstance {
                                        screen_rect,
                                        color: match glyph.color_opt {
                                            Some(c) => color_to_f32(c),
                                            None => area.default_color,
                                        },
                                        glyph_offset: v1_entry.glyph_offset,
                                        cmd_texel_count: v1_entry.cmd_texel_count,
                                        depth: metadata_to_depth(glyph.metadata),
                                        ppem: glyph.font_size * text_area.scale,
                                    });
                                } else {
                                    complete = false;
                                }
                            } else {
                                complete = false;
                            }
                            continue;
                        }

                        if entry.is_color_vector() {
                            if let Some(color_entry) = atlas.color_glyphs.get(&wi.key) {
                                let foreground_color = match glyph.color_opt {
                                    Some(c) => color_to_f32(c),
                                    None => area.default_color,
                                };
                                let scale =
                                    glyph.font_size * text_area.scale / color_entry.units_per_em;
                                let glyph_x =
                                    text_area.left + (glyph.x + glyph.x_offset) * text_area.scale;
                                let glyph_y =
                                    text_area.top + (wi.line_y + glyph.y_offset) * text_area.scale;
                                let depth = metadata_to_depth(glyph.metadata);
                                let ppem = glyph.font_size * text_area.scale;

                                for layer in &color_entry.layers {
                                    if layer.entry.is_non_vector() {
                                        continue;
                                    }
                                    let [min_x, min_y, max_x, max_y] = layer.entry.bounds;
                                    let screen_x = glyph_x + min_x * scale;
                                    let screen_y = glyph_y - max_y * scale;
                                    let screen_w = (max_x - min_x) * scale;
                                    let screen_h = (max_y - min_y) * scale;

                                    let screen_rect = [screen_x, screen_y, screen_w, screen_h];
                                    if !vector_rect_visible(screen_rect, scroll, bounds) {
                                        complete = false;
                                        continue;
                                    }

                                    let color = if layer.use_foreground {
                                        foreground_color
                                    } else {
                                        layer.color
                                    };

                                    area_instances.push(GlyphInstance {
                                        screen_rect,
                                        color,
                                        glyph_offset: layer.entry.glyph_offset,
                                        cmd_texel_count: 0,
                                        depth,
                                        ppem,
                                    });
                                }
                            } else {
                                complete = false;
                            }
                            continue;
                        }

                        let scale = glyph.font_size * text_area.scale / entry.units_per_em;
                        let [min_x, min_y, max_x, max_y] = entry.bounds;

                        let glyph_x = text_area.left + (glyph.x + glyph.x_offset) * text_area.scale;
                        let glyph_y =
                            text_area.top + (wi.line_y + glyph.y_offset) * text_area.scale;

                        let screen_x = glyph_x + min_x * scale;
                        let screen_y = glyph_y - max_y * scale;
                        let screen_w = (max_x - min_x) * scale;
                        let screen_h = (max_y - min_y) * scale;

                        let screen_rect = [screen_x, screen_y, screen_w, screen_h];
                        if !vector_rect_visible(screen_rect, scroll, bounds) {
                            complete = false;
                            continue;
                        }

                        let color = match glyph.color_opt {
                            Some(c) => color_to_f32(c),
                            None => area.default_color,
                        };

                        area_instances.push(GlyphInstance {
                            screen_rect,
                            color,
                            glyph_offset: entry.glyph_offset,
                            cmd_texel_count: 0,
                            depth: metadata_to_depth(glyph.metadata),
                            ppem: glyph.font_size * text_area.scale,
                        });
                    }

                    area_keys.sort_unstable();
                    area_keys.dedup();

                    self.instances.extend_from_slice(&area_instances);
                    non_vector_collector.extend_from_slice(&area_non_vector);

                    self.text_area_cache.insert(
                        area.cache_key,
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
                            scroll,
                            complete,
                        },
                    );
                }
            }
        }

        let occurrences = &self.text_area_occurrences;
        self.text_area_cache.retain(|key, _| {
            occurrences
                .get(&key.buffer_ptr)
                .is_some_and(|count| key.occurrence < *count)
        });

        atlas.flush_uploads(queue);

        self.raster_instances =
            atlas.rasterize_glyphs(queue, font_system, &non_vector_collector, scroll);
        self.raster_glyphs_to_render = self.raster_instances.len() as u32;

        // Only direct hits leave the vector vertex buffer unchanged.
        if all_direct_hits && self.instances.len() == self.glyphs_to_render as usize {
            self.upload_raster_vertices(device, queue);
            self.prepared_atlas_generation = atlas_gen;
            self.last_prepare_stats = prepare_stats;
            return Ok(());
        }

        self.upload_vertices(device, queue);
        self.upload_raster_vertices(device, queue);
        self.prepared_atlas_generation = atlas_gen;
        self.last_prepare_stats = prepare_stats;
        Ok(())
    }

    /// Resolve a glyph: return cached entry or extract + upload on miss.
    /// Cold path: called only on cache miss. Extracts outline, builds bands,
    /// uploads glyph blob, and inserts into cache.
    fn resolve_glyph_miss(
        &mut self,
        font_system: &mut cosmic_text::FontSystem,
        atlas: &mut TextAtlas,
        key: GlyphKey,
    ) -> Result<crate::glyph_cache::GlyphEntry, PrepareError> {
        if let Some(entry) = atlas.restore_cached_glyph(key)? {
            return Ok(entry);
        }
        let font_weight = cosmic_text::Weight(key.font_weight);
        let cache_key = (key.font_id, font_weight);
        if let std::collections::hash_map::Entry::Vacant(slot) = self.font_cache.entry(cache_key) {
            let face_index = font_system
                .db()
                .face(key.font_id)
                .map(|info| info.index)
                .unwrap_or(0);
            let font = match font_system.get_font(key.font_id, font_weight) {
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
            slot.insert(CachedFont {
                font,
                face_index,
                units_per_em,
                has_colr,
            });
        }
        // Clone the Arc out of font_cache so we can call &mut self methods
        // (upload_colr_v0_layers needs prep_scratch) without holding a borrow
        // on self.font_cache.
        let (font_arc, face_index, units_per_em, has_colr) = {
            let cached = &self.font_cache[&cache_key];
            (
                Arc::clone(&cached.font),
                cached.face_index,
                cached.units_per_em,
                cached.has_colr,
            )
        };
        let font_data = font_arc.data();

        let wght_tag = skrifa::Tag::new(b"wght");
        let location = [VariationSetting::new(wght_tag, key.font_weight as f32)];

        // Check for COLR color glyph first - COLRv0 fonts often have fallback
        // monochrome outlines, so extract_outline would succeed but miss the color.
        // Skip the COLR check entirely for fonts without a COLR table.
        let color_info = if has_colr {
            extract_color_info(font_data, face_index, key.glyph_id, &location)
        } else {
            None
        };
        let entry = match color_info {
            Some(ColorGlyphInfo::V0Layers(layers)) => {
                let fake_italic = key
                    .cache_key_flags
                    .contains(cosmic_text::CacheKeyFlags::FAKE_ITALIC);
                match self.upload_colr_v0_layers(
                    atlas,
                    font_data,
                    face_index,
                    units_per_em,
                    &location,
                    &layers,
                    fake_italic,
                    key,
                ) {
                    Ok(entry) => entry,
                    Err(_) => NON_VECTOR_GLYPH,
                }
            }
            Some(ColorGlyphInfo::V1(mut v1_data)) => {
                match atlas.upload_color_v1(key, &mut v1_data, units_per_em) {
                    Ok(v1_entry) => {
                        atlas.color_v1_glyphs.insert(key, v1_entry);
                        COLOR_V1_VECTOR_GLYPH
                    }
                    Err(_) => NON_VECTOR_GLYPH,
                }
            }
            None => {
                // No color data - try regular outline, else raster fallback.
                match extract_outline(font_data, face_index, key.glyph_id, &location) {
                    Some(mut outline) => {
                        if key
                            .cache_key_flags
                            .contains(cosmic_text::CacheKeyFlags::FAKE_ITALIC)
                        {
                            apply_italic_shear(&mut outline);
                        }
                        let band_count = band_count_for_curves(outline.curves.len());
                        match prepare_mono(
                            &outline,
                            band_count,
                            band_count,
                            units_per_em,
                            &mut self.prep_scratch,
                        ) {
                            Some(prepared) => atlas.commit_mono(key, &prepared)?,
                            None => NON_VECTOR_GLYPH,
                        }
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
        &mut self,
        atlas: &mut TextAtlas,
        font_data: &[u8],
        face_index: u32,
        units_per_em: f32,
        location: &[VariationSetting],
        layers: &[crate::outline::ColorLayer],
        fake_italic: bool,
        key: GlyphKey,
    ) -> Result<crate::glyph_cache::GlyphEntry, PrepareError> {
        let mut prepared_layers = Vec::with_capacity(layers.len());

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
            let prepared = prepare_mono(
                &outline,
                band_count,
                band_count,
                units_per_em,
                &mut self.prep_scratch,
            );
            prepared_layers.push((prepared, layer.color, layer.use_foreground));
        }

        if prepared_layers.is_empty() {
            return Ok(NON_VECTOR_GLYPH);
        }

        let entry = atlas.commit_color_v0(key, &prepared_layers, units_per_em)?;
        atlas.color_glyphs.insert(key, entry);

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
            self.vertex_buffer
                .slice(..)
                .get_mapped_range_mut()
                .slice(..vertices_raw.len())
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
    pub fn render(
        &self,
        atlas: &TextAtlas,
        viewport: &Viewport,
        pass: &mut RenderPass<'_>,
    ) -> Result<(), RenderError> {
        // Detect trim(compaction) between prepare() and render(): the atlas was
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

    /// Vector instances emitted by the last `prepare()` call.
    ///
    /// This exists for integration tests that assert placement without a
    /// render pass; it is not part of the stable API.
    #[doc(hidden)]
    pub fn prepared_instances(&self) -> &[GlyphInstance] {
        &self.instances
    }

    #[doc(hidden)]
    pub fn last_prepare_stats(&self) -> PrepareStats {
        self.last_prepare_stats
    }
}

/// Determine the band count for a glyph based on its curve complexity.
/// Matches harfbuzz: 1:1 up to a cap of 16 bands.
fn band_count_for_curves(num_curves: usize) -> u32 {
    (num_curves as u32).clamp(1, 16)
}

fn clipped_bounds(bounds: TextBounds, resolution: crate::types::Resolution) -> [i32; 4] {
    [
        bounds.left.max(0),
        bounds.top.max(0),
        bounds.right.min(resolution.width as i32),
        bounds.bottom.min(resolution.height as i32),
    ]
}

fn run_is_visible(
    top: f32,
    scale: f32,
    scroll_y: f32,
    run: &cosmic_text::LayoutRun,
    bounds_min_y: i32,
    bounds_max_y: i32,
) -> bool {
    let start_y = top + run.line_top * scale + scroll_y;
    let end_y = start_y + run.line_height * scale;
    start_y <= bounds_max_y as f32 && bounds_min_y as f32 <= end_y
}

fn vector_rect_visible(screen_rect: [f32; 4], scroll: [f32; 2], bounds: [i32; 4]) -> bool {
    let [x, y, width, height] = screen_rect;
    let x = x + scroll[0];
    let y = y + scroll[1];
    x + width + 1.0 >= bounds[0] as f32
        && x - 1.0 <= bounds[2] as f32
        && y + height + 1.0 >= bounds[1] as f32
        && y - 1.0 <= bounds[3] as f32
}

/// Classify a placement-valid cache hit. `Direct` when nothing about the
/// placement changed; `ReCull` when the cached candidate set is complete and
/// can be shifted/re-culled (raster candidates forbid a left/top shift
/// because `LayoutGlyph::physical` recomputes integer placement and subpixel
/// bins from the origin); `Miss` otherwise.
fn classify_placement(
    dx: f32,
    dy: f32,
    scroll_matches: bool,
    complete: bool,
    has_raster_candidates: bool,
) -> PlacementClass {
    if dx == 0.0 && dy == 0.0 && scroll_matches {
        return PlacementClass::Direct;
    }
    if complete && (!has_raster_candidates || (dx == 0.0 && dy == 0.0)) {
        return PlacementClass::ReCull;
    }
    PlacementClass::Miss
}

fn re_cull_vector_instances(
    instances: &[GlyphInstance],
    complete: bool,
    dx: f32,
    dy: f32,
    scroll: [f32; 2],
    bounds: [i32; 4],
) -> (Vec<GlyphInstance>, bool) {
    let mut visible = Vec::with_capacity(instances.len());
    let mut complete = complete;
    for instance in instances {
        let mut adjusted = *instance;
        adjusted.screen_rect[0] += dx;
        adjusted.screen_rect[1] += dy;
        if vector_rect_visible(adjusted.screen_rect, scroll, bounds) {
            visible.push(adjusted);
        } else {
            complete = false;
        }
    }
    (visible, complete)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn run(line_top: f32, line_height: f32) -> cosmic_text::LayoutRun<'static> {
        cosmic_text::LayoutRun {
            line_i: 0,
            text: "",
            rtl: false,
            glyphs: &[],
            decorations: &[],
            line_y: 0.0,
            line_top,
            line_height,
            line_w: 0.0,
        }
    }

    #[test]
    fn run_visibility_applies_fractional_and_negative_scroll_at_inclusive_edges() {
        let layout_run = run(10.0, 5.0);
        assert!(run_is_visible(0.0, 1.0, -15.0, &layout_run, 0, 10));
        assert!(run_is_visible(0.5, 1.0, -10.5, &layout_run, 0, 5));
        assert!(!run_is_visible(0.0, 1.0, -15.1, &layout_run, 0, 10));
    }

    #[test]
    fn vector_culling_uses_scroll_and_one_pixel_margin() {
        let rect = [10.0, 10.0, 5.0, 5.0];
        assert!(vector_rect_visible(rect, [-16.0, 0.0], [0, 0, 10, 20]));
        assert!(!vector_rect_visible(rect, [-16.1, 0.0], [0, 0, 10, 20]));
    }

    #[test]
    fn placement_classification_covers_all_transitions() {
        use PlacementClass::{Direct, Miss, ReCull};
        // Exact placement: direct hit regardless of completeness or raster.
        assert!(matches!(
            classify_placement(0.0, 0.0, true, false, true),
            Direct
        ));
        // Scroll-only change: re-cull, even with raster candidates.
        assert!(matches!(
            classify_placement(0.0, 0.0, false, true, true),
            ReCull
        ));
        // Position change, complete, vector-only: re-cull.
        assert!(matches!(
            classify_placement(2.5, -1.0, true, true, false),
            ReCull
        ));
        // Position change with raster candidates: miss.
        assert!(matches!(
            classify_placement(0.0, 1.0, true, true, true),
            Miss
        ));
        // Incomplete cache: any placement change is a miss.
        assert!(matches!(
            classify_placement(1.0, 0.0, true, false, false),
            Miss
        ));
        assert!(matches!(
            classify_placement(0.0, 0.0, false, false, false),
            Miss
        ));
    }

    #[test]
    fn re_cull_keeps_completeness_only_when_every_vector_survives() {
        let instance = GlyphInstance {
            screen_rect: [2.0, 2.0, 2.0, 2.0],
            color: [0.0; 4],
            glyph_offset: 0,
            cmd_texel_count: 0,
            depth: 0.0,
            ppem: 0.0,
        };
        let (visible, complete) =
            re_cull_vector_instances(&[instance], true, 1.5, -0.5, [0.0, 0.0], [0, 0, 10, 10]);
        assert!(complete);
        assert_eq!(visible[0].screen_rect[..2], [3.5, 1.5]);

        let (visible, complete) =
            re_cull_vector_instances(&[instance], true, -10.0, 0.0, [0.0, 0.0], [0, 0, 10, 10]);
        assert!(visible.is_empty());
        assert!(!complete);
    }
}
