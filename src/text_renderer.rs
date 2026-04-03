use crate::GlyphInstance;
use crate::glyph_cache::{GlyphKey, NON_VECTOR_GLYPH};
use crate::outline::extract_outline;
use crate::prepare::apply_italic_shear;
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

/// Cached per-font data to avoid re-parsing font tables on every glyph miss.
struct CachedFont {
    font: Arc<cosmic_text::Font>,
    face_index: u32,
    units_per_em: f32,
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

        let pipeline = atlas.get_or_create_pipeline(device, multisample, depth_stencil);

        Self {
            vertex_buffer,
            vertex_buffer_size,
            pipeline,
            instances: Vec::new(),
            glyphs_to_render: 0,
            font_cache: FxHashMap::default(),
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
        _encoder: &mut CommandEncoder,
        font_system: &mut cosmic_text::FontSystem,
        atlas: &mut TextAtlas,
        viewport: &Viewport,
        text_areas: impl IntoIterator<Item = TextArea<'a>>,
        _cache: &mut cosmic_text::SwashCache,
        mut metadata_to_depth: impl FnMut(usize) -> f32,
    ) -> Result<(), PrepareError> {
        self.instances.clear();

        let resolution = viewport.resolution();

        for text_area in text_areas {
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
                    // --- Phase 1: Cache lookup or extraction ---
                    let entry = self.resolve_glyph(device, font_system, atlas, glyph)?;

                    if entry.is_non_vector() {
                        continue;
                    }

                    // --- Phase 2: Screen position + culling ---
                    let scale = glyph.font_size * text_area.scale / entry.units_per_em;
                    let [min_x, min_y, max_x, max_y] = entry.bounds;

                    let glyph_x = text_area.left + (glyph.x + glyph.x_offset) * text_area.scale;
                    let glyph_y = text_area.top + (run.line_y + glyph.y_offset) * text_area.scale;

                    let screen_x = glyph_x + min_x * scale;
                    let screen_y = glyph_y - max_y * scale;
                    let screen_w = (max_x - min_x) * scale;
                    let screen_h = (max_y - min_y) * scale;

                    if screen_x + screen_w + 1.0 < bounds_min_x as f32
                        || screen_x - 1.0 > bounds_max_x as f32
                        || screen_y + screen_h + 1.0 < bounds_min_y as f32
                        || screen_y - 1.0 > bounds_max_y as f32
                    {
                        continue;
                    }

                    // --- Phase 4: Instance packing ---
                    let color = match glyph.color_opt {
                        Some(c) => color_to_f32(c),
                        None => default_color,
                    };

                    self.instances.push(GlyphInstance {
                        screen_rect: [screen_x, screen_y, screen_w, screen_h],
                        em_rect: [min_x, min_y, max_x, max_y],
                        band_transform: entry.band_transform,
                        glyph_data: [
                            entry.band_offset,
                            entry.band_max_x,
                            entry.band_max_y,
                            0,
                        ],
                        color,
                        depth: metadata_to_depth(glyph.metadata),
                        ppem: glyph.font_size * text_area.scale,
                        _pad: [0.0; 2],
                    });
                }
            }
        }

        atlas.flush_uploads(queue);
        self.upload_vertices(device, queue);
        Ok(())
    }

    /// Resolve a glyph: return cached entry or extract + upload on miss.
    fn resolve_glyph(
        &mut self,
        device: &Device,
        font_system: &mut cosmic_text::FontSystem,
        atlas: &mut TextAtlas,
        glyph: &cosmic_text::LayoutGlyph,
    ) -> Result<crate::glyph_cache::GlyphEntry, PrepareError> {
        let key = GlyphKey::from_layout_glyph(glyph);

        if let Some(e) = atlas.glyphs.get_and_mark_used(&key) {
            return Ok(e);
        }

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
            let units_per_em = skrifa::FontRef::from_index(font.data(), face_index)
                .ok()
                .and_then(|f| {
                    use skrifa::raw::TableProvider;
                    f.head().map(|h| h.units_per_em() as f32).ok()
                })
                .unwrap_or(1000.0);
            self.font_cache.insert(
                cache_key,
                CachedFont {
                    font,
                    face_index,
                    units_per_em,
                },
            );
        }
        let cached = &self.font_cache[&cache_key];
        let (font_data, face_index, units_per_em) =
            (cached.font.data(), cached.face_index, cached.units_per_em);

        let wght_tag = skrifa::Tag::new(b"wght");
        let location = [VariationSetting::new(wght_tag, glyph.font_weight.0 as f32)];

        let entry = match extract_outline(font_data, face_index, glyph.glyph_id, &location) {
            Some(mut outline) => {
                if glyph
                    .cache_key_flags
                    .contains(cosmic_text::CacheKeyFlags::FAKE_ITALIC)
                {
                    apply_italic_shear(&mut outline);
                }
                let band_count = band_count_for_curves(outline.curves.len());
                atlas.upload_glyph(device, &outline, band_count, band_count, units_per_em)?
            }
            None => NON_VECTOR_GLYPH,
        };

        Ok(atlas.glyphs.insert_and_mark_used(key, entry))
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

    /// Prepares all of the provided text areas for rendering.
    #[allow(clippy::too_many_arguments)] // matches cryoglyph's API
    pub fn prepare<'a>(
        &mut self,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
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
        if self.glyphs_to_render == 0 {
            return Ok(());
        }

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &viewport.bind_group, &[]);
        pass.set_bind_group(1, &atlas.bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.draw(0..4, 0..self.glyphs_to_render);

        Ok(())
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
