use crate::glyph_cache::{GlyphKey, NON_VECTOR_GLYPH};
use crate::outline::extract_outline;
use crate::prepare::{apply_italic_shear, prepare_outline};
use crate::text_atlas::TextAtlas;
use crate::types::{PrepareError, RenderError, TextArea};
use crate::viewport::Viewport;
use crate::GlyphInstance;

use skrifa::setting::VariationSetting;

use wgpu::{
    Buffer, BufferDescriptor, BufferUsages, CommandEncoder, DepthStencilState, Device,
    MultisampleState, Queue, RenderPass, RenderPipeline, COPY_BUFFER_ALIGNMENT,
};

/// A text renderer that uses the Slug algorithm to render text into an
/// existing render pass.
pub struct TextRenderer {
    vertex_buffer: Buffer,
    vertex_buffer_size: u64,
    pipeline: RenderPipeline,
    instances: Vec<GlyphInstance>,
    glyphs_to_render: u32,
    units_per_em_cache: std::collections::HashMap<cosmic_text::fontdb::ID, f32>,
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
            units_per_em_cache: std::collections::HashMap::new(),
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
                    let entry = self.resolve_glyph(device, queue, font_system, atlas, glyph)?;

                    if entry.is_non_vector() {
                        continue;
                    }

                    // --- Phase 2: Font metrics ---
                    let units_per_em = match resolve_units_per_em(
                        &mut self.units_per_em_cache,
                        font_system,
                        glyph.font_id,
                        glyph.font_weight,
                    ) {
                        Some(v) => v,
                        None => continue,
                    };

                    // --- Phase 3: Screen position + culling ---
                    let scale = glyph.font_size * text_area.scale / units_per_em;
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
                            entry.band_offset % crate::BAND_TEXTURE_WIDTH,
                            entry.band_offset / crate::BAND_TEXTURE_WIDTH,
                            entry.band_max_x,
                            entry.band_max_y,
                        ],
                        color,
                        depth: metadata_to_depth(glyph.metadata),
                    });
                }
            }
        }

        self.upload_vertices(device, queue);
        Ok(())
    }

    /// Resolve a glyph: return cached entry or extract + upload on miss.
    fn resolve_glyph(
        &mut self,
        device: &Device,
        queue: &Queue,
        font_system: &mut cosmic_text::FontSystem,
        atlas: &mut TextAtlas,
        glyph: &cosmic_text::LayoutGlyph,
    ) -> Result<crate::glyph_cache::GlyphEntry, PrepareError> {
        let key = GlyphKey::from_layout_glyph(glyph);

        if let Some(e) = atlas.glyphs.get_and_mark_used(&key) {
            return Ok(e);
        }

        // Cache miss — extract outline and upload
        let face_index = font_system
            .db()
            .face(glyph.font_id)
            .map(|info| info.index)
            .unwrap_or(0);
        let font = match font_system.get_font(glyph.font_id, glyph.font_weight) {
            Some(f) => f,
            None => {
                log::warn!("Font not found for glyph {key:?}");
                // Cache as non-vector to avoid re-attempting every frame
                return Ok(atlas.glyphs.insert_and_mark_used(key, NON_VECTOR_GLYPH));
            }
        };

        // Populate units_per_em cache while we have the font ref
        if let std::collections::hash_map::Entry::Vacant(e) = self.units_per_em_cache.entry(glyph.font_id)
            && let Ok(skrifa_font) = skrifa::FontRef::from_index(font.data(), face_index)
        {
            use skrifa::raw::TableProvider;
            let v = skrifa_font.head().map(|h| h.units_per_em() as f32).unwrap_or(1000.0);
            e.insert(v);
        }

        let wght_tag = skrifa::Tag::new(b"wght");
        let location = [VariationSetting::new(wght_tag, glyph.font_weight.0 as f32)];

        let entry = match extract_outline(font.data(), face_index, glyph.glyph_id, &location) {
            Some(outline) => {
                let mut gpu_outline = prepare_outline(&outline);
                if glyph.cache_key_flags.contains(cosmic_text::CacheKeyFlags::FAKE_ITALIC) {
                    apply_italic_shear(&mut gpu_outline);
                }
                let band_count = band_count_for_curves(gpu_outline.curves.len());
                atlas.upload_glyph(device, queue, &gpu_outline, band_count, band_count)?
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
            self.vertex_buffer
                .slice(..)
                .get_mapped_range_mut()[..vertices_raw.len()]
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
fn band_count_for_curves(num_curves: usize) -> u32 {
    if num_curves < 10 {
        4
    } else if num_curves < 30 {
        8
    } else {
        12
    }
}

/// Look up units_per_em for a font, populating the cache on miss.
fn resolve_units_per_em(
    cache: &mut std::collections::HashMap<cosmic_text::fontdb::ID, f32>,
    font_system: &mut cosmic_text::FontSystem,
    font_id: cosmic_text::fontdb::ID,
    font_weight: cosmic_text::Weight,
) -> Option<f32> {
    if let Some(&v) = cache.get(&font_id) {
        return Some(v);
    }
    let face_index = font_system
        .db()
        .face(font_id)
        .map(|info| info.index)
        .unwrap_or(0);
    let font = font_system.get_font(font_id, font_weight)?;
    let skrifa_font = skrifa::FontRef::from_index(font.data(), face_index).ok()?;
    use skrifa::raw::TableProvider;
    let v = skrifa_font
        .head()
        .map(|h| h.units_per_em() as f32)
        .unwrap_or(1000.0);
    cache.insert(font_id, v);
    Some(v)
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
