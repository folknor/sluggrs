use crate::glyph_cache::GlyphKey;
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

    /// Prepares all of the provided text areas for rendering, with depth.
    #[allow(clippy::too_many_arguments)] // matches cryoglyph's API
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

        // Persistent cache — avoids HashMap allocation per frame and
        // avoids re-parsing skrifa FontRef + head table on warm frames.

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

            // Precompute default color for this text area (avoids per-glyph division)
            let dc = text_area.default_color;
            let default_color = [
                dc.r() as f32 / 255.0,
                dc.g() as f32 / 255.0,
                dc.b() as f32 / 255.0,
                dc.a() as f32 / 255.0,
            ];

            for run in layout_runs {
                for glyph in run.glyphs {
                    let key = GlyphKey::from_layout_glyph(glyph);

                    // Cache lookup or extract
                    if !atlas.glyphs.contains_key(&key) {
                        // Face index within font collections (TTC)
                        let face_index = font_system
                            .db()
                            .face(glyph.font_id)
                            .map(|info| info.index)
                            .unwrap_or(0);
                        let font = font_system.get_font(glyph.font_id, glyph.font_weight);
                        let entry = match font {
                            Some(font) => {
                                // Populate units_per_em cache while we have the font ref,
                                // avoiding a second get_font() call on the hot path below.
                                if !self.units_per_em_cache.contains_key(&glyph.font_id) {
                                    if let Ok(skrifa_font) = skrifa::FontRef::from_index(font.data(), face_index) {
                                        use skrifa::raw::TableProvider;
                                        let v = skrifa_font.head().map(|h| h.units_per_em() as f32).unwrap_or(1000.0);
                                        self.units_per_em_cache.insert(glyph.font_id, v);
                                    }
                                }

                                // Set up variation coordinates (weight axis for variable fonts)
                                let wght_tag = skrifa::Tag::new(b"wght");
                                let location = [VariationSetting::new(wght_tag, glyph.font_weight.0 as f32)];
                                match extract_outline(font.data(), face_index, glyph.glyph_id, &location) {
                                    Some(outline) => {
                                        let mut gpu_outline = prepare_outline(&outline);
                                        if glyph.cache_key_flags.contains(
                                            cosmic_text::CacheKeyFlags::FAKE_ITALIC,
                                        ) {
                                            apply_italic_shear(&mut gpu_outline);
                                        }
                                        let num_curves = gpu_outline.curves.len();
                                        let band_count = if num_curves < 10 {
                                            4
                                        } else if num_curves < 30 {
                                            8
                                        } else {
                                            12
                                        };
                                        atlas.upload_glyph(device, queue, &gpu_outline, band_count, band_count)
                                    }
                                    None => {
                                        // Non-vector glyph — mark cached and in-use
                                        atlas.mark_non_vector(key);
                                        atlas.glyphs.mark_used(key);
                                        continue;
                                    }
                                }
                            }
                            None => {
                                log::warn!("Font not found for glyph {key:?}");
                                continue;
                            }
                        };
                        atlas.glyphs.insert(key, entry);
                    }

                    let entry = match atlas.glyphs.get(&key) {
                        Some(&e) => {
                            atlas.glyphs.mark_used(key);
                            if e.is_non_vector() {
                                continue;
                            }
                            e
                        }
                        _ => continue,
                    };

                    // Get font metrics for scaling (cached per font_id).
                    // Usually populated in the glyph cache-miss block above;
                    // this fallback handles glyphs cached from a prior frame.
                    let units_per_em = match self.units_per_em_cache.get(&glyph.font_id) {
                        Some(&v) => v,
                        None => {
                            let face_index = font_system
                                .db()
                                .face(glyph.font_id)
                                .map(|info| info.index)
                                .unwrap_or(0);
                            let font = match font_system.get_font(glyph.font_id, glyph.font_weight) {
                                Some(f) => f,
                                None => continue,
                            };
                            let v = {
                                let skrifa_font = match skrifa::FontRef::from_index(font.data(), face_index) {
                                    Ok(f) => f,
                                    Err(_) => continue,
                                };
                                use skrifa::raw::TableProvider;
                                skrifa_font.head().map(|h| h.units_per_em() as f32).unwrap_or(1000.0)
                            };
                            self.units_per_em_cache.insert(glyph.font_id, v);
                            v
                        }
                    };

                    let scale = glyph.font_size * text_area.scale / units_per_em;
                    let [min_x, min_y, max_x, max_y] = entry.bounds;

                    // Screen position: glyph position from layout + text area offset
                    let glyph_x = text_area.left + (glyph.x + glyph.x_offset) * text_area.scale;
                    let glyph_y = text_area.top + (run.line_y + glyph.y_offset) * text_area.scale;

                    // Screen rect: undilated quad (shader handles 1px dilation)
                    let screen_x = glyph_x + min_x * scale;
                    let screen_y = glyph_y - max_y * scale; // flip Y: font is Y-up
                    let screen_w = (max_x - min_x) * scale;
                    let screen_h = (max_y - min_y) * scale;

                    // Skip if entirely outside bounds (1px margin for shader dilation)
                    if screen_x + screen_w + 1.0 < bounds_min_x as f32
                        || screen_x - 1.0 > bounds_max_x as f32
                        || screen_y + screen_h + 1.0 < bounds_min_y as f32
                        || screen_y - 1.0 > bounds_max_y as f32
                    {
                        continue;
                    }

                    let color = match glyph.color_opt {
                        Some(c) => [
                            c.r() as f32 / 255.0,
                            c.g() as f32 / 255.0,
                            c.b() as f32 / 255.0,
                            c.a() as f32 / 255.0,
                        ],
                        None => default_color,
                    };

                    let depth = metadata_to_depth(glyph.metadata);

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
                        depth,
                    });
                }
            }
        }

        self.glyphs_to_render = self.instances.len() as u32;

        if self.instances.is_empty() {
            return Ok(());
        }

        // Upload vertex buffer
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

        Ok(())
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

fn next_copy_buffer_size(size: u64) -> u64 {
    let align_mask = COPY_BUFFER_ALIGNMENT - 1;
    ((size.next_power_of_two() + align_mask) & !align_mask).max(COPY_BUFFER_ALIGNMENT)
}

fn zero_depth(_: usize) -> f32 {
    0.0
}
