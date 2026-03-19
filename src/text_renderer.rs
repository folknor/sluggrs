use crate::band::build_bands;
use crate::glyph_cache::GlyphKey;
use crate::outline::extract_outline;
use crate::prepare::prepare_outline;
use crate::text_atlas::TextAtlas;
use crate::types::{PrepareError, RenderError, TextArea};
use crate::viewport::Viewport;
use crate::GlyphInstance;

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
        }
    }

    /// Prepares all of the provided text areas for rendering, with depth.
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

            for run in layout_runs {
                for glyph in run.glyphs.iter() {
                    let key = GlyphKey::from_layout_glyph(glyph);

                    // Cache lookup or extract
                    if !atlas.glyphs.contains_key(&key) {
                        let font = font_system.get_font(glyph.font_id, glyph.font_weight);
                        let entry = match font {
                            Some(font) => {
                                match extract_outline(font.data(), glyph.glyph_id) {
                                    Some(outline) => {
                                        let gpu_outline = prepare_outline(&outline);
                                        let num_curves = gpu_outline.curves.len();
                                        let band_count = if num_curves < 10 {
                                            4
                                        } else if num_curves < 30 {
                                            8
                                        } else {
                                            12
                                        };
                                        // Build bands with dummy locations — upload_glyph
                                        // will rebuild with absolute locations
                                        let dummy_locs: Vec<_> = (0..num_curves)
                                            .map(|i| crate::band::CurveLocation {
                                                x: (i as u32) * 2,
                                                y: 0,
                                            })
                                            .collect();
                                        let band_data = build_bands(
                                            &gpu_outline,
                                            &dummy_locs,
                                            band_count,
                                            band_count,
                                        );
                                        atlas.upload_glyph(device, queue, &gpu_outline, &band_data)
                                    }
                                    None => {
                                        // Non-vector glyph
                                        atlas.mark_non_vector(key);
                                        continue;
                                    }
                                }
                            }
                            None => {
                                log::warn!("Font not found for glyph {:?}", key);
                                continue;
                            }
                        };
                        atlas.glyphs.insert(key, entry);
                    }

                    let entry = match atlas.glyphs.get(&key) {
                        Some(e) if !e.is_non_vector() => e,
                        _ => continue, // Skip non-vector or missing
                    };

                    // Get font metrics for scaling
                    let font = match font_system.get_font(glyph.font_id, glyph.font_weight) {
                        Some(f) => f,
                        None => continue,
                    };
                    let units_per_em = {
                        let skrifa_font = match skrifa::FontRef::new(font.data()) {
                            Ok(f) => f,
                            Err(_) => continue,
                        };
                        use skrifa::raw::TableProvider;
                        skrifa_font.head().map(|h| h.units_per_em() as f32).unwrap_or(1000.0)
                    };

                    let scale = glyph.font_size * text_area.scale / units_per_em;
                    let [min_x, min_y, max_x, max_y] = entry.bounds;

                    // Screen position: glyph position from layout + text area offset
                    let glyph_x = text_area.left + (glyph.x + glyph.x_offset) * text_area.scale;
                    let glyph_y = text_area.top + (run.line_y + glyph.y_offset) * text_area.scale;

                    // Screen rect: position + size of the quad
                    let screen_x = glyph_x + min_x * scale;
                    let screen_y = glyph_y - max_y * scale; // flip Y: font is Y-up
                    let screen_w = (max_x - min_x) * scale;
                    let screen_h = (max_y - min_y) * scale;

                    // Skip if entirely outside bounds
                    if screen_x + screen_w < bounds_min_x as f32
                        || screen_x > bounds_max_x as f32
                        || screen_y + screen_h < bounds_min_y as f32
                        || screen_y > bounds_max_y as f32
                    {
                        continue;
                    }

                    let color = match glyph.color_opt {
                        Some(c) => {
                            let r = c.r() as f32 / 255.0;
                            let g = c.g() as f32 / 255.0;
                            let b = c.b() as f32 / 255.0;
                            let a = c.a() as f32 / 255.0;
                            [r, g, b, a]
                        }
                        None => {
                            let c = text_area.default_color;
                            let r = c.r() as f32 / 255.0;
                            let g = c.g() as f32 / 255.0;
                            let b = c.b() as f32 / 255.0;
                            let a = c.a() as f32 / 255.0;
                            [r, g, b, a]
                        }
                    };

                    let _depth = metadata_to_depth(glyph.metadata);

                    self.instances.push(GlyphInstance {
                        screen_rect: [screen_x, screen_y, screen_w, screen_h],
                        em_rect: [min_x, min_y, max_x, max_y],
                        band_transform: entry.band_transform,
                        glyph_data: [entry.band_offset, 0, entry.band_max_x, entry.band_max_y],
                        color,
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
        pass.set_bind_group(0, &atlas.bind_group, &[]);
        pass.set_bind_group(1, &viewport.bind_group, &[]);
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
