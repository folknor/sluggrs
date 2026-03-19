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
        _device: &Device,
        _queue: &Queue,
        _encoder: &mut CommandEncoder,
        _font_system: &mut cosmic_text::FontSystem,
        _atlas: &mut TextAtlas,
        _viewport: &Viewport,
        _text_areas: impl IntoIterator<Item = TextArea<'a>>,
        _cache: &mut cosmic_text::SwashCache,
        _metadata_to_depth: impl FnMut(usize) -> f32,
    ) -> Result<(), PrepareError> {
        // Phase A stub: do nothing, render no text.
        self.instances.clear();
        self.glyphs_to_render = 0;
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
