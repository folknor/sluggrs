// wgpu's backend type graph (Global -> Hub -> Registry -> RwLock -> Storage ->
// ...) nests deeper than the default limit of 128, so auto-trait resolution for
// anything holding a wgpu resource (e.g. `gpu_cache::Inner: Send`) overflows.
#![recursion_limit = "256"]

// Public API modules - stable interface matching cryoglyph
pub mod gpu_cache;
pub mod text_atlas;
pub mod text_renderer;
pub mod types;
pub mod viewport;

// Low-level modules - public for custom renderers (like the demo) but
// not part of the stable iced integration API. Internal representations
// may change.
pub mod band;
pub(crate) mod blob_cache;
pub mod glyph_cache;
pub mod outline;
pub mod prep;
pub mod prepare;
pub(crate) mod raster_text;

// Public API - matches cryoglyph's interface for iced integration
#[doc(hidden)]
pub use blob_cache::BlobCacheStats;
pub use glyph_cache::GlyphKey;
pub use gpu_cache::Cache;
pub use text_atlas::TextAtlas;
pub use text_renderer::TextRenderer;
pub use types::{ColorMode, PrepareError, RenderError, Resolution, TextArea, TextBounds};
pub use viewport::Viewport;

// Re-export cosmic_text types that iced's text.rs uses via cryoglyph
pub use cosmic_text::{self, Buffer, CacheKey, Color, FontSystem, SwashCache};
use wgpu::VertexFormat;

// Shader sources
pub const SIMPLE_SHADER_WGSL: &str = include_str!("simple_shader.wgsl");
// Full shader (with dilation) is not yet synced with simple_shader fixes.
// Kept internal until it's brought up to parity.
const _SHADER_WGSL: &str = include_str!("shader.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GlyphInstance {
    pub screen_rect: [f32; 4],
    pub color: [f32; 4],
    pub border_color: [f32; 4],
    pub glyph_offset: u32,
    pub cmd_texel_count: u32,
    pub depth: f32,
    pub ppem: f32,
    pub border_width: f32,
}

impl GlyphInstance {
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: size_of::<GlyphInstance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: 16,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: 32,
                    shader_location: 2,
                },
                wgpu::VertexAttribute {
                    format: VertexFormat::Uint32,
                    offset: 48,
                    shader_location: 3,
                },
                wgpu::VertexAttribute {
                    format: VertexFormat::Uint32,
                    offset: 52,
                    shader_location: 4,
                },
                wgpu::VertexAttribute {
                    format: VertexFormat::Float32,
                    offset: 56,
                    shader_location: 5,
                },
                wgpu::VertexAttribute {
                    format: VertexFormat::Float32,
                    offset: 60,
                    shader_location: 6,
                },
                wgpu::VertexAttribute {
                    format: VertexFormat::Float32,
                    offset: 64,
                    shader_location: 7,
                },
            ]
        }
    }
}

const _: () = assert!(size_of::<GlyphInstance>() == 68);

#[cfg(test)]
mod glyph_instance_tests {
    use super::GlyphInstance;
    use std::mem::{offset_of, size_of};

    // #[test]
    // fn glyph_instance_abi() {
    //     assert_eq!(size_of::<GlyphInstance>(), 80);
    //     assert_eq!(offset_of!(GlyphInstance, screen_rect), 0);
    //     assert_eq!(offset_of!(GlyphInstance, color), 16);
    //     assert_eq!(offset_of!(GlyphInstance, glyph_offset), 32);
    //     assert_eq!(offset_of!(GlyphInstance, cmd_texel_count), 36);
    //     assert_eq!(offset_of!(GlyphInstance, depth), 40);
    //     assert_eq!(offset_of!(GlyphInstance, ppem), 44);
    // }
}
