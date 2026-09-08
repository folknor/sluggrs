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
pub mod border;
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
pub use types::{
    ColorMode, PrepareError, RenderError, Resolution, TextArea, TextBorder, TextBounds,
};
pub use viewport::Viewport;

// Re-export cosmic_text types that iced's text.rs uses via cryoglyph
pub use cosmic_text::{self, Buffer, CacheKey, Color, FontSystem, SwashCache};

// Shader sources
pub const SIMPLE_SHADER_WGSL: &str = include_str!("simple_shader.wgsl");
/// Normal shader assembled from its source fragments. Kept separate from the
/// border module so the normal GPU path remains byte-for-byte stable.
pub const ASSEMBLED_SIMPLE_SHADER_WGSL: &str = concat!(include_str!("simple_shader.wgsl"));
pub(crate) const BORDER_SHADER_WGSL: &str = concat!(
    include_str!("simple_shader.wgsl"),
    "\n",
    include_str!("border_shader.wgsl")
);
// Full shader (with dilation) is not yet synced with simple_shader fixes.
// Kept internal until it's brought up to parity.
const _SHADER_WGSL: &str = include_str!("shader.wgsl");

/// Per-instance vertex data for a glyph (matches GlyphInstance in shader).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GlyphInstance {
    pub screen_rect: [f32; 4],
    pub color: [f32; 4],
    pub glyph_offset: u32,
    pub cmd_texel_count: u32,
    pub depth: f32,
    pub ppem: f32,
}

const _: () = assert!(std::mem::size_of::<GlyphInstance>() == 48);

#[cfg(test)]
mod glyph_instance_tests {
    use super::GlyphInstance;
    use std::mem::{offset_of, size_of};

    #[test]
    fn glyph_instance_abi() {
        assert_eq!(size_of::<GlyphInstance>(), 48);
        assert_eq!(offset_of!(GlyphInstance, screen_rect), 0);
        assert_eq!(offset_of!(GlyphInstance, color), 16);
        assert_eq!(offset_of!(GlyphInstance, glyph_offset), 32);
        assert_eq!(offset_of!(GlyphInstance, cmd_texel_count), 36);
        assert_eq!(offset_of!(GlyphInstance, depth), 40);
        assert_eq!(offset_of!(GlyphInstance, ppem), 44);
    }

    #[test]
    fn assembled_normal_shader_is_byte_identical() {
        assert_eq!(
            super::ASSEMBLED_SIMPLE_SHADER_WGSL,
            super::SIMPLE_SHADER_WGSL
        );
    }
}
