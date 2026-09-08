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
    ColorMode, DecorationError, DecorationMode, PrepareError, RenderError, Resolution, TextArea,
    TextBounds, TextDecoration, validate_decorations,
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

    /// A zero-spread decoration is specified to reproduce the glyph shape
    /// exactly, which holds only if both vertex shaders map screen pixels to
    /// em space identically. The denominator is the place they can silently
    /// disagree: clamping to one pixel versus guarding the raw dimension
    /// near zero differ for any glyph dimension between 0 and 1 px, and the
    /// divergence appears only on sub-pixel glyphs that no ordinary scene
    /// renders.
    #[test]
    fn both_vertex_shaders_use_the_same_ems_per_pixel_denominator() {
        const DENOMINATOR: &str = "em_size / max(instance.screen_rect.zw, vec2<f32>(1.0, 1.0))";
        assert!(
            super::SIMPLE_SHADER_WGSL.contains(DENOMINATOR),
            "fill vertex shader changed its screen-to-em mapping"
        );
        assert!(
            include_str!("border_shader.wgsl").contains(DENOMINATOR),
            "decoration vertex shader must match the fill's screen-to-em mapping"
        );
    }
}
