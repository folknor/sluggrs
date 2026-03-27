// Public API modules — stable interface matching cryoglyph
pub mod gpu_cache;
pub mod text_atlas;
pub mod text_renderer;
pub mod types;
pub mod viewport;

// Low-level modules — public for custom renderers (like the demo) but
// not part of the stable iced integration API. Internal representations
// may change.
pub mod band;
pub mod glyph_cache;
pub mod outline;
pub mod prepare;

// Public API — matches cryoglyph's interface for iced integration
pub use gpu_cache::Cache;
pub use text_atlas::TextAtlas;
pub use text_renderer::TextRenderer;
pub use types::{ColorMode, PrepareError, RenderError, Resolution, TextArea, TextBounds};
pub use viewport::Viewport;

// Re-export cosmic_text types that iced's text.rs uses via cryoglyph
pub use cosmic_text::{
    self, Buffer, CacheKey, Color, FontSystem, SwashCache,
};

// Shader sources
pub const SIMPLE_SHADER_WGSL: &str = include_str!("simple_shader.wgsl");
// Full shader (with dilation) is not yet synced with simple_shader fixes.
// Kept internal until it's brought up to parity.
const _SHADER_WGSL: &str = include_str!("shader.wgsl");

/// Band texture width assumed by the shaders' wrapping logic.
/// The uploaded texture must be exactly this wide; rows wrap at this boundary.
pub const BAND_TEXTURE_WIDTH: u32 = 4096;

/// Per-instance vertex data for a glyph (matches GlyphInstance in shader).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GlyphInstance {
    pub screen_rect: [f32; 4],     // x, y, width, height
    pub em_rect: [f32; 4],         // min_x, min_y, max_x, max_y
    pub band_transform: [f32; 4],  // scale_x, scale_y, offset_x, offset_y
    pub glyph_data: [u32; 4],      // band_loc_x, band_loc_y, band_max_x, band_max_y
    pub color: [f32; 4],           // RGBA
    pub depth: f32,                // z-depth for iced widget layering
}
