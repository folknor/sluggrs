pub mod band;
pub mod outline;
pub mod prepare;

pub const SIMPLE_SHADER_WGSL: &str = include_str!("simple_shader.wgsl");
pub const SHADER_WGSL: &str = include_str!("shader.wgsl");

/// Band texture width assumed by the shaders' wrapping logic.
/// The uploaded texture must be exactly this wide; rows wrap at this boundary.
pub const BAND_TEXTURE_WIDTH: u32 = 4096;
