/// The screen resolution to use when rendering text.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Resolution {
    pub width: u32,
    pub height: u32,
}

/// Controls the visible area of the text. Any text outside of the visible
/// area will be clipped.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TextBounds {
    pub left: i32,
    pub top: i32,
    pub right: i32,
    pub bottom: i32,
}

impl Default for TextBounds {
    fn default() -> Self {
        Self {
            left: i32::MIN,
            top: i32::MIN,
            right: i32::MAX,
            bottom: i32::MAX,
        }
    }
}

/// A text area containing text to be rendered along with its overflow behavior.
#[derive(Clone)]
pub struct TextArea<'a> {
    pub buffer: &'a cosmic_text::Buffer,
    pub left: f32,
    pub top: f32,
    pub scale: f32,
    pub bounds: TextBounds,
    pub default_color: cosmic_text::Color,
}

/// The color mode of the text atlas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorMode {
    /// Accurate color management (sRGB texture for colored glyphs).
    Accurate,
    /// Web color management (linear RGB texture with sRGB colors).
    Web,
}

/// An error that occurred while preparing text for rendering.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrepareError {
    AtlasFull,
}

impl std::fmt::Display for PrepareError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Prepare error: glyph texture atlas is full")
    }
}

impl std::error::Error for PrepareError {}

/// An error that occurred while rendering text.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RenderError {
    RemovedFromAtlas,
    ScreenResolutionChanged,
}

impl std::fmt::Display for RenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            RenderError::RemovedFromAtlas => {
                write!(f, "Render error: glyph no longer exists within the texture atlas")
            }
            RenderError::ScreenResolutionChanged => {
                write!(f, "Render error: screen resolution changed since last prepare call")
            }
        }
    }
}

impl std::error::Error for RenderError {}
