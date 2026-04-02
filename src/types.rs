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
#[derive(Clone, Copy)]
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
///
/// Currently `render()` always returns `Ok(())` — these variants exist for
/// cryoglyph API compatibility but are never produced. If trim/atlas-reset
/// detection is added in the future, `RemovedFromAtlas` would be returned.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RenderError {
    RemovedFromAtlas,
    ScreenResolutionChanged,
}

impl std::fmt::Display for RenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            RenderError::RemovedFromAtlas => {
                write!(
                    f,
                    "Render error: glyph no longer exists within the texture atlas"
                )
            }
            RenderError::ScreenResolutionChanged => {
                write!(
                    f,
                    "Render error: screen resolution changed since last prepare call"
                )
            }
        }
    }
}

impl std::error::Error for RenderError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_bounds_default_is_unbounded() {
        let bounds = TextBounds::default();
        assert_eq!(bounds.left, i32::MIN);
        assert_eq!(bounds.top, i32::MIN);
        assert_eq!(bounds.right, i32::MAX);
        assert_eq!(bounds.bottom, i32::MAX);
    }

    #[test]
    fn prepare_error_display() {
        let err = PrepareError::AtlasFull;
        let msg = format!("{err}");
        assert!(msg.contains("atlas"), "Display should mention atlas: {msg}");
    }

    #[test]
    fn render_error_display_variants() {
        let msg1 = format!("{}", RenderError::RemovedFromAtlas);
        assert!(msg1.contains("atlas"), "Should mention atlas: {msg1}");

        let msg2 = format!("{}", RenderError::ScreenResolutionChanged);
        assert!(
            msg2.contains("resolution"),
            "Should mention resolution: {msg2}"
        );
    }

    #[test]
    fn error_types_implement_error_trait() {
        let pe: Box<dyn std::error::Error> = Box::new(PrepareError::AtlasFull);
        assert!(pe.to_string().contains("atlas"));

        let re: Box<dyn std::error::Error> = Box::new(RenderError::RemovedFromAtlas);
        assert!(re.to_string().contains("atlas"));
    }

    #[test]
    fn error_types_are_copy() {
        let e1 = PrepareError::AtlasFull;
        let e2 = e1; // Copy
        assert_eq!(e1, e2);

        let r1 = RenderError::RemovedFromAtlas;
        let r2 = r1; // Copy
        assert_eq!(r1, r2);
    }

    #[test]
    fn resolution_equality() {
        let a = Resolution {
            width: 1920,
            height: 1080,
        };
        let b = Resolution {
            width: 1920,
            height: 1080,
        };
        let c = Resolution {
            width: 1280,
            height: 720,
        };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn color_mode_equality() {
        assert_eq!(ColorMode::Accurate, ColorMode::Accurate);
        assert_ne!(ColorMode::Accurate, ColorMode::Web);
    }
}
