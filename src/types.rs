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

/// A solid outline drawn underneath monochrome vector glyphs.
///
/// COLRv0 layers, COLRv1 glyphs, and raster fallback glyphs are deliberately
/// excluded. Raster fallback is rendered in the renderer's final raster tail,
/// after all per-area vector border and fill draws.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextBorder {
    pub color: cosmic_text::Color,
    /// Width in logical pixels. Non-finite, zero, and negative widths are ignored.
    pub width: f32,
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
///
/// `border` affects monochrome vector glyphs only. COLRv0, COLRv1, and raster
/// fallback glyphs remain borderless. Raster fallback glyphs retain the global
/// raster-tail ordering and are drawn after the per-area vector draws.
#[derive(Clone, Copy)]
pub struct TextArea<'a> {
    pub buffer: &'a cosmic_text::Buffer,
    pub left: f32,
    pub top: f32,
    pub scale: f32,
    pub bounds: TextBounds,
    pub default_color: cosmic_text::Color,
    pub border: Option<TextBorder>,
}

impl TextArea<'_> {
    /// Returns a validated border width in physical pixels.
    pub(crate) fn physical_border_width(&self) -> Option<f32> {
        physical_border_width(self.border, self.scale)
    }
}

pub(crate) fn physical_border_width(border: Option<TextBorder>, scale: f32) -> Option<f32> {
    let logical = border?.width;
    if !logical.is_finite() || logical <= 0.0 || !scale.is_finite() || scale <= 0.0 {
        return None;
    }
    let width = logical * scale;
    (width.is_finite() && width > 0.0).then_some(width)
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
/// `render()` returns `RemovedFromAtlas` when the prepared atlas identity or
/// generation no longer matches the atlas passed to render.
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
                    "Render error: prepared atlas data is invalid or unavailable"
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

    #[test]
    fn physical_border_width_validation() {
        let border = |width| {
            Some(TextBorder {
                color: cosmic_text::Color::rgb(255, 255, 255),
                width,
            })
        };
        assert_eq!(physical_border_width(border(2.0), 1.5), Some(3.0));
        for width in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0, 0.0] {
            assert_eq!(physical_border_width(border(width), 1.0), None);
        }
        assert_eq!(physical_border_width(border(f32::MAX), 2.0), None);
        assert_eq!(physical_border_width(border(-2.0), -3.0), None);
        assert_eq!(physical_border_width(None, 1.0), None);
    }
}
