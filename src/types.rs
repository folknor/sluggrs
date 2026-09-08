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

/// A solid shape drawn underneath monochrome vector glyphs: an outline, a
/// hard drop shadow, or both.
///
/// The glyph is dilated by `spread` and translated by `offset`, then filled
/// with `color`. `spread` alone gives the outline; `offset` alone gives a
/// hard shadow of the exact glyph shape; together they give a spread shadow.
///
/// This is a MORPHOLOGICAL effect, computed from the signed distance to the
/// glyph boundary. It cannot express a blurred shadow, which is a
/// convolution of the whole glyph mask rather than a function of the nearest
/// boundary distance.
///
/// COLRv0 layers, COLRv1 glyphs, and raster fallback glyphs are deliberately
/// excluded. Raster fallback is rendered in the renderer's final raster tail,
/// after all per-area vector decoration and fill draws.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextDecoration {
    pub color: cosmic_text::Color,
    /// Dilation radius in logical pixels. Zero is meaningful: it gives an
    /// undilated copy of the glyph, which is what a plain drop shadow is.
    /// Negative (erosion) is not supported and drops the decoration.
    pub spread: f32,
    /// Translation in logical pixels, positive y downward. Does not enter the
    /// glyph's distance-query radius - it moves the quad, not the query.
    pub offset: [f32; 2],
}

impl TextDecoration {
    /// A plain outline: dilation with no offset, the shipped border shape.
    pub fn outline(color: cosmic_text::Color, width: f32) -> Self {
        Self {
            color,
            spread: width,
            offset: [0.0, 0.0],
        }
    }

    /// A hard drop shadow: the glyph shape translated, undilated.
    pub fn shadow(color: cosmic_text::Color, dx: f32, dy: f32) -> Self {
        Self {
            color,
            spread: 0.0,
            offset: [dx, dy],
        }
    }
}

/// A decoration resolved to physical pixels, ready for the GPU.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PhysicalDecoration {
    pub color: [f32; 4],
    pub spread: f32,
    pub offset: [f32; 2],
}

/// Resolve one decoration to physical pixels, or `None` if it cannot be
/// drawn. Validates the PHYSICAL result so a finite logical value that
/// overflows under `scale` is rejected too.
pub(crate) fn physical_decoration(
    decoration: TextDecoration,
    scale: f32,
) -> Option<PhysicalDecoration> {
    if !scale.is_finite() || scale <= 0.0 {
        return None;
    }
    let spread = decoration.spread * scale;
    let offset = [decoration.offset[0] * scale, decoration.offset[1] * scale];
    // Spread of exactly zero is valid - that is an unspread drop shadow.
    if !spread.is_finite() || spread < 0.0 || !offset[0].is_finite() || !offset[1].is_finite() {
        return None;
    }
    Some(PhysicalDecoration {
        color: crate::text_renderer::color_to_f32(decoration.color),
        spread,
        offset,
    })
}

/// How far an area's decorations extend past its glyph fills, per side, in
/// physical pixels. Culling needs the four sides separately: an offset is
/// directional, so flipping its sign at constant magnitude reveals glyphs on
/// the opposite side that a single scalar margin would have kept hidden.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct DecorationExtents {
    pub left: f32,
    pub right: f32,
    pub top: f32,
    pub bottom: f32,
}

impl DecorationExtents {
    /// Component-wise union over a resolved decoration list. `aa` is the
    /// antialiasing allowance the shader adds to every dilation.
    pub fn of(decorations: &[PhysicalDecoration], aa: f32) -> Self {
        let mut extents = Self::default();
        for decoration in decorations {
            let radius = decoration.spread + aa;
            let [dx, dy] = decoration.offset;
            extents.left = extents.left.max(radius - dx);
            extents.right = extents.right.max(radius + dx);
            extents.top = extents.top.max(radius - dy);
            extents.bottom = extents.bottom.max(radius + dy);
        }
        extents
    }
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
/// `decorations` affect monochrome vector glyphs only. COLRv0, COLRv1, and
/// raster fallback glyphs remain undecorated. Raster fallback glyphs retain
/// the global raster-tail ordering and are drawn after the per-area vector
/// draws.
///
/// Decoration ORDER IS BACK-TO-FRONT, matching CSS `text-shadow`: the first
/// entry paints on top of later ones, and all of them paint under the fill.
#[derive(Clone, Copy)]
pub struct TextArea<'a> {
    pub buffer: &'a cosmic_text::Buffer,
    pub left: f32,
    pub top: f32,
    pub scale: f32,
    pub bounds: TextBounds,
    pub default_color: cosmic_text::Color,
    pub decorations: &'a [TextDecoration],
}

impl TextArea<'_> {
    /// Resolve every drawable decoration to physical pixels, in input order.
    pub(crate) fn physical_decorations(&self) -> Vec<PhysicalDecoration> {
        self.decorations
            .iter()
            .filter_map(|d| physical_decoration(*d, self.scale))
            .collect()
    }
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

    fn white() -> cosmic_text::Color {
        cosmic_text::Color::rgb(255, 255, 255)
    }

    #[test]
    fn physical_decoration_scales_spread_and_offset() {
        let decoration = TextDecoration {
            color: white(),
            spread: 2.0,
            offset: [3.0, -4.0],
        };
        let physical = physical_decoration(decoration, 1.5).expect("valid");
        assert_eq!(physical.spread, 3.0);
        assert_eq!(physical.offset, [4.5, -6.0]);
    }

    /// Zero spread is a plain drop shadow, not an absent decoration. This is
    /// the one place the decoration rule differs from the old border rule,
    /// which treated zero width as nothing to draw.
    #[test]
    fn zero_spread_is_a_valid_shadow() {
        let shadow = TextDecoration::shadow(white(), 2.0, 2.0);
        let physical = physical_decoration(shadow, 1.0).expect("zero spread is drawable");
        assert_eq!(physical.spread, 0.0);
        assert_eq!(physical.offset, [2.0, 2.0]);
    }

    #[test]
    fn physical_decoration_validation() {
        let with_spread = |spread| TextDecoration {
            color: white(),
            spread,
            offset: [0.0, 0.0],
        };
        for spread in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0] {
            assert_eq!(physical_decoration(with_spread(spread), 1.0), None);
        }
        // Finite logically, overflows under scale.
        assert_eq!(physical_decoration(with_spread(f32::MAX), 2.0), None);

        let with_offset = |offset| TextDecoration {
            color: white(),
            spread: 1.0,
            offset,
        };
        assert_eq!(physical_decoration(with_offset([f32::NAN, 0.0]), 1.0), None);
        assert_eq!(
            physical_decoration(with_offset([0.0, f32::INFINITY]), 1.0),
            None
        );
        assert_eq!(
            physical_decoration(with_offset([f32::MAX, 0.0]), 4.0),
            None,
            "offset must be rejected on physical overflow like spread is"
        );

        for scale in [0.0, -1.0, f32::NAN] {
            assert_eq!(physical_decoration(with_spread(1.0), scale), None);
        }
    }

    /// A directional offset must widen only the side it points at. A scalar
    /// margin cannot express this, which is why the cache tracks four sides.
    #[test]
    fn extents_are_directional() {
        let decoration = PhysicalDecoration {
            color: [1.0; 4],
            spread: 1.0,
            offset: [4.0, 0.0],
        };
        let extents = DecorationExtents::of(&[decoration], 0.5);
        assert_eq!(extents.right, 5.5, "radius 1.5 plus offset 4 to the right");
        assert_eq!(extents.left, 0.0, "offset 4 exceeds radius 1.5, so clamped");
        assert_eq!(extents.top, 1.5);
        assert_eq!(extents.bottom, 1.5);
    }

    /// Two shadows pointing opposite ways must widen BOTH sides. Tracking a
    /// maximum radius alone would lose one of them.
    #[test]
    fn extents_union_opposing_offsets() {
        let at = |dx: f32| PhysicalDecoration {
            color: [1.0; 4],
            spread: 0.0,
            offset: [dx, 0.0],
        };
        let extents = DecorationExtents::of(&[at(6.0), at(-3.0)], 0.5);
        assert_eq!(extents.right, 6.5);
        assert_eq!(extents.left, 3.5);
    }

    #[test]
    fn empty_decoration_list_has_no_extents() {
        assert_eq!(
            DecorationExtents::of(&[], 0.5),
            DecorationExtents::default()
        );
    }
}
