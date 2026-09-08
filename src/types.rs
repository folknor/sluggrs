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
    /// Gaussian sigma in logical pixels. Zero means an analytic decoration
    /// drawn from the distance field; above zero the decoration becomes a
    /// FILTERED shadow - the area's glyph coverage is rendered to a mask,
    /// blurred separably, and composited.
    ///
    /// The two are different mechanisms, not a parameter of one another. A
    /// falloff over the nearest-boundary distance is a feathered dilation:
    /// it keeps counters open that a real blur hazes shut, holds thin stems
    /// at full opacity where a blur thins them, and treats touching glyphs as
    /// separate shapes where a blur merges them.
    pub blur: f32,
    pub mode: DecorationMode,
}

/// A Gaussian has infinite support; the kernel is cut off at this many sigma.
/// One axis loses about 0.0063% of its mass beyond 4 sigma. Culling, mask
/// bounds, tap count and composite bounds must all use the SAME rounded
/// support or the shadow will be clipped somewhere in the chain.
pub const BLUR_SUPPORT_SIGMAS: f32 = 4.0;

/// Largest physical sigma that will be drawn.
///
/// This is a resource bound, not a style choice. Support is `4 * sigma` on
/// every side, the mask texture is the shadow's rect grown by it, and the
/// fragment loops `2 * support + 1` times per pixel in each of two passes.
/// Sigma 64 already means a 512px border on every side and a 513-tap kernel;
/// an unbounded value produces a texture past `max_texture_dimension_2d` and
/// a loop long enough to trip the GPU watchdog. Three such textures are built
/// per job, every frame, with no pooling.
pub const MAX_BLUR_SIGMA: f32 = 64.0;

/// Kernel support in physical pixels for a physical sigma.
pub(crate) fn blur_support(sigma: f32) -> f32 {
    if sigma <= 0.0 {
        return 0.0;
    }
    (sigma.min(MAX_BLUR_SIGMA) * BLUR_SUPPORT_SIGMAS).ceil()
}

/// What a decoration paints.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum DecorationMode {
    /// The whole dilated glyph, painted under the fill. An outline when the
    /// fill is opaque, since the fill hides the interior.
    #[default]
    Solid,
    /// Only the band between the glyph edge and the dilated edge, with the
    /// fill painted by the SAME draw as a disjoint region. This is what
    /// outline-only (hollow) text needs: a `Solid` decoration under a
    /// transparent fill is a solid fat glyph, not a hollow one.
    ///
    /// A ring subtracted from an outer coverage and then composited under a
    /// separate fill draw does not reconstruct the union - source-over gives
    /// `o - f(o-f)`, not `o` - so ring and fill must come from one fragment.
    /// That is why the constraints in [`DecorationError`] exist.
    Ring,
}

/// Why a decoration list cannot be rendered as written.
///
/// These are not stylistic limits. Each names a combination the one-fragment
/// ring-plus-fill execution model cannot draw correctly, so they are refused
/// rather than silently mis-rendered.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecorationError {
    /// Two `Ring` decorations would each emit the fill, compositing it twice.
    MultipleRings,
    /// One fragment cannot emit a ring at one screen position and the fill at
    /// another, so a `Ring` must not be displaced from its glyph.
    RingWithOffset,
    /// The `Ring` owns the fill, and the fill paints above every decoration.
    /// A decoration ordered above the ring would paint over that fill, so the
    /// ring must be first in the back-to-front list.
    RingNotTopmost,
    /// A ring is evaluated analytically from the glyph's distance field and a
    /// blur is a convolution of a rendered mask. One draw cannot be both.
    RingWithBlur,
    /// A filtered decoration's mask is rendered from the undilated glyph, so a
    /// spread on it would be silently dropped. Reject rather than mis-render.
    BlurWithSpread,
    /// Sigma past [`MAX_BLUR_SIGMA`] in physical pixels. Support scales with
    /// it in both texture size and per-fragment tap count, so an unbounded
    /// value is a resource hazard rather than a slow frame.
    BlurTooLarge,
}

impl std::fmt::Display for DecorationError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::MultipleRings => write!(f, "at most one Ring decoration per text area"),
            Self::RingWithOffset => write!(f, "a Ring decoration must have a zero offset"),
            Self::RingNotTopmost => {
                write!(f, "a Ring decoration must be first in the decoration list")
            }
            Self::RingWithBlur => write!(f, "a Ring decoration cannot be blurred"),
            Self::BlurWithSpread => {
                write!(f, "a blurred decoration cannot also carry a spread")
            }
            Self::BlurTooLarge => write!(f, "blur sigma exceeds {MAX_BLUR_SIGMA} physical pixels"),
        }
    }
}

impl std::error::Error for DecorationError {}

/// Check the constraints the execution model imposes on a decoration list.
///
/// Convenience wrapper for callers who want to check ahead of time; `prepare`
/// validates the RESOLVED list instead, via [`validate_physical`].
pub fn validate_decorations(decorations: &[TextDecoration]) -> Result<(), DecorationError> {
    let resolved: Vec<PhysicalDecoration> = decorations
        .iter()
        .filter_map(|d| physical_decoration(*d, 1.0))
        .collect();
    validate_physical(&resolved)
}

/// Check the resolved list, which is what actually gets drawn.
///
/// Validating the logical list instead lets the two disagree: a decoration
/// that `physical_decoration` drops (negative spread, non-finite, overflow
/// under scale) is still counted for ordering, so a dropped entry ahead of a
/// ring would report `RingNotTopmost` and fail the whole frame even though
/// the ring really would have been drawn first. A logical sigma that flushes
/// to zero under a small scale would likewise validate as filtered and then
/// draw as analytic.
pub(crate) fn validate_physical(decorations: &[PhysicalDecoration]) -> Result<(), DecorationError> {
    let mut seen_ring = false;
    for (index, decoration) in decorations.iter().enumerate() {
        if decoration.blur > MAX_BLUR_SIGMA {
            return Err(DecorationError::BlurTooLarge);
        }
        if decoration.mode != DecorationMode::Ring {
            // The mask for a filtered decoration is rendered from the
            // undilated glyph, so a spread would be accepted then ignored.
            // Checked after the Ring branch so a blurred ring reports the
            // more specific RingWithBlur.
            if decoration.is_filtered() && decoration.spread > 0.0 {
                return Err(DecorationError::BlurWithSpread);
            }
            continue;
        }
        if seen_ring {
            return Err(DecorationError::MultipleRings);
        }
        seen_ring = true;
        if decoration.offset != [0.0, 0.0] {
            return Err(DecorationError::RingWithOffset);
        }
        if index != 0 {
            return Err(DecorationError::RingNotTopmost);
        }
        if decoration.is_filtered() {
            return Err(DecorationError::RingWithBlur);
        }
    }
    Ok(())
}

impl TextDecoration {
    /// A plain outline: dilation with no offset, the shipped border shape.
    /// Painted under the fill, so an opaque fill hides its interior.
    pub fn outline(color: cosmic_text::Color, width: f32) -> Self {
        Self {
            color,
            spread: width,
            offset: [0.0, 0.0],
            blur: 0.0,
            mode: DecorationMode::Solid,
        }
    }

    /// A hard drop shadow: the glyph shape translated, undilated.
    pub fn shadow(color: cosmic_text::Color, dx: f32, dy: f32) -> Self {
        Self {
            color,
            spread: 0.0,
            offset: [dx, dy],
            blur: 0.0,
            mode: DecorationMode::Solid,
        }
    }

    /// A blurred drop shadow: the CSS `text-shadow` effect. `sigma` is the
    /// Gaussian standard deviation in logical pixels.
    ///
    /// This is a convolution of the area's whole glyph mask, so touching
    /// glyphs blur together and counters haze shut, exactly as in a browser.
    pub fn blurred_shadow(color: cosmic_text::Color, dx: f32, dy: f32, sigma: f32) -> Self {
        Self {
            color,
            spread: 0.0,
            offset: [dx, dy],
            blur: sigma,
            mode: DecorationMode::Solid,
        }
    }

    /// An outline that does NOT paint under the fill: the ring and the fill
    /// are emitted as disjoint regions by one draw. Pair with a transparent
    /// fill color for hollow text, or a translucent one to see the backdrop
    /// through the counter while the ring stays solid.
    pub fn ring(color: cosmic_text::Color, width: f32) -> Self {
        Self {
            color,
            spread: width,
            offset: [0.0, 0.0],
            blur: 0.0,
            mode: DecorationMode::Ring,
        }
    }
}

/// A decoration resolved to physical pixels, ready for the GPU.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PhysicalDecoration {
    pub color: [f32; 4],
    pub spread: f32,
    pub offset: [f32; 2],
    /// Physical sigma. Above zero this is a filtered shadow rather than an
    /// analytic one, and takes an entirely different path through the
    /// renderer: mask render, separable blur, composite.
    pub blur: f32,
    pub mode: DecorationMode,
}

impl PhysicalDecoration {
    pub fn is_filtered(&self) -> bool {
        self.blur > 0.0
    }

    /// How far this decoration's paint reaches past the glyph, before the
    /// offset is applied. Analytic decorations reach by their dilation plus
    /// the antialiasing allowance; a filtered one reaches by its kernel
    /// support, which the mask, the taps and the composite all share.
    pub fn radius(&self, aa: f32) -> f32 {
        if self.is_filtered() {
            self.spread + blur_support(self.blur)
        } else {
            self.spread + aa
        }
    }
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
    let blur = decoration.blur * scale;
    // Spread of exactly zero is valid - that is an unspread drop shadow.
    if !spread.is_finite() || spread < 0.0 || !offset[0].is_finite() || !offset[1].is_finite() {
        return None;
    }
    // A non-finite or negative sigma is not a blur; a zero one is simply the
    // analytic path. Reject the former, allow the latter.
    if !blur.is_finite() || blur < 0.0 || !blur_support(blur).is_finite() {
        return None;
    }
    Some(PhysicalDecoration {
        color: crate::text_renderer::color_to_f32(decoration.color),
        spread,
        offset,
        blur,
        mode: decoration.mode,
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
            let radius = decoration.radius(aa);
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
    /// Resolve every drawable decoration to physical pixels, in input order,
    /// and check the resolved list against what the renderer can draw.
    pub(crate) fn physical_decorations(&self) -> Result<Vec<PhysicalDecoration>, DecorationError> {
        let resolved: Vec<PhysicalDecoration> = self
            .decorations
            .iter()
            .filter_map(|d| physical_decoration(*d, self.scale))
            .collect();
        validate_physical(&resolved)?;
        Ok(resolved)
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
    /// A text area's decoration list names a combination the renderer cannot
    /// draw. Refused rather than mis-rendered.
    Decoration(DecorationError),
}

impl std::fmt::Display for PrepareError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::AtlasFull => write!(f, "Prepare error: glyph texture atlas is full"),
            Self::Decoration(error) => write!(f, "Prepare error: {error}"),
        }
    }
}

impl From<DecorationError> for PrepareError {
    fn from(error: DecorationError) -> Self {
        Self::Decoration(error)
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
            blur: 0.0,
            mode: DecorationMode::Solid,
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
            blur: 0.0,
            mode: DecorationMode::Solid,
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
            blur: 0.0,
            mode: DecorationMode::Solid,
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
            blur: 0.0,
            mode: DecorationMode::Solid,
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
            blur: 0.0,
            mode: DecorationMode::Solid,
        };
        let extents = DecorationExtents::of(&[at(6.0), at(-3.0)], 0.5);
        assert_eq!(extents.right, 6.5);
        assert_eq!(extents.left, 3.5);
    }

    /// Each rule below exists because the one-fragment ring-plus-fill model
    /// cannot draw the rejected combination correctly - not as a style
    /// preference.
    #[test]
    fn ring_constraints_are_enforced() {
        let ring = TextDecoration::ring(white(), 2.0);
        let shadow = TextDecoration::shadow(white(), 2.0, 2.0);

        assert_eq!(validate_decorations(&[]), Ok(()));
        assert_eq!(validate_decorations(&[ring]), Ok(()));
        assert_eq!(validate_decorations(&[shadow, shadow]), Ok(()));
        assert_eq!(
            validate_decorations(&[ring, shadow]),
            Ok(()),
            "a ring above other decorations is the supported arrangement"
        );

        // Two rings would each emit the fill.
        assert_eq!(
            validate_decorations(&[ring, ring]),
            Err(DecorationError::MultipleRings)
        );
        // A decoration above the ring would paint over the fill the ring owns.
        assert_eq!(
            validate_decorations(&[shadow, ring]),
            Err(DecorationError::RingNotTopmost)
        );
        // One fragment cannot emit the ring and the fill at different places.
        let displaced = TextDecoration {
            offset: [1.0, 0.0],
            ..ring
        };
        assert_eq!(
            validate_decorations(&[displaced]),
            Err(DecorationError::RingWithOffset)
        );
    }

    /// A filtered decoration's mask is rendered from the undilated glyph, so
    /// a spread on it would be accepted and then silently dropped.
    #[test]
    fn blur_with_spread_is_refused() {
        let both = TextDecoration {
            color: white(),
            spread: 2.0,
            offset: [0.0, 0.0],
            blur: 3.0,
            mode: DecorationMode::Solid,
        };
        assert_eq!(
            validate_decorations(&[both]),
            Err(DecorationError::BlurWithSpread)
        );
        assert_eq!(
            validate_decorations(&[TextDecoration::blurred_shadow(white(), 1.0, 1.0, 3.0)]),
            Ok(())
        );
    }

    /// Sigma drives both the mask's size and the per-fragment tap count, so
    /// an unbounded value is a resource hazard, not a slow frame.
    #[test]
    fn oversized_blur_is_refused() {
        let huge = TextDecoration::blurred_shadow(white(), 0.0, 0.0, MAX_BLUR_SIGMA + 1.0);
        assert_eq!(
            validate_decorations(&[huge]),
            Err(DecorationError::BlurTooLarge)
        );
        let at_limit = TextDecoration::blurred_shadow(white(), 0.0, 0.0, MAX_BLUR_SIGMA);
        assert_eq!(validate_decorations(&[at_limit]), Ok(()));
    }

    /// The limit is on the PHYSICAL sigma, so a logical value under the limit
    /// that scales past it is still refused.
    #[test]
    fn the_blur_limit_applies_after_scaling() {
        let area_scale = 4.0;
        let logical = MAX_BLUR_SIGMA / 2.0;
        let decoration = TextDecoration::blurred_shadow(white(), 0.0, 0.0, logical);
        let physical = physical_decoration(decoration, area_scale).expect("resolves");
        assert_eq!(
            validate_physical(&[physical]),
            Err(DecorationError::BlurTooLarge)
        );
    }

    /// Validation runs on the RESOLVED list. A decoration that resolution
    /// drops must not still count toward the ring's position, or a list the
    /// renderer would have drawn correctly fails the whole frame.
    #[test]
    fn a_dropped_decoration_does_not_displace_the_ring() {
        let dropped = TextDecoration {
            color: white(),
            spread: -1.0,
            offset: [0.0, 0.0],
            blur: 0.0,
            mode: DecorationMode::Solid,
        };
        let ring = TextDecoration::ring(white(), 2.0);
        assert_eq!(
            validate_decorations(&[dropped, ring]),
            Ok(()),
            "the dropped entry is not drawn, so the ring really is first"
        );
    }

    #[test]
    fn ring_cannot_be_blurred() {
        let blurred_ring = TextDecoration {
            blur: 2.0,
            ..TextDecoration::ring(white(), 1.0)
        };
        assert_eq!(
            validate_decorations(&[blurred_ring]),
            Err(DecorationError::RingWithBlur)
        );
    }

    #[test]
    fn constructors_carry_their_mode() {
        assert_eq!(
            TextDecoration::outline(white(), 1.0).mode,
            DecorationMode::Solid
        );
        assert_eq!(
            TextDecoration::shadow(white(), 1.0, 1.0).mode,
            DecorationMode::Solid
        );
        assert_eq!(
            TextDecoration::ring(white(), 1.0).mode,
            DecorationMode::Ring
        );
    }

    /// Mode must not disturb the culling envelope: Solid and Ring at equal
    /// spread and offset occupy the same pixels, so a mode switch is draw
    /// topology, never a geometry change.
    #[test]
    fn mode_does_not_change_extents() {
        let solid =
            physical_decoration(TextDecoration::outline(white(), 3.0), 1.0).expect("valid solid");
        let ring =
            physical_decoration(TextDecoration::ring(white(), 3.0), 1.0).expect("valid ring");
        assert_eq!(
            DecorationExtents::of(&[solid], 0.5),
            DecorationExtents::of(&[ring], 0.5)
        );
    }

    #[test]
    fn empty_decoration_list_has_no_extents() {
        assert_eq!(
            DecorationExtents::of(&[], 0.5),
            DecorationExtents::default()
        );
    }
}
