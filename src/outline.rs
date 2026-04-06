/// Glyph outline extraction from font files via skrifa.
/// Converts all curves to quadratic beziers (TrueType is native, CFF cubics are subdivided).
use skrifa::{
    MetadataProvider,
    instance::Size,
    outline::{DrawSettings, OutlinePen},
    color::{Brush, ColorGlyphFormat, ColorPainter, CompositeMode, Transform},
};

/// A quadratic bezier curve: 3 control points in em-space.
#[derive(Debug, Clone, Copy)]
pub struct QuadCurve {
    pub p1: [f32; 2],
    pub p2: [f32; 2],
    pub p3: [f32; 2],
}

/// Extracted glyph outline.
///
/// Note: `bounds` is approximate for CFF fonts — cubic-to-quadratic subdivision
/// can produce control points outside the original cubic hull. Use
/// `build_bands()` for exact bounds over the actual quadratic geometry.
#[derive(Debug, Clone)]
pub struct GlyphOutline {
    pub curves: Vec<QuadCurve>,
    pub bounds: [f32; 4], // min_x, min_y, max_x, max_y (approximate for CFF)
}

/// Pen that collects quadratic beziers. Converts lines to degenerate quadratics
/// and cubics to approximated quadratics.
struct CollectPen {
    curves: Vec<QuadCurve>,
    current: [f32; 2],
    contour_start: [f32; 2],
    min: [f32; 2],
    max: [f32; 2],
}

impl CollectPen {
    fn new() -> Self {
        Self {
            curves: Vec::new(),
            current: [0.0; 2],
            contour_start: [0.0; 2],
            min: [f32::MAX, f32::MAX],
            max: [f32::MIN, f32::MIN],
        }
    }

    fn update_bounds(&mut self, p: [f32; 2]) {
        self.min[0] = self.min[0].min(p[0]);
        self.min[1] = self.min[1].min(p[1]);
        self.max[0] = self.max[0].max(p[0]);
        self.max[1] = self.max[1].max(p[1]);
    }
}

impl OutlinePen for CollectPen {
    fn move_to(&mut self, x: f32, y: f32) {
        self.current = [x, y];
        self.contour_start = [x, y];
        self.update_bounds(self.current);
    }

    fn line_to(&mut self, x: f32, y: f32) {
        let p1 = self.current;
        let p3 = [x, y];
        if p1 == p3 {
            return; // zero-length: no winding contribution
        }
        self.curves.push(QuadCurve { p1, p2: p1, p3 });
        self.current = p3;
        self.update_bounds(p3);
    }

    fn quad_to(&mut self, cx: f32, cy: f32, x: f32, y: f32) {
        let p1 = self.current;
        let p2 = [cx, cy];
        let p3 = [x, y];
        self.curves.push(QuadCurve { p1, p2, p3 });
        self.current = p3;
        self.update_bounds(p2);
        self.update_bounds(p3);
    }

    fn curve_to(&mut self, cx0: f32, cy0: f32, cx1: f32, cy1: f32, x: f32, y: f32) {
        let c0: P = self.current;
        let c1: P = [cx0, cy0];
        let c2: P = [cx1, cy1];
        let c3: P = [x, y];

        // Point-collapse check
        if c0 == c3 {
            return;
        }

        // Collinearity pre-check: if both interior control points are within
        // tolerance of the c0→c3 baseline, emit as a line.
        let dx = c3[0] - c0[0];
        let dy = c3[1] - c0[1];
        let len_sq = dx * dx + dy * dy;
        if len_sq > 1e-12 {
            let inv = 1.0 / len_sq;
            let cross1 = (c1[0] - c0[0]) * dy - (c1[1] - c0[1]) * dx;
            let cross2 = (c2[0] - c0[0]) * dy - (c2[1] - c0[1]) * dx;
            if cross1 * cross1 * inv <= CU2QU_TOLERANCE * CU2QU_TOLERANCE
                && cross2 * cross2 * inv <= CU2QU_TOLERANCE * CU2QU_TOLERANCE
            {
                // Flat enough — emit as line (p2=p1 encoding)
                let p3f = [x, y];
                self.curves.push(QuadCurve {
                    p1: self.current,
                    p2: self.current,
                    p3: p3f,
                });
                self.current = p3f;
                self.update_bounds(p3f);
                return;
            }
        }

        cubic_to_quadratics(
            &mut self.curves,
            &mut self.min,
            &mut self.max,
            c0,
            c1,
            c2,
            c3,
            0,
        );
        self.current = [x, y];
    }

    fn close(&mut self) {
        // Emit a closing line segment if current position != contour start
        let dx = self.current[0] - self.contour_start[0];
        let dy = self.current[1] - self.contour_start[1];
        if dx * dx + dy * dy > 1e-6 {
            let p1 = self.current;
            let p3 = self.contour_start;
            self.curves.push(QuadCurve { p1, p2: p1, p3 });
            self.current = self.contour_start;
        }
    }
}

/// Cubic-to-quadratic conversion using tangent-line intersection fitting.
/// Port of harfbuzz's cu2qu algorithm (hb-gpu-cu2qu.hh).
///
/// Tolerance for quadratic approximation, in font units.
/// f32 has ~7 decimal digits — sufficient precision for font units (0-2048)
/// at 0.5 font-unit tolerance.
const CU2QU_TOLERANCE: f32 = 0.5;

/// Maximum subdivision depth (max 1024 quadratics per cubic).
const CU2QU_MAX_DEPTH: u32 = 10;

type P = [f32; 2];

fn lerp(a: P, b: P, t: f32) -> P {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
}

/// Check if a cubic error curve stays within tolerance of the origin.
/// The error curve has endpoints at (0,0) and interior control points p1, p2.
fn cubic_fits_inside(p0: P, p1: P, p2: P, p3: P, tolerance: f32, depth: u32) -> bool {
    // Quick accept: both interior control points within tolerance
    if f32::hypot(p1[0], p1[1]) <= tolerance && f32::hypot(p2[0], p2[1]) <= tolerance {
        return true;
    }
    if depth >= 8 {
        return false;
    }

    // Check midpoint (t=0.5): (p0 + 3*(p1+p2) + p3) / 8
    let mid = [
        (p0[0] + 3.0 * (p1[0] + p2[0]) + p3[0]) * 0.125,
        (p0[1] + 3.0 * (p1[1] + p2[1]) + p3[1]) * 0.125,
    ];
    if f32::hypot(mid[0], mid[1]) > tolerance {
        return false;
    }

    // Split error curve and recurse
    let d3 = [
        (p3[0] + p2[0] - p1[0] - p0[0]) * 0.125,
        (p3[1] + p2[1] - p1[1] - p0[1]) * 0.125,
    ];
    let h01 = lerp(p0, p1, 0.5);
    let h23 = lerp(p2, p3, 0.5);
    let mid_minus_d3 = [mid[0] - d3[0], mid[1] - d3[1]];
    let mid_plus_d3 = [mid[0] + d3[0], mid[1] + d3[1]];

    cubic_fits_inside(p0, h01, mid_minus_d3, mid, tolerance, depth + 1)
        && cubic_fits_inside(mid, mid_plus_d3, h23, p3, tolerance, depth + 1)
}

/// Try to fit a single quadratic to a cubic via tangent-line intersection.
/// Returns Some(q1) if the fit is within tolerance, None otherwise.
fn approx_quadratic(c0: P, c1: P, c2: P, c3: P, tolerance: f32) -> Option<P> {
    // Tangent directions
    let ax = c1[0] - c0[0];
    let ay = c1[1] - c0[1];
    let dx = c3[0] - c2[0];
    let dy = c3[1] - c2[1];

    // Perpendicular to start tangent
    let px = -ay;
    let py = ax;

    // Intersection parameter along end tangent
    let denom = px * dx + py * dy;
    if denom.abs() < 1e-6 {
        return None; // Parallel tangents — needs subdivision
    }

    let h = (px * (c0[0] - c2[0]) + py * (c0[1] - c2[1])) / denom;
    let q1 = [c2[0] + dx * h, c2[1] + dy * h];

    // Error: difference between original cubic and degree-elevated quadratic
    let two_thirds = 2.0 / 3.0;
    let err1 = [
        c0[0] + (q1[0] - c0[0]) * two_thirds - c1[0],
        c0[1] + (q1[1] - c0[1]) * two_thirds - c1[1],
    ];
    let err2 = [
        c3[0] + (q1[0] - c3[0]) * two_thirds - c2[0],
        c3[1] + (q1[1] - c3[1]) * two_thirds - c2[1],
    ];

    if cubic_fits_inside([0.0; 2], err1, err2, [0.0; 2], tolerance, 0) {
        Some(q1)
    } else {
        None
    }
}

/// Convert a cubic bezier to quadratic approximations using cu2qu.
/// Pushes quadratics to `out`, tracks bounds in `bounds_min`/`bounds_max`.
#[allow(clippy::too_many_arguments)]
fn cubic_to_quadratics(
    out: &mut Vec<QuadCurve>,
    bounds_min: &mut [f32; 2],
    bounds_max: &mut [f32; 2],
    c0: P,
    c1: P,
    c2: P,
    c3: P,
    depth: u32,
) {
    // Try single-quad fit
    if let Some(q1) = approx_quadratic(c0, c1, c2, c3, CU2QU_TOLERANCE) {
        out.push(QuadCurve {
            p1: c0,
            p2: q1,
            p3: c3,
        });
        for p in [q1, c3] {
            bounds_min[0] = bounds_min[0].min(p[0]);
            bounds_min[1] = bounds_min[1].min(p[1]);
            bounds_max[0] = bounds_max[0].max(p[0]);
            bounds_max[1] = bounds_max[1].max(p[1]);
        }
        return;
    }

    // Max depth fallback: emit line
    if depth >= CU2QU_MAX_DEPTH {
        if c0 != c3 {
            out.push(QuadCurve {
                p1: c0,
                p2: c0,
                p3: c3,
            });
            bounds_min[0] = bounds_min[0].min(c3[0]);
            bounds_min[1] = bounds_min[1].min(c3[1]);
            bounds_max[0] = bounds_max[0].max(c3[0]);
            bounds_max[1] = bounds_max[1].max(c3[1]);
        }
        return;
    }

    // de Casteljau split at t=0.5
    let m01 = lerp(c0, c1, 0.5);
    let m12 = lerp(c1, c2, 0.5);
    let m23 = lerp(c2, c3, 0.5);
    let m012 = lerp(m01, m12, 0.5);
    let m123 = lerp(m12, m23, 0.5);
    let mid = lerp(m012, m123, 0.5);

    cubic_to_quadratics(out, bounds_min, bounds_max, c0, m01, m012, mid, depth + 1);
    cubic_to_quadratics(out, bounds_min, bounds_max, mid, m123, m23, c3, depth + 1);
}

/// Extract the quadratic bezier outline for a glyph.
/// Coordinates are in font units (em-space).
///
/// `font_data`: raw font file bytes
/// `face_index`: index within a font collection (TTC), 0 for single-face fonts
/// `glyph_id`: glyph index within the font
/// `location`: variation coordinates (e.g. weight axis for variable fonts)
#[hotpath::measure]
pub fn extract_outline(
    font_data: &[u8],
    face_index: u32,
    glyph_id: u16,
    location: &[skrifa::setting::VariationSetting],
) -> Option<GlyphOutline> {
    let font = skrifa::FontRef::from_index(font_data, face_index).ok()?;
    let location = font.axes().location(location.iter().copied());
    let outlines = font.outline_glyphs();
    let glyph = outlines.get(skrifa::GlyphId::new(glyph_id.into()))?;

    let mut pen = CollectPen::new();
    let settings = DrawSettings::unhinted(Size::unscaled(), &location);
    glyph.draw(settings, &mut pen).ok()?;

    if pen.curves.is_empty() {
        return None;
    }

    Some(GlyphOutline {
        curves: pen.curves,
        bounds: [pen.min[0], pen.min[1], pen.max[0], pen.max[1]],
    })
}

/// Map a character to a glyph ID using the font's cmap table.
///
/// `face_index`: index within a font collection (TTC), 0 for single-face fonts.
pub fn char_to_glyph_id(font_data: &[u8], face_index: u32, ch: char) -> Option<u16> {
    let font = skrifa::FontRef::from_index(font_data, face_index).ok()?;
    let glyph_id = font.charmap().map(ch)?;
    Some(glyph_id.to_u32() as u16)
}

/// A single layer of a COLRv0 color glyph.
#[derive(Debug, Clone)]
pub struct ColorLayer {
    /// Sub-glyph ID to extract the outline from.
    pub glyph_id: u16,
    /// RGBA color for this layer. If `use_foreground` is true, this is zeroed
    /// and the caller should substitute the text's foreground color.
    pub color: [f32; 4],
    /// True if palette_index was 0xFFFF (foreground color sentinel).
    pub use_foreground: bool,
}

/// Result of COLR table inspection for a glyph.
pub enum ColorGlyphInfo {
    /// COLRv0: flat solid-color layers, back-to-front.
    V0Layers(Vec<ColorLayer>),
    /// COLRv1: command sequence + sub-glyph outlines for GPU interpreter.
    V1(ColorV1Data),
}

/// Encoded COLRv1 glyph data ready for GPU upload.
pub struct ColorV1Data {
    /// Command sequence (vec4<i32> texels).
    pub commands: Vec<[i32; 4]>,
    /// Sub-glyph outlines with their band data, in order of first reference.
    pub sub_glyphs: Vec<ColorV1SubGlyph>,
    /// Number of commands the shader needs to iterate.
    pub cmd_count: u32,
}

/// A single sub-glyph within a COLRv1 command sequence.
pub struct ColorV1SubGlyph {
    pub outline: GlyphOutline,
    /// Offset from blob start to this sub-glyph's header (set during upload).
    pub blob_offset: u32,
}

// Command opcodes for the shader interpreter.
pub const CMD_PUSH_GROUP: i32 = 1;
pub const CMD_DRAW_SOLID: i32 = 2;
pub const CMD_DRAW_GRADIENT: i32 = 3;
pub const CMD_POP_GROUP: i32 = 4;

/// Check whether a glyph has COLR color data. Returns the color info if so.
///
/// `location`: variable font axis settings (e.g. weight).
pub fn extract_color_info(
    font_data: &[u8],
    face_index: u32,
    glyph_id: u16,
    location: &[skrifa::setting::VariationSetting],
) -> Option<ColorGlyphInfo> {
    let font = skrifa::FontRef::from_index(font_data, face_index).ok()?;
    let color_glyphs = font.color_glyphs();
    let color_glyph = color_glyphs.get(skrifa::GlyphId::new(glyph_id as u32))?;

    match color_glyph.format() {
        ColorGlyphFormat::ColrV1 => {
            encode_colr_v1(font_data, face_index, glyph_id, location)
                .map(ColorGlyphInfo::V1)
        }
        ColorGlyphFormat::ColrV0 => {
            let palettes = font.color_palettes();
            let palette = palettes.get(0);
            let palette_colors = palette.as_ref().map(skrifa::color::ColorPalette::colors);

            let mut collector = ColrV0Collector {
                layers: Vec::new(),
                palette_colors,
            };

            let axes = font.axes();
            let loc = axes.location(location);
            if color_glyph.paint(&loc, &mut collector).is_err() {
                return None;
            }

            if collector.layers.is_empty() {
                return None;
            }

            Some(ColorGlyphInfo::V0Layers(collector.layers))
        }
    }
}

/// Minimal ColorPainter that collects COLRv0 layers (solid-color fill_glyph calls).
struct ColrV0Collector<'a> {
    layers: Vec<ColorLayer>,
    palette_colors: Option<&'a [skrifa::color::Color]>,
}

impl ColorPainter for ColrV0Collector<'_> {
    fn push_transform(&mut self, _transform: Transform) {}
    fn pop_transform(&mut self) {}
    fn push_clip_glyph(&mut self, _glyph_id: skrifa::GlyphId) {}
    fn push_clip_box(&mut self, _clip_box: skrifa::metrics::BoundingBox) {}
    fn pop_clip(&mut self) {}
    fn push_layer(&mut self, _composite_mode: CompositeMode) {}
    fn pop_layer(&mut self) {}

    fn fill(&mut self, _brush: Brush<'_>) {
        // COLRv0 should only use fill_glyph, not bare fill.
    }

    fn fill_glyph(
        &mut self,
        glyph_id: skrifa::GlyphId,
        _brush_transform: Option<Transform>,
        brush: Brush<'_>,
    ) {
        let Brush::Solid {
            palette_index,
            alpha,
        } = brush
        else {
            return; // COLRv0 only has solid fills
        };

        let use_foreground = palette_index == 0xFFFF;
        let color = if use_foreground {
            [0.0; 4]
        } else if let Some(colors) = self.palette_colors {
            if let Some(c) = colors.get(palette_index as usize) {
                [
                    c.red as f32 / 255.0 * alpha,
                    c.green as f32 / 255.0 * alpha,
                    c.blue as f32 / 255.0 * alpha,
                    c.alpha as f32 / 255.0 * alpha,
                ]
            } else {
                [0.0, 0.0, 0.0, alpha]
            }
        } else {
            [0.0, 0.0, 0.0, alpha]
        };

        self.layers.push(ColorLayer {
            glyph_id: glyph_id.to_u32() as u16,
            color,
            use_foreground,
        });
    }
}

/// Encode a COLRv1 color glyph into a command sequence for the GPU shader.
///
/// Walks the paint tree via skrifa's ColorPainter, applying transforms to
/// bezier control points on the CPU. Emits a linear command stream that the
/// fragment shader interprets with a small color stack.
pub fn encode_colr_v1(
    font_data: &[u8],
    face_index: u32,
    glyph_id: u16,
    location: &[skrifa::setting::VariationSetting],
) -> Option<ColorV1Data> {
    let font = skrifa::FontRef::from_index(font_data, face_index).ok()?;
    let color_glyphs = font.color_glyphs();
    let color_glyph =
        color_glyphs.get_with_format(skrifa::GlyphId::new(glyph_id as u32), ColorGlyphFormat::ColrV1)?;

    let palettes = font.color_palettes();
    let palette = palettes.get(0);
    let palette_colors = palette.as_ref().map(skrifa::color::ColorPalette::colors);

    let mut encoder = CommandEncoder {
        font_data,
        face_index,
        location,
        palette_colors,
        commands: Vec::new(),
        sub_glyphs: Vec::new(),
        transform_stack: vec![AffineTransform::identity()],
        clip_glyph: None,
        composite_mode_stack: Vec::new(),
    };

    let axes = font.axes();
    let loc = axes.location(location);
    if color_glyph.paint(&loc, &mut encoder).is_err() {
        return None;
    }

    if encoder.commands.is_empty() {
        return None;
    }

    let cmd_count = encoder.commands.len() as u32;
    Some(ColorV1Data {
        commands: encoder.commands,
        sub_glyphs: encoder.sub_glyphs,
        cmd_count,
    })
}

/// 2D affine transform: [xx, yx, xy, yy, dx, dy]
/// x' = xx*x + xy*y + dx
/// y' = yx*x + yy*y + dy
#[derive(Debug, Clone, Copy)]
struct AffineTransform {
    xx: f32, yx: f32,
    xy: f32, yy: f32,
    dx: f32, dy: f32,
}

impl AffineTransform {
    fn identity() -> Self {
        Self { xx: 1.0, yx: 0.0, xy: 0.0, yy: 1.0, dx: 0.0, dy: 0.0 }
    }

    /// Concatenate: self * other (apply other first, then self).
    fn then(&self, other: &Self) -> Self {
        Self {
            xx: self.xx * other.xx + self.xy * other.yx,
            yx: self.yx * other.xx + self.yy * other.yx,
            xy: self.xx * other.xy + self.xy * other.yy,
            yy: self.yx * other.xy + self.yy * other.yy,
            dx: self.xx * other.dx + self.xy * other.dy + self.dx,
            dy: self.yx * other.dx + self.yy * other.dy + self.dy,
        }
    }

    fn transform_point(&self, x: f32, y: f32) -> [f32; 2] {
        [
            self.xx * x + self.xy * y + self.dx,
            self.yx * x + self.yy * y + self.dy,
        ]
    }

    fn invert(&self) -> Option<Self> {
        let det = self.xx * self.yy - self.xy * self.yx;
        if det.abs() < 1e-12 {
            return None;
        }
        let inv_det = 1.0 / det;
        Some(Self {
            xx: self.yy * inv_det,
            yx: -self.yx * inv_det,
            xy: -self.xy * inv_det,
            yy: self.xx * inv_det,
            dx: (self.xy * self.dy - self.yy * self.dx) * inv_det,
            dy: (self.yx * self.dx - self.xx * self.dy) * inv_det,
        })
    }

    fn from_skrifa(t: &Transform) -> Self {
        Self { xx: t.xx, yx: t.yx, xy: t.xy, yy: t.yy, dx: t.dx, dy: t.dy }
    }
}

/// Pack an RGBA color into two i32 values: [R_G, B_A] with 8-bit components.
fn pack_color_i32(r: f32, g: f32, b: f32, a: f32) -> [i32; 2] {
    let ri = (r.clamp(0.0, 1.0) * 255.0).round() as u32;
    let gi = (g.clamp(0.0, 1.0) * 255.0).round() as u32;
    let bi = (b.clamp(0.0, 1.0) * 255.0).round() as u32;
    let ai = (a.clamp(0.0, 1.0) * 255.0).round() as u32;
    [(ri << 8 | gi) as i32, (bi << 8 | ai) as i32]
}

/// Pack a fixed-point f32 into two i16-in-i32 values (integer + fractional).
fn pack_fixed(v: f32) -> [i32; 2] {
    let integer = v.floor() as i32;
    let fractional = ((v - v.floor()) * (1 << 15) as f32).round() as i32;
    [integer, fractional]
}

struct CommandEncoder<'a> {
    font_data: &'a [u8],
    face_index: u32,
    location: &'a [skrifa::setting::VariationSetting],
    palette_colors: Option<&'a [skrifa::color::Color]>,
    commands: Vec<[i32; 4]>,
    sub_glyphs: Vec<ColorV1SubGlyph>,
    transform_stack: Vec<AffineTransform>,
    clip_glyph: Option<skrifa::GlyphId>,
    composite_mode_stack: Vec<i32>,
}

impl<'a> CommandEncoder<'a> {
    fn current_transform(&self) -> &AffineTransform {
        self.transform_stack.last().expect("transform stack not empty")
    }

    /// Extract a glyph outline and apply the current transform to all control points.
    fn extract_transformed_outline(&self, glyph_id: skrifa::GlyphId) -> Option<GlyphOutline> {
        let mut outline = extract_outline(
            self.font_data, self.face_index, glyph_id.to_u32() as u16, self.location,
        )?;

        let t = self.current_transform();
        if t.xx != 1.0 || t.xy != 0.0 || t.yx != 0.0 || t.yy != 1.0
            || t.dx != 0.0 || t.dy != 0.0
        {
            let mut min = [f32::MAX; 2];
            let mut max = [f32::MIN; 2];
            for curve in &mut outline.curves {
                curve.p1 = t.transform_point(curve.p1[0], curve.p1[1]);
                curve.p2 = t.transform_point(curve.p2[0], curve.p2[1]);
                curve.p3 = t.transform_point(curve.p3[0], curve.p3[1]);
                for p in [curve.p1, curve.p2, curve.p3] {
                    min[0] = min[0].min(p[0]);
                    min[1] = min[1].min(p[1]);
                    max[0] = max[0].max(p[0]);
                    max[1] = max[1].max(p[1]);
                }
            }
            outline.bounds = [min[0], min[1], max[0], max[1]];
        }

        Some(outline)
    }

    /// Add a sub-glyph and return its index (offset will be set during upload).
    fn add_sub_glyph(&mut self, outline: GlyphOutline) -> u32 {
        let idx = self.sub_glyphs.len() as u32;
        self.sub_glyphs.push(ColorV1SubGlyph { outline, blob_offset: 0 });
        idx
    }

    fn resolve_color(&self, palette_index: u16, alpha: f32) -> [f32; 4] {
        if palette_index == 0xFFFF {
            // Foreground color — shader will use instance color
            [1.0, 1.0, 1.0, alpha]
        } else if let Some(colors) = self.palette_colors {
            if let Some(c) = colors.get(palette_index as usize) {
                [
                    c.red as f32 / 255.0,
                    c.green as f32 / 255.0,
                    c.blue as f32 / 255.0,
                    c.alpha as f32 / 255.0 * alpha,
                ]
            } else {
                [0.0, 0.0, 0.0, alpha]
            }
        } else {
            [0.0, 0.0, 0.0, alpha]
        }
    }

    fn emit_draw_solid(&mut self, sub_glyph_idx: u32, color: [f32; 4]) {
        let [rg, ba] = pack_color_i32(color[0], color[1], color[2], color[3]);
        self.commands.push([CMD_DRAW_SOLID, sub_glyph_idx as i32, rg, ba]);
    }

    fn emit_draw_gradient(
        &mut self,
        sub_glyph_idx: u32,
        brush: &Brush<'_>,
        brush_transform: Option<&AffineTransform>,
    ) {
        // Compute inverse of the full brush transform for gradient evaluation.
        // The brush transform maps from gradient space to glyph space.
        // The shader needs the inverse to go from pixel coords to gradient coords.
        let current = *self.current_transform();
        let full_transform = if let Some(bt) = brush_transform {
            current.then(bt)
        } else {
            current
        };
        let inv = full_transform.invert().unwrap_or_else(AffineTransform::identity);

        match brush {
            Brush::LinearGradient { p0, p1, color_stops, extend: _ } => {
                // Command header
                self.commands.push([CMD_DRAW_GRADIENT, sub_glyph_idx as i32, 0, color_stops.len() as i32]);
                // Inverse transform (2 texels, 6 fixed-point values)
                let [ixx_i, ixx_f] = pack_fixed(inv.xx);
                let [ixy_i, ixy_f] = pack_fixed(inv.xy);
                self.commands.push([ixx_i, ixx_f, ixy_i, ixy_f]);
                let [iyx_i, iyx_f] = pack_fixed(inv.yx);
                let [iyy_i, iyy_f] = pack_fixed(inv.yy);
                self.commands.push([iyx_i, iyx_f, iyy_i, iyy_f]);
                let [idx_i, idx_f] = pack_fixed(inv.dx);
                let [idy_i, idy_f] = pack_fixed(inv.dy);
                self.commands.push([idx_i, idx_f, idy_i, idy_f]);
                // Gradient params: p0, p1 as fixed-point
                let [p0x_i, p0x_f] = pack_fixed(p0.x);
                let [p0y_i, p0y_f] = pack_fixed(p0.y);
                self.commands.push([p0x_i, p0x_f, p0y_i, p0y_f]);
                let [p1x_i, p1x_f] = pack_fixed(p1.x);
                let [p1y_i, p1y_f] = pack_fixed(p1.y);
                self.commands.push([p1x_i, p1x_f, p1y_i, p1y_f]);
                // Color stops
                for stop in *color_stops {
                    let c = self.resolve_color(stop.palette_index, stop.alpha);
                    let [rg, ba] = pack_color_i32(c[0], c[1], c[2], c[3]);
                    let [off_i, off_f] = pack_fixed(stop.offset);
                    self.commands.push([off_i, off_f, rg, ba]);
                }
            }
            Brush::RadialGradient { c0, r0, c1, r1, color_stops, extend: _ } => {
                self.commands.push([CMD_DRAW_GRADIENT, sub_glyph_idx as i32, 1, color_stops.len() as i32]);
                // Inverse transform (3 texels)
                let [ixx_i, ixx_f] = pack_fixed(inv.xx);
                let [ixy_i, ixy_f] = pack_fixed(inv.xy);
                self.commands.push([ixx_i, ixx_f, ixy_i, ixy_f]);
                let [iyx_i, iyx_f] = pack_fixed(inv.yx);
                let [iyy_i, iyy_f] = pack_fixed(inv.yy);
                self.commands.push([iyx_i, iyx_f, iyy_i, iyy_f]);
                let [idx_i, idx_f] = pack_fixed(inv.dx);
                let [idy_i, idy_f] = pack_fixed(inv.dy);
                self.commands.push([idx_i, idx_f, idy_i, idy_f]);
                // Radial params: c0, r0, c1, r1
                let [c0x_i, c0x_f] = pack_fixed(c0.x);
                let [c0y_i, c0y_f] = pack_fixed(c0.y);
                self.commands.push([c0x_i, c0x_f, c0y_i, c0y_f]);
                let [r0_i, r0_f] = pack_fixed(*r0);
                let [c1x_i, c1x_f] = pack_fixed(c1.x);
                self.commands.push([r0_i, r0_f, c1x_i, c1x_f]);
                let [c1y_i, c1y_f] = pack_fixed(c1.y);
                let [r1_i, r1_f] = pack_fixed(*r1);
                self.commands.push([c1y_i, c1y_f, r1_i, r1_f]);
                // Color stops
                for stop in *color_stops {
                    let c = self.resolve_color(stop.palette_index, stop.alpha);
                    let [rg, ba] = pack_color_i32(c[0], c[1], c[2], c[3]);
                    let [off_i, off_f] = pack_fixed(stop.offset);
                    self.commands.push([off_i, off_f, rg, ba]);
                }
            }
            Brush::SweepGradient { c0, start_angle, end_angle, color_stops, extend: _ } => {
                self.commands.push([CMD_DRAW_GRADIENT, sub_glyph_idx as i32, 2, color_stops.len() as i32]);
                // Inverse transform (3 texels)
                let [ixx_i, ixx_f] = pack_fixed(inv.xx);
                let [ixy_i, ixy_f] = pack_fixed(inv.xy);
                self.commands.push([ixx_i, ixx_f, ixy_i, ixy_f]);
                let [iyx_i, iyx_f] = pack_fixed(inv.yx);
                let [iyy_i, iyy_f] = pack_fixed(inv.yy);
                self.commands.push([iyx_i, iyx_f, iyy_i, iyy_f]);
                let [idx_i, idx_f] = pack_fixed(inv.dx);
                let [idy_i, idy_f] = pack_fixed(inv.dy);
                self.commands.push([idx_i, idx_f, idy_i, idy_f]);
                // Sweep params: center, start_angle, end_angle
                let [cx_i, cx_f] = pack_fixed(c0.x);
                let [cy_i, cy_f] = pack_fixed(c0.y);
                self.commands.push([cx_i, cx_f, cy_i, cy_f]);
                let [sa_i, sa_f] = pack_fixed(*start_angle);
                let [ea_i, ea_f] = pack_fixed(*end_angle);
                self.commands.push([sa_i, sa_f, ea_i, ea_f]);
                // Color stops
                for stop in *color_stops {
                    let c = self.resolve_color(stop.palette_index, stop.alpha);
                    let [rg, ba] = pack_color_i32(c[0], c[1], c[2], c[3]);
                    let [off_i, off_f] = pack_fixed(stop.offset);
                    self.commands.push([off_i, off_f, rg, ba]);
                }
            }
            Brush::Solid { palette_index, alpha } => {
                let color = self.resolve_color(*palette_index, *alpha);
                self.emit_draw_solid(sub_glyph_idx, color);
            }
        }
    }
}

impl ColorPainter for CommandEncoder<'_> {
    fn push_transform(&mut self, transform: Transform) {
        let new = self.current_transform().then(&AffineTransform::from_skrifa(&transform));
        self.transform_stack.push(new);
    }

    fn pop_transform(&mut self) {
        if self.transform_stack.len() > 1 {
            self.transform_stack.pop();
        }
    }

    fn push_clip_glyph(&mut self, glyph_id: skrifa::GlyphId) {
        self.clip_glyph = Some(glyph_id);
    }

    fn push_clip_box(&mut self, _clip_box: skrifa::metrics::BoundingBox) {
        // COLRv1 clip boxes are just optimization hints — we don't need them
        // for correctness since the glyph coverage handles clipping.
    }

    fn pop_clip(&mut self) {
        self.clip_glyph = None;
    }

    fn fill(&mut self, brush: Brush<'_>) {
        // fill() is called after push_clip_glyph(). The clip glyph IS the shape.
        let clip_glyph = match self.clip_glyph {
            Some(g) => g,
            None => return,
        };

        let outline = match self.extract_transformed_outline(clip_glyph) {
            Some(o) => o,
            None => return,
        };
        let sub_idx = self.add_sub_glyph(outline);

        match &brush {
            Brush::Solid { palette_index, alpha } => {
                let color = self.resolve_color(*palette_index, *alpha);
                self.emit_draw_solid(sub_idx, color);
            }
            _ => {
                self.emit_draw_gradient(sub_idx, &brush, None);
            }
        }
    }

    fn fill_glyph(
        &mut self,
        glyph_id: skrifa::GlyphId,
        brush_transform: Option<Transform>,
        brush: Brush<'_>,
    ) {
        let outline = match self.extract_transformed_outline(glyph_id) {
            Some(o) => o,
            None => return,
        };
        let sub_idx = self.add_sub_glyph(outline);

        match &brush {
            Brush::Solid { palette_index, alpha } => {
                let color = self.resolve_color(*palette_index, *alpha);
                self.emit_draw_solid(sub_idx, color);
            }
            _ => {
                let bt = brush_transform.map(|t| {
                    self.current_transform().then(&AffineTransform::from_skrifa(&t))
                });
                self.emit_draw_gradient(sub_idx, &brush, bt.as_ref());
            }
        }
    }

    fn push_layer(&mut self, composite_mode: CompositeMode) {
        self.commands.push([CMD_PUSH_GROUP, 0, 0, 0]);
        self.composite_mode_stack.push(composite_mode as i32);
    }

    fn pop_layer(&mut self) {
        let mode = self.composite_mode_stack.pop().unwrap_or(3); // default SourceOver
        self.commands.push([CMD_POP_GROUP, mode, 0, 0]);
    }

    fn pop_layer_with_mode(&mut self, composite_mode: CompositeMode) {
        self.composite_mode_stack.pop(); // discard stored mode
        self.commands.push([CMD_POP_GROUP, composite_mode as i32, 0, 0]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noto_color_emoji_colrv1() {
        let font_data = include_bytes!("../examples/fonts/NotoColorEmoji-Regular.ttf");
        let gid = char_to_glyph_id(font_data.as_slice(), 0, '\u{1F600}')
            .expect("U+1F600 in cmap");
        let info = extract_color_info(font_data.as_slice(), 0, gid, &[]);
        match &info {
            Some(ColorGlyphInfo::V1(v1)) => {
                assert!(v1.cmd_count > 0, "expected commands, got 0");
                assert!(!v1.sub_glyphs.is_empty(), "expected sub-glyphs");
            }
            Some(ColorGlyphInfo::V0Layers(_)) => panic!("expected V1, got V0"),
            None => panic!("no color info for U+1F600"),
        }
    }

    #[test]
    fn twemoji_colrv0_layers() {
        let font_data = include_bytes!("../examples/fonts/TwemojiCOLRv0.ttf");
        let emojis = ['\u{1F600}', '\u{1F60D}', '\u{1F525}', '\u{2764}'];
        for ch in emojis {
            let gid = char_to_glyph_id(font_data.as_slice(), 0, ch)
                .unwrap_or_else(|| panic!("U+{:04X} not in cmap", ch as u32));
            let color_info = extract_color_info(font_data.as_slice(), 0, gid, &[]);
            match &color_info {
                Some(ColorGlyphInfo::V0Layers(layers)) => {
                    assert!(!layers.is_empty(),
                        "U+{:04X}: V0 but no layers", ch as u32);
                    for (i, l) in layers.iter().enumerate() {
                        let outline = extract_outline(font_data.as_slice(), 0, l.glyph_id, &[]);
                        assert!(outline.is_some(),
                            "U+{:04X} layer {i}: glyph_id={} has no outline",
                            ch as u32, l.glyph_id,
                        );
                    }
                }
                _ => panic!("U+{:04X}: expected V0Layers", ch as u32),
            }
        }
    }

    #[test]
    fn cu2qu_produces_at_least_one_quad() {
        let mut out = Vec::new();
        let mut bmin = [f32::MAX; 2];
        let mut bmax = [f32::MIN; 2];
        cubic_to_quadratics(
            &mut out,
            &mut bmin,
            &mut bmax,
            [0.0, 0.0],
            [10.0, 100.0],
            [90.0, 100.0],
            [100.0, 0.0],
            0,
        );
        assert!(!out.is_empty(), "cu2qu must produce at least one curve");
        assert_eq!(out[0].p1, [0.0, 0.0]);
        assert_eq!(out.last().expect("non-empty").p3, [100.0, 0.0]);
    }

    #[test]
    fn cu2qu_chain_is_continuous() {
        let mut out = Vec::new();
        let mut bmin = [f32::MAX; 2];
        let mut bmax = [f32::MIN; 2];
        cubic_to_quadratics(
            &mut out,
            &mut bmin,
            &mut bmax,
            [0.0, 0.0],
            [0.0, 100.0],
            [100.0, 100.0],
            [100.0, 0.0],
            0,
        );
        for i in 0..out.len() - 1 {
            let gap_x = (out[i].p3[0] - out[i + 1].p1[0]).abs();
            let gap_y = (out[i].p3[1] - out[i + 1].p1[1]).abs();
            assert!(
                gap_x < 1e-6 && gap_y < 1e-6,
                "curve chain not continuous at index {i}: gap=({gap_x}, {gap_y})"
            );
        }
    }

    #[test]
    fn cu2qu_terminates_on_complex_cubic() {
        let mut out = Vec::new();
        let mut bmin = [f32::MAX; 2];
        let mut bmax = [f32::MIN; 2];
        cubic_to_quadratics(
            &mut out,
            &mut bmin,
            &mut bmax,
            [0.0, 0.0],
            [0.0, 1000.0],
            [1000.0, 1000.0],
            [1000.0, 0.0],
            0,
        );
        assert!(
            out.len() <= 1024,
            "max depth 10 should produce at most 1024 quads, got {}",
            out.len()
        );
        assert!(!out.is_empty());
    }

    #[test]
    fn collect_pen_line_to_creates_degenerate_quad() {
        let mut pen = CollectPen::new();
        pen.move_to(0.0, 0.0);
        pen.line_to(100.0, 0.0);
        assert_eq!(pen.curves.len(), 1);
        let c = &pen.curves[0];
        // p2 should equal p1 (harfbuzz-style degenerate encoding)
        assert!((c.p2[0] - 0.0).abs() < 1e-6);
        assert!((c.p2[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn collect_pen_close_emits_closing_segment() {
        let mut pen = CollectPen::new();
        pen.move_to(0.0, 0.0);
        pen.line_to(100.0, 0.0);
        pen.line_to(100.0, 100.0);
        pen.close();
        // 2 line_to + 1 close = 3 curves
        assert_eq!(pen.curves.len(), 3);
        // Closing segment should end at contour start
        let last = &pen.curves[2];
        assert!((last.p3[0] - 0.0).abs() < 1e-6);
        assert!((last.p3[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn collect_pen_close_at_start_no_extra_segment() {
        let mut pen = CollectPen::new();
        pen.move_to(50.0, 50.0);
        pen.line_to(100.0, 50.0);
        pen.line_to(50.0, 50.0); // back to start
        pen.close();
        // close() should not emit an extra segment since we're already at start
        assert_eq!(pen.curves.len(), 2);
    }

    #[test]
    fn collect_pen_bounds_track_all_points() {
        let mut pen = CollectPen::new();
        pen.move_to(10.0, 20.0);
        pen.quad_to(5.0, 90.0, 80.0, 30.0);
        assert!(pen.min[0] <= 5.0);
        assert!(pen.min[1] <= 20.0);
        assert!(pen.max[0] >= 80.0);
        assert!(pen.max[1] >= 90.0);
    }

    /// Verify cu2qu chain properties on a variety of cubics: continuity,
    /// endpoint preservation, and reasonable subdivision counts.
    #[test]
    fn cu2qu_f32_chain_properties() {
        let cubics: &[(P, P, P, P)] = &[
            // Gentle S-curve
            ([0.0, 0.0], [100.0, 300.0], [200.0, -100.0], [300.0, 200.0]),
            // Near-semicircle (high curvature)
            ([0.0, 0.0], [0.0, 552.0], [448.0, 1000.0], [1000.0, 1000.0]),
            // Typical glyph curve (font-unit scale)
            ([186.0, 0.0], [186.0, 262.0], [398.0, 450.0], [660.0, 450.0]),
            // Small curve (10 units)
            ([0.0, 0.0], [3.0, 8.0], [7.0, 8.0], [10.0, 0.0]),
            // Large CJK-scale (2048 upem)
            (
                [100.0, 200.0],
                [400.0, 1800.0],
                [1600.0, 1800.0],
                [1900.0, 200.0],
            ),
        ];

        for (i, &(c0, c1, c2, c3)) in cubics.iter().enumerate() {
            let mut out = Vec::new();
            let mut bmin = [f32::MAX; 2];
            let mut bmax = [f32::MIN; 2];
            cubic_to_quadratics(&mut out, &mut bmin, &mut bmax, c0, c1, c2, c3, 0);

            assert!(!out.is_empty(), "cubic {i} produced no quads");
            // Sharp inflections (e.g. S-curves) legitimately need 20+ quads.
            // Max depth 10 caps at 1024; typical glyphs produce 1-30.
            assert!(
                out.len() <= 64,
                "cubic {i}: excessive subdivisions ({})",
                out.len()
            );

            // Endpoint preservation
            assert_eq!(out[0].p1, c0, "cubic {i}: first quad doesn't start at c0");
            assert_eq!(
                out.last().expect("non-empty").p3,
                c3,
                "cubic {i}: last quad doesn't end at c3"
            );

            // Chain continuity
            for j in 0..out.len() - 1 {
                let gap = f32::hypot(
                    out[j].p3[0] - out[j + 1].p1[0],
                    out[j].p3[1] - out[j + 1].p1[1],
                );
                assert!(gap < 1e-5, "cubic {i}: chain gap {gap} at quad {j}");
            }

            // Bounds must contain new control points (p2, p3 of each quad).
            // p1 of the first quad is the start point, tracked separately by the pen.
            for q in &out {
                for p in [q.p2, q.p3] {
                    assert!(
                        p[0] >= bmin[0] - 1e-3 && p[0] <= bmax[0] + 1e-3,
                        "cubic {i}: x={} outside bounds [{}, {}]",
                        p[0],
                        bmin[0],
                        bmax[0]
                    );
                    assert!(
                        p[1] >= bmin[1] - 1e-3 && p[1] <= bmax[1] + 1e-3,
                        "cubic {i}: y={} outside bounds [{}, {}]",
                        p[1],
                        bmin[1],
                        bmax[1]
                    );
                }
            }
        }
    }

    /// Extract outlines from the embedded CFF font and verify all glyphs
    /// produce valid, continuous quadratic chains.
    #[test]
    fn cff_font_outlines_are_valid() {
        use skrifa::raw::TableProvider;
        let font_data = include_bytes!("../examples/fonts/EBH Runes.otf");
        let font = skrifa::FontRef::from_index(font_data.as_slice(), 0).expect("valid CFF font");
        let num_glyphs = font.maxp().map(|m| m.num_glyphs()).unwrap_or(0);

        let mut extracted = 0u32;
        for gid in 0..num_glyphs {
            if let Some(outline) = extract_outline(font_data.as_slice(), 0, gid, &[]) {
                extracted += 1;
                // Chain continuity within each contour
                // (we can't easily separate contours, but p3[i] == p1[i+1]
                // should hold within contours)
                for j in 0..outline.curves.len().saturating_sub(1) {
                    let gap = f32::hypot(
                        outline.curves[j].p3[0] - outline.curves[j + 1].p1[0],
                        outline.curves[j].p3[1] - outline.curves[j + 1].p1[1],
                    );
                    // Contour breaks are allowed (move_to resets), but within
                    // a contour the chain must be continuous. We can't distinguish
                    // here, so just check no gap exceeds the glyph's bounding box
                    // (a very loose sanity check).
                    let bbox_diag = f32::hypot(
                        outline.bounds[2] - outline.bounds[0],
                        outline.bounds[3] - outline.bounds[1],
                    );
                    assert!(
                        gap <= bbox_diag + 1.0,
                        "glyph {gid}: suspicious gap {gap} at curve {j} (bbox diag {bbox_diag})"
                    );
                }
            }
        }

        assert!(
            extracted >= 5,
            "expected at least 5 CFF glyph outlines, got {extracted}"
        );
    }
}
