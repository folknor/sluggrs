/// Glyph outline extraction from font files via skrifa.
/// Converts all curves to quadratic beziers (TrueType is native, CFF cubics are subdivided).
use skrifa::{
    MetadataProvider,
    instance::Size,
    outline::{DrawSettings, OutlinePen},
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
                self.curves.push(QuadCurve { p1: self.current, p2: self.current, p3: p3f });
                self.current = p3f;
                self.update_bounds(p3f);
                return;
            }
        }

        cubic_to_quadratics(&mut self.curves, &mut self.min, &mut self.max, c0, c1, c2, c3, 0);
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
    c0: P, c1: P, c2: P, c3: P,
    depth: u32,
) {
    // Try single-quad fit
    if let Some(q1) = approx_quadratic(c0, c1, c2, c3, CU2QU_TOLERANCE) {
        out.push(QuadCurve { p1: c0, p2: q1, p3: c3 });
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
            out.push(QuadCurve { p1: c0, p2: c0, p3: c3 });
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cu2qu_produces_at_least_one_quad() {
        let mut out = Vec::new();
        let mut bmin = [f32::MAX; 2];
        let mut bmax = [f32::MIN; 2];
        cubic_to_quadratics(
            &mut out, &mut bmin, &mut bmax,
            [0.0, 0.0], [10.0, 100.0], [90.0, 100.0], [100.0, 0.0],
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
            &mut out, &mut bmin, &mut bmax,
            [0.0, 0.0], [0.0, 100.0], [100.0, 100.0], [100.0, 0.0],
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
            &mut out, &mut bmin, &mut bmax,
            [0.0, 0.0], [0.0, 1000.0], [1000.0, 1000.0], [1000.0, 0.0],
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
            ([100.0, 200.0], [400.0, 1800.0], [1600.0, 1800.0], [1900.0, 200.0]),
        ];

        for (i, &(c0, c1, c2, c3)) in cubics.iter().enumerate() {
            let mut out = Vec::new();
            let mut bmin = [f32::MAX; 2];
            let mut bmax = [f32::MIN; 2];
            cubic_to_quadratics(&mut out, &mut bmin, &mut bmax, c0, c1, c2, c3, 0);

            assert!(!out.is_empty(), "cubic {i} produced no quads");
            // Sharp inflections (e.g. S-curves) legitimately need 20+ quads.
            // Max depth 10 caps at 1024; typical glyphs produce 1-30.
            assert!(out.len() <= 64, "cubic {i}: excessive subdivisions ({})", out.len());

            // Endpoint preservation
            assert_eq!(out[0].p1, c0, "cubic {i}: first quad doesn't start at c0");
            assert_eq!(out.last().expect("non-empty").p3, c3, "cubic {i}: last quad doesn't end at c3");

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
                    assert!(p[0] >= bmin[0] - 1e-3 && p[0] <= bmax[0] + 1e-3,
                        "cubic {i}: x={} outside bounds [{}, {}]", p[0], bmin[0], bmax[0]);
                    assert!(p[1] >= bmin[1] - 1e-3 && p[1] <= bmax[1] + 1e-3,
                        "cubic {i}: y={} outside bounds [{}, {}]", p[1], bmin[1], bmax[1]);
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
