/// Glyph outline extraction from font files via skrifa.
/// Converts all curves to quadratic beziers (TrueType is native, CFF cubics are subdivided).
use skrifa::{
    instance::Size,
    outline::{DrawSettings, OutlinePen},
    MetadataProvider,
};

/// A quadratic bezier curve: 3 control points in em-space.
#[derive(Debug, Clone, Copy)]
pub struct QuadCurve {
    pub p1: [f32; 2],
    pub p2: [f32; 2],
    pub p3: [f32; 2],
}

/// Extracted glyph outline.
#[derive(Debug, Clone)]
pub struct GlyphOutline {
    pub curves: Vec<QuadCurve>,
    pub bounds: [f32; 4], // min_x, min_y, max_x, max_y
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
        let p2 = [(p1[0] + p3[0]) * 0.5, (p1[1] + p3[1]) * 0.5];
        self.curves.push(QuadCurve { p1, p2, p3 });
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
        // Cubic bezier → approximate with quadratics via midpoint subdivision.
        // For a first pass, split once into two quadratics (good enough for most glyphs).
        let p0 = self.current;
        let p1 = [cx0, cy0];
        let p2 = [cx1, cy1];
        let p3 = [x, y];

        subdivide_cubic(&mut self.curves, p0, p1, p2, p3, 0);

        self.current = p3;
        self.update_bounds(p1);
        self.update_bounds(p2);
        self.update_bounds(p3);
    }

    fn close(&mut self) {
        // Emit a closing line segment if current position != contour start
        let dx = self.current[0] - self.contour_start[0];
        let dy = self.current[1] - self.contour_start[1];
        if dx * dx + dy * dy > 1e-6 {
            let p1 = self.current;
            let p3 = self.contour_start;
            let p2 = [(p1[0] + p3[0]) * 0.5, (p1[1] + p3[1]) * 0.5];
            self.curves.push(QuadCurve { p1, p2, p3 });
            self.current = self.contour_start;
        }
    }
}

/// Recursively subdivide a cubic bezier into quadratic approximations.
/// max_depth controls quality vs curve count tradeoff.
fn subdivide_cubic(
    out: &mut Vec<QuadCurve>,
    p0: [f32; 2],
    p1: [f32; 2],
    p2: [f32; 2],
    p3: [f32; 2],
    depth: u32,
) {
    const MAX_DEPTH: u32 = 3;

    // Check if quadratic approximation is good enough.
    // Error metric: max distance between cubic and its quadratic approximation.
    // The quadratic control point is at (3*(p1+p2) - p0 - p3) / 4
    let qx = (3.0 * (p1[0] + p2[0]) - p0[0] - p3[0]) / 4.0;
    let qy = (3.0 * (p1[1] + p2[1]) - p0[1] - p3[1]) / 4.0;

    // Error is proportional to |p1 - q| + |p2 - q|
    let err = ((p1[0] - qx).powi(2) + (p1[1] - qy).powi(2)).sqrt()
        + ((p2[0] - qx).powi(2) + (p2[1] - qy).powi(2)).sqrt();

    if err < 0.5 || depth >= MAX_DEPTH {
        // Good enough — emit single quadratic
        out.push(QuadCurve {
            p1: p0,
            p2: [qx, qy],
            p3,
        });
        return;
    }

    // Split cubic at t=0.5 using de Casteljau
    let m01 = mid(p0, p1);
    let m12 = mid(p1, p2);
    let m23 = mid(p2, p3);
    let m012 = mid(m01, m12);
    let m123 = mid(m12, m23);
    let m0123 = mid(m012, m123);

    subdivide_cubic(out, p0, m01, m012, m0123, depth + 1);
    subdivide_cubic(out, m0123, m123, m23, p3, depth + 1);
}

fn mid(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5]
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
pub fn char_to_glyph_id(font_data: &[u8], ch: char) -> Option<u16> {
    let font = skrifa::FontRef::new(font_data).ok()?;
    let glyph_id = font.charmap().map(ch)?;
    Some(glyph_id.to_u32() as u16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mid_computes_midpoint() {
        assert_eq!(mid([0.0, 0.0], [10.0, 20.0]), [5.0, 10.0]);
        assert_eq!(mid([-5.0, 3.0], [5.0, -3.0]), [0.0, 0.0]);
    }

    #[test]
    fn subdivide_cubic_produces_at_least_one_quad() {
        let mut out = Vec::new();
        subdivide_cubic(
            &mut out,
            [0.0, 0.0],
            [10.0, 100.0],
            [90.0, 100.0],
            [100.0, 0.0],
            0,
        );
        assert!(!out.is_empty(), "subdivision must produce at least one curve");
        // First curve should start at p0
        assert_eq!(out[0].p1, [0.0, 0.0]);
        // Last curve should end at p3
        assert_eq!(out.last().expect("non-empty").p3, [100.0, 0.0]);
    }

    #[test]
    fn subdivide_cubic_chain_is_continuous() {
        let mut out = Vec::new();
        subdivide_cubic(
            &mut out,
            [0.0, 0.0],
            [0.0, 100.0],
            [100.0, 100.0],
            [100.0, 0.0],
            0,
        );
        // Each curve's p3 should equal the next curve's p1
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
    fn subdivide_respects_max_depth() {
        // A highly curved cubic should still terminate (MAX_DEPTH=3 → max 8 quads)
        let mut out = Vec::new();
        subdivide_cubic(
            &mut out,
            [0.0, 0.0],
            [0.0, 1000.0],
            [1000.0, 1000.0],
            [1000.0, 0.0],
            0,
        );
        assert!(out.len() <= 8, "max depth 3 should produce at most 8 quads, got {}", out.len());
        assert!(!out.is_empty());
    }

    #[test]
    fn collect_pen_line_to_creates_degenerate_quad() {
        let mut pen = CollectPen::new();
        pen.move_to(0.0, 0.0);
        pen.line_to(100.0, 0.0);
        assert_eq!(pen.curves.len(), 1);
        let c = &pen.curves[0];
        // p2 should be the midpoint
        assert!((c.p2[0] - 50.0).abs() < 1e-6);
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
}
