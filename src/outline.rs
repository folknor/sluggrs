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
pub fn extract_outline(font_data: &[u8], glyph_id: u16) -> Option<GlyphOutline> {
    let font = skrifa::FontRef::new(font_data).ok()?;
    let outlines = font.outline_glyphs();
    let glyph = outlines.get(skrifa::GlyphId::new(glyph_id.into()))?;

    let mut pen = CollectPen::new();
    let settings = DrawSettings::unhinted(Size::unscaled(), skrifa::instance::LocationRef::default());
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
