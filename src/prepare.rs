/// GPU preparation stage for glyph outlines.
///
/// Outlines pass through unchanged since line segments use p2=p1 encoding
/// and the shader handles degenerate quadratics via exact-zero detection.
/// The only mutation is fake-italic shear for ~1% of glyphs.
use crate::outline::GlyphOutline;

/// Italic shear factor: tan(14°) ≈ 0.2493, matching cosmic_text's
/// swash integration for FAKE_ITALIC.
const ITALIC_SHEAR: f32 = 0.2493;

/// Apply a fake-italic shear transform to a glyph outline.
///
/// The shear is x' = x + y * tan(14°), y' = y, matching the angle
/// used by cosmic_text's swash integration. Bounds are recomputed
/// after the transform.
pub fn apply_italic_shear(outline: &mut GlyphOutline) {
    if outline.curves.is_empty() {
        return;
    }

    let mut min = [f32::MAX, f32::MAX];
    let mut max = [f32::MIN, f32::MIN];

    for curve in &mut outline.curves {
        for p in [&mut curve.p1, &mut curve.p2, &mut curve.p3] {
            p[0] += p[1] * ITALIC_SHEAR;
            min[0] = min[0].min(p[0]);
            min[1] = min[1].min(p[1]);
            max[0] = max[0].max(p[0]);
            max[1] = max[1].max(p[1]);
        }
    }

    outline.bounds = [min[0], min[1], max[0], max[1]];
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::outline::QuadCurve;

    fn real_quad(p1: [f32; 2], p2: [f32; 2], p3: [f32; 2]) -> QuadCurve {
        QuadCurve { p1, p2, p3 }
    }

    #[test]
    fn italic_shear_only_affects_x() {
        let mut outline = GlyphOutline {
            curves: vec![real_quad([10.0, 100.0], [50.0, 200.0], [90.0, 50.0])],
            bounds: [10.0, 50.0, 90.0, 200.0],
        };
        let original_ys: Vec<[f32; 3]> = outline
            .curves
            .iter()
            .map(|c| [c.p1[1], c.p2[1], c.p3[1]])
            .collect();

        apply_italic_shear(&mut outline);

        for (i, c) in outline.curves.iter().enumerate() {
            assert!((c.p1[1] - original_ys[i][0]).abs() < 1e-6);
            assert!((c.p2[1] - original_ys[i][1]).abs() < 1e-6);
            assert!((c.p3[1] - original_ys[i][2]).abs() < 1e-6);
        }
    }

    #[test]
    fn italic_shear_zero_y_unchanged() {
        let mut outline = GlyphOutline {
            curves: vec![real_quad([10.0, 0.0], [50.0, 0.0], [90.0, 0.0])],
            bounds: [10.0, 0.0, 90.0, 0.0],
        };
        let original_xs = [10.0f32, 50.0, 90.0];
        apply_italic_shear(&mut outline);

        let c = &outline.curves[0];
        assert!((c.p1[0] - original_xs[0]).abs() < 1e-6);
        assert!((c.p2[0] - original_xs[1]).abs() < 1e-6);
        assert!((c.p3[0] - original_xs[2]).abs() < 1e-6);
    }

    #[test]
    fn italic_shear_positive_y_shifts_right() {
        let mut outline = GlyphOutline {
            curves: vec![real_quad([0.0, 100.0], [0.0, 200.0], [0.0, 300.0])],
            bounds: [0.0, 100.0, 0.0, 300.0],
        };
        apply_italic_shear(&mut outline);

        let c = &outline.curves[0];
        assert!(c.p1[0] > 0.0, "positive y should shift x right");
        assert!(c.p2[0] > c.p1[0], "larger y should shift x more");
        assert!(c.p3[0] > c.p2[0], "largest y should shift x most");
    }

    #[test]
    fn italic_shear_bounds_updated() {
        let mut outline = GlyphOutline {
            curves: vec![real_quad([0.0, 0.0], [50.0, 500.0], [100.0, 0.0])],
            bounds: [0.0, 0.0, 100.0, 500.0],
        };
        apply_italic_shear(&mut outline);

        // The sheared p2 at y=500 should have moved x significantly right
        // so max_x in bounds should reflect that
        assert!(outline.bounds[2] > 100.0, "shear should widen max_x bound");
    }
}
