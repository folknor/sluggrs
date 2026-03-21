/// GPU preparation stage for glyph outlines.
///
/// Transforms exact font geometry ([`GlyphOutline`]) into solver-safe
/// geometry ([`GpuOutline`]) suitable for the Slug fragment shader.
/// The key transformation is perturbing degenerate quadratics (line segments
/// encoded with p2 at the midpoint) so the shader's quadratic solver never
/// encounters collinear control points.
use crate::outline::{GlyphOutline, QuadCurve};

/// Glyph outline prepared for GPU rendering.
///
/// Curves may differ slightly from the true font geometry: line segments
/// have p2 offset along the edge normal to avoid degenerate quadratics
/// in the shader solver. Bounds are recomputed to contain the actual
/// control points.
#[derive(Debug, Clone)]
pub struct GpuOutline {
    pub curves: Vec<QuadCurve>,
    pub bounds: [f32; 4], // min_x, min_y, max_x, max_y
}

/// Prepare a glyph outline for GPU rendering.
///
/// Line segments (degenerate quadratics where p2 is the midpoint of p1–p3)
/// are perturbed slightly along the edge normal so the Slug shader's
/// quadratic solver always has a nonzero second-degree coefficient.
#[hotpath::measure]
pub fn prepare_outline(outline: &GlyphOutline) -> GpuOutline {
    let mut curves = Vec::with_capacity(outline.curves.len());
    let mut min = [f32::MAX, f32::MAX];
    let mut max = [f32::MIN, f32::MIN];

    for curve in &outline.curves {
        let p2 = if is_linear(curve) {
            perturb_midpoint(curve.p1, curve.p3)
        } else {
            curve.p2
        };

        let out = QuadCurve {
            p1: curve.p1,
            p2,
            p3: curve.p3,
        };
        curves.push(out);

        // Recompute bounds over all control points (including perturbed p2)
        for p in [out.p1, out.p2, out.p3] {
            min[0] = min[0].min(p[0]);
            min[1] = min[1].min(p[1]);
            max[0] = max[0].max(p[0]);
            max[1] = max[1].max(p[1]);
        }
    }

    GpuOutline {
        curves,
        bounds: [min[0], min[1], max[0], max[1]],
    }
}

/// Check whether a quadratic curve is actually a line segment
/// (p2 is at the midpoint of p1–p3).
fn is_linear(curve: &QuadCurve) -> bool {
    let mid_x = (curve.p1[0] + curve.p3[0]) * 0.5;
    let mid_y = (curve.p1[1] + curve.p3[1]) * 0.5;
    (curve.p2[0] - mid_x).abs() < 1e-6 && (curve.p2[1] - mid_y).abs() < 1e-6
}

/// Italic shear factor: tan(14°) ≈ 0.2493, matching cosmic_text's
/// swash integration for FAKE_ITALIC.
const ITALIC_SHEAR: f32 = 0.2493;

/// Apply a fake-italic shear transform to a GPU outline.
///
/// The shear is x' = x + y * tan(14°), y' = y, matching the angle
/// used by cosmic_text's swash integration. Bounds are recomputed
/// after the transform.
pub fn apply_italic_shear(outline: &mut GpuOutline) {
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

/// Offset the midpoint of a line segment along the edge normal.
/// This turns the degenerate quadratic into a tiny but genuine curve,
/// preventing division-by-near-zero in the shader's quadratic solver.
fn perturb_midpoint(p1: [f32; 2], p3: [f32; 2]) -> [f32; 2] {
    let dx = p3[0] - p1[0];
    let dy = p3[1] - p1[1];
    let len = (dx * dx + dy * dy).sqrt().max(1.0);
    // Scaled to segment length, clamped to a safe range in em-space.
    let eps = (len * 1e-5).clamp(0.01, 0.1);
    let nx = -dy / len;
    let ny = dx / len;
    [
        (p1[0] + p3[0]) * 0.5 + nx * eps,
        (p1[1] + p3[1]) * 0.5 + ny * eps,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line_segment(p1: [f32; 2], p3: [f32; 2]) -> QuadCurve {
        let p2 = [(p1[0] + p3[0]) * 0.5, (p1[1] + p3[1]) * 0.5];
        QuadCurve { p1, p2, p3 }
    }

    fn real_quad(p1: [f32; 2], p2: [f32; 2], p3: [f32; 2]) -> QuadCurve {
        QuadCurve { p1, p2, p3 }
    }

    #[test]
    fn is_linear_detects_line_segment() {
        let curve = line_segment([0.0, 0.0], [100.0, 200.0]);
        assert!(is_linear(&curve));
    }

    #[test]
    fn is_linear_rejects_real_quadratic() {
        let curve = real_quad([0.0, 0.0], [50.0, 100.0], [100.0, 0.0]);
        assert!(!is_linear(&curve));
    }

    #[test]
    fn perturb_midpoint_stays_near_midpoint() {
        let p1 = [0.0, 0.0];
        let p3 = [100.0, 0.0];
        let perturbed = perturb_midpoint(p1, p3);
        // Should be close to (50, 0) but offset along the normal (0, -1)
        assert!((perturbed[0] - 50.0).abs() < 1.0);
        // Normal offset should be small but nonzero
        assert!(perturbed[1].abs() > 1e-6);
        assert!(perturbed[1].abs() < 1.0);
    }

    #[test]
    fn perturb_midpoint_produces_nonlinear_curve() {
        let p1 = [10.0, 20.0];
        let p3 = [50.0, 80.0];
        let p2 = perturb_midpoint(p1, p3);
        let curve = QuadCurve { p1, p2, p3 };
        assert!(!is_linear(&curve), "Perturbed midpoint should not be linear");
    }

    #[test]
    fn prepare_outline_perturbs_lines_preserves_curves() {
        let outline = GlyphOutline {
            curves: vec![
                line_segment([0.0, 0.0], [100.0, 0.0]),       // line → perturbed
                real_quad([0.0, 0.0], [50.0, 80.0], [100.0, 0.0]), // curve → unchanged
            ],
            bounds: [0.0, 0.0, 100.0, 80.0],
        };
        let gpu = prepare_outline(&outline);
        assert_eq!(gpu.curves.len(), 2);

        // First curve was linear → should now be nonlinear
        assert!(!is_linear(&gpu.curves[0]));

        // Second curve was already quadratic → p2 should be unchanged
        assert!((gpu.curves[1].p2[0] - 50.0).abs() < 1e-6);
        assert!((gpu.curves[1].p2[1] - 80.0).abs() < 1e-6);
    }

    #[test]
    fn prepare_outline_empty_input() {
        let outline = GlyphOutline {
            curves: vec![],
            bounds: [0.0, 0.0, 0.0, 0.0],
        };
        let gpu = prepare_outline(&outline);
        assert!(gpu.curves.is_empty());
    }

    #[test]
    fn prepare_outline_bounds_contain_all_points() {
        let outline = GlyphOutline {
            curves: vec![
                line_segment([10.0, 20.0], [90.0, 80.0]),
                real_quad([5.0, 5.0], [50.0, 95.0], [95.0, 5.0]),
            ],
            bounds: [5.0, 5.0, 95.0, 95.0],
        };
        let gpu = prepare_outline(&outline);
        let [min_x, min_y, max_x, max_y] = gpu.bounds;

        for curve in &gpu.curves {
            for p in [curve.p1, curve.p2, curve.p3] {
                assert!(p[0] >= min_x - 1e-6, "point x={} < min_x={min_x}", p[0]);
                assert!(p[1] >= min_y - 1e-6, "point y={} < min_y={min_y}", p[1]);
                assert!(p[0] <= max_x + 1e-6, "point x={} > max_x={max_x}", p[0]);
                assert!(p[1] <= max_y + 1e-6, "point y={} > max_y={max_y}", p[1]);
            }
        }
    }

    #[test]
    fn italic_shear_only_affects_x() {
        let mut outline = GpuOutline {
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
        let mut outline = GpuOutline {
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
        let mut outline = GpuOutline {
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
        let mut outline = GpuOutline {
            curves: vec![real_quad([0.0, 0.0], [50.0, 500.0], [100.0, 0.0])],
            bounds: [0.0, 0.0, 100.0, 500.0],
        };
        apply_italic_shear(&mut outline);

        // The sheared p2 at y=500 should have moved x significantly right
        // so max_x in bounds should reflect that
        assert!(outline.bounds[2] > 100.0, "shear should widen max_x bound");
    }
}
