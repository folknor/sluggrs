//! GPU regression coverage for the quadratic solver cancellation fix.

use sluggrs::{
    SIMPLE_SHADER_WGSL,
    outline::{GlyphOutline, QuadCurve},
    prep::{PrepScratch, prepare_mono},
};

mod common;

#[derive(Copy, Clone)]
enum Axis {
    Horizontal,
    Vertical,
}

impl Axis {
    fn fragment_entry(self) -> &'static str {
        match self {
            Self::Horizontal => "fs_solver_horizontal",
            Self::Vertical => "fs_solver_vertical",
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Horizontal => "horizontal",
            Self::Vertical => "vertical",
        }
    }
}

/// A box whose right (resp. top) edge is a genuine midpoint-controlled
/// quadratic with an exactly-linear active-axis coordinate: control values
/// (-1000, 0, 1000), so a = q1 - 2*q2 + q3 == 0 exactly in the stored
/// quarter-integer encoding. The three other edges are p2 = p1 lines.
/// Pre-fix shaders computed `a` from render_coord-shifted values, where
/// rounding yields a = -2^-15 instead of 0 at the witness sample, the
/// quadratic branch runs, and cancellation selects root t = 2.0 instead of
/// the correct linear root t ~= 0.744002.
fn outline(axis: Axis) -> GlyphOutline {
    let curves = match axis {
        Axis::Horizontal => vec![
            QuadCurve {
                p1: [0.0, -1000.0],
                p2: [500.0, 0.0],
                p3: [0.0, 1000.0],
            },
            QuadCurve {
                p1: [0.0, 1000.0],
                p2: [0.0, 1000.0],
                p3: [-1000.0, 1000.0],
            },
            QuadCurve {
                p1: [-1000.0, 1000.0],
                p2: [-1000.0, 1000.0],
                p3: [-1000.0, -1000.0],
            },
            QuadCurve {
                p1: [-1000.0, -1000.0],
                p2: [-1000.0, -1000.0],
                p3: [0.0, -1000.0],
            },
        ],
        Axis::Vertical => vec![
            QuadCurve {
                p1: [-1000.0, 0.0],
                p2: [0.0, 500.0],
                p3: [1000.0, 0.0],
            },
            QuadCurve {
                p1: [1000.0, 0.0],
                p2: [1000.0, 0.0],
                p3: [1000.0, -1000.0],
            },
            QuadCurve {
                p1: [1000.0, -1000.0],
                p2: [1000.0, -1000.0],
                p3: [-1000.0, -1000.0],
            },
            QuadCurve {
                p1: [-1000.0, -1000.0],
                p2: [-1000.0, -1000.0],
                p3: [-1000.0, 0.0],
            },
        ],
    };
    let bounds = match axis {
        Axis::Horizontal => [-1000.0, -1000.0, 500.0, 1000.0],
        Axis::Vertical => [-1000.0, -1000.0, 1000.0, 500.0],
    };

    GlyphOutline { curves, bounds }
}

fn glyph_data(axis: Axis) -> Vec<i32> {
    let mut scratch = PrepScratch::default();
    let prepared = prepare_mono(&outline(axis), 1, 1, 2000.0, &mut scratch)
        .expect("synthetic outline must fit the packed i16 representation");
    let mut data = vec![
        prepared.bounds[0].to_bits() as i32,
        prepared.bounds[1].to_bits() as i32,
        prepared.bounds[2].to_bits() as i32,
        prepared.bounds[3].to_bits() as i32,
        prepared.band_transform[0].to_bits() as i32,
        prepared.band_transform[1].to_bits() as i32,
        prepared.band_transform[2].to_bits() as i32,
        prepared.band_transform[3].to_bits() as i32,
        sluggrs::prep::pack_i16_pair(
            (prepared.band_count_x - 1) as i16,
            (prepared.band_count_y - 1) as i16,
        ),
        0,
    ];
    data.extend_from_slice(&prepared.blob_data);
    data
}

/// Appends test-only fragment entry points that call the real
/// render_single with the exact f32 witness sample. On the witness curve
/// y(t) = -1000 + 2000t, the sample's active coordinate 488.003997802734375
/// gives t ~= 0.744002 and curve x(t) ~= 190.463; the sample x
/// 187.9630126953125 sits 2.5 decoded units = 0.25 px inside
/// (pixels_per_em 0.1), so correct coverage is exactly 0.75. The broken
/// variant's wrong root drives coverage far from 0.75 (measured 0 on this
/// geometry). The vertical entry point swaps the two coordinates.
fn shader_source(broken: bool) -> String {
    let original = "let a = q12.xy - q12.zw * 2.0 + q3;\n            let b = q12.xy - q12.zw;";
    let shifted = "let a = p12.xy - p12.zw * 2.0 + p3;\n            let b = p12.xy - p12.zw;";
    let occurrences = SIMPLE_SHADER_WGSL.matches(original).count();
    assert_eq!(
        occurrences, 2,
        "solver regression patch must match both solver blocks, found {occurrences}"
    );

    let source = if broken {
        SIMPLE_SHADER_WGSL.replace(original, shifted)
    } else {
        SIMPLE_SHADER_WGSL.to_owned()
    };
    format!(
        "{source}\n\
@fragment\n\
fn fs_solver_horizontal(input: VertexOutput) -> @location(0) vec4<f32> {{\n\
    let coverage = render_single(\n\
        vec2<f32>(187.9630126953125, 488.003997802734375),\n\
        vec2<f32>(0.1, 0.1),\n\
        input.banding,\n\
        u32(input.glyph.x),\n\
        input.glyph.yz,\n\
    );\n\
    return vec4<f32>(coverage, coverage, coverage, coverage);\n\
}}\n\
@fragment\n\
fn fs_solver_vertical(input: VertexOutput) -> @location(0) vec4<f32> {{\n\
    let coverage = render_single(\n\
        vec2<f32>(488.003997802734375, 187.9630126953125),\n\
        vec2<f32>(0.1, 0.1),\n\
        input.banding,\n\
        u32(input.glyph.x),\n\
        input.glyph.yz,\n\
    );\n\
    return vec4<f32>(coverage, coverage, coverage, coverage);\n\
}}\n"
    )
}

#[test]
#[ignore = "Requires GPU or software renderer (wgpu adapter)"]
fn solver_cancellation_regression() {
    let (device, queue) = common::create_test_device();
    let tolerance = 2.0 / 255.0;

    for axis in [Axis::Horizontal, Axis::Vertical] {
        let data = glyph_data(axis);
        let fixed = common::render_coverage(
            &device,
            &queue,
            axis.fragment_entry(),
            shader_source(false),
            &data,
        );
        let broken = common::render_coverage(
            &device,
            &queue,
            axis.fragment_entry(),
            shader_source(true),
            &data,
        );
        assert!(
            (fixed - 0.75).abs() <= tolerance,
            "{} fixed coverage {fixed}, broken coverage {broken}; expected fixed coverage near 0.75",
            axis.name(),
        );
        assert!(
            (broken - 0.75).abs() > tolerance,
            "{} fixed coverage {fixed}, broken coverage {broken}; expected broken coverage away from 0.75",
            axis.name(),
        );
    }
}
