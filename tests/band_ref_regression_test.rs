//! GPU A/B coverage for curve-reference perpendicular bounds.

use sluggrs::{
    SIMPLE_SHADER_WGSL,
    outline::{GlyphOutline, QuadCurve},
    prep::{PrepScratch, prepare_mono},
};

mod common;

#[derive(Copy, Clone)]
enum Variant {
    Optimized,
    NoPrecheck,
    HNonStrict,
    VNonStrict,
}

fn glyph_data() -> Vec<i32> {
    let outline = GlyphOutline {
        curves: vec![QuadCurve {
            p1: [-2.0, -2.0],
            p2: [0.0, 0.0],
            p3: [3.0, 2.0],
        }],
        bounds: [-2.0, -2.0, 3.0, 2.0],
    };
    let mut scratch = PrepScratch::default();
    let prepared = prepare_mono(&outline, 1, 1, 5.0, &mut scratch)
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

fn shader_source(variant: Variant) -> String {
    let horizontal = "        if f32(curve_ref.y) > render_coord_q.y || f32(curve_ref.z) < render_coord_q.y { continue; }\n";
    let vertical = "        if f32(curve_ref.y) > render_coord_q.x || f32(curve_ref.z) < render_coord_q.x { continue; }\n";
    assert_eq!(
        SIMPLE_SHADER_WGSL.matches(horizontal).count(),
        1,
        "horizontal precheck patch must match exactly one loop"
    );
    assert_eq!(
        SIMPLE_SHADER_WGSL.matches(vertical).count(),
        1,
        "vertical precheck patch must match exactly one loop"
    );

    // The equality witnesses sit in the half-pixel AA falloff band just past
    // the curve endpoint (3, 2), where coverage comes from a single ray and
    // the max-bound equality (curve_ref.z == render_coord_q) decides whether
    // that ray sees the curve at all: (3.125, 2.0) reads 96/255 through the
    // horizontal ray and 0 if it is wrongly skipped; (3.0, 1.625) reads
    // 223/255 through the vertical ray and 0 if skipped. The other axis is
    // strictly outside its range at both samples, so the opposite mutation
    // cannot change them. Only the max bound has observable equality
    // semantics: at exact MIN equality every perpendicular delta is >= 0,
    // which can never form a mixed sign pattern in calc_root_code, so the
    // strict `>` on curve_ref.y is defensive and untestable by construction.
    let source = match variant {
        Variant::Optimized => SIMPLE_SHADER_WGSL.to_owned(),
        Variant::NoPrecheck => SIMPLE_SHADER_WGSL.replace(horizontal, "").replace(vertical, ""),
        Variant::HNonStrict => SIMPLE_SHADER_WGSL.replace(
            horizontal,
            "        if f32(curve_ref.y) >= render_coord_q.y || f32(curve_ref.z) <= render_coord_q.y { continue; }\n",
        ),
        Variant::VNonStrict => SIMPLE_SHADER_WGSL.replace(
            vertical,
            "        if f32(curve_ref.y) >= render_coord_q.x || f32(curve_ref.z) <= render_coord_q.x { continue; }\n",
        ),
    };
    format!(
        "{source}\n\
@fragment\n\
fn fs_band_h_outside(input: VertexOutput) -> @location(0) vec4<f32> {{\n\
    let coverage = render_single(vec2<f32>(3.0, 3.0), vec2<f32>(1.0, 1.0), input.banding, u32(input.glyph.x), input.glyph.yz);\n\
    return vec4<f32>(coverage);\n\
}}\n\
@fragment\n\
fn fs_band_v_outside(input: VertexOutput) -> @location(0) vec4<f32> {{\n\
    let coverage = render_single(vec2<f32>(4.0, 2.0), vec2<f32>(1.0, 1.0), input.banding, u32(input.glyph.x), input.glyph.yz);\n\
    return vec4<f32>(coverage);\n\
}}\n\
@fragment\n\
fn fs_band_h_eq(input: VertexOutput) -> @location(0) vec4<f32> {{\n\
    let coverage = render_single(vec2<f32>(3.125, 2.0), vec2<f32>(1.0, 1.0), input.banding, u32(input.glyph.x), input.glyph.yz);\n\
    return vec4<f32>(coverage);\n\
}}\n\
@fragment\n\
fn fs_band_v_eq(input: VertexOutput) -> @location(0) vec4<f32> {{\n\
    let coverage = render_single(vec2<f32>(3.0, 1.625), vec2<f32>(1.0, 1.0), input.banding, u32(input.glyph.x), input.glyph.yz);\n\
    return vec4<f32>(coverage);\n\
}}\n"
    )
}

fn coverage(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    data: &[i32],
    entry: &str,
    variant: Variant,
) -> f32 {
    common::render_coverage(device, queue, entry, shader_source(variant), data)
}

#[test]
#[ignore = "Requires GPU or software renderer (wgpu adapter)"]
fn curve_ref_perpendicular_bounds_match_baseline() {
    let (device, queue) = common::create_test_device();
    let data = glyph_data();
    let tolerance = 1.0 / 255.0;

    for entry in [
        "fs_band_h_outside",
        "fs_band_v_outside",
        "fs_band_h_eq",
        "fs_band_v_eq",
    ] {
        let optimized = coverage(&device, &queue, &data, entry, Variant::Optimized);
        let baseline = coverage(&device, &queue, &data, entry, Variant::NoPrecheck);
        assert!(
            (optimized - baseline).abs() <= tolerance,
            "{entry}: optimized coverage {optimized} differs from baseline {baseline}"
        );
    }

    let h_baseline = coverage(&device, &queue, &data, "fs_band_h_eq", Variant::NoPrecheck);
    let h_non_strict = coverage(&device, &queue, &data, "fs_band_h_eq", Variant::HNonStrict);
    let h_other_axis = coverage(&device, &queue, &data, "fs_band_h_eq", Variant::VNonStrict);
    assert!(
        h_baseline >= 10.0 / 255.0,
        "horizontal equality baseline coverage {h_baseline} is too small"
    );
    assert!(
        (h_non_strict - h_baseline).abs() > 8.0 / 255.0,
        "horizontal non-strict mutation did not change coverage: {h_non_strict} vs {h_baseline}"
    );
    assert!(
        (h_other_axis - h_baseline).abs() <= tolerance,
        "vertical mutation affected horizontal equality witness: {h_other_axis} vs {h_baseline}"
    );

    let v_baseline = coverage(&device, &queue, &data, "fs_band_v_eq", Variant::NoPrecheck);
    let v_non_strict = coverage(&device, &queue, &data, "fs_band_v_eq", Variant::VNonStrict);
    let v_other_axis = coverage(&device, &queue, &data, "fs_band_v_eq", Variant::HNonStrict);
    assert!(
        v_baseline >= 10.0 / 255.0,
        "vertical equality baseline coverage {v_baseline} is too small"
    );
    assert!(
        (v_non_strict - v_baseline).abs() > 8.0 / 255.0,
        "vertical non-strict mutation did not change coverage: {v_non_strict} vs {v_baseline}"
    );
    assert!(
        (v_other_axis - v_baseline).abs() <= tolerance,
        "horizontal mutation affected vertical equality witness: {v_other_axis} vs {v_baseline}"
    );
}
