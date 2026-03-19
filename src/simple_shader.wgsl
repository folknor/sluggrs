// Simplified Slug shader for the proof-of-concept.
// Uses a simple 2D orthographic projection (no dilation).
// The fragment shader is the full Slug curve evaluator.

const K_LOG_BAND_TEXTURE_WIDTH: u32 = 12u;
const K_BAND_TEXTURE_WIDTH: u32 = 4096u;

struct Params {
    screen_size: vec2<f32>,
    _pad: vec2<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(1) @binding(0) var curve_texture: texture_2d<f32>;
@group(1) @binding(1) var band_texture: texture_2d<u32>;

// Per-instance data for a glyph
struct GlyphInstance {
    // Screen-space position and size of the glyph quad
    @location(0) screen_rect: vec4<f32>,     // x, y, width, height
    // Em-space bounds of the glyph
    @location(1) em_rect: vec4<f32>,         // min_x, min_y, max_x, max_y
    // Band transform
    @location(2) band_transform: vec4<f32>,  // scale_x, scale_y, offset_x, offset_y
    // Packed glyph data (same as Slug: glyph_loc.x, glyph_loc.y, band_max.x, band_max.y_with_flags)
    @location(3) glyph_data: vec4<u32>,
    // Color
    @location(4) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) texcoord: vec2<f32>,               // em-space coordinates
    @location(2) @interpolate(flat) banding: vec4<f32>,
    @location(3) @interpolate(flat) glyph: vec4<i32>,
}

@vertex
fn vs_main(instance: GlyphInstance, @builtin(vertex_index) vid: u32) -> VertexOutput {
    var output: VertexOutput;

    // Generate quad corners from vertex index (0..3 as triangle strip)
    let corner = vec2<f32>(
        f32(vid & 1u),        // 0, 1, 0, 1
        f32((vid >> 1u) & 1u) // 0, 0, 1, 1
    );

    // Screen-space position
    let screen_pos = instance.screen_rect.xy + corner * instance.screen_rect.zw;

    // Convert to NDC: [0, screen_size] → [-1, 1], flip Y
    let ndc = vec2<f32>(
        screen_pos.x / params.screen_size.x * 2.0 - 1.0,
        -(screen_pos.y / params.screen_size.y * 2.0 - 1.0),
    );

    output.position = vec4<f32>(ndc, 0.0, 1.0);

    // Interpolate em-space coordinates across the quad
    output.texcoord = vec2<f32>(
        mix(instance.em_rect.x, instance.em_rect.z, corner.x),
        // Flip Y for em-space (font coords are Y-up, screen is Y-down)
        mix(instance.em_rect.w, instance.em_rect.y, corner.y),
    );

    output.banding = instance.band_transform;
    output.glyph = vec4<i32>(instance.glyph_data);
    output.color = instance.color;

    return output;
}

// --- Fragment shader: full Slug curve evaluation ---

fn calc_root_code(y1: f32, y2: f32, y3: f32) -> u32 {
    // Simple sign-based version: avoid bitcast tricks that may differ across GPUs
    let s1 = select(0u, 1u, y1 < 0.0);
    let s2 = select(0u, 1u, y2 < 0.0);
    let s3 = select(0u, 1u, y3 < 0.0);
    let shift = s1 | (s2 << 1u) | (s3 << 2u);
    return (0x2E74u >> shift) & 0x0101u;
}

fn solve_horiz_poly(p12: vec4<f32>, p3: vec2<f32>) -> vec2<f32> {
    let a = p12.xy - p12.zw * 2.0 + p3;
    let b = p12.xy - p12.zw;

    var t1: f32;
    var t2: f32;

    // Threshold must exceed the perturbation applied to line segments
    // on the CPU side (|a| ≤ 0.2 for perturbed lines, genuine curves
    // have |a| in the tens+). 0.5 safely catches all near-linear cases.
    if abs(a.y) < 0.5 {
        let rb = 0.5 / b.y;
        let lin = p12.y * rb;
        t1 = lin;
        t2 = lin;
    } else {
        let ra = 1.0 / a.y;
        let d = sqrt(max(b.y * b.y - a.y * p12.y, 0.0));
        t1 = (b.y - d) * ra;
        t2 = (b.y + d) * ra;
    }

    return vec2<f32>(
        (a.x * t1 - b.x * 2.0) * t1 + p12.x,
        (a.x * t2 - b.x * 2.0) * t2 + p12.x,
    );
}

fn solve_vert_poly(p12: vec4<f32>, p3: vec2<f32>) -> vec2<f32> {
    let a = p12.xy - p12.zw * 2.0 + p3;
    let b = p12.xy - p12.zw;

    var t1: f32;
    var t2: f32;

    if abs(a.x) < 0.5 {
        let rb = 0.5 / b.x;
        let lin = p12.x * rb;
        t1 = lin;
        t2 = lin;
    } else {
        let ra = 1.0 / a.x;
        let d = sqrt(max(b.x * b.x - a.x * p12.x, 0.0));
        t1 = (b.x - d) * ra;
        t2 = (b.x + d) * ra;
    }

    return vec2<f32>(
        (a.y * t1 - b.y * 2.0) * t1 + p12.y,
        (a.y * t2 - b.y * 2.0) * t2 + p12.y,
    );
}

fn calc_band_loc(glyph_loc: vec2<i32>, offset: u32) -> vec2<i32> {
    var band_loc = vec2<i32>(glyph_loc.x + i32(offset), glyph_loc.y);
    band_loc.y += band_loc.x >> K_LOG_BAND_TEXTURE_WIDTH;
    band_loc.x &= i32(K_BAND_TEXTURE_WIDTH) - 1;
    return band_loc;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let render_coord = input.texcoord;
    let band_transform = input.banding;
    let glyph_data = input.glyph;

    let ems_per_pixel = fwidth(render_coord);
    let pixels_per_em = 1.0 / ems_per_pixel;

    var band_max = glyph_data.zw;
    band_max.y &= 0x00FF;

    let band_index = clamp(
        vec2<i32>(render_coord * band_transform.xy + band_transform.zw),
        vec2<i32>(0, 0),
        band_max,
    );
    let glyph_loc = glyph_data.xy;

    // --- Horizontal ray casting ---
    var xcov = 0.0;
    var xwgt = 0.0;

    let hband_data = textureLoad(band_texture, vec2<i32>(glyph_loc.x + band_index.y, glyph_loc.y), 0).xy;
    let hband_loc = calc_band_loc(glyph_loc, hband_data.y);

    for (var ci = 0; ci < i32(hband_data.x); ci++) {
        let curve_ref = textureLoad(band_texture, vec2<i32>(hband_loc.x + ci, hband_loc.y), 0).xy;
        let curve_loc = vec2<i32>(curve_ref);
        let p12 = textureLoad(curve_texture, curve_loc, 0) - vec4<f32>(render_coord, render_coord);
        let p3 = textureLoad(curve_texture, vec2<i32>(curve_loc.x + 1, curve_loc.y), 0).xy - render_coord;

        if max(max(p12.x, p12.z), p3.x) * pixels_per_em.x < -0.5 {
            break;
        }

        let code = calc_root_code(p12.y, p12.w, p3.y);
        if code != 0u {
            let r = solve_horiz_poly(p12, p3) * pixels_per_em.x;

            if (code & 1u) != 0u {
                xcov += clamp(r.x + 0.5, 0.0, 1.0);
                xwgt = max(xwgt, clamp(1.0 - abs(r.x) * 2.0, 0.0, 1.0));
            }
            if code > 1u {
                xcov -= clamp(r.y + 0.5, 0.0, 1.0);
                xwgt = max(xwgt, clamp(1.0 - abs(r.y) * 2.0, 0.0, 1.0));
            }
        }
    }

    // --- Vertical ray casting ---
    var ycov = 0.0;
    var ywgt = 0.0;

    let vband_data = textureLoad(band_texture, vec2<i32>(glyph_loc.x + band_max.y + 1 + band_index.x, glyph_loc.y), 0).xy;
    let vband_loc = calc_band_loc(glyph_loc, vband_data.y);

    for (var ci = 0; ci < i32(vband_data.x); ci++) {
        let curve_ref = textureLoad(band_texture, vec2<i32>(vband_loc.x + ci, vband_loc.y), 0).xy;
        let curve_loc = vec2<i32>(curve_ref);
        let p12 = textureLoad(curve_texture, curve_loc, 0) - vec4<f32>(render_coord, render_coord);
        let p3 = textureLoad(curve_texture, vec2<i32>(curve_loc.x + 1, curve_loc.y), 0).xy - render_coord;

        if max(max(p12.y, p12.w), p3.y) * pixels_per_em.y < -0.5 {
            break;
        }

        let code = calc_root_code(p12.x, p12.z, p3.x);
        if code != 0u {
            let r = solve_vert_poly(p12, p3) * pixels_per_em.y;

            if (code & 1u) != 0u {
                ycov -= clamp(r.x + 0.5, 0.0, 1.0);
                ywgt = max(ywgt, clamp(1.0 - abs(r.x) * 2.0, 0.0, 1.0));
            }
            if code > 1u {
                ycov += clamp(r.y + 0.5, 0.0, 1.0);
                ywgt = max(ywgt, clamp(1.0 - abs(r.y) * 2.0, 0.0, 1.0));
            }
        }
    }

    // Combine coverage (original Slug formula)
    let combined = abs(xcov * xwgt + ycov * ywgt) / max(xwgt + ywgt, 1.0 / 65536.0);
    let fallback = min(abs(xcov), abs(ycov));
    let final_coverage = clamp(max(combined, fallback), 0.0, 1.0);
    return input.color * final_coverage;
}
