// ===================================================
// Slug algorithm — WGSL translation from HLSL reference.
// Original: Copyright 2017, Eric Lengyel (MIT License).
// ===================================================

// Constants
const K_LOG_BAND_TEXTURE_WIDTH: u32 = 12u;
const K_BAND_TEXTURE_WIDTH: u32 = 4096u;

// Uniforms
struct Params {
    // MVP matrix rows
    matrix_0: vec4<f32>,
    matrix_1: vec4<f32>,
    matrix_2: vec4<f32>,
    matrix_3: vec4<f32>,
    // Viewport dimensions in pixels (xy)
    viewport: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(1) @binding(0) var curve_texture: texture_2d<f32>;
@group(1) @binding(1) var band_texture: texture_2d<u32>;

// Vertex input — one SlugVertex per instance, 4 vertices per quad
struct VertexInput {
    // Per-instance attributes
    @location(0) pos: vec4<f32>,    // xy = object-space position, zw = normal
    @location(1) tex: vec4<f32>,    // xy = em-space coords, zw = packed glyph data
    @location(2) jac: vec4<f32>,    // inverse Jacobian (00, 01, 10, 11)
    @location(3) bnd: vec4<f32>,    // band scale xy, band offset zw
    @location(4) col: vec4<f32>,    // RGBA color
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) texcoord: vec2<f32>,
    @location(2) @interpolate(flat) banding: vec4<f32>,
    @location(3) @interpolate(flat) glyph: vec4<i32>,
}

// --- Vertex shader ---

fn slug_unpack(tex: vec4<f32>, bnd: vec4<f32>) -> vec4<i32> {
    let gz = bitcast<u32>(tex.z);
    let gw = bitcast<u32>(tex.w);
    return vec4<i32>(
        i32(gz & 0xFFFFu),
        i32(gz >> 16u),
        i32(gw & 0xFFFFu),
        i32(gw >> 16u),
    );
}

fn slug_dilate(
    pos: vec4<f32>,
    tex: vec4<f32>,
    jac: vec4<f32>,
    m0: vec4<f32>,
    m1: vec4<f32>,
    m3: vec4<f32>,
    dim: vec2<f32>,
) -> vec3<f32> {
    // Returns: xy = dilated em-space texcoord, z is unused
    // Also computes dilated position, which we return via a separate path.
    // We pack both outputs: first call for texcoord, recompute for position.
    // Actually, let's return 4 floats: dilated_pos.xy, dilated_tex.xy
    // We'll use a different approach since WGSL doesn't have out params.

    let n = normalize(pos.zw);
    let s = dot(m3.xy, pos.xy) + m3.w;
    let t = dot(m3.xy, n);

    let u = (s * dot(m0.xy, n) - t * (dot(m0.xy, pos.xy) + m0.w)) * dim.x;
    let v = (s * dot(m1.xy, n) - t * (dot(m1.xy, pos.xy) + m1.w)) * dim.y;

    let s2 = s * s;
    let st = s * t;
    let uv = u * u + v * v;
    let d = pos.zw * (s2 * (st + sqrt(uv)) / (uv - st * st));

    // This function can't easily return both pos and tex in WGSL without a struct.
    // We'll inline it in the vertex main instead.
    return vec3<f32>(d.x, d.y, 0.0); // placeholder
}

struct DilateResult {
    dilated_pos: vec2<f32>,
    dilated_tex: vec2<f32>,
}

fn compute_dilate(
    pos: vec4<f32>,
    tex: vec4<f32>,
    jac: vec4<f32>,
    m0: vec4<f32>,
    m1: vec4<f32>,
    m3: vec4<f32>,
    dim: vec2<f32>,
) -> DilateResult {
    let n = normalize(pos.zw);
    let s = dot(m3.xy, pos.xy) + m3.w;
    let t = dot(m3.xy, n);

    let u = (s * dot(m0.xy, n) - t * (dot(m0.xy, pos.xy) + m0.w)) * dim.x;
    let v = (s * dot(m1.xy, n) - t * (dot(m1.xy, pos.xy) + m1.w)) * dim.y;

    let s2 = s * s;
    let st = s * t;
    let uv = u * u + v * v;
    let d = pos.zw * (s2 * (st + sqrt(uv)) / (uv - st * st));

    let dilated_pos = pos.xy + d;
    let dilated_tex = vec2<f32>(
        tex.x + dot(d, jac.xy),
        tex.y + dot(d, jac.zw),
    );

    return DilateResult(dilated_pos, dilated_tex);
}

@vertex
fn vs_main(input: VertexInput, @builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var output: VertexOutput;

    // Apply dynamic dilation
    let dilated = compute_dilate(
        input.pos, input.tex, input.jac,
        params.matrix_0, params.matrix_1, params.matrix_3,
        params.viewport.xy,
    );

    let p = dilated.dilated_pos;
    output.texcoord = dilated.dilated_tex;

    // Apply MVP matrix to dilated vertex position
    output.position = vec4<f32>(
        p.x * params.matrix_0.x + p.y * params.matrix_0.y + params.matrix_0.w,
        p.x * params.matrix_1.x + p.y * params.matrix_1.y + params.matrix_1.w,
        p.x * params.matrix_2.x + p.y * params.matrix_2.y + params.matrix_2.w,
        p.x * params.matrix_3.x + p.y * params.matrix_3.y + params.matrix_3.w,
    );

    // Unpack glyph data
    output.glyph = slug_unpack(input.tex, input.bnd);
    output.banding = input.bnd;
    output.color = input.col;

    return output;
}

// --- Fragment shader ---

fn calc_root_code(y1: f32, y2: f32, y3: f32) -> u32 {
    let i1 = bitcast<u32>(y1) >> 31u;
    let i2 = bitcast<u32>(y2) >> 30u;
    let i3 = bitcast<u32>(y3) >> 29u;

    var shift = (i2 & 2u) | (i1 & ~2u);
    shift = (i3 & 4u) | (shift & ~4u);

    return (0x2E74u >> shift) & 0x0101u;
}

fn solve_horiz_poly(p12: vec4<f32>, p3: vec2<f32>) -> vec2<f32> {
    let a = p12.xy - p12.zw * 2.0 + p3;
    let b = p12.xy - p12.zw;
    let ra = 1.0 / a.y;
    let rb = 0.5 / b.y;

    let d = sqrt(max(b.y * b.y - a.y * p12.y, 0.0));
    var t1 = (b.y - d) * ra;
    var t2 = (b.y + d) * ra;

    if abs(a.y) < 1.0 / 65536.0 {
        let lin = p12.y * rb;
        t1 = lin;
        t2 = lin;
    }

    return vec2<f32>(
        (a.x * t1 - b.x * 2.0) * t1 + p12.x,
        (a.x * t2 - b.x * 2.0) * t2 + p12.x,
    );
}

fn solve_vert_poly(p12: vec4<f32>, p3: vec2<f32>) -> vec2<f32> {
    let a = p12.xy - p12.zw * 2.0 + p3;
    let b = p12.xy - p12.zw;
    let ra = 1.0 / a.x;
    let rb = 0.5 / b.x;

    let d = sqrt(max(b.x * b.x - a.x * p12.x, 0.0));
    var t1 = (b.x - d) * ra;
    var t2 = (b.x + d) * ra;

    if abs(a.x) < 1.0 / 65536.0 {
        let lin = p12.x * rb;
        t1 = lin;
        t2 = lin;
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

fn calc_coverage(xcov: f32, ycov: f32, xwgt: f32, ywgt: f32) -> f32 {
    let combined = abs(xcov * xwgt + ycov * ywgt) / max(xwgt + ywgt, 1.0 / 65536.0);
    let fallback = min(abs(xcov), abs(ycov));
    let coverage = max(combined, fallback);
    // Nonzero fill rule
    return clamp(coverage, 0.0, 1.0);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let render_coord = input.texcoord;
    let band_transform = input.banding;
    let glyph_data = input.glyph;

    // Effective pixel dimensions in em-space
    let ems_per_pixel = fwidth(render_coord);
    let pixels_per_em = 1.0 / ems_per_pixel;

    var band_max = glyph_data.zw;
    band_max.y &= 0x00FF;

    // Determine band indices
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
        let curve_loc = vec2<i32>(textureLoad(band_texture, vec2<i32>(hband_loc.x + ci, hband_loc.y), 0).xy);
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
        let curve_loc = vec2<i32>(textureLoad(band_texture, vec2<i32>(vband_loc.x + ci, vband_loc.y), 0).xy);
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

    let coverage = calc_coverage(xcov, ycov, xwgt, ywgt);
    return input.color * coverage;
}
