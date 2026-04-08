// Simplified Slug shader for the proof-of-concept.
// Uses a simple 2D orthographic projection (no dilation).
// The fragment shader is the full Slug curve evaluator.

struct Params {
    screen_size: vec2<f32>,
    scroll_offset: vec2<f32>,
    flags: u32,       // bit 0: enable MSAA+stem darkening
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
const INV_UNITS: f32 = 0.25; // 1.0 / 4.0 units_per_em
// Packed storage: each i32 holds two i16 values. A logical "texel" of 4 i16
// values occupies 2 consecutive i32 elements. Texel addressing: element index
// = texel_offset * 2. Halves bandwidth vs the old array<vec4<i32>> layout.
@group(1) @binding(0) var<storage, read> atlas: array<i32>;

/// Unpack two signed i16 values from a packed i32: low 16 bits, high 16 bits.
fn unpack_lo(v: i32) -> i32 { return (v << 16) >> 16; }
fn unpack_hi(v: i32) -> i32 { return v >> 16; }

/// Read a packed texel (4 i16 values) from 2 consecutive i32 elements.
fn read_texel(idx: u32) -> vec4<i32> {
    let base = idx * 2u;
    let ab = atlas[base];
    let cd = atlas[base + 1u];
    return vec4<i32>(unpack_lo(ab), unpack_hi(ab), unpack_lo(cd), unpack_hi(cd));
}

/// Read a raw i32 directly from the flat buffer (for COLRv1 unpacked data).
fn read_raw(idx: u32) -> i32 { return atlas[idx]; }

/// Read 4 raw (non-packed) i32 values as a vec4.
/// COLRv1 commands and data are stored unpacked: each old "texel" of 4 i32
/// values occupies 4 consecutive slots in the flat buffer (2 packed texels).
/// `texel_idx` is the COLRv1 texel index; actual buffer offset = texel_idx * 4.
fn read_raw4(base: u32) -> vec4<i32> {
    return vec4<i32>(atlas[base], atlas[base + 1u], atlas[base + 2u], atlas[base + 3u]);
}

// Per-instance data for a glyph
struct GlyphInstance {
    // Screen-space position and size of the glyph quad
    @location(0) screen_rect: vec4<f32>,     // x, y, width, height
    // Em-space bounds of the glyph
    @location(1) em_rect: vec4<f32>,         // min_x, min_y, max_x, max_y
    // Band transform
    @location(2) band_transform: vec4<f32>,  // scale_x, scale_y, offset_x, offset_y
    // Packed glyph data: glyph_offset, band_max.x, band_max.y_with_flags, 0
    @location(3) glyph_data: vec4<u32>,
    // Color
    @location(4) color: vec4<f32>,
    // Depth for widget layer ordering (from iced's metadata_to_depth)
    @location(5) depth: f32,
    // Pixels per em (for MSAA/darkening thresholds)
    @location(6) ppem: f32,
    @location(7) _pad: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) texcoord: vec2<f32>,               // em-space coordinates
    @location(2) @interpolate(flat) banding: vec4<f32>,
    @location(3) @interpolate(flat) glyph: vec4<i32>,
    @location(4) @interpolate(flat) pixels_per_em: vec2<f32>,
}

@vertex
fn vs_main(instance: GlyphInstance, @builtin(vertex_index) vid: u32) -> VertexOutput {
    var output: VertexOutput;

    // Generate quad corners from vertex index (0..3 as triangle strip)
    let corner = vec2<f32>(
        f32(vid & 1u),        // 0, 1, 0, 1
        f32((vid >> 1u) & 1u) // 0, 0, 1, 1
    );

    // Outward normal for this corner: (-1,-1), (1,-1), (-1,1), (1,1)
    let normal = corner * 2.0 - 1.0;

    // Undilated screen-space position
    let base_pos = instance.screen_rect.xy + corner * instance.screen_rect.zw;

    // Apply scroll offset and dilate (push each corner outward by half a pixel)
    let screen_pos = base_pos + params.scroll_offset + normal * 0.5;

    // Convert to NDC: [0, screen_size] → [-1, 1], flip Y
    let ndc = vec2<f32>(
        screen_pos.x / params.screen_size.x * 2.0 - 1.0,
        -(screen_pos.y / params.screen_size.y * 2.0 - 1.0),
    );

    output.position = vec4<f32>(ndc, instance.depth, 1.0);

    // Undilated em-space texcoord at this corner
    let base_texcoord = vec2<f32>(
        mix(instance.em_rect.x, instance.em_rect.z, corner.x),
        // Flip Y for em-space (font coords are Y-up, screen is Y-down)
        mix(instance.em_rect.w, instance.em_rect.y, corner.y),
    );

    // Convert half-pixel dilation to em-space offset
    let em_size = vec2<f32>(
        instance.em_rect.z - instance.em_rect.x,
        instance.em_rect.w - instance.em_rect.y,
    );
    let ems_per_pixel = em_size / max(instance.screen_rect.zw, vec2<f32>(1.0, 1.0));

    // Adjust texcoord for dilation (Y negated: em Y-up, screen Y-down)
    output.texcoord = base_texcoord + vec2<f32>(normal.x, -normal.y) * ems_per_pixel * 0.5;

    output.banding = instance.band_transform;
    output.glyph = vec4<i32>(instance.glyph_data);
    output.color = instance.color;
    output.pixels_per_em = vec2<f32>(instance.ppem, instance.ppem);

    return output;
}

// --- Fragment shader: full Slug curve evaluation ---

fn calc_root_code(y1: f32, y2: f32, y3: f32) -> u32 {
    // Comparison-based sign extraction. The Slug reference uses
    // bitcast(y) >> 31 (sign bit extraction) which is faster on NVIDIA
    // but treats -0.0 as negative. On Intel Arc, intermediate calculations
    // can produce -0.0 where NVIDIA produces +0.0, causing wrong root
    // eligibility and rendering artifacts (unfilled regions). The select()
    // version handles -0.0 correctly per IEEE 754 (not less than zero).
    let s1 = select(0u, 1u, y1 < 0.0);
    let s2 = select(0u, 1u, y2 < 0.0);
    let s3 = select(0u, 1u, y3 < 0.0);
    let shift = s1 | (s2 << 1u) | (s3 << 2u);
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

    if a.y == 0.0 {
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

    if a.x == 0.0 {
        let lin = p12.x * rb;
        t1 = lin;
        t2 = lin;
    }

    return vec2<f32>(
        (a.y * t1 - b.y * 2.0) * t1 + p12.y,
        (a.y * t2 - b.y * 2.0) * t2 + p12.y,
    );
}

/// Decode an i16-stored offset: recover the original u16 value via mask.
fn decode_offset(v: i32) -> u32 {
    return u32(v) & 0xFFFFu;
}

/// Evaluate Slug coverage at a single sample point.
fn render_single(
    render_coord: vec2<f32>,
    pixels_per_em: vec2<f32>,
    band_transform: vec4<f32>,
    glyph_base: u32,
    band_max: vec2<i32>,
) -> f32 {
    let band_index = clamp(
        vec2<i32>(render_coord * band_transform.xy + band_transform.zw),
        vec2<i32>(0, 0),
        band_max,
    );

    // --- Horizontal ray casting (direction-aware) ---
    var xcov = 0.0;
    var xwgt = 0.0;

    let hband_data = read_texel(glyph_base + u32(band_index.y));
    let h_split = f32(hband_data.w) * INV_UNITS;
    let h_left_ray = render_coord.x < h_split;
    let h_data_offset = decode_offset(select(hband_data.y, hband_data.z, h_left_ray));

    for (var ci = 0u; ci < u32(hband_data.x); ci++) {
        let curve_ref = read_texel(glyph_base + h_data_offset + ci);
        let curve_offset = decode_offset(curve_ref.x);
        let raw12 = read_texel(glyph_base + curve_offset);
        let p12 = vec4<f32>(raw12) * INV_UNITS - vec4<f32>(render_coord, render_coord);
        let raw3 = read_texel(glyph_base + curve_offset + 1u);
        let p3 = vec2<f32>(raw3.xy) * INV_UNITS - render_coord;

        if h_left_ray {
            if min(min(p12.x, p12.z), p3.x) * pixels_per_em.x > 0.5 { break; }
        } else {
            if max(max(p12.x, p12.z), p3.x) * pixels_per_em.x < -0.5 { break; }
        }

        let code = calc_root_code(p12.y, p12.w, p3.y);
        if code != 0u {
            let r = solve_horiz_poly(p12, p3) * pixels_per_em.x;

            if (code & 1u) != 0u {
                let cov = select(r.x + 0.5, 0.5 - r.x, h_left_ray);
                xcov += clamp(cov, 0.0, 1.0);
                xwgt = max(xwgt, clamp(1.0 - abs(r.x) * 2.0, 0.0, 1.0));
            }
            if code > 1u {
                let cov = select(r.y + 0.5, 0.5 - r.y, h_left_ray);
                xcov -= clamp(cov, 0.0, 1.0);
                xwgt = max(xwgt, clamp(1.0 - abs(r.y) * 2.0, 0.0, 1.0));
            }
        }
    }

    // --- Vertical ray casting (direction-aware) ---
    var ycov = 0.0;
    var ywgt = 0.0;

    let vband_data = read_texel(glyph_base + u32(band_max.y + 1 + band_index.x));
    let v_split = f32(vband_data.w) * INV_UNITS;
    let v_left_ray = render_coord.y < v_split;
    let v_data_offset = decode_offset(select(vband_data.y, vband_data.z, v_left_ray));

    for (var ci = 0u; ci < u32(vband_data.x); ci++) {
        let curve_ref = read_texel(glyph_base + v_data_offset + ci);
        let curve_offset = decode_offset(curve_ref.x);
        let raw12 = read_texel(glyph_base + curve_offset);
        let p12 = vec4<f32>(raw12) * INV_UNITS - vec4<f32>(render_coord, render_coord);
        let raw3 = read_texel(glyph_base + curve_offset + 1u);
        let p3 = vec2<f32>(raw3.xy) * INV_UNITS - render_coord;

        if v_left_ray {
            if min(min(p12.y, p12.w), p3.y) * pixels_per_em.y > 0.5 { break; }
        } else {
            if max(max(p12.y, p12.w), p3.y) * pixels_per_em.y < -0.5 { break; }
        }

        let code = calc_root_code(p12.x, p12.z, p3.x);
        if code != 0u {
            let r = solve_vert_poly(p12, p3) * pixels_per_em.y;

            if (code & 1u) != 0u {
                let cov = select(r.x + 0.5, 0.5 - r.x, v_left_ray);
                ycov -= clamp(cov, 0.0, 1.0);
                ywgt = max(ywgt, clamp(1.0 - abs(r.x) * 2.0, 0.0, 1.0));
            }
            if code > 1u {
                let cov = select(r.y + 0.5, 0.5 - r.y, v_left_ray);
                ycov += clamp(cov, 0.0, 1.0);
                ywgt = max(ywgt, clamp(1.0 - abs(r.y) * 2.0, 0.0, 1.0));
            }
        }
    }

    // Combine coverage (original Slug formula)
    let combined = abs(xcov * xwgt + ycov * ywgt) / max(xwgt + ywgt, 1.0 / 65536.0);
    let fallback = min(abs(xcov), abs(ycov));
    return clamp(max(combined, fallback), 0.0, 1.0);
}

/// Stem darkening: thicken thin stems at small sizes.
/// Gamma curve with brightness-dependent exponent, no-op above 48ppem.
fn darken(coverage: f32, brightness: f32, ppem: f32) -> f32 {
    return pow(coverage,
        mix(pow(2.0, brightness - 0.5), 1.0, smoothstep(8.0, 48.0, ppem)));
}

// ── COLRv1 color glyph support ──────────────────────────────────────────

// Command opcodes (must match CMD_* constants in outline.rs)
const CMD_PUSH_GROUP: i32 = 1;
const CMD_DRAW_SOLID: i32 = 2;
const CMD_DRAW_GRADIENT: i32 = 3;
const CMD_POP_GROUP: i32 = 4;

// Read a fixed-point value from two i16-in-i32 fields (integer + fractional).
fn read_fixed(integer: i32, fractional: i32) -> f32 {
    return f32(integer) + f32(fractional) / f32(1 << 15);
}

// Unpack RGBA from two packed i32 values [R_G, B_A].
fn unpack_color(rg: i32, ba: i32) -> vec4<f32> {
    let rgu = u32(rg) & 0xFFFFu;
    let bau = u32(ba) & 0xFFFFu;
    return vec4<f32>(
        f32(rgu >> 8u) / 255.0,
        f32(rgu & 0xFFu) / 255.0,
        f32(bau >> 8u) / 255.0,
        f32(bau & 0xFFu) / 255.0,
    );
}

// Evaluate coverage for a sub-glyph whose header is at blob_base + sub_offset.
// Sub-glyph header layout (3 packed texels = 6 i32 slots):
//   texel 0: band_max_x, band_max_y (packed i16 pair) + padding
//   texels 1-2: band_transform (4 raw i32, bitcast f32)
fn render_sub_glyph(
    render_coord: vec2<f32>,
    pixels_per_em: vec2<f32>,
    blob_base: u32,
    sub_offset: u32,
) -> f32 {
    let header_base = blob_base + sub_offset;
    let h0 = read_texel(header_base);
    let band_max = vec2<i32>(h0.x, h0.y);
    // band_transform is 4 raw i32 values at i32 indices (header_base + 1) * 2 + 2
    // They span texels 1-2 as raw (non-packed) i32 data.
    let raw_base = header_base * 2u + 2u; // skip texel 0 (2 i32s)
    let band_transform = vec4<f32>(
        bitcast<f32>(read_raw(raw_base)),
        bitcast<f32>(read_raw(raw_base + 1u)),
        bitcast<f32>(read_raw(raw_base + 2u)),
        bitcast<f32>(read_raw(raw_base + 3u)),
    );
    // Sub-glyph band+curve data starts right after the 3-texel header
    let sub_glyph_base = header_base + 3u;
    return render_single(render_coord, pixels_per_em, band_transform, sub_glyph_base, band_max);
}

// Porter-Duff and blend mode compositing.
// mode values match skrifa::color::CompositeMode enum order.
fn composite_colors(src: vec4<f32>, dst: vec4<f32>, mode: i32) -> vec4<f32> {
    // All inputs/outputs are premultiplied alpha.
    let sa = src.a;
    let da = dst.a;
    switch mode {
        // 0: Clear
        case 0: { return vec4<f32>(0.0); }
        // 1: Source
        case 1: { return src; }
        // 2: Destination
        case 2: { return dst; }
        // 3: SourceOver (default)
        case 3: { return src + dst * (1.0 - sa); }
        // 4: DestinationOver
        case 4: { return dst + src * (1.0 - da); }
        // 5: SourceIn
        case 5: { return src * da; }
        // 6: DestinationIn
        case 6: { return dst * sa; }
        // 7: SourceOut
        case 7: { return src * (1.0 - da); }
        // 8: DestinationOut
        case 8: { return dst * (1.0 - sa); }
        // 9: SourceAtop
        case 9: { return src * da + dst * (1.0 - sa); }
        // 10: DestinationAtop
        case 10: { return dst * sa + src * (1.0 - da); }
        // 11: Xor
        case 11: { return src * (1.0 - da) + dst * (1.0 - sa); }
        // 12: Plus (Lighter)
        case 12: { return min(src + dst, vec4<f32>(1.0)); }
        // 13: Screen
        case 13: { return src + dst - src * dst; }
        // 14: Multiply
        case 14: { return src * dst + src * (1.0 - da) + dst * (1.0 - sa); }
        // Fallback: SourceOver
        default: { return src + dst * (1.0 - sa); }
    }
}

// Interpolate along a color line (gradient color stops).
// Stops are encoded as raw i32 quads: [offset_int, offset_frac, R_G_packed, B_A_packed]
// raw_base is the raw i32 offset; each stop occupies 4 raw i32 values.
fn evaluate_color_line(raw_base: u32, num_stops: i32, t: f32) -> vec4<f32> {
    if num_stops <= 0 { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
    if num_stops == 1 {
        let s = read_raw4(raw_base);
        return unpack_color(s.z, s.w);
    }
    // Clamp t to [first_stop, last_stop]
    let first = read_raw4(raw_base);
    let last = read_raw4(raw_base + u32(num_stops - 1) * 4u);
    let t_first = read_fixed(first.x, first.y);
    let t_last = read_fixed(last.x, last.y);
    let tc = clamp(t, t_first, t_last);

    // Find the two stops surrounding tc
    var c0 = unpack_color(first.z, first.w);
    var c1 = c0;
    var t0 = t_first;
    var t1 = t_first;
    for (var i = 1; i < num_stops && i < 16; i++) {
        let s = read_raw4(raw_base + u32(i) * 4u);
        t1 = read_fixed(s.x, s.y);
        c1 = unpack_color(s.z, s.w);
        if t1 >= tc { break; }
        c0 = c1;
        t0 = t1;
    }
    let range = t1 - t0;
    if range < 1e-6 { return c1; }
    let frac = (tc - t0) / range;
    return mix(c0, c1, frac);
}

// Evaluate linear gradient: project point onto line p0→p1, return parameter t.
fn eval_linear_gradient(p0: vec2<f32>, p1: vec2<f32>, uv: vec2<f32>) -> f32 {
    let d = p1 - p0;
    let len_sq = dot(d, d);
    if len_sq < 1e-12 { return 0.0; }
    return dot(uv - p0, d) / len_sq;
}

// Evaluate radial gradient between two circles (c0,r0) and (c1,r1).
fn eval_radial_gradient(c0: vec2<f32>, r0: f32, c1: vec2<f32>, r1: f32, uv: vec2<f32>) -> f32 {
    let cd = c1 - c0;
    let rd = r1 - r0;
    let pd = uv - c0;
    let a = dot(cd, cd) - rd * rd;
    let b = dot(pd, cd) - r0 * rd;
    let c = dot(pd, pd) - r0 * r0;

    if abs(a) < 1e-6 {
        if abs(b) < 1e-6 { return 0.0; }
        return -c / (2.0 * b);
    }
    let disc = b * b - a * c;
    if disc < 0.0 { return 0.0; }
    let sq = sqrt(disc);
    // Pick largest t where radius is non-negative
    let t1 = (b + sq) / a;
    let t2 = (b - sq) / a;
    if r0 + t1 * rd >= 0.0 { return t1; }
    if r0 + t2 * rd >= 0.0 { return t2; }
    return 0.0;
}

// Evaluate sweep (conical) gradient.
fn eval_sweep_gradient(center: vec2<f32>, start_angle: f32, end_angle: f32, uv: vec2<f32>) -> f32 {
    let d = uv - center;
    var angle = atan2(-d.y, d.x); // clockwise, matching COLRv1 spec
    // Normalize to degrees 0..360
    angle = angle * (180.0 / 3.14159265359);
    if angle < 0.0 { angle += 360.0; }
    let range = end_angle - start_angle;
    if abs(range) < 1e-6 { return 0.0; }
    return (angle - start_angle) / range;
}

// Read inverse transform matrix from 3 raw i32 quads (12 raw i32 values).
// raw_base is in raw i32 units.
fn read_inv_transform(raw_base: u32) -> mat3x3<f32> {
    let t0 = read_raw4(raw_base);
    let t1 = read_raw4(raw_base + 4u);
    let t2 = read_raw4(raw_base + 8u);
    return mat3x3<f32>(
        read_fixed(t0.x, t0.y), read_fixed(t0.z, t0.w), 0.0,
        read_fixed(t1.x, t1.y), read_fixed(t1.z, t1.w), 0.0,
        read_fixed(t2.x, t2.y), read_fixed(t2.z, t2.w), 1.0,
    );
}

// Main COLRv1 command interpreter.
// blob_base and cmd_count are in packed texel units. COLRv1 commands
// and auxiliary data are stored as raw i32 values (4 per old texel = 2
// packed texels). Sub-glyph offsets are in packed texel units.
fn render_color(
    render_coord: vec2<f32>,
    pixels_per_em: vec2<f32>,
    blob_base: u32,
    cmd_count: u32,
) -> vec4<f32> {
    var stack: array<vec4<f32>, 8>;
    stack[0] = vec4<f32>(0.0);
    var sp: i32 = 0;

    // Raw i32 cursor: each packed texel = 2 raw i32s, each old texel = 4 raw i32s
    let raw_base = blob_base * 2u;
    var raw_cursor: u32 = raw_base;
    let raw_end = raw_base + cmd_count * 2u;

    for (var iter = 0u; iter < 64u && raw_cursor < raw_end; iter++) {
        let cmd = read_raw4(raw_cursor);
        raw_cursor += 4u; // each command = 4 raw i32 values

        switch cmd.x {
            case 1: { // CMD_PUSH_GROUP
                sp = min(sp + 1, 7);
                stack[sp] = vec4<f32>(0.0);
            }
            case 2: { // CMD_DRAW_SOLID
                let sub_offset = u32(cmd.y); // in packed texel units
                let draw_color = unpack_color(cmd.z, cmd.w);
                let coverage = render_sub_glyph(render_coord, pixels_per_em, blob_base, sub_offset);
                let premul = vec4<f32>(draw_color.rgb * draw_color.a * coverage, draw_color.a * coverage);
                stack[sp] = composite_colors(premul, stack[sp], 3);
            }
            case 3: { // CMD_DRAW_GRADIENT
                let sub_offset = u32(cmd.y); // in packed texel units
                let gradient_type = cmd.z;
                let num_stops = cmd.w;

                // Read inverse transform (3 raw quads = 12 raw i32)
                let inv_mat = read_inv_transform(raw_cursor);
                raw_cursor += 12u;

                let uv = (inv_mat * vec3<f32>(render_coord, 1.0)).xy;

                var grad_t: f32 = 0.0;

                switch gradient_type {
                    case 0: { // Linear
                        let g0 = read_raw4(raw_cursor);
                        let g1 = read_raw4(raw_cursor + 4u);
                        raw_cursor += 8u;
                        let p0 = vec2<f32>(read_fixed(g0.x, g0.y), read_fixed(g0.z, g0.w));
                        let p1 = vec2<f32>(read_fixed(g1.x, g1.y), read_fixed(g1.z, g1.w));
                        grad_t = eval_linear_gradient(p0, p1, uv);
                    }
                    case 1: { // Radial
                        let g0 = read_raw4(raw_cursor);
                        let g1 = read_raw4(raw_cursor + 4u);
                        let g2 = read_raw4(raw_cursor + 8u);
                        raw_cursor += 12u;
                        let c0 = vec2<f32>(read_fixed(g0.x, g0.y), read_fixed(g0.z, g0.w));
                        let r0 = read_fixed(g1.x, g1.y);
                        let c1x = read_fixed(g1.z, g1.w);
                        let c1y = read_fixed(g2.x, g2.y);
                        let r1 = read_fixed(g2.z, g2.w);
                        grad_t = eval_radial_gradient(c0, r0, vec2<f32>(c1x, c1y), r1, uv);
                    }
                    case 2: { // Sweep
                        let g0 = read_raw4(raw_cursor);
                        let g1 = read_raw4(raw_cursor + 4u);
                        raw_cursor += 8u;
                        let center = vec2<f32>(read_fixed(g0.x, g0.y), read_fixed(g0.z, g0.w));
                        let start_a = read_fixed(g1.x, g1.y);
                        let end_a = read_fixed(g1.z, g1.w);
                        grad_t = eval_sweep_gradient(center, start_a, end_a, uv);
                    }
                    default: {}
                }

                // Color stops immediately follow gradient params
                let grad_color = evaluate_color_line(raw_cursor, num_stops, grad_t);
                raw_cursor += u32(num_stops) * 4u; // skip past color stops

                let coverage = render_sub_glyph(render_coord, pixels_per_em, blob_base, sub_offset);
                let premul = vec4<f32>(grad_color.rgb * grad_color.a * coverage, grad_color.a * coverage);
                stack[sp] = composite_colors(premul, stack[sp], 3);
            }
            case 4: { // CMD_POP_GROUP
                let mode = cmd.y;
                let popped = stack[max(sp, 0)];
                sp = max(sp - 1, 0);
                stack[sp] = composite_colors(popped, stack[sp], mode);
            }
            default: {}
        }
    }

    if sp >= 0 { return stack[sp]; }
    return vec4<f32>(0.0);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let render_coord = input.texcoord;
    let band_transform = input.banding;
    let glyph_data = input.glyph;

    // Per-pixel coverage needs the actual em-to-pixel ratio from fwidth
    let ems_per_pixel = max(fwidth(render_coord), vec2<f32>(1.0 / 65536.0));
    let pixels_per_em = 1.0 / ems_per_pixel;
    // Stable per-glyph ppem from instance data (for MSAA/darkening thresholds)
    let ppem = input.pixels_per_em.x;

    let glyph_base = u32(glyph_data.x);

    // COLRv1 color glyph: glyph_data.w holds command count (non-zero).
    if glyph_data.w != 0 {
        let cmd_count = u32(glyph_data.w);
        return render_color(render_coord, pixels_per_em, glyph_base, cmd_count);
    }

    var band_max = glyph_data.yz;
    band_max.y &= 0x00FF;

    // Single-sample coverage
    var coverage = render_single(render_coord, pixels_per_em, band_transform, glyph_base, band_max);

    if (params.flags & 1u) != 0u {
        // 4x MSAA for small sizes: blend in gradually from 16ppem down to 8ppem
        if ppem < 16.0 {
            let d = ems_per_pixel * (1.0 / 3.0);
            let msaa = 0.25 * (
                render_single(render_coord + vec2<f32>(-d.x, -d.y), pixels_per_em, band_transform, glyph_base, band_max) +
                render_single(render_coord + vec2<f32>( d.x, -d.y), pixels_per_em, band_transform, glyph_base, band_max) +
                render_single(render_coord + vec2<f32>(-d.x,  d.y), pixels_per_em, band_transform, glyph_base, band_max) +
                render_single(render_coord + vec2<f32>( d.x,  d.y), pixels_per_em, band_transform, glyph_base, band_max)
            );
            coverage = mix(coverage, msaa, smoothstep(16.0, 8.0, ppem));
        }

        // Stem darkening: brightness-aware gamma for thin stems at small sizes.
        // No-op above 48ppem (exponent = 1.0), so skip the pow entirely.
        if ppem < 48.0 {
            let brightness = dot(input.color.rgb, vec3<f32>(0.299, 0.587, 0.114));
            coverage = darken(coverage, brightness, ppem);
        }
    }

    // In Web color mode (linear-RGB framebuffer), convert sRGB vertex
    // colors to linear before blending. Accurate mode (sRGB framebuffer)
    // passes through as-is — the hardware does the conversion.
    var rgb = input.color.rgb;
    if (params.flags & 2u) != 0u {
        rgb = pow(rgb, vec3<f32>(2.2));
    }

    // Premultiplied alpha output
    let alpha = input.color.a * coverage;
    return vec4<f32>(rgb * alpha, alpha);
}
