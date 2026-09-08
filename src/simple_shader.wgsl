struct Params {
    screen_size: vec2<f32>,
    scroll_offset: vec2<f32>,
    flags: u32,
    _pad: u32
}

@group(0) @binding(0) var<uniform> params: Params;
const INV_UNITS: f32 = 0.25;
const GLYPH_HEADER_TEXELS: u32 = 5u;
@group(1) @binding(0) var<storage, read> atlas: array<i32>;

fn unpack_lo(v: i32) -> i32 { return (v << 16) >> 16; }
fn unpack_hi(v: i32) -> i32 { return v >> 16; }

fn read_texel(idx: u32) -> vec4<i32> {
    let base = idx * 2u;
    let ab = atlas[base];
    let cd = atlas[base + 1u];
    return vec4<i32>(unpack_lo(ab), unpack_hi(ab), unpack_lo(cd), unpack_hi(cd));
}

fn read_raw(idx: u32) -> i32 { return atlas[idx]; }

fn read_raw4(base: u32) -> vec4<i32> {
    return vec4<i32>(atlas[base], atlas[base + 1u], atlas[base + 2u], atlas[base + 3u]);
}
fn dist_to_segment(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let ab = b - a;
    let t = clamp(dot(p - a, ab) / max(dot(ab, ab), 1e-12), 0.0, 1.0);
    return length(p - (a + ab * t));
}
struct GlyphInstance {
    @location(0) screen_rect: vec4<f32>,
    @location(1) color: vec4<f32>,
    @location(2) border_color: vec4<f32>,
    @location(3) glyph_offset: u32,
    @location(4) cmd_texel_count: u32,
    @location(5) depth: f32,
    @location(6) ppem: f32,
    @location(7) border_width: f32,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) texcoord: vec2<f32>,
    @location(2) @interpolate(flat) banding: vec4<f32>,
    @location(3) @interpolate(flat) glyph: vec4<i32>,
    @location(4) @interpolate(flat) pixels_per_em: vec2<f32>,
    @location(5) @interpolate(flat) border_color: vec4<f32>,
    @location(6) @interpolate(flat) border_width: f32,
    @location(7) @interpolate(flat) em_size: vec2<f32>,
}

@vertex
fn vs_main(instance: GlyphInstance, @builtin(vertex_index) vid: u32) -> VertexOutput {
    var output: VertexOutput;
    let corner = vec2<f32>(
        f32(vid & 1u),
        f32((vid >> 1u) & 1u)
    );
    let header_raw = instance.glyph_offset * 2u;
    let em_rect = vec4<f32>(
        bitcast<f32>(atlas[header_raw]),
        bitcast<f32>(atlas[header_raw + 1u]),
        bitcast<f32>(atlas[header_raw + 2u]),
        bitcast<f32>(atlas[header_raw + 3u]),
    );
    let band_transform = vec4<f32>(
        bitcast<f32>(atlas[header_raw + 4u]),
        bitcast<f32>(atlas[header_raw + 5u]),
        bitcast<f32>(atlas[header_raw + 6u]),
        bitcast<f32>(atlas[header_raw + 7u]),
    );
    let band_max = read_texel(instance.glyph_offset + GLYPH_HEADER_TEXELS - 1u).xy;
    let base_texcoord = vec2<f32>(
        mix(em_rect.x, em_rect.z, corner.x),
        mix(em_rect.w, em_rect.y, corner.y),
    );
    let em_size = vec2<f32>(
        em_rect.z - em_rect.x,
        em_rect.w - em_rect.y
    );

    let border_margin_px = instance.border_width * (instance.screen_rect.z + instance.screen_rect.w);

    let normal = corner * 2.0 - 1.0;
    let expand = border_margin_px + 0.5;
    let base_pos = instance.screen_rect.xy - vec2<f32>(expand) +
        corner * (instance.screen_rect.zw + vec2<f32>(expand * 2.0));
    let screen_pos = base_pos + params.scroll_offset;
    let ndc = vec2<f32>(
        screen_pos.x / params.screen_size.x * 2.0 - 1.0,
        -(screen_pos.y / params.screen_size.y * 2.0 - 1.0),
    );
    output.position = vec4<f32>(ndc, instance.depth, 1.0);
    let ems_per_pixel = em_size / max(instance.screen_rect.zw, vec2<f32>(1.0, 1.0));
    output.texcoord =
        base_texcoord +
        vec2<f32>(normal.x, -normal.y) *
        ems_per_pixel *
        expand;
    output.banding = band_transform;
    output.glyph = vec4<i32>(
        i32(instance.glyph_offset + GLYPH_HEADER_TEXELS),
        band_max.x,
        band_max.y,
        i32(instance.cmd_texel_count),
    );
    output.color = instance.color;
    output.border_color = instance.border_color;
    output.border_width = instance.border_width;
    output.pixels_per_em = vec2<f32>(instance.ppem, instance.ppem);
    output.em_size = em_size;
    return output;
}

fn sd_bezier(pos: vec2<f32>, p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>) -> f32 {
    let a = p1 - p0;
    let b = p0 - p1 * 2.0 + p2;
    let c = a * 2.0;
    let d = p0 - pos;

    let kk = 1.0 / max(dot(b, b), 1e-9);
    let kx = kk * dot(a, b);
    let ky = kk * (2.0 * dot(a, a) + dot(d, b)) / 3.0;
    let kz = kk * dot(d, a);

    var res = 0.0;
    let p = ky - kx * kx;
    let p3 = p * p * p;
    let q = kx * (2.0 * kx * kx - 3.0 * ky) + kz;
    var h = q * q + 4.0 * p3;

    if h >= 0.0 {
        h = sqrt(h);
        let x = (vec2<f32>(h, -h) - vec2<f32>(q, q)) / 2.0;
        let uv = sign(x) * pow(abs(x), vec2<f32>(1.0 / 3.0));
        let t = clamp(uv.x + uv.y - kx, 0.0, 1.0);
        let qq = d + (c + b * t) * t;
        res = dot(qq, qq);
    } else {
        let z = sqrt(-p);
        let v = acos(q / (p * z * 2.0)) / 3.0;
        let m = cos(v);
        let n = sin(v) * 1.732050808;
        let t = clamp(vec2<f32>(m + m, -n - m) * z - vec2<f32>(kx, kx), vec2<f32>(0.0), vec2<f32>(1.0));
        let qx = d + (c + b * t.x) * t.x;
        let dx = dot(qx, qx);
        let qy = d + (c + b * t.y) * t.y;
        let dy = dot(qy, qy);
        res = min(dx, dy);
    }

    return sqrt(res);
}
fn calc_root_code(y1: f32, y2: f32, y3: f32) -> u32 {
    let s1 = select(0u, 1u, y1 < 0.0);
    let s2 = select(0u, 1u, y2 < 0.0);
    let s3 = select(0u, 1u, y3 < 0.0);
    let shift = s1 | (s2 << 1u) | (s3 << 2u);
    return (0x2E74u >> shift) & 0x0101u;
}

fn solve_horiz_poly(a: vec2<f32>, b: vec2<f32>, p1: vec2<f32>) -> vec2<f32> {
    let ra = 1.0 / a.y;
    let rb = 0.5 / b.y;
    let d = sqrt(max(b.y * b.y - a.y * p1.y, 0.0));
    var t1 = (b.y - d) * ra;
    var t2 = (b.y + d) * ra;

    if a.y == 0.0 {
        let lin = p1.y * rb;
        t1 = lin;
        t2 = lin;
    }

    return vec2<f32>(
        (a.x * t1 - b.x * 2.0) * t1 + p1.x,
        (a.x * t2 - b.x * 2.0) * t2 + p1.x,
    );
}

fn solve_vert_poly(a: vec2<f32>, b: vec2<f32>, p1: vec2<f32>) -> vec2<f32> {
    let ra = 1.0 / a.x;
    let rb = 0.5 / b.x;
    let d = sqrt(max(b.x * b.x - a.x * p1.x, 0.0));
    var t1 = (b.x - d) * ra;
    var t2 = (b.x + d) * ra;

    if a.x == 0.0 {
        let lin = p1.x * rb;
        t1 = lin;
        t2 = lin;
    }

    return vec2<f32>(
        (a.y * t1 - b.y * 2.0) * t1 + p1.y,
        (a.y * t2 - b.y * 2.0) * t2 + p1.y,
    );
}

fn decode_offset(v: i32) -> u32 {
    return u32(v) & 0xFFFFu;
}

fn render_single_sd(
    render_coord: vec2<f32>,
    pixels_per_em: vec2<f32>,
    band_transform: vec4<f32>,
    glyph_base: u32,
    band_max: vec2<i32>,
    search_radius: f32
) -> vec2<f32> {
    let render_coord_q = render_coord * 4.0;
    let band_index = clamp(
        vec2<i32>(render_coord * band_transform.xy + band_transform.zw),
        vec2<i32>(0, 0),
        band_max,
    );

    var xcov = 0.0;
    var xwgt = 0.0;
    var min_dist = 1e6;

    let hband_data = read_texel(glyph_base + u32(band_index.y));
    let h_split = f32(hband_data.w) * INV_UNITS;
    let h_left_ray = render_coord.x < h_split;
    let h_data_offset = decode_offset(select(hband_data.y, hband_data.z, h_left_ray));
    let h_em_radius = search_radius / pixels_per_em.x;

    for (var ci = 0u; ci < u32(hband_data.x); ci++) {
        let curve_ref = read_texel(glyph_base + h_data_offset + ci);
        let wq = (f32(curve_ref.w) * INV_UNITS - render_coord.x) * pixels_per_em.x;
        if h_left_ray {
            if wq > search_radius { break; }
        } else {
            if wq < -search_radius { break; }
        }
        if f32(curve_ref.y) - h_em_radius * 4.0 > render_coord_q.y || f32(curve_ref.z) + h_em_radius * 4.0 < render_coord_q.y { continue; }
        let curve_offset = decode_offset(curve_ref.x);
        let raw12 = read_texel(glyph_base + curve_offset);
        let raw3 = read_texel(glyph_base + curve_offset + 1u);
        let q12 = vec4<f32>(raw12) * INV_UNITS;
        let q3 = vec2<f32>(raw3.xy) * INV_UNITS;
        let p12 = q12 - vec4<f32>(render_coord, render_coord);
        let p3 = q3 - render_coord;

        min_dist = min(min_dist, sd_bezier(render_coord, q12.xy, q12.zw, q3) * pixels_per_em.x);

        let code = calc_root_code(p12.y, p12.w, p3.y);
        if code != 0u {
            let a = q12.xy - q12.zw * 2.0 + q3;
            let b = q12.xy - q12.zw;
            let r = solve_horiz_poly(a, b, p12.xy) * pixels_per_em.x;

            if (code & 1u) != 0u {
                let cov = select(r.x + 0.5, 0.5 - r.x, h_left_ray);
                xcov += clamp(cov, 0.0, 1.0);
                xwgt = max(xwgt, clamp(1.0 - abs(r.x) * 2.0, 0.0, 1.0));
                min_dist = min(min_dist, abs(r.x));
            }
            if code > 1u {
                let cov = select(r.y + 0.5, 0.5 - r.y, h_left_ray);
                xcov -= clamp(cov, 0.0, 1.0);
                xwgt = max(xwgt, clamp(1.0 - abs(r.y) * 2.0, 0.0, 1.0));
                min_dist = min(min_dist, abs(r.y));
            }
        }
    }

    var ycov = 0.0;
    var ywgt = 0.0;

    let vband_data = read_texel(glyph_base + u32(band_max.y + 1 + band_index.x));
    let v_split = f32(vband_data.w) * INV_UNITS;
    let v_left_ray = render_coord.y < v_split;
    let v_data_offset = decode_offset(select(vband_data.y, vband_data.z, v_left_ray));
    let v_em_radius = search_radius / pixels_per_em.y;

    for (var ci = 0u; ci < u32(vband_data.x); ci++) {
        let curve_ref = read_texel(glyph_base + v_data_offset + ci);
        let wq = (f32(curve_ref.w) * INV_UNITS - render_coord.y) * pixels_per_em.y;
        if v_left_ray {
            if wq > search_radius { break; }
        } else {
            if wq < -search_radius { break; }
        }
        if f32(curve_ref.y) - v_em_radius * 4.0 > render_coord_q.x || f32(curve_ref.z) + v_em_radius * 4.0 < render_coord_q.x { continue; }
        let curve_offset = decode_offset(curve_ref.x);
        let raw12 = read_texel(glyph_base + curve_offset);
        let raw3 = read_texel(glyph_base + curve_offset + 1u);
        let q12 = vec4<f32>(raw12) * INV_UNITS;
        let q3 = vec2<f32>(raw3.xy) * INV_UNITS;
        let p12 = q12 - vec4<f32>(render_coord, render_coord);
        let p3 = q3 - render_coord;

        min_dist = min(min_dist, sd_bezier(render_coord, q12.xy, q12.zw, q3) * pixels_per_em.y);

        let code = calc_root_code(p12.x, p12.z, p3.x);
        if code != 0u {
            let a = q12.xy - q12.zw * 2.0 + q3;
            let b = q12.xy - q12.zw;
            let r = solve_vert_poly(a, b, p12.xy) * pixels_per_em.y;

            if (code & 1u) != 0u {
                let cov = select(r.x + 0.5, 0.5 - r.x, v_left_ray);
                ycov -= clamp(cov, 0.0, 1.0);
                ywgt = max(ywgt, clamp(1.0 - abs(r.x) * 2.0, 0.0, 1.0));
                min_dist = min(min_dist, abs(r.x));
            }
            if code > 1u {
                let cov = select(r.y + 0.5, 0.5 - r.y, v_left_ray);
                ycov += clamp(cov, 0.0, 1.0);
                ywgt = max(ywgt, clamp(1.0 - abs(r.y) * 2.0, 0.0, 1.0));
                min_dist = min(min_dist, abs(r.y));
            }
        }
    }

    let combined = abs(xcov * xwgt + ycov * ywgt) / max(xwgt + ywgt, 1.0 / 65536.0);
    let fallback = min(abs(xcov), abs(ycov));
    let coverage = clamp(max(combined, fallback), 0.0, 1.0);
    let signed_dist = select(min_dist, -min_dist, coverage > 0.5);

    return vec2<f32>(coverage, signed_dist);
}

fn render_single(
    render_coord: vec2<f32>,
    pixels_per_em: vec2<f32>,
    band_transform: vec4<f32>,
    glyph_base: u32,
    band_max: vec2<i32>,
) -> f32 {
    return render_single_sd(render_coord, pixels_per_em, band_transform, glyph_base, band_max, 0.5).x;
}

fn darken(coverage: f32, brightness: f32, ppem: f32) -> f32 {
    return pow(coverage,
        mix(pow(2.0, brightness - 0.5), 1.0, smoothstep(8.0, 48.0, ppem)));
}

const CMD_PUSH_GROUP: i32 = 1;
const CMD_DRAW_SOLID: i32 = 2;
const CMD_DRAW_GRADIENT: i32 = 3;
const CMD_POP_GROUP: i32 = 4;

fn read_fixed(integer: i32, fractional: i32) -> f32 {
    return f32(integer) + f32(fractional) / f32(1 << 15);
}

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

fn render_sub_glyph(
    render_coord: vec2<f32>,
    pixels_per_em: vec2<f32>,
    blob_base: u32,
    sub_offset: u32,
) -> f32 {
    let header_base = blob_base + sub_offset;
    let h0 = read_texel(header_base);
    let band_max = vec2<i32>(h0.x, h0.y);
    let raw_base = header_base * 2u + 2u;
    let band_transform = vec4<f32>(
        bitcast<f32>(read_raw(raw_base)),
        bitcast<f32>(read_raw(raw_base + 1u)),
        bitcast<f32>(read_raw(raw_base + 2u)),
        bitcast<f32>(read_raw(raw_base + 3u)),
    );
    let sub_glyph_base = header_base + 3u;
    return render_single(render_coord, pixels_per_em, band_transform, sub_glyph_base, band_max);
}

fn render_sub_glyph_sd(
    render_coord: vec2<f32>,
    pixels_per_em: vec2<f32>,
    blob_base: u32,
    sub_offset: u32,
    search_radius: f32,
) -> vec2<f32> {
    let header_base = blob_base + sub_offset;
    let h0 = read_texel(header_base);
    let band_max = vec2<i32>(h0.x, h0.y);
    let raw_base = header_base * 2u + 2u;
    let band_transform = vec4<f32>(
        bitcast<f32>(read_raw(raw_base)),
        bitcast<f32>(read_raw(raw_base + 1u)),
        bitcast<f32>(read_raw(raw_base + 2u)),
        bitcast<f32>(read_raw(raw_base + 3u)),
    );
    let sub_glyph_base = header_base + 3u;
    return render_single_sd(render_coord, pixels_per_em, band_transform, sub_glyph_base, band_max, search_radius);
}

fn composite_colors(src: vec4<f32>, dst: vec4<f32>, mode: i32) -> vec4<f32> {
    let sa = src.a;
    let da = dst.a;
    switch (mode) {
        case 0: { return vec4<f32>(0.0); }
        case 1: { return src; }
        case 2: { return dst; }
        case 3: { return src + dst * (1.0 - sa); }
        case 4: { return dst + src * (1.0 - da); }
        case 5: { return src * da; }
        case 6: { return dst * sa; }
        case 7: { return src * (1.0 - da); }
        case 8: { return dst * (1.0 - sa); }
        case 9: { return src * da + dst * (1.0 - sa); }
        case 10: { return dst * sa + src * (1.0 - da); }
        case 11: { return src * (1.0 - da) + dst * (1.0 - sa); }
        case 12: { return min(src + dst, vec4<f32>(1.0)); }
        case 13: { return src + dst - src * dst; }
        case 14: { return src * dst + src * (1.0 - da) + dst * (1.0 - sa); }
        default: { return src + dst * (1.0 - sa); }
    }
}

fn evaluate_color_line(raw_base: u32, num_stops: i32, t: f32) -> vec4<f32> {
    if num_stops <= 0 { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
    if num_stops == 1 {
        let s = read_raw4(raw_base);
        return unpack_color(s.z, s.w);
    }
    let first = read_raw4(raw_base);
    let last = read_raw4(raw_base + u32(num_stops - 1) * 4u);
    let t_first = read_fixed(first.x, first.y);
    let t_last = read_fixed(last.x, last.y);
    let tc = clamp(t, t_first, t_last);

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

fn eval_linear_gradient(p0: vec2<f32>, p1: vec2<f32>, uv: vec2<f32>) -> f32 {
    let d = p1 - p0;
    let len_sq = dot(d, d);
    if len_sq < 1e-12 { return 0.0; }
    return dot(uv - p0, d) / len_sq;
}

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
    let t1 = (b + sq) / a;
    let t2 = (b - sq) / a;
    if r0 + t1 * rd >= 0.0 { return t1; }
    if r0 + t2 * rd >= 0.0 { return t2; }
    return 0.0;
}

fn eval_sweep_gradient(center: vec2<f32>, start_angle: f32, end_angle: f32, uv: vec2<f32>) -> f32 {
    let d = uv - center;
    var angle = atan2(-d.y, d.x);
    angle = angle * (180.0 / 3.14159265359);
    if angle < 0.0 { angle += 360.0; }
    let range = end_angle - start_angle;
    if abs(range) < 1e-6 { return 0.0; }
    return (angle - start_angle) / range;
}

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

fn render_color(
    render_coord: vec2<f32>,
    pixels_per_em: vec2<f32>,
    blob_base: u32,
    cmd_count: u32,
) -> vec4<f32> {
    var stack: array<vec4<f32>, 8>;
    stack[0] = vec4<f32>(0.0);
    var sp: i32 = 0;

    let raw_base = blob_base * 2u;
    var raw_cursor: u32 = raw_base;
    let raw_end = raw_base + cmd_count * 2u;

    for (var iter = 0u; iter < 64u && raw_cursor < raw_end; iter++) {
        let cmd = read_raw4(raw_cursor);
        raw_cursor += 4u;

        switch (cmd.x) {
            case 1: {
                sp = min(sp + 1, 7);
                stack[sp] = vec4<f32>(0.0);
            }
            case 2: {
                let sub_offset = u32(cmd.y);
                let draw_color = unpack_color(cmd.z, cmd.w);
                let coverage = render_sub_glyph(render_coord, pixels_per_em, blob_base, sub_offset);
                let premul = vec4<f32>(draw_color.rgb * draw_color.a * coverage, draw_color.a * coverage);
                stack[sp] = composite_colors(premul, stack[sp], 3);
            }
            case 3: {
                let sub_offset = u32(cmd.y);
                let gradient_type = cmd.z;
                let num_stops = cmd.w;

                let inv_mat = read_inv_transform(raw_cursor);
                raw_cursor += 12u;

                let uv = (inv_mat * vec3<f32>(render_coord, 1.0)).xy;

                var grad_t: f32 = 0.0;

                switch (gradient_type) {
                    case 0: {
                        let g0 = read_raw4(raw_cursor);
                        let g1 = read_raw4(raw_cursor + 4u);
                        raw_cursor += 8u;
                        let p0 = vec2<f32>(read_fixed(g0.x, g0.y), read_fixed(g0.z, g0.w));
                        let p1 = vec2<f32>(read_fixed(g1.x, g1.y), read_fixed(g1.z, g1.w));
                        grad_t = eval_linear_gradient(p0, p1, uv);
                    }
                    case 1: {
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
                    case 2: {
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

                let grad_color = evaluate_color_line(raw_cursor, num_stops, grad_t);
                raw_cursor += u32(num_stops) * 4u;

                let coverage = render_sub_glyph(render_coord, pixels_per_em, blob_base, sub_offset);
                let premul = vec4<f32>(grad_color.rgb * grad_color.a * coverage, grad_color.a * coverage);
                stack[sp] = composite_colors(premul, stack[sp], 3);
            }
            case 4: {
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

fn render_color_alpha_dist(
    render_coord: vec2<f32>,
    pixels_per_em: vec2<f32>,
    blob_base: u32,
    cmd_count: u32,
    search_radius: f32,
) -> f32 {
    var min_dist: f32 = 1e6;

    let raw_base = blob_base * 2u;
    var raw_cursor = raw_base;
    let raw_end = raw_base + cmd_count * 2u;

    for (var iter = 0u; iter < 64u && raw_cursor < raw_end; iter++) {
        let cmd = read_raw4(raw_cursor);
        raw_cursor += 4u;

        switch (cmd.x) {
            case CMD_DRAW_SOLID: {
                let sub_offset = u32(cmd.y);
                let sd = render_sub_glyph_sd(render_coord, pixels_per_em, blob_base, sub_offset, search_radius);
                min_dist = min(min_dist, abs(sd.y));
            }
            case CMD_DRAW_GRADIENT: {
                let sub_offset = u32(cmd.y);
                let gradient_type = cmd.z;
                let num_stops = cmd.w;

                raw_cursor += 12u;

                switch (gradient_type) {
                    case 0: { raw_cursor += 8u; }
                    case 1: { raw_cursor += 12u; }
                    case 2: { raw_cursor += 8u; }
                    default: {}
                }

                raw_cursor += u32(num_stops) * 4u;

                let sd = render_sub_glyph_sd(render_coord, pixels_per_em, blob_base, sub_offset, search_radius);
                min_dist = min(min_dist, abs(sd.y));
            }
            default: {}
        }
    }

    return min_dist;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let render_coord = input.texcoord;
    let band_transform = input.banding;
    let glyph_data = input.glyph;

    let ems_per_pixel = max(
        fwidth(render_coord),
        vec2<f32>(1.0 / 65536.0)
    );

    let pixels_per_em = 1.0 / ems_per_pixel;

    let ppem = input.pixels_per_em.x;
    let glyph_base = u32(glyph_data.x);

    var fill_rgb = input.color.rgb;
    var border_rgb = input.border_color.rgb;

    if (params.flags & 2u) != 0u {
        fill_rgb = pow(fill_rgb, vec3<f32>(2.2));
        border_rgb = pow(border_rgb, vec3<f32>(2.2));
    }

    let glyph_screen_size = input.em_size * pixels_per_em;

    // Resizing works, but the borders are inconsisten per glyph. An 'I' and an 'r' get a thinner border than an 'e'.
    //let border_width_physical = input.border_width * (glyph_screen_size.x + glyph_screen_size.y);

    // Consistent borders, but borders resize weirdly based on Zoom
    //let border_width_physical = input.border_width * ppem;

    // PERFECT, every glyph has the same border thickness and resizing/zooming makes the border remain just a PERCENTAGE of the glyph, instead of a hardcoded width!
    let border_width_physical = input.border_width * (pixels_per_em.x + pixels_per_em.y) * 5.0;

    if glyph_data.w != 0 {
        let cmd_count = u32(glyph_data.w);
        let fill = render_color(render_coord, pixels_per_em, glyph_base, cmd_count);
        let fill_alpha = fill.a;

        if border_width_physical <= 0.0 {
            return fill;
        }

        let search_radius = border_width_physical + 0.5;
        let min_dist = render_color_alpha_dist(
            render_coord, pixels_per_em, glyph_base, cmd_count, search_radius
        );

        let signed_dist = select(min_dist, -min_dist, fill_alpha > 0.5);
        let outer_alpha = clamp(border_width_physical + 0.5 - signed_dist, 0.0, 1.0);
        let border_coverage = max(outer_alpha - fill_alpha, 0.0);
        let border_alpha = input.border_color.a * border_coverage;

        return vec4<f32>(
            fill.rgb + border_rgb * border_alpha,
            fill_alpha + border_alpha
        );
    }

    var band_max = glyph_data.yz;
    band_max.y &= 0x00FF;

    var geometric_coverage = render_single(
        render_coord,
        pixels_per_em,
        band_transform,
        glyph_base,
        band_max
    );

    var coverage = geometric_coverage;

    if (params.flags & 1u) != 0u {
        if ppem < 16.0 {
            let d = ems_per_pixel * (1.0 / 3.0);

            let msaa = 0.25 * (
                render_single(
                    render_coord + vec2<f32>(-d.x, -d.y),
                    pixels_per_em,
                    band_transform,
                    glyph_base,
                    band_max
                ) +
                render_single(
                    render_coord + vec2<f32>( d.x, -d.y),
                    pixels_per_em,
                    band_transform,
                    glyph_base,
                    band_max
                ) +
                render_single(
                    render_coord + vec2<f32>(-d.x,  d.y),
                    pixels_per_em,
                    band_transform,
                    glyph_base,
                    band_max
                ) +
                render_single(
                    render_coord + vec2<f32>( d.x,  d.y),
                    pixels_per_em,
                    band_transform,
                    glyph_base,
                    band_max
                )
            );

            coverage = mix(
                coverage,
                msaa,
                smoothstep(16.0, 8.0, ppem)
            );
        }

        if ppem < 48.0 {
            let brightness = dot(
                input.color.rgb,
                vec3<f32>(0.299, 0.587, 0.114)
            );

            coverage = darken(
                coverage,
                brightness,
                ppem
            );
        }
    }

    let fill_alpha = input.color.a * coverage;

    if border_width_physical <= 0.0 {
        return vec4<f32>(fill_rgb * fill_alpha, fill_alpha);
    }

    let search_radius = border_width_physical + 0.5;
    let sd = render_single_sd(
        render_coord, pixels_per_em, band_transform, glyph_base, band_max, search_radius
    );

    let signed_dist = sd.y;
    let outer_coverage = clamp(border_width_physical + 0.5 - signed_dist, 0.0, 1.0);
    let border_coverage = max(outer_coverage - geometric_coverage, 0.0);
    let border_alpha = input.border_color.a * border_coverage;

    let alpha = fill_alpha + border_alpha;
    let rgb = fill_rgb * fill_alpha + border_rgb * border_alpha;

    return vec4<f32>(rgb, alpha);
}