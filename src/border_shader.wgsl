// Border-only entry points. This file is appended to the unchanged normal
// shader source, so the shared unpacking and quadratic root helpers above are
// compiled from one source fragment by both pipelines.

struct BorderParams {
    color: vec4<f32>,
    width_px: f32,
    _pad0: f32,
    _pad1: vec2<f32>,
}

@group(2) @binding(0) var<uniform> border: BorderParams;

struct BorderVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) texcoord: vec2<f32>,
    @location(1) @interpolate(flat) descriptor: u32,
    @location(2) @interpolate(flat) pixels_per_em: f32,
}

@vertex
fn vs_border(instance: GlyphInstance, @builtin(vertex_index) vid: u32) -> BorderVertexOutput {
    var output: BorderVertexOutput;
    let corner = vec2<f32>(f32(vid & 1u), f32((vid >> 1u) & 1u));
    let normal = corner * 2.0 - 1.0;
    let descriptor_raw = instance.glyph.x * 2u;
    let fill_offset = u32(atlas[descriptor_raw]);
    let fill_raw = fill_offset * 2u;
    let em_rect = vec4<f32>(
        bitcast<f32>(atlas[fill_raw]), bitcast<f32>(atlas[fill_raw + 1u]),
        bitcast<f32>(atlas[fill_raw + 2u]), bitcast<f32>(atlas[fill_raw + 3u]),
    );
    let dilation = border.width_px + 0.5;
    let base_pos = instance.screen_rect.xy + corner * instance.screen_rect.zw;
    let screen_pos = base_pos + params.scroll_offset + normal * dilation;
    let ndc = vec2<f32>(
        screen_pos.x / params.screen_size.x * 2.0 - 1.0,
        -(screen_pos.y / params.screen_size.y * 2.0 - 1.0),
    );
    output.position = vec4<f32>(ndc, instance.depth_ppem.x, 1.0);
    let base_uv = vec2<f32>(
        mix(em_rect.x, em_rect.z, corner.x),
        mix(em_rect.w, em_rect.y, corner.y),
    );
    let em_size = vec2<f32>(em_rect.z - em_rect.x, em_rect.w - em_rect.y);
    let safe_screen_size = select(instance.screen_rect.zw, vec2<f32>(1e-20), abs(instance.screen_rect.zw) < vec2<f32>(1e-20));
    let ems_per_pixel = em_size / safe_screen_size;
    output.texcoord = base_uv
        + vec2<f32>(normal.x, -normal.y) * ems_per_pixel * dilation;
    output.descriptor = instance.glyph.x;
    output.pixels_per_em = instance.depth_ppem.y;
    return output;
}

fn border_curve(base: u32, index: u32) -> mat3x2<f32> {
    let raw = (base + index * 3u) * 2u;
    let p1 = vec2<f32>(bitcast<f32>(atlas[raw]), bitcast<f32>(atlas[raw + 1u]));
    let p2 = vec2<f32>(bitcast<f32>(atlas[raw + 2u]), bitcast<f32>(atlas[raw + 3u]));
    let p3 = vec2<f32>(bitcast<f32>(atlas[raw + 4u]), bitcast<f32>(atlas[raw + 5u]));
    return mat3x2<f32>(p1, p2, p3);
}

fn winding_curve(base: u32, index: u32) -> mat3x2<f32> {
    let first = read_texel(base + index * 2u);
    let last = read_texel(base + index * 2u + 1u);
    return mat3x2<f32>(
        vec2<f32>(first.xy) * INV_UNITS,
        vec2<f32>(first.zw) * INV_UNITS,
        vec2<f32>(last.xy) * INV_UNITS,
    );
}

fn segment_distance(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let d = b - a;
    let dd = dot(d, d);
    if dd <= 1e-12 { return length(p - a); }
    return length(p - (a + d * clamp(dot(p - a, d) / dd, 0.0, 1.0)));
}

fn quadratic_distance(p: vec2<f32>, curve: mat3x2<f32>, pixels_per_unit: f32) -> f32 {
    let p1 = curve[0];
    let p2 = curve[1];
    let p3 = curve[2];
    let curvature = length(p1 - p2 * 2.0 + p3) * pixels_per_unit;
    if curvature <= 0.125 {
        return segment_distance(p, p1, p3);
    }
    var best = 1e30;
    var previous = p1;
    for (var i = 1u; i <= 16u; i++) {
        let t = f32(i) / 16.0;
        let q = mix(mix(p1, p2, t), mix(p2, p3, t), t);
        best = min(best, segment_distance(p, previous, q));
        previous = q;
    }
    return best;
}

fn border_winding(p: vec2<f32>, band_base: u32, bounds: vec4<f32>, band_count: u32) -> i32 {
    var winding = 0;
    let height = max(bounds.w - bounds.y, 1e-6);
    let band = u32(clamp(floor((p.y - bounds.y) * f32(band_count) / height), 0.0, f32(band_count - 1u)));
    let header = read_texel(band_base + band);
    let left = p.x < f32(header.w) * INV_UNITS;
    let list = decode_offset(select(header.y, header.z, left));
    for (var i = 0u; i < u32(header.x); i++) {
        let reference = read_texel(band_base + list + i);
        let curve_offset = decode_offset(reference.x);
        let first = read_texel(band_base + curve_offset);
        let last = read_texel(band_base + curve_offset + 1u);
        let p0 = vec2<f32>(first.xy) * INV_UNITS;
        let p1 = vec2<f32>(first.zw) * INV_UNITS;
        let p2 = vec2<f32>(last.xy) * INV_UNITS;
        if all(p1 == p0) || all(p1 == p2) {
            if p0.y == p2.y { continue; }
            let upward = p0.y <= p.y && p.y < p2.y;
            let downward = p2.y <= p.y && p.y < p0.y;
            let t = (p.y - p0.y) / (p2.y - p0.y);
            let x = mix(p0.x, p2.x, t);
            if x > p.x {
                if upward { winding += 1; }
                if downward { winding -= 1; }
            }
            continue;
        }
        let a = p0.y - 2.0 * p1.y + p2.y;
        let b = 2.0 * (p1.y - p0.y);
        let c = p0.y - p.y;
        var roots = vec2<f32>(-1.0);
        if abs(a) < 1e-8 {
            if abs(b) >= 1e-8 { roots.x = -c / b; }
        } else {
            let discriminant = b * b - 4.0 * a * c;
            if discriminant >= 0.0 {
                let root = sqrt(discriminant);
                roots = vec2<f32>((-b - root) / (2.0 * a), (-b + root) / (2.0 * a));
            }
        }
        for (var ri = 0u; ri < 2u; ri++) {
            let t = roots[ri];
            if t < 0.0 || t > 1.0 { continue; }
            let q = mix(mix(p0, p1, t), mix(p1, p2, t), t);
            if q.x <= p.x { continue; }
            let direction = 2.0 * a * t + b;
            if direction > 0.0 && t == 1.0 { continue; }
            if direction < 0.0 && t == 0.0 { continue; }
            if direction > 0.0 { winding += 1; }
            if direction < 0.0 { winding -= 1; }
        }
    }
    return winding;
}

@fragment
fn fs_border(input: BorderVertexOutput) -> @location(0) vec4<f32> {
    let raw = input.descriptor * 2u;
    let winding_offset = u32(atlas[raw + 1u]);
    let grid_offset = u32(atlas[raw + 2u]);
    let boundary_offset = u32(atlas[raw + 3u]);
    let boundary_count = u32(atlas[raw + 4u]);
    let bounds = vec4<f32>(
        bitcast<f32>(atlas[raw + 8u]), bitcast<f32>(atlas[raw + 9u]),
        bitcast<f32>(atlas[raw + 10u]), bitcast<f32>(atlas[raw + 11u]),
    );
    let columns = u32(atlas[raw + 12u]);
    let rows = u32(atlas[raw + 13u]);
    let curve_count = u32(atlas[raw + 15u]);
    let units_per_em = bitcast<f32>(atlas[raw + 6u]);
    let pixels_per_unit = input.pixels_per_em / units_per_em;
    let band_count = clamp(curve_count, 1u, 16u);
    let winding_base = input.descriptor + winding_offset;
    // Corrected band data precedes the packed winding curves. Its size is the
    // distance from winding_offset to grid_offset minus two texels per curve.
    let inside = border_winding(input.texcoord, winding_base, bounds, band_count) != 0;

    var distance = 1e30;
    let outside_grid = input.texcoord.x < bounds.x || input.texcoord.y < bounds.y
        || input.texcoord.x > bounds.z || input.texcoord.y > bounds.w;
    let brute_force = atlas[raw + 7u] != 0 || outside_grid || columns == 0u || rows == 0u;
    let boundary_base = input.descriptor + boundary_offset;
    if brute_force {
        for (var i = 0u; i < boundary_count; i++) {
            distance = min(distance, quadratic_distance(input.texcoord, border_curve(boundary_base, i), pixels_per_unit));
        }
    } else {
        let cell_size = (bounds.zw - bounds.xy) / vec2<f32>(f32(columns), f32(rows));
        let cell = clamp(vec2<u32>((input.texcoord - bounds.xy) / cell_size), vec2<u32>(0u), vec2<u32>(columns - 1u, rows - 1u));
        let cell_index = cell.y * columns + cell.x;
        let offsets_base = input.descriptor + grid_offset;
        let first = u32(atlas[(offsets_base + cell_index) * 2u]);
        let last = u32(atlas[(offsets_base + cell_index + 1u) * 2u]);
        let candidates_base = offsets_base + columns * rows + 1u;
        for (var i = first; i < last; i++) {
            let piece = u32(atlas[(candidates_base + i) * 2u]);
            distance = min(distance, quadratic_distance(input.texcoord, border_curve(boundary_base, piece), pixels_per_unit));
        }
    }
    let signed_distance_px = select(distance, -distance, inside) * pixels_per_unit;
    let coverage = clamp(border.width_px + 0.5 - signed_distance_px, 0.0, 1.0);
    let alpha = border.color.a * coverage;
    let rgb = select(border.color.rgb, pow(border.color.rgb, vec3<f32>(2.2)), (params.flags & 2u) != 0u);
    return vec4<f32>(rgb * alpha, alpha);
}
