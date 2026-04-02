// Simplified Slug shader for the proof-of-concept.
// Uses a simple 2D orthographic projection (no dilation).
// The fragment shader is the full Slug curve evaluator.

const K_LOG_BAND_TEXTURE_WIDTH: u32 = 12u;
const K_BAND_TEXTURE_WIDTH: u32 = 4096u;

struct Params {
    screen_size: vec2<f32>,
    scroll_offset: vec2<f32>,
    flags: u32,       // bit 0: enable MSAA+stem darkening
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
const INV_UNITS: f32 = 0.25; // 1.0 / 4.0 units_per_em
@group(1) @binding(0) var curve_texture: texture_2d<i32>;
@group(1) @binding(1) var band_texture: texture_2d<i32>;

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

fn calc_band_loc(glyph_loc: vec2<i32>, offset: u32) -> vec2<i32> {
    var band_loc = vec2<i32>(glyph_loc.x + i32(offset), glyph_loc.y);
    band_loc.y += band_loc.x >> K_LOG_BAND_TEXTURE_WIDTH;
    band_loc.x &= i32(K_BAND_TEXTURE_WIDTH) - 1;
    return band_loc;
}

/// Evaluate Slug coverage at a single sample point.
fn render_single(
    render_coord: vec2<f32>,
    pixels_per_em: vec2<f32>,
    band_transform: vec4<f32>,
    glyph_loc: vec2<i32>,
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

    let hband_header_loc = calc_band_loc(glyph_loc, u32(band_index.y));
    let hband_data = textureLoad(band_texture, hband_header_loc, 0);
    let h_split = f32(hband_data.w) * INV_UNITS;
    let h_left_ray = render_coord.x < h_split;
    let h_data_offset = u32(select(hband_data.y, hband_data.z, h_left_ray));

    for (var ci = 0u; ci < u32(hband_data.x); ci++) {
        let curve_ref_loc = calc_band_loc(glyph_loc, h_data_offset + ci);
        let curve_ref = textureLoad(band_texture, curve_ref_loc, 0).xy;
        let curve_loc = vec2<i32>(curve_ref);
        let raw12 = textureLoad(curve_texture, curve_loc, 0);
        let p12 = vec4<f32>(raw12) * INV_UNITS - vec4<f32>(render_coord, render_coord);
        let raw3 = textureLoad(curve_texture, vec2<i32>(curve_loc.x + 1, curve_loc.y), 0);
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

    let vband_header_loc = calc_band_loc(glyph_loc, u32(band_max.y + 1 + band_index.x));
    let vband_data = textureLoad(band_texture, vband_header_loc, 0);
    let v_split = f32(vband_data.w) * INV_UNITS;
    let v_left_ray = render_coord.y < v_split;
    let v_data_offset = u32(select(vband_data.y, vband_data.z, v_left_ray));

    for (var ci = 0u; ci < u32(vband_data.x); ci++) {
        let curve_ref_loc = calc_band_loc(glyph_loc, v_data_offset + ci);
        let curve_ref = textureLoad(band_texture, curve_ref_loc, 0).xy;
        let curve_loc = vec2<i32>(curve_ref);
        let raw12 = textureLoad(curve_texture, curve_loc, 0);
        let p12 = vec4<f32>(raw12) * INV_UNITS - vec4<f32>(render_coord, render_coord);
        let raw3 = textureLoad(curve_texture, vec2<i32>(curve_loc.x + 1, curve_loc.y), 0);
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

    var band_max = glyph_data.zw;
    band_max.y &= 0x00FF;
    let glyph_loc = glyph_data.xy;

    // Single-sample coverage
    var coverage = render_single(render_coord, pixels_per_em, band_transform, glyph_loc, band_max);

    if (params.flags & 1u) != 0u {
        // 4x MSAA for small sizes: blend in gradually from 16ppem down to 8ppem
        if ppem < 16.0 {
            let d = ems_per_pixel * (1.0 / 3.0);
            let msaa = 0.25 * (
                render_single(render_coord + vec2<f32>(-d.x, -d.y), pixels_per_em, band_transform, glyph_loc, band_max) +
                render_single(render_coord + vec2<f32>( d.x, -d.y), pixels_per_em, band_transform, glyph_loc, band_max) +
                render_single(render_coord + vec2<f32>(-d.x,  d.y), pixels_per_em, band_transform, glyph_loc, band_max) +
                render_single(render_coord + vec2<f32>( d.x,  d.y), pixels_per_em, band_transform, glyph_loc, band_max)
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

    // Premultiplied alpha output
    let alpha = input.color.a * coverage;
    return vec4<f32>(input.color.rgb * alpha, alpha);
}
