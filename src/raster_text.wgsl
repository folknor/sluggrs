// Textured quad shader for raster glyph fallback (emoji, bitmap fonts).
// Shares the Params uniform layout with sluggrs (group 0).

struct Params {
    screen_size: vec2<f32>,
    scroll_offset: vec2<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(1) @binding(0) var atlas_tex: texture_2d<f32>;
@group(1) @binding(1) var atlas_sampler: sampler;

struct Instance {
    @location(0) screen_pos: vec2<f32>,
    @location(1) screen_size: vec2<f32>,
    @location(2) atlas_pos: vec2<f32>,
    @location(3) atlas_size: vec2<f32>,
    @location(4) color: vec4<f32>,
    @location(5) depth: f32,
}

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
}

@vertex
fn vs_main(inst: Instance, @builtin(vertex_index) vid: u32) -> VsOut {
    let corner = vec2<f32>(f32(vid & 1u), f32((vid >> 1u) & 1u));
    let screen = inst.screen_pos + corner * inst.screen_size + params.scroll_offset;
    let ndc = vec2<f32>(
        screen.x / params.screen_size.x * 2.0 - 1.0,
        -(screen.y / params.screen_size.y * 2.0 - 1.0),
    );

    var out: VsOut;
    out.pos = vec4<f32>(ndc, inst.depth, 1.0);
    out.uv = inst.atlas_pos + corner * inst.atlas_size;
    out.color = inst.color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let tex = textureSample(atlas_tex, atlas_sampler, in.uv);
    // tex is premultiplied alpha (color emoji) or expanded mask (white * alpha).
    // Multiply by vertex color for tinting (mask glyphs) or pass-through (color glyphs).
    return tex * in.color;
}
