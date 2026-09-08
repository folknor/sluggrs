// Composite of a finished blurred shadow.
//
// A separate module from the blur itself, not merely a separate entry point:
// both need a uniform, a texture and a sampler at group 0, and two different
// globals cannot share one binding index within a module.

struct ShadowParams {
    color: vec4<f32>,
    // Destination rect in physical pixels: origin then size.
    rect: vec4<f32>,
    screen_size: vec2<f32>,
    // Bit 1 matches the main shader's Params.flags: Web colour mode.
    flags: u32,
    _pad: f32,
}

@group(0) @binding(0) var<uniform> shadow: ShadowParams;
@group(0) @binding(1) var shadow_texture: texture_2d<f32>;
@group(0) @binding(2) var shadow_sampler: sampler;

struct ShadowVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_shadow(@builtin(vertex_index) vid: u32) -> ShadowVertexOutput {
    var output: ShadowVertexOutput;
    let corner = vec2<f32>(f32(vid & 1u), f32((vid >> 1u) & 1u));
    output.uv = corner;
    let screen_pos = shadow.rect.xy + corner * shadow.rect.zw;
    output.position = vec4<f32>(
        screen_pos.x / shadow.screen_size.x * 2.0 - 1.0,
        -(screen_pos.y / shadow.screen_size.y * 2.0 - 1.0),
        0.0,
        1.0,
    );
    return output;
}

@fragment
fn fs_shadow(input: ShadowVertexOutput) -> @location(0) vec4<f32> {
    let coverage = textureSampleLevel(shadow_texture, shadow_sampler, input.uv, 0.0).r;
    // Colour conversion happens HERE, on the tint, matching the fill and
    // border shaders. The blurred coverage stays linear.
    let web = (shadow.flags & 2u) != 0u;
    let rgb = select(shadow.color.rgb, pow(shadow.color.rgb, vec3<f32>(2.2)), web);
    let alpha = shadow.color.a * coverage;
    return vec4<f32>(rgb * alpha, alpha);
}
