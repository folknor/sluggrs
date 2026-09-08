// Separable Gaussian blur and shadow composite for filtered decorations.
//
// The blur runs on a SCALAR LINEAR coverage mask, never on colour. Every
// pixel of one decoration shares a colour, so tinting after the convolution
// is algebraically identical to convolving premultiplied colour, and it
// avoids both the extra storage and the chance of blurring sRGB values.
// Coverage is linear geometry: it is never gamma-converted.

struct BlurParams {
    // Texel size of the source texture, for tap spacing.
    texel: vec2<f32>,
    // Tap direction: (1,0) horizontal, (0,1) vertical.
    direction: vec2<f32>,
    sigma: f32,
    // Kernel half-width in texels. Must be the same ceil(4 * sigma) the CPU
    // used to size the mask and to cull contributors, or the shadow is
    // clipped somewhere in the chain.
    support: i32,
    _pad: vec2<f32>,
}

@group(0) @binding(0) var<uniform> blur: BlurParams;
@group(0) @binding(1) var mask_texture: texture_2d<f32>;
@group(0) @binding(2) var mask_sampler: sampler;

struct BlurVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_blur(@builtin(vertex_index) vid: u32) -> BlurVertexOutput {
    var output: BlurVertexOutput;
    let corner = vec2<f32>(f32(vid & 1u), f32((vid >> 1u) & 1u));
    output.uv = corner;
    output.position = vec4<f32>(corner * 2.0 - 1.0, 0.0, 1.0);
    // The quad is built in UV space where y grows downward, so flip for NDC.
    output.position.y = -output.position.y;
    return output;
}

@fragment
fn fs_blur(input: BlurVertexOutput) -> @location(0) f32 {
    // Weights are computed rather than looked up: the support is dynamic and
    // a table would need one entry per sigma.
    let inv_two_sigma_sq = 1.0 / (2.0 * blur.sigma * blur.sigma);
    var total = 0.0;
    var weight_sum = 0.0;
    for (var i = -blur.support; i <= blur.support; i++) {
        let offset = f32(i);
        let weight = exp(-offset * offset * inv_two_sigma_sq);
        let uv = input.uv + blur.direction * blur.texel * offset;
        // ZERO extension, not clamp-to-edge. Repeating the edge texel would
        // duplicate any coverage that reaches the mask boundary once per
        // out-of-bounds tap and brighten the shadow into a stripe. The source
        // rectangle is padded by the full support, but a destination clipped
        // by the area bounds can still put a contributing glyph against that
        // boundary, so the edge is not guaranteed empty.
        let inside = all(uv >= vec2<f32>(0.0)) && all(uv <= vec2<f32>(1.0));
        let sample = select(
            0.0,
            textureSampleLevel(mask_texture, mask_sampler, uv, 0.0).r,
            inside,
        );
        total += sample * weight;
        // The full truncated kernel weight stays in the denominator: the mask
        // is zero-extended, so an out-of-bounds tap contributes zero to the
        // numerator and must not renormalise the rest upward.
        weight_sum += weight;
    }
    // Normalising by the actual sum rather than an analytic constant keeps
    // the result unbiased at the truncation boundary.
    return total / max(weight_sum, 1e-8);
}

