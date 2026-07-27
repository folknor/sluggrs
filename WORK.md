# WORK

Implement a GPU regression test pinning the solver cancellation fix from
commit 17bb626 in `src/simple_shader.wgsl`. Double-sided: the shipped
shader must produce the analytically expected coverage at an exact f32
witness coordinate, and a deliberately re-broken shader variant must
fail the same predicate.

## Agreed plan (implement exactly this)

New file `tests/solver_regression_test.rs`, one test:

    #[test]
    #[ignore = "Requires GPU or software renderer (wgpu adapter)"]
    fn solver_cancellation_regression()

Device creation like tests/atlas_lifecycle_test.rs. No TextAtlas or
TextRenderer: build a standalone pipeline modeled on examples/demo.rs
(~:1070), using the public `sluggrs::GlyphInstance` 48-byte ABI and its
4-attribute vertex layout, Params uniform at group 0 (screen_size =
[1.0, 1.0], zero scroll, flags = 0), read-only storage buffer at
group 1.

### Geometry (horizontal-solver case)

Synthetic `sluggrs::outline::GlyphOutline`, contour order:

1. quadratic (0,-1000) ctrl (500,0) to (0,1000)   <- witness curve,
   real midpoint control point, exactly-linear y: -1000 - 2*0 + 1000 = 0
2. line (0,1000) to (-1000,1000)      (p2 = p1 encoding)
3. line (-1000,1000) to (-1000,-1000) (p2 = p1)
4. line (-1000,-1000) to (0,-1000)    (p2 = p1)

bounds = [-1000, -1000, 500, 1000].
`prepare_mono(&outline, 1, 1, 2000.0, &mut scratch)`.

Vertical-solver case: transpose every point (swap x/y everywhere),
bounds = [-1000, -1000, 1000, 500].

### Witness sample (exact f32 literals)

Horizontal: render_coord = (187.9630126953125, 488.003997802734375),
pixels_per_em = (0.1, 0.1). On the witness curve y(t) = -1000 + 2000t,
t = 0.744002..., x(t) = 1000 t (1-t) = 190.4630127...; the sample is
2.5 decoded units = 0.25 px inside, so correct coverage = 0.75 exactly
(combined term: |0.75 * 0.5| / 0.5; fallback term: min(0.75, 1)).
Vertical: swap the two coordinates.

### Test-only fragment entry points

Append two extra fragment entry points to `sluggrs::SIMPLE_SHADER_WGSL`
at runtime (string concatenation), one per axis, each calling the real
`render_single` with the literal witness coordinates:

    let coverage = render_single(
        vec2<f32>(187.9630126953125, 488.003997802734375),
        vec2<f32>(0.1, 0.1),
        input.banding,
        u32(input.glyph.x),
        input.glyph.yz,
    );
    return vec4<f32>(coverage, coverage, coverage, coverage);

Production `vs_main` is used unchanged, so header decode, payload-base
forwarding, band lookup, curve decode, root classification, and coverage
combination are all under test; only interpolation is bypassed.

### Buffers

Prepend the 5-texel header to `PreparedMono::blob_data` (local encoder
mirroring examples/demo.rs:66): 4x bounds f32::to_bits as i32, 4x
band_transform to_bits, pack_i16_pair(band_count_x-1, band_count_y-1),
reserved 0. Instance: screen_rect [0,0,1,1], color [1;4],
glyph_offset 0, cmd_texel_count 0, depth 0, ppem 200.

### Broken shader variant

Patch the source string: replace both occurrences of

    let a = q12.xy - q12.zw * 2.0 + q3;
    let b = q12.xy - q12.zw;

with the pre-fix shifted forms

    let a = p12.xy - p12.zw * 2.0 + p3;
    let b = p12.xy - p12.zw;

Assert (hard test assertion) that the original block occurs exactly
twice before patching - a future shader refactor must fail loudly here
rather than silently defusing the sensitivity check.

### Render + readback + assertions

1x1 Bgra8UnormSrgb target, RENDER_ATTACHMENT | COPY_SRC, transparent
clear, 256-byte-padded readback (pattern from examples/snapshot.rs).
Read alpha (byte 3, unaffected by sRGB). For each axis and each shader
(fixed, broken), render and read coverage = alpha / 255.

Mandatory assertions:
- fixed: |coverage - 0.75| <= 2/255, both axes;
- broken: NOT(|coverage - 0.75| <= 2/255), both axes.

Do not assert a specific broken value (0 vs saturated is empirical);
include the measured fixed and broken values in every assertion message
so a failure is self-diagnosing.

## Constraints

- Do not run cargo or brokkr; the orchestrator runs all builds and
  tests.
- Rust edition 2024, wgpu 29. Match the existing GPU test conventions
  (#[ignore] reason string, device creation, no unwrap-lint allows
  beyond what tests/ already uses).
- No non-ASCII characters in code or comments.

## Implementation summary

Implemented as tests/solver_regression_test.rs per plan; both diff
reviews (direct + resumed deep session) found no defects. Empirically
double-sided on this host: fixed shader hits 0.75 within 2/255 on both
solver axes, the string-patched broken variant fails the predicate on
both (measured coverage 0), and the occurrence assertion fails closed if
the solver blocks are ever refactored. Full suite: 74 tests + 14 GPU
tests pass. Explanatory comments on the witness geometry and constants
added per review polish. No shader or library changes - test only, so no
visual or benchmark run needed this loop.
