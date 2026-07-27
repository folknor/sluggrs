# WORK

Fix a reachable f32 cancellation bug in the Slug solvers of
`src/simple_shader.wgsl` by computing the polynomial coefficients `a` and
`b` from unshifted curve coordinates, matching the harfbuzz reference
(`research/harfbuzz/src/hb-gpu-fragment.wgsl:184-262`).

## Problem

`solve_horiz_poly` / `solve_vert_poly` compute `a = p12.xy - p12.zw * 2.0
+ p3` and `b = p12.xy - p12.zw` from coordinates already shifted by
`render_coord`. `a` and `b` are translation-invariant; computed from the
unshifted (stored) coordinates they are exact, because decoded
coordinates are quarter-integers (i16 * 0.25) and every intermediate
stays well inside f32's exact range. Computed from shifted coordinates,
rounding breaks the cancellation: a curve whose active-axis coordinate is
exactly linear (q1 - 2*q2 + q3 == 0; hundreds exist in the bundled fonts)
can produce a tiny nonzero `a`, skipping the exact `a == 0.0` linear
branch and computing a catastrophically wrong root (e.g. t = 2.0 instead
of 0.744 for q = -1000, 0, 1000 at render_coord 488.004), yielding
missing or spurious coverage.

## Agreed plan (implement exactly this)

All in `src/simple_shader.wgsl`:

1. Change both solver signatures to the harfbuzz shape:

   fn solve_horiz_poly(a: vec2<f32>, b: vec2<f32>, p1: vec2<f32>) -> vec2<f32>
   fn solve_vert_poly(a: vec2<f32>, b: vec2<f32>, p1: vec2<f32>) -> vec2<f32>

   Remove the internal a,b computation; replace `p12.x/.y` uses in the
   discriminant, linear branch, and returned polynomial with `p1.x/.y`.
   Preserve the exact `a.y == 0.0` / `a.x == 0.0` tests - no epsilon.

2. In both loops of `render_single`, decode unshifted coordinates first
   and derive the shifted ones from them:

   let q12 = vec4<f32>(raw12) * INV_UNITS;
   let q3 = vec2<f32>(raw3.xy) * INV_UNITS;
   let p12 = q12 - vec4<f32>(render_coord, render_coord);
   let p3 = q3 - render_coord;

3. Inside the `if code != 0u` blocks, compute

   let a = q12.xy - q12.zw * 2.0 + q3;
   let b = q12.xy - q12.zw;

   and call the solver as `solve_horiz_poly(a, b, p12.xy)` (resp
   `solve_vert_poly(a, b, p12.xy)`).

4. Leave untouched: the early-exit tests and `calc_root_code` (both
   correctly use shifted coordinates), the direction-aware coverage
   accumulation, the MSAA loop, and everything in the COLRv1 interpreter
   (`render_sub_glyph` inherits the fix through `render_single`).

5. `src/prepare.rs`: fix the stale top comment. Lines use `p2 = p1`
   encoding to avoid midpoint-degenerate coefficients; the exact-zero
   branch handles exactly-linear real quadratics, not line segments in
   general.

Out of scope: the vertex-shader dilation (evaluated and rejected this
loop), `src/shader.wgsl` (dead code), any solver threshold or epsilon.

## Implementation summary

Implemented by the build session exactly as planned; both diff reviews
(direct + resumed deep session) found no correctness issues. brokkr check
passes including the 13 GPU tests; all four approved visual snapshots
pass at 0.0% pixel diff, confirming non-regression (the failure needs a
specific subpixel alignment none of the scenes happen to hit).

The zero-dilation proposal was rejected with math (recorded under "Not
worth pursuing" in TODO.md): AA support outside the boundary is half a
pixel in screen space at every ppem.

Deferred: a constructed-atlas regression test for the cancellation
witness (recorded in TODO.md under Polish; needs a raw-blob + readback
harness that tests/ lacks).

## Benchmark verdict (plantasjen, same-host A/B vs parent 99276d7)

- gpu_text_render_us: 7 -> 7. Unchanged, as expected (same arithmetic
  count; the change is exactness and reference alignment).
- email2: 14.769 -> 14.717 ms (-0.4%), noise.
- render wall: 31.726 -> 29.280 ms (-7.7%), but the delta sits entirely
  in CPU-side prepare KVs (cold 7466 -> 7189 us, warm 879 -> 798, mixed
  268 -> 216) which a fragment-shader change cannot affect: host-state
  variance, not a real effect. A null A/B on identical code earlier the
  same day showed +/-1.5% on this wall.

## Constraints

- Do not run cargo or brokkr; the orchestrator runs all builds, tests,
  snapshots, and benchmarks.
- WGSL under naga/wgpu 29. Rust edition 2024.
- No non-ASCII characters in code or comments.
- Keep the diff minimal: this is a surgical change to two files.
