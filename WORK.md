# WORK

GPU shader optimization: perpendicular-range pre-check + ref-texel break
keys for the band curve loops in `src/simple_shader.wgsl`.

## Problem

`render_single()` in `src/simple_shader.wgsl` casts a horizontal and a
vertical ray per sample. Each ray iterates its band's curve list
(sorted for direction-aware early exit along the ray axis). A curve is
included in a horizontal band when its bbox intersects the band's y
slab, but the sample's exact y may still fall outside the curve's own
y-range - such curves produce root code 0 and contribute nothing, yet
each costs a curve_ref texel read plus two curve texel reads plus the
root-code computation. At ppem < 16 with MSAA enabled, `render_single`
runs 5 times per fragment, multiplying the waste.

Each curve-ref texel carries 3 unused i16 lanes (written as zeros in
the four writer loops near the end of `build_bands` in `src/band.rs`).
Filling them costs no band-data growth.

## Agreed plan (implement exactly this)

### `src/band.rs`

1. Private quantization helper matching the curve packing pipeline
   EXACTLY (wrap, not saturate):

       fn quantize_i16(v: f32) -> i16 { ((v * 4.0).round() as i32) as i16 }

   Curve texels are produced by `(v * 4.0).round() as i32` (prep.rs
   `quantize`, text_atlas.rs upload_color_v1 `q`) followed by `as i16`
   at pack time, which wraps. A direct f32-to-i16 cast saturates and
   would diverge for out-of-range COLRv1 coordinates. Bounds must be
   min/max over the three POST-CAST i16 values per axis (wrapping
   destroys monotonicity, so quantizing f32 extrema is wrong).

2. `CurveMeta` gains four i16 fields: `min_x_q`, `max_x_q`, `min_y_q`,
   `max_y_q`. Phase 1 quantizes the six control-point coordinates once
   per curve and takes per-axis min/max of the post-cast values.
   Continuation curves share p1 with the previous p3 by value, so
   per-curve bounds from the outline points match the deduplicated
   stored texels.

3. The existing f32 keys stay unchanged for band membership, sorting,
   and `find_split`. Only the ref-texel payload changes.

4. Fill the three zero lanes in all four writer loops:

   | Reference list        | `.y`    | `.z`    | `.w`    |
   |-----------------------|---------|---------|---------|
   | Horizontal descending | min_y_q | max_y_q | max_x_q |
   | Horizontal ascending  | min_y_q | max_y_q | min_x_q |
   | Vertical descending   | min_x_q | max_x_q | max_y_q |
   | Vertical ascending    | min_x_q | max_x_q | min_y_q |

   `.y`/`.z` = perpendicular range (skip test). `.w` = the
   direction-specific break key: a left/bottom ray selects the
   ascending list and breaks on the minimum ray-axis coordinate; a
   right/top ray selects the descending list and breaks on the maximum.
   `.x` (offset) and all header lanes remain unchanged.

5. Update layout comments to document the ref lanes and the
   list-dependent `.w` meaning.

### `src/simple_shader.wgsl`

Only `render_single`'s two loops change. Root solving, coverage
accumulation, `calc_root_code`, COLRv1 functions, and `src/shader.wgsl`
are untouched. IMPORTANT: `tests/solver_regression_test.rs` string-
patches the two `let a = q12.xy - q12.zw * 2.0 + q3;` blocks and
asserts exactly 2 occurrences - those lines must survive verbatim.

1. Compute `let render_coord_q = render_coord * 4.0;` once near the top
   of `render_single`.

2. In each loop, immediately after `read_texel` of `curve_ref` and
   BEFORE decoding `.x` or reading the two curve texels:

   a. Direction-aware early break from `.w` (bit-identical to the old
      break: min/max commutes with the affine transform, so
      `f32(w) * INV_UNITS - render_coord.axis` equals the old
      min/max over p values). Horizontal loop:

          let wq = (f32(curve_ref.w) * INV_UNITS - render_coord.x) * pixels_per_em.x;
          left ray:  if wq > 0.5  { break; }   // w = min_x_q (asc list)
          right ray: if wq < -0.5 { break; }   // w = max_x_q (desc list)

      Vertical loop mirrors with `.y` axis and `pixels_per_em.y`.

   b. Perpendicular range skip (continue, NEVER break - the list is
      not sorted on the perpendicular axis). Horizontal loop:

          if f32(curve_ref.y) > render_coord_q.y || f32(curve_ref.z) < render_coord_q.y { continue; }

      Vertical loop uses `render_coord_q.x`. Strict comparisons: the
      equality cases are retained (a coordinate exactly at the sample
      counts as nonnegative and can form a mixed sign pattern).

   The break must come first: a perpendicularly irrelevant curve can
   also be the terminating curve, and continuing past it would disable
   early termination for the rest of the list.

3. Remove the old early-break block that ran after the `p12`/`p3`
   computation (now represented exactly by `.w`).

### Exactness invariants (why this is output-identical)

- Root code is nonzero only for mixed sign patterns of
  `p_i = q_i * 0.25 - coord` (all-nonnegative and all-negative both
  yield 0 via the 0x2E74 table). `min_q > 4*coord` implies all p
  strictly positive; `max_q < 4*coord` implies all strictly negative;
  both are safe skips. i16-to-f32 conversion is exact; `* 4.0` and
  `* 0.25` are exact binary scalings.
- No perpendicular half-pixel margin: the half-pixel in the break is
  along the RAY axis (roots farther than 0.5 px have zero clamped
  coverage). Perpendicular eligibility is a root-existence question at
  the exact sample coordinate; MSAA samples are offset before the call.
- Bounds computed from post-cast i16 values keep the skip consistent
  with whatever the stored curve texels contain, including pre-existing
  COLRv1 wrap-around behavior (do NOT fix that here; band membership
  still uses unwrapped f32 - out of scope).

## Tests

In the `band.rs` unit-test module:

- `curve_refs_encode_quantized_bounds_and_break_keys`: known
  non-quarter-integer coordinates, 1 horizontal + 1 vertical band;
  assert the complete 4-lane contents of all four refs (h-desc, h-asc,
  v-desc, v-asc) per the table.
- A continuation-curve case (curve i's p1 == curve i-1's p3):
  identify refs by decoded offset; assert the continuation's bounds
  include the shared point.
- `curve_ref_bounds_match_wrapped_curve_texels`: coordinates just past
  the i16 boundary; assert bounds equal the post-wrap values (guards
  against a saturating float-to-i16 cast).
- Keep every existing entry-length assertion unchanged (proves no band
  data growth).

GPU regression (new ignored test alongside solver_regression_test.rs,
reusing its pattern: string-patch SIMPLE_SHADER_WGSL with an asserted
occurrence count, append test-only fragment entry points that call
`render_single` at exact witness coordinates, render 1x1, read back):

- Baseline variant = shader with the new perpendicular-continue lines
  patched out (assert the patch matches exactly 2 occurrences).
- Witnesses: horizontal and vertical rays; a sample outside a curve's
  perpendicular range; a sample exactly at a curve's max bound where
  equality must still produce a root.
- Assert optimized and baseline coverage agree within 1/255.

## Constraints

- Do not run cargo or brokkr; the orchestrator runs all builds, tests,
  snapshots, and benchmarks.
- Rust edition 2024. Performance is the top priority.
- No public API changes. No non-ASCII characters in code or comments.
- Only `src/band.rs`, `src/simple_shader.wgsl`, and tests change.

## Acceptance (run by the orchestrator)

- brokkr check + ignored GPU tests pass.
- brokkr visual --all: all snapshots 0.0% divergence.
- band_texels KV unchanged (no band data growth).
- gpu_text_render_us same or better on the render target; build_bands
  shows no material CPU regression.

## Benchmark verdict (plantasjen, same-host A/B vs 4914a28)

- Identical storage: buffer 22787 texels / 182296 bytes on both
  commits - the three spare curve-ref lanes carry the new data at
  zero size cost.
- gpu_text_render_us 7 vs 7: unchanged on the 92-glyph headless
  scene, which is already at the measurement floor. The skip targets
  small-ppem MSAA (5x render_single) and dense glyphs; this scene
  exercises neither.
- Walls inside the +/-4% noise band: render -1.5%, email2 +0.2%.
- build_bands 298 -> 324 us total across 91 glyphs (+0.28 us/glyph):
  the expected cost of six per-curve quantizations, 0.3% of cold
  prepare.

## Implementation summary

Shipped per the agreed plan; the deep review of the diff found no
production defects and three test-side findings, all fixed:

- Shared GPU harness moved to tests/common/mod.rs; the solver test no
  longer runs twice under --ignored (17 -> 16 ignored).
- The continuation unit test now builds through prepare_mono and
  validates ref bounds against the actual deduplicated curve texels it
  addresses (offsets 10/11 sharing a texel).
- The equality witnesses were redesigned empirically: a throwaway
  64-sample sweep across the equality lines showed the original
  witnesses sat where the coverage combiner masks the tested axis
  (baseline == mutated). The decisive samples live in the half-pixel
  AA falloff band just past the curve endpoint: (3.125, 2.0) reads
  96/255 via the horizontal ray vs 0 when wrongly skipped;
  (3.0, 1.625) reads 223/255 via the vertical ray vs 0. Each is inert
  under the opposite axis's mutation. The sweep also confirmed
  optimized == baseline at all 256 sampled points.
- Sweep discovery, recorded in the test comment: only the MAX-bound
  equality (`curve_ref.z < coord`) is semantically observable. At
  exact MIN equality all perpendicular deltas are >= 0, which cannot
  form a mixed sign pattern in calc_root_code, so the strict `>` on
  `curve_ref.y` is defensive and untestable by construction.

Validation: 82 tests + 16 GPU-gated tests pass; all four visual
snapshots 0.0%. Band data layout unchanged (all length assertions
intact); the three spare curve-ref lanes now carry perpendicular
bounds + break key at zero size cost.

## Fixed review findings (for the record)

The implementation shipped; a review round found three test-side
weaknesses. These supersede the corresponding parts of the Tests
section above.

### 1. Shared GPU test harness (kills the duplicated test run)

`tests/band_ref_regression_test.rs` currently includes
`tests/solver_regression_test.rs` via `#[path] mod`, which compiles the
solver's `#[test]` into a second crate so it runs twice under
--ignored, and forced unrelated helpers to become pub(crate).

Fix: create `tests/common/mod.rs` (a subdirectory module is not
compiled as its own test crate) holding the shared harness: the Params
struct, `create_test_device()`, and the generalized
`render_coverage(device, queue, entry_point, source, glyph_data)`
including the pipeline/vertex-layout/readback plumbing. Both test files
declare `mod common;` and keep their own outlines, shader_source
patchers, witnesses, and `#[test]` functions at top level. Remove the
`#[path]` include and revert the solver file's items to private where
possible. Add `#![allow(dead_code)]` at the top of common/mod.rs if
per-crate usage differs.

### 2. Axis-isolated equality witnesses + mutation variants

The current `fs_band_h_max`/`fs_band_v_max` both sample (3.0, 2.0),
where the h and v contributions mask each other: a one-axis strictness
regression (`>` -> `>=` or `<` -> `<=`) still passes. Rework
`band_ref_regression_test.rs`:

- shader_source takes a variant enum: `Optimized` (unpatched),
  `NoPrecheck` (both continue lines removed), `HNonStrict` (horizontal
  continue line's `>` replaced with `>=` and `<` with `<=`),
  `VNonStrict` (same for the vertical line). Count the horizontal and
  vertical patch patterns SEPARATELY, asserting exactly 1 occurrence
  each (not a summed 2).
- Same synthetic curve as now: p1 (-2,-2), p2 (0,0), p3 (3,2),
  quantized (-8,-8) (0,0) (12,8), upem 5, 1x1 bands,
  pixels_per_em (1,1).
- Witness entry points (all call render_single directly like today):
  - `fs_band_h_outside` at (3.0, 3.0): y strictly above max_y.
  - `fs_band_v_outside` at (4.0, 2.0): x strictly beyond max_x.
  - `fs_band_h_eq` at (2.75, 2.0): y4 = 8 == max_y_q (equality on the
    horizontal loop's perpendicular test); x4 = 11 strictly inside
    (-8, 12) so VNonStrict cannot affect it. The curve's equality root
    is the endpoint (3, 2) at t = 1; expected coverage is ~0.082 when
    retained and ~0.332 when wrongly skipped.
  - `fs_band_v_eq` at (3.0, 1.9375): x4 = 12 == max_x_q (equality on
    the vertical loop's test); y4 = 7.75 strictly inside (-8, 8) so
    HNonStrict cannot affect it. Expected ~0.031 retained, ~0.406
    wrongly skipped.
- Assertions (tolerance 1/255 for equalities):
  - Every witness: Optimized == NoPrecheck within 1/255.
  - fs_band_h_eq: NoPrecheck >= 10/255; |HNonStrict - NoPrecheck| >
    8/255; |VNonStrict - NoPrecheck| <= 1/255.
  - fs_band_v_eq: NoPrecheck >= 4/255; |VNonStrict - NoPrecheck| >
    8/255; |HNonStrict - NoPrecheck| <= 1/255.
  The expected-value figures above are hand-computed guidance; the
  assertions use the bounds, not the exact figures. Do not weaken the
  bounds; if a bound seems wrong, say so in your report instead of
  adjusting it.

### 3. Continuation test must exercise the real deduplicated layout

`continuation_curve_refs_include_shared_point_in_bounds` in
src/band.rs passes `sequential_locations(2)` (offsets 0, 2), but real
continuation dedup produces offsets 0, 1 (the previous p3 texel is the
continuation's p1/p2 texel). Replace the test with one that builds the
payload through `crate::prep::prepare_mono` (same crate, no cycle:
this is a unit test) on a two-curve continuation outline, then:

- decode the band refs from `PreparedMono::blob_data` (layout:
  [band texels][curve texels], 2 i32 per texel, i16 pairs packed
  low/high; ref offsets are biased by the band element count),
- for each ref, read the addressed curve texel pair, reconstruct the
  three control points (texel0 = p1,p2; texel1.xy = p3),
- assert the ref's .y/.z lanes equal the min/max of those stored
  values on the correct axis, and .w equals the min or max on the ray
  axis per list membership,
- covering both the plain curve and the continuation curve (which
  must share a texel with its predecessor - assert the two refs
  address overlapping texels to prove dedup happened).

## Constraints (unchanged)

- Do not run cargo or brokkr; the orchestrator runs all builds/tests.
- No non-ASCII characters. Only test files change:
  tests/band_ref_regression_test.rs, tests/solver_regression_test.rs,
  new tests/common/mod.rs, and the band.rs unit-test module (tests
  only, not production code).
