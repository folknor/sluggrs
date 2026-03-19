# Slug PoC Investigation Summary (2026-03-18)

## What works

The proof-of-concept successfully renders text using GPU-evaluated bezier curves via the Slug algorithm:

- **Outline extraction** from Inter Variable via `skrifa` works correctly — `quad_to`, `line_to`, `close` callbacks produce the right curve data
- **Band acceleration structure** builds correctly — CPU simulation confirms the shader would read the right data
- **WGSL shader translation** of the Slug algorithm is functionally correct — all curved glyphs (letters, period, !) render perfectly
- **wgpu pipeline** (textures, bind groups, instanced rendering) works end-to-end
- **Multiple text lines** at different sizes render correctly (after fixing the offset chaining between `prepare_text` calls)

## Bugs found and fixed during the session

1. **`band_max` vs `band_count`** — Shader expects max band *index* (0-based), we were passing the count. Off-by-one shifted vertical band header lookups. Fixed by passing `band_count - 1`.

2. **Missing `close()` segments** — The outline pen wasn't emitting a closing line segment back to the contour start. Glyphs like 'l' (rectangles) were missing one edge, breaking winding number. Fixed by tracking `contour_start` and emitting a line_to on close.

3. **Offset reset across `prepare_text` calls** — Second text line's curve/band offsets started at 0, overlapping with first line's data in the shared textures. Fixed by passing `base_curve_offset` and `base_band_offset`.

## Remaining issue: straight-line-only glyphs (the comma)

### Symptoms

Inter Variable's comma glyph is a parallelogram made of 4 straight lines (`line_to` only, no `quad_to`). It renders with horizontal stripe artifacts — alternating filled and empty scanlines inside the shape. All curved glyphs render perfectly.

### Root cause identified

The Slug coverage formula uses two perpendicular ray casts (horizontal and vertical) and combines them:

```
fallback = min(abs(xcov), abs(ycov))
```

This requires BOTH directions to detect the pixel as "inside." The **band acceleration structure** groups curves by spatial region. For straight-line shapes, some vertical bands don't contain curves needed for the vertical ray to detect "inside" — the curve's x-range simply doesn't overlap that band. This causes `ycov = 0` for pixels that ARE inside the shape, and `min(1, 0) = 0`.

### Attempted fixes

| Approach | Result |
|----------|--------|
| NaN-safe solver (if/else instead of compute-then-overwrite) | No change — NaN wasn't the issue |
| Simplified `calc_root_code` (avoid bitcast tricks) | No change — sign extraction was correct |
| `max(abs(xcov), abs(ycov))` when weights ≈ 0 | Comma fills correctly, but false fills leak outside the shape to the bounding box |
| 1 band per direction (disable banding) | Untested (shader compile error from leftover duplicate, session ended) |

### Revised diagnosis (2026-03-19, external review)

The original theory — "the comma needs 1 band" — is **invalidated**. The all-linear detection in `main.rs:156-170` is already setting `band_count=1` for the comma. The 1-band path executes, and the comma still stripes. This means the bug is **not in band splitting** but downstream in the fragment shader.

The new primary suspects:

1. **The degenerate-quadratic solver paths** — `simple_shader.wgsl:93` (`solve_horiz_poly`) and `simple_shader.wgsl:119` (`solve_vert_poly`). The comma's curves are degenerate quadratics (control point at midpoint of endpoints, so `a = p1 - 2*p2 + p3 ≈ 0`). The `abs(a) < 1/65536` threshold gates whether the linear or quadratic path runs. If floating point puts `a` just above the threshold, it falls into the quadratic path and computes `1.0 / a` with a near-zero denominator — producing garbage roots. Alternatively, the linear path itself may have issues when `b` is near-zero in the solving component (e.g. a near-horizontal or near-vertical edge).

2. **The coverage combine logic** — `simple_shader.wgsl:228-230`. Even if both ray casts produce correct individual coverage, the `min(abs(xcov), abs(ycov))` fallback zeros out coverage when one direction happens to produce zero for geometric reasons unrelated to inside/outside.

3. **`calc_root_code`** — `simple_shader.wgsl:76-83`. For degenerate quadratics where all three y-values (or x-values) are very close, sign classification could be unstable. A control point at exactly the midpoint means `p2` is always between `p1` and `p3`, so its sign should agree with one of them — but floating point edge cases could produce wrong root counts.

### Revised plan

#### Step 1: Verify band data (CPU, already instrumented)

Debug dumps have been added to `main.rs` (extended comma/period debug block). Run and check:

- `h0.count == 4` and `v0.count == 4` for the comma
- Both lists reference curve indices 0,1,2,3

If either count is less than 4, the bug is still in `band.rs` despite `band_count=1`. If both are 4, the builder is confirmed correct and the problem is purely in the shader.

#### Step 2: Isolate failing ray direction (shader, already instrumented)

The shader return has been changed to `vec4(abs(xcov), abs(ycov), 0.0, 1.0)`. Run and interpret:

| Red channel (xcov) | Green channel (ycov) | Diagnosis |
|---------------------|----------------------|-----------|
| Solid | Striped | Vertical ray solver is failing |
| Striped | Solid | Horizontal ray solver is failing |
| Both striped | Both striped | Degenerate-quadratic solver itself is broken |
| Both solid | Both solid | Combine logic (`min`/`combined`) is the problem |

**Bonus probe**: swap to `vec4(xwgt, ywgt, 0.0, 1.0)` to check whether coverage values are nonzero but weights are zero (which would also zero out the `combined` path).

#### Step 3: Fix the identified component

Based on step 2 results:

**If a solver is failing** (one channel striped): The `1/65536` threshold in `solve_horiz_poly` / `solve_vert_poly` is likely too tight for degenerate quadratics. Try:
- Raising the threshold (e.g. `1/256` or `1/64`)
- Or restructuring: compute both linear and quadratic solutions, then select based on `abs(a)` magnitude (blend or hard switch with a safer threshold)
- Check whether `b` can also be near-zero for certain edge orientations, which would make the linear path (`0.5 / b`) equally unstable

**If both solvers are fine but combine is wrong**: The `min(abs(xcov), abs(ycov))` fallback is fundamentally problematic for shapes where one ray direction has legitimate zero coverage at certain pixels. Consider:
- Using `max` when one weight is zero: `if xwgt < epsilon { abs(ycov) } else if ywgt < epsilon { abs(xcov) } else { original formula }`
- Or weight-only combination: skip `fallback` entirely and rely on `combined` when weights are available

**If `calc_root_code` is unstable**: For degenerate quadratics, the sign bits of near-zero values are unreliable. Could add an epsilon-band: treat values within `±epsilon` as zero and handle those curves specially.

#### Step 4: Validate fix against more all-linear glyphs

The comma is one case. Other glyphs to test:
- Pipe `|`, underscore `_`, box-drawing characters, dash `—`
- Any glyph that is a pure polygon with no true curves
- Glyphs with mixed linear and curved segments (should still work after fix)

#### Secondary issue: band texture width assumption

`simple_shader.wgsl:137` (`calc_band_loc`) assumes a 4096-wide wrapped texture, but the PoC uploads a 1-row texture sized to content (`main.rs:382`). This works now because data fits in one row but is fragile. Not the current bug, but should be fixed before scaling up — either pad the texture to 4096 width or remove the wrapping logic.

## Architecture validated

Despite the comma issue, the session validated the core architecture:

- `skrifa` glyph outlines → quadratic beziers: **works**
- Band acceleration structure: **works for curved glyphs**
- WGSL Slug shader (curve evaluation per pixel): **works**
- wgpu instanced rendering pipeline: **works**
- The entire approach of replacing atlas-based text rendering with GPU curve evaluation: **viable**

## Files

```
slug-glyph-proto/
├── Cargo.toml
├── fonts/InterVariable.ttf
└── src/
    ├── main.rs           # wgpu harness, glyph preparation, CPU simulation
    ├── outline.rs         # skrifa outline extraction (with debug tracing)
    ├── band.rs            # band acceleration structure builder
    ├── shader.wgsl        # full Slug shader (unused — has dilation)
    └── simple_shader.wgsl # simplified shader used by the PoC (currently has debug output)
```
