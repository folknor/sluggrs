# Slug PoC Investigation Summary

## What works

The proof-of-concept successfully renders text using GPU-evaluated bezier curves via the Slug algorithm:

- **Outline extraction** from Inter Variable via `skrifa` works correctly — `quad_to`, `line_to`, `close` callbacks produce the right curve data
- **Band acceleration structure** builds correctly for all glyph types
- **WGSL shader translation** of the Slug algorithm is functionally correct — all glyphs render correctly including all-linear shapes
- **wgpu pipeline** (textures, bind groups, instanced rendering) works end-to-end
- **Multiple text lines** at different sizes render correctly

## Bugs found and fixed

1. **`band_max` vs `band_count`** — Shader expects max band *index* (0-based), we were passing the count. Off-by-one shifted vertical band header lookups. Fixed by passing `band_count - 1`.

2. **Missing `close()` segments** — The outline pen wasn't emitting a closing line segment back to the contour start. Glyphs like 'l' (rectangles) were missing one edge, breaking winding number. Fixed by tracking `contour_start` and emitting a line_to on close.

3. **Offset reset across `prepare_text` calls** — Second text line's curve/band offsets started at 0, overlapping with first line's data in the shared textures. Fixed by passing `base_curve_offset` and `base_band_offset`.

4. **Band boundary over-assignment** — The band builder used `ceil()` for the max bound, causing curves whose extent lands exactly on a band boundary to spill into the adjacent band. Fixed with an epsilon-biased exclusive upper bound: `(max - 1e-5).floor()`.

5. **Degenerate quadratic solver artifacts (the comma bug)** — Line segments encoded as degenerate quadratics (p2 at midpoint of p1–p3) caused horizontal stripe artifacts. The shader's quadratic solver falls through to a linear path when `a ≈ 0`, but for horizontal/vertical edges both `a` and `b` are near zero, producing `0.5 / 0.0 = inf` and garbage coverage values.

   **Root cause**: The Slug solver assumes all input curves are genuine quadratics with nonzero second-degree coefficients. Degenerate quadratics violate this assumption.

   **Fix**: Perturb line segments on the CPU side during GPU preparation (`prepare_outline`). The control point p2 is offset by a tiny amount along the edge normal, turning the degenerate quadratic into a genuine (but visually imperceptible) curve. The shader's quadratic solver then always has a nonzero `a` coefficient and works correctly.

   This fix is applied in `src/prepare.rs`, keeping `src/outline.rs` exact for non-rendering use cases. Several shader-side approaches were attempted first (degenerate edge guards, bias functions, dedicated linear solvers) but all either failed to fully fix the issue or broke other glyphs.

6. **Band texture width mismatch** — The shaders hardcode a 4096-wide band texture with row wrapping (`calc_band_loc`), but the demo was uploading a single row sized to content. Fixed by padding to `BAND_TEXTURE_WIDTH` and computing correct row count.

## Intentional shader divergences from Slug reference

Our WGSL shaders are translated from the Slug HLSL reference
(github.com/EricLengyel/Slug, MIT). As of March 2026 the reference
has had no functional changes since our translation — only comment
additions and license header updates. Two intentional divergences
exist:

### 1. Linearity threshold: 0.25 vs 1/65536

The reference solver uses `abs(a) < 1.0 / 65536.0` to detect
near-linear curves and fall back to a linear solver. Our shader
uses `abs(a) < 0.25`.

**Why**: Our CPU-side perturbation in `prepare.rs::perturb_midpoint()`
produces curves with `|a|` values of 0.02–0.2. The reference threshold
(~0.00002) doesn't catch these, so they enter the quadratic solver with
a near-zero denominator, producing unstable roots — visible as unfilled
regions in glyphs with short horizontal features (e.g. bold 'r' arm in
Inter Variable). See commit b7bf831.

The threshold was originally set to 0.5, but review found that genuine
quadratics with modest curvature (`|a|` around 0.3) could enter the
linear fallback path where `0.5 / b.y` with small `b.y` produces
garbage roots. Lowered to 0.25 which still catches all perturbed line
segments (`|a| ≤ 0.2`) with margin, while keeping real quadratics on
the quadratic path.

**Coupling**: The shader threshold and `perturb_midpoint()` eps range
(currently `(len * 1e-5).clamp(0.01, 0.1)`) are coupled. If the
perturbation magnitude changes, the shader threshold must track it.
The reference doesn't need this because Lengyel's pipeline presumably
doesn't perturb line segments.

### 2. CalcRootCode: select() vs bitcast sign extraction

The reference uses `asuint(y) >> 31U` (bitcast to uint, extract sign
bit) which reduces to a single `LOP3` instruction on NVIDIA. Our
translation uses `select(0u, 1u, y < 0.0)` which generates conditional
moves.

**These are NOT equivalent for negative zero.** `bitcast<u32>(-0.0) >> 31`
returns 1 (sign bit set), but `select(0u, 1u, -0.0 < 0.0)` returns 0
(per IEEE 754, -0.0 is not less than 0.0). On Intel Arc, intermediate
shader calculations can produce -0.0 where NVIDIA produces +0.0. The
bitcast version then misclassifies these as negative, producing wrong
root eligibility codes and visible rendering artifacts (unfilled regions).

We use select() for correctness across all GPUs. The performance
difference is negligible (verified: 13µs/frame on RTX 3080 either way).

## Architecture

The crate separates concerns into three stages:

1. **Outline extraction** (`outline.rs`) — True font geometry via skrifa. `GlyphOutline` has exact control points and bounds.
2. **GPU preparation** (`prepare.rs`) — Transforms outlines for the shader solver. `GpuOutline` has perturbed line segments and recomputed bounds.
3. **Band building** (`band.rs`) — Spatial acceleration structure operating on GPU-prepared geometry.

## Files

```
src/
├── lib.rs              # Re-exports, shader constants
├── outline.rs          # Font geometry extraction (exact)
├── prepare.rs          # GPU preparation (perturbed for solver)
├── band.rs             # Band acceleration structure
├── simple_shader.wgsl  # Simplified Slug shader (no dilation)
└── shader.wgsl         # Full Slug shader (with dilation, not yet wired up)
examples/
└── demo.rs             # Standalone wgpu/winit demo (cargo run --example demo)
```
