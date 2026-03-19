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

### Most promising next step

The **1-band approach** (effectively disabling the band optimization for all-linear glyphs) should confirm the diagnosis. If the comma renders correctly with 1 band, the fix is to improve the band builder — either:

1. **Detect all-linear glyphs** and use 1 band (simple, already coded, just needs the shader duplicate fixed)
2. **Expand curve assignment** in the band builder — add padding to the x/y range when assigning curves to bands, so curves near band boundaries appear in adjacent bands
3. **Use a smarter coverage formula** — when both weights are near zero, use `max` for the direction that has nonzero weight, and `min` only when both have weight

## Architecture validated

Despite the comma issue, the session validated the core architecture:

- `skrifa` glyph outlines → quadratic beziers: **works**
- Band acceleration structure: **works for curved glyphs**
- WGSL Slug shader (curve evaluation per pixel): **works**
- wgpu instanced rendering pipeline: **works**
- The entire approach of replacing atlas-based text rendering with GPU curve evaluation: **viable**

## Files

```
docs/research/slug-glyph-proto/
├── Cargo.toml
├── fonts/InterVariable.ttf
└── src/
    ├── main.rs           # wgpu harness, glyph preparation, CPU simulation
    ├── outline.rs         # skrifa outline extraction (with debug tracing)
    ├── band.rs            # band acceleration structure builder
    ├── shader.wgsl        # full Slug shader (unused — has dilation)
    └── simple_shader.wgsl # simplified shader used by the PoC

docs/research/
├── slug-font-rendering.md   # background research (Slug + Loop-Blinn patents)
├── slug-glyph-design.md     # design doc for iced integration
└── slug-glyph-investigation.md  # this file
```
