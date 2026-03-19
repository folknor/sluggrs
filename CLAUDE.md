# sluggrs

GPU-based vector text rendering using the Slug algorithm. Replaces atlas-based glyph rendering (bitmap rasterization) with per-pixel bezier curve evaluation in fragment shaders — resolution-independent, no texture atlas needed.

## Project structure

- `slug-font-rendering.md` — Background research on Slug + Loop-Blinn patents
- `slug-glyph-design.md` — Design doc for eventual iced integration (replacing cryoglyph)
- `slug-glyph-investigation.md` — PoC investigation log, bugs found/fixed, remaining issues
- `slug-glyph-proto/` — Working proof-of-concept (standalone wgpu app)
- `repos/` — External dependency repos (gitignored)

## Proto crate (`slug-glyph-proto/`)

Standalone wgpu/winit app that renders text using Slug shaders.

### Key modules
- `main.rs` — wgpu harness, glyph preparation, CPU simulation for debugging
- `outline.rs` — Glyph outline extraction via `skrifa`, cubic→quadratic subdivision
- `band.rs` — Band acceleration structure (spatial index for shader curve lookup)
- `simple_shader.wgsl` — Simplified Slug shader used by the PoC (no dilation)
- `shader.wgsl` — Full Slug shader translated from HLSL reference (with dilation, not currently used)

### Build & run

```sh
cd slug-glyph-proto
RUST_LOG=info cargo run
```

### Known issue

Straight-line-only glyphs (e.g. comma in Inter Variable) render with horizontal stripe artifacts. Root cause: band acceleration structure doesn't ensure both ray directions have coverage for all-linear shapes. The code already detects all-linear glyphs and uses 1 band, but this needs further debugging. See `slug-glyph-investigation.md` for details.

## Tech stack

- Rust (edition 2024)
- wgpu 28, winit 0.30, skrifa 0.40
- WGSL shaders (translated from Slug HLSL reference, MIT licensed)

## Broader context

This is Phase 1 (standalone PoC) of a potential replacement for `cryoglyph` in the iced GUI framework's text rendering pipeline. The eventual goal is a `slug-glyph` crate that provides the same API surface as cryoglyph but uses GPU curve evaluation instead of bitmap atlases. See `slug-glyph-design.md` for the full integration plan.
