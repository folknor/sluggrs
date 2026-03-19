# sluggrs

GPU-based vector text rendering using the Slug algorithm. Replaces atlas-based glyph rendering (bitmap rasterization) with per-pixel bezier curve evaluation in fragment shaders — resolution-independent, no texture atlas needed.

## Project structure

- `src/lib.rs` — Library root, re-exports modules and shader source constants
- `src/outline.rs` — Glyph outline extraction via `skrifa`, cubic→quadratic subdivision
- `src/band.rs` — Band acceleration structure (spatial index for shader curve lookup)
- `src/simple_shader.wgsl` — Simplified Slug shader (no dilation)
- `src/shader.wgsl` — Full Slug shader translated from HLSL reference (with dilation, not currently used)
- `examples/demo.rs` — Standalone wgpu/winit demo app
- `docs/` — Design docs and investigation logs

## Build & run

```sh
RUST_LOG=info cargo run --example demo
```

## Tech stack

- Rust (edition 2024, MSRV 1.92)
- skrifa 0.40 (glyph outline extraction)
- wgpu 28, winit 0.30 (demo only)
- WGSL shaders (translated from Slug HLSL reference, MIT licensed)

## Broader context

This crate is a potential replacement for `cryoglyph` in the iced GUI framework's text rendering pipeline. See `docs/slug-glyph-design.md` for the full integration plan.
