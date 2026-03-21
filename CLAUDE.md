# sluggrs

GPU-based vector text rendering using the Slug algorithm. Drop-in replacement for cryoglyph in iced's wgpu text rendering pipeline. Evaluates quadratic bezier curves per-pixel in fragment shaders — resolution-independent, no texture atlas needed.

## Project structure

### Library (`src/`)
- `lib.rs` — Public API, re-exports, cosmic_text re-exports, shader constants
- `outline.rs` — Glyph outline extraction via `skrifa`, cubic→quadratic subdivision
- `prepare.rs` — GPU preparation: line segment perturbation, FAKE_ITALIC shear
- `band.rs` — Band acceleration structure (spatial index for shader curve lookup)
- `glyph_cache.rs` — GlyphKey, GlyphEntry, GlyphMap for resolution-independent caching
- `gpu_cache.rs` — Shared GPU state (shader, bind group layouts, pipeline cache)
- `text_atlas.rs` — Curve + band texture management, glyph upload, texture growth
- `text_renderer.rs` — prepare() + render() pipeline matching cryoglyph's interface
- `viewport.rs` — Screen resolution uniform buffer
- `types.rs` — Resolution, TextBounds, TextArea, ColorMode, error types
- `simple_shader.wgsl` — Simplified Slug shader (no dilation)
- `shader.wgsl` — Full Slug shader (with dilation, not yet wired up)

### Other
- `examples/demo.rs` — Standalone wgpu/winit demo
- `examples/hotpath.rs` — Profiling binary for brokkr (`brokkr sluggrs hotpath`)
- `tests/` — Spike tests and unit tests (62 passing, 8 ignored GPU-only)
- `docs/` — Design docs, investigation log, integration spec
- `repos/` — gitignored checkouts of iced, cosmic-text, cryoglyph for reference

## Bash rules
- Never use sed, find, awk, or complex bash commands
- Never chain commands with &&
- Never chain commands with ;
- Never pipe commands with |
- Never read or write from /tmp. All data lives in the project.
- Never run raw cargo, curl, pkill. Use `brokkr`. Exception: non-sluggrs projects (iced, etc.).

## brokkr commands

### Available in sluggrs
```sh
brokkr check                                  # clippy + tests
brokkr check -- --test glyph_pipeline_test    # run one test file
brokkr check -- -- --ignored                  # run ignored (GPU-only) tests
brokkr sluggrs hotpath                        # timing profile, stored in results.db
brokkr sluggrs hotpath --alloc                # allocation profile
brokkr sluggrs hotpath --runs 5               # best-of-5
brokkr sluggrs hotpath --commit abc123        # benchmark an old commit via worktree
brokkr sluggrs hotpath --force                # run on dirty tree (results not stored)
brokkr results                                # last 20 results
brokkr results <uuid>                         # look up by UUID prefix
brokkr results --compare-last --command hotpath  # compare two most recent hotpath runs
brokkr results --commit abc1                  # filter by commit prefix
brokkr env                                    # show environment info
brokkr clean                                  # clean build artifacts and scratch data
brokkr history                                # browse command history
```

### Not yet available in sluggrs
- `brokkr run` — no main binary (sluggrs is a library)
- `brokkr bench` — no benchmark suite configured
- `brokkr sluggrs test/list/approve/status` — visual snapshot testing (no snapshots defined yet)

## Profiling

Five functions are instrumented with `#[hotpath::measure]`:
- `extract_outline()`, `prepare_outline()`, `build_bands()`, `upload_glyph()`, `prepare_with_depth()`

`.brokkr/results.db` is committed to git — always commit it after profiling runs so performance data is tracked alongside the code.

The hotpath example emits KV pairs to stderr (captured by brokkr):
`distinct_glyphs`, `curve_texels`, `band_texels`, `cold_prepare_us`,
`warm_prepare_avg_us`, `mixed_prepare_avg_us`, `curve_texture_bytes`,
`band_texture_bytes`.

## Lints

Cargo.toml has 27 clippy deny-level rules covering style, error handling, async safety, and no-debug-code. Performance-constraining lints (`cast_*`, `float_cmp`, `indexing_slicing`) are intentionally excluded — speed at all costs.

## iced integration

The `repos/iced/` checkout (branch `sluggrs` on `folknor/iced`) has `text.rs` swapped from cryoglyph to sluggrs. To test in ratatoskr, point its iced dependency at the fork.

## Tech stack

- Rust (edition 2024, MSRV 1.92)
- cosmic-text 0.18 (shaping, layout, font system)
- skrifa 0.40 (glyph outline extraction)
- wgpu 28 (GPU textures, render pipeline)
- hotpath 0.14 (function-level profiling, brokkr integration)
- WGSL shaders (translated from Slug HLSL reference, MIT licensed)
