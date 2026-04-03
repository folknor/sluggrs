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

If brokkr reports a lock (`already locked by PID`), another project is using it.
Wait and retry — the lock exists to prevent concurrent benchmark interference.

### Available in sluggrs
```sh
brokkr check                                  # clippy + tests
brokkr check -- --test glyph_pipeline_test    # run one test file
brokkr check -- -- --ignored                  # run ignored (GPU-only) tests
brokkr hotpath                                # timing profile (1 run, stored in results.db)
brokkr hotpath -n 3                           # 3 runs
brokkr hotpath --alloc                        # allocation profile
brokkr hotpath --alloc -n 5                   # 5 alloc runs
brokkr hotpath --target email                 # email-client-scale benchmark (8k+ glyphs)
brokkr hotpath --target email --alloc         # email benchmark with allocation tracking
brokkr test [snapshot] [--all]                # run visual snapshot tests
brokkr list                                   # list snapshots and approval state
brokkr approve <snapshot>                     # record current output as accepted baseline
brokkr report <run_id>                        # show detailed results for a past run
brokkr visual-status                          # dashboard: all snapshots vs approved baselines
brokkr results                                # last 20 results
brokkr results <uuid>                         # look up by UUID prefix
brokkr results --compare-last --command hotpath  # compare two most recent hotpath runs
brokkr results --commit abc1                  # filter by commit prefix
brokkr env                                    # show environment info
brokkr clean                                  # clean build artifacts and scratch data
brokkr history                                # browse command history
```

## Profiling

Five functions are instrumented with `#[hotpath::measure]`:
- `extract_outline()`, `prepare_outline()`, `build_bands()`, `upload_glyph()`, `prepare_with_depth()`

`.brokkr/results.db` is committed to git — always commit it after profiling runs so performance data is tracked alongside the code. Brokkr requires a clean git tree to store results, but allows a dirty `results.db` or markdown file changes — so you don't need to commit CLAUDE.md edits before running profiling.

The hotpath example emits KV pairs to stderr (captured by brokkr):
`distinct_glyphs`, `curve_texels`, `band_texels`, `cold_prepare_us`,
`warm_prepare_avg_us`, `mixed_prepare_avg_us`, `curve_texture_bytes`,
`band_texture_bytes`, `gpu_text_render_us`.

### GPU profiling

Both CPU and GPU profiling run headless — no user interaction needed.

- `brokkr sluggrs hotpath` measures CPU-side prepare AND GPU fragment shader
  time via wgpu-profiler timestamp queries. Renders to an offscreen 1920x1080
  texture. The `gpu_text_render_us` KV is stored in results.db.
- `cargo run --example demo` also emits `gpu_text_render_ms` to stderr on
  each redraw (5 warmup frames at startup flush the profiler pipeline). A
  window opens but GPU timing is captured automatically:
  `timeout 3 cargo run --example demo 2>&1 | grep gpu_`
- Requires TIMESTAMP_QUERY + TIMESTAMP_QUERY_INSIDE_PASSES wgpu features
  (NVIDIA, AMD, Intel desktop all support these). Gracefully disabled if
  unavailable.

Baseline (RTX 3080, 92 glyphs): CPU prepare 753us, GPU render 11us
(headless offscreen) / 71us (windowed with compositor).

## Lints

Cargo.toml has 27 clippy deny-level rules covering style, error handling, async safety, and no-debug-code. Performance-constraining lints (`cast_*`, `float_cmp`, `indexing_slicing`) are intentionally excluded — speed at all costs.

## iced integration

The `repos/iced/` checkout (branch `sluggrs` on `folknor/iced`) has `text.rs` swapped from cryoglyph to sluggrs. To test in ratatoskr, point its iced dependency at the fork.

## Code review

`.review.toml` defines 7 reviewer archetypes with persistent Claude + Codex sessions:
- **sweep** = bugs, perf, slug, gpu (every change)
- **design** = fonts, wgpu, arch (structural changes)
- **everything** = all 7

Usage: `echo "review prompt" | review <archetype|group|all>`

`research/` (gitignored) contains reference repos and docs for reviewer context:
Slug reference shaders, fontations, cosmic-text, vello, mesa, wgpu-naga, gpuweb spec,
Vulkan spec, NVIDIA shader guides, JCGT paper, and other Slug implementations.

When adding TODO items from review findings, tag them with `**<archetype> review**`
to track which reviewer caught what.

## Tech stack

- Rust (edition 2024, MSRV 1.92)
- cosmic-text 0.18 (shaping, layout, font system)
- skrifa 0.40 (glyph outline extraction)
- wgpu 28 (GPU textures, render pipeline)
- hotpath 0.14 (function-level profiling, brokkr integration)
- WGSL shaders (translated from Slug HLSL reference, MIT licensed)
