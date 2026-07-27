# sluggrs

GPU-based vector text rendering using the Slug algorithm. Drop-in
replacement for cryoglyph in iced's wgpu text rendering pipeline. Evaluates
quadratic bezier curves per-pixel in fragment shaders -
resolution-independent, no texture atlas needed.

## Project structure

### Library (`src/`)
- `lib.rs` - Public API, re-exports, cosmic_text re-exports, GlyphInstance,
  shader constants
- `outline.rs` - Glyph outline extraction via `skrifa`, cubic->quadratic
  subdivision, COLR color emoji
- `prepare.rs` - GPU preparation: line segment perturbation, FAKE_ITALIC shear
- `prep.rs` - Mono glyph prep phase (`prepare_mono`), parallel-ready seam
  paired with `text_atlas::commit_mono`
- `band.rs` - Band acceleration structure (spatial index for shader curve lookup)
- `glyph_cache.rs` - GlyphKey, GlyphEntry, GlyphMap for resolution-independent caching
- `gpu_cache.rs` - Shared GPU state (shader, bind group layouts, pipeline cache)
- `text_atlas.rs` - Curve + band texture management, glyph upload, texture growth
- `text_renderer.rs` - prepare() + render() pipeline matching cryoglyph's interface
- `raster_text.rs` + `raster_text.wgsl` - Raster fallback for non-vector
  glyphs (absorbed from iced)
- `viewport.rs` - Screen resolution uniform buffer
- `types.rs` - Resolution, TextBounds, TextArea, ColorMode, error types
- `simple_shader.wgsl` - Simplified Slug shader (no dilation)
- `shader.wgsl` - Full Slug shader (with dilation, not yet wired up)

### Other
- `examples/demo.rs` - Standalone wgpu/winit demo
- `examples/demo2.rs` - Demo with viewport scroll + MSAA
- `examples/hotpath.rs` - Profiling binary for brokkr
- `examples/email_bench.rs`, `email2_bench.rs`, `gpu_bench.rs` - Benchmark
  targets (email-client scale, mixed-locale, GPU timing)
- `tests/` - Spike tests and unit tests (63 passing, 11 ignored GPU-only)
- `docs/` - Design docs, investigation log, integration spec
- `repos/` - gitignored checkouts of iced, cosmic-text, cryoglyph for reference

## brokkr

All builds, tests, and profiling go through `brokkr`, the shared dev tool -
never raw `cargo` (exception: non-sluggrs projects like iced). Whether a
given session may run brokkr at all is stated per session; when in doubt,
don't - the orchestrator runs the checks.

If brokkr reports a lock (`already locked by PID`), another project is using it.
Wait and retry - the lock exists to prevent concurrent benchmark interference.

### Available in sluggrs
```sh
brokkr check                                  # clippy + tests
brokkr check -- --test glyph_pipeline_test    # run one test file
brokkr check -- -- --ignored                  # run ignored (GPU-only) tests
brokkr hotpath                                # timing profile (1 run, stored in results.db)
brokkr hotpath --hotpath 3                    # 3 timing runs (run count rides on the mode flag)
brokkr hotpath --alloc                        # allocation profile
brokkr hotpath --alloc 5                      # 5 alloc runs
brokkr hotpath --bench                        # uninstrumented build, 3 runs - walls comparable across commits
brokkr hotpath --bench --commit 736e18c       # build + bench an old commit (brokkr-managed worktree)
brokkr hotpath --target email                 # email-client-scale benchmark (8k+ glyphs)
brokkr hotpath --target email2                # mixed-locale inbox (CJK/Arabic/Hindi, 200 messages)
brokkr hotpath --target email --alloc         # email benchmark with allocation tracking
brokkr hotpath -v                             # full build/bench/result output
brokkr visual [snapshot] [--all]              # run visual snapshot tests
brokkr fmt                                    # cargo fmt (args forwarded raw)
brokkr list                                   # list snapshots and approval state
brokkr approve <snapshot>                     # record current output as accepted baseline
brokkr report <run_id>                        # show detailed results for a past run
brokkr visual-status                          # dashboard: all snapshots vs approved baselines
brokkr results                                # last 20 results
brokkr results <uuid>                         # look up by UUID prefix
brokkr results --compare abc1 def2 --mode bench   # compare two commits side-by-side
brokkr results --commit abc1                  # filter by commit prefix
brokkr env                                    # show environment info
brokkr clean                                  # clean build artifacts and scratch data
brokkr history                                # browse command history
```

The `--target` flag is a free-form string. `brokkr hotpath --target foo` builds
`examples/foo_bench.rs` and files results under the command name "foo" (the
default `hotpath` target files under "render"). The measurement mode is a
separate axis: exact values `bench`, `hotpath`, `alloc`. To add a new
benchmark target, create `examples/{name}_bench.rs` and a `[[example]]` entry
in `Cargo.toml`.

## Lints

Cargo.toml has 27 clippy deny-level rules covering style, error handling,
async safety, and no-debug-code. Performance-constraining lints (`cast_*`,
`float_cmp`, `indexing_slicing`) are intentionally excluded - speed at all
costs.

## Tech stack

- Rust (edition 2024, MSRV 1.97)
- cosmic-text 0.19 (shaping, layout, font system)
- skrifa 0.40 (glyph outline extraction)
- wgpu 29 (GPU textures, render pipeline)
- hotpath 0.22 (function-level profiling, brokkr integration)
- WGSL shaders (translated from Slug HLSL reference, MIT licensed)

### Deliberate version pins

`ccu` will report skrifa and wgpu as outdated. Both are pinned on purpose:

- **skrifa tracks cosmic-text, not latest.** cosmic-text 0.19 depends on
  skrifa 0.40 directly *and* on skrifa 0.42 via swash. Our 0.40 pin dedups
  with cosmic-text's copy; bumping to 0.45 would put a third skrifa in every
  downstream build for no gain, since skrifa is internal to sluggrs (not
  re-exported, no types cross the iced boundary). Bump only when cosmic-text
  does.
- **wgpu must match iced.** wgpu types (`Device`, `RenderPass`,
  `TextureFormat`) cross the sluggrs/iced API boundary, so wgpu 30 has to
  wait for upstream iced.

hotpath has no downstream coupling and can be bumped freely.

## General rules

- Don't use gremlins! Em-dash, en-dash, strange quotes, whatever - they're
  all verboten.
