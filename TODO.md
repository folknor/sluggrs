# TODO

## Done

- [x] Centralize shared shader constants (BAND_TEXTURE_WIDTH exported from lib.rs)
- [x] Update investigation doc with comma bug resolution
- [x] Update design doc to reflect sluggrs naming
- [x] Build the crate API surface (Cache, TextAtlas, TextRenderer, Viewport)
- [x] Wire into iced's `iced_wgpu/src/text.rs` as a cryoglyph replacement
- [x] FAKE_ITALIC shear transform support
- [x] Visual testing in ratatoskr — text renders correctly
- [x] Dilation — 1px quad expansion for smooth AA at edges
- [x] Variable font support (face index + weight axis variation)
- [x] Multi-row curve texture (avoids exceeding device texture limits)
- [x] Band offset 2D conversion (linear → wrapped texture coordinates)
- [x] Solver threshold raised to handle near-degenerate perturbed curves
- [x] Clippy lint hardening (27 deny-level rules)
- [x] Inline unit tests (62 passing across prepare, band, outline, types)
- [x] Hotpath instrumentation + brokkr integration (timing + alloc profiling)
- [x] KV metric emissions (glyph counts, texture usage, cold/warm timings)

## Before shipping

- [ ] Non-vector glyph fallback — color emoji and bitmap-only fonts currently
  produce no output. Required before the iced swap ships.

  **Dual-render-all spike failed** (repos/iced branch): rendering all text
  through both sluggrs and cryoglyph produces 1-2px vertical offset between
  the two renderers at small sizes (different rounding strategies for em-space
  vs pixel-snapped positioning). Not shippable.

  **Next approach: selective routing.** Requires glyph-level or run-segment-level
  partitioning so cryoglyph only renders non-vector glyphs. The hard part is
  constructing fallback-only draw inputs without disturbing text.rs's batching
  model — preserving layout positions, glyph advances, and run boundaries while
  routing only specific glyphs to cryoglyph. Likely needs a classification API
  from sluggrs and a two-pass approach in text.rs. See spike plan in
  .plans/effervescent-chasing-metcalfe.md for full context.
- [x] Trim/eviction — per-frame usage tracking via glyphs_in_use HashSet.
  Pressure-based reset when <50% of cached glyphs are in use (threshold 256).
  Matches cryoglyph frame-boundary semantics.
- [ ] Depth plumbing — prepare_with_depth accepts the callback but discards
  the depth value. Any iced path relying on layered text depth will render
  incorrectly. Either implement or document as unsupported.

## Optimization — CPU allocation reduction

Baseline: 92 glyphs, cold_prepare 1.9ms, warm_prepare 0.7ms, 9.7 MB total alloc.

### Done

- [x] Eliminate band_texels + add atlas scratch buffers — scratch_curve_texels and
  scratch_curve_locations on TextAtlas, band_texels eliminated via bytemuck::cast_slice.
  Result: upload_glyph alloc -37.5%, timing -23.8%.

- [x] Persist units_per_em_cache on TextRenderer — HashMap moved from local to field.
  Result: prepare_with_depth alloc -33.3%.

- [x] Cheap capacity fixes in build_bands — inner vectors pre-sized with heuristic,
  offset vectors and entries with exact capacity. Result: build_bands alloc -43.3%.

- [x] Pre-compute band sort keys + sort_unstable_by — avoids redundant max() in
  comparators, avoids sort temp allocation. Minimal timing impact (sort is cheap
  at 3-8 elements/band), but cleaner code.

Cumulative: 9.7 MB → 9.1 MB total alloc (-6.2%).

The remaining 8.5 MB is dominated by cosmic_text shaping, wgpu buffer management,
and font system internals — areas outside our control. Further micro-allocation
cleanup in upload_glyph/build_bands/prepare_with_depth would be diminishing returns.

build_bands improving less in time (-18.5%) than allocation (-43.3%) confirms it is
partly compute/sort bound (per-band sorting in band.rs), not just allocator bound.

### Parked — pursue only if profiling points here again

- [ ] Band builder algorithmic work — build_bands is partly compute-bound. Targets:
  - Temporary data layout: Vec<Vec<usize>> has poor locality; flat layout may help.
  - Reusable context: store scratch vectors on TextAtlas or BandBuilder struct.
    Requires API change (build_bands currently returns owned BandData).

- [ ] Color multiplication constant — replace `/ 255.0` with `* INV_255`. Cleanup
  win, not priority perf. Same for pre-computing default_color per text area.

## Optimization — GPU shader

### Profiling infrastructure (do first)

- [ ] Add wgpu-profiler to demo — wraps wgpu timestamp queries, outputs Chrome
  trace format. Gives per-frame GPU time for the text render pass. Currently we
  only measure CPU-side prep, not actual shader execution cost. Requires
  `Features::TIMESTAMP_QUERY` + `Features::TIMESTAMP_QUERY_INSIDE_PASSES`.

- [ ] RenderDoc inspection — capture a frame via Vulkan backend
  (`WGPU_BACKEND=vulkan`), verify early-exit is triggering, check per-pixel
  loop iteration counts. The `renderdoc` crate provides programmatic capture.

### Shader correctness / sync with reference

- [ ] Diff against Slug reference — our shaders were originally translated from
  the Slug HLSL reference (github.com/EricLengyel/Slug, MIT). Check for any
  upstream changes since our translation. Key areas:
  - CalcRootCode: reference uses bitwise sign extraction (`asuint(y) >> 31`)
    which reduces to a single LOP3 on NVIDIA. Ours uses `select()`. Verify
    naga produces equivalent SPIR-V, or switch to bitcast if it doesn't.
  - Band split (dual-sort) was removed from reference — we never had it, good.
  - Supersampling was removed — we never had it, good.
  - Dilation: reference uses dynamic vertex-shader dilation via inverse
    Jacobian. We use fixed 1px expansion — compare quality and performance.

### Shader optimization targets

- [ ] Texture fetch audit — each curve costs 2 textureLoad (p12, p3) + 1 for
  band ref. With ~8 curves/band, that's ~24 fetches per ray direction. Our
  2-texels-per-curve layout is already optimal for RGBA32Float. Verify we're
  not doing redundant loads.

- [ ] Branch divergence assessment — the `if abs(a.y) < 0.5` branch in
  solve_horiz_poly causes warp divergence between linear and quadratic paths.
  The linear case only fires for perturbed line segments (rare), so the branch
  should predict well. Confirm with Nsight or RGP if available.

- [ ] Band bounding-box pre-check — if a band's y-range doesn't intersect the
  current pixel (within half a pixel), skip the entire band loop. Could
  eliminate the band header textureLoad for edge pixels.

### Not worth pursuing

- Band split (dual-sort) — Lengyel removed it; hurts small text more than
  it helps large text. We never had it.
- Supersampling — removed from reference; dilation handles it.
- Compute shader rewrite — osor.io's wave-level approach is impressive but
  requires fundamentally different architecture (compute dispatch, tile-based
  curve caching). Not compatible with our render-pass integration.

## Future / long-term

- [ ] Unbounded retained memory — curve_data and band_data in TextAtlas
  (text_atlas.rs:39-40) grow with each uploaded glyph and are never compacted.
  This is intentional: they exist so texture growth can re-upload prior contents.
  Not a leak (memory is reachable and purposeful), but unbounded retained memory
  until eviction/compaction exists. Fix options: GPU texture-to-texture copy on
  growth, or LRU eviction that compacts the staging buffers.

- [ ] Store units_per_em on GlyphEntry — capture during atlas upload, eliminates the
  second get_font() call and skrifa re-parse entirely. Adds 4 bytes to GlyphEntry.

- [ ] Texture growth batching — if many glyphs are added in frame 1, the texture
  may grow multiple times sequentially. Predict final size from glyph count and
  pre-allocate once.

- [ ] Band count heuristic tuning — currently 4/8/12 bands based on curve count
  thresholds (10/30). Profile whether different thresholds or continuous scaling
  improves shader early-exit rates.

## Polish

- [ ] naga_oil for shader dedup — share fragment shader code between
  simple_shader.wgsl and shader.wgsl via `#import`. Eliminates copy-paste
  divergence risk. Alternative: WESL (wesl-rs) is newer but less mature.
- [ ] Band texture width configurable rather than hardcoded 4096
- [ ] ColorMode handling (currently stubbed — Slug renders in linear space)
- [ ] Texture growth under heavy load — stress test with CJK, mixed fonts
