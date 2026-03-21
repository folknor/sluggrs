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
  produce no output. Required before the iced swap ships. See integration
  spec (docs/integration-spec.md) for strategy options.
- [ ] Trim/eviction — currently trim() is a no-op. Implement usage tracking
  and LRU eviction to match cryoglyph's frame-boundary semantics.
- [ ] Depth plumbing — prepare_with_depth accepts the callback but discards
  the depth value. Any iced path relying on layered text depth will render
  incorrectly. Either implement or document as unsupported.

## Optimization — allocation reduction

Baseline: 92 glyphs, cold_prepare 1.9ms, warm_prepare 0.7ms, 9.7 MB total alloc.

### Done

- [x] Eliminate band_texels + add atlas scratch buffers — scratch_curve_texels and
  scratch_curve_locations on TextAtlas, band_texels eliminated via bytemuck::cast_slice.
  Result: upload_glyph alloc -37.5%, timing -23.8%.

- [x] Persist units_per_em_cache on TextRenderer — HashMap moved from local to field.
  Result: prepare_with_depth alloc -33.3%.

- [x] Cheap capacity fixes in build_bands — inner vectors pre-sized with heuristic,
  offset vectors and entries with exact capacity. Result: build_bands alloc -43.3%.

Cumulative: 9.7 MB → 9.1 MB total alloc (-6.2%).

The remaining 8.5 MB is dominated by cosmic_text shaping, wgpu buffer management,
and font system internals — areas outside our control. Further micro-allocation
cleanup in upload_glyph/build_bands/prepare_with_depth would be diminishing returns.

build_bands improving less in time (-18.5%) than allocation (-43.3%) confirms it is
partly compute/sort bound (per-band sorting in band.rs), not just allocator bound.

### Parked — pursue only if profiling points here again

- [ ] Reusable band-builder context — store scratch vectors on TextAtlas or a
  BandBuilder struct. Requires API change: build_bands currently owns all temporaries
  and returns owned BandData. Real refactor, not a drop-in change.

- [ ] Flatten band Vec<Vec<usize>> — 2-pass flat allocation with exact capacity.
  Better cache locality. Only worth it if band assignment + sorting remain a hotspot.

- [ ] Color multiplication constant — replace `/ 255.0` with `* INV_255`. Cleanup
  win, not priority perf. Same for pre-computing default_color per text area.

## Future / long-term

- [ ] Unbounded retained memory — curve_data and band_data in TextAtlas
  (text_atlas.rs:39-40) grow with each uploaded glyph and are never compacted.
  This is intentional: they exist so texture growth can re-upload prior contents
  (text_atlas.rs:103). Not a leak (memory is reachable and purposeful), but
  unbounded retained memory until eviction/compaction exists. Fix options:
  GPU texture-to-texture copy on growth, or LRU eviction that compacts the
  staging buffers.

- [ ] Store units_per_em on GlyphEntry — capture during atlas upload, eliminates the
  second get_font() call and skrifa re-parse entirely. Adds 4 bytes to GlyphEntry.

- [ ] Texture growth batching — if many glyphs are added in frame 1, the texture
  may grow multiple times sequentially. Predict final size from glyph count and
  pre-allocate once.

- [ ] Band count heuristic tuning — currently 4/8/12 bands based on curve count
  thresholds (10/30). Profile whether different thresholds or continuous scaling
  improves shader early-exit rates.

## Polish

- [ ] Band texture width configurable rather than hardcoded 4096
- [ ] Shader constant centralization between simple_shader.wgsl and shader.wgsl
- [ ] ColorMode handling (currently stubbed — Slug renders in linear space)
- [ ] Texture growth under heavy load — stress test with CJK, mixed fonts
