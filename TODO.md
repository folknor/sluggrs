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

### 1. Eliminate band_texels + add atlas scratch buffers

Remove the intermediate `band_texels: Vec<[u32; 4]>` in upload_glyph (text_atlas.rs:145)
by passing `band_data.entries` directly to upload_wrapped_texels_u32 via bytemuck::cast_slice.
Eliminates ~1.8 KB allocation per glyph.

Add `scratch_curve_texels: Vec<[f32; 4]>` and `scratch_curve_locations: Vec<CurveLocation>`
to TextAtlas. Clear+reuse across upload_glyph() calls instead of allocating fresh per glyph
(text_atlas.rs:121, text_atlas.rs:130). Eliminates ~180 allocations (2 per glyph × 91 glyphs).

Expected: ~90% reduction in upload_glyph per-call alloc (currently 18.1 KB/call, 1.6 MB total).

### 2. Persist units_per_em_cache on TextRenderer

Move the `HashMap<fontdb::ID, f32>` from a local in prepare_with_depth (text_renderer.rs:71)
to a field on TextRenderer. Avoids HashMap allocation per frame and avoids re-parsing skrifa
FontRef + head table on warm frames. Real warm-frame improvement.

### 3. Cheap capacity fixes in build_bands

- `hband_offsets` and `vband_offsets` (band.rs:142): use Vec::with_capacity — sizes are
  known (band_count_y and band_count_x).
- `entries` (band.rs:132): reserve up front. The final size is known after the offset pass
  (`current_offset` gives the total texel count), so entries can be allocated with exact
  capacity before writing headers + curve refs. Avoids growth reallocations.
- Inner vectors in hband_curves/vband_curves (band.rs:54): use capacity hint based on
  `curves.len() / band_count` as a heuristic.

### 4. Reusable band-builder context

Store reusable scratch vectors for build_bands() on TextAtlas or a dedicated BandBuilder
struct (hband_curves, vband_curves, offsets, entries). Clear+reuse across glyphs instead
of allocating Vec<Vec<usize>> fresh per call.

Note: build_bands currently owns all temporaries and returns an owned BandData (band.rs:35).
The API needs to change to accept a mutable context or move builder logic onto a state object.
This is a real refactor, not a drop-in change.

### 5. Flatten band structure (only if profiling still points here)

Replace nested Vec<Vec<usize>> with a 2-pass flat allocation: count band assignments,
allocate single Vec<u32> with exact capacity, fill. Better cache locality, eliminates
inner Vec overhead. Bigger refactor (~50 lines).

### Cleanup wins (not priority perf)

- [ ] Color multiplication constant — replace `/ 255.0` with `* (1.0 / 255.0)` in
  prepare_with_depth color conversion (text_renderer.rs:192). Fine to do, but on a
  path measured at 16µs warm, the payoff is cleanup-level, not priority.
- [ ] Pre-compute default_color — convert text_area.default_color to [f32; 4] once
  per text area instead of per glyph in the None branch. Same category.

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
