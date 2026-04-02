# TODO

## Bugs found by review (2026-03-27)

- [x] `prepare_with_depth` does too much — text_renderer.rs:57 handles font
  lookup, TTC face resolution, variation setup, outline extraction, fake
  italic, cache insertion, non-vector classification, instance packing, and
  vertex upload. Renderer and atlas are tightly coupled through pub(crate)
  internals.

- [x] Dead API surface — `RenderError` variants `RemovedFromAtlas` and
  `ScreenResolutionChanged` are public but `render()` always returns
  `Ok(())`. `SwashCache` and `CommandEncoder` are in the compatibility
  signature but unused.

- [ ] Emoji classification should be an explicit API — non-vector glyphs are
  currently a sentinel `GlyphEntry` in glyph_cache.rs:53, skipped as a side
  effect of prepare(). For clean two-pass routing (sluggrs + cryoglyph
  fallback), classification needs to be a separate step, not incidental to
  prepare(). **arch review**: the silent drop with no error/placeholder/signal
  to iced violates the integration contract — users see missing characters
  in otherwise normal text with no indication anything is wrong.

- [ ] Signed/unsigned confusion in shader texture addressing — the shader
  casts `vec2<u32>` to `vec2<i32>` for textureLoad coordinates, and uses
  arithmetic right shift on signed `i32` for row calculation. If band_loc.x
  goes negative (from offset addition wrapping or a bug), the arithmetic
  shift propagates the sign bit, corrupting row calculation with a silent
  wrong-texel read. Works today because numbers are small, but no safety
  margin or validation. **wgpu review**

## Bugs found by deep review (2026-03-27)

- [ ] trim() can invalidate already-prepared draw data — if
  prepare→trim(reset)→render happens in sequence, render() draws from
  the old instance buffer against the new (empty) atlas textures.
  RemovedFromAtlas error variant exists but is never raised due to no
  generation tracking. **bugs review**

- [ ] Scroll offset not accounted for in CPU-side culling —
  text_renderer.rs clips glyphs against screen bounds without scroll_offset,
  but shader applies scroll_offset to screen position. Dormant while
  scroll_offset has no public setter on Viewport, but will cause
  first/last visible glyphs to flicker once scroll API is exposed.
  **bugs review**

- [x] Curve texel pair row-straddling invariant undocumented — shader reads
  p3 at curve_loc.x + 1, same row. If curve_loc.x were 4095, this is OOB.
  Currently safe because curve_cursor is always even and texture width is
  even, so x is never 4095. Invariant is implicit. Add debug assertion
  or document. **gpu, fonts, wgpu review**

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

### Cold-frame waste (found by deep perf review)

- [ ] Per-glyph write_texture calls — each cache-miss glyph produces 2+
  write_texture calls. Batching curve uploads for all glyphs into a single
  write would reduce driver overhead. At 92 glyphs = 184+ wgpu command
  submissions on cold frame. **perf review**

- [ ] CPU-side texture mirrors only clear() on reset, not shrink — after a
  large working-set spike, curve_data and band_data keep their capacity
  indefinitely even though GPU textures are recreated at 1 row. Use
  `shrink_to_fit()` or `= Vec::new()` in reset_atlas. **perf review**

### Parked — pursue only if profiling points here again

- [ ] Color multiplication constant — replace `/ 255.0` with `* INV_255`. Cleanup
  win, not priority perf. Same for pre-computing default_color per text area.

## Optimization — GPU shader

### Profiling infrastructure (do first)

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

- [ ] Branch divergence assessment — the `if abs(a.y) < 0.25` branch in
  solve_horiz_poly causes warp divergence between linear and quadratic paths.
  Threshold lowered from 0.5 to 0.25 (see bugs section), reducing the
  number of genuine quadratics entering the linear path, but some shallow
  CFF-derived curves may still qualify. Confirm divergence rate with
  Nsight or RGP if available.

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

## Divergences from harfbuzz Slug implementation (found by hb review)

### Done

- [x] Exact geometry for lines — p2=p1 encoding, removed CPU perturbation **hb review**
- [x] Implicit p1 contour sharing — -45.6% curve texels **hb review**
- [x] Axis-aligned curve filtering — skip horiz from hbands, vert from vbands **hb review**
- [x] Dual sorted bands with split point — direction-aware early exit **hb review**
- [x] RGBA16I texture format (Stages A+B) — halved texture memory **hb review**
- [x] Half-pixel dilation — reduced from 1px to 0.5px **hb review**
- [x] Zero-length curve rejection — filter p1==p3 in outline extraction **hb review**
- [x] Shader MSAA — 4x supersampling below 16ppem, toggled with E key **hb review**
- [x] Stem darkening — ppem-aware gamma, no-op above 48ppem **hb review**
- [x] GpuOutline → type alias — prepare_outline is now a clone **hb review**
- [x] Cu2qu — tangent-line intersection, f64, tolerance 0.5 font units **hb review**
- [x] i16 overflow guard — reject glyphs exceeding quantization range **hb review**
- [x] Band count policy — 1:1 up to cap of 16 (matching harfbuzz) **hb review**
- [x] Unified storage buffer (Stage C) — single array<vec4<i32>>, -352 lines **hb review**

### Remaining

- [ ] Jacobian-based vertex dilation — full MVP-aware half-pixel expansion.
  Only needed if we support rotation/non-uniform scaling.
  Harfbuzz: `hb-gpu-vertex.wgsl:49-81`. **hb review**

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
  may grow multiple times sequentially. Each growth does a full re-upload of all
  accumulated texture data, so multiple growths in a single prepare() call
  produce O(n) full re-uploads. On a frame that loads a new script (CJK, Arabic)
  with 200+ new glyphs across growth boundaries, this is an unbounded stall.
  Predict final size from glyph count and pre-allocate once. **arch review**

- [ ] Band count heuristic tuning — currently 4/8/12 bands based on curve count
  thresholds (10/30). Profile whether different thresholds or continuous scaling
  improves shader early-exit rates.

## Architecture (found by deep review)

- [ ] GlyphInstance 84-byte stride not 16-byte aligned — depth field makes
  struct 84 bytes (5×16 + 4). Some GPU architectures fetch vertex data in
  16-byte chunks, wasting 12 bytes per instance on bus. Pad to 96 bytes
  (add `_pad: [f32; 3]`) or pack depth into glyph_data.w. Verify
  84-byte stride works on all backends first. **perf, arch review**

- [ ] Scroll offset has no public API — Viewport::update only accepts
  Resolution, scroll_offset in Params is always [0,0]. The shader reads
  it but it's unreachable from the library API. Need
  `Viewport::set_scroll_offset()` or extend `update()`. **wgpu, arch review**

- [ ] ColorMode has no effect — TextAtlas::color_mode stored with
  `#[allow(dead_code)]` but never read. Shader always does same blend.
  May produce wrong results in linear-RGB framebuffers with sRGB colors
  (the Web mode). **arch review**

- [ ] TextRenderer/TextAtlas coupling — renderer reaches into atlas
  internals via pub(crate). Policy, cache state, and upload orchestration
  spread across both types. Any change to classification, eviction, or
  fallback routing cuts across both modules. **arch review**

- [ ] API doesn't encode TextRenderer↔TextAtlas lifetime relationship —
  render() accepts any &TextAtlas but the pipeline was baked from a
  specific atlas at construction. Mispairing is type-correct but
  produces wrong rendering. **arch review**

- [ ] CommandEncoder parameter is unnecessarily restrictive —
  prepare_with_depth takes `&mut CommandEncoder` but never uses it.
  The `&mut` borrow prevents caller from using encoder during prepare.
  Cryoglyph compatibility constraint. **arch review**

## Polish

- [ ] naga_oil for shader dedup — share fragment shader code between
  simple_shader.wgsl and shader.wgsl via `#import`. Eliminates copy-paste
  divergence risk. Alternative: WESL (wesl-rs) is newer but less mature.
- [ ] Band texture width configurable rather than hardcoded 4096
- [ ] ColorMode handling (currently stubbed — Slug renders in linear space)
- [ ] Texture growth under heavy load — stress test with CJK, mixed fonts
