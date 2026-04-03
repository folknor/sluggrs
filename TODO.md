# TODO

## Bugs

- [ ] **Emoji classification needs explicit API** — non-vector glyphs are a
  sentinel `GlyphEntry`, skipped silently during prepare(). For two-pass
  routing (sluggrs + cryoglyph fallback), classification must be a separate
  step. Users see missing characters with no indication anything is wrong.
  **arch review**

- [ ] **Signed/unsigned confusion in shader texture addressing** — shader
  casts `vec2<u32>` to `vec2<i32>` for textureLoad coordinates, uses
  arithmetic right shift on signed `i32` for row calculation. If band_loc.x
  wraps negative, sign bit propagates and corrupts row calculation silently.
  Works today because numbers are small. **wgpu review**

- [ ] **trim() can invalidate already-prepared draw data** — if
  prepare→trim(reset)→render happens in sequence, render() draws from
  the old instance buffer against new (empty) atlas. RemovedFromAtlas error
  variant exists but is never raised — no generation tracking. **bugs review**

- [ ] **Scroll offset not in CPU-side culling** — glyphs clipped against
  screen bounds without scroll_offset, but shader applies it. Dormant while
  scroll_offset has no public setter on Viewport, but will cause first/last
  visible glyphs to flicker once scroll API is exposed. **bugs review**

## CPU — Cold path

Baseline: 92 glyphs, ~753µs cold prepare on RTX 3080.

### High priority — all 6 reviewers

- [x] **BandScratch — reuse all Vecs in build_bands()** — 14 fresh
  allocations per glyph replaced with a `BandScratch` struct on `TextAtlas`,
  cleared and reused each call.

- [x] **Remove prepare_outline() clone** — `GpuOutline` was a type alias for
  `GlyphOutline`, `prepare_outline` was a pointless clone. Removed function,
  type alias, and all references. Outline passed directly, only cloned for
  italic shear.

- [x] **Batch queue.write_buffer** — per-glyph `write_buffer` replaced with
  single `flush_uploads()` at end of prepare. `grow_buffer` no longer
  re-uploads; sets flush cursor to 0 so flush covers everything.

- [x] **Eliminate band_i32 widening alloc** — replaced `collect()` into
  temporary `Vec<[i32; 4]>` with direct `extend()` from iterator into
  `buffer_data`. Zero-alloc widening.

### Strong consensus — 4-5 reviewers

- [x] **Cache FontRef per font_id** — `CachedFont` struct holds
  `Arc<Font>`, face_index, and units_per_em per `(font_id, weight)`.
  Eliminates db().face(), get_font() lock, and FontRef re-parse on
  repeated misses from the same font.

- [x] **Pre-compute curve_data_offset** — `build_bands` now computes
  `band_element_count` internally and adds it to curve ref offsets when
  writing them. Eliminates the fixup loop in `upload_glyph`.

### Additional cold-path items

- [ ] **Split prepare_with_depth into two passes** — currently miss
  processing happens inline during glyph iteration. Pass 1: collect visible
  glyphs and distinct misses. Pass 2: build missing blobs (enables future
  parallelism). Pass 3: fast instance packing. Enables batching and cleaner
  architecture. Medium effort. *Multiple reviewers independently.*

- [ ] **Cache per-curve band metadata** — `build_bands` iterates all curves
  twice (counting then assignment), recomputing min/max/band-range both
  times. Cache `(hband_min, hband_max, vband_min, vband_max,
  is_horizontal, is_vertical)` per curve in first pass. Avoids redundant
  float math. Small effort. **hb review**

- [ ] **f32 cu2qu instead of f64** — cu2qu inherited f64 from harfbuzz. At
  0.5 font-unit tolerance, f32 has sufficient precision. Halves register
  pressure, improves vectorization. Also consider converting recursion to
  iteration with explicit stack. 5-10% cold for CFF fonts. Medium effort.

- [ ] **Atlas initial capacity** — starts at 8192 elements, `grow_buffer()`
  doubles with full re-upload. Start at 1-4MB for known workloads, or
  predict final size from glyph count and pre-allocate once. Eliminates
  growth-copy stalls on first cold frame. Small effort.

- [ ] **Shrink CPU buffer mirrors on reset** — `buffer_data` keeps capacity
  after `reset_atlas()`. After a large working-set spike, CPU memory stays
  high even though GPU buffer is recreated. Use `shrink_to_fit()` or
  `= Vec::new()` in reset. **perf review**

## CPU — Warm path

- [x] **Store units_per_em on GlyphEntry** — captured during atlas upload,
  eliminates `resolve_units_per_em()` and `units_per_em_cache` HashMap
  entirely. Warm path reads directly from entry.

- [x] **FxHash for GlyphMap** — switched from SipHash to rustc-hash FxHash.
  2-3x faster hashing for 12-byte GlyphKey.

- [ ] **Inline cache-hit path in prepare_with_depth** — `resolve_glyph(&mut
  self)` borrows mutably on every glyph, even warm hits. Inline the HashMap
  lookup, only call `resolve_glyph_miss()` on miss. Better branch
  prediction, fewer function call overheads. ~5-10µs warm. **hb review**

- [ ] **Add sluggrs-only warm benchmark** — hotpath.rs recreates and reshapes
  a fresh `cosmic_text::Buffer` every iteration, so `warm_prepare_avg_us`
  includes shaping/layout cost that isn't sluggrs. Add a benchmark with a
  pre-shaped persistent Buffer. Needed before optimizing warm path.
  *Multiple reviewers.*

## GPU — Shader

Baseline: 11µs headless / 71µs windowed (RTX 3080, 92 glyphs). Windowed
dominated by compositor/surface, not text math.

### Profiling infrastructure (do first)

- [ ] **RenderDoc inspection** — capture a frame via Vulkan backend
  (`WGPU_BACKEND=vulkan`), verify early-exit is triggering, check per-pixel
  loop iteration counts. The `renderdoc` crate provides programmatic
  capture.

### Optimization targets

- [ ] **Pack i16 pairs into i32** — halve storage buffer bandwidth. Current
  `array<vec4<i32>>` is 16 bytes/texel. Pack two i16 lanes per u32 with
  shader unpack helpers. 10-20% GPU. Medium effort. *Majority consensus.*

- [ ] **Analytical AA / reduce 5x MSAA** — below 16ppem, shader runs
  `render_single` 5 times (full curve evaluation each). Biggest GPU cost
  at small text sizes. Evaluate curve intersection once and compute
  coverage analytically, or reduce to 2 samples. Up to 3-4x GPU at small
  ppem. Large effort.

- [ ] **Move em_rect/band_transform into blob header** — shrink
  GlyphInstance from 96→64 bytes, reduce vertex bandwidth. Shader decodes
  from blob header instead of per-instance vertex attributes. Medium effort.
  *Majority consensus.*

- [ ] **Precompute a,b from unshifted coords** — compute `a` and `b`
  outside solver from unshifted `q12`/`q3`, pass as arguments. Matches
  harfbuzz at `hb-gpu-fragment.wgsl:205-206`. Preserves exact zeros for
  `== 0.0` branch, saves 2 vec2 subtractions per curve. **hb review**

- [ ] **Eliminate redundant decode_offset per curve read** — every curve
  does `decode_offset` (integer add of 32768) per fragment per ray
  direction. Store curve refs as already-decoded offsets, or pre-add
  `glyph_base` on CPU for absolute addressing. 16 adds/fragment saved.
  Buffer is append-only so absolute addressing is safe. **perf review**

- [ ] **Zero dilation for large text (48ppem+)** — coverage at exact glyph
  boundary is already 0.0 or 1.0 at large sizes. Set dilation to 0 in
  vertex shader, reducing fragment count ~4-8%. **hb review**

- [ ] Texture fetch audit — verify no redundant loads in curve inner loop
- [ ] Branch divergence assessment — `abs(a.y) < 0.25` warp divergence
  between linear and quadratic paths. Confirm with Nsight/RGP if available.
- [ ] Band bounding-box pre-check — skip band loop if y-range doesn't
  intersect pixel (within half a pixel)

### Correctness / reference sync

- [ ] **Diff against Slug reference** — check for upstream changes since
  translation. Key areas:
  - CalcRootCode: reference uses bitwise sign extraction (`asuint(y) >> 31`,
    single LOP3 on NVIDIA). Ours uses `select()`. Verify naga SPIR-V
    equivalence or switch to bitcast.
  - Dilation: reference uses dynamic vertex-shader dilation via inverse
    Jacobian. We use fixed 0.5px — compare quality and performance.

### Not worth pursuing

- Band split (dual-sort) — Lengyel removed it; hurts small text
- Supersampling — removed from reference; dilation handles it
- Compute shader rewrite — fundamentally different architecture, not
  compatible with render-pass integration

## Architecture

- [ ] **GlyphInstance vertex bandwidth** — struct is 96 bytes with 8 bytes
  of padding (`_pad: [f32; 2]`). Pack depth into `glyph_data.w` and ppem
  into color.w (or similar) to shrink to 80 or 64 bytes. See also blob
  header approach under GPU targets. **perf, arch review**

- [ ] **Scroll offset has no public API** — Viewport::update only accepts
  Resolution, scroll_offset in Params is always [0,0]. Shader reads it but
  unreachable from library API. **wgpu, arch review**

- [ ] **ColorMode has no effect** — stored with `#[allow(dead_code)]`, never
  read. Shader always does same blend. May produce wrong results in
  linear-RGB framebuffers with sRGB colors. **arch review**

- [ ] **TextRenderer/TextAtlas coupling** — renderer reaches into atlas via
  pub(crate). Policy, cache state, and upload orchestration spread across
  both. Any change to classification, eviction, or fallback routing cuts
  across both. **arch review**

- [ ] **API doesn't encode TextRenderer↔TextAtlas lifetime** — render()
  accepts any &TextAtlas but pipeline was baked from a specific atlas.
  Mispairing is type-correct but wrong rendering. **arch review**

- [ ] **CommandEncoder param unnecessarily restrictive** —
  prepare_with_depth takes `&mut CommandEncoder` but never uses it. The
  `&mut` borrow prevents caller from using encoder during prepare.
  Cryoglyph compatibility constraint. **arch review**

## Future / Long-term

- [ ] **Parallel cold glyph processing (rayon)** — miss processing is
  embarrassingly parallel per distinct glyph. Collect missing keys, process
  in parallel into blobs, commit serially. 2-4x cold speedup. Depends on
  two-pass split. Medium effort.

- [ ] **Second-level blob cache** — cache encoded glyph blobs independent of
  atlas residency. Atlas reset drops GPU residency only, re-upload is
  memcpy. Huge for mixed/trim workloads. Large effort.

- [ ] **Retained prepared-text cache** — skip instance rebuild for unchanged
  text. Cache per-text-area instances keyed by buffer generation, viewport,
  transform. Biggest possible warm-path win. Large effort.

- [ ] **Unbounded retained memory** — buffer_data grows with each uploaded
  glyph, never compacted. Intentional (needed for growth re-upload). Fix:
  GPU buffer-to-buffer copy on growth, or LRU eviction with compaction.

- [ ] **Texture growth batching** — multiple growths in one prepare() produce
  O(n) full re-uploads. Predict final size from glyph count and
  pre-allocate once. **arch review**

### Harfbuzz divergences remaining

- [ ] **Jacobian-based vertex dilation** — full MVP-aware half-pixel
  expansion. Only needed for rotation/non-uniform scaling. Harfbuzz:
  `hb-gpu-vertex.wgsl:49-81`. **hb review**

## Polish

- [ ] naga_oil for shader dedup — `#import` to share code between
  simple_shader.wgsl and shader.wgsl. Eliminates copy-paste divergence.
- [ ] ColorMode handling (currently stubbed — Slug renders in linear space)
- [ ] Texture growth stress test with CJK, mixed fonts

### Parked

- [ ] Color multiplication — `/ 255.0` → `* INV_255`. Cleanup, not priority.

## Done

### Bugs fixed
- [x] prepare_with_depth decomposed (was doing too much)
- [x] Dead API surface removed — unused RenderError variants, SwashCache, CommandEncoder
- [x] Curve texel pair row-straddling invariant documented

### CPU allocation reduction

Baseline: 9.7 MB → 9.1 MB total alloc (-6.2%). Remaining ~8.5 MB dominated
by cosmic_text shaping, wgpu buffer management, and font internals.

- [x] Eliminate band_texels + add atlas scratch buffers (-37.5% upload_glyph alloc, -23.8% timing)
- [x] Persist units_per_em_cache on TextRenderer (-33.3% prepare_with_depth alloc)
- [x] Cheap capacity fixes in build_bands (-43.3% build_bands alloc)
- [x] Pre-compute band sort keys + sort_unstable_by (minimal timing, cleaner code)

### Harfbuzz convergence (14 items)
- [x] Exact geometry for lines — p2=p1 encoding
- [x] Implicit p1 contour sharing — -45.6% curve texels
- [x] Axis-aligned curve filtering — skip horiz from hbands, vert from vbands
- [x] Dual sorted bands with split point — direction-aware early exit
- [x] RGBA16I texture format (Stages A+B) — halved texture memory
- [x] Half-pixel dilation — reduced from 1px to 0.5px
- [x] Zero-length curve rejection — filter p1==p3 in outline extraction
- [x] Shader MSAA — 4x supersampling below 16ppem
- [x] Stem darkening — ppem-aware gamma, no-op above 48ppem
- [x] GpuOutline → type alias
- [x] Cu2qu — tangent-line intersection, f64, tolerance 0.5 font units
- [x] i16 overflow guard — reject glyphs exceeding quantization range
- [x] Band count policy — 1:1 up to cap of 16 (matching harfbuzz)
- [x] Unified storage buffer (Stage C) — single array<vec4<i32>>, -352 lines
