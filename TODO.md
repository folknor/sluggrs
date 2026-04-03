# TODO

## iced fork (`repos/iced/`, branch `sluggrs` on `folknor/iced`)

Cleanup opportunities in `wgpu/src/text.rs` — cryoglyph heritage and dual-pipeline leftovers.

- [ ] **Arc\<RwLock\<TextAtlas\>\> friction** — write-locked during prepare, read-locked during render. RwLockReadGuard dies before RenderPass<'a>. Had to hoist lock to lib.rs + Pipeline::atlas() accessor. Cleaner with a callback pattern or pre-locked render context.
- [ ] **Lazy raster vertex buffer** — Storage creates per-TextRenderer vertex + raster buffers. Most groups never see a non-vector glyph. Allocate raster buffer on first use.
- [ ] **Dead SwashCache allocation** — atlas owns a persistent one now, but iced still creates one per frame for the `_cache` API param (ignored). Goes away with the API cleanup below.
- [ ] **Inline prepare() free function** — thin wrapper that just calls renderer.prepare(). Existed for the old raster.prepare() call. Inline at its two call sites (State::prepare, Storage::prepare).
- [ ] **Shared shift-or-invalidate** — vector and raster cache-hit paths both do integer-delta adjustments, duplicated. Vector adjusts screen_rect[0..1], raster adjusts physical.x/y. Unify.
- [ ] **Remove unused `_encoder` and `_cache` params** — thread through 4 functions, never used. Cryoglyph API compat.


## Bugs

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

Baseline: 92 glyphs, ~1.8ms cold prepare on RTX 3080.
Mixed-locale baseline: 364 glyphs, ~4.6ms cold prepare (`brokkr hotpath --target email2`).

### Additional cold-path items

- [ ] **Split prepare_with_depth into two passes** — currently miss
  processing happens inline during glyph iteration. Pass 1: collect visible
  glyphs and distinct misses. Pass 2: build missing blobs (enables future
  parallelism). Pass 3: fast instance packing. Enables batching and cleaner
  architecture. Medium effort. *Multiple reviewers independently.*

- [x] **Cache per-curve band metadata** — `CurveMeta` struct cached in
  Phase 1 of `build_bands`, reused in Phase 2. 2.8× cold-path speedup on
  mixed-locale benchmark (364 distinct glyphs). **hb review**

- [x] **f32 cu2qu instead of f64** — converted all cu2qu math from f64 to
  f32. Identical subdivision counts verified against f64 baseline. CFF font
  test validates real glyph outlines. Recursion→iteration still possible
  as a follow-up.

- [ ] **Atlas initial capacity** — starts at 8192 elements, `grow_buffer()`
  doubles with full re-upload. Start at 1-4MB for known workloads, or
  predict final size from glyph count and pre-allocate once. Eliminates
  growth-copy stalls on first cold frame. Small effort.

- [ ] **Shrink CPU buffer mirrors on reset** — `buffer_data` keeps capacity
  after `reset_atlas()`. After a large working-set spike, CPU memory stays
  high even though GPU buffer is recreated. Use `shrink_to_fit()` or
  `= Vec::new()` in reset. **perf review**

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
