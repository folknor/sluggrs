# sluggrs Integration Spec: Replacing cryoglyph in iced

## Overview

Replace cryoglyph (bitmap atlas text renderer) with sluggrs (GPU curve
evaluation) in iced's wgpu backend. The integration point is a single file:
`iced/wgpu/src/text.rs` (~650 lines).

## Blockers (resolve before implementation)

### Blocker 1: Font byte access

**Status**: Largely resolved — needs a confirming spike.

`prepare()` needs raw font bytes for skrifa outline extraction.
cosmic_text provides viable paths:

- `FontSystem::db()` returns the `fontdb::Database`
- `Font::data()` exposes raw bytes (`repos/cosmic-text/src/font/mod.rs:114`)
- `FontSystem::get_font(font_id)` returns a `Font` with `.data()` access

The remaining question is the cleanest extraction path: whether to use
`FontSystem::get_font(id).data()` directly or reach into fontdb. The spike
should produce a standalone test that:

1. Creates a `FontSystem`, loads a font
2. Shapes text to get `LayoutGlyph`s with font_id + glyph_id
3. Extracts raw bytes via the chosen API path
4. Passes those bytes to `skrifa::FontRef::new()` + `extract_outline()`
5. Produces a valid `GlyphOutline`

### Blocker 2: Glyph cache key

The glyph cache key must uniquely identify a glyph's outline geometry.
`(font_id, glyph_id)` alone is NOT sufficient. cosmic_text's shaping
pipeline includes additional outline-affecting state:

- **font_weight** (`LayoutGlyph::font_weight`): Affects outline selection
  in variable fonts. SwashCache uses this when producing outlines
  (`repos/cosmic-text/src/swash.rs:88`).
- **cache_key_flags** (`LayoutGlyph::cache_key_flags`): Includes
  `FAKE_ITALIC` which affects outline generation via a shear transform.
- **Variation coordinates**: For variable fonts with axes beyond weight
  (width, optical size, etc.), the specific instance affects outlines.

The key must capture all of these:

```rust
#[derive(Hash, Eq, PartialEq, Clone, Copy)]
pub struct GlyphKey {
    /// cosmic_text font identifier — uniquely identifies the face
    /// within the font database, including collection index
    font_id: cosmic_text::fontdb::ID,
    /// Glyph index within the font
    glyph_id: u16,
    /// Font weight used during shaping — affects outline selection
    /// in variable fonts
    font_weight: u16,
    /// Flags that affect outline generation (e.g. FAKE_ITALIC)
    cache_key_flags: cosmic_text::CacheKeyFlags,
}
```

**Note on FAKE_ITALIC**: If `cache_key_flags` includes FAKE_ITALIC,
sluggrs must apply the same shear transform that SwashCache applies.
This may need to happen during outline extraction or as a post-process
on the extracted curves. The spike should determine whether skrifa
handles this or if we need to apply it ourselves.

**Note on variation coordinates**: `font_weight` covers the weight axis,
which is the most common variable font axis. If cosmic_text exposes
additional variation state at layout time, it should be included in the
key. The spike should enumerate what's available on `LayoutGlyph` and
`Font`.

**Important**: The cache key deliberately excludes size, position, and
subpixel offset. Outlines are resolution-independent — vertex attributes
handle the per-instance transform. This is the core architectural advantage
over cryoglyph's `CacheKey` which includes physical position.

### Blocker 3: Non-vector glyph fallback

Slug cannot render bitmap glyphs (color emoji, bitmap-only fonts).
Silently skipping these produces user-visible missing text and is not
acceptable at any phase.

**Hybrid rendering is required before swapping the dependency in iced.**
This is not a polish item — iced's `State::prepare` and `render` assume
the renderer handles all text (`text.rs:304`, `text.rs:374`). A renderer
that drops glyphs will break real applications.

**Strategy**: Detect non-vector glyphs (no outline from skrifa) during
prepare and route them to a bitmap fallback path.

Options for the bitmap path:

1. **Embedded cryoglyph**: sluggrs depends on cryoglyph as a library
   and delegates non-vector glyphs to it. Heavy dependency but zero
   new bitmap code. Two draw calls per text batch.

2. **Minimal bitmap renderer**: Implement a stripped-down bitmap atlas
   (SwashCache rasterization + texture packing) for non-vector glyphs
   only. Lighter dependency, more code.

3. **Upstream cryoglyph in iced**: Keep cryoglyph in iced's workspace
   alongside sluggrs, use it as the fallback at the `text.rs` level.
   Avoids making sluggrs depend on cryoglyph.

Recommendation: Option 3 for initial integration (least coupling),
migrate to Option 2 when the integration is stable.

**Phase 1 minimum**: During the spike phase, rendering a visible
placeholder (missing-glyph box) for non-vector glyphs is acceptable
for development/testing. But the fallback must be implemented before
the iced dependency swap ships.

## API Surface

sluggrs must export the same public types that `text.rs` imports from
cryoglyph. The types are listed here with their cryoglyph behavior and
the sluggrs equivalent.

### Types (identical interface)

These types are simple and can match cryoglyph exactly:

| Type | Purpose | Notes |
|------|---------|-------|
| `Resolution` | `{ width: u32, height: u32 }` | Identical |
| `TextBounds` | `{ left, top, right, bottom: i32 }` | Identical |
| `TextArea<'a>` | `{ buffer, left, top, scale, bounds, default_color }` | Identical, references `cosmic_text::Buffer` |
| `PrepareError` | `enum { AtlasFull }` | Keep variant, even though our "atlas" is different |
| `RenderError` | `enum { RemovedFromAtlas, ScreenResolutionChanged }` | Keep variants for API compat |
| `ColorMode` | `enum { Accurate, Web }` | Stub — Slug doesn't distinguish gamma modes yet |

### Types (different internals, same interface)

#### `Cache`

Cryoglyph: Holds shader module, sampler, bind group layouts, pipeline
layout, cached render pipelines. Shared across all atlases.

sluggrs: Same role — holds our Slug shader module, bind group layouts for
curve/band textures and params uniform, pipeline layout, cached pipelines.
No sampler needed (we use `textureLoad`, not `textureSample`).

```rust
pub struct Cache {
    shader: wgpu::ShaderModule,
    texture_layout: wgpu::BindGroupLayout,   // curve + band textures
    params_layout: wgpu::BindGroupLayout,     // screen resolution uniform
    pipeline_layout: wgpu::PipelineLayout,
    pipelines: Mutex<Vec<(TextureFormat, MultisampleState, Option<DepthStencilState>, RenderPipeline)>>,
}

impl Cache {
    pub fn new(device: &wgpu::Device) -> Self;
}
```

#### `Viewport`

Cryoglyph: Params buffer with screen resolution, bind group for the
uniform.

sluggrs: Identical structure and behavior. The shader uniform layout
matches (screen_size vec2 + padding).

```rust
pub struct Viewport {
    params_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl Viewport {
    pub fn new(device: &wgpu::Device, cache: &Cache) -> Self;
    pub fn update(&mut self, queue: &wgpu::Queue, resolution: Resolution);
}
```

#### `TextAtlas`

Cryoglyph: Two bitmap atlas textures (color + mask), etagere packer, LRU
glyph cache keyed by `cosmic_text::CacheKey` (physical glyph position).

sluggrs: Two data textures (curve Rgba32Float + band Rgba32Uint), a glyph
outline cache keyed by `GlyphKey` (resolution-independent), and a bind
group referencing both textures.

The curve texture stores control points. The band texture stores the
acceleration structure. Both grow as new glyphs are encountered. The band
texture is always `BAND_TEXTURE_WIDTH` (4096) wide and grows in height.

##### GlyphEntry

The bridge between glyph cache, texture data, and vertex packing.
Every field here is consumed downstream — this is the central data
contract.

```rust
/// Location and metadata for a cached glyph in the GPU textures.
struct GlyphEntry {
    /// Texel offset of this glyph's band headers in the band texture.
    /// Stored in LINEAR texel space (a flat index into the logical
    /// texture). The shader's calc_band_loc() converts this to 2D
    /// coordinates using BAND_TEXTURE_WIDTH-wide wrapping:
    ///   x = (glyph_loc.x + offset) & (BAND_TEXTURE_WIDTH - 1)
    ///   y = (glyph_loc.x + offset) >> LOG_BAND_TEXTURE_WIDTH
    /// CPU code must pack band data contiguously at this offset.
    /// Offsets within a glyph's band data (header offsets, curve list
    /// offsets) are also linear and interpreted the same way.
    band_offset: u32,

    /// Number of horizontal bands (y-direction) minus 1. Passed to
    /// the shader as band_max.y.
    band_max_y: u32,
    /// Number of vertical bands (x-direction) minus 1. Passed to
    /// the shader as band_max.x.
    band_max_x: u32,

    /// Scale + offset to map em-space coordinates to band indices.
    /// Computed by build_bands(). Passed to the shader as-is.
    band_transform: [f32; 4],  // [scale_x, scale_y, offset_x, offset_y]

    /// Glyph bounding box in em-space (from GpuOutline.bounds).
    /// Used to compute screen_rect and em_rect vertex attributes.
    bounds: [f32; 4],  // [min_x, min_y, max_x, max_y]
}
```

Vertex packing reads from GlyphEntry:
- `screen_rect` = bounds scaled by font_size/units_per_em, positioned
  by cosmic_text layout
- `em_rect` = bounds directly
- `band_transform` = band_transform directly
- `glyph_data` = `[band_offset, 0, band_max_x, band_max_y]`

```rust
pub struct TextAtlas {
    cache: Cache,
    curve_texture: wgpu::Texture,
    curve_view: wgpu::TextureView,
    band_texture: wgpu::Texture,
    band_view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
    format: wgpu::TextureFormat,
    color_mode: ColorMode,

    // Glyph cache: GlyphKey → location in textures
    glyph_cache: HashMap<GlyphKey, GlyphEntry>,

    // Write cursors (append-only within a frame)
    curve_cursor: u32,  // next free texel in curve texture
    band_cursor: u32,   // next free texel in band texture (linear, pre-wrap)
}
```

**Key difference from cryoglyph**: Glyph data is resolution-independent.
A glyph cached once serves all sizes — only the vertex attributes change.
This eliminates re-rasterization on scale changes.

##### Texture growth and invalidation

Curve texture: single row, starts at width 1024. When `curve_cursor`
exceeds width, reallocate at 2x width, re-upload all existing data, and
rebind. Existing offsets remain valid because data is append-only and
positions don't change.

Band texture: fixed width `BAND_TEXTURE_WIDTH` (4096), starts at height 1.
When `band_cursor / BAND_TEXTURE_WIDTH` exceeds current height, reallocate
at 2x height, re-upload, and rebind. Same invariant: existing offsets
are stable.

Rebind: after any texture reallocation, recreate the bind group that
references both texture views. The `TextRenderer` holds a pipeline (which
references the bind group layout, not the bind group itself), so it
remains valid. The bind group is set per-draw in `render()`.

##### trim() semantics

**Design decision**: sluggrs's `trim()` retains cached glyph data.

cryoglyph's `trim()` clears the per-frame "glyphs in use" sets but
retains all atlas contents and the LRU cache. This is important because
iced calls `trim()` every frame (`text.rs:424`).

sluggrs must NOT clear the glyph cache on trim. Instead:

- Track which glyphs were referenced this frame (glyphs_in_use set)
- On trim, clear glyphs_in_use but retain glyph_cache and texture data
- Eviction (if needed) happens only when textures are full: evict
  least-recently-used glyphs not in the current frame's usage set

This matches cryoglyph's behavior and avoids re-extracting every glyph
every frame. Since vector glyph data is small (10–60 texels per glyph
vs thousands of pixels per bitmap), texture pressure is much lower and
eviction may rarely trigger in practice.

```rust
impl TextAtlas {
    pub fn new(device, queue, cache, format) -> Self;
    pub fn with_color_mode(device, queue, cache, format, color_mode) -> Self;
    pub fn trim(&mut self);  // clears usage set, retains cache

    /// Upload a glyph's curve + band data. Returns the GlyphEntry.
    /// Grows textures if needed.
    fn upload_glyph(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        gpu_outline: &GpuOutline,
        band_data: &BandData,
    ) -> GlyphEntry;
}
```

#### `TextRenderer`

Cryoglyph: Collects `GlyphToRender` instances (pos, atlas UV, color, depth),
uploads vertex buffer, draws instanced triangle strips.

sluggrs: Collects `GlyphInstance` instances (screen_rect, em_rect,
band_transform, glyph_data, color), uploads vertex buffer, draws instanced
triangle strips. The vertex format is different but the flow is the same.

```rust
pub struct TextRenderer {
    vertex_buffer: wgpu::Buffer,
    vertex_buffer_size: u64,
    pipeline: wgpu::RenderPipeline,
    instances: Vec<GlyphInstance>,
    instance_count: u32,
}

impl TextRenderer {
    pub fn new(atlas: &mut TextAtlas, device, multisample, depth_stencil) -> Self;

    pub fn prepare_with_depth<'a>(
        &mut self,
        device, queue, encoder, font_system,
        atlas: &mut TextAtlas,
        viewport: &Viewport,
        text_areas: impl IntoIterator<Item = TextArea<'a>>,
        cache: &mut SwashCache,  // accepted for API compat, not used
        metadata_to_depth: impl FnMut(usize) -> f32,
    ) -> Result<(), PrepareError>;

    pub fn prepare<'a>(
        &mut self,
        device, queue, encoder, font_system,
        atlas: &mut TextAtlas,
        viewport: &Viewport,
        text_areas: impl IntoIterator<Item = TextArea<'a>>,
        cache: &mut SwashCache,
    ) -> Result<(), PrepareError>;
    // Delegates to prepare_with_depth with zero_depth

    pub fn render(
        &self,
        atlas: &TextAtlas,
        viewport: &Viewport,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> Result<(), RenderError>;
}
```

### Re-exports

cryoglyph re-exports most of `cosmic_text` for convenience. `text.rs` uses:

- `cosmic_text::Buffer` (via `cryoglyph::Buffer`)
- `cosmic_text::SwashCache` (via `cryoglyph::SwashCache`)
- `cosmic_text::Color` (via `cryoglyph::Color`)
- `cosmic_text::FontSystem` (via `cryoglyph::FontSystem`)

sluggrs must re-export these identically:

```rust
pub use cosmic_text::{
    self, Buffer, CacheKey, Color, FontSystem, SwashCache,
    // ... other types text.rs may reference transitively
};
```

## The prepare() Hot Path

This is where the real work happens. For each `TextArea`:

### 1. Walk layout runs

```
for run in buffer.layout_runs() {
    for glyph in run.glyphs {
        // process each glyph
    }
}
```

Same as cryoglyph — cosmic_text does the shaping/layout.

### 2. Classify and cache glyph

```rust
let key = GlyphKey {
    font_id: glyph.font_id,
    glyph_id: glyph.glyph_id,
    font_weight: glyph.font_weight,
    cache_key_flags: glyph.cache_key_flags,
};

if !atlas.glyph_cache.contains_key(&key) {
    let font_data = font_system.get_font(glyph.font_id)?.data();
    match extract_outline(font_data, glyph.glyph_id) {
        Some(outline) => {
            let gpu_outline = prepare_outline(&outline);
            let bands = build_bands(&gpu_outline, ...);
            atlas.upload_glyph(device, queue, &gpu_outline, &bands);
        }
        None => {
            // Non-vector glyph (emoji, bitmap font).
            // Route to bitmap fallback (see Blocker 3).
            atlas.mark_non_vector(key);
        }
    }
}
```

### 3. Build vertex instance

For each visible vector glyph, compute from `GlyphEntry`:

- **screen_rect**: `entry.bounds` scaled by `font_size / units_per_em`,
  positioned by cosmic_text layout (glyph.x, run.line_y) + TextArea
  (left, top, scale)
- **em_rect**: `entry.bounds` directly
- **band_transform**: `entry.band_transform` directly
- **glyph_data**: `[entry.band_offset, 0, entry.band_max_x, entry.band_max_y]`
- **color**: `glyph.color_opt.unwrap_or(text_area.default_color)`
- **depth**: from `metadata_to_depth(glyph.metadata)`

### 4. Upload and draw

Upload instance buffer via staging belt (same pattern as cryoglyph).
Set pipeline, bind groups (atlas textures + viewport uniform), vertex
buffer. Draw instanced triangle strips: 4 vertices × instance_count.

## Changes to iced

### `iced/Cargo.toml` (workspace)

```diff
-cryoglyph = { git = "https://github.com/iced-rs/cryoglyph.git", rev = "1d68895..." }
+sluggrs = { path = "../../../sluggrs" }  # or git dep
```

### `iced/wgpu/Cargo.toml`

```diff
-cryoglyph.workspace = true
+sluggrs.workspace = true
```

### `iced/wgpu/src/text.rs`

This is NOT a mechanical find-replace. While the public type names match,
the semantic differences require targeted changes in `text.rs`.

#### Behavioral differences from cryoglyph

| Behavior | cryoglyph | sluggrs | text.rs impact |
|----------|-----------|---------|----------------|
| **Namespace** | `cryoglyph::*` | `sluggrs::*` | Mechanical rename |
| **SwashCache** | Rasterizes glyphs | Accepted, unused | No change needed; wasted alloc is negligible |
| **ColorMode** | Controls sRGB texture format | Accepted, ignored | No change; may need revisiting for color correctness |
| **trim()** | Clears usage sets, retains atlas | Clears usage sets, retains cache | Compatible — same external behavior |
| **Atlas invalidation** | group.version increments on trim when atlas is shared | Same mechanism applies | Preserved — uploads check group_version |
| **PrepareError::AtlasFull** | Bitmap atlas hit max texture size | Curve/band textures hit device limits | Swallowed at `text.rs:358` — same behavior |
| **Per-glyph clipping** | Crops glyph quads against TextBounds | Relies on scissor rect | See Clip Bounds section |
| **prepare_with_depth** | Supports depth metadata per glyph | Same signature, depth passed through | Compatible |

#### Preserved text.rs assumptions

These iced behaviors are maintained by sluggrs:

1. **Group versioning** (`text.rs:163`, `text.rs:247`): Atlas trim
   increments group version, forcing re-prepare of uploads. sluggrs
   preserves this because trim changes texture contents (even if it
   only clears the usage set, regrown textures after eviction would
   invalidate old offsets).

2. **Scissor rect per batch** (`text.rs:397`): Always set before any
   render call. sluggrs relies on this for clipping.

3. **AtlasFull swallowed** (`text.rs:358`): The pipeline gracefully
   degrades by skipping the batch. Same behavior with sluggrs since
   our textures are harder to fill.

4. **Multiple TextRenderer instances** (`text.rs:301`, `text.rs:332`):
   State maintains a Vec of renderers, one per prepare layer. Each
   renderer has its own vertex buffer. Compatible with sluggrs.

## Clip Bounds

**Design decision**: sluggrs relies on the scissor rect for clipping,
not per-glyph clip testing.

cryoglyph clips individual glyph quads against `TextBounds` and adjusts
atlas UVs to crop partially visible glyphs. sluggrs renders full glyph
quads and depends on the scissor rect that `text.rs:397` sets via
`render_pass.set_scissor_rect(...)`.

This is valid because:
1. `State::render()` always sets a scissor rect before drawing
   (`text.rs:397`)
2. GPU scissor clipping is per-fragment and correctly clips glyph quads
3. Storage-based cached renders go through the same render pass with
   the same scissor

**Risk**: Oversized glyph quads increase fragment shader invocations
for pixels outside the visible area. For most text this is negligible
(quads are small). For pathological cases (huge glyphs with tight clip
bounds), this wastes GPU work. If this becomes measurable, add early-out
bounds checking in prepare() to skip fully off-screen glyphs.

**No UV adjustment needed**: Unlike cryoglyph, we don't need to crop
texture coordinates because our fragment shader evaluates curves
analytically — there are no UVs to adjust.

## Phasing

### Phase 0: Spike (pre-implementation)

Resolve Blockers 1 and 2 with working code:
- Extract a glyph outline from cosmic_text's font system via skrifa
- Determine the correct GlyphKey fields (font_weight, cache_key_flags,
  variation state)
- Handle FAKE_ITALIC if needed (shear transform on extracted curves)
- Produce a test that exercises the full path from `LayoutGlyph` to
  `GpuOutline`

### Phase A: API skeleton

Create the sluggrs API types with the correct signatures. Implement
trivial stubs (prepare does nothing, render draws nothing). Swap the
dependency in iced and verify it compiles and runs (with no visible text).

### Phase B: Outline extraction in prepare

Implement the font data access path. Extract outlines via skrifa in
prepare(). Build GpuOutline + bands. Upload to textures. Don't draw yet.
Verify glyph cache population with logging.

### Phase C: Vertex packing and rendering

Build GlyphInstance data from layout runs + cached glyph data. Upload
vertex buffer. Wire up the render pipeline with Slug shader. First
visual output.

### Phase D: Non-vector fallback

Implement Blocker 3 — bitmap fallback for emoji and non-vector glyphs.
This must be complete before the iced dependency swap ships.

### Phase E: Polish

Handle edge cases (empty buffers, zero-size glyphs, missing outlines).
Texture growth. Eviction under pressure. Performance tuning.
