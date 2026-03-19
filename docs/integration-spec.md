# sluggrs Integration Spec: Replacing cryoglyph in iced

## Overview

Replace cryoglyph (bitmap atlas text renderer) with sluggrs (GPU curve
evaluation) in iced's wgpu backend. The integration point is a single file:
`iced/wgpu/src/text.rs` (~650 lines).

## Blockers (resolve before implementation)

### Blocker 1: Font byte access

**Status**: Unresolved — requires a spike.

`prepare()` needs raw font bytes for skrifa outline extraction. The path
from `cosmic_text::LayoutGlyph::font_id` → `&[u8]` font data through
cosmic_text's `FontSystem` must be validated with working code before
Phase B can start. Candidate API path:

```
FontSystem::db() → fontdb::Database
fontdb::Database::face_source(font_id) → Source::Binary(Arc<[u8]>)
```

If this doesn't work cleanly, alternatives include:
- Maintaining a parallel font cache keyed by font_id
- Using cosmic_text's swash integration to get font refs

This spike should produce a standalone test that extracts an outline from
a glyph returned by cosmic_text layout.

### Blocker 2: Glyph cache key

The glyph cache key must uniquely identify a glyph's outline geometry.
`(font_id, glyph_id)` alone is NOT sufficient because:

- **Variable fonts**: The same font_id + glyph_id produces different
  outlines at different variation coordinates (weight, width, etc.)
- **Font collections**: A font_id might map to different faces within
  a collection

The key must capture everything that affects outline shape:

```rust
#[derive(Hash, Eq, PartialEq)]
pub struct GlyphKey {
    /// cosmic_text font identifier — must uniquely identify the face
    /// within the font database, including collection index
    font_id: cosmic_text::fontdb::ID,
    /// Glyph index within the font
    glyph_id: u16,
    /// Normalized variation coordinates, if any. For static fonts this
    /// is empty. For variable fonts, this captures the specific instance.
    /// Quantized to avoid floating-point hash instability.
    variation_hash: u64,
}
```

The exact fields depend on what cosmic_text exposes. The spike in Blocker 1
should also determine what variation state is available at layout time.

**Important**: The cache key deliberately excludes size, position, and
subpixel offset. Outlines are resolution-independent — vertex attributes
handle the per-instance transform. This is the core architectural advantage
over cryoglyph's `CacheKey` which includes physical position.

### Blocker 3: Non-vector glyph fallback

Slug cannot render bitmap glyphs (color emoji, bitmap-only fonts).
Silently skipping these is not acceptable — it produces user-visible
missing text.

**Required strategy for Phase 1**: Detect non-vector glyphs (no outline
data from skrifa) and fall back to a bitmap path. Options:

1. **Hybrid rendering**: Keep a minimal cryoglyph instance for bitmap-only
   glyphs. Detect during prepare, route to the appropriate renderer.
   Two draw calls per text batch (one Slug, one bitmap).

2. **Skip in Phase 1, hard error in debug**: Render a visible placeholder
   (e.g. missing-glyph box) for non-vector glyphs so the omission is
   obvious during testing. Implement bitmap fallback before any user-facing
   release.

3. **cryoglyph as dependency**: sluggrs depends on cryoglyph for bitmap
   fallback. Heavy, but zero new code for the bitmap path.

Recommendation: Option 2 for Phase 1 (visible placeholder), Option 1 for
production. The spec must not treat this as optional.

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
    /// The shader reads band headers starting at (band_offset, 0) and
    /// wrapping at BAND_TEXTURE_WIDTH.
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

On `trim()`: clear the glyph_cache, reset cursors to 0. Next frame
repopulates on demand. This is simpler than LRU eviction and acceptable
because outline extraction + band building is fast (no rasterization).

Rebind: after any texture reallocation, recreate the bind group that
references both texture views. The `TextRenderer` holds a pipeline (which
references the bind group layout, not the bind group itself), so it
remains valid. The bind group is set per-draw in `render()`.

```rust
impl TextAtlas {
    pub fn new(device, queue, cache, format) -> Self;
    pub fn with_color_mode(device, queue, cache, format, color_mode) -> Self;
    pub fn trim(&mut self);

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

    pub fn prepare<'a>(
        &mut self,
        device, queue, encoder, font_system,
        atlas: &mut TextAtlas,
        viewport: &Viewport,
        text_areas: impl IntoIterator<Item = TextArea<'a>>,
        cache: &mut SwashCache,  // accepted for API compat, not used
    ) -> Result<(), PrepareError>;

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
let key = GlyphKey::from_glyph(font_system, &glyph);

// Check if this glyph has a vector outline
if !atlas.glyph_cache.contains_key(&key) {
    match extract_outline(font_data, glyph_id) {
        Some(outline) => {
            let gpu_outline = prepare_outline(&outline);
            let bands = build_bands(&gpu_outline, ...);
            atlas.upload_glyph(device, queue, &gpu_outline, &bands);
        }
        None => {
            // Non-vector glyph (emoji, bitmap font).
            // Phase 1: render placeholder box.
            // Production: route to bitmap fallback.
            atlas.mark_non_vector(key);
            continue;
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
the semantic differences require targeted changes:

- **Namespace swap**: `cryoglyph::` → `sluggrs::` for type references.
  This part is mechanical.

- **SwashCache usage**: cryoglyph creates `SwashCache::new()` per prepare
  call and passes it for glyph rasterization. sluggrs accepts the parameter
  for API compat but doesn't use it. No code change needed, but the
  allocation is wasted. Acceptable for now.

- **ColorMode**: cryoglyph uses `TextAtlas::with_color_mode()` to control
  sRGB handling. sluggrs accepts the parameter but ignores it (Slug renders
  in linear space). May need revisiting for correct color output.

- **Atlas trim semantics**: cryoglyph's `trim()` clears per-frame glyph
  usage sets for LRU eviction. sluggrs's `trim()` clears the entire cache
  and resets texture cursors. This means glyph re-extraction on the next
  frame after trim, which is fast but worth profiling.

- **Error handling**: `PrepareError::AtlasFull` in cryoglyph means the
  bitmap atlas hit max texture size. In sluggrs this maps to curve/band
  textures hitting device limits, which is much harder to reach (vector
  data is dramatically smaller than bitmaps).

## Clip Bounds

**Design decision**: sluggrs relies on the scissor rect for clipping,
not per-glyph clip testing.

cryoglyph clips individual glyph quads against `TextBounds` and adjusts
atlas UVs to crop partially visible glyphs. sluggrs renders full glyph
quads and depends on the scissor rect that `text.rs:397` sets via
`render_pass.set_scissor_rect(...)`.

This is valid **if and only if**:
1. Every text render pass sets a scissor rect before drawing
2. Glyph quads that extend beyond the scissor are correctly clipped by
   the GPU (guaranteed by the spec)
3. No batching or layering assumptions break when quads extend beyond
   their logical bounds

Verification needed: review every call to `State::render()` and
`Storage::get()` to confirm scissor rects are always set. Check that
depth/stencil (if used) doesn't interact badly with oversized quads.

If this proves problematic, add per-glyph bounds checking in prepare()
to skip off-screen glyphs entirely (cheaper than cryoglyph's UV cropping
since we don't need to adjust texture coordinates).

## Phasing

### Phase 0: Spike (pre-implementation)

Resolve Blockers 1 and 2 with working code:
- Extract a glyph outline from cosmic_text's font system via skrifa
- Determine the correct GlyphKey fields for variable fonts
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

### Phase D: Polish

Handle edge cases (empty buffers, zero-size glyphs, missing outlines).
Texture growth. Trim/eviction. Performance tuning. Non-vector glyph
placeholders.
