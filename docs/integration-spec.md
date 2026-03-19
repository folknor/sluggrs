# sluggrs Integration Spec: Replacing cryoglyph in iced

## Overview

Replace cryoglyph (bitmap atlas text renderer) with sluggrs (GPU curve
evaluation) in iced's wgpu backend. The integration point is a single file:
`iced/wgpu/src/text.rs` (~650 lines).

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
outline cache keyed by `(font_id, glyph_id)` (resolution-independent — a
key advantage), and a bind group referencing both textures.

The curve texture stores control points. The band texture stores the
acceleration structure. Both grow as new glyphs are encountered. The band
texture is always `BAND_TEXTURE_WIDTH` wide and grows in height.

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

    // Glyph cache: (font_id, glyph_id) → location in textures
    glyph_cache: HashMap<GlyphKey, GlyphEntry>,
    // Current write cursors
    curve_cursor: u32,
    band_cursor: u32,
    // Pending texture data to upload
    curve_data: Vec<[f32; 4]>,
    band_data: Vec<[u32; 4]>,
}
```

**Key difference from cryoglyph**: Glyph data is resolution-independent.
A glyph cached at 12px works identically at 72px — only the vertex
attributes change. This eliminates re-rasterization on scale changes.

```rust
impl TextAtlas {
    pub fn new(device, queue, cache, format) -> Self;
    pub fn with_color_mode(device, queue, cache, format, color_mode) -> Self;
    pub fn trim(&mut self);
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
        cache: &mut SwashCache,  // ignored — we use skrifa, but keep for API compat
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

### 2. Look up or extract glyph outline

Cache key: `(font_id, glyph_id)` — NOT physical position.
This is a major win: one outline serves all sizes.

```
let key = GlyphKey { font_id, glyph_id };
if !atlas.glyph_cache.contains_key(&key) {
    let outline = extract_outline(font_data, glyph_id);
    let gpu_outline = prepare_outline(&outline);
    let bands = build_bands(&gpu_outline, ...);
    // Upload curve + band data to textures
    atlas.upload_glyph(key, &gpu_outline, &bands);
}
```

### 3. Build vertex instance

For each visible glyph, compute:

- **screen_rect**: position and size in screen pixels (from cosmic_text
  layout + TextArea position/scale)
- **em_rect**: the glyph's bounding box in em-space (from GpuOutline.bounds)
- **band_transform**: scale + offset to map em-space → band index
  (from BandData)
- **glyph_data**: packed texture offsets for band lookup
  (from GlyphEntry in atlas cache)
- **color**: RGBA from glyph.color_opt or TextArea.default_color

### 4. Upload and draw

Same as cryoglyph: upload vertex buffer, set pipeline + bind groups, draw
instanced triangle strips.

## Font Data Access

cryoglyph uses `SwashCache::get_image_uncached(font_system, cache_key)` to
rasterize glyphs. We need raw font bytes to extract outlines via skrifa.

`cosmic_text::FontSystem` provides access to font data through its
`fontdb` database. We can get font bytes via:

```rust
let font_id = glyph.font_id;  // from cosmic_text::LayoutGlyph
// cosmic_text::FontSystem -> fontdb::Database
// fontdb::Database::face_source(id) -> Source::Binary(Arc<[u8]>)
```

This needs investigation — the exact API path from a `LayoutGlyph`'s
font reference to raw font bytes through cosmic_text's font system.

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

Mechanical find-replace of `cryoglyph::` → `sluggrs::` if we match the
API exactly. The only expected difference is that `SwashCache::new()` is
still called (for API compat) but ignored internally during prepare.

## Open Questions

### 1. Font byte access

How to get from `cosmic_text::FontSystem` + `LayoutGlyph::font_id` to
raw `&[u8]` font data for skrifa. This is the critical path for outline
extraction. Needs code-level investigation of cosmic_text internals.

### 2. Glyph cache key

cosmic_text's `CacheKey` includes subpixel position. Our outlines are
resolution-independent, so we cache by `(font_id, glyph_id)` only. But
the vertex attributes (screen position) still need the subpixel offset
from cosmic_text's layout. Need to verify this doesn't cause issues.

### 3. Texture growth strategy

cryoglyph grows its atlas by 2x when full. Our curve texture is a single
row that grows in width; the band texture is fixed-width (4096) and grows
in height. Need a growth strategy that doesn't invalidate existing data
(append-only within a frame, repack on trim).

### 4. Color emoji

Slug can't render bitmap emoji. For now, these glyphs will be silently
skipped (no outline to extract). A bitmap fallback is tracked in TODO.md.

### 5. Clip bounds

cryoglyph does per-glyph clip testing against TextBounds and crops atlas
UVs. We render full glyph quads and rely on the scissor rect that
`text.rs` already sets. This should be equivalent but needs verification
that no glyphs leak outside clip bounds.

### 6. The SwashCache parameter

`text.rs` passes `&mut SwashCache::new()` to every prepare call. Since
we don't rasterize, we accept but ignore it. We could also newtype it
to avoid the allocation, but the cost is negligible.

## Phasing

### Phase A: API skeleton

Create the sluggrs API types with the correct signatures. Implement
trivial stubs (prepare does nothing, render draws nothing). Swap the
dependency in iced and verify it compiles.

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
Texture growth. Trim/eviction. Performance tuning.
