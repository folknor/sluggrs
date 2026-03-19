# sluggrs: Slug-Based Text Renderer for iced

> **Note**: This design doc was written before the PoC. Some details
> (module names, API signatures) may have evolved. See `src/` for the
> current implementation and `TODO.md` for outstanding work.

Design doc for replacing cryoglyph (atlas-based) with a Slug-based (GPU curve evaluation) text renderer in the iced/wgpu pipeline.

## Background

iced currently renders text by rasterizing glyphs to bitmaps on the CPU (via swash/zeno), packing them into a GPU texture atlas (via cryoglyph/etagere), and drawing textured quads. This produces acceptable results but has known issues:

- Slightly off rendering at fractional scale factors
- Atlas memory pressure with many unique glyphs (CJK, emoji)
- Re-rasterization on atlas eviction or scale change
- No resolution independence — cached glyphs are pixel-locked

Slug evaluates quadratic bezier curves **per-pixel in the fragment shader**. Glyph outlines go to the GPU as control point data, not bitmaps. This gives resolution independence, zero atlas overhead for vector glyphs, and clean rendering at any scale/DPI.

## Architecture Overview

```
cosmic-text                      (keep — shaping, bidi, line-breaking)
    │
    │  produces: glyph IDs + positions + font refs
    ▼
sluggrs                       (new crate — replaces cryoglyph)
    │
    ├── outline extraction       (skrifa — read bezier curves from font)
    ├── band builder             (CPU — spatial acceleration structure)
    ├── curve/band textures      (GPU — two wgpu textures)
    ├── vertex packing           (CPU — 5×vec4 per glyph vertex)
    ├── WGSL shaders             (GPU — translated from Slug HLSL reference)
    └── render pipeline          (wgpu — draw calls into render pass)
    │
    ▼
iced_wgpu/src/text.rs            (modified — calls sluggrs instead of cryoglyph)
```

## Integration into iced

### Dependency chain (current → proposed)

```
Current:
  app → squidowl/iced (git rev b201e4f)
          → iced_wgpu → cryoglyph (git dep) → wgpu 28

Proposed:
  app → our-fork/iced (git)
          → iced_wgpu → sluggrs (git or path dep) → wgpu 28
```

### What changes in iced

**One file**: `iced_wgpu/src/text.rs` (~650 lines). This is the sole integration point where cryoglyph is imported and used. The rest of iced_wgpu is unaffected.

**Workspace Cargo.toml**: Swap cryoglyph dependency for sluggrs.

### The contract text.rs expects

From studying cryoglyph's public API and how text.rs calls it, the replacement must provide:

```rust
// Shared GPU state (shader modules, bind group layouts, sampler)
pub struct Cache { .. }
impl Cache {
    pub fn new(device: &wgpu::Device) -> Self;
}

// Viewport uniform buffer (screen resolution)
pub struct Viewport { .. }
impl Viewport {
    pub fn new(device: &wgpu::Device, cache: &Cache) -> Self;
    pub fn update(&mut self, queue: &wgpu::Queue, resolution: Resolution);
}

// Glyph data store (replaces texture atlas)
// In sluggrs this holds curve + band textures instead of bitmap atlases
pub struct TextAtlas { .. }
impl TextAtlas {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, cache: &Cache,
               format: wgpu::TextureFormat) -> Self;
    pub fn trim(&mut self);  // End-of-frame cleanup
}

// Per-layer renderer
pub struct TextRenderer { .. }
impl TextRenderer {
    pub fn new(atlas: &mut TextAtlas, device: &wgpu::Device,
               multisample: wgpu::MultisampleState,
               depth_stencil: Option<wgpu::DepthStencilState>) -> Self;

    pub fn prepare_with_depth<'a>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        font_system: &mut cosmic_text::FontSystem,
        atlas: &mut TextAtlas,
        viewport: &Viewport,
        text_areas: impl IntoIterator<Item = TextArea<'a>>,
        cache: &mut GlyphCache,  // was SwashCache — now our outline cache
        metadata_to_depth: impl FnMut(usize) -> f32,
    ) -> Result<(), PrepareError>;

    pub fn render(
        &self,
        atlas: &TextAtlas,
        viewport: &Viewport,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> Result<(), RenderError>;
}

// Input type — same shape as cryoglyph::TextArea
pub struct TextArea<'a> {
    pub buffer: &'a cosmic_text::Buffer,
    pub left: f32,
    pub top: f32,
    pub scale: f32,
    pub bounds: TextBounds,
    pub default_color: Color,
}
```

The key difference is internal: `prepare_with_depth` no longer rasterizes glyphs to bitmaps. Instead it:
1. Looks up glyph outlines (quadratic beziers) via skrifa
2. Builds/caches curve + band data for each glyph
3. Packs vertex attributes (position, em-coords, Jacobian, band transform, color)
4. Uploads curve/band textures if new glyphs appeared

And `render` sets the Slug pipeline + textures and draws instanced quads.

## sluggrs Crate Design

### Dependencies

```toml
[dependencies]
wgpu = "28"
cosmic-text = "0.18"    # for Buffer, LayoutRun, LayoutGlyph types
skrifa = "0.40"          # font outline extraction (already in cosmic-text's dep tree)
etagere = "0.2"          # reuse for packing curves into texture (optional)
lru = "0.16"             # glyph outline cache
rustc-hash = "2"         # fast hashing
```

### Module structure

```
sluggrs/
├── Cargo.toml
└── src/
    ├── lib.rs              Public API (Cache, TextAtlas, TextRenderer, Viewport, etc.)
    ├── outline.rs          Glyph outline extraction from skrifa → quadratic beziers
    ├── band.rs             Band acceleration structure builder
    ├── glyph_store.rs      Curve + band texture management, glyph caching
    ├── prepare.rs          Convert cosmic-text buffers → vertex data
    ├── render.rs           wgpu pipeline setup and draw calls
    ├── vertex.rs           Vertex attribute packing (5×vec4)
    ├── shader.wgsl         Slug shaders translated from HLSL
    └── types.rs            TextArea, TextBounds, Color, Resolution, errors
```

### Outline extraction (outline.rs)

Uses `skrifa` to read glyph outlines. Font files contain cubic and quadratic beziers; Slug operates on quadratics. TrueType fonts are natively quadratic. OpenType/CFF fonts use cubics and need conversion.

```rust
/// Extract quadratic bezier curves for a glyph.
/// Returns control points in em-space coordinates.
pub fn extract_outline(
    font: &skrifa::FontRef,
    glyph_id: GlyphId,
    size: f32,
) -> Option<GlyphOutline> { .. }

pub struct GlyphOutline {
    /// Quadratic bezier curves: each is 3 control points (p1, p2, p3)
    pub curves: Vec<[Vec2; 3]>,
    /// Bounding box in em-space
    pub bounds: Rect,
}
```

**Cubic → quadratic conversion**: When we encounter cubic beziers (CFF/OpenType), subdivide into quadratics. This is a well-known operation — split the cubic at midpoints until the quadratic approximation is within tolerance. The `lyon_geom` crate or a manual implementation can handle this. Most text fonts are TrueType (natively quadratic), so this path is less common.

### Band builder (band.rs)

The band structure is Slug's spatial acceleration — it divides the glyph bounding box into horizontal and vertical strips, each storing which curves intersect it. This avoids testing every curve for every pixel.

```rust
/// Build the band acceleration structure for a glyph's curves.
pub fn build_bands(
    outline: &GlyphOutline,
    band_count_x: u32,  // typically 4–16 depending on glyph complexity
    band_count_y: u32,
) -> BandData { .. }

pub struct BandData {
    /// Per-band curve indices and offsets
    pub entries: Vec<BandEntry>,
    /// Band grid dimensions
    pub band_count: [u32; 2],
    /// Transform from em-space to band index
    pub band_transform: [f32; 4],  // scale_x, scale_y, offset_x, offset_y
}
```

### Glyph store (glyph_store.rs)

Manages the two GPU textures that hold curve and band data for all active glyphs.

**Curve texture** (`Rgba32Float` or `Rg32Float`):
- Stores control points for quadratic beziers
- Each curve = 2 texels: `(p1.x, p1.y, p2.x, p2.y)` and `(p3.x, p3.y, _, _)`
- Append-only within a frame; compacted on trim

**Band texture** (`Rgba32Uint`, width 4096):
- Stores band entries: curve count + offset pairs
- Per-glyph region referenced by vertex attribute `tex.z`

```rust
pub struct GlyphStore {
    curve_texture: wgpu::Texture,
    band_texture: wgpu::Texture,
    // Cache: glyph key → location in textures
    cache: LruCache<GlyphKey, GlyphLocation>,
    // Allocation tracking
    curve_cursor: u32,
    band_cursor: u32,
}

struct GlyphLocation {
    band_origin: [u16; 2],     // where this glyph's band data starts in band texture
    band_max: [u8; 2],         // band grid dimensions
    band_transform: [f32; 4],  // em-space → band index transform
    curve_count: u16,
}
```

**Memory characteristics**: Curve data is *tiny* compared to bitmap atlases. A typical Latin glyph has 10–30 curves = 20–60 texels. A full Latin character set (~200 glyphs) fits in ~12K texels. CJK glyphs are larger (~50–100 curves) but still far smaller than their bitmap equivalents at high DPI.

### Vertex packing (vertex.rs)

Each glyph is rendered as a quad (4 vertices, triangle strip). The Slug vertex shader expects 5 × vec4 attributes per vertex:

| Attribute | Contents | Source |
|-----------|----------|--------|
| `pos` | xy = object-space position, zw = normal | Glyph bounding box corners + edge normals |
| `tex` | xy = em-space coords, z = packed glyph location, w = packed band max + flags | From glyph store allocation |
| `jac` | Inverse Jacobian matrix (2×2) | Computed from object→em-space transform |
| `bnd` | Band scale xy + band offset zw | From band builder |
| `col` | RGBA color | From TextArea default_color or per-glyph color |

```rust
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SlugVertex {
    pub pos: [f32; 4],
    pub tex: [f32; 4],
    pub jac: [f32; 4],
    pub bnd: [f32; 4],
    pub col: [f32; 4],
}
// 80 bytes per vertex, 320 bytes per glyph (4 vertices)
```

### Shader translation (shader.wgsl)

The reference Slug shaders are 375 lines of HLSL. Translation to WGSL is mechanical:

| HLSL | WGSL |
|------|------|
| `float4` | `vec4<f32>` |
| `uint4` | `vec4<u32>` |
| `Texture2D<float4>` | `texture_2d<f32>` |
| `Texture2D<uint4>` | `texture_2d<u32>` |
| `cbuffer` | `@group(0) @binding(N) var<uniform>` |
| `SV_Position` | `@builtin(position)` |
| `fwidth()` | `fwidth()` (available in WGSL) |
| `asfloat()/asuint()` | `bitcast<f32>()/bitcast<u32>()` |
| `nointerpolation` | `@interpolate(flat)` |
| `saturate()` | `clamp(x, 0.0, 1.0)` |
| `tex.Load(loc, 0)` | `textureLoad(tex, loc, 0)` |

Key concerns:
- WGSL requires explicit bind group/binding annotations
- Texture load coordinates may need `vec2<i32>` vs `int3` adjustment
- Bit manipulation syntax differs slightly but all ops are available

### Render pipeline (render.rs)

```rust
impl TextRenderer {
    pub fn render(
        &self,
        atlas: &TextAtlas,  // holds curve + band textures
        viewport: &Viewport,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> Result<(), RenderError> {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &atlas.bind_group, &[]);    // curve + band textures
        pass.set_bind_group(1, &viewport.bind_group, &[]);  // MVP + viewport
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.draw(0..4, 0..self.glyph_count);  // 4 verts per glyph, instanced
        Ok(())
    }
}
```

**Pipeline descriptor**:
- Vertex: SlugVertex layout, 80 bytes stride, instance step mode (4 vertices per glyph via vertex_index)
- Fragment: Alpha blending (premultiplied), writes to target format
- Depth/stencil: Passthrough (same as cryoglyph)
- Multisample: Match iced's settings

## Open Questions

### Color emoji

Color emoji (Apple Color Emoji, Noto Color Emoji) are bitmap-based — they don't have vector outlines. Slug can't render these. Options:

1. **Hybrid approach**: Detect bitmap glyphs, fall back to a small atlas (like cryoglyph) for just those glyphs. This is the pragmatic answer.
2. **SVG emoji**: Some emoji fonts (Noto Color Emoji SVG) have vector outlines. These could potentially use the Slug pipeline, though they involve fills, not just outlines.
3. **Skip for now**: Email clients don't render emoji-heavy content as often as chat apps. Start with vector-only, add bitmap fallback later.

Recommendation: option 1 — keep a minimal bitmap atlas path for color emoji only.

### Cubic bezier conversion quality

CFF/OpenType fonts use cubic beziers. Converting to quadratics introduces approximation error. At typical text sizes this is invisible, but at extreme zoom levels it could matter. The tolerance for subdivision needs tuning — start with 0.1 em-units (well below pixel threshold at any reasonable size).

### Performance characteristics

Slug trades CPU work (atlas rasterization) for GPU work (per-pixel curve evaluation). This is favorable on modern GPUs but could be a concern for:

- **Integrated GPUs**: Lower shader throughput. Need to benchmark.
- **Very dense text**: Email thread list with hundreds of visible lines. Band acceleration helps, but this needs profiling.
- **Overdraw**: Glyph quads overlap. The fragment shader runs for every covered pixel. Depth testing could help if glyphs are opaque backgrounds, but text alpha-blends.

### skrifa API for outline access

Need to verify that `skrifa` exposes per-glyph outlines as individual curve segments we can iterate. The `OutlinePen` trait should work — it provides `move_to`, `line_to`, `quad_to`, `curve_to` callbacks. We'd collect these into our `GlyphOutline` struct.

### Subpixel rendering (LCD)

Slug's reference implementation does grayscale anti-aliasing. Subpixel (LCD) rendering would require running the coverage calculation three times with sub-pixel offsets for R, G, B. This is possible but triples fragment shader cost. Probably not worth it on HiDPI displays where subpixel rendering is unnecessary.

## Phasing

### Phase 1: Proof of concept (standalone)

- Translate Slug HLSL → WGSL
- Build a standalone wgpu app that renders a few glyphs using skrifa + Slug shaders
- Validate visual quality against cryoglyph output
- No iced integration yet

### Phase 2: sluggrs crate

- Implement full crate with the API described above
- Handle Latin + CJK + common Unicode ranges
- Glyph caching, texture management, vertex packing
- Benchmark against cryoglyph

### Phase 3: iced integration

- Fork squidowl/iced
- Swap cryoglyph → sluggrs in workspace Cargo.toml
- Rewrite `iced_wgpu/src/text.rs` integration
- Point ratatoskr app at forked iced
- Validate in the actual email client

### Phase 4: Polish

- Color emoji fallback (bitmap hybrid)
- Performance tuning (band count heuristics, texture sizing)
- Upstream discussion with iced maintainers if results are good
