# Emoji / Non-Vector Glyph Fallback Spec

## Problem

sluggrs cannot render bitmap-only glyphs (color emoji, bitmap fonts).
These glyphs have no vector outline — `extract_outline()` returns `None`.
Currently they are silently skipped, producing invisible text. This is
a user-visible bug that must be fixed before the iced integration ships.

## Background: How the current stack handles it

### cosmic_text

SwashCache rasterizes glyphs via swash with a priority chain:

```
Source::ColorOutline(0)   → COLR/CPAL tables (vector color glyphs)
Source::ColorBitmap(...)  → CBDT/SBIX tables (bitmap emoji)
Source::Outline           → glyf/CFF tables (regular vector outlines)
```

The result is a `SwashImage` with:
- `Content::Color` → RGBA pixels (4 bytes/pixel) — emoji, color glyphs
- `Content::Mask` → alpha pixels (1 byte/pixel) — regular text
- placement coordinates (top, left, width, height)

### cryoglyph

Handles both paths transparently:
- Two atlas textures: `color_atlas` (Rgba8) and `mask_atlas` (R8)
- etagere packer allocates space in the appropriate atlas
- Fragment shader uses `content_type` uniform to pick blend mode
- The caller (iced's text.rs) doesn't know or care which path a glyph takes

### sluggrs currently

- Tries `extract_outline()` → works for vector glyphs
- Returns `None` for bitmap-only glyphs → marks as `NON_VECTOR_GLYPH`
- Vertex packing skips non-vector entries → glyph disappears

## Options

### Option A: Implement bitmap atlas in sluggrs

Add a third texture (Rgba8) to TextAtlas for bitmap glyphs, with a
simple packer (etagere or manual). When `extract_outline()` returns
None, fall back to `SwashCache::get_image_uncached()` and pack the
bitmap into the atlas.

**Shader changes**: The fragment shader needs a branch — curve evaluation
for vector glyphs, texture sampling for bitmap glyphs. This could be
signaled via a flag in `glyph_data` (e.g. `band_offset == u32::MAX`
means "sample the bitmap atlas instead").

**Vertex format**: Bitmap glyphs need different per-instance data:
atlas UV coordinates instead of em_rect/band_transform/glyph_data.
Could reuse the same fields with different semantics when the flag
is set, or use a union-style layout.

**New dependencies**: etagere (or manual packer), a sampler in the
bind group.

**Pros**:
- Self-contained — sluggrs handles all text rendering
- Single crate dependency for iced
- Can optimize the bitmap path independently
- Full control over atlas management, growth, eviction

**Cons**:
- 500-800 lines of new code (packer, atlas texture, shader branch)
- Shader branching may impact performance (divergent warp)
- Duplicates work already done well in cryoglyph
- Bitmap cache is resolution-dependent (unlike our vector cache),
  adding complexity to cache key and eviction logic
- Need to handle the SwashCache parameter properly (currently ignored)

**Maintenance burden**: Medium. The bitmap atlas is conceptually simple
but has its own set of edge cases (atlas growth, eviction under
pressure, subpixel positioning for bitmap glyphs).

### Option B: Delegate bitmap glyphs to cryoglyph

Keep cryoglyph in iced's workspace (it's already there). During
`prepare()`, split glyphs into two lists: vector glyphs (sluggrs)
and bitmap glyphs (cryoglyph). In `render()`, issue two draw calls.

**Implementation**: sluggrs's TextRenderer holds an optional
cryoglyph::TextRenderer internally. When a non-vector glyph is
detected, it's routed to cryoglyph's prepare path. The render
call draws sluggrs glyphs first, then cryoglyph glyphs.

**No shader changes**: cryoglyph handles its own pipeline, shaders,
and atlas textures entirely.

**Pros**:
- Minimal new code (~100 lines of routing logic)
- Proven bitmap rendering (cryoglyph is battle-tested)
- No shader modifications
- Bitmap cache management already handled
- SwashCache is already wired through the API

**Cons**:
- cryoglyph becomes a real dependency (currently just in workspace)
- Two render pipelines active simultaneously
- Two sets of GPU resources (cryoglyph's atlas textures + sluggrs's
  curve/band textures)
- Version coupling — cryoglyph and sluggrs must agree on wgpu version
- The cryoglyph Cache, TextAtlas, and Viewport must be initialized
  and maintained alongside sluggrs's equivalents
- cryoglyph's TextArea/TextBounds types are identical to ours but
  technically separate types — need conversion or shared definition

**Maintenance burden**: Low code, but dependency coupling is a long-term
risk.

### Option C: Delegate at the iced level (text.rs)

Don't change sluggrs at all. Instead, modify iced's `text.rs` to
maintain both a sluggrs renderer and a cryoglyph renderer. Route
each glyph to the appropriate renderer based on whether it has a
vector outline.

**Pros**:
- sluggrs stays focused on vector rendering only
- Clean separation of concerns
- iced already manages the cryoglyph dependency
- No changes to sluggrs's API or shader

**Cons**:
- Requires modifying iced's text.rs more substantially
- Detection logic ("does this glyph have a vector outline?") must
  happen at the iced level, which doesn't have easy access to skrifa
- Each glyph needs to be classified before routing, adding overhead
  to the prepare path
- Two full text rendering pipelines in iced's render pass
- Makes the iced fork harder to rebase

## Recommendation

**Option C for now, Option A as a future goal.**

Rationale:

1. **Option C is the fastest to ship.** The iced fork already has
   cryoglyph in the workspace. We can keep cryoglyph active alongside
   sluggrs with minimal text.rs changes. Non-vector glyphs go through
   the existing proven path.

2. **The detection problem is solvable.** sluggrs can export a function
   like `has_vector_outline(font_data, face_index, glyph_id)` that
   iced's text.rs calls to decide the routing. This is a cheap check
   (just attempt outline extraction, no curve building).

3. **Option A is the right long-term answer** but it's a substantial
   investment that doesn't need to block the initial integration. Once
   sluggrs is stable in production, adding a bitmap atlas is a natural
   next step that eliminates the cryoglyph dependency entirely.

4. **Option B is the worst of both worlds.** It couples sluggrs to
   cryoglyph at the library level, which is harder to undo than
   coupling at the integration level (Option C).

## Option C Implementation Sketch

### sluggrs changes

Add a detection function:

```rust
/// Check whether a glyph has a vector outline that sluggrs can render.
/// Returns false for bitmap-only glyphs (emoji, bitmap fonts).
pub fn has_vector_outline(
    font_data: &[u8],
    face_index: u32,
    glyph_id: u16,
) -> bool {
    extract_outline(font_data, face_index, glyph_id, &[]).is_some()
}
```

Export it from lib.rs.

### iced text.rs changes

In the `prepare()` function, classify each TextArea's glyphs and
split into two batches:

```rust
// Rough sketch — actual implementation needs more thought
let mut sluggrs_areas = vec![];  // TextAreas with only vector glyphs
let mut cryoglyph_areas = vec![]; // TextAreas with only bitmap glyphs

// For mixed TextAreas, we'd need to render the same TextArea through
// both renderers, each skipping glyphs it can't handle.
```

**Problem**: This sketch is oversimplified. A single TextArea can
contain both vector and bitmap glyphs (e.g. "Hello 👋 world"). We
can't split at the TextArea level — we'd need glyph-level routing.

**Better approach**: Render every TextArea through both renderers.
sluggrs skips non-vector glyphs (it already does). cryoglyph renders
all glyphs. The visual result is correct: sluggrs draws vector text,
cryoglyph draws everything but its vector glyphs are drawn on top
of (or behind) sluggrs's.

**Even better**: Render every TextArea through both renderers, but
have cryoglyph skip vector glyphs. This avoids double-rendering.
This requires cryoglyph to know which glyphs to skip, which brings
us back to the detection problem.

**Simplest correct approach**: Render all text through sluggrs (which
skips non-vector glyphs) AND all text through cryoglyph (which renders
everything). Since cryoglyph renders behind sluggrs, the vector glyphs
from cryoglyph are hidden behind sluggrs's higher-quality vector
rendering. Emoji from cryoglyph shows through because sluggrs
doesn't draw anything there.

This works if:
- Both renderers use the same positions (they do — same TextArea)
- sluggrs draws on top of cryoglyph (draw order in render pass)
- Alpha blending doesn't cause artifacts for double-drawn vector
  glyphs (it shouldn't — sluggrs's output is opaque in the glyph
  interior, and both renderers produce the same shape)

**This might be the pragmatic answer**: zero routing logic, both
renderers see all text, sluggrs handles what it can, cryoglyph
catches what falls through. Double rendering of vector glyphs
wastes some GPU work but produces correct output.

## Open Questions

1. **Draw order and blending**: If cryoglyph draws first and sluggrs
   draws on top, does alpha blending produce correct results for
   overlapping vector glyphs? Both produce similar but not identical
   coverage — could cause visible fringing.

2. **Performance**: Rendering all text through both pipelines doubles
   the GPU work. For text-heavy applications (email client) this may
   be measurable. Profiling needed.

3. **COLR/CPAL vector emoji**: Some emoji fonts use COLR/CPAL tables
   which define emoji as layered vector outlines (not bitmaps). swash
   can rasterize these to RGBA bitmaps, but skrifa can also extract
   the outlines. Should sluggrs try to render these natively? This is
   complex (layered colored fills) but would avoid the bitmap path for
   modern emoji fonts.
