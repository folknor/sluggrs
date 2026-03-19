# sluggrs

GPU vector text rendering using the [Slug algorithm](https://terathon.com/blog/decade-slug.html). Evaluates quadratic bezier curves per-pixel in fragment shaders — resolution-independent, no texture atlas needed.

Drop-in replacement for [cryoglyph](https://github.com/iced-rs/cryoglyph) in the [iced](https://github.com/iced-rs/iced) GUI framework's wgpu text rendering pipeline.

## How it works

Traditional text renderers (cryoglyph, glyphon) rasterize glyphs to bitmaps on the CPU, pack them into a GPU texture atlas, and draw textured quads. This works but has downsides: atlas memory pressure, re-rasterization on scale changes, and pixel-locked rendering.

sluggrs takes a different approach: glyph outlines (quadratic bezier curves) are uploaded to the GPU as control point data. The fragment shader evaluates these curves per-pixel to determine coverage, producing resolution-independent rendering with zero atlas overhead for vector glyphs.

Key advantages:
- **Resolution independent** — one cached outline serves all sizes
- **No atlas pressure** — vector data is ~100x smaller than bitmaps
- **Clean at any scale** — no rasterization artifacts at fractional DPI
- **Same API** — matches cryoglyph's interface for iced integration

## Status

Work in progress. The core rendering pipeline is functional:
- Outline extraction from any font via [skrifa](https://github.com/googlefonts/fontations)
- Band acceleration structure for efficient curve lookup
- WGSL fragment shader with full Slug curve evaluation
- cryoglyph-compatible API (Cache, TextAtlas, TextRenderer, Viewport)
- Wired into iced's `text.rs` via [forked iced](https://github.com/folknor/iced/tree/sluggrs)

Not yet implemented: dilation (AA at small sizes), non-vector glyph fallback (emoji), trim/eviction.

## License

MIT OR Apache-2.0 OR Zlib
