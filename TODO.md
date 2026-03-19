# TODO

## Pre-integration

- [ ] Centralize shared shader constants (BAND_TEXTURE_WIDTH, texture layout
  assumptions) so `simple_shader.wgsl` and `shader.wgsl` cannot drift
- [ ] Update `docs/slug-glyph-investigation.md` with comma bug resolution
- [ ] Update `docs/slug-glyph-design.md` to reflect `sluggrs` naming

## Integration (Phase 2–3)

- [ ] Implement dilation in vertex packing — expand glyph quads by ~1px so
  the fragment shader has room for full AA coverage ramp at edges. The full
  shader (`src/shader.wgsl`) already has dilation support translated from
  the Slug reference; the simplified shader does not. Dilation matters at
  small sizes and for thin glyph features.
- [ ] Build the crate API surface (Cache, TextAtlas, TextRenderer, Viewport)
  matching the contract in `docs/slug-glyph-design.md`
- [ ] Wire into iced's `iced_wgpu/src/text.rs` as a cryoglyph replacement
- [ ] Color emoji fallback (bitmap hybrid for non-vector glyphs)

## Cleanup

- [ ] Band texture width should eventually be configurable rather than a
  hardcoded 4096 constant shared between Rust and WGSL
