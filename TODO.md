# TODO

## Done

- [x] Centralize shared shader constants (BAND_TEXTURE_WIDTH exported from lib.rs)
- [x] Update investigation doc with comma bug resolution
- [x] Update design doc to reflect sluggrs naming
- [x] Build the crate API surface (Cache, TextAtlas, TextRenderer, Viewport)
- [x] Wire into iced's `iced_wgpu/src/text.rs` as a cryoglyph replacement
- [x] FAKE_ITALIC shear transform support

## Before shipping

- [ ] Visual testing in ratatoskr — verify text renders correctly
- [ ] Dilation — expand glyph quads by ~1px so the fragment shader has room
  for full AA coverage ramp at edges. Matters at small sizes and thin
  features. The full shader (`src/shader.wgsl`) has dilation support; the
  simplified shader does not.
- [ ] Non-vector glyph fallback — color emoji and bitmap-only fonts currently
  produce no output. Required before the iced swap ships. See integration
  spec for strategy options.
- [ ] Texture growth under load — verify curve/band textures grow correctly
  when rendering large amounts of diverse text (CJK, mixed fonts)
- [ ] Trim/eviction — currently trim() is a no-op. Implement usage tracking
  and LRU eviction to match cryoglyph's frame-boundary semantics.

## Polish

- [ ] Band texture width configurable rather than hardcoded 4096
- [ ] Shader constant centralization between simple_shader.wgsl and shader.wgsl
- [ ] Performance profiling against cryoglyph
- [ ] ColorMode handling (currently stubbed — Slug renders in linear space)
