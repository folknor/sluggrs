# TODO

## Done

- [x] Centralize shared shader constants (BAND_TEXTURE_WIDTH exported from lib.rs)
- [x] Update investigation doc with comma bug resolution
- [x] Update design doc to reflect sluggrs naming
- [x] Build the crate API surface (Cache, TextAtlas, TextRenderer, Viewport)
- [x] Wire into iced's `iced_wgpu/src/text.rs` as a cryoglyph replacement
- [x] FAKE_ITALIC shear transform support
- [x] Visual testing in ratatoskr — text renders correctly
- [x] Dilation — 1px quad expansion for smooth AA at edges
- [x] Variable font support (face index + weight axis variation)
- [x] Multi-row curve texture (avoids exceeding device texture limits)
- [x] Band offset 2D conversion (linear → wrapped texture coordinates)
- [x] Solver threshold raised to handle near-degenerate perturbed curves

## Before shipping

- [ ] Non-vector glyph fallback — color emoji and bitmap-only fonts currently
  produce no output. Required before the iced swap ships. See integration
  spec (docs/integration-spec.md) for strategy options.
- [ ] Trim/eviction — currently trim() is a no-op. Implement usage tracking
  and LRU eviction to match cryoglyph's frame-boundary semantics.
- [ ] Depth plumbing — prepare_with_depth accepts the callback but discards
  the depth value. Any iced path relying on layered text depth will render
  incorrectly. Either implement or document as unsupported.

## Polish

- [ ] Band texture width configurable rather than hardcoded 4096
- [ ] Shader constant centralization between simple_shader.wgsl and shader.wgsl
- [ ] Performance profiling against cryoglyph
- [ ] ColorMode handling (currently stubbed — Slug renders in linear space)
- [ ] Texture growth under heavy load — stress test with CJK, mixed fonts
