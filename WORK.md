# WORK

Shrink GlyphInstance from 96 to 48 bytes by moving per-glyph constants
(em_rect, band_transform, band_max) into a 5-texel glyph header stored in
the atlas storage buffer, decoded by the vertex shader. Rendering must be
pixel-identical: every value the shader consumes must be bit-identical to
what the old per-instance fields carried.

## Agreed plan (implement exactly this)

### New instance ABI (src/lib.rs, mirrored in simple_shader.wgsl)

48-byte repr(C), four vertex attributes:

| byte | field                              | vertex format        |
|-----:|------------------------------------|----------------------|
|    0 | screen_rect: [f32; 4]              | loc 0, Float32x4     |
|   16 | color: [f32; 4]                    | loc 1, Float32x4     |
|   32 | glyph_offset: u32, cmd_texel_count: u32 | loc 2, Uint32x2 |
|   40 | depth: f32, ppem: f32              | loc 3, Float32x2     |

Add a compile-time size assertion (48) and offset tests. Keep bytemuck
Pod/Zeroable.

### Universal glyph header (5 packed texels = 10 raw i32)

| texel | raw slots | contents                                        |
|------:|----------:|-------------------------------------------------|
|     0 |      0, 1 | bounds.min_x, bounds.min_y (exact f32::to_bits) |
|     1 |      2, 3 | bounds.max_x, bounds.max_y                      |
|     2 |      4, 5 | band_transform.scale_x, scale_y                 |
|     3 |      6, 7 | band_transform.offset_x, offset_y               |
|     4 |      8, 9 | pack_i16_pair(band_max_x, band_max_y), reserved 0 |

Layout per glyph: `[header: 5 texels][payload]` where payload is the
existing band+curve blob (mono/COLRv0) or command stream (COLRv1).
`glyph_offset` points at the header; payload base = glyph_offset + 5.
All 16-bit offsets inside the payload stay payload-relative - no rebasing
anywhere. COLRv1 outer header: union bounds, zero transform, zero maxima;
the existing 3-texel sub-glyph headers are unchanged.

The header values must be the exact same f32 bits the emission sites
previously wrote into per-instance em_rect/band_transform. If any
emission site turns out to apply a per-instance adjustment to one of
these fields, stop: that field is not per-glyph constant and the plan
needs revisiting.

### Shader + pipeline (src/gpu_cache.rs, src/simple_shader.wgsl)

- Atlas bind group visibility: FRAGMENT -> VERTEX | FRAGMENT. Read-only;
  no VERTEX_WRITABLE_STORAGE, no usage change.
- Replace the eight vertex attributes with the four above.
- vs_main decodes the header (bitcast raw slots for rects/transform,
  read_texel(glyph_offset + 4).xy for maxima), generates texcoords and
  dilation exactly as today, and forwards the existing flat varyings.
  Forward the PAYLOAD base, not the header base:
    mono/COLRv0: glyph = [glyph_offset + 5, max_x, max_y, 0]
    COLRv1:      glyph = [glyph_offset + 5, 0, 0, cmd_texel_count]
  fs_main, render_single, render_color, render_sub_glyph stay
  semantically unchanged (keep the band_max.y &= 0x00FF mask).
- Document the wgpu DownlevelFlags::VERTEX_STORAGE requirement (baseline
  WebGPU provides it; GLES-style adapters without vertex storage become
  unsupported).

### Upload + cache (src/text_atlas.rs, src/glyph_cache.rs, src/prep.rs)

- text_atlas: private header encoder returning [i32; 10]. commit_mono:
  capacity-check 5 + prepared.blob_size, append header then payload,
  store the header offset in the entry. Keep the blob_size <= 65535
  check (offsets are payload-relative). upload_color_v1: same pattern -
  header before commands, sub-glyph offsets computed exactly as today,
  return the header offset.
- prep.rs: blob stays header-free and payload-relative; clarify the seam
  comment.
- glyph_cache: rename band_offset -> glyph_offset; remove band_max_x,
  band_max_y, band_transform; keep bounds, units_per_em,
  last_used_epoch. Update the three sentinels (u32::MAX family) and
  GlyphEntry::new. Rename ColorV1GlyphEntry::blob_offset ->
  glyph_offset and cmd_count -> cmd_texel_count.

### Emission (src/text_renderer.rs, three sites ~:443/:506/:551)

- Mono: glyph_offset = entry.glyph_offset, cmd_texel_count = 0.
- COLRv0: per layer, layer.entry.glyph_offset, 0.
- COLRv1: glyph_offset, cmd_texel_count.
- screen_rect construction keeps using cached entry bounds on the CPU.
- CachedTextArea / HitDirect / re-cull logic unchanged (Vec shrinks).
- Update the local GlyphInstance test fixture (~:1128).

### Supporting changes

- examples/demo.rs builds its own standalone pipeline and blobs - migrate
  its layout, header creation, offsets, bind visibility, and vertex
  attributes to match. (demo2 uses the library path; verify, and migrate
  only if it too hand-builds instances.)
- tests/glyph_pipeline_test.rs and tests/glyph_key_and_fallback_test.rs:
  adapt to the smaller GlyphEntry. Add header-encoding tests and the
  48-byte ABI assertion.
- docs/integration-spec.md: GlyphEntry/instance sections + the
  VERTEX_STORAGE note.
- Leave untouched: shader.wgsl (dead), raster_text.rs/.wgsl, public
  prepare/render signatures, email benchmark corpus text (even where it
  mentions 96 bytes - benchmark input must stay comparable).

## Constraints

- Do not run cargo or brokkr; the orchestrator runs all builds, tests,
  snapshots, and benchmarks.
- WGSL under naga/wgpu 29. Rust edition 2024. bytemuck derive for the
  instance struct.
- No non-ASCII characters in code or comments.
- Pixel-identical rendering is a hard requirement; when in doubt, prefer
  copying the exact existing expression over re-deriving a value.

## Implementation summary

Implemented by the build session per plan; both diff reviews (direct +
resumed deep session) confirmed correctness with no blocking defects.
Bit-identity of the header values was verified by reading: all three
emission sites destructured entry bounds verbatim into the old em_rect,
which is exactly what encode_glyph_header now stores via f32::to_bits
(unit test pins NaN/-0.0/infinity preservation).

Review follow-ups applied by the orchestrator: sub-glyph offsets renamed
command-payload-relative in comments (outline.rs, text_atlas.rs); the
demo header now encodes band_data.band_count_x/y - 1 instead of
recomputing from curve count; the WGSL header offsets use a
GLYPH_HEADER_TEXELS constant instead of literal 4u/5u.

Validation: 74 tests + 13 GPU tests pass; all four approved snapshots at
0.0% pixel diff (pixel-identical requirement met).
