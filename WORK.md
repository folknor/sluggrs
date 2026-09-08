# WORK

Implement text border/outline support with a zero-cost normal path.

Do NOT run cargo or brokkr; the orchestrator runs all builds, tests, and
formatting. Read and write code only. Do not commit. Do not touch
`repos/`, `.review.toml`, or markdown files other than this one.

## Background

PR #1 (diff saved at `notes/pr1.diff`, reference only - do NOT apply it)
added borders by growing `GlyphInstance` to 68 bytes and threading a
signed-distance path through the hot shader. We are reimplementing the
feature natively on the pinned deps (wgpu 29, skrifa 0.40 - do not bump
anything) under a hard requirement: text without borders must render
through byte-for-byte identical GPU state - same 48-byte `GlyphInstance`,
same shader module source, same pipeline descriptor, same single draw.

## Agreed design (settled; do not relitigate)

### API (`types.rs`)
- `pub struct TextBorder { pub color: cosmic_text::Color, pub width: f32 }`
- `TextArea` gains `pub border: Option<TextBorder>`.
- `width` is in logical pixels, multiplied by `TextArea.scale` exactly
  once on the CPU into physical pixels. Validate the PHYSICAL result:
  non-finite, zero, negative, or overflowed => treated as `None`.
  Physical border width is constant across font sizes (CSS-outline-like).

### Scope contract
- Borders apply to monochrome vector glyphs only. COLRv0 layer
  instances, COLRv1 glyphs, and raster-fallback glyphs render exactly as
  today, borderless. Eligibility is decided per RESOLVED glyph before
  flattening into instances (note: COLRv0 layers currently flatten into
  ordinary-looking instances with `cmd_texel_count == 0`; that field
  cannot carry eligibility - track it explicitly).

### Compositing semantic
- Per area: draw a dilated UNDERLAY of each eligible glyph in the border
  color (full dilated coverage, not an exterior ring), then draw the
  area's fills over it with the existing normal pipeline. Underlay
  coverage = winding-inside OR distance <= physical width (AA over the
  half-pixel edge). Translucent fills show the underlay through; that is
  the defined semantic. Complete each area's underlay phase before its
  fill phase; preserve area order; never let one area's underlay land
  over an earlier area's fill, and do not regress the current intra-area
  ordering of fills including excluded color/raster glyphs (beware the
  current global raster tail in `render()`).
- Border (underlay) pass: depth test without depth write; no duplicated
  stencil side effects. Distinct dynamic-uniform offsets per draw -
  never rewrite one uniform slot between encoded draws.

### GPU structure
- `GlyphInstance` stays exactly 48 bytes; restore the ABI test.
- Normal instances stay input-ordered in the existing stream; an
  all-normal frame must produce the exact current single draw.
- Bordered glyphs additionally emit into a SEPARATE border instance
  stream (separate vertex buffer) using the SAME 48-byte layout, whose
  `glyph_offset` field references the glyph's border descriptor instead
  of the fill blob. Draw metadata is an ordered list of
  {stream, instance range, mode, uniform offset}.
- Border shader lives in a SEPARATE, lazily created WGSL module.
  Shared functions (curve evaluator, root solver, unpack helpers) live
  in one source fragment concatenated into both module strings at build
  time; the assembled normal module must be byte-identical to today's
  `simple_shader.wgsl`. Verify with a unit test comparing the assembled
  string to the current file content.
- Border pipeline: lazily created in `gpu_cache.rs` (extend the cache
  key), own pipeline layout adding a dynamic-offset uniform bind group
  (border color + physical width; mind the 32-byte struct size and
  `min_uniform_buffer_offset_alignment` spacing, one lazily allocated
  buffer + one bind group). The NORMAL pipeline layout, entrypoints,
  and descriptor must remain untouched.
- Border vertex shader dilates the quad by the physical border width
  plus AA allowance; the same single physical width value drives quad
  dilation, candidate radius, fragment threshold, and CPU culling
  margins. Never clamp dilation or thresholds (the radius cap below
  changes lookup strategy only).

### Per-glyph border blob (in the shared atlas)
Built lazily on first bordered use of a glyph; contains:
1. A descriptor pointing at the fill blob and the blob's own sections.
2. BORDER-CORRECTED ray bands for winding: same builder as `band.rs`
   but membership includes boundary-touching curves (no
   `BAND_EPSILON` upper-extent exclusion), consistent half-open
   endpoint ownership at shared vertices, built from the same quantized
   curves the shader evaluates. The border entrypoint uses ONLY these
   bands; normal bands are untouched. Render-time inside test is
   NONZERO winding (not parity): accumulate the existing signed root
   contributions as integer crossings (no AA clamp), counting an
   intersection only strictly along the selected ray direction; left
   and right rays may flip the sign, `winding != 0` is invariant.
   Full corrected contours (including canceled ones) stay in the
   winding bands - they sum to zero and are harmless there.
3. A 2D distance grid over FILLED-SET BOUNDARY pieces only, with
   conservative per-cell candidate lists satisfying: for every point p
   in cell C, every boundary piece within the supported radius R of p is
   in candidates(C). False positives allowed; deduplicate at prep.
   Explicit lookup rule for queries outside the represented domain.
   R must cover physical width + AA allowance via the actual
   glyph-to-screen mapping ("em-space" shader coords are font units;
   convert through units_per_em). Radius capacity grows lazily in
   power-of-two buckets; above a hard cap (~1 em) the fragment falls
   back to brute-force distance over all boundary pieces (also
   selectable when the candidate list degenerates to everything).
   Collect the frame's max radius requirement per glyph before emitting
   references. The blob records a ppem validity ceiling for its
   boundary approximation; use above the ceiling triggers rebuild.
4. Boundary extraction (CPU, prep): remove geometry that does not bound
   the nonzero-filled set, so canceled or covered contours produce no
   phantom borders. Method: CERTIFIED SUBDIVISION, not sample
   agreement - samples only SUGGEST classifications; conservative curve
   bounds and recursive Bezier subdivision must certify an interval has
   no possible classification transition before keeping/dropping it;
   unresolved events are isolated spatially until the boundary
   approximation meets a declared SCREEN-SPACE error budget at the
   blob's max supported ppem; coincident spans handled explicitly
   (identical equally-oriented traces remain boundary; opposite
   orientation cancels); conservative arithmetic for separation tests.
   No closed-form curve intersections required. Never discard an
   interval solely because finitely many samples agree; a small
   surviving component is not droppable by diameter alone (dilation
   magnifies it).
5. Distance evaluation in-shader: unsigned distance to candidate pieces
   (`sd_bezier`-style analytic solve) with an exactly-linear /
   near-linear segment fallback gated by a screen-space error tolerance
   (collinearity alone insufficient - a quadratic can overshoot and
   return), scale-aware degeneracy thresholds, sign from the winding
   test only. The border fragment does NOT reproduce fill AA coverage.
- Blob participates fully in atlas residency, relocation, compaction,
  eviction, and prepared-reference invalidation (`text_atlas.rs`
  currently tracks one resident blob per glyph key - extend it).

### Renderer / retained cache (`text_renderer.rs`)
- `prepare` partitions per area into underlay range(s) + fill range(s);
  fills use the normal pipeline and existing stream.
- Cache semantics: border color is paint-only (uniform update, no
  instance rebuild, no cache miss). Border width and presence affect
  geometry: they enter culling margins, draw-range selection, and
  placement validity; width growth can expose glyphs absent from an
  incomplete cached candidate set and must force a fresh walk.
  Eligibility and original glyph identity must survive caching.
- Every visibility layer learns the border margin: run selection
  (`run_is_visible`), `vector_rect_visible`, and
  `re_cull_vector_instances` - a glyph whose fill is out of bounds but
  whose border enters them must not disappear.
- The direct-hit fast path (skip vertex upload) stays valid for
  unchanged instances, but border uniforms and ordered draw ranges are
  rebuilt or proven unchanged independently.
- `render()` walks the ordered draw metadata; all-normal frames take
  the existing two-branch path (vector draw + raster tail) unchanged.

### Tests to add
- Assembled normal shader source is byte-identical to the previous
  `simple_shader.wgsl` content.
- `GlyphInstance` ABI test restored at 48 bytes.
- Boundary extraction: canceled opposite-winding contour pair yields no
  candidates; identical equally-oriented pair keeps its trace; covered
  stroke with a short exposed arc keeps only the arc (within budget).
- Winding: band-boundary endpoint crossing case (upward segment ending
  exactly on a band boundary) counted exactly once.
- Width validation: NaN/inf/negative/zero and scale-overflow => None.
- Cache: border color change invalidates nothing but the uniform; width
  change re-culls; bordered->unbordered returns identical instances to
  a never-bordered prepare.
- CPU-only prep tests for the grid invariant on a few real glyphs
  (every cell's candidate list covers a dense point sample of the cell
  at radius R).

### Boundary certification (as shipped)

The certified-subdivision requirement in item 4 above is implemented in
`border.rs` as follows, sign-off by deep review:
- Coincident spans are handled EXPLICITLY and EXACTLY before
  certification: all coincidence decisions run in exact integer
  arithmetic over the 4x-scaled quarter-unit quantized grid (i64 cross
  products and line scalars for segments; i128 blossom polynomial
  identity over rationalized parameters for quads, float arithmetic
  only generating candidates). Discovery is pairwise and bidirectional
  with exact rational inversion, so grouping is a true equivalence
  relation; spans, cuts, containment, net multiplicity, and min-index
  ownership stay exact rationals.
- `certify_piece` keeps/drops whole intervals only under a
  probe-pair-plus-separation certificate (recursive AABB clearance
  against subdivided other-curve hulls); degenerate-line quads have
  exact zero flatness.
- Unresolved sub-tolerance pieces are retained only with a nonzero
  winding witness within the probe radius (piece then lies within
  probe + diameter <= tol of the filled set); with no witness, and at
  the subdivision depth limit, certification fails conservatively with
  `AtlasFull` - never a silent drop, because dilation magnifies both
  phantom and deleted components beyond any screen-space budget.
- A hard 65536-piece budget bounds the final refined boundary vector
  (and hence the blob); exceeding it is the same conservative failure.

## Not in scope
- COLR/raster borders (documented exclusion), the PR's dep bumps, any
  change to `prepare()`'s public signature beyond adding the field to
  `TextArea`, shader comment removal (keep existing comments intact;
  comment new shader code in the same style).
