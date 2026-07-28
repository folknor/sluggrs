# WORK

Architecture: consolidate glyph resolution inside TextAtlas and verify
renderer/atlas pairing at render time. Two TODO items in one loop:

1. "TextRenderer/TextAtlas coupling" - the renderer reaches into the
   atlas via pub(crate) fields and methods; policy, cache state, and
   upload orchestration are spread across both files.
2. "API doesn't encode TextRenderer-TextAtlas lifetime" - render()
   accepts any &TextAtlas but the instance buffer was prepared against
   a specific atlas's offsets; mispairing is type-correct but renders
   garbage.

## Problem detail (verified earlier this session)

`src/text_renderer.rs` currently:

- reads/writes `atlas.glyphs` directly (get_and_mark_used in pass 1,
  get in pass 3, insert_and_mark_used in resolve_glyph_miss),
- reads `atlas.color_glyphs` / `atlas.color_v1_glyphs` in pass 3 and
  inserts into them in the resolve path,
- drives the whole cold-glyph pipeline itself (`resolve_glyph_miss`,
  `upload_colr_v0_layers`): restore_cached_glyph, font lookup +
  FontRef parse (its own `font_cache`), extract_outline/COLR checks,
  prepare_mono with its own `prep_scratch`, then atlas.commit_mono /
  upload_color_v1 / commit_color_v0,
- uses `atlas.bind_group` in render(), plus flush_uploads,
  rasterize_glyphs, render_raster_pass, generation(),
  get_or_create_pipeline, init_raster.

`src/text_atlas.rs` exposes pub(crate) fields: cache, glyph_buffer,
bind_group, format, glyphs, color_glyphs, color_v1_glyphs.

Pairing: `TextRenderer::new(atlas, ...)` bakes the pipeline from that
atlas's Cache+format; `prepare*(atlas)` builds instances referencing
that atlas's buffer offsets; `render(atlas)` binds whatever atlas it
is handed. The only guard is the generation counter, and two distinct
atlases both start at generation 0, so cross-atlas mispairing passes
the check and renders garbage. types.rs also carries a stale doc
comment claiming render() never returns errors (the generation guard
has returned RemovedFromAtlas for a while) - fix it in this loop.

## Benchmark verdict (plantasjen, same-host, vs stored b7f0ca9 results)

Perf-neutral as required, using stored baselines only (no worktree
reruns): email2 15.102 ms vs 14.987 ms 5-run baseline (+0.8%), render
29.869 ms vs 30.745 ms (-2.8%), shared_buffer 1.322 ms vs 1.478 ms
(-10.6% on a 1.5 ms target - noise, right direction). The one
per-frame addition is a u64 compare in render() and the assert in
prepare().

## Implementation summary

Shipped per the agreed plan. The resumed deep session verified the
seam list is exact (no extra atlas reaches), all seven fields and
four resolution helpers private, source-level behavior preserved
across every resolution path (error routing, NON_VECTOR fallbacks,
font-cache population, COLR v0/v1, fake italic, weight variation),
and both pairing checks placed correctly. It found no production
defects; two low test-quality findings were fixed by the
orchestrator:

- The cross-atlas render test now uses bundled InterVariable and
  asserts at least one prepared vector instance (it could previously
  pass with zero instances on a font-less host).
- The prepare-pairing test now checks the panic message and proves
  recovery: after the rejected prepare, preparing with the
  constructor atlas succeeds and emits instances (pinning the
  "assert before any state mutation" property).

Validation: 82 unit + 25 GPU tests pass; all four snapshots 0.0%
(emoji-colr exercises mono + COLRv0 + COLRv1 through the moved
resolution path). Resolves both arch TODO items.

## Agreed plan (implemented exactly as written)

1. **Atlas identity.** `TextAtlas` gets a `u64` id from a static
   `AtomicU64` (relaxed ordering), assigned only in
   `with_initial_buffer_capacity()`. pub(crate) `id()` accessor. The
   id is instance identity and survives compaction (generation covers
   layout changes).

2. **Move resolution into TextAtlas.** Move `CachedFont`,
   `font_cache`, `prep_scratch`, `resolve_glyph_miss`,
   `upload_colr_v0_layers`, and `band_count_for_curves` from
   text_renderer.rs into text_atlas.rs. Rename the entry point to
   `resolve_glyph(&mut self, font_system, key) -> Result<GlyphEntry,
   PrepareError>`. Make it defensively check the resident glyph map
   first, then blob restoration, then extraction+commit - correct
   independent of the pass-1 precondition, no warm-frame cost (misses
   only). After the move, make `restore_cached_glyph`, `commit_mono`,
   `upload_color_v1`, and `commit_color_v0` PRIVATE.

3. **Narrow seam + privacy.** Add pub(crate) accessors:
   `glyph_mark_used(&mut self, &GlyphKey) -> Option<GlyphEntry>`,
   `glyph(&self, &GlyphKey) -> Option<GlyphEntry>`,
   `color_glyph(&self, &GlyphKey) -> Option<&ColorGlyphEntry>`,
   `color_v1_glyph(&self, &GlyphKey) -> Option<&ColorV1GlyphEntry>`,
   `bind_group(&self) -> &BindGroup`. Then make all seven fields
   (cache, glyph_buffer, bind_group, format, glyphs, color_glyphs,
   color_v1_glyphs) private. Verified: no source outside
   text_renderer.rs touches them; examples/tests use public methods
   only. The complete post-move seam is: constructor id()/
   get_or_create_pipeline()/init_raster(); prepare id()/generation()/
   glyph_mark_used()/resolve_glyph()/glyph()/color_glyph()/
   color_v1_glyph()/flush_uploads()/rasterize_glyphs(); render id()/
   generation()/bind_group()/render_raster_pass(). Any other atlas
   access remaining in text_renderer.rs is a plan violation.

4. **Pairing enforcement.** `TextRenderer` stores `atlas_id` from the
   constructor atlas. At the very start of `prepare_with_depth()`
   (before any state mutation): unconditional
   `assert_eq!(atlas.id(), self.atlas_id, ...)` with a clear message -
   this is a programmer invariant, `PrepareError` has no suitable
   variant, and a debug-only check would let release builds populate
   retained state from the wrong atlas (the retained cache validates
   only generation, so a second atlas with the same glyph keys could
   reuse wrong offsets silently). In `render()`: check id FIRST and
   return `RenderError::RemovedFromAtlas` on mismatch, then the
   existing generation check. Do NOT add an enum variant (public
   non-#[non_exhaustive] enum; cryoglyph drop-in compatibility) and do
   NOT use debug_assert in render.

5. **types.rs.** Rewrite the stale RenderError doc comment (render()
   does return RemovedFromAtlas: on identity mismatch and on atlas
   generation change) and broaden the Display wording for
   RemovedFromAtlas to "prepared atlas data is invalid or unavailable"
   -style phrasing covering both cases.

## Tests (tests/atlas_lifecycle_test.rs or a new pairing test file)

There is currently NO test that calls render() after a generation
change - atlas_lifecycle_test.rs:180 only compares generation values.
Add real render-path tests (GPU, ignored, existing harness patterns):

- Two atlases A and B from the same Cache/device/format (both
  generation 0). Renderer constructed+prepared with A using
  deterministic non-empty vector text; inside a valid render pass:
  `render(B)` returns Err(RemovedFromAtlas); `render(A)` returns
  Ok(()).
- Constructor pairing: renderer `new(A)`, then `prepare(B)` panics
  (use `std::panic::catch_unwind` or `#[should_panic]` as fits the
  harness; the wgpu resources must not be poisoned - a dedicated
  small test is fine).
- Real generation test: prepare with A, force compaction (small
  initial capacity constructor + disjoint glyph sets across trims, as
  atlas_lifecycle_test already does) until generation changes, then
  assert `render(A)` returns Err(RemovedFromAtlas) BEFORE
  re-preparing, and Ok after re-preparing.

Existing suites must pass unchanged: 82 unit + 22 GPU tests, four
snapshots at 0.0% (pure refactor for correctly paired usage;
emoji-colr exercises mono + COLRv0 + COLRv1 through the moved
resolution path).

## Superseded proposal (for context only)

### A. Move glyph resolution into TextAtlas

Move `resolve_glyph_miss` and `upload_colr_v0_layers` - together with
the `font_cache: FxHashMap<(fontdb::ID, Weight), CachedFont>` and
`prep_scratch: PrepScratch` fields they use - from TextRenderer into
TextAtlas. Public-ish seam (pub(crate)):

    atlas.resolve_glyph(font_system, key) -> Result<GlyphEntry, PrepareError>

The renderer's pass 2 becomes a loop over distinct misses calling
that. Rationale: everything the resolve path touches (blob-cache
restore, font tables, outline extraction policy, NON_VECTOR fallback
routing, commit) is atlas policy and atlas state; the renderer only
needs the resulting entry. This makes classification/eviction/fallback
changes single-file. The "one scratch while serial, one per worker
under future rayon" note moves along with prep_scratch.

### B. Narrow the crate-internal surface

Make `glyphs`, `color_glyphs`, `color_v1_glyphs`, `bind_group`,
`cache`, `glyph_buffer`, `format` private to text_atlas.rs. Add the
minimal pub(crate) accessors the renderer actually needs:

- pass 1: `glyph_mark_used(&mut self, key) -> Option<GlyphEntry>`
  (wraps glyphs.get_and_mark_used)
- pass 3: `glyph(&self, key) -> Option<GlyphEntry>`,
  `color_glyph(&self, key) -> Option<&ColorGlyphEntry>`,
  `color_v1_glyph(&self, key) -> Option<&ColorV1GlyphEntry>`
- render: `bind_group(&self) -> &BindGroup`

Existing pub(crate) methods (flush_uploads, rasterize_glyphs,
render_raster_pass, init_raster, get_or_create_pipeline) stay. The
public `glyph_map()` accessor stays (external tests use it).

### C. Runtime pairing verification

- TextAtlas gets a unique instance id: `id: u64` from a static
  `AtomicU64` counter, assigned in the constructor, surviving
  compaction (compaction already bumps generation; the id is the
  instance identity, not the layout identity). pub(crate) or
  #[doc(hidden)] accessor.
- TextRenderer records `prepared_atlas_id` alongside
  `prepared_atlas_generation` in prepare_with_depth.
- render() verifies id first, then generation. On id mismatch return
  `RenderError::RemovedFromAtlas`? OPEN QUESTION below.

Public API signatures stay identical (cryoglyph drop-in; iced is out
of scope). No behavior change for correctly paired usage.

### D. types.rs doc fix

Rewrite the stale RenderError doc comment: render() DOES return
RemovedFromAtlas when the atlas generation (or now identity) does not
match the prepared state.

## Open questions for review

1. Do font_cache and prep_scratch belong in TextAtlas, or should a
   third internal type (e.g. a GlyphResolver owned by the atlas) hold
   them to keep TextAtlas from growing into a god object? Judge
   against the file as it exists (it already owns the blob cache,
   raster state, swash cache, and compaction).
2. Mispair error semantics: reuse RemovedFromAtlas (no API change,
   slightly wrong name) vs a new RenderError variant (additive public
   enum change - check whether repos/iced's text.rs matches on
   RenderError exhaustively before recommending it) vs debug_assert +
   RemovedFromAtlas in release. Recommend one.
3. Should prepare() also verify (or record-and-warn) that the passed
   atlas matches the constructor-time pipeline source? The pipeline
   depends only on Cache identity + format, so a strict check may be
   too strong; a debug_assert on Arc::ptr_eq(cache) + format equality
   may be right. Or skip constructor pairing entirely and only pin
   prepare-vs-render. Recommend.
4. Is there any remaining reach-in this plan misses? Enumerate every
   `atlas.` access in text_renderer.rs against the proposed seam.
5. Anything in examples/ or tests/ that relies on the fields going
   private? (External crates can only see public items already, but
   confirm nothing in src/ outside the two files uses them.)
6. Test plan critique (below).

## Planned tests

- New GPU test (tests/prepare_behavior_test.rs or a new file): two
  atlases, one renderer; prepare against atlas A, render into a pass
  with atlas B; assert Err(...) instead of silent garbage; render with
  A succeeds. Also cover: prepare(A), compact/trim A until generation
  bumps... (existing generation test covers that; do not duplicate).
- Existing suites must pass unchanged: 82 unit + 22 GPU tests, all
  four snapshots at 0.0% (pure refactor, zero behavior change for
  paired usage).

## Constraints

- Do not run cargo or brokkr; the orchestrator runs all builds, tests,
  snapshots, and benchmarks.
- Rust edition 2024. Perf-neutral refactor: no new per-frame work on
  the warm path beyond the one id comparison in render().
- Public API: signatures unchanged; additions limited to what question
  2 decides plus #[doc(hidden)]/pub(crate) accessors.
- No non-ASCII characters in code or comments.
- Files: src/text_renderer.rs, src/text_atlas.rs, src/types.rs, tests.

## Acceptance (run by the orchestrator)

- brokkr check + ignored GPU tests pass; brokkr visual --all at 0.0%.
- render/email2/shared_buffer benches: no regression vs stored results
  at b7f0ca9 / edd0f7d.
