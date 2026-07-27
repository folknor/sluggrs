# WORK

Scroll offset breaks culling and the retained text-area cache.

## Symptom

In `examples/demo2.rs` (zoom via mousewheel, pan via drag, both applied
through `Viewport::set_scroll_offset` and viewport resolution): zooming in
and panning down permanently cuts off the bottom of the content. Zooming
back out (which changes TextArea bounds and invalidates caches) brings it
back.

## Root cause (verified by reading, needs independent confirmation)

All in `src/text_renderer.rs`, `prepare_with_depth()`:

1. Pass 1's `is_run_visible` culls layout runs against the area bounds
   using `text_area.top + run.line_top * scale` WITHOUT the viewport
   scroll offset, while pass 3's instance culling adds it
   (`vis_y = screen_y + scroll[1]`). The two culls disagree whenever
   scroll is nonzero: runs that scroll would bring into view are dropped
   before instances are ever built.

2. The retained `CachedTextArea` stores the culled instance list, and its
   validity check (redraw flag, scale, bounds, default_color,
   atlas_generation) does not include scroll. After a scroll change,
   `HitDirect` replays the stale pre-culled list and `HitShifted` re-culls
   only the instances that survived the original cull. Content culled
   under one scroll offset never reappears until something else (bounds
   or resolution change) forces a full miss rebuild.

Note: iced never sets a scroll offset (it is always [0,0]; there is no
public API for it, see TODO.md), so iced rendering is unaffected. The bug
bites any consumer that uses `Viewport::set_scroll_offset`, currently the
demos.

## Performance constraints

- The static-frame fast path is the crown jewel: all-hit frames with no
  position changes skip instance building and vertex upload entirely
  (warm prepare ~5us). Any fix must preserve that path unchanged.
- The scroll uniform exists so scrolling does not require re-shaping.
  Re-building instances on scroll change is acceptable (the mixed path
  costs ~100-226us per frame on measured workloads); re-shaping is not
  needed since cosmic-text buffers are untouched.

## Candidate directions (pick, combine, or improve)

1. Treat scroll as part of cache validity: store the scroll offset in
   `CachedTextArea`; a scroll mismatch demotes the area to the miss path
   (full rebuild with scroll-aware run culling). `is_run_visible` adds
   `scroll[1]` to its y-range test so the rebuild contains exactly what
   is visible under the current scroll. Simple and correct; scrolling
   frames pay the mixed-path rebuild cost.
2. Cache unculled instances and cull at emit time. Preserves a cheap
   scroll path (re-cull + re-upload, no instance rebuild), but requires
   dropping or rethinking run-level culling for cached areas, which
   exists to keep cold cost proportional to the visible portion of large
   buffers. Higher risk, larger change.

## Deliverables for this session

Verify the root-cause analysis against the code (challenge anything that
does not match), then produce a concrete implementation plan: files,
functions, edge cases, and which candidate direction (or a better one)
to take. Consider at least: interaction of scroll with `HitShifted`'s
dx/dy adjustment, the whole-frame fast-path condition
(`all_hit && !any_position_changed`), non-vector (raster) glyphs whose
clip bounds are stored per instance, and what invariants tests can pin
without a GPU.

## Constraints

- This is a planning session: do not modify the tree, and do not run
  cargo or brokkr. The orchestrator runs all builds, tests, and
  benchmarks.
- Rust edition 2024. Performance is the top priority; the deny-level
  clippy set lives in Cargo.toml, and perf-constraining lints are
  deliberately absent from it.

## Plan

Agreed after two consolidation rounds. Core idea: exact placement
(`left`, `top`, scroll) becomes part of cache validity; a placement-only
change re-culls the cached instance set when that set is provably
complete, and rebuilds otherwise. The unsound `HitShifted` rounding path
is removed. Additionally verified during planning: the raster path has
its own scroll bug (CPU cull at `raster_text.rs:334` ignores scroll, the
shader at `raster_text.wgsl:31` applies it), and a `left/top` shift is
never safe for cached raster glyphs because `LayoutGlyph::physical`
recomputes integer placement and the subpixel cache-key bin from the new
origin.

All in `src/text_renderer.rs` unless noted.

1. `CachedTextArea` gains `scroll: [f32; 2]` and `complete: bool`.
   `complete` documents that the cache retains every placement-dependent
   candidate needed for re-culling: no run skipped by run culling, no
   mono/COLRv1 instance rejected, no individual COLRv0 layer rejected.
   Raster candidates always count as retained (they are stored before
   raster clipping).

2. Add an `AreaPlan::ReCull` variant (buffer pointer, dx, dy, current
   bounds, current scroll, state to update the cache after emission).

3. Classification (existing redraw/scale/bounds/color/atlas-generation/
   glyph-residency checks remain prerequisites for both retained paths):
   - same `left`, `top`, scroll: `HitDirect`;
   - placement changed AND `cached.complete` AND (no raster candidates
     OR `dx == 0.0 && dy == 0.0`): `ReCull`;
   - otherwise: `Miss`. `HitShifted` is removed, including its rounded
     raster dx/dy adjustment.

4. Replace the `skip_while`/`take_while` run iteration with an explicit
   loop over a shared scroll-aware run predicate (pure helper, f32
   arithmetic: `start_y = top + line_top * scale + scroll_y`,
   `end_y = start_y + line_height * scale`, inclusive comparison against
   clipped bounds). Preserve early-stop; record
   `all_runs_included = false` when a run is skipped or the visible
   range terminates early.

5. Miss-path completeness: `complete = all_runs_included && no vector
   instance was rejected`. Raster CPU clipping does not affect it.

6. One shared pure vector rectangle-intersection helper (base
   screen_rect, scroll, clipped bounds, the existing 1 px margin) used
   by mono, COLRv0, COLRv1, and ReCull. Every rejected vector instance
   or color layer clears completeness.

7. ReCull emission: shift vector `screen_rect` origins by dx/dy, re-cull
   with current scroll and bounds via the shared helper, preserve
   ordering; retain unchanged `NonVectorGlyph` candidates for scroll-only
   raster reuse. Then update the cache: new `left`/`top`/scroll, cached
   vector instances replaced by the newly visible list, `complete`
   preserved only if nothing was rejected. (Replacing the cache after
   clearing `complete` is safe: the next unchanged frame is `HitDirect`,
   any later placement change rebuilds.)

8. Replace `all_hit`/`any_position_changed` with explicit state: only an
   all-`HitDirect` frame takes the vector upload fast path; any `ReCull`
   or `Miss` uploads. Keep the existing instance-count guard. The frame
   after a re-cull is direct again (~5us path restored).

9. `src/text_atlas.rs` + `src/raster_text.rs`: pass the current scroll
   through `rasterize_glyphs` into `RasterState`; raster visibility
   tests use `x + scroll[0]` / `y + scroll[1]` while emitted
   `RasterVertex.screen_pos` stays unscrolled (the shader applies the
   uniform exactly once).

10. GPU-free unit tests beside the pure helpers:
    - scroll-aware run visibility, fractional and negative offsets,
      inclusive edges;
    - initial completeness true with all runs included; false when a
      run, mono glyph, COLRv1 glyph, or any COLRv0 layer is rejected;
    - raster clipping not destroying completeness;
    - complete vector-only cache accepts combined dx/dy/scroll re-cull;
      raster cache accepts scroll-only, rejects left/top;
    - re-cull preserves `complete` when every vector survives, clears it
      when one is rejected;
    - exact placement after re-cull takes `HitDirect`; a later placement
      change after an incomplete re-cull takes `Miss`;
    - any `ReCull` disables the whole-frame vector upload fast path.

11. `TODO.md`: fix the stale claim that scroll has no public API
    (sluggrs exposes `Viewport::set_scroll_offset`; the iced wrapper
    does not use it).

Out of scope for the build session: benchmarks and validation runs (the
orchestrator does those), commits.

## Implementation summary

Implemented scroll-aware retained text caching: complete vector caches can
re-cull on placement changes, while incomplete or raster-position changes
rebuild. Unified vector culling and run visibility now account for scroll;
raster CPU culling does too without changing shader-space vertex positions.
Added GPU-free helper tests and corrected the TODO note about the public
sluggrs scroll API.

## Review fixes

The deep session's diff review found one defect and one coverage gap, both
fixed by the orchestrator:

- ReCull (and the pre-existing Miss insert) mutated the cache mid-emission;
  a later plan for the same buffer pointer (iced deduplicates identical
  text to one buffer) would emit another area's placement. All cache
  writes are now collected as `PendingCacheWrite`s and applied after the
  plan loop, last occurrence winning.
- Classification extracted into a pure `classify_placement` helper and the
  raster cull into `raster_rect_visible`, both unit-tested. Added GPU
  regression tests: two TextAreas sharing one buffer keep distinct
  placements across retained frames, and content culled under one scroll
  offset reappears after the offset changes (the original demo2 symptom).

## Benchmark verdict (plantasjen, same-host A/B vs parent 10ab57c)

- render: warm 860us (pre 837-879), mixed 237us (pre 226-236). Unchanged.
- email2: warm 170-171us (pre 175-177), cold 4288us (pre 4096-4523).
  Unchanged. One 13.4ms cold outlier was page-cache noise (extract_outline
  P50 normal, P99 195us; immediate re-run normal).
- email (the scroll-phase workload): warm 223 to 227/215us and cold
  ~2250us unchanged; scroll phase 255 to 401/381us. The +130-150us per
  scrolling frame is the cost of correct re-culling. Decisive detail:
  final glyphs went 157 to 159 and buffer texels 47028 to 47368 - the
  fixed path renders two glyphs the old shift path silently dropped,
  benchmark-visible proof of the unsoundness this loop fixed.
