# WORK

Retained-cache rework: give each TextArea occurrence its own cache
entry so shared-buffer areas stop ping-ponging.

## Benchmark verdict (plantasjen, same-host A/B vs 69e077d)

- shared_buffer (new target, stored as baseline): 32 areas on one
  buffer, warm frames all-direct (32/0/0 asserted), warm prepare
  14 us/frame, cold 688 us, wall 1.478 ms. This is the scenario the
  rework exists for; no pre-change equivalent exists to compare
  against (the old code cannot express it as all-direct).
- email2: 14.987 ms (5-run) vs 15.021 ms at 69e077d, -0.2%. An
  initial single-run 15.726 ms (+4.7%) was an outlier; comparisons
  against yesterday's 5fc1a85 numbers carry large host-state deltas
  and are not like-for-like.
- render: 30.745 ms vs 30.241 ms, +1.7% (noise). Instrumented
  prepare_with_depth +2.9% on the all-miss render workload, inside
  the noise floor shown by untouched functions (extract_outline
  +5.9%).
- email: 27.396 ms vs 26.485 ms at 5fc1a85, +3.4% cross-day within
  the noise band.

## Implementation summary

Shipped per the agreed plan. The resumed deep session found no defect
in the renderer implementation; five test/bench findings were fixed by
the orchestrator:

- (high) The Cargo example was named `shared_buffer`, but brokkr's
  `--target shared_buffer` resolves to `shared_buffer_bench`; renamed.
- (medium) The bench lacked the HotpathGuard and counting-allocator
  declarations, so --hotpath/--alloc modes would silently measure
  nothing; added, following hotpath.rs.
- (medium) The color-variant test compared cold vs warm frames but
  never asserted the two areas keep DIFFERENT colors; it now asserts
  red for the first emission group and green for the second.
- (low) The order-swap test now pins group identity against the cold
  frame (swapped halves equal the cold halves shifted +/-100 within
  re-cull rounding tolerance) instead of only checking a 100px delta.
- (low) The bench's per-frame assertion moved out of the timed loop
  (direct-hit counts accumulate; validated after timing).

Validation: 82 tests + 22 GPU-gated tests pass; all four snapshots
0.0%. The 22 include six new shared-buffer tests covering the miss ->
all-direct transition, order-swap convergence, shrink/grow retention,
and color/scale/bounds validity separation - all asserted through the
new hidden PrepareStats accessor. Raster shared-buffer coverage was
skipped: no deterministic non-vector fixture exists in the repo.

## Problem (verified)

`TextRenderer`'s retained cache (`src/text_renderer.rs:122`) is keyed
by buffer pointer alone and holds ONE placement. iced deduplicates
identical text, so N TextAreas can share one `cosmic_text::Buffer` at
different placements or with different bounds/colors/scales. Verified
behavior today for two vector-only occurrences at left=0 and left=100:
cold frame both Miss (last writer caches left=100); next frame left=0
ReCulls and left=100 is Direct, and the deferred update flips the
retained placement; classifications alternate every frame. The cache
ping-pongs, `all_direct_hits` stays false, and the vector vertex
upload at line ~609 re-runs every frame. Occurrences with different
scale/bounds/default_color miss validity and rebuild every frame
(placement mismatches can also be Miss, not just ReCull, when the
cached candidates are incomplete or contain raster glyphs). Fully
identical duplicate areas DO all go Direct today - the pathology is
specifically differing placement/validity fields. The whole
`PendingCacheWrite` deferral (lines ~96-109, ~343, ~575-597) exists
only to keep the shared-pointer aliasing sound.

## Agreed plan (implement exactly this)

All in `src/text_renderer.rs` unless stated.

### 1. Key types and fields

    type BufferPtr = *const cosmic_text::Buffer;

    #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
    struct TextAreaCacheKey {
        buffer_ptr: BufferPtr,
        occurrence: usize,
    }

Renderer fields:

    text_area_cache: FxHashMap<TextAreaCacheKey, CachedTextArea>,
    text_area_occurrences: FxHashMap<BufferPtr, usize>,
    last_prepare_stats: PrepareStats,

The occurrence map lives on the renderer for capacity reuse; clear it
at the start of every prepare.

### 2. Stats (public, hidden)

    #[doc(hidden)]
    #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
    pub struct PrepareStats {
        pub direct_hits: usize,
        pub reculls: usize,
        pub misses: usize,
    }

    #[doc(hidden)]
    pub fn last_prepare_stats(&self) -> PrepareStats

Count at the exact branch where each plan is finalized; commit the
field only on successful prepare (both return points), so it always
means "last successful prepare".

### 3. Key assignment (pass 1, before any classification)

    let buffer_ptr = text_area.buffer as BufferPtr;
    let count = self.text_area_occurrences.entry(buffer_ptr).or_default();
    let cache_key = TextAreaCacheKey { buffer_ptr, occurrence: *count };
    *count += 1;

Use `cache_key` for the cache lookup and store it in whichever plan is
created. EVERY plan variant carries it - including `MissArea`, which
gains a `cache_key` field (the occurrence index is not recomputable in
pass 3).

### 4. Immediate pass-3 writes

Keys are unique within a frame, so no two plans touch the same entry:

- HitDirect: look up by key, append cached vectors.
- ReCull: one `get_mut(&cache_key)`; read `instances`/`complete`,
  produce the owned re-culled vector, append it to the frame output,
  then immediately update that entry's left/top/scroll/instances/
  complete.
- Miss: build exactly as today, then immediately
  `insert(area.cache_key, CachedTextArea { ... })`.

Delete `PendingCacheWrite`, `pending_cache_writes`, and the post-pass
application loop entirely.

### 5. Retention

    let occurrences = &self.text_area_occurrences;
    self.text_area_cache.retain(|key, _| {
        occurrences
            .get(&key.buffer_ptr)
            .is_some_and(|count| key.occurrence < *count)
    });

Replaces `used_ptrs` (delete it).

### 6. Untouched

Pass 2 miss resolution, `classify_placement`,
`re_cull_vector_instances`, visibility helpers, atlas generation
checks, raster collection/upload. The raster vertex buffer is uploaded
even on the all-direct fast path; this change restores VECTOR buffer
reuse for shared-buffer frames.

### 7. New benchmark: `examples/shared_buffer_bench.rs`

Plus an `[[example]]` entry in Cargo.toml (target name
`shared_buffer`, pattern: existing email benches). Headless device
like the other benches. One shaped buffer of distinct Latin
codepoints; 32 TextAreas at distinct (left, top) placements, full
screen bounds; `set_redraw(false)` after the cold frame. One cold
prepare, then 50 warm frames. Assert (in the bench) that warm frames
report 32 direct hits via `last_prepare_stats()`. Emit KVs:
`elapsed_ms` (mandatory), `cold_prepare_us`, `warm_prepare_avg_us`,
`direct_hits`, `reculls`, `misses`.

No cross-commit A/B exists for a new target (the old commit lacks the
example); the bench demonstrates the win on this commit and guards
future regressions.

## Tests (tests/prepare_behavior_test.rs)

Keep `shared_buffer_areas_keep_distinct_placements` and
`scrolled_content_reappears_after_cache_hit` green and UNCHANGED.

New GPU tests (same harness):

- Three-frame distinct-placement shared-buffer test asserting stats:
  frame 1 = 2 misses, frames 2 and 3 = 2 direct hits; emitted
  instances identical across frames (modulo nothing - placements are
  stable).
- Order-swap: frame 1 areas [A@0, B@100]; frame 2 [B@100, A@0]; frame
  3 [B@100, A@0] again. Frame 2 instances must match the swapped input
  order (correctness regardless of classification); frame 3 must be 2
  direct hits (convergence after heuristic mismatch).
- Shrink/grow retention: frame 1 two occurrences, frame 2 one, frame 3
  two. Frame 3 = 1 direct hit + 1 miss (stale occurrence entry was
  retained out).
- Differing-validity variants, one test each asserting frame 2 = 2
  direct hits with correct emitted output: (a) different default_color
  per area, (b) different scale, (c) different bounds.
- Raster shared-buffer coverage: ONLY if a deterministic non-vector
  fixture exists in the repo (no system-font dependence); otherwise
  skip and note it.

## Constraints

- Do not run cargo or brokkr; the orchestrator runs all builds, tests,
  snapshots, and benchmarks.
- Rust edition 2024. The warm path must gain no overhead beyond the
  occurrence-counter hash per area.
- No public API changes beyond the #[doc(hidden)] stats accessor.
- No non-ASCII characters in code or comments.
- Files: src/text_renderer.rs, tests/prepare_behavior_test.rs,
  examples/shared_buffer_bench.rs, Cargo.toml only.

## Acceptance (run by the orchestrator)

- brokkr check + ignored GPU tests pass; brokkr visual --all at 0.0%.
- render/email/email2 benches: no regression vs stored results at
  69e077d (email vs 5fc1a85).
- shared_buffer bench: warm frames all-direct, stored as new baseline.
