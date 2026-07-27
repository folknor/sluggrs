# WORK

Cold-frame stalls from glyph storage buffer growth in sluggrs.

## Problem

All in `src/text_atlas.rs`:

- `INITIAL_BUFFER_CAPACITY = 8192` texels; one texel is 8 bytes (2 packed
  i32), so the initial GPU storage buffer is 64 KiB (`create_glyph_buffer`).
- `commit_mono()` and `upload_color_v1()` append glyph blobs. When
  `new_end > buffer_capacity`, `grow_buffer()` doubles until it fits,
  creates a new wgpu buffer and bind group, and resets
  `gpu_flush_cursor = 0`, which makes the next `flush_uploads()` rewrite
  the entire CPU-side `buffer_data` to the GPU.
- Growths happen mid-prepare, per glyph, so one cold frame triggers
  several buffer recreations and the final flush is preceded by
  progressively larger wasted re-upload state.
- `trim()` uses `buffer_capacity > INITIAL_BUFFER_CAPACITY * 4` as its
  "substantial growth" trigger for atlas reset; `reset_atlas()` recreates
  at `INITIAL_BUFFER_CAPACITY`. Any capacity change must keep this
  heuristic coherent.

Measured workloads (brokkr results.db, commit e98d6d1):

- default render bench: 93 glyphs, 22,139 texels (177 KiB) final. The
  cold frame grows 8192 to 32768 (2 growths).
- email2 mixed-locale bench: 367 glyphs, 123,149 texels (985 KiB) final.
  The cold frame grows 8192 to 131072 (4 growths).

Even the smallest workload outgrows the initial buffer immediately.
Per-glyph cost observed: roughly 240-340 texels (latin vs CJK-heavy).

## Goal

Eliminate growth-copy stalls on the first cold frame. Candidate
directions from the backlog (pick, combine, or improve):

1. Raise the initial capacity (backlog suggests 1-4 MB).
2. Predict required size and pre-allocate once per prepare: misses are
   known after the classify pass, and `prep::prepare_mono` produces each
   blob (including `blob_size`) before `commit_mono` writes it, so a
   frame's total could be summed before any commit.
3. At minimum, collapse multiple growths within one prepare into one.

Deliverables for this session: verify the problem statement above against
the code, then produce a concrete implementation plan (files, functions,
edge cases). Consider at least: coherence of the `trim()` reset heuristic,
the `max_storage_buffer_binding_size` clamp in `commit_mono()`, the COLRv1
path (`upload_color_v1` already computes `total_blob_size` before
appending), and the retained-cache `generation` counter on `TextAtlas`.

## Constraints

- This is a planning session: do not modify the tree, and do not run
  cargo or brokkr. The orchestrator runs all builds, tests, and
  benchmarks.
- Rust edition 2024. Performance is the top priority; the deny-level
  clippy set lives in Cargo.toml, and perf-constraining lints are
  deliberately absent from it.

## Plan

Agreed after independent verification. Core idea: since `flush_uploads()`
is the only GPU write and runs once per prepare, the GPU buffer never
needs to grow mid-prepare. Commits become CPU-only appends; the flush
sizes the GPU buffer exactly once.

All in `src/text_atlas.rs` unless noted.

1. Raise `INITIAL_BUFFER_CAPACITY` from 8192 to 131072 texels (1 MiB).
   Covers both measured workloads (22k and 123k texels) with zero growth.
   Fix its doc comment while there: a texel is 8 bytes (2 packed i32),
   not 16.

2. Cache two limits at construction time:
   - actual initial capacity: the constant clamped to the device limit;
   - max capacity: `max_storage_buffer_binding_size / BYTES_PER_TEXEL`.
   `reset_atlas()` recreates at the actual initial capacity, not the raw
   constant.

3. Add one checked validation helper used by both `commit_mono()` and
   `upload_color_v1()` before they touch `buffer_data` or `buffer_cursor`:
   - `checked_add` for `buffer_cursor + blob_size`;
   - reject ends beyond the cached max capacity with
     `PrepareError::AtlasFull`;
   - keep the existing per-blob 65535 check in the mono path.
   COLRv1 blob-size construction switches its usize-to-u32 conversions,
   multiplications, and offset additions to checked forms feeding the same
   error.

4. Defer GPU growth to `flush_uploads()`:
   - `commit_mono()` / `upload_color_v1()` only append to CPU storage
     after validation (no `grow_buffer` calls, no Device needed);
   - at the start of `flush_uploads()`, if `buffer_cursor` exceeds
     `buffer_capacity`, allocate once: next power of two, capped at max
     capacity, never below the required end; rebuild the bind group;
     reset `gpu_flush_cursor` to 0; then do the existing single write.
   Larger-than-initial prepares get exactly one growth per frame.

5. Drop the now-unused `Device` parameters from `commit_mono()` and
   `upload_color_v1()` (the atlas owns a Device clone). Update callers in
   `src/text_renderer.rs` (`resolve_glyph_miss`, `upload_colr_v0_layers`).

6. Make the trim policy explicit: named growth-factor constant, compare
   `buffer_capacity >= initial_capacity * 4` (the current strict `>` on
   power-of-two capacities makes the effective threshold 8x), and document
   the resulting policy: reset eligibility at 4 MiB, reset returns to
   1 MiB. The in-use/cached working-set test stays unchanged.

7. Generation semantics unchanged: growth does not bump `generation`
   (offsets stay valid; the single flush lands before prepare records its
   generation), only `reset_atlas()` increments it.

8. Tests (`tests/atlas_lifecycle_test.rs`):
   - The capacity-planning arithmetic (power-of-two target, device-limit
     clamp, overflow to `AtlasFull`) lands in a pure helper with unit
     tests that need no GPU.
   - Tests 7/8 currently force growth past the reset threshold with ~140
     glyphs; that no longer works at 1 MiB initial. Add a `#[doc(hidden)]`
     constructor taking an explicit initial capacity so lifecycle tests
     can exercise growth and reset with small buffers, and update the
     stale texture-era comments.

Out of scope for the build session: benchmarks and validation runs (the
orchestrator does those), TODO.md bookkeeping, commits.

## Implementation summary

Raised the default glyph storage buffer to 1 MiB, cached device-clamped
capacity limits, deferred buffer replacement to the single upload flush, and
added checked mono/COLRv1 capacity accounting. Updated trim's 4x policy,
lifecycle coverage for small-buffer growth/reset behavior, and pure planning
arithmetic tests. No cargo or brokkr commands were run.
