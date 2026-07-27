@AGENTS.md

# Claude-side process

Project knowledge - what sluggrs is, code structure, brokkr, lints, tech
stack and version pins - lives in `AGENTS.md`: imported above for Claude,
picked up natively by codex. This file holds what codex sessions never
touch: orchestration (the review tool, the fix loop), git, bash rules,
profiling, the iced repo setup, and Claude-harness specifics.

## Bash rules

- Each Bash invocation runs exactly one command. To run several, send
  multiple Bash calls (in parallel when independent). This subsumes `&&`,
  `;`, `|`, and multi-line scripts in one Bash call.
- Never use `sed`, `find`, `awk`, `head`, `tail`, or complex bash commands.
- Never chain commands with `&&`.
- Never chain commands with `;`.
- Never chain/pipe commands with `|`. Exception: piping into `review` is
  allowed (writing scratch prompt files is wasteful).
- Never capture stdout into env vars (`UUID=$(...)`).
- Never read or write from `/tmp`. All data lives in the project.
- Never run raw `cargo`, `curl`, `pkill`. Use `brokkr`. Exception:
  non-sluggrs projects (iced, etc.).
- Never run `git` with `-C <path>`. Run `git` from the current working
  directory.

## Git commit rules

- Always run `brokkr fmt` before a commit.
- Never commit markdown changes and/or `.brokkr/results.db` alone. Bundle
  them with upcoming code commits.
- When committing other changes: always tag along markdown files and
  `.brokkr/results.db` if dirty. (`sidecar.db` stays out of git - way too
  large - which is why `.gitignore` un-ignores only `results.db` from
  `.brokkr/`.)
- Write substantive engineering-focused commit messages.
- Has `Cargo.lock` changed? Commit it.
- Never `git push` unless the user explicitly asks. Stop after the commit.

## Memory rules

Do not use your Memory functionality. Do not read, write, or update
memories. Do not suggest saving things to memory. Durable context belongs in
CLAUDE.md or the relevant docs, not in per-session memory files - this
project is developed across several hosts and users, and memory does not
transfer between them; CLAUDE.md does.

## General rules

- Don't remind the user of CLAUDE.md rules. They wrote them, so they know
  them.

## Profiling

Five functions are instrumented with `#[hotpath::measure]`:
- `extract_outline()`, `prepare_outline()`, `build_bands()`, `upload_glyph()`, `prepare_with_depth()`

`.brokkr/results.db` is committed to git - always commit it after profiling runs so performance data is tracked alongside the code. Brokkr requires a clean git tree to store results, but allows a dirty `results.db` or markdown file changes - so you don't need to commit CLAUDE.md edits before running profiling.

The hotpath example emits KV pairs to stderr (captured by brokkr):
`distinct_glyphs`, `curve_texels`, `band_texels`, `cold_prepare_us`,
`warm_prepare_avg_us`, `mixed_prepare_avg_us`, `curve_texture_bytes`,
`band_texture_bytes`, `gpu_text_render_us`.

### Cross-commit A/B

Same-host A/B verdicts no longer need manual checkout juggling:

- `brokkr hotpath --bench` builds the example bare (no hotpath
  instrumentation) and runs it 3 times by default - these uninstrumented
  walls are the numbers to compare across commits.
- `--commit <sha>` builds and benchmarks an old commit in a brokkr-managed
  worktree, leaving the working tree alone. Worktrees are persistent (with
  their own target dir, deliberately not the shared one); remove them with
  `brokkr clean --worktrees` when done.
- `brokkr results --compare <sha_a> <sha_b> --mode bench` shows the verdict
  side-by-side.

Run counts ride on the mode flags (`--bench 5`, `--hotpath 3`, `--alloc 5`);
the old `-n` flag and `results --compare-last` are gone.

### GPU profiling

Both CPU and GPU profiling run headless - no user interaction needed.

- `brokkr sluggrs hotpath` measures CPU-side prepare AND GPU fragment shader
  time via wgpu-profiler timestamp queries. Renders to an offscreen 1920x1080
  texture. The `gpu_text_render_us` KV is stored in results.db.
- `cargo run --example demo` also emits `gpu_text_render_ms` to stderr on
  each redraw (5 warmup frames at startup flush the profiler pipeline). A
  window opens but GPU timing is captured automatically:
  `timeout 3 cargo run --example demo 2>&1 | grep gpu_`
- Requires TIMESTAMP_QUERY + TIMESTAMP_QUERY_INSIDE_PASSES wgpu features
  (NVIDIA, AMD, Intel desktop all support these). Gracefully disabled if
  unavailable.

Baseline (RTX 3080, 92 glyphs): CPU prepare 753us, GPU render 11us
(headless offscreen) / 71us (windowed with compositor).

## iced integration

`repos/iced/` is the canonical checkout. Branch `sluggrs`, with `text.rs`
swapped from cryoglyph to sluggrs (cryoglyph is removed from the workspace
entirely). To test in ratatoskr, point its iced dependency at the fork.

Remotes in `repos/iced/` are named unusually - check before pulling:
- `origin` = `squidowl/iced`, branch `arboard-full-patch` (**upstream**)
- `fork` = `folknor/iced`, branch `sluggrs` (ours)

So `git pull` there pulls from *upstream*, not the fork.

### The path dependency

iced's `Cargo.toml` uses `sluggrs = { path = "../../../sluggrs" }`. That
resolves only from a checkout exactly three levels below `/home/folk/Programs`,
i.e. `repos/iced/`. A clone anywhere else - e.g. `/home/folk/Programs/iced` -
resolves to `/home/sluggrs` and fails to build.

`repos/iced/` is the **only** iced checkout, and the one downstream projects
point their iced dependency at. Don't make a second one - `research/` briefly
held a shallow copy and it only caused confusion about which tree was live.

Because it's a path dep and not a git rev, iced always sees the working tree
of the local sluggrs checkout. No push or rev bump needed to test a change.

### Catching up with upstream

The fork is kept at exactly **one commit** ahead of upstream ("Replace
cryoglyph with sluggrs..."), which makes catch-ups mechanical:

```sh
git fetch origin arboard-full-patch
git rebase --onto origin/arboard-full-patch <our-commit>^ sluggrs
cargo check --workspace
git push --force-with-lease fork sluggrs
```

Upstream **rebases and rewords** its branch, so our older copies of upstream
commits will not match by patch-id - `git cherry` reports them as new. Verify
by commit subject instead, and drop the duplicates. Never `git pull` (merge)
the fork after a rebase; reset to `fork/sluggrs` instead.

## Code review

`.review.toml` defines three archetypes, codex-only by default:

- **bare** = empty priming prompt; the piped prompt is the whole
  instruction. The fix loop uses this so no hidden persona shapes any
  stage's output and every stage stays auditable after the fact.
- **bugs** = correctness-bug hunter that inspects the current repo state.
- **goal** = prefixes the prompt with `/goal `.

Profiles are tiers, not roles - any archetype can take any profile:

- `--profile deep` = gpt-5.6-sol, xhigh effort, read-only sandbox. Spec
  critique and planning.
- `--profile build` = gpt-5.6-terra, medium effort, workspace-write
  sandbox. Implementation.

Usage: `echo "prompt" | review bare --profile deep`. The session ID is
printed above the response; follow up with `--session <ID>`.

`research/` (gitignored) contains reference repos and docs for reviewer context:
Slug reference shaders, fontations, cosmic-text, vello, mesa, wgpu-naga, gpuweb spec,
Vulkan spec, NVIDIA shader guides, JCGT paper, and other Slug implementations.

TODO.md tags like `**hb review**` date from the previous 7-archetype setup;
they record which reviewer caught what and stay as provenance.

## Fix loop

The process driving `WORK.md` against the backlog in `TODO.md`. `WORK.md`
holds only the current loop's problem statement and plan - codex sessions
read it and should see the problem, not the process. Loop history lives in
git, not in `WORK.md`.

One loop:

1. Pick the next target(s) from `TODO.md`.
2. Shallow-verify the target(s) are real by reading the cited code.
3. Write the problem statement into `WORK.md`.
4. Launch `review bare --profile deep` pointed at `WORK.md`: verify the
   targets independently, produce an implementation plan. Runs in the
   background so step 5 overlaps it; everything else is synchronous.
5. While 4 runs: read the target code in depth and produce an independent
   plan. Read-only work only - no edits while a reviewer is running.
6. Consolidate. Argue with the reviewer until the plans agree. Write the
   agreed plan into `WORK.md`.
7. Launch `review bare --profile build` to implement.
8. Review the diff twice: once directly, once by resuming the deep session
   from step 4 with `--session <ID>`.
9. Actionable findings get fixed - by a build session (findings written
   into `WORK.md`) or by hand.
10. Delete the shipped entry from `TODO.md`, run `brokkr fmt` +
    `brokkr check`, commit. For changes that can alter rendered output,
    also run `brokkr visual --all` against the approved baselines. For
    performance work, profile with `brokkr hotpath` and commit
    `.brokkr/results.db` alongside the code.

Notes:

- **Point sessions at `WORK.md` and nothing else.** Never tell a review or
  build session to read `CLAUDE.md` or `AGENTS.md`. codex picks up
  `AGENTS.md` by convention, and `CLAUDE.md` is Claude-side process that
  only pollutes a codex session's context. Standing project constraints
  reach codex through `AGENTS.md`; `WORK.md` must carry everything
  loop-specific the session needs, restated inline - above all "do not run
  cargo/brokkr" (the orchestrator runs all checks). If a session needed to
  know something and did not, that is a gap in `WORK.md`, not a missing
  reading assignment.
- **Codex refusal fallback.** If codex refuses or errors out on an item,
  redo that item with the `Agent` tool, then go back to codex for the next
  item. Do not water down the prompt, and do not treat a refusal as a
  finding about the work. Reviewing/planning (steps 4, 8):
  `model: "fable"`. Implementing (step 7): `model: "opus"`. Either way the
  agent reads and writes only; the orchestrator runs `brokkr fmt` /
  `brokkr check` in the main conversation, so parallel agents never
  contend over a build.
- **Never mutate the working tree while a review session is running.** A
  `review` session starts fresh and fetches code itself: it runs
  `git diff` and opens files in the live tree. A `git stash`,
  `git checkout`, or any edit during that window silently changes what it
  reviews, and the verdict that comes back cannot be placed. Concretely:
  baseline profiling (`brokkr hotpath`) happens before launching the
  reviewer or after it returns, never alongside it. The step-5 overlap is
  read-only work only.

