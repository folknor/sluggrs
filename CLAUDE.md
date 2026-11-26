@AGENTS.md

# Claude-side process

Project knowledge - what sluggrs is, code structure, brokkr, lints, tech
stack and version pins - lives in `AGENTS.md`: imported above for Claude,
picked up natively by codex. This file holds what codex sessions never
touch: git, bash rules, profiling, the iced repo setup, and Claude-harness
specifics.

## More rules

### Memory rules

Do not use your Memory functionality. Durable context belongs in CLAUDE.md or the relevant docs.

### Bash rules

- Never use `sed`, `find`, `awk`, `head`, `tail`, or complex bash commands.
- Never `find /`.
- One Bash() invocation === one command.
- Never chain commands with `&&`.
- Never chain commands with `;`.
- Never chain/pipe commands with `|`. Exception: piping into `review` is allowed.
- Never capture stdout into env vars (`UUID=$(...)`).
- Keep `git commit -m` messages free of zsh metacharacters - braces `{}`, brackets `[]`, parens `()`, angle brackets `<>`, `#`. They trip the permission matcher and block the commit. Spell lists out (`syntax, vm, data and runner`, not `{syntax,vm,data,runner}`), write `5.1 per bar` not `5.1/bar`, name attributes in prose not `#[attr]`.

### Git commit rules

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

### Subagents

- Always get permission from the user before launching subagents.
- Do NOT use git worktree isolation for parallel agents. Worktrees create merge conflicts that silently drop agent work. Instead, launch agents in the same tree with strict file ownership - zero overlap.

Agent coordination rules:
- Each agent gets exclusive ownership of specific files. No two agents touch the same file.
- Agents must read their target file FIRST. Do not replace existing code with placeholders or stub it out.
- Agents must NOT run `brokkr` or `cargo`. The orchestrator validates between agents.

Audit protocol:
- Do not trust agent claims of completion. Verify existence + wiring + behavior.
- Use the 3-pass audit structure: domain-specific verification, then cross-cutting reconciliation (is the new code actually wired into its callers?), then editorial normalization.
- Any discrepancies doc should contain only current gaps, not historical records. Remove resolved items entirely.

Subagent prompt rules:
- Scope the investigation, not the report. Caps like "under 1500 chars" or "max 15 findings" throw away signal you asked them to surface.
- Invite lateral findings up front. If they notice a bug, optimization, smell, or anything surprising while doing the scoped work, they should flag it, even when it's outside the immediate task.
- Name the question, not the method. Don't prescribe tools ("use `git diff`", "use `Read`"), don't prescribe steps ("read in full, not just hunks"), don't enumerate files when the scope already implies them. Prescribing the method wastes tokens and signals distrust.
- Don't restate rules the agent already inherits. Subagents load the same CLAUDE.md / AGENTS.md as the main session, so the bash rules, no-cargo, no-worktrees, gremlins, etc. are already in scope. Re-listing them is noise.
- Do pass anything learned in *this* conversation that the agent can't see: the user's framing, prior decisions, what's already been ruled out, the specific claim being audited.

### Codex agents

Never tell a codex agent to read CLAUDE.md (it is Claude-specific and contradicts their job), and never tell them to read AGENTS.md (codex loads it automatically). Put any rule they need directly in the prompt.

## Profiling

Five functions are instrumented with `#[hotpath::measure]`:
- `extract_outline()`, `prepare_outline()`, `build_bands()`, `upload_glyph()`, `prepare_with_depth()`

Benches run through `brokkr bench` (Lua script benches, see AGENTS.md). Runs
go to the per-user bench store / results server, not to git.
`.brokkr/results.db` holds the history from the retired `brokkr hotpath`
command; it stays committed but new runs no longer land in it. A run on a
dirty tree is stored but marked dirty and never compared, so commit before
benchmarking anything you want to compare.

Each example's `key=value` stderr lines land in the run's `record.kv`
(the render bench: `distinct_glyphs`, `cold_prepare_us`,
`warm_prepare_avg_us`, `mixed_prepare_avg_us`, `buffer_bytes`,
`gpu_text_render_us` and more). `brokkr results UUID` shows them.

### Cross-commit A/B

- `brokkr bench render` builds the example bare (no hotpath
  instrumentation) and runs it 3 times - these uninstrumented walls are
  the numbers to compare across commits. `--hotpath` and `--alloc` runs
  pair only with their own kind.
- `--commit <sha>` builds and benchmarks an old commit in a brokkr-managed
  worktree, leaving the working tree alone. Worktrees are persistent (with
  their own target dir, deliberately not the shared one); remove them with
  `brokkr clean --worktrees` when done.
- `brokkr bench --rounds 6 --commit A --commit B render` runs an
  interleaved A/B (ABBA order); `brokkr results --invocation ID`
  summarizes it.

### GPU profiling

Both CPU and GPU profiling run headless - no user interaction needed.

- `brokkr bench render` measures CPU-side prepare AND GPU fragment shader
  time via wgpu-profiler timestamp queries. Renders to an offscreen 1920x1080
  texture. The `gpu_text_render_us` KV is stored in the run's record.
  `brokkr bench gpu` is the dedicated fragment-shader timing bench.
- `brokkr run demo` also emits `gpu_text_render_ms` to stderr on each
  redraw (5 warmup frames at startup flush the profiler pipeline). A
  window opens but GPU timing is captured automatically.
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

The fork is kept at exactly **one commit** ahead of
`origin/arboard-full-patch` ("Replace cryoglyph with sluggrs..."), which
makes catch-ups mechanical:

```sh
git fetch origin
git rebase --onto origin/arboard-full-patch <our-commit>^ sluggrs
brokkr check
git push --force-with-lease fork sluggrs
```

**Before analyzing, confirm the tracking ref is live.** The clone was once
single-branch: `remote.origin.fetch` covered only `master`, so
`git fetch origin arboard-full-patch` updated `FETCH_HEAD` but left
`origin/arboard-full-patch` frozen for months - which once led a whole
catch-up astray (a spurious merge of squidowl master plus a redundant
re-port of the arboard patches, force-pushed away in ad1e240a). The refspec
should be `+refs/heads/*:refs/remotes/origin/*`, but it has regressed to
master-only before, so check `git config --get-all remote.origin.fetch` and
restore it if needed. Then `git ls-remote origin arboard-full-patch` and
check the SHA matches the local tracking ref after fetching.

Upstream iced pins cosmic-text to a git rev on an experimental fork. sluggrs
must pin the identical rev (see "Deliberate version pins" in AGENTS.md), so
check whether a catch-up moved it and bump sluggrs's `Cargo.toml` to match.

Upstream **rebases and rewords** its branch, so our older copies of upstream
commits will not match by patch-id - `git cherry` reports them as new. Verify
by commit subject instead, and drop the duplicates. squidowl adapts the
arboard patches to upstream refactors themselves (e.g. the unified
`core::text` editing) - their versions are canonical; never keep our own
port of the same behavior alongside theirs. Never `git pull` (merge) the
fork after a rebase; reset to `fork/sluggrs` instead.
