# Known Rendering Artifacts

## Active

### 1. Bold 'r' arm fill (FIXED in b7bf831)

Solver threshold too low for near-degenerate perturbed curves.
Fixed by raising linear-fallback threshold to 0.5.

### 2. Ampersand '&' bottom

Reported: unfilled region at the bottom of '&' glyphs. Not yet
investigated. Likely same class as bold 'r' — near-degenerate curve
at a join or terminal. Need to dump the glyph geometry and check for
tiny segments with small `a` coefficients.

### 3. Lowercase 'a' bottom-left

Reported: artifact at the bottom-left of lowercase 'a'. Not yet
investigated. May be related to the bowl-to-stem join where curves
meet at a sharp angle.

## Investigation checklist (for both #2 and #3)

1. Dump the glyph outline (original + GPU-prepared) at the affected weight
2. Identify the specific curves in the affected region
3. Check `a` coefficients — are any near-degenerate (|a| < 0.5)?
4. If so, the threshold fix may need to be higher, or the perturbation
   epsilon needs increasing
5. If `a` is well above 0.5, this is a different bug class — likely
   band assignment or coverage combine logic
6. Use debug visualization (xcov/ycov) to isolate the failing ray direction

## Pattern

All artifacts so far have been at curve joins or small features where:
- Line segments meet curves at shallow angles
- The perturbation produces near-degenerate quadratics
- The solver's linear/quadratic threshold is too tight

The fix has consistently been ensuring the solver uses the linear path
for these cases, either by increasing perturbation or raising the threshold.
