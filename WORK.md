# WORK

Generalize text borders into ordered text decorations: outline-only
(hollow) text, hard offset shadows, and blurred shadows.

Do NOT run cargo or brokkr; the orchestrator runs all builds, tests, and
formatting. Read and write code only. Do not commit. Do not touch
`repos/`, `.review.toml`, or markdown files other than this one.

## Background

`TextArea` currently carries `Option<TextBorder { color, width }>`. The
border is drawn as a solid dilated underlay of each eligible monochrome
vector glyph, in border color, before that area's fills; see the shipped
design notes in git history for the blob and certification details.

We are adding the three text-decoration features CSS authors actually
use, in this order:

1. Outline-only (hollow) text: a fill that does not paint, so only the
   outline shows. The CSS `-webkit-text-fill-color` effect.
2. Hard offset shadow: color plus `dx`/`dy`, no blur.
3. Blurred shadow: the real CSS `text-shadow`.

Deps stay pinned (wgpu 29, skrifa 0.40, cosmic-text 0.19). Pre-1.0:
breaking the public surface, including the iced-facing `prepare`
signature, is acceptable where it is the right shape.

## Prerequisite bug: border blob capacity aggregation

`text_renderer.rs` builds `border_requirements: FxHashMap<GlyphKey,
(f32, f32)>` by maximizing ppem and pixel radius INDEPENDENTLY and then
resolving that pair. `text_atlas::resolve_border_blob` derives
`required_units = wanted_bucket * units_per_em / ppem`, so pairing the
maximum ppem with the maximum pixel radius yields the SMALLEST unit
radius. A glyph appearing in one frame at two bordered sizes therefore
gets a pre-pass blob that is under-provisioned for the smaller size.

Emission then calls `resolve_border_glyph` again per instance at that
instance's real ppem, the `grid_radius_units` check fails, and the blob
is rebuilt - the same-frame supersession the pre-pass comment says it
exists to prevent.

Consequence, stated precisely: each emitted instance still receives a
descriptor that satisfied its own request at emission time, and replaced
blobs stay resident in retained texels, so no undersized grid is ever
sampled and outlines do not truncate. The damage is repeated
preparation, duplicate atlas storage, rebuilds recurring every frame,
and premature `AtlasFull`.

Fix, and do this FIRST because the decoration work builds on it:

- Aggregate two independent capacities per `GlyphKey`: the maximum ppem
  needed for boundary accuracy, and the maximum required radius IN FONT
  UNITS for distance queries (fold the existing pixel-radius bucketing
  into the unit-radius computation, or drop that redundant metadata
  consistently).
- Resolve once per key from those two independent capacities. The blob
  builder must accept them independently instead of deriving both from
  one `(ppem, radius_px)` pair.
- Emission becomes lookup-only. Delete the per-instance re-resolution at
  the three call sites.
- One blob per key is sufficient: a boundary refined at the maximum ppem
  is valid at every lower ppem, and a grid covering the maximum unit
  radius covers every smaller query. Replacement is growth-only -
  preserve existing capacities.

## Agreed design

Settled by spar; do not relitigate the mechanism. Correctness gaps in
the mechanism are still worth raising.

### Two execution kinds, not one primitive

A signed-distance field supports morphological effects (dilation,
erosion, rings) but NOT convolution. A Gaussian shadow is a convolution
of the glyph mask; a falloff over nearest-boundary distance is a
feathered dilation and differs visibly on real glyphs: counters in `e`,
`a`, `8`, `@` haze shut under a true blur but not under an SDF; thin
stems and small punctuation lose peak opacity under a true blur and do
not under an SDF; energy accumulates in the concavities of `V`, `W`,
`M`; and tightly kerned or overlapping glyphs blur as one combined mask
rather than as independent per-glyph shadows. Substituting a
Gaussian-shaped falloff `exp(-d^2/2s^2)` does not fix this - it is still
a function of one nearest distance, not an integral over coverage.

So the decoration list holds two kinds of entry:

- **Analytic decorations** (solid dilation, hard offset shadow, ring),
  which reuse the existing border blob and border pipeline.
- **Filtered shadows** (blur), which are a mask-render plus separable
  blur at AREA granularity.

### API shape (`types.rs`)

`TextArea` carries an ordered decoration list replacing
`Option<TextBorder>`. Analytic entries carry `{ color, offset: [f32; 2],
spread: f32, mode: Solid | Ring }`; filtered entries carry
`{ color, offset: [f32; 2], sigma: f32 }`.

- Today's border is `{ Solid, offset 0, spread: width }` and must render
  unchanged.
- Widths, offsets and sigma are LOGICAL pixels, multiplied by
  `TextArea::scale` exactly once on the CPU, validated on the physical
  result (non-finite, negative => that decoration is dropped).
- **Order is back-to-front, matching CSS**: the FIRST entry in a CSS
  `text-shadow` list paints on TOP of later ones. Define the list
  explicitly so authors do not get the reverse of what they expect.
- Negative `spread` (erosion) is NOT supported in this pass. Reject it
  in validation rather than leaving it to the dilation formula, which
  only handles positive growth and would need different quad
  construction.
- Fill color semantics must be stated explicitly: whether it overrides
  per-glyph rich-text colors, only replaces `default_color`, recolors
  `use_foreground` COLR layers, or applies to monochrome vector glyphs
  only. A transparent fill cannot make a COLRv1 emoji hollow and the
  ring path cannot decorate one.

### Hollow text: one combined fragment, not two draws

Ring coverage subtracted from outer coverage does NOT compose correctly
under source-over. With `o` outer, `f` fill, ring `r = o - f`, drawn
ring-then-fill, the composite alpha is `f + (o-f)(1-f) = o - f(o-f)`,
which equals `o` only when `f = 0` or `f = o`. At an inner edge with
`o = 1, f = 0.5` it gives `0.75`: a coverage deficit. Exactness in `f`
moves the error, it does not remove it.

A compensated two-draw form exists (underlay alpha
`q = b(o-f)/(1-a*f)`, then fill at `a*f`) and is algebraically valid,
but it makes the underlay depend on the specific fill that follows it,
and the area-wide underlay phase lets another glyph's fill intervene.
Rejected.

**Therefore**: ring and fill are emitted by ONE fragment that computes
both contributions and returns the disjoint partition

```
premultiplied rgb = fill_rgb * a*f + ring_rgb * b*(o-f)
alpha             = a*f + b*(o-f)
```

and the ordinary fill draw OMITS those glyphs. This supports translucent
fill rather than refusing it. Requirements:

- The border module already concatenates the whole normal shader
  (`lib.rs`), so `render_single` and the banding helpers are compiled in
  and reachable - no shared-source refactor needed. What is missing is
  that `vs_border` does not emit the fill-side varyings and `fs_border`
  never calls the evaluator.
- Matching the fill's coverage means matching its whole policy: the
  extra sampling below 16 ppem and the brightness-dependent stem
  darkening below 48 ppem, with fill color as an input.
- There is a real coordinate mismatch to reproduce: `vs_main` divides
  its half-pixel UV expansion by `max(screen_rect.zw, 1)` while
  `vs_border` divides by the actual dimensions with a near-zero guard,
  so sub-pixel glyph dimensions get different interpolated coordinates
  and derivatives.
- `f <= o` is NOT guaranteed at small spread with stem darkening. Apply
  an explicit nesting rule `effective_outer = max(sdf_outer, f)`,
  accepting that it can enlarge the effective outer edge.
- For an OPAQUE fill the shipped solid underlay is already exactly
  right; Ring is only required for zero-alpha or translucent fills.
- A zero-alpha normal draw should be omitted rather than emitted, since
  zero color output does not disable depth or stencil side effects.

### Offset shadows

- Offset MUST NOT enter the blob radius requirement. It translates the
  quad; it does not change which glyph-space boundary is nearest. The
  radius requirement stays `spread + AA support`.
- `vs_border` currently dilates by exactly `width_px + 0.5`. The quad
  dilation and texcoord expansion must use the decoration's COMPLETE
  finite support or the fragment falloff is clipped at the quad edge.
- Because border instances carry the same `screen_rect` as the fill and
  dilate in the vertex shader, moving offset and dilation into the
  per-draw uniform lets ALL analytic decorations of an area draw the
  SAME instance range with only a uniform rebind. Do not duplicate the
  instance stream per decoration.

### Blur: encoding phase

There is currently nowhere to encode blur passes: `prepare_with_depth`
takes `&CommandEncoder` (immutable, unused) and `render` takes
`&mut RenderPass`. A pass cannot begin on a shared encoder, nor inside
an active pass.

**Change `prepare` and `prepare_with_depth` to take
`&mut CommandEncoder`**, and encode the mask and blur passes there,
after preparation and resource uploads complete. `render` then
composites the prepared shadow texture inside the caller's pass. This
keeps the existing division (preparation produces what rendering
consumes) and leaves submission ordering with the caller. Creating a
private encoder inside `prepare` is possible but forfeits that ordering
against unsubmitted caller work; rejected.

The iced fork in `repos/iced/` calls this surface and must be updated to
match. It is a path dependency, so no rev bump is involved.

Correctness details:

- Build the mask from every source glyph that can contribute THROUGH the
  kernel, including sources outside the area bounds. Clip the final
  shadow to the area bounds; clipping the source mask first cuts off
  contributions near the edge.
- A Gaussian has infinite support. Define an explicit finite-support
  cutoff proportional to sigma; culling, texture sizing, and allocation
  all derive from it.
- Intermediate textures and their inputs must stay valid until the
  encoded work executes; a second `prepare` before submission must not
  reuse that storage or overwrite those uniforms.
- Per-glyph blur is a DIFFERENT semantic and cannot silently substitute:
  independently blurred glyphs composited source-over do not equal a
  blur of the combined mask.

### Culling and the retained cache

- Culling needs DIRECTIONAL extents, not one scalar margin. For a
  decoration with offset `(dx, dy)` and support radius `r`:
  `left = max(0, r - dx)`, `right = max(0, r + dx)`,
  `top = max(0, r - dy)`, `bottom = max(0, r + dy)` (positive `dy`
  down), unioned component-wise across the list. A scalar
  `r + max(|dx|, |dy|)` is conservative but wrong as the cache
  invariant: flipping offset DIRECTION at constant magnitude reveals
  candidates on the opposite side.
- `run_is_visible` takes a symmetric vertical margin today and must take
  top and bottom extents separately; `vector_rect_visible` and
  `re_cull_vector_instances` need all four.
- Keep candidate-envelope validity and blob-capacity validity as
  SEPARATE checks. An unchanged envelope does not prove descriptor
  validity: a far-offset narrow decoration and a centered wide one can
  share an envelope while needing different grid radii. Conversely a
  changed envelope only forces a fresh walk when the retained candidates
  cannot prove coverage of the new envelope - a complete cache can be
  re-culled.
- Fill color is NOT paint-only. `fs_main` derives stem darkening from
  fill RGB brightness, so changing fill color can change COVERAGE, not
  just paint. Any cache path treating color as a uniform-only update is
  wrong.
- Decoration order and uniform order are part of the rebuilt draw
  metadata even when no vertex upload happens.
- Depth: analytic decoration draws test depth without writing. Multiple
  draws at one glyph depth interact with earlier area fills and
  caller-owned depth. A filtered-shadow composite has no unique
  source-glyph depth once masks overlap; state its depth semantic
  explicitly.

### Known pre-existing hole: the global raster tail

`render()` collects every area's raster-fallback glyphs and draws them
AFTER all vector draws, so area A's bitmap glyph already lands over area
B's fill. Decorations widen the consequences (area B's shadow cannot sit
beneath area B's raster glyph while respecting A/B order) but do not
create the bug. Fixing it means per-area raster ranges inside the same
ordered draw graph. If decorations stay monochrome-vector-only, say so
in the API docs; that still does not repair cross-area raster ordering.
Record it; do not silently rely on the current ordering.

## Recorded, not fixed

The solid underlay is not an exact disjoint partition where BOTH
coverages are partial: `f + o(1-f)` can exceed `o`. Under an idealized
shared distance ramp, `spread >= 1` physical pixel guarantees `o = 1`
wherever `f > 0` and the artifact vanishes. That bound does NOT transfer
strictly to the shipped shaders, which use analytic ray coverage plus
optional extra samples on one side and an approximate Euclidean boundary
distance on the other. The honest statement: the artifact is
concentrated at narrow borders and the ideal threshold is one physical
pixel, times `TextArea::scale`.
