# Verify round 6 (sign-off): symmetric exact grouping in src/border.rs

Scope: only the change since round 5, then the finding-2 verdict.
Round 5 accepted: the exact integer identity predicate (blossom
controls, den^2 scaling, i128 bounds), the fully exact i64 segment
path, the lerp fix. The two blockers were (1) asymmetric candidate
discovery via param_on's [-1, 2] range breaking equivalence-class
ownership, and (2) verified rationals converted to f32 before cut
sorting/containment.

## The change

1. Symmetry by construction: coincidence discovery now runs once per
   PAIR (i < j, both non-segment), trying `coincident_params` in both
   directions; a single exact match populates BOTH curves' span tables,
   the other side derived by exact rational affine inversion
   (`invert_params` on the common-denominator triple `(n0, n1, den)`
   returned by `exact_subspan`). If either direction can discover the
   match, both curves get exact entries - a shorter curve can never be
   blind to a longer one. Completeness of the [-1, 2] candidate range:
   for any pair with overlapping domains on the shared arc, at least
   one direction has the other's endpoints within one curve-length of
   its domain; if both directions are out of range the domains are
   disjoint on the (injective) arc and no cancellation is relevant.
2. Exact rationals to the end: `Rat` (i128 num/den, gcd-reduced,
   positive den, exact cross-multiplied Ord) carries span endpoints
   through clamping, cut collection, sort, dedup, containment, net,
   and ownership. Since cuts include every span endpoint, a window is
   exactly inside or outside each span, and containment is
   `span_lo <= window_lo && window_hi <= span_hi` - no midpoint
   arithmetic. f32 conversion happens only in `subcurve` construction
   of surviving pieces (geometry approximation, not a grouping
   decision). Magnitudes: rationalized triples have den <= 2^40;
   inversion produces num/den <= 2^43; Ord cross-multiplies once
   (~2^86), inside i128.

The prescribed regression test is added and passes: grid-exact parent
arc (0,0),(8,16),(16,0) plus its oppositely oriented [0, 0.25]
subcurve (4,6),(2,4),(0,0) at the LOWER index - the quarter cancels
(no source-0 pieces, nothing left of x 3.5) and the parent's remainder
survives.

100 unit tests, 28 GPU-gated, 4 visual snapshots pass.

## Question

Any remaining blocker to signing off WORK.md finding 2?

Do not run cargo or brokkr; read only.
