//! CPU preparation for lazily-created per-glyph border blobs.

use crate::band::{BandScratch, CurveLocation, build_border_bands};
use crate::outline::{GlyphOutline, QuadCurve};
use crate::prep::pack_i16_pair;
use rustc_hash::FxHashSet;

/// Screen-space accuracy used when isolating filled-set boundary events.
pub const BOUNDARY_ERROR_PX: f32 = 0.25;
/// Wider borders use the shader's all-boundary brute force path.
pub const MAX_GRID_RADIUS_EM: f32 = 1.0;
const MAX_SUBDIVISION_DEPTH: u32 = 18;

#[derive(Clone, Copy, Debug)]
pub struct BoundaryPiece {
    pub curve: QuadCurve,
    pub source_curve: u32,
    pub t0: f32,
    pub t1: f32,
}

#[derive(Clone, Debug)]
pub struct DistanceGrid {
    pub bounds: [f32; 4],
    pub columns: u16,
    pub rows: u16,
    pub radius: f32,
    pub offsets: Vec<u32>,
    pub candidates: Vec<u32>,
    pub brute_force: bool,
}

impl DistanceGrid {
    /// Outside the represented domain the shader must use every boundary
    /// piece. Inside it, this returns the conservative cell candidate list.
    pub fn candidates_at(&self, point: [f32; 2], piece_count: usize) -> GridCandidates<'_> {
        if self.brute_force
            || point[0] < self.bounds[0]
            || point[1] < self.bounds[1]
            || point[0] > self.bounds[2]
            || point[1] > self.bounds[3]
        {
            return GridCandidates::All(0..piece_count as u32);
        }
        let width = (self.bounds[2] - self.bounds[0]).max(f32::EPSILON);
        let height = (self.bounds[3] - self.bounds[1]).max(f32::EPSILON);
        let x = (((point[0] - self.bounds[0]) / width) * f32::from(self.columns))
            .floor()
            .clamp(0.0, f32::from(self.columns - 1)) as usize;
        let y = (((point[1] - self.bounds[1]) / height) * f32::from(self.rows))
            .floor()
            .clamp(0.0, f32::from(self.rows - 1)) as usize;
        let cell = y * usize::from(self.columns) + x;
        GridCandidates::Slice(
            &self.candidates[self.offsets[cell] as usize..self.offsets[cell + 1] as usize],
        )
    }
}

pub enum GridCandidates<'a> {
    Slice(&'a [u32]),
    All(std::ops::Range<u32>),
}

#[derive(Clone, Copy, Debug)]
pub struct BorderDescriptor {
    pub fill_offset: u32,
    pub winding_offset: u32,
    pub grid_offset: u32,
    pub boundary_offset: u32,
    pub boundary_count: u32,
    /// Distance-query capacity in FONT UNITS. Infinite for a brute-force
    /// blob, which loops every boundary piece and so serves any radius.
    /// Font units, not pixels: the same pixel radius needs a larger unit
    /// radius at a smaller ppem, so a pixel capacity cannot be compared
    /// across the sizes one glyph is drawn at within a single frame.
    pub grid_radius_units: f32,
    pub ppem_ceiling: f32,
}

#[derive(Clone, Debug)]
pub struct PreparedBorder {
    pub descriptor: BorderDescriptor,
    pub data: Vec<i32>,
    pub texel_len: u32,
    pub boundary: Vec<BoundaryPiece>,
    pub grid: DistanceGrid,
}

fn q(v: f32) -> f32 {
    (v * 4.0).round() * 0.25
}

pub fn quantized_outline(outline: &GlyphOutline) -> GlyphOutline {
    let curves = outline
        .curves
        .iter()
        .map(|c| QuadCurve {
            p1: [q(c.p1[0]), q(c.p1[1])],
            p2: [q(c.p2[0]), q(c.p2[1])],
            p3: [q(c.p3[0]), q(c.p3[1])],
        })
        .collect::<Vec<_>>();
    let mut bounds = [f32::MAX, f32::MAX, f32::MIN, f32::MIN];
    for c in &curves {
        for p in [c.p1, c.p2, c.p3] {
            bounds[0] = bounds[0].min(p[0]);
            bounds[1] = bounds[1].min(p[1]);
            bounds[2] = bounds[2].max(p[0]);
            bounds[3] = bounds[3].max(p[1]);
        }
    }
    GlyphOutline { curves, bounds }
}

fn eval(c: QuadCurve, t: f32) -> [f32; 2] {
    let u = 1.0 - t;
    [
        u * u * c.p1[0] + 2.0 * u * t * c.p2[0] + t * t * c.p3[0],
        u * u * c.p1[1] + 2.0 * u * t * c.p2[1] + t * t * c.p3[1],
    ]
}

fn split(c: QuadCurve) -> (QuadCurve, QuadCurve) {
    let a = [(c.p1[0] + c.p2[0]) * 0.5, (c.p1[1] + c.p2[1]) * 0.5];
    let b = [(c.p2[0] + c.p3[0]) * 0.5, (c.p2[1] + c.p3[1]) * 0.5];
    let m = [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5];
    (
        QuadCurve {
            p1: c.p1,
            p2: a,
            p3: m,
        },
        QuadCurve {
            p1: m,
            p2: b,
            p3: c.p3,
        },
    )
}

/// Endpoints of a curve whose trace is a straight segment. `line_to` emits
/// lines as degenerate quads with the control point on an endpoint; the
/// parameterization is then non-uniform (t^2), so line curves must be
/// classified on the trace, not on the quadratic's roots and derivative.
fn as_segment(c: QuadCurve) -> Option<([f32; 2], [f32; 2])> {
    (c.p2 == c.p1 || c.p2 == c.p3).then_some((c.p1, c.p3))
}

fn winding_at(curves: &[QuadCurve], p: [f32; 2]) -> i32 {
    let mut winding = 0;
    for c in curves {
        if let Some((a, b)) = as_segment(*c) {
            if a[1] == b[1] {
                continue;
            }
            // Half-open ownership: the start vertex belongs to this curve,
            // the end vertex to its successor.
            let upward = a[1] <= p[1] && p[1] < b[1];
            let downward = b[1] <= p[1] && p[1] < a[1];
            let s = (p[1] - a[1]) / (b[1] - a[1]);
            if (upward || downward) && a[0] + (b[0] - a[0]) * s > p[0] {
                winding += if upward { 1 } else { -1 };
            }
            continue;
        }
        let ay = c.p1[1] - 2.0 * c.p2[1] + c.p3[1];
        let by = 2.0 * (c.p2[1] - c.p1[1]);
        let cy = c.p1[1] - p[1];
        let mut roots = [f32::NAN; 2];
        let count = if ay.abs() < 1e-7 {
            if by.abs() < 1e-7 {
                0
            } else {
                roots[0] = -cy / by;
                1
            }
        } else {
            let d = by * by - 4.0 * ay * cy;
            if d < 0.0 {
                0
            } else {
                let s = d.sqrt();
                roots[0] = (-by - s) / (2.0 * ay);
                roots[1] = (-by + s) / (2.0 * ay);
                2
            }
        };
        for &t in &roots[..count] {
            // Half-open ownership: a contour vertex belongs to its outgoing
            // interval only. This is also the rule used by corrected bands.
            if !(0.0..=1.0).contains(&t) || eval(*c, t)[0] <= p[0] {
                continue;
            }
            let dy = 2.0 * ((1.0 - t) * (c.p2[1] - c.p1[1]) + t * (c.p3[1] - c.p2[1]));
            let owns = if dy > 0.0 {
                t < 1.0
            } else {
                dy < 0.0 && t > 0.0
            };
            if !owns {
                continue;
            }
            winding += if dy > 0.0 {
                1
            } else if dy < 0.0 {
                -1
            } else {
                0
            };
        }
    }
    winding
}

fn deriv(c: QuadCurve, u: f32) -> [f32; 2] {
    [
        2.0 * ((1.0 - u) * (c.p2[0] - c.p1[0]) + u * (c.p3[0] - c.p2[0])),
        2.0 * ((1.0 - u) * (c.p2[1] - c.p1[1]) + u * (c.p3[1] - c.p2[1])),
    ]
}

/// The exact quadratic sub-span of `c` over `[lo, hi]` (de Casteljau).
fn subcurve(c: QuadCurve, lo: f32, hi: f32) -> QuadCurve {
    if lo == 0.0 && hi == 1.0 {
        return c;
    }
    let p1 = eval(c, lo);
    let d = deriv(c, lo);
    QuadCurve {
        p1,
        p2: [
            p1[0] + (hi - lo) * d[0] * 0.5,
            p1[1] + (hi - lo) * d[1] * 0.5,
        ],
        p3: eval(c, hi),
    }
}

/// Parameter `u` (allowed slightly outside [0, 1]) where `c` passes through
/// `p` within `eps` on both axes, if any.
fn param_on(c: QuadCurve, p: [f32; 2], eps: f32) -> Option<f32> {
    let mut candidates = [f32::NAN; 4];
    let mut count = 0;
    for axis in 0..2 {
        let a = c.p1[axis] - 2.0 * c.p2[axis] + c.p3[axis];
        let b = 2.0 * (c.p2[axis] - c.p1[axis]);
        let k = c.p1[axis] - p[axis];
        if a.abs() < 1e-7 {
            if b.abs() >= 1e-7 {
                candidates[count] = -k / b;
                count += 1;
            }
        } else {
            let d = b * b - 4.0 * a * k;
            if d >= 0.0 {
                let s = d.sqrt();
                candidates[count] = (-b - s) / (2.0 * a);
                candidates[count + 1] = (-b + s) / (2.0 * a);
                count += 2;
            }
        }
    }
    candidates[..count].iter().copied().find(|&u| {
        u.is_finite() && (-1.0..=2.0).contains(&u) && {
            let q = eval(c, u);
            (q[0] - p[0]).abs() <= eps && (q[1] - p[1]).abs() <= eps
        }
    })
}

/// Candidate slack for float endpoint-parameter generation. Generosity is
/// safe: every candidate must pass the EXACT integer polynomial identity in
/// `exact_subspan` before it can cancel anything.
const CANDIDATE_EPS: f32 = 0.1;

/// Exact rational parameter, kept exact through cut sorting, window
/// containment, net calculation, and ownership grouping - converted to f32
/// only when the surviving subcurve geometry is finally constructed.
/// Magnitudes stay far inside i128: rationalized parameters have
/// denominators <= 2^40, inversion squares that at worst, and comparisons
/// cross-multiply once.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Rat {
    n: i128,
    d: i128,
}

impl Rat {
    const ZERO: Rat = Rat { n: 0, d: 1 };
    const ONE: Rat = Rat { n: 1, d: 1 };
    fn new(n: i128, d: i128) -> Rat {
        let sign = if d < 0 { -1 } else { 1 };
        let (mut n, mut d) = (n * sign, d * sign);
        let g = gcd(n.unsigned_abs(), d.unsigned_abs());
        if g > 1 {
            n /= g as i128;
            d /= g as i128;
        }
        Rat { n, d }
    }
    fn to_f32(self) -> f32 {
        (self.n as f64 / self.d as f64) as f32
    }
}

impl Ord for Rat {
    fn cmp(&self, other: &Rat) -> std::cmp::Ordering {
        (self.n * other.d).cmp(&(other.n * self.d))
    }
}

impl PartialOrd for Rat {
    fn partial_cmp(&self, other: &Rat) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.max(1)
}

/// One direction of exact coincidence discovery: `other`'s endpoints as
/// exact parameters on `base`'s arc over a common denominator
/// `(n0, n1, den)`, certified by `exact_subspan`. Float arithmetic only
/// GENERATES the candidates; nothing matches without the exact integer
/// identity. Keeping the raw triple (denominators <= 2^40) means the
/// affine inversion below stays a small pure-integer ratio.
fn coincident_params(base: QuadCurve, other: QuadCurve) -> Option<(i64, i64, i64)> {
    let u0 = param_on(base, other.p1, CANDIDATE_EPS)?;
    let u1 = param_on(base, other.p3, CANDIDATE_EPS)?;
    if (u1 - u0).abs() <= f32::EPSILON {
        return None;
    }
    exact_subspan(base, other, u0, u1)
}

/// The exact affine-inverse endpoints: given `other = base o phi` with
/// phi(0) = n0/den, phi(1) = n1/den, `base`'s [0, 1] endpoints in
/// `other`'s parameterization are phi^-1(0) and phi^-1(1).
fn invert_params((n0, n1, den): (i64, i64, i64)) -> (Rat, Rat) {
    let n0 = i128::from(n0);
    let n1 = i128::from(n1);
    let den = i128::from(den);
    (Rat::new(-n0, n1 - n0), Rat::new(den - n0, n1 - n0))
}

/// Sub-span of `[0, 1]` covered, with relative orientation - or None when
/// the exact span misses the domain.
fn clamp_span(u0: Rat, u1: Rat) -> Option<(Rat, Rat, i32)> {
    let lo = u0.min(u1).max(Rat::ZERO);
    let hi = u0.max(u1).min(Rat::ONE);
    (lo < hi).then_some((lo, hi, if u1 > u0 { 1 } else { -1 }))
}

/// Extract the boundary of the nonzero-filled set from the exact quantized
/// curves consumed by the shader. Fails (conservatively, never silently
/// dropping certified-unknown geometry) when certification exhausts its
/// subdivision depth or the piece budget.
pub fn extract_boundary(
    outline: &GlyphOutline,
    max_ppem: f32,
    units_per_em: f32,
) -> Result<Vec<BoundaryPiece>, crate::types::PrepareError> {
    let quantized = quantized_outline(outline);
    let tolerance = BOUNDARY_ERROR_PX * units_per_em / max_ppem.max(f32::MIN_POSITIVE);
    // Coincident-arc discovery runs once per PAIR, trying both directions,
    // and a single exact match populates BOTH curves' span tables via exact
    // rational inversion. Discovery is therefore symmetric by construction
    // (float candidate range limits cannot make one side blind), which the
    // min-index ownership rule below requires. Spans, cuts, and containment
    // stay exact rationals until subcurve construction.
    let mut arcs: Vec<Vec<(Rat, Rat, i32, usize)>> = vec![Vec::new(); quantized.curves.len()];
    for i in 0..quantized.curves.len() {
        if as_segment(quantized.curves[i]).is_some() {
            continue;
        }
        for j in i + 1..quantized.curves.len() {
            if as_segment(quantized.curves[j]).is_some() {
                continue;
            }
            let ci = quantized.curves[i];
            let cj = quantized.curves[j];
            // Forward: j's endpoints on i's arc; else reverse and invert.
            let (span_i, span_j) = if let Some(fwd) = coincident_params(ci, cj) {
                let (u0, u1) = (
                    Rat::new(i128::from(fwd.0), i128::from(fwd.2)),
                    Rat::new(i128::from(fwd.1), i128::from(fwd.2)),
                );
                let (v0, v1) = invert_params(fwd);
                ((u0, u1), (v0, v1))
            } else if let Some(rev) = coincident_params(cj, ci) {
                let (v0, v1) = (
                    Rat::new(i128::from(rev.0), i128::from(rev.2)),
                    Rat::new(i128::from(rev.1), i128::from(rev.2)),
                );
                let (u0, u1) = invert_params(rev);
                ((u0, u1), (v0, v1))
            } else {
                continue;
            };
            if let Some((lo, hi, orientation)) = clamp_span(span_i.0, span_i.1) {
                arcs[i].push((lo, hi, orientation, j));
            }
            if let Some((lo, hi, orientation)) = clamp_span(span_j.0, span_j.1) {
                arcs[j].push((lo, hi, orientation, i));
            }
        }
    }
    let mut out = Vec::new();
    for (i, &curve) in quantized.curves.iter().enumerate() {
        if let Some((a, b)) = as_segment(curve) {
            // All coincidence decisions run in EXACT integer arithmetic on
            // the 4x-scaled quarter-unit grid: collinearity is an exact
            // cross product, and coincident spans live on the shared line
            // as integer scalars (P - a) . d. Distinct quantized traces
            // therefore never cancel; only identical ones do.
            let ai = int_point(a);
            let bi = int_point(b);
            let di = [bi[0] - ai[0], bi[1] - ai[1]];
            let dd = di[0] * di[0] + di[1] * di[1];
            if dd == 0 {
                continue;
            }
            let mut covers = Vec::new();
            for (j, &other) in quantized.curves.iter().enumerate() {
                if j == i {
                    continue;
                }
                let Some((o0, o1)) = as_segment(other) else {
                    continue;
                };
                let o0i = int_point(o0);
                let o1i = int_point(o1);
                let cross0 = di[0] * (o0i[1] - ai[1]) - di[1] * (o0i[0] - ai[0]);
                let cross1 = di[0] * (o1i[1] - ai[1]) - di[1] * (o1i[0] - ai[0]);
                if cross0 != 0 || cross1 != 0 {
                    continue;
                }
                let s0 = (o0i[0] - ai[0]) * di[0] + (o0i[1] - ai[1]) * di[1];
                let s1 = (o1i[0] - ai[0]) * di[0] + (o1i[1] - ai[1]) * di[1];
                if s0 == s1 {
                    continue;
                }
                let orientation = if s1 > s0 { 1i32 } else { -1 };
                covers.push((s0.min(s1), s0.max(s1), orientation, j));
            }
            let mut cuts = vec![0i64, dd];
            for &(s0, s1, _, _) in &covers {
                for s in [s0, s1] {
                    if 0 < s && s < dd {
                        cuts.push(s);
                    }
                }
            }
            cuts.sort_unstable();
            cuts.dedup();
            for window in cuts.windows(2) {
                let s_lo = window[0];
                let s_hi = window[1];
                // Window midpoint doubled stays integer, so span containment
                // below is exact too.
                let m2 = s_lo + s_hi;
                let mut net = 1i32;
                let mut owner = i;
                let mut coincident = Vec::new();
                for &(s0, s1, orientation, j) in &covers {
                    if 2 * s0 <= m2 && m2 <= 2 * s1 {
                        net += orientation;
                        owner = owner.min(j);
                        coincident.push(j);
                    }
                }
                if owner == i && net != 0 {
                    let f_lo = s_lo as f32 / dd as f32;
                    let f_hi = s_hi as f32 / dd as f32;
                    // Cut scalars are linear positions along the trace, but a
                    // degenerate quad parameterizes its chord as t^2 - so the
                    // window endpoints must be lerped, not eval'd.
                    let p0 = [a[0] + (b[0] - a[0]) * f_lo, a[1] + (b[1] - a[1]) * f_lo];
                    let p1 = [a[0] + (b[0] - a[0]) * f_hi, a[1] + (b[1] - a[1]) * f_hi];
                    certify_piece(
                        &quantized.curves,
                        line_curve(p0, p1),
                        i as u32,
                        f_lo,
                        f_hi,
                        tolerance,
                        0,
                        &coincident,
                        &mut out,
                    )?;
                }
            }
            continue;
        }
        // Curved path: mirror the segment cut-and-net treatment on the
        // quadratic arc, so equivalent traces with DIFFERENT segmentation
        // cancel (or merge) explicitly instead of reaching certification
        // as unresolvable coincident geometry. Every span endpoint is an
        // exact rational, and cuts include all of them, so each window is
        // exactly inside or outside every span - containment compares the
        // window's endpoints, no midpoint arithmetic needed.
        let mut cuts = vec![Rat::ZERO, Rat::ONE];
        for &(lo, hi, _, _) in &arcs[i] {
            for u in [lo, hi] {
                if Rat::ZERO < u && u < Rat::ONE {
                    cuts.push(u);
                }
            }
        }
        cuts.sort_unstable();
        cuts.dedup();
        for window in cuts.windows(2) {
            let w_lo = window[0];
            let w_hi = window[1];
            let mut net = 1i32;
            let mut owner = i;
            let mut coincident = Vec::new();
            for &(lo, hi, orientation, j) in &arcs[i] {
                if lo <= w_lo && w_hi <= hi {
                    net += orientation;
                    owner = owner.min(j);
                    coincident.push(j);
                }
            }
            if owner == i && net != 0 {
                let f_lo = w_lo.to_f32();
                let f_hi = w_hi.to_f32();
                certify_piece(
                    &quantized.curves,
                    subcurve(curve, f_lo, f_hi),
                    i as u32,
                    f_lo,
                    f_hi,
                    tolerance,
                    0,
                    &coincident,
                    &mut out,
                )?;
            }
        }
    }
    Ok(out)
}

fn line_curve(a: [f32; 2], b: [f32; 2]) -> QuadCurve {
    QuadCurve {
        p1: a,
        p2: a,
        p3: b,
    }
}

fn refine_for_distance(
    piece: BoundaryPiece,
    ppem_ceiling: f32,
    units_per_em: f32,
    out: &mut Vec<BoundaryPiece>,
) -> Result<(), crate::types::PrepareError> {
    let c = piece.curve;
    // A quadratic's maximum deviation from each of sixteen equal-parameter
    // chords is bounded by |p1 - 2*p2 + p3| / (4*16^2).
    let second = (c.p1[0] - 2.0 * c.p2[0] + c.p3[0]).hypot(c.p1[1] - 2.0 * c.p2[1] + c.p3[1]);
    let error_px = second * ppem_ceiling / units_per_em / 1024.0;
    if error_px <= BOUNDARY_ERROR_PX {
        // The piece budget covers the FINAL refined vector, so the blob
        // size the atlas sees is actually bounded by it.
        return push_piece(c, piece.source_curve, piece.t0, piece.t1, out);
    }
    let (a, b) = split(c);
    let tm = (piece.t0 + piece.t1) * 0.5;
    refine_for_distance(
        BoundaryPiece {
            curve: a,
            t1: tm,
            ..piece
        },
        ppem_ceiling,
        units_per_em,
        out,
    )?;
    refine_for_distance(
        BoundaryPiece {
            curve: b,
            t0: tm,
            ..piece
        },
        ppem_ceiling,
        units_per_em,
        out,
    )
}

/// Retaining an unresolved sub-tolerance piece charges `probe + diameter`
/// against the screen-space budget, so both are half the tolerance.
const UNRESOLVED_FRACTION: f32 = 0.5;
/// A quantized coordinate on the exact 4x-scaled integer grid. Quantized
/// values are quarter-unit multiples well inside f32's exact-integer range,
/// so this conversion is lossless.
fn qi(v: f32) -> i64 {
    (v * 4.0).round() as i64
}

fn int_point(p: [f32; 2]) -> [i64; 2] {
    [qi(p[0]), qi(p[1])]
}

/// Best rational approximation p/q of `u` with q bounded, via continued
/// fractions. Coincidence parameters of exactly matching quantized traces
/// are rationals with small denominators; anything the bound misses only
/// fails verification, which is conservative.
fn rationalize(u: f32) -> Option<(i64, i64)> {
    const MAX_DEN: i64 = 1 << 20;
    let target = f64::from(u);
    let mut x = target;
    let (mut p0, mut q0, mut p1, mut q1) = (1i64, 0i64, x.floor() as i64, 1i64);
    for _ in 0..40 {
        if f64::abs(q1 as f64 * target - p1 as f64) <= 1e-9 * q1 as f64 {
            break;
        }
        let frac = x - x.floor();
        if frac.abs() < 1e-12 {
            break;
        }
        x = 1.0 / frac;
        let a = x.floor() as i64;
        let p2 = a.checked_mul(p1)?.checked_add(p0)?;
        let q2 = a.checked_mul(q1)?.checked_add(q0)?;
        if q2 > MAX_DEN {
            break;
        }
        (p0, q0, p1, q1) = (p1, q1, p2, q2);
    }
    (q1 > 0).then_some((p1, q1))
}

/// Exact verification that `other` equals the sub-span of `base` over the
/// rationalizations of `[u0, u1]`, by polynomial identity over the 4x
/// integer grid (i128 arithmetic; the de Casteljau sub-span controls are
/// the blossom values b(u0,u0), b(u0,u1), b(u1,u1)). Only an exact match
/// returns the snapped parameters (as `(n0, n1, den)` over a common
/// denominator), so approximate float candidates can never cancel a
/// distinct trace.
fn exact_subspan(base: QuadCurve, other: QuadCurve, u0: f32, u1: f32) -> Option<(i64, i64, i64)> {
    let (p0, q0) = rationalize(u0)?;
    let (p1, q1) = rationalize(u1)?;
    let den = q0.checked_mul(q1)?;
    let sn0 = p0.checked_mul(q1)?;
    let sn1 = p1.checked_mul(q0)?;
    let n0 = i128::from(sn0);
    let n1 = i128::from(sn1);
    let d = i128::from(den);
    let w0 = d - n0;
    let w1 = d - n1;
    for axis in 0..2 {
        let c0 = i128::from(qi(base.p1[axis]));
        let c1 = i128::from(qi(base.p2[axis]));
        let c2 = i128::from(qi(base.p3[axis]));
        let dd = d * d;
        let start = w0 * w0 * c0 + 2 * n0 * w0 * c1 + n0 * n0 * c2;
        let control = w1 * (w0 * c0 + n0 * c1) + n1 * (w0 * c1 + n0 * c2);
        let end = w1 * w1 * c0 + 2 * n1 * w1 * c1 + n1 * n1 * c2;
        if start != i128::from(qi(other.p1[axis])) * dd
            || control != i128::from(qi(other.p2[axis])) * dd
            || end != i128::from(qi(other.p3[axis])) * dd
        {
            return None;
        }
    }
    Some((sn0, sn1, den))
}
/// Hard budget on certified boundary pieces per glyph. Certification fails
/// conservatively past this, instead of building an atlas-busting blob.
const MAX_BOUNDARY_PIECES: usize = 1 << 16;

#[allow(clippy::too_many_arguments)]
fn certify_piece(
    all: &[QuadCurve],
    c: QuadCurve,
    source: u32,
    t0: f32,
    t1: f32,
    tol: f32,
    depth: u32,
    excluded: &[usize],
    out: &mut Vec<BoundaryPiece>,
) -> Result<(), crate::types::PrepareError> {
    let mid = eval(c, 0.5);
    let tangent = [c.p3[0] - c.p1[0], c.p3[1] - c.p1[1]];
    let len = (tangent[0] * tangent[0] + tangent[1] * tangent[1]).sqrt();
    if len <= f32::EPSILON {
        return Ok(());
    }
    let probe = tol * UNRESOLVED_FRACTION;
    let normal = [-tangent[1] / len * probe, tangent[0] / len * probe];
    let inside_pos = winding_at(all, [mid[0] + normal[0], mid[1] + normal[1]]) != 0;
    let inside_neg = winding_at(all, [mid[0] - normal[0], mid[1] - normal[1]]) != 0;
    let different = inside_pos != inside_neg;
    // A degenerate-line quad traces its chord exactly, so its flatness
    // deviation is zero regardless of where the control point sits.
    let flat = if as_segment(c).is_some() {
        0.0
    } else {
        ((c.p2[0] - (c.p1[0] + c.p3[0]) * 0.5).powi(2)
            + (c.p2[1] - (c.p1[1] + c.p3[1]) * 0.5).powi(2))
        .sqrt()
    };
    let bounds = curve_bounds(c);
    // Curves whose trace was certified coincident with this piece's span
    // are excluded from the separation test: their multiplicity is already
    // accounted in the extraction net, and a coincident trace cannot move
    // the boundary away from itself.
    let separated = all.iter().enumerate().all(|(index, other)| {
        index == source as usize
            || excluded.contains(&index)
            || clear_of_curve(bounds, *other, probe, CLEARANCE_DEPTH)
    });
    if different && flat <= tol && separated {
        push_piece(c, source, t0, t1, out)?;
        return Ok(());
    }
    // Agreement of the probes is only a suggestion. It becomes a drop
    // certificate when every other control hull is conservatively separated
    // from the swept probe neighborhood, so no classification event can lie
    // in this parameter interval.
    if !different && flat <= tol && separated {
        return Ok(());
    }
    let diameter = (bounds[2] - bounds[0]).hypot(bounds[3] - bounds[1]);
    if diameter <= tol * UNRESOLVED_FRACTION {
        // Unresolved at sub-tolerance size. A nonzero winding sample within
        // `probe` of the midpoint is an existence witness for filled matter,
        // so the whole piece lies within probe + diameter <= tol of the
        // filled set and retaining it stays inside the declared budget. The
        // ladder of shrinking offsets rescues thin cusps the outermost pair
        // straddles. With no witness the piece can be neither kept (phantom
        // border in empty space, magnified by dilation) nor dropped (it may
        // bound a sub-probe real component whose border dilation is equally
        // magnified), so certification fails conservatively.
        for scale in [1.0f32, 0.25, 0.0625] {
            for side in [1.0f32, -1.0] {
                let p = [
                    mid[0] + normal[0] * scale * side,
                    mid[1] + normal[1] * scale * side,
                ];
                if winding_at(all, p) != 0 {
                    push_piece(c, source, t0, t1, out)?;
                    return Ok(());
                }
            }
        }
        return Err(crate::types::PrepareError::AtlasFull);
    }
    if depth >= MAX_SUBDIVISION_DEPTH {
        // Neither keeping (above tolerance) nor dropping (no certificate)
        // is sound here, so certification fails conservatively and the
        // caller rejects the blob.
        return Err(crate::types::PrepareError::AtlasFull);
    }
    let (a, b) = split(c);
    let tm = (t0 + t1) * 0.5;
    certify_piece(all, a, source, t0, tm, tol, depth + 1, excluded, out)?;
    certify_piece(all, b, source, tm, t1, tol, depth + 1, excluded, out)
}

fn push_piece(
    c: QuadCurve,
    source: u32,
    t0: f32,
    t1: f32,
    out: &mut Vec<BoundaryPiece>,
) -> Result<(), crate::types::PrepareError> {
    if out.len() >= MAX_BOUNDARY_PIECES {
        return Err(crate::types::PrepareError::AtlasFull);
    }
    out.push(BoundaryPiece {
        curve: c,
        source_curve: source,
        t0,
        t1,
    });
    Ok(())
}

const CLEARANCE_DEPTH: u32 = 6;

/// Conservative clearance test between a piece's bounds and another curve's
/// trace. A whole-curve AABB is far too coarse - a long diagonal's box
/// covers most of the glyph - so on overlap the OTHER curve is subdivided
/// and each half retested; only if some leaf box still overlaps at the
/// depth limit is the pair declared unseparated. False negatives
/// (declaring overlap where there is none) are conservative: they merely
/// force further subdivision of the piece.
fn clear_of_curve(piece_bounds: [f32; 4], other: QuadCurve, margin: f32, depth: u32) -> bool {
    if !bounds_overlap(piece_bounds, curve_bounds(other), margin) {
        return true;
    }
    if depth == 0 {
        return false;
    }
    let (a, b) = split(other);
    clear_of_curve(piece_bounds, a, margin, depth - 1)
        && clear_of_curve(piece_bounds, b, margin, depth - 1)
}

fn curve_bounds(c: QuadCurve) -> [f32; 4] {
    [
        c.p1[0].min(c.p2[0]).min(c.p3[0]),
        c.p1[1].min(c.p2[1]).min(c.p3[1]),
        c.p1[0].max(c.p2[0]).max(c.p3[0]),
        c.p1[1].max(c.p2[1]).max(c.p3[1]),
    ]
}

fn bounds_overlap(a: [f32; 4], b: [f32; 4], margin: f32) -> bool {
    a[2] + margin >= b[0] && a[0] - margin <= b[2] && a[3] + margin >= b[1] && a[1] - margin <= b[3]
}

pub fn radius_bucket(radius_units: f32, units_per_em: f32) -> Option<f32> {
    if radius_units > units_per_em * MAX_GRID_RADIUS_EM {
        return None;
    }
    let quantum = (units_per_em / 64.0).max(0.25);
    Some(next_power_of_two_f32((radius_units / quantum).max(1.0)) * quantum)
}

fn next_power_of_two_f32(value: f32) -> f32 {
    2.0_f32.powf(value.log2().ceil())
}

pub fn build_distance_grid(
    pieces: &[BoundaryPiece],
    bounds: [f32; 4],
    radius: f32,
    units_per_em: f32,
) -> DistanceGrid {
    let Some(radius) = radius_bucket(radius, units_per_em) else {
        return DistanceGrid {
            bounds,
            columns: 1,
            rows: 1,
            radius,
            offsets: vec![0, 0],
            candidates: Vec::new(),
            brute_force: true,
        };
    };
    let aspect = ((bounds[2] - bounds[0]).max(1.0) / (bounds[3] - bounds[1]).max(1.0)).sqrt();
    let columns = (8.0 * aspect).round().clamp(1.0, 32.0) as u16;
    let rows = (8.0 / aspect).round().clamp(1.0, 32.0) as u16;
    let mut offsets = Vec::with_capacity(usize::from(columns) * usize::from(rows) + 1);
    let mut candidates = Vec::new();
    offsets.push(0);
    let cw = (bounds[2] - bounds[0]).max(f32::EPSILON) / f32::from(columns);
    let ch = (bounds[3] - bounds[1]).max(f32::EPSILON) / f32::from(rows);
    for y in 0..rows {
        for x in 0..columns {
            let cell = [
                bounds[0] + f32::from(x) * cw,
                bounds[1] + f32::from(y) * ch,
                bounds[0] + f32::from(x + 1) * cw,
                bounds[1] + f32::from(y + 1) * ch,
            ];
            let mut seen = FxHashSet::default();
            for (i, p) in pieces.iter().enumerate() {
                let b = curve_bounds(p.curve);
                if b[2] >= cell[0] - radius
                    && b[0] <= cell[2] + radius
                    && b[3] >= cell[1] - radius
                    && b[1] <= cell[3] + radius
                    && seen.insert(i)
                {
                    candidates.push(i as u32);
                }
            }
            offsets.push(candidates.len() as u32);
        }
    }
    let brute_force = offsets
        .windows(2)
        .all(|w| w[1] - w[0] == pieces.len() as u32);
    DistanceGrid {
        bounds,
        columns,
        rows,
        radius,
        offsets,
        candidates,
        brute_force,
    }
}

/// Prepare and pack the CPU border blob. `radius_px` includes the AA allowance.
/// `ppem` sizes the boundary approximation's error budget; `radius_units`
/// sizes the distance grid. They are INDEPENDENT capacities and must be
/// passed as such: deriving the unit radius from a pixel radius at one
/// ppem under-provisions every smaller ppem the same glyph is drawn at.
pub fn prepare_border(
    outline: &GlyphOutline,
    fill_offset: u32,
    units_per_em: f32,
    ppem: f32,
    radius_units: f32,
) -> Result<PreparedBorder, crate::types::PrepareError> {
    let safe_ppem = ppem.max(f32::MIN_POSITIVE);
    let ppem_ceiling = next_power_of_two_f32(safe_ppem * 2.0);
    // Floor the grid at one physical pixel's worth of font units, matching
    // the AA allowance every border query carries.
    let radius_units = radius_units.max(units_per_em / safe_ppem);
    let quantized = quantized_outline(outline);
    // Curve locations are i*2, so a large-enough curve count alone overflows
    // the u16 offset encoding; reject before the expensive boundary pass.
    if quantized.curves.len() * 2 > usize::from(u16::MAX) {
        return Err(crate::types::PrepareError::AtlasFull);
    }
    let certified_boundary = extract_boundary(&quantized, ppem_ceiling, units_per_em)?;
    let mut boundary = Vec::new();
    for piece in certified_boundary {
        refine_for_distance(piece, ppem_ceiling, units_per_em, &mut boundary)?;
    }
    let grid = build_distance_grid(&boundary, quantized.bounds, radius_units, units_per_em);
    let mut locations = Vec::with_capacity(quantized.curves.len());
    for i in 0..quantized.curves.len() {
        locations.push(CurveLocation {
            offset: (i * 2) as u32,
        });
    }
    let band_count = (quantized.curves.len() as u32).clamp(1, 16);
    if !border_band_offsets_fit(&quantized, band_count) {
        return Err(crate::types::PrepareError::AtlasFull);
    }
    let bands = build_border_bands(
        &quantized,
        &locations,
        band_count,
        band_count,
        Vec::new(),
        &mut BandScratch::default(),
    );
    let descriptor_texels = 8u32;
    let winding_offset = descriptor_texels;
    let winding_texels = (bands.entries.len() / 4) as u32;
    let winding_curve_texels = (quantized.curves.len() as u32) * 2;
    let grid_offset = winding_offset + winding_texels + winding_curve_texels;
    if grid_offset > u32::from(u16::MAX)
        || locations
            .iter()
            .any(|location| location.offset > u32::from(u16::MAX))
    {
        return Err(crate::types::PrepareError::AtlasFull);
    }
    let grid_texels = grid.offsets.len() as u32 + grid.candidates.len() as u32;
    let boundary_offset = grid_offset + grid_texels;
    // A brute-force grid loops every boundary piece, so it answers any
    // query radius; record that as unlimited capacity rather than the
    // radius that happened to trigger the fallback.
    let grid_radius_units = if grid.brute_force {
        f32::INFINITY
    } else {
        grid.radius
    };
    let descriptor = BorderDescriptor {
        fill_offset,
        winding_offset,
        grid_offset,
        boundary_offset,
        boundary_count: boundary.len() as u32,
        grid_radius_units,
        ppem_ceiling,
    };
    let mut data = vec![
        fill_offset as i32,
        winding_offset as i32,
        grid_offset as i32,
        boundary_offset as i32,
        boundary.len() as i32,
        // Slot 5 is CPU bookkeeping; the shader reads units_per_em at 6.
        grid.radius.to_bits() as i32,
        units_per_em.to_bits() as i32,
        if grid.brute_force { 1 } else { 0 },
        quantized.bounds[0].to_bits() as i32,
        quantized.bounds[1].to_bits() as i32,
        quantized.bounds[2].to_bits() as i32,
        quantized.bounds[3].to_bits() as i32,
        i32::from(grid.columns),
        i32::from(grid.rows),
        grid.radius.to_bits() as i32,
        quantized.curves.len() as i32,
    ];
    let (band_quads, _) = bands.entries.as_chunks::<4>();
    for c in band_quads {
        data.push(pack_i16_pair(c[0], c[1]));
        data.push(pack_i16_pair(c[2], c[3]));
    }
    for c in &quantized.curves {
        data.push(pack_i16_pair(
            (c.p1[0] * 4.0).round() as i16,
            (c.p1[1] * 4.0).round() as i16,
        ));
        data.push(pack_i16_pair(
            (c.p2[0] * 4.0).round() as i16,
            (c.p2[1] * 4.0).round() as i16,
        ));
        data.push(pack_i16_pair(
            (c.p3[0] * 4.0).round() as i16,
            (c.p3[1] * 4.0).round() as i16,
        ));
        data.push(0);
    }
    for &v in &grid.offsets {
        data.extend_from_slice(&[v as i32, 0]);
    }
    for &v in &grid.candidates {
        data.extend_from_slice(&[v as i32, 0]);
    }
    for p in &boundary {
        for point in [p.curve.p1, p.curve.p2, p.curve.p3] {
            data.push(point[0].to_bits() as i32);
            data.push(point[1].to_bits() as i32);
        }
    }
    let texel_len = (data.len() / 2) as u32;
    Ok(PreparedBorder {
        descriptor,
        data,
        texel_len,
        boundary,
        grid,
    })
}

fn border_band_offsets_fit(outline: &GlyphOutline, band_count: u32) -> bool {
    let [min_x, min_y, max_x, max_y] = outline.bounds;
    let width = (max_x - min_x).max(f32::MIN_POSITIVE);
    let height = (max_y - min_y).max(f32::MIN_POSITIVE);
    let mut references = 0u64;
    for curve in &outline.curves {
        let curve_min_x = curve.p1[0].min(curve.p2[0]).min(curve.p3[0]);
        let curve_max_x = curve.p1[0].max(curve.p2[0]).max(curve.p3[0]);
        let curve_min_y = curve.p1[1].min(curve.p2[1]).min(curve.p3[1]);
        let curve_max_y = curve.p1[1].max(curve.p2[1]).max(curve.p3[1]);
        if curve_min_y != curve_max_y {
            let first = ((curve_min_y - min_y) * band_count as f32 / height)
                .floor()
                .clamp(0.0, band_count as f32 - 1.0) as u64;
            let last = ((curve_max_y - min_y) * band_count as f32 / height)
                .floor()
                .clamp(0.0, band_count as f32 - 1.0) as u64;
            references += last - first + 1;
        }
        if curve_min_x != curve_max_x {
            let first = ((curve_min_x - min_x) * band_count as f32 / width)
                .floor()
                .clamp(0.0, band_count as f32 - 1.0) as u64;
            let last = ((curve_max_x - min_x) * band_count as f32 / width)
                .floor()
                .clamp(0.0, band_count as f32 - 1.0) as u64;
            references += last - first + 1;
        }
    }
    let band_texels = u64::from(band_count) * 2 + references * 2;
    let last_curve = outline.curves.len().saturating_sub(1) as u64 * 2;
    band_texels <= u64::from(u16::MAX) && band_texels + last_curve <= u64::from(u16::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn line(a: [f32; 2], b: [f32; 2]) -> QuadCurve {
        QuadCurve {
            p1: a,
            p2: a,
            p3: b,
        }
    }
    fn outline(curves: Vec<QuadCurve>) -> GlyphOutline {
        GlyphOutline {
            curves,
            bounds: [0.0, 0.0, 10.0, 10.0],
        }
    }
    fn square(clockwise: bool) -> Vec<QuadCurve> {
        let p = [[0., 0.], [10., 0.], [10., 10.], [0., 10.]];
        let mut out = Vec::new();
        for i in 0..4 {
            let (a, b) = if clockwise {
                (p[i], p[(i + 1) % 4])
            } else {
                (p[(i + 1) % 4], p[i])
            };
            out.push(line(a, b));
        }
        out
    }
    fn rectangle(min: [f32; 2], max: [f32; 2]) -> Vec<QuadCurve> {
        let p = [min, [max[0], min[1]], max, [min[0], max[1]]];
        (0..4).map(|i| line(p[i], p[(i + 1) % 4])).collect()
    }
    #[test]
    fn opposite_coincident_contours_cancel() {
        let mut c = square(true);
        c.extend(square(false));
        assert!(
            extract_boundary(&outline(c), 64., 10.)
                .expect("certification converges")
                .is_empty()
        );
    }
    #[test]
    fn equally_oriented_trace_survives() {
        let mut c = square(true);
        c.extend(square(true));
        assert!(
            !extract_boundary(&outline(c), 64., 10.)
                .expect("certification converges")
                .is_empty()
        );
    }
    #[test]
    fn oppositely_oriented_trace_with_different_segmentation_cancels() {
        let curves = vec![
            line([0., 0.], [10., 0.]),
            line([10., 0.], [5., 0.]),
            line([5., 0.], [0., 0.]),
        ];
        assert!(
            extract_boundary(&outline(curves), 64., 10.)
                .expect("certification converges")
                .is_empty()
        );
    }
    #[test]
    fn oppositely_oriented_curved_trace_with_different_segmentation_cancels() {
        let arc = QuadCurve {
            p1: [0., 0.],
            p2: [5., 8.],
            p3: [10., 0.],
        };
        let (a, b) = split(arc);
        let rev = |c: QuadCurve| QuadCurve {
            p1: c.p3,
            p2: c.p2,
            p3: c.p1,
        };
        let curves = vec![arc, rev(b), rev(a)];
        assert!(
            extract_boundary(&outline(curves), 64., 10.)
                .expect("certification converges")
                .is_empty()
        );
    }
    #[test]
    fn covered_span_keeps_only_exposed_parts() {
        let mut c = rectangle([0., 0.], [10., 10.]);
        c.extend(rectangle([8., 4.], [12., 6.]));
        let pieces = extract_boundary(&outline(c), 128., 10.).expect("certification converges");
        let right = pieces
            .iter()
            .filter(|p| p.source_curve == 1)
            .collect::<Vec<_>>();
        assert!(right.iter().any(|p| eval(p.curve, 0.5)[1] < 3.0));
        assert!(!right.iter().any(|p| {
            let y = eval(p.curve, 0.5)[1];
            (4.25..5.75).contains(&y)
        }));
    }
    #[test]
    fn asymmetric_length_coincident_pair_cancels_symmetrically() {
        // Parent arc (0,0),(8,16),(16,0); its exact [0, 0.25] subcurve is
        // (0,0),(2,4),(4,6). The reversed quarter sits at the LOWER index,
        // so cancellation must be discovered from the whole curve's side
        // and still clear the quarter's own span table.
        let parent = QuadCurve {
            p1: [0., 0.],
            p2: [8., 16.],
            p3: [16., 0.],
        };
        let quarter_rev = QuadCurve {
            p1: [4., 6.],
            p2: [2., 4.],
            p3: [0., 0.],
        };
        let pieces = extract_boundary(
            &GlyphOutline {
                curves: vec![quarter_rev, parent],
                bounds: [0., 0., 16., 16.],
            },
            64.,
            10.,
        )
        .expect("certification converges");
        assert!(!pieces.is_empty());
        assert!(!pieces.iter().any(|p| p.source_curve == 0));
        assert!(!pieces.iter().any(|p| eval(p.curve, 0.5)[0] < 3.5));
    }
    #[test]
    fn quantum_separated_opposite_segments_are_not_canceled() {
        let curves = vec![line([5., 0.], [5., 10.]), line([5.25, 10.], [5.25, 0.])];
        assert!(
            !extract_boundary(&outline(curves), 64., 10.)
                .expect("certification converges")
                .is_empty()
        );
    }
    #[test]
    fn quantum_separated_opposite_arcs_are_not_canceled() {
        let a = QuadCurve {
            p1: [0., 0.],
            p2: [5., 8.],
            p3: [10., 0.],
        };
        let b = QuadCurve {
            p1: [10., 0.25],
            p2: [5., 8.25],
            p3: [0., 0.25],
        };
        assert!(
            !extract_boundary(&outline(vec![a, b]), 64., 10.)
                .expect("certification converges")
                .is_empty()
        );
    }
    #[test]
    fn half_open_vertex_crossing_is_once() {
        let curves = [line([0., 0.], [0., 5.]), line([0., 5.], [0., 10.])];
        assert_eq!(winding_at(&curves, [-1., 5.]), 1);
    }
    #[test]
    fn local_extremum_tangency_has_zero_winding() {
        let curves = vec![
            line([0., 0.], [1., 1.]),
            line([1., 1.], [0., 2.]),
            line([0., 2.], [-1., 1.]),
            line([-1., 1.], [0., 0.]),
        ];
        assert_eq!(winding_at(&curves, [-2., 0.]), 0);
    }
    #[test]
    fn split_segment_winding_matches_unsplit_trace() {
        let unsplit = [line([4., 0.], [4., 10.])];
        let split = [line([4., 0.], [4., 5.]), line([4., 5.], [4., 10.])];
        for point in [[0., 2.], [0., 5.], [0., 8.], [8., 5.]] {
            assert_eq!(winding_at(&unsplit, point), winding_at(&split, point));
        }
    }
    #[test]
    fn split_edge_cpu_and_gpu_winding_paths_agree() {
        let curves = vec![
            line([0., 0.], [10., 0.]),
            line([10., 0.], [10., 5.]),
            line([10., 5.], [10., 10.]),
            line([10., 10.], [0., 10.]),
            line([0., 10.], [0., 0.]),
        ];
        assert_ne!(winding_at(&curves, [5., 5.]), 0);
        let shader = include_str!("border_shader.wgsl");
        assert!(shader.contains("if all(p1 == p0) || all(p1 == p2)"));
        assert!(shader.contains("if upward { winding += 1; }"));
        assert!(shader.contains("if downward { winding -= 1; }"));
    }
    #[test]
    fn oversized_winding_offsets_are_rejected() {
        let curves = (0..32_769)
            .map(|i| line([i as f32, 0.], [i as f32, 1.]))
            .collect();
        assert_eq!(
            prepare_border(&outline(curves), 0, 1000.0, 16.0, 1.5)
                .expect_err("u16 offsets must not wrap"),
            crate::types::PrepareError::AtlasFull
        );
    }
    #[test]
    fn corrected_band_includes_upper_boundary_endpoint() {
        let o = outline(vec![line([0., 0.], [1., 5.]), line([1., 5.], [2., 10.])]);
        let locations = [CurveLocation { offset: 0 }, CurveLocation { offset: 2 }];
        let bands = build_border_bands(
            &o,
            &locations,
            1,
            2,
            Vec::new(),
            &mut BandScratch::default(),
        );
        assert_eq!(bands.entries[0], 1);
        assert_eq!(bands.entries[4], 2);
    }
    #[test]
    fn grid_bbox_candidates_are_conservative() {
        let pieces = vec![BoundaryPiece {
            curve: line([2., 2.], [8., 8.]),
            source_curve: 0,
            t0: 0.,
            t1: 1.,
        }];
        let g = build_distance_grid(&pieces, [0., 0., 10., 10.], 2., 10.);
        for y in 0..20 {
            for x in 0..20 {
                let p = [x as f32 * 0.5, y as f32 * 0.5];
                let near = (0..101).any(|i| {
                    let q = eval(pieces[0].curve, i as f32 / 100.);
                    (q[0] - p[0]).hypot(q[1] - p[1]) <= g.radius
                });
                if near {
                    match g.candidates_at(p, 1) {
                        GridCandidates::Slice(s) => assert!(s.contains(&0)),
                        GridCandidates::All(_) => {}
                    }
                }
            }
        }
    }
}
