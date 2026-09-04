"""Stage 1: bracket every eigenvalue in a window, by exclusion in the wavenumber.

TEST SUPPORT ONLY -- a scored candidate. `docs/design/candidates.md` section 1 is the seam,
`CHLADNI_V1.md` section 2 is the design, and this docstring records the parts that are load
bearing so they are not optimized away by someone who has not measured them.

    seed  ->  cover  ->  gate  ->  locate  ->  confirm

# Work in `k = sqrt(lam)`

Not a numerical trick: it is the Helmholtz wavenumber, the natural variable of the MPS
formulation. The wall slope satisfies `c(lam) = C/sqrt(lam)`, so under `u = lam**p`,
`dsigma/du = (C/p)*lam**(1/2 - p)`, which is constant for `p = 1/2` and no other power.
Measured slope spread over 15 levels:

    transform   lam    lam^(3/4)   lam^(1/4)   log lam   sqrt(lam)
    spread      57%       27%         17%        32%        5%

**That was worth 1.8x of a 2.6x total improvement -- more than every algorithmic change
combined.** A single global slope bound is conservative by the ratio of the largest local slope
to the smallest, and paying that everywhere is the largest avoidable cost in the method.

Brackets are returned in `k`; the harness converts to `lam` and widens by 4 ulps, because the
round trip is not exact and a bracket correctly fencing `k*` can fail to fence `lam* = k*^2`
after squaring.

# The cost law, which retires a family of ideas

Excluding a wall pair at distances `u_a > u_b` requires `u_b/u_a >= (safety-1)/(safety+1)`,
**independent of `c`**. So a wall is covered by a geometric ladder of ratio
`(safety+1)/(safety-1)`, costing `log(range)/log(ratio)` per side. Predicted against measured:

    safety      1.2    1.5    2.0    3.0    5.0
    predicted  11.00   5.00   3.00   2.00   1.50
    measured   11.00   5.00   3.00   2.00   1.50

Two consequences:

* **A better prediction of where the eigenvalue is cannot help the cover phase.** A sample
  placed exactly ON an eigenvalue leaves the live region unchanged; so do two straddling it to
  1e-9. What is live is the WALL, and the wall yields to the ladder or to nothing. **Six
  attempts to make cover cheaper by sampling more cleverly all failed for this one reason. A
  seventh will too.**
* A larger safety factor is a finer ladder, not merely margin: it costs `1/log(ratio)` per side.

# What is not claimed

`L` is **not a certificate.** No finite sample set bounds a Lipschitz constant from above; the
largest observed secant is a LOWER bound and inflating it produces a number with no formal
status. The completeness claim is conditional and must be reported as such -- "no `x` in
`[a, b]` with `sigma_1(x) <= tau` exists outside the returned set, GIVEN `L`" -- which is why
every run reports `L`, its history, and how often it grew after something relied on it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from objective import Declined

__all__ = ["Bracket", "SearchDiagnostics", "SearchResult", "bracket_interval", "v1_brackets",
           "live_regions", "wall_ladder_ratio", "width_is_pure", "descent_certificate",
           "isolation"]

# --- the constants ---------------------------------------------------------------------------
#
# All dimensionless, per `search.md` section 2.1: a constant may be FIXED only if it is
# dimensionless and separates two populations whose separation does not shrink as `prec`
# tightens. Anything carrying units of `lam`, and anything expressing HOW ACCURATE, is computed
# at its site from `prec`.
#
# Every value here was measured on ONE family -- the (1+delta) x 1 rectangle -- and
# `CHLADNI_V1.md` section 8 says to re-measure all of them. The ladder LAW transfers, being
# derived from the envelope test alone; the values do not.

#: Inflation of the largest observed secant. RECALIBRATE on tier 3: a domain with a re-entrant
#: angle may show a larger slope spread in `k` than the 5% measured here, and if it does the
#: case for a LOCAL slope model returns. Measured: 1.2 -> 2.0 costs +40% evaluations for
#: identical coverage, so margin is cheap in the seed; and using the 99th-percentile slope
#: instead of the max wrongly excluded 13 of 600 samples, so a 1.4% underestimate loses
#: coverage. The bound is nearly tight.
SAFETY = 1.5

#: The `sigma_2` isolation gate. Certify a region as holding one eigenvalue when the estimated
#: neighbour distance exceeds `MARGIN * region_width`. RECALIBRATE: this gate has produced
#: coverage failures twice, and `CHLADNI_V1.md` section 8 calls stopping at isolation the claim
#: most in need of confirmation off the rectangle.
MARGIN = 1.5

#: Fraction of the bracket width to place a sample away from the parabola's prediction. A sharp
#: optimum: 0.01 -> +1%, 0.02 -> 0, 0.04 -> +89%, 0.08 -> +425%, 0.15 -> +823%.
#:
#: RECALIBRATE: the optimum is pinned by a prediction error dominated by the two branches having
#: slightly different slopes, about 1.7% on the rectangle. A domain whose cone is more
#: asymmetric moves it, and `CHLADNI_V1.md` section 10 item 2 proposes fitting `c_left` and
#: `c_right` separately, which would shrink it. Swept only on rectangles.
STRADDLE = 0.02

#: Starting sample density, per expected Weyl gap. The starting density barely matters -- 0.5,
#: 1.0 and 1.5 all cost the same, because the doubling adapts -- so take the larger for margin.
SEED_PER_GAP = 1.5

#: Secant growth below which the slope estimate is called settled, per doubling.
SETTLE = 0.10

#: Bracket widths of clearance to the estimated neighbour before a bracket is handed on.
ISOLATION_WIDTHS = 4.0

#: Evaluations spent looking for a descent in a region that has none before emitting it anyway.
CONFIRM_EVALS = 3

# --- three stopping rules that were tried and removed -----------------------------------------
#
# Left here as comments and not as constants, because each is a plausible thing to reach for and
# each is measured wrong. The rule that survives is `width_is_pure` alone: a region leaves the
# cover loop when it is narrow enough to be pure whatever it holds, or when the gate certifies
# it single.
#
# **A positive threshold on the envelope (`tau_scan`).** `env >= 0` looks unsound, because
# `sigma_1` bottoms at `eps > 0` rather than zero, so as the spacing `dk` falls the bound
# `(v_i + v_j)/2 - L*dk/2` rises toward a positive floor value and a finely sampled well can
# exclude itself. Replacing it with `env >= 0.5 * plateau` broke EVERY case, 0 of 66: the cover
# loop stops on purity by width long before that regime is reached, so the zero threshold never
# bites.
#
# **A flat-bottom stop from `4*sigma_min/c`.** Meant to stop narrowing at the resolution limit
# `eps/c`. But `sigma` at the current deepest sample is an UPPER bound on the floor, so the
# estimate read 0.04 while the samples were still far from the bottom and stopped narrowing at
# once -- trading the coverage failures it fixed for purity failures on pairs that were
# perfectly resolvable. The bias has to go the other way.
#
# **A prominence stop on the certificate triple.** Sound in principle -- a triple of width `w`
# about a cone's minimum has flanks at `~c*w/2` against a centre at `eps`, so prominence
# reaching 1 is exactly `w ~ 2*eps/c` -- and unsafe early, because a cone bottoming out mid-cell
# has almost equidistant flanks and scores ~1 on a coarse grid. Telling the two apart needs the
# LEVEL as well as the shape, at which point it is doing `width_is_pure`'s job less directly.
#
# The finding that made all three unnecessary is elsewhere: being asked for more precision than
# the objective can deliver is not a failure, so the loop reports `attained_prec` and stops,
# rather than needing a rule to detect the floor in advance.

#: Sample pairs closer than this fraction of the span are ignored when estimating the slope.
#: An unguarded secant across such a pair is rounding rather than slope, and one reached
#: **7.3e10** against a true 0.6 -- a slope model that never shrinks cannot recover from
#: believing that.
MIN_SECANT_FRAC = 1e-9

#: Doublings the seed may spend before the slope estimate is declared unstable.
MAX_DOUBLINGS = 6

#: Seed size when the objective offers no count estimate.
N_SEED = 64

#: Growths of `L` after exclusions have already relied on it before declining.
#:
#: **Growth is COST, not error.** Exclusions are recomputed from the current `L` on every pass,
#: so an underestimate becomes extra work rather than a silent miss -- which is the whole reason
#: for recomputing. The counter exists to REPORT that the bound was being pushed; a run with
#: many growths deserves a rerun with a larger safety factor, but it is not wrong.
#:
#: It was 4, and that threw away solvable cases. On the multiplicity-3 cluster at `50*pi^2` the
#: seed settles at a max secant of 0.31 while the true wall slope is 0.683 -- the window is
#: narrow and the cluster's dip is far finer than any seed grid resolves, so the settling test
#: passes on the PLATEAU's own variation and the cover phase then discovers the walls one
#: doubling at a time. It takes 7 growths to reach `L = 1.082 = 1.5 * 0.72`, which is the right
#: answer. At 4 the case declined having found nothing; at 8 it passes with coverage 1.00 at 80
#: evaluations. 16 leaves margin without making the decline unreachable.
MAX_GROWTHS = 16

#: Cover-loop samples that buy no reduction in the unresolved regions before declining.
STALL_LIMIT = 200

#: Locator steps per region before giving up on it.
LOCATE_CAP = 200

#: Cover/locate rounds. A re-opened region goes back through cover; five is ample.
MAX_ROUNDS = 5

#: Region-widths of clearance the re-check demands before a narrowed bracket is kept.
REOPEN_MARGIN = 1.0

#: How far below its flanks a region's samples must fall to be kept as a dip. Asked ONCE, of a
#: region, to decide whether a minimizer is there at all -- never to choose among certificates.
#: Depth necessarily tends to 1 as a bracket tightens around a cone, because the flanks approach
#: the middle, so filtering on it would reject exactly the tight brackets the search works
#: toward and a locator built that way stalls partway down.
DIP_RATIO = 0.5

#: Evaluations per expected eigenvalue before declining. Never flat: a flat cap silently
#: penalises dense windows and turned a 38-evaluation miss into the recorded claim "nobody
#: handles family E".
#:
#: Deliberately well above the scorer's own `COST_CAP_PER_EIG`, because these two numbers do
#: different jobs and confusing them corrupts both. The scorer's cap is a VERDICT -- "this was
#: too expensive" -- while this one only stops a pathological run, and if it binds first then
#: every expensive case is reported as DECLINED rather than as expensive, which hides the
#: distinction the decline mechanism exists to make. Measured: at 120 the 35-eigenvalue window
#: declined at 1320 evaluations with 30 regions still live, i.e. the budget was the failure.
BUDGET_PER_EIG = 600


@dataclass(frozen=True)
class Bracket:
    """One interval believed to hold an eigenvalue, in `k`, with why it is believed."""
    lo: float
    hi: float
    k_hint: float = float("nan")    # the deepest sample inside; free for the refiner to reuse
    sigma: float = float("nan")
    neighbour: float = float("inf")  # estimated distance to the next eigenvalue
    certified: bool = False          # a three-point descent certificate on SAMPLED points
    isolated: bool = False           # the sigma_2 gate passed, for the REGION
    floor_limited: bool = False      # stopped at `eps/c`, not at `prec`: the PENCIL is binding

    def __iter__(self):
        return iter((self.lo, self.hi))


@dataclass(frozen=True)
class SearchDiagnostics:
    """Required output, not instrumentation. The completeness claim is conditional on `L`."""
    evaluations: int = 0
    seed_evals: int = 0
    cover_evals: int = 0
    locate_evals: int = 0
    confirm_evals: int = 0
    L: float = float("nan")
    L_history: tuple = ()
    L_growth_after_use: int = 0     # how often the "bound" rose after something relied on it
    max_secant: float = float("nan")
    slope_spread: float = float("nan")   # the 5% claim of V1 section 2.1, measured per run
    expected_count: float = float("nan")
    n_brackets: int = 0
    n_unconfirmed: int = 0
    n_live_regions: int = 0
    declined: bool = False
    decline_reason: str = ""
    #: Relative width in `lam` of the WIDEST bracket returned -- the precision actually
    #: achieved. Compare against `prec_requested`: equal or smaller means the request was met,
    #: larger means the objective could not support it and this is the honest bound.
    attained_prec: float = float("nan")
    prec_requested: float = float("nan")
    #: True when some region stopped narrowing because the objective, not the budget, ran out.
    floor_limited: bool = False

    @property
    def met_request(self):
        """Did every bracket reach the requested precision?"""
        return self.attained_prec <= self.prec_requested

    @property
    def margin(self):
        """`L` over the largest secant actually seen. A small margin means the constant was
        being pushed and the run deserves a rerun with a larger safety factor."""
        return self.L / self.max_secant if self.max_secant else float("inf")


@dataclass
class SearchResult:
    brackets: tuple = ()
    diagnostics: SearchDiagnostics = field(default_factory=SearchDiagnostics)

    @property
    def declined(self):
        return self.diagnostics.declined

    def __iter__(self):
        return iter(tuple((b.lo, b.hi) for b in self.brackets))

    def __len__(self):
        return len(self.brackets)


# --- evidence ---------------------------------------------------------------------------------

def _secants(ks, vs, span):
    """Absolute secant slopes, ignoring pairs too close to carry slope information."""
    dk = np.diff(ks)
    ok = dk > MIN_SECANT_FRAC * span
    if not np.any(ok):
        return np.empty(0)
    return np.abs(np.diff(vs)[ok] / dk[ok])


def _envelope_min(ks, vs, i, L):
    """Piyavskii lower bound on `sigma_1` between samples `i` and `i+1`.

    Two cones of slope `L` dropping from the two samples cross at `mean(v) - L*gap/2`. At or
    above zero, no minimizer can lie between them -- conditional on `L` really bounding the
    slope, which is the assumption the whole method rests on and which `_slope` refuses to
    dress up as a certificate.
    """
    return 0.5 * (vs[i] + vs[i + 1]) - 0.5 * L * (ks[i + 1] - ks[i])


def _piyavskii_point(ks, vs, i, L):
    """Where that lower bound is attained -- the most informative point in the pair."""
    x = 0.5 * (ks[i] + ks[i + 1]) + (vs[i] - vs[i + 1]) / (2.0 * L)
    return min(max(x, np.nextafter(ks[i], ks[i + 1])), np.nextafter(ks[i + 1], ks[i]))


def wall_ladder_ratio(safety):
    """Coarsest geometric sample spacing that still excludes a cone's wall.

    Exclusion of a pair at distances `u_a > u_b` from the minimizer needs
    `c*(u_a + u_b)/2 - L*(u_a - u_b)/2 >= 0` with `L = safety*c`, which reduces to
    `u_b/u_a >= (safety - 1)/(safety + 1)` -- **independent of `c`**. So a wall is covered by a
    geometric ladder of ratio `(safety + 1)/(safety - 1)` and costs `log(range)/log(ratio)` per
    side. Predicted against measured, exactly:

        safety      1.2    1.5    2.0    3.0    5.0
        predicted  11.00   5.00   3.00   2.00   1.50
        measured   11.00   5.00   3.00   2.00   1.50
    """
    safety = float(safety)
    return (safety + 1.0) / (safety - 1.0) if safety > 1.0 else float("inf")


def live_regions(ks, vs, L):
    """Maximal runs of adjacent sample pairs that exclusion has not ruled out, MERGED.

    Merging matters: the pairs either side of a minimizer are both live, and reported
    separately they would look like two candidates where there is one.
    """
    out = []
    for i in range(len(ks) - 1):
        if _envelope_min(ks, vs, i, L) >= 0.0:
            continue
        lo, hi = float(ks[i]), float(ks[i + 1])
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], hi)
        else:
            out.append((lo, hi))
    return out


def _within(ks, lo, hi):
    """Indices of samples inside `[lo, hi]`, ascending."""
    return np.nonzero((ks >= lo) & (ks <= hi))[0]


# --- what licenses a bracket -------------------------------------------------------------------

def width_is_pure(lo, hi, prec_u):
    """Narrow enough that it cannot be impure, whatever it holds.

    **`prec` is a stopping rule, not a merge licence.** The search never reasons about whether
    merging is permitted: merging saves the refinement of one bracket, certifying it is legal
    means resolving both minimizers anyway, and being wrong is a hard purity failure. Splitting
    beyond the licence is explicitly not an error. The economics are entirely one-sided, so the
    merges this search performs are only the ones it could not resolve.

    Keyed to the upper end, which is the binding member -- the scorer's own rule, so the search
    stops exactly where the scorer stops caring and not one evaluation later.
    """
    return (hi - lo) <= float(prec_u) * max(abs(float(hi)), 1.0)


def descent_certificate(ks, vs, lo, hi):
    """The three-point descent centred on the LOWEST sample in `[lo, hi]`, or `None`.

    `f(p) > f(q) < f(r)` on sampled points proves a local minimum in `(p, r)` under continuity
    alone, and nothing weaker may define an endpoint. In particular an extrapolated root may
    not: two-point secant roots fitted from opposite walls of a close pair were measured to
    CROSS -- the left root landing above the right -- which yields an inverted bracket, or a
    tiny one containing neither minimizer. Extrapolation may choose where to sample and whether
    to split; it may not name an endpoint.

    **Anchoring on the region's minimum is what makes the bracket tighten monotonically**:
    every new sample nearer the bottom becomes the new centre, and its immediate neighbours are
    necessarily closer than the old ones. Selecting by DEPTH instead stalls, because depth tends
    to 1 as the flanks approach the middle.

    Immediate neighbours only: a descent read off non-adjacent samples would skip whatever lies
    between them and could straddle two cones with a barrier in between. And a minimum on the
    edge fences nothing, so it is refused.
    """
    idx = _within(ks, lo, hi)
    if idx.size < 3:
        return None
    k = int(idx[int(np.argmin(vs[idx]))])
    if k == idx[0] or k == idx[-1] or k == 0 or k + 1 >= len(ks):
        return None
    if not (vs[k - 1] > vs[k] < vs[k + 1]):
        return None
    return (int(k), float(ks[k - 1]), float(ks[k]), float(ks[k + 1]))


def _depth(vs, cert):
    """How far below its flanks the middle sits, as a ratio. SMALLER is deeper.

    Normalized by the FURTHER flank -- the larger of the two values -- not the nearer one.
    Using the nearer one systematically rejects real minimizers: when a cone bottoms out near
    the middle of a sampling cell its two flanking samples are almost equidistant, so their
    values are almost equal and the ratio approaches 1 however deep the cone actually is.
    Measured, that cost **27% of coverage -- 19 of 26 minimizers instead of 26** -- and no
    amount of extra sampling fixes it, because refining the grid just moves the problem to a
    different cell.

    Against the further flank a mid-cell cone scores ~0.33 while plateau ripple still scores
    ~0.98, which is the separation this ratio is for.
    """
    k = cert[0]
    flank = max(vs[k - 1], vs[k + 1])
    return vs[k] / flank if flank > 0 else 0.0


# --- the isolation gate ------------------------------------------------------------------------

def isolation(obj, ks, vs, lo, hi, c_bound, margin=MARGIN):
    """Does this region hold ONE minimizer? `(single, neighbour, c_local)` or `None`.

    One evaluation returns the whole ascending sigma vector, so `sigma_2` costs nothing extra.
    Near a minimizer it is the distance to the NEXT one scaled by the same wall slope. Measured
    against the true gap: a close neighbour (<= 1e-2) reads at ratio 0.98-1.08, a far one
    (>= 19) at 0.56-0.87.

    Two things matter about those numbers. It is accurate precisely where the decision is HARD
    -- when a neighbour is close enough to be a purity risk -- and where it is inaccurate it
    UNDERESTIMATES the gap. So a gate phrased as "is the neighbour further away than this
    region is wide?" errs toward refusing to certify: conservative by construction, not by
    tuning.

    `c` is estimated from the descent triple itself, cancellation-free: the flanks straddle the
    minimizer, so on a cone `v_lo = c*(lam* - lo)` and `v_hi = c*(hi - lam*)`, hence
    `c = (v_lo + v_hi)/(hi - lo)` with no subtraction of nearly-equal quantities and no
    dependence on where the minimum is. But it is then raised to `max(local, c_bound)`: where a
    region holds two minimizers its flanks span BOTH cones, the local estimate comes out too
    small, the neighbour looks further than it is, and the gate certifies exactly what it must
    not. Cost of getting that wrong: 3 of 26 eigenvalues.

    **Silence is not permission.** No certificate means no decision, and `None` keeps the
    region on the slow path.

    Not read: the level's MULTIPLICITY off the sigma vector. Counting branches at the floor is
    correct in principle and saves 40% on exactly degenerate cases, but it is ambiguous at range
    by construction -- approaching an unresolved pair from distance `u`,
    `sigma_2/sigma_1 = 1 + d/u`, which is 1.0006 at the start of a cover pass -- and swept over
    two windows the failure count is NOT MONOTONE in the threshold (0/1/0/1/2 failures at
    1.0/1.05/1.2/1.5/2.0). A parameter whose safety alternates as it is tightened has no safe
    setting, only settings that pass the cases tried.
    """
    cert = descent_certificate(ks, vs, lo, hi)
    if cert is None:
        return None
    k, a, mid, b = cert
    c_local = (vs[k - 1] + vs[k + 1]) / (b - a) if b > a else 0.0
    c = max(c_local, float(c_bound))
    s2 = obj.sigma(float(ks[k]), 1)
    neighbour = (s2 / c) if (np.isfinite(s2) and c > 0) else float("inf")
    width = hi - lo
    return dict(single=bool(neighbour > margin * width), neighbour=float(neighbour),
                c_local=float(c_local), c=float(c), width=float(width), cert=cert)


# --- the locator -------------------------------------------------------------------------------

def _locator_step(ks, vs, lo, hi, prec_u):
    """Where to sample next inside a region certified to hold one minimizer, or `None`.

    `sigma_1**2` is exactly quadratic near an eigenvalue, so the vertex through three points is
    the answer rather than an approximation. **But prediction alone does not tighten a
    bracket**: the width is set by whichever side of the prediction has no sample near it, and
    the prediction error has a SYSTEMATIC sign -- the two branches meeting at the bottom have
    slightly different slopes -- so every scheme that samples AT the prediction converges from
    one side and the far flank never moves.

    So place the sample a fixed fraction of the bracket to one side of the prediction, on
    whichever side is not already fenced. Four designs were tried before this one worked, and
    each is a plausible thing to try again: plain V-interpolation stalled with the bracket
    stuck; bisecting the lopsided side cured the stall at no better than exclusion; mirroring
    the near side across the prediction oscillated.
    """
    cert = descent_certificate(ks, vs, lo, hi)
    if cert is None:
        return None
    k, a, mid, b = cert
    if width_is_pure(a, b, prec_u):
        return None
    x = np.array([a, mid, b])
    y = np.array([vs[k - 1], vs[k], vs[k + 1]]) ** 2
    d01, d12, d20 = x[0] - x[1], x[1] - x[2], x[2] - x[0]
    denom = d01 * d12 * d20
    pred = None
    if denom != 0.0:
        c2 = -(d12 * y[0] + d20 * y[1] + d01 * y[2]) / denom
        c1 = ((x[1] ** 2 - x[2] ** 2) * y[0] + (x[2] ** 2 - x[0] ** 2) * y[1]
              + (x[0] ** 2 - x[1] ** 2) * y[2]) / denom
        if c2 > 0 and np.isfinite(c1):
            pred = float(np.clip(-0.5 * c1 / c2, x[0], x[2]))
    if pred is None:
        # The fit says this is not a single minimum. Bisect the wider side rather than nudge:
        # the straddle is an offset from a prediction, and with no prediction a 2% step shrinks
        # the bracket by 2% per evaluation. Measured, the bracket crept 6.2e-08 -> 3.4e-08 over
        # twelve evaluations and stopped.
        left, right = x[1] - x[0], x[2] - x[1]
        pred = 0.5 * (x[0] + x[1]) if left >= right else 0.5 * (x[1] + x[2])
        return float(pred), float(a), float(b)
    gap_left, gap_right = pred - x[0], x[2] - pred
    step = STRADDLE * (x[2] - x[0])
    target = pred - step if gap_left >= gap_right else pred + step
    return float(np.clip(target, np.nextafter(a, b), np.nextafter(b, a))), float(a), float(b)


# --- the loop ----------------------------------------------------------------------------------

def _seed(obj, lo, hi, prec_u):
    """Seed from Weyl, then double until the slope estimate stops growing.

    **The bootstrap hazard runs the opposite way to intuition.** An UNDERESTIMATED `L` makes
    exclusion radii too LARGE, so a search that has not yet seen the true slope excludes
    everything and terminates having found nothing. Measured, the largest observed secant
    against a true 0.0656 as a function of uniform seed density:

        n          8      16      32      64     128     256
        max     .0046   .0128   .0386   .0599   .0647   .0652
        of true    7%     20%     59%     91%     99%     99%

    At `n = 8` the radii would be 14x too large. So the seed doubles until the estimate stops
    growing -- which is both a sound bootstrap and, when it never stops, exactly the evidence
    that the Lipschitz premise does not hold here. That is the slope-instability decline, and
    **the bootstrap IS the falsification test.**

    Doubling is `n -> 2n-1`, by inserting midpoints, so each grid CONTAINS the previous one
    exactly. `n -> 2n` leaves new points at irrational offsets from the old, some an ulp away,
    and a secant across such a pair is rounding rather than slope: one reached **7.3e10**
    against a true 0.6, and a slope model that never shrinks cannot recover from believing it.
    Midpoint insertion also reuses the old points bit-identically, so the cache returns them
    and a doubling costs only the genuinely new samples.
    """
    expected = obj.expected_count()
    n = N_SEED
    if expected is not None and np.isfinite(expected) and expected > 0:
        n = max(N_SEED // 4, int(np.ceil(SEED_PER_GAP * expected)))
    ks = list(np.linspace(lo, hi, n))
    for k in ks:
        obj.tension(k)
    hist, previous = [], 0.0
    for _ in range(MAX_DOUBLINGS):
        ks_arr, vs_arr = obj.samples
        raw = float(np.max(_secants(ks_arr, vs_arr, hi - lo), initial=0.0))
        hist.append(raw)
        if previous > 0.0 and raw <= (1.0 + SETTLE) * previous:
            return ks_arr, vs_arr, tuple(hist), True
        previous = raw
        mids = [0.5 * (ks[i] + ks[i + 1]) for i in range(len(ks) - 1)]
        for x in mids:
            obj.tension(x)
        ks = sorted(set(ks) | set(mids))
    ks_arr, vs_arr = obj.samples
    return ks_arr, vs_arr, tuple(hist), False


def bracket_interval(obj, *, prec=None, budget=None):
    """Brackets for every eigenvalue in the objective's window. Returned in `k`.

    **One loop, not two.** The obvious structure -- cover to find candidates, then refine each
    candidate separately -- is wrong, and the baselines measure how wrong: `grid_then_bisect`
    starts at coverage 1.0 and ends at **0.577**, because refining a candidate that holds a
    close pair converges on one member and leaves the other outside the bracket it returns. Any
    per-candidate refiner anchored on "the lowest point I can see" does this.

    So there is one loop, and it refines REGIONS rather than triples. A region holding two
    minimizers splits into two of its own accord the moment a sample between them is excluded.

    DECLINING is a first-class third outcome, never an exception, and the brackets found so far
    come back with it.
    """
    prec = obj.prec if prec is None else float(prec)
    obj = obj if getattr(obj, "variable", None) == "k" else _KView(obj)
    lo, hi = obj.lo, obj.hi
    span = hi - lo
    # Purity is judged on `dlam` against `prec*lam`; a bracket `[u1, u2]` maps to
    # `dlam = (dlam/du)*du`, so for `u = lam**p` the relative precision in the transformed
    # variable is `p*prec` -- a HALF for the wavenumber. Conservative, being evaluated at the
    # upper end.
    prec_u = 0.5 * prec

    ks, vs, hist, settled = _seed(obj, lo, hi, prec_u)
    seed_evals = obj.n_evals
    expected = obj.expected_count()

    # The plateau, measured once from the seed. Every threshold on a TENSION is a fraction of
    # this and never an absolute value: the floor varies five orders within a single scan on a
    # corner domain, while the plateau spans only 0.12 to 0.36 across seven domains.
    plateau = float(np.median(vs))

    max_secant = float(np.max(_secants(ks, vs, span), initial=0.0))
    L = SAFETY * max_secant
    L_hist, L_growth = [L], 0

    def diag(**kw):
        sec = _secants(ks, vs, span)
        base = dict(evaluations=obj.n_evals, seed_evals=seed_evals, L=L,
                    L_history=tuple(L_hist), L_growth_after_use=L_growth,
                    max_secant=max_secant,
                    slope_spread=float(np.ptp(sec) / np.mean(sec)) if sec.size else float("nan"),
                    expected_count=float(expected) if expected is not None else float("nan"))
        base.update(kw)
        return SearchDiagnostics(**base)

    if not settled:
        return SearchResult((), diag(
            declined=True, n_live_regions=len(live_regions(ks, vs, L)),
            decline_reason=("the slope estimate never settled: "
                            f"{['%.4g' % s for s in hist]} over {MAX_DOUBLINGS} doublings, "
                            f"still rising by more than {SETTLE:.0%}. Nothing bounds the local "
                            "slope, so no exclusion this method makes would mean anything.")))

    if budget is None:
        seen = len(live_regions(ks, vs, L))
        budget = int(BUDGET_PER_EIG * max(expected or seen, 1.0))

    cover_evals = locate_evals = confirm_evals = 0
    regions, floor_limited = [], False

    for _ in range(MAX_ROUNDS):
        mark = obj.n_evals
        try:
            ks, vs, L, L_hist, L_growth, regions, stalled = _cover(
                obj, ks, vs, L, L_hist, L_growth, prec_u, span, budget)
        except _Aborted as exc:
            cover_evals += obj.n_evals - mark
            return SearchResult((), diag(declined=True, decline_reason=str(exc),
                                         cover_evals=cover_evals,
                                         n_live_regions=len(live_regions(ks, vs, L))))
        cover_evals += obj.n_evals - mark
        floor_limited |= stalled

        mark = obj.n_evals
        ks, vs, regions, reopened = _locate(obj, ks, vs, regions, prec_u, L, budget)
        locate_evals += obj.n_evals - mark
        if not reopened:
            break
        # A re-opened region held more than one minimizer after all. It goes back through cover
        # rather than being returned as it stands: the samples the locator just placed sit right
        # beside one member, so exclusion separates the pair cheaply from here and each half is
        # judged on its own. Returning it unsplit keeps coverage but fails PURITY wherever
        # `prec` forbids merging -- measured, two pairs came back merged and the case failed
        # with coverage 1.0.

    mark = obj.n_evals
    ks, vs, kept, dropped = _confirm(obj, ks, vs, regions, L, budget, plateau)
    confirm_evals += obj.n_evals - mark

    brackets = tuple(Bracket(lo=a, hi=b, k_hint=_hint(ks, vs, a, b),
                             sigma=_hint_sigma(ks, vs, a, b),
                             certified=descent_certificate(ks, vs, a, b) is not None,
                             floor_limited=floor_limited and not width_is_pure(a, b, prec_u))
                     for a, b in kept)
    # The precision ACHIEVED, over the regions that were still being NARROWED when the loop
    # stopped -- not over every bracket. A bracket holding one eigenvalue is emitted wide on
    # purpose: purity does not constrain its width, and refining past isolation buys nothing.
    # Reporting the widest bracket instead read 9e-02 on runs that met a 1e-08 request exactly,
    # which would tell a caller its answer was eight orders worse than it was.
    unresolved = [b for b in brackets if b.floor_limited]
    attained = max((2.0 * (b.hi - b.lo) / max(b.hi, np.spacing(1.0)) for b in unresolved),
                   default=float(prec))
    return SearchResult(brackets, diag(
        n_brackets=len(brackets), n_unconfirmed=dropped,
        n_live_regions=len(regions), cover_evals=cover_evals,
        locate_evals=locate_evals, confirm_evals=confirm_evals,
        attained_prec=float(attained), floor_limited=bool(floor_limited),
        prec_requested=float(prec)))


class _KView:
    """A view of an `Objective` in `k = sqrt(lam)`. The searcher owns its own abscissa.

    `docs/conventions.md` puts `lam` above the `mps` seam and the harness states every case in
    the case's own variable, so an objective arrives in `lam` on a real pencil. But the whole
    cost argument for this method is that the wall slope is constant in `k` -- 5% spread against
    57% -- so the searcher transforms internally rather than asking the caller to.

    The cache is the underlying objective's, so `n_evals` stays the harness's charged count and
    a point read through both views is paid for once.
    """

    __slots__ = ("_obj", "variable", "lo", "hi", "prec")

    def __init__(self, obj):
        self._obj = obj
        self.variable = "k"
        self.prec = obj.prec
        if obj.variable == "k":
            self.lo, self.hi = obj.lo, obj.hi
        else:
            self.lo, self.hi = float(np.sqrt(obj.lo)), float(np.sqrt(obj.hi))

    def _to_native(self, k):
        return float(k) if self._obj.variable == "k" else float(k) * float(k)

    def tension(self, k):
        return self._obj.tension(self._to_native(k))

    def sigma(self, k, j):
        return self._obj.sigma(self._to_native(k), j)

    def in_lam(self, k):
        return float(k) * float(k)

    def from_lam(self, lam):
        return float(np.sqrt(lam))

    @property
    def n_evals(self):
        return self._obj.n_evals

    @property
    def samples(self):
        xs, vs = self._obj.samples
        if self._obj.variable != "k":
            xs = np.sqrt(xs)
        keep = (xs >= self.lo) & (xs <= self.hi)
        return xs[keep], vs[keep]

    def expected_count(self, lo=None, hi=None):
        lo = self.lo if lo is None else lo
        hi = self.hi if hi is None else hi
        return self._obj.expected_count(self._to_native(lo), self._to_native(hi))


class _Aborted(Exception):
    """A decline raised from inside a phase, caught and reported at the top."""


def _cover(obj, ks, vs, L, L_hist, L_growth, prec_u, span, budget):
    """Refine by exclusion until every live region is pure by width or certified single.

    A region leaves this loop once it is narrow enough to be pure whatever it holds, OR once
    `sigma_2` certifies it holds a single minimizer -- at which point the fast locator can
    finish it. Anything else stays here, where clusters split by exclusion. A region the gate
    cannot judge counts as NOT certified: silence is not permission.
    """
    stalled, last_total = 0, np.inf
    while True:
        regions = live_regions(ks, vs, L)
        todo = [r for r in regions
                if not width_is_pure(r[0], r[1], prec_u)
                and not _certified(obj, ks, vs, r, L)]
        if not todo:
            return ks, vs, L, L_hist, L_growth, regions, False

        total = sum(b - a for a, b in todo)
        stalled = stalled + 1 if total >= last_total else 0
        if stalled > STALL_LIMIT or obj.n_evals >= budget:
            # **Being asked for more precision than the objective can deliver is not a
            # failure, and must not be reported as one.** The regions in hand still bracket
            # their eigenvalues -- they are merely wider than `prec` asked for -- so they are
            # returned, flagged, with the width actually ACHIEVED. `search.md` clause (2) says
            # exactly this: meet `prec`, OR report the attained bound and say which bound was
            # active.
            #
            # An earlier version raised here instead. That turned "I got 3e-12 where you asked
            # for 4e-13" into "DECLINED, nothing found", which is worse in both directions: it
            # loses brackets that were correct, and it hides the one number the caller needs to
            # decide whether to accept the answer or improve the basis.
            return ks, vs, L, L_hist, L_growth, regions, True
        last_total = total

        # The region with the deepest unexplained dip: the most negative lower bound is where a
        # minimizer is least ruled out.
        best = None
        for a, b in todo:
            idx = _within(ks, a, b)
            for i in idx[:-1]:
                e = _envelope_min(ks, vs, int(i), L)
                if e < 0 and (best is None or e < best[0]):
                    best = (e, int(i))
        if best is None:
            return ks, vs, L, L_hist, L_growth, regions, False

        x = _piyavskii_point(ks, vs, best[1], L)
        before = obj.n_evals
        ks, vs = _add(obj, x)
        if obj.n_evals == before:
            # Nothing new lies between them: the floating-point floor, which is the objective's
            # own limit rather than a failure of the search.
            return ks, vs, L, L_hist, L_growth, regions, True

        raw = float(np.max(_secants(ks, vs, span), initial=0.0))
        if SAFETY * raw > L:
            L = SAFETY * raw
            L_hist.append(L)
            L_growth += 1
            if L_growth > MAX_GROWTHS:
                raise _Aborted(
                    f"the slope estimate grew {L_growth} times after exclusions had already "
                    f"relied on it (now {L:.4g}). Each growth invalidates work and suggests no "
                    "bound is being approached.")


def _certified(obj, ks, vs, region, c_bound):
    iso = isolation(obj, ks, vs, region[0], region[1], c_bound)
    return iso is not None and iso["single"]


def _locate(obj, ks, vs, regions, prec_u, c_bound, budget):
    """Drive every certified-single region down with the fast locator.

    **The gate is checked AGAIN on the result.** A locator converges to one minimizer, so
    pointing it at a pair loses one, and the cheapest insurance is to re-read `sigma_2` at the
    located point -- which costs nothing, the evaluation is already paid for. If it now says a
    neighbour lies inside the region that was just collapsed, the ORIGINAL region is kept
    instead of the narrowed one, and the extra brackets are the price of not being wrong.

    Compared against the region, never against the narrow bracket that came out of it: any
    neighbour is enormous next to a window of `prec*lam`, so that check would always answer
    "single" and be decoration. Measured with that mistake in place, the searcher lost the
    second member of two pairs, collapsed onto their partners.
    """
    out, reopened = [], 0
    for a0, b0 in regions:
        if width_is_pure(a0, b0, prec_u) or not _certified(obj, ks, vs, (a0, b0), c_bound):
            out.append((a0, b0))
            continue
        a, b, n = a0, b0, 0
        while n < LOCATE_CAP and obj.n_evals < budget:
            step = _locator_step(ks, vs, a, b, prec_u)
            if step is None:
                break
            x, a, b = step
            before = obj.n_evals
            ks, vs = _add(obj, x)
            if obj.n_evals == before:
                break
            n += 1
            # Re-ask the gate every step. It is a pure function of the evidence, so it costs
            # nothing, and the answer CHANGES: a region certified from far away -- where an
            # unresolved pair reads as one level -- shows its second level as soon as the
            # locator drives in far enough to separate the branches. Without this the whole
            # descent runs to `prec` before the check below notices, and the wasted work is
            # ~19 evaluations per pair rather than ~2.
            here = isolation(obj, ks, vs, a, b, c_bound)
            if here is not None and here["neighbour"] <= REOPEN_MARGIN * (b0 - a0):
                break
            if here is not None and here["single"] and _isolated_enough(ks, vs, a, b, here):
                break
        after = isolation(obj, ks, vs, a, b, c_bound)
        # **Silence is not permission here either.** `after is None` means the narrowed bracket
        # carries no descent certificate, so there is no evidence that what it fences is a
        # single minimizer -- and keeping it then throws away whatever else the original region
        # held. Measured at `prec = 1e-14` on the (2,1)/(1,2) pair 1.3e-14 apart: the locator
        # collapsed onto one member, the post-check could not certify, the narrowed bracket was
        # kept anyway and the second member was left outside every bracket. Coverage 0.75 with
        # purity intact, which is the failure mode the gate exists to catch.
        #
        # So a bracket is kept only when the gate positively says it is isolated; otherwise the
        # original region goes back, and being too wide is the price of not being wrong.
        if after is None or after["neighbour"] <= REOPEN_MARGIN * (b0 - a0):
            out.append((a0, b0))
            reopened += 1
        else:
            out.append((a, b))
    return ks, vs, out, reopened


def _isolated_enough(ks, vs, lo, hi, iso):
    """May this bracket be handed on now, instead of driven down to `prec*lam`?

    Purity does not constrain the width of a bracket holding ONE minimizer -- its span of
    contained minimizers is zero whatever the interval -- so refining past isolation buys
    nothing the scorer asks for, and the refiner reaches machine precision from a wide bracket
    in about seven evaluations anyway.

    What refining to `prec*lam` DOES buy is a guarantee independent of the gate: a bracket that
    narrow cannot be impure however wrong `sigma_2` was. Stopping early gives that up, so the
    rule demands its own margin -- the neighbour must be `ISOLATION_WIDTHS` bracket-widths
    away, not merely outside -- and it requires a descent certificate, so the bracket is still
    fenced by sampled points. This is the claim most in need of confirmation off the rectangle.
    """
    if descent_certificate(ks, vs, lo, hi) is None:
        return False
    return iso["neighbour"] >= ISOLATION_WIDTHS * (hi - lo)


def _confirm(obj, ks, vs, regions, L, budget, plateau):
    """Emit only regions with a demonstrated dip, buying a few samples to find out.

    An exclusion-derived region is one not yet ruled OUT, which is not the same as one shown to
    hold something -- and a global slope model, conservative everywhere by the margin it needs
    at its steepest point, leaves live slivers on the walls beside real dips. Measured at
    `prec = 1e-3`: **18 of 33 regions held no minimizer.**

    It probes the PIYAVSKII POINT -- the best available estimate of where the minimizer is --
    and not the widest unsampled gap. Probing the widest gap needs many more samples to land
    near a bottom, and at `prec = 1e-8` it exhausted a three-sample budget on a real region and
    dropped it, costing a minimizer. **Coverage failure by a confirmation step is the worst
    possible trade**: the region was correctly found and then discarded.
    """
    kept, dropped = [], 0
    for a, b in regions:
        for _ in range(CONFIRM_EVALS):
            if descent_certificate(ks, vs, a, b) is not None or obj.n_evals >= budget:
                break
            idx = _within(ks, a, b)
            if idx.size < 2:
                break
            i = min(idx[:-1], key=lambda j: _envelope_min(ks, vs, int(j), L))
            x = _piyavskii_point(ks, vs, int(i), L)
            before = obj.n_evals
            ks, vs = _add(obj, x)
            if obj.n_evals == before:
                break
        if (descent_certificate(ks, vs, a, b) is not None
                or _looks_like_a_dip(ks, vs, a, b, plateau)):
            kept.append((a, b))
        else:
            dropped += 1
    return ks, vs, kept, dropped


def _looks_like_a_dip(ks, vs, lo, hi, plateau):
    """Do a region's samples fall far enough to be a cone rather than a wall segment?

    The tie-breaker for a region that could not be FENCED. Dropping it costs a minimizer;
    keeping it costs the refinement layer a handful of evaluations, which it spends discarding
    an empty bracket without being misled. Against that asymmetry the answer is not close, so
    anything that looks like a dip is kept and only demonstrably flat regions are dropped.

    **Two ways to be a dip, and the second one is not optional.** The within-region ratio
    catches a region straddling a wall. But a region lying entirely INSIDE a well has every
    sample on the floor, so its internal ratio is ~1 and it reads as flat -- and flatness at
    the FLOOR is the opposite of flatness on the plateau.

    Measured, at `prec = 1e-14` on the (2,1)/(1,2) pair 6.5e-13 apart: the locator fenced the
    upper member in a bracket 2.6e-13 wide, and the lower member's region -- every sample at
    3e-15 against a plateau of 0.35 -- was dropped here as flat. Coverage 0.75 with purity
    intact, the region correctly found and then discarded, which is the worst trade available.

    So the level test is a FRACTION OF THE MEASURED PLATEAU, never an absolute tension: the
    floor varies five orders within a single scan on a corner domain, while the plateau spans
    only 0.12 to 0.36 across seven domains.
    """
    idx = _within(ks, lo, hi)
    if idx.size < 2:
        return True                      # too little seen to call it flat
    top = float(np.max(vs[idx]))
    bottom = float(np.min(vs[idx]))
    if top <= 0.0 or bottom / top <= DIP_RATIO:
        return True
    return bottom <= DIP_RATIO * float(plateau)


def _add(obj, k):
    """Evaluate at `k` and return the sample arrays, ascending."""
    obj.tension(float(k))
    return obj.samples


def _hint(ks, vs, lo, hi):
    idx = _within(ks, lo, hi)
    return float(ks[idx[int(np.argmin(vs[idx]))]]) if idx.size else float("nan")


def _hint_sigma(ks, vs, lo, hi):
    idx = _within(ks, lo, hi)
    return float(np.min(vs[idx])) if idx.size else float("nan")


def v1_brackets(**kw):
    """The candidate, on the stage-1 seam: `brackets(tensions, a, b, *, prec, modes=None)`.

    Wraps the callable in an `Objective` so V1 is scored on the identical instrument as the
    baselines rather than through one built for it.
    """
    def brackets(tensions, a, b, *, prec, modes=None, objective=None, expected_count=None):
        from objective import Objective
        obj = objective if objective is not None else Objective(
            tensions, a, b, prec=prec, variable="k", expected_count=expected_count)
        res = bracket_interval(obj, prec=prec, **kw)
        brackets.last = res
        # Brackets come back in the objective's OWN variable, which is what the harness
        # converts from; the `k` transform is internal to the search.
        out = [(x.lo, x.hi) if obj.variable == "k" else (x.lo ** 2, x.hi ** 2)
               for x in res.brackets]
        if res.declined:
            return Declined(res.diagnostics.decline_reason, partial=out,
                            diagnostics=res.diagnostics)
        return out

    brackets.last = None
    return brackets
