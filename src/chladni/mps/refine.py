"""Stage 2: refine a bracket to a located eigenvalue, by iterated parabolic interpolation.

Promoted from `tests/v1_refine.py`, which is now a re-export shim and stays one only until the
scored candidates are repointed at this module. `docs/design/candidates.md` section 1 is the
seam it was scored on; `CHLADNI_V1.md` section 3 is the design.

# Why this algorithm, and why not Brent

`sigma_1**2` is exactly quadratic near an eigenvalue -- for a simple smallest generalized
singular value it is an eigenvalue of an analytic Hermitian family, and the cone model says it
equals `eps**2 + c**2*delta**2`. So the vertex through three points is THE ANSWER rather than an
approximation, and iterating it uses information that a bracketing method throws away. Measured
over 26 eigenvalues at `prec = 1e-14`, on brackets from 1e-6 to 1e-1 wide:

    refiner                                  median rel error   evals/bracket
    iterated parabolic on sigma_1**2              1.6e-16          5.5 - 7.9
    Brent on sigma_1**2                    5.4e-10 to 5.6e-5       5.0 - 11.7
    straddled interpolation on sigma_1          0 (exact)          48 - 63

**Do not reach for `scipy.optimize.minimize_scalar`.** Brent's parabolic step is guarded by a
golden-section fallback and it terminates long before the curvature information is used up.

And note what does NOT go wrong: **squaring the objective does not cost half the significant
digits.** The `sqrt(eps_machine)` limit belongs to methods that locate a minimum by comparing
function VALUES; solving for the vertex from the curvature does not suffer it.

# The variable is `lam`

Even though the search covers in `k`. Fitting a quadratic to `sigma_1**2` over identical
physical sample points, the residual relative to the data range is smaller in `lam` in **17 of
17 cases** across rect 1x1, 2x1 and 5x1, the disk, eq_tri and the L-shape -- median 5.1x, range
2.5x to 15.4x -- and the located vertex is correspondingly more accurate. `docs/conventions.md`
wants `lam` above the `mps` seam anyway, and here the measurement agrees with the convention.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["RefineResult", "refine", "DELTA_SAFETY"]

#: Evaluations before giving up. Generous: a bracket costs 3 to 12 in practice, and the point
#: of a cap is to stop a pathological case rather than to bound a healthy one.
MAX_EVALS = 40

#: Steps after which a minimum still sitting on a handed-in endpoint is given up on. The
#: bracket does not hold an interior minimum and no further stepping will produce one.
_ENDPOINT_STEPS = 4

#: Relative closeness at which two sample points are treated as the same point. Below this the
#: vertex formula's `dx` is rounding rather than geometry.
_SAME_POINT = 4.0

#: Consecutive vertices landing on already-sampled points, away from the best one, before the
#: refinement is declared `stalled`. Confirming the best point is handled by fencing and never
#: counts here, so a repeat is noise in the vertex, and three in a row is not going to clear.
_REPEATS = 3

#: Fraction of `_xatol` within which the vertex must agree with the best point before the best
#: point is fenced. A fraction rather than the whole tolerance, so the point handed on has been
#: LOCATED by a vertex and not merely found inside the width rule; the residual disagreement is
#: carried into the bar, so the choice changes cost and never honesty.
_SETTLE = 0.05

#: A best point at a bracket END is rejected as a root when a probe past the end, aimed at the
#: cone's predicted minimum `sigma/c` out, reads below this fraction of its `sigma` and lies past
#: the end. A phantom sits on the wall of a neighbour's well, so the curve keeps falling past it --
#: 7.0e-10 at the end, about 2.7e-10 one width out, measured at 130*pi^2 -- while a genuine
#: eigenvalue AT the end sits at the floor and the probe climbs the cone.
#: The two differ by orders, so the factor only has to sit between them; it is `bracket.DIP_RATIO`'s
#: value for the same question asked of a region.
_DESCENT_RATIO = 0.5
#: Probe steps past a bracket end, as fractions of the best point's reach `sigma/c`. Each
#: probe falls only within 1.5x of the minimum's distance; halving twice covers a slope
#: estimate up to 6x too shallow. One evaluation each, stopping at the first that falls.
_PROBE_STEPS = (1.0, 0.5, 0.25)

#: Margin on the reported `delta_res`. The located error is `eps/c` to a measured ratio of 0.99
#: to 1.27 over 12 levels, three domains and two basis families -- so `eps/c` is the right
#: SCALE but not an upper bound, and a quantity that clause (2) compares against `prec*lam` has
#: to err upward. 2.0 leaves about 1.6x over the worst observed ratio. Dimensionless, and the
#: margin does not shrink as `prec` tightens, which is `search.md` section 2.1's rule for a
#: constant that may be fixed at all.
DELTA_SAFETY = 2.0


@dataclass(frozen=True)
class RefineResult:
    """What one bracket refined to, and what it cost."""
    lam: float                  # the located minimum
    sigma: float                # sigma_1 there
    accepted: bool
    reason: str                 # converged | stalled | flat | no_parabola | endpoint | budget
    evaluations: int            # what the harness charges for this bracket
    bracket: tuple = ()         # the interval it was handed
    delta_res: float = float("nan")   # attained bound, from the parabola fit
    steps: int = 0
    #: Width of the final straddling triple. `converged` is a claim that this is at most
    #: `_xatol`; `stalled` is the admission that it is not and could not be narrowed.
    width: float = float("nan")

    @property
    def rejected(self):
        return not self.accepted


def _deepest_inside(obj, lo, hi):
    """The lowest ALREADY-EVALUATED point strictly inside `(lo, hi)`, in `lam`, or `None`.

    A cache read, so it costs nothing. The objective may be parameterized by `k`, in which case
    its samples are too, and they are mapped back here.
    """
    xs, vs = obj.samples
    if xs.size == 0:
        return None
    lams = np.array([obj.in_lam(x) for x in xs])
    inside = (lams > lo) & (lams < hi)
    if not np.any(inside):
        return None
    return float(lams[inside][int(np.argmin(vs[inside]))])


def _xatol(lam, prec):
    """Stopping width, from the target precision and the arithmetic.

    `max(prec*|lam|, 4*ulp(lam))`, computed from the CURRENT iterate rather than from the
    bracket. A tolerance derived from `lam` alone can EXCEED the bracket it is refining -- a
    2.7e-13 bracket handed a tolerance of 7.9e-13 -- so the refiner terminates before doing
    anything and returns a point that is not the minimum.
    """
    return max(prec * abs(lam), 4.0 * np.spacing(abs(lam)))


def _vertex(x, y):
    """Vertex of the parabola through three points, or `None` if there is not one.

    Returns `(x_star, curvature)`. `curvature <= 0` means the three points do not describe a
    single minimum -- guard 1 -- and `dx == 0` is guarded here rather than at the call site
    because that is where the division is.
    """
    x0, x1, x2 = (float(v) for v in x)
    y0, y1, y2 = (float(v) for v in y)
    d01, d12, d20 = x0 - x1, x1 - x2, x2 - x0
    denom = d01 * d12 * d20
    if denom == 0.0:
        return None
    # `c2` is the curvature, the parabola's leading coefficient, from differences only.
    c2 = -(d12 * y0 + d20 * y1 + d01 * y2) / denom
    if c2 <= 0.0 or not np.isfinite(c2):
        return None
    # The vertex CENTRED on the middle point, from offsets and never from `x**2`. The monomial
    # form `-c1/(2*c2)` squares each `x` about the origin while the samples sit near `lam`, so
    # every square carries its own `ulp(lam**2)` and the vertex is off by about
    # `ulp(lam**2)/h` for triple spacing `h` -- 2e-7 relative at `h = 1e-10*lam`. It stalls at
    # `h ~ lam*sqrt(eps)`, which put the refiner's "converged" at 1e-11 to 1e-9 on pencils
    # whose floor is 1e-15. Centred, the error is a few ulps of `lam` at every spacing.
    a, b = x1 - x0, x1 - x2
    den = a * (y1 - y2) - b * (y1 - y0)
    if den == 0.0:
        return None
    step = 0.5 * (a * a * (y1 - y2) - b * b * (y1 - y0)) / den
    if not np.isfinite(step):
        return None
    return x1 - step, c2


def refine(obj, lo, hi, *, prec=None, max_evals=MAX_EVALS, lam_hint=None, interior=None):
    """Locate the minimum of `sigma_1**2` in `[lo, hi]`. Both endpoints in `lam`.

    `obj` is an `Objective`; it may be parameterized by `k`, in which case the conversion
    happens here and the objective's cache is shared, so re-reading a point costs nothing.

    Returns a `RefineResult` whose `evaluations` is what the harness charges for this bracket.
    That number is the whole reason this stage is built before the searcher: a flat per-bracket
    charge is wrong in both directions -- 3 evaluations for a bracket of width 1e-12, 12 for
    width 20, and 2 to 7 to REJECT an empty one -- and using one reversed the answer to
    "should the search refine or hand off?" twice.
    """
    prec = obj.prec if prec is None else float(prec)
    lo, hi = float(min(lo, hi)), float(max(lo, hi))
    start = obj.n_evals

    def f(lam):
        """`sigma_1**2` at `lam`, clamped into the bracket."""
        lam = float(np.clip(lam, lo, hi))
        s = obj.tension(obj.from_lam(lam))
        return lam, s * s, s

    # Three points to start: the ends, and the DEEPEST SAMPLE ALREADY INSIDE the bracket.
    #
    # The arithmetic midpoint is the wrong third point, and badly so on a bracket that stage 1
    # stopped refining on width purity: **a bracket is guaranteed to CONTAIN the eigenvalue, not
    # to have a sampled descent inside it.** On the multiplicity-3 degeneracy the well is about
    # 1e-07 wide inside an 8.6e-07 bracket, so the midpoint misses it entirely and the starting
    # triple reads 1.5e-09 -> 5.6e-09 -> 1.3e-08, monotone. The parabola cannot close on that
    # and the refiner walks to an endpoint.
    #
    # The search already paid for a sample in the well -- that is what earned the bracket -- so
    # the fix is to read it back rather than to sample again. Measured, on the degeneracies:
    #
    #     level          midpoint start   deepest-sample start
    #     D:mult=3          1.8e-10             7.4e-12
    #     D:mult=4          9.1e-10             3.3e-11
    #
    # An explicit `lam_hint` still wins, for a caller that knows better.
    mid = None
    if lam_hint is not None:
        mid = float(np.clip(lam_hint, lo, hi))
    if mid is None or not lo < mid < hi:
        mid = _deepest_inside(obj, lo, hi)
    if mid is None or not lo < mid < hi:
        mid = 0.5 * (lo + hi)

    # Every point this call has seen, so the working triple can be RE-SELECTED rather than
    # patched. The first version kept `[i-1, i, i+1]` of the sorted list, which deadlocks: with
    # the minimum at position 1 the left endpoint is pinned forever, every new point lands
    # outside the retained triple and is discarded, and the same vertex is recomputed until the
    # budget runs out. Machine-exact answer, 40 wasted steps, and `reason` stuck at "budget".
    hist = [f(lo), f(mid), f(hi)]

    # Guard 3: track the lowest point seen INDEPENDENTLY of the triple. The triple is re-formed
    # by position rather than by value, so a bad vertex can discard the sub-interval holding
    # the minimum -- and then the answer is gone even though it was once in hand.
    best = min(hist, key=lambda p: p[1])

    steps = 0
    progressed = False
    reason = "budget"
    repeats = 0
    settle = 0.0      # the vertex's last disagreement with the fenced best point
    curvature = float("nan")

    def triple():
        """Best point, plus its nearest neighbour on each side.

        Straddling is what makes the interval shrink: new points accumulate near the minimum,
        so the two neighbours close in on it monotonically. Where the best point is at an end
        -- the minimum lies outside the bracket -- there is no straddle and the three nearest
        points are used instead, which is the configuration `_accept` then rejects.
        """
        b = min(hist, key=lambda p: p[1])
        left = [p for p in hist if p[0] < b[0]]
        right = [p for p in hist if p[0] > b[0]]
        if left and right:
            return [max(left, key=lambda p: p[0]), b, min(right, key=lambda p: p[0])]
        return sorted(sorted(hist, key=lambda p: abs(p[0] - b[0]))[:3], key=lambda p: p[0])

    # The loop is bounded by STEPS, not by `obj.n_evals`. `n_evals` counts cache MISSES, and a
    # refiner that re-asks for a point it has already seen -- which happens as soon as the
    # triple stops moving -- adds none. Budgeting on it spins forever at zero cost, which is
    # exactly what the first version of this function did.
    while steps < max_evals:
        pts = triple()
        v = _vertex([p[0] for p in pts], [p[1] for p in pts])
        if v is None:
            # Guards 1 and 2, and the difference between them is the whole point. `C <= 0`
            # fires BOTH on a bracket that holds no single minimum -- empty, or spanning two
            # well-separated eigenvalues where the fit is across a W -- and at the END of a
            # good refinement, once the three points have collapsed onto the floor and the
            # curvature is gone. Treating the second as failure discards correct answers.
            reason = "flat" if progressed else "no_parabola"
            break

        x_star, curvature = v
        # Guard 4: keep the step inside the interval, and nudge it off a duplicate point --
        # clamped, so the nudge cannot leave the bracket it is supposed to be searching.
        x_star = float(np.clip(x_star, lo, hi))
        # The vertex BEFORE guard 4's nudge. A vertex landing on the best point is the answer,
        # and the nudge would move it to the midpoint toward a neighbour -- which, evaluated,
        # halves the distance per step and never settles: 18 evaluations from an exact start.
        raw = x_star
        for p in pts:
            if abs(x_star - p[0]) <= _SAME_POINT * np.spacing(abs(p[0]) + (hi - lo)):
                near = min((q for q in pts if q[0] != p[0]),
                           key=lambda q: abs(q[0] - p[0]), default=None)
                target = 0.5 * (p[0] + near[0]) if near is not None else 0.5 * (lo + hi)
                x_star = float(np.clip(target, lo, hi))
                break

        # The vertex is within tolerance of the best point: the LOCATION is settled, and what
        # remains is to certify the WIDTH. Iterated vertices approach from one side -- the far
        # flank of the triple never moves, which is `bracket._locator_step`'s observation -- so
        # waiting for the triple to narrow on its own costs a halving per step and can exhaust
        # the budget. Fence the best point at the tolerance on each side not already fenced:
        # two evaluations at most, and a unimodal minimum is then bounded by the triple.
        tol = _xatol(best[0], prec)
        # Fence only a best point that is INTERIOR and that the vertex CONFIRMS. Triggering on
        # `|raw - best| <= tol` alone fired before any vertex was evaluated whenever the handed-in
        # bracket was already inside the tolerance -- the routine case, since stage 1 stops on
        # purity at about `prec*lam` -- and the fence pass then settled on whichever of the three
        # starting points was lowest. With the root nearer an end that was an ENDPOINT, `_accept`
        # refused it, and the eigenvalue was dropped. Short of both conditions, take the step.
        if lo < best[0] < hi and abs(raw - best[0]) <= _SETTLE * tol:
            settle = max(settle, abs(raw - best[0]))
            # A side is already fenced when it holds a sample within HALF the tolerance of the
            # best point; then a pass that evaluates nothing leaves a triple at most `0.9*tol`
            # wide, which converges below. Using the full tolerance there left triples up to
            # `1.8*tol` wide with nothing to evaluate, and the loop recomputed the same vertex
            # without spending a step -- forever.
            fenced = 0
            for side in (-1.0, 1.0):
                fence = float(np.clip(best[0] + side * 0.45 * tol, lo, hi))
                if any(0.0 <= side * (q[0] - best[0]) <= 0.45 * tol and q[0] != best[0]
                       for q in hist) or any(fence == q[0] for q in hist):
                    continue
                new = f(fence)
                hist.append(new)
                steps += 1
                fenced += 1
                if new[1] < best[1]:
                    best = new
                    progressed = True
            pts = triple()
            if pts[-1][0] - pts[0][0] <= _xatol(best[0], prec):
                reason = "converged"
                progressed = True
                break
            if not fenced:
                # Termination must not rest on the argument above: nothing new was learned, so
                # stepping again would compute the same vertex.
                reason = "stalled" if progressed else "flat"
                break
            continue

        if any(x_star == p[0] for p in hist):
            # Landing on a sampled point that is NOT the best one is noise in the vertex, and it
            # says nothing about the width of the triple. With the old monomial vertex this was
            # reported `converged` at 1e-11 to 1e-9 on pencils whose floor is 1e-15. Split the
            # wider gap instead, and a vertex that keeps repeating is `stalled`, never converged.
            pts = triple()
            if pts[-1][0] - pts[0][0] <= tol:
                reason = "converged" if progressed else "flat"
                break
            repeats += 1
            gap_left, gap_right = pts[1][0] - pts[0][0], pts[2][0] - pts[1][0]
            if gap_left >= gap_right:
                x_star = 0.5 * (pts[0][0] + pts[1][0])
            else:
                x_star = 0.5 * (pts[1][0] + pts[2][0])
            x_star = float(np.clip(x_star, lo, hi))
            if repeats > _REPEATS or any(x_star == p[0] for p in hist):
                reason = "stalled" if progressed else "flat"
                break
        else:
            repeats = 0

        new = f(x_star)
        hist.append(new)
        steps += 1
        if new[1] < best[1]:
            best = new
            progressed = True

        pts = triple()
        if pts[-1][0] - pts[0][0] <= _xatol(best[0], prec):
            reason = "converged"
            progressed = True
            break

        # Rejecting must be CHEAP. On a bracket whose minimum sits at an end -- empty, or one
        # wall of a root just outside -- the vertex keeps stepping toward that end and the
        # parabola keeps closing, so nothing above stops it: measured 28 evaluations to reject
        # what `_accept` was always going to refuse. The cost model prices a spurious bracket
        # at 2 to 7 evaluations, and a search is entitled to keep a doubtful region at that
        # price; 28 would change the architectural answer about where to draw the stage
        # boundary, which is exactly the mistake a flat charge made twice.
        if steps >= _ENDPOINT_STEPS and (best[0] <= lo or best[0] >= hi):
            reason = "endpoint"
            break

    # `delta_res = DELTA_SAFETY * eps / c`: the RESOLUTION LIMIT, from the floor the refiner
    # reached and the curvature the parabola actually showed. `search.md` section 1 names
    # `eps/c` as the scale on which the cone flattens into its quadratic bottom, and a minimum
    # cannot be located finer than the region over which the curve is flat.
    #
    # **Why this is a bound and not a fit.** It is SELF-CALIBRATING through the cone. `eps`
    # here is `sigma_1` where the refiner actually stopped, not the floor of the well, so on
    # the wall -- where `sigma ~ c*delta` -- `eps/c` IS the remaining distance to the minimum,
    # recovered from the one value already in hand. Inside the flat bottom `sigma` saturates
    # at the true floor and the estimate becomes conservative, which is the direction a bound
    # is allowed to be wrong in. That is why it holds across five orders of `eps` rather than
    # over a fitted range.
    #
    # This replaced `sqrt(eps * eps_mach / curvature)`, which assumed the tension carries
    # ABSOLUTE noise of order machine epsilon. It does not, and `search.md` section 3 had
    # already recorded that the assumption was wrong without the consequence being applied
    # here. The two differ by `sqrt(eps_mach / eps)`, so the shortfall GROWS as the floor
    # rises: on the disk through the real pipeline the old form contained the true error at
    # ZERO of six levels, under-reporting by 280x to 2000x, where this one contains all six.
    # `curvature` is `_vertex`'s `c2`, the parabola's LEADING COEFFICIENT, so `c = sqrt(c2)`
    # directly -- `f = eps**2 + c**2*delta**2` means `c2 == c**2`, not `2*c**2`. Getting that
    # factor wrong is a silent sqrt(2) on the reported bound.
    eps_floor = max(best[2], np.finfo(float).eps)
    delta_res = float(DELTA_SAFETY * eps_floor / np.sqrt(curvature)) if (
        np.isfinite(curvature) and curvature > 0) else float("nan")
    # No location is better than the arithmetic. On a floor of 1e-15 at `lam = 1e5` the model
    # reads 4.7e-12 while the located point is 2 ulps -- 2.9e-11 -- from the minimum, so a bar
    # below a few ulps under-reports. The same floor `_xatol` stops at.
    if np.isfinite(delta_res):
        delta_res = max(delta_res, 4.0 * np.spacing(abs(best[0])), 2.0 * settle)

    # The triple the minimum sits in bounds the error whatever the fit said. A `stalled`
    # refinement could not narrow it, so the bar must cover it; one that ended with no closing
    # parabola -- collapsed onto the floor -- has no other bound in hand. Either way a NaN bar
    # would now read as unmet (`search.Level.met`), which is right only if it is really unknown.
    pts = triple()
    width = float(pts[-1][0] - pts[0][0])
    if reason == "stalled":
        delta_res = max(delta_res, width) if np.isfinite(delta_res) else width
    elif not np.isfinite(delta_res):
        delta_res = width

    ends = (hist[0][2], hist[2][2])      # sigma_1 at the handed-in endpoints; already paid for
    if reason == "no_parabola":
        accepted = False
    elif interior is None:
        accepted = _accept(best, lo, hi, prec, ends)
    else:
        # `[lo, hi]` was WIDENED from the search's bracket `interior` so that a minimum whose exact
        # square rounds just outside it is not lost. That makes the ends artificial, so "interior"
        # has to be judged by where the best point came from:
        #
        #   well inside   the search's bracket: strictly inside the widened one is enough. The
        #                 4-ulp margin against widened ends is what defeated the widening -- a
        #                 minimum one ulp outside the search's end sits three inside the widened one.
        #   at an end     within a few ulps of the search's end, or beyond it: judged by the curve
        #                 past the end. The search's deepest sample can BE its bracket end, and 4
        #                 ulps of widening against a 4-ulp margin let exactly that sample -- on the
        #                 wall of a NEIGHBOUR's well, 0.48 licences from the eigenvalue -- through as
        #                 a root, below the widened end by rounding alone.
        ilo, ihi = (float(v) for v in interior)
        x_best, _, s_best = best
        tie = _SAME_POINT * np.spacing(abs(x_best))
        side = 1.0 if abs(x_best - ihi) <= abs(x_best - ilo) else -1.0
        end = ihi if side > 0 else ilo
        # `c` is the STEEPEST secant from the best point to a sample at least a quarter of the
        # search bracket away. Not the fitted curvature: the last triple sits in the ulp-scale
        # noise pressed against the end, and its curvature read a slope 25x too large -- the
        # probe fell back onto the phantom's own wall and passed it. The quarter-width floor
        # keeps that noise out of every secant.
        #
        # Not the far end alone either, which assumed a phantom's bracket lies on ONE monotone
        # wall (1.14e-2 against a true 1.17e-2 at 130*pi^2). A bracket over a HUMP breaks it:
        # 125*pi^2, prec 1e-8, band 1.6, from member 3's wall over the hump to member 4's, the far
        # end read 7.09e-8 against the phantom's 6.60e-8, a slope 18x too shallow; the probe
        # overshot member 4 onto its far wall (1.15e-6) and the phantom passed as a fifth level.
        # Samples past a hump lie BELOW the wall, so they only make a secant shallower -- the
        # steepest is the wall's; the mid sample there, on the hump, gave 1.15e-2.
        span = max(ihi - ilo, np.spacing(abs(x_best)))
        secants = [abs(p[2] - s_best) / abs(p[0] - x_best) for p in hist
                   if abs(p[0] - x_best) >= 0.25 * span]
        if not secants:
            far_x, _, far_s = hist[0] if side > 0 else hist[2]
            secants = [abs(far_s - s_best) / max(abs(far_x - x_best), np.spacing(abs(x_best)))]
        c_wall = max(secants)
        # The best point's own resolution radius: the cone puts the minimum `sigma/c` away.
        reach = s_best / c_wall if c_wall > 0 else (ihi - ilo)
        # "At an end" is a question about that RADIUS, not about ulps. A 4-ulp tie let a phantom
        # 4 ulps inside the upper end through -- 125*pi^2, prec 1e-8, band 1.6, a bracket
        # narrower than `_xatol`, sigma 6.6e-8 against 1e-10 at the real roots -- as a fifth
        # level of a four-member cluster. Its radius reached 0.7 bracket widths past the end.
        at_end = side * (end - x_best) <= max(tie, reach)
        if not at_end:
            inside = lo < x_best < hi
        else:
            # The best point is ON an end of the search's bracket, in the sliver the widening
            # added, or close enough that its radius crosses the end. Neither position nor
            # provenance can say which case this is: the phantom and a genuine eigenvalue exactly
            # at the search's end (the evenly spaced cluster at 1e-14, band 2.0) both arrive
            # there, and the first version of this test -- keyed on the point being bit-identical
            # to a HANDED one -- was defeated by a vertex step landing a tenth of an ulp away.
            # What differs is the curve BEYOND the end. One evaluation, spent only when this fires.
            #
            # Aimed at where the cone puts the minimum: `sigma/c` outward. A FIXED reach cannot
            # serve both phantoms measured -- one bracket width landed on the well 0.48 licences
            # out at 130*pi^2 and OVERSHOT the one 0.255 licences out at 25*pi^2, reading the far
            # wall and passing it. On a genuine root at the end `sigma` IS the floor and the reach
            # is a few ulps, where the probe reads the same.
            #
            # A probe reads "falling" only within 1.5x of the minimum's true distance, so a slope
            # still off by a factor would overshoot again. Halving twice covers a reach up to 6x
            # long, at one evaluation a step, stopping at the first probe that falls.
            phantom = False
            for frac in _PROBE_STEPS:
                step = max(frac * reach, 4.0 * np.spacing(abs(x_best)))
                probe = x_best + side * step
                # Straight through the objective, NOT through `f`: `f` clamps into the bracket,
                # and the whole point of this probe is to look outside it. The first version
                # called `f` and the probe landed back on the widened end, four ulps away, where
                # it read the same as the phantom and passed it.
                s_probe = obj.tension(obj.from_lam(probe))
                if s_probe < _DESCENT_RATIO * s_best:
                    # A falling curve alone does not make a phantom once the radius, not the tie,
                    # decides "at an end": a genuine root located only to `_xatol` close to the
                    # end has a radius that crosses the end too, and its probe lands ON the root
                    # and reads low. Where the minimum is separates them. A probe falls only
                    # within half a step of the minimum, so a minimum inside the bracket leaves
                    # the probe at most half a step past the end; the phantom's lies outside.
                    phantom = side * (probe - end) > max(tie, 0.5 * step)
                    break
            inside = (lo < x_best < hi) and not phantom
        accepted = bool(inside and s_best < min(ends))
    if not accepted and reason in ("converged", "flat", "stalled"):
        reason = "not_a_minimum"

    return RefineResult(lam=best[0], sigma=best[2], accepted=bool(accepted), reason=reason,
                        evaluations=obj.n_evals - start, bracket=(lo, hi),
                        delta_res=delta_res, steps=steps, width=width)


def _accept(best, lo, hi, prec, ends):
    """Is the located point a genuine INTERIOR minimum of the bracket it was handed?

    Two conditions, both free -- the endpoints were evaluated on the way in, so this costs
    nothing and cannot be skipped for economy.

    1. **Strictly interior.** Bounded minimisation returns an ENDPOINT when the true minimum
       lies outside the bracket, and next to a deep root that endpoint still passes any
       threshold test: measured before this check existed, 8 phantom roots against 20 real
       ones, each 2e-05 to 4e-04 from a genuine root with `sigma_1` between 1.6e-04 and
       9.8e-04. Smallness is not minimality.
    2. **Below both ends.** Which is the "compare against the cone OUTSIDE, where it must be
       rising" rule, read off the points the bracket already carries.

    An earlier version probed at `lam +/- 4*span` instead, meaning to sample the rising cone
    beyond the bracket. On an empty bracket one wall away from a root that probe **jumped over
    the eigenvalue entirely** and landed on the plateau beyond it, where the tension is high --
    so both sides read "rising" and the wall point was accepted. A probe whose reach is set by
    the bracket width cannot know what it is stepping across; the endpoints can, because they
    are where the bracket actually ends.

The interior margin is a few ULPS, not a fraction of the width. It was
    `min(xatol, width/4)`, and the `width/4` term rejected genuine minima that sit off-centre:
    on the multiplicity-3 degeneracy at `493.48022`, the bracket `[493.480219963,
    493.480220823]` really does contain the level, the refiner locates it to 7.4e-12 relative,
    and the true minimum sits 8.8e-08 from the left end against a demanded margin of 2.15e-07.
    Every degeneracy in the suite was rejected that way, and the composed pipeline scored
    separation 0.00 on all three.

    The margin was redundant as well as wrong. Condition 2 already excludes a locator that
    converged to an endpoint, because there `s_best == s(end)` and the strict `<` fails. So the
    interior test only has to mean "not AT an endpoint", which is a question about ulps.
    """
    lam, _, s_best = best
    margin = 4.0 * np.spacing(max(abs(lo), abs(hi)))
    if not (lo + margin <= lam <= hi - margin):
        return False
    return bool(s_best < min(ends))
