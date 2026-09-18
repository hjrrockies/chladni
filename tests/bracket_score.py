"""Scoring a BRACKET SET against complete closed-form truth.

TEST SUPPORT ONLY. No multiplicities appear anywhere here, which is the point: the joint
scorer's defect (section 27.2) is that recall-with-multiplicity is one number measuring two
things, so a method that never splits a pair but claims multiplicity 2 scores perfectly. A
bracket set carries no multiplicity, so that exploit does not exist.

# The acceptance rule is a LATTICE, not a bijection

    fine end    one bracket per true eigenvalue
    coarse end  one bracket per level of S(prec)

and anything between is acceptable. An accidental split of a within-`prec` cluster is NOT a
failure: downstream, refinement finds both, Phase D sees two roots inside `tol_merge` and the
answer is two levels of multiplicity 1, which is legal because merging is permitted and never
required. Penalising it would contradict the multiplicity convention.

Bounding the count rather than fixing it is also what kills the two degenerate strategies:
`[(a, b)]` fails purity, and ten thousand slivers fail spurious and cost.

# Coverage and purity are HARD; cost is what rejects the last degenerate strategy

A partition of the window returned *without evaluating anything* satisfies coverage, purity and
a search-only budget. Only a cost model that charges for the REFINEMENT HANDED ON rejects it,
which is why `CostReport` exists and why a flat per-bracket charge will not do: measured against
a flat 10, a flat charge overcharged tight brackets threefold, undercharged spurious ones by the
same factor, and reversed the answer to "should the search refine or hand off?" TWICE
(`CHLADNI_V1.md` section 3.4).

# Declining is neither a pass nor a failure

A declined run reports `declined=True` and can never be `ok`. The decline RATE is quoted beside
the pass rate over the same case set -- a method that declines whenever it is about to be wrong
has a perfect record over what it accepted and has demonstrated nothing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["BracketScore", "score_brackets", "CostReport", "Separation", "run_bracketer",
           "truth_in_lam", "objective_for", "run_case",
           "charge_refinement",
           "classify_separation", "predicted_refine_cost", "to_lam_brackets", "REJECT_COST",
           "COST_CAP_PER_EIG", "PREC_LADDER", "SEPARATION_BANDS"]

#: The `prec` ladder every candidate is scored on. Six points rather than
#: `CHLADNI_V1.md` section 5.5's three, so the merge/split crossover is located at each
#: precision separately instead of being sampled three times. Section 29 recorded results that
#: were IDENTICAL at 1e-8, 1e-11 and 1e-14 because one global separation ladder left the
#: requirement unchanged at all three -- the sweep has to move with `prec` or it measures cost
#: while claiming to measure capability.
PREC_LADDER = (1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14)

#: Multiples of `prec * lam` at which a case's tightest pair is placed. The first two must
#: MERGE and the next two must SPLIT, so every precision is exercised on both sides of its
#: own licence; the last two are far from any crossover and are kept once per window rather
#: than once per precision. A method that always merges and one that always splits both look
#: correct on a sweep that stays on one side.
SEPARATION_BANDS = (0.5, 0.8, 1.3, 3.0, 30.0, 300.0)

#: Evaluations per eigenvalue above which a bracket set is rejected on COST, counting the
#: refinement it hands on. Deliberately generous: `CHLADNI_V1.md` section 4 measures a working
#: method at ~30, a uniform grid dense enough to detect everything at ~43 (and it fails purity
#: anyway), and grid-then-bisect at 74-95 (rejected on COVERAGE, not here). Successful
#: family-B costs in section 29 run 33 to 366. So this rejects absurdity, not inefficiency --
#: efficiency is reported and compared, never a pass/fail line.
#:
#: 200 is where the two cost adversaries land on the wrong side and every honest method lands
#: on the right one: a 4000-point fine grid costs ~800 per eigenvalue on a 5-eigenvalue window
#: and a purity-guaranteeing unevaluated partition ~6000, against 30 for a working method, 43
#: for a uniform grid and 74-95 for grid-then-bisect. Nothing in the study sits between 95 and
#: 800, so the line is in a genuine gap rather than tuned to a case.
#:
#: It is PER EIGENVALUE and never flat. Section 29.2: a flat 1500-evaluation cap silently
#: penalised the 29-eigenvalue window and turned a 38-evaluation miss into the recorded claim
#: "nobody handles family E".
COST_CAP_PER_EIG = 200.0


# --- cost ----------------------------------------------------------------------------------

@dataclass(frozen=True)
class CostReport:
    """What a bracket set actually cost, including the refinement it hands on."""
    search_evals: int = 0
    refine_evals: int = 0
    per_bracket: tuple = ()
    measured: bool = True       # False when `refine_evals` came from the fitted model

    @property
    def total(self):
        return self.search_evals + self.refine_evals

    def per_eigenvalue(self, n_true):
        return self.total / n_true if n_true else float("inf")


def predicted_refine_cost(width, lam, prec):
    """Evaluations the refiner will spend on a bracket of this width. TIER 1 ONLY.

    The surrogate tier has no pencil to charge honestly against, so it gets a model. The shape
    is `a + b*log10(width/xatol)`, which is right because iterated parabolic interpolation
    contracts its interval by a roughly fixed factor per step.

    **Fitted on tier 2, not assumed.** 126 accepted refinements over rectangle families A and D
    at `prec = 1e-14`, bracket widths 1e-12 to 1e-1:

        evals = 3.73 + 0.274 * log10(width / xatol)      median |residual| 0.84, max 3.29

    against a measured range of 4 to 10. The provisional version of this function was fitted to
    `CHLADNI_V1.md` section 3.4's two quoted anchors -- 3 evaluations at width 1e-12 and 12 at
    width 20 -- and it reads 1.46 evaluations off on the same data, over-predicting wide
    brackets by half again (10.7 against 7.0 measured) and under-predicting tight ones. The
    anchors were not wrong about their own refiner; the slope simply does not transfer, so it
    was measured here.

    `REJECT_COST` is separate and flat, because rejection measured flat: 3 evaluations on every
    one of 10 empty brackets, against V1's quoted 2 to 7.
    """
    xatol = max(prec * abs(float(lam)), 4.0 * np.spacing(float(lam)))
    ratio = max(float(width), xatol) / xatol
    return float(np.clip(3.733 + 0.274 * np.log10(ratio), 2.0, 40.0))


#: What a spurious bracket costs the refiner to throw away. Measured flat at 3 evaluations
#: across 10 empty brackets placed mid-gap on real pencils.
#:
#: This number decides an architectural question, which is why it is measured rather than
#: guessed: keeping a doubtful region costs ~3 evaluations and dropping one costs an
#: eigenvalue, so the search should keep anything that still LOOKS like a dip. An early version
#: of the refiner took 28 evaluations to reject the same brackets -- it ground down a monotone
#: wall that `_accept` was always going to refuse -- and at that price the trade reverses.
REJECT_COST = 3.0


# --- the variable seam ---------------------------------------------------------------------

#: Ulps of slack added to a bracket endpoint on the `k -> lam` conversion. The round trip is
#: not exact: a bracket correctly fencing `k*` can fail to fence `lam* = k*^2` after squaring
#: because the endpoint rounds the wrong way. Observed as a one-ulp coverage loss whose
#: occurrence depended only on seed density -- i.e. it is luck, not algorithm, and paying four
#: ulps to remove it costs nothing measurable.
ULP_SLACK = 4


def to_lam_brackets(brackets, variable="lam", slack=ULP_SLACK):
    """Convert a bracket set to `lam` and widen by `slack` ulps.

    Done HERE, once, identically for every candidate, rather than in each method: squaring is
    monotone so the containment structure is preserved exactly, and the only thing at stake is
    the rounding -- which is a property of the conversion, not of the search that produced the
    interval. Leaving it to each method would let one candidate be scored on a wider bracket
    than another for no reason a reader could see.
    """
    out = []
    for lo, hi in brackets:
        lo, hi = float(lo), float(hi)
        if variable == "k":
            lo, hi = lo * lo, hi * hi
        for _ in range(int(slack)):
            lo = np.nextafter(lo, -np.inf)
            hi = np.nextafter(hi, np.inf)
        out.append((float(lo), float(hi)))
    return out


# --- separation classification -------------------------------------------------------------

@dataclass(frozen=True)
class Separation:
    """A case's tightest true separation, and which band of `prec` it ACHIEVED.

    Classify by what was achieved, never by what was requested. Near the float64 floor the
    dial is quantized (~2.7% at 1e-14) and an unrepresentable request rounds the perturbation
    to zero, silently yielding an EXACT DEGENERACY wearing the split-pair label -- the opposite
    case with the right name on it.
    """
    min_gap: float = float("inf")       # absolute, in lam
    min_gap_rel: float = float("inf")   # relative to the larger member
    ratio: float = float("inf")         # min_gap_rel / prec
    band: float | None = None           # the SEPARATION_BANDS entry it landed on, if any
    degenerate: bool = False            # the tightest pair is bit-identical
    exercisable: bool = True            # False when it hit neither a band nor degeneracy
    at: tuple = ()                      # the two eigenvalues realizing the tightest gap

    @property
    def must_split(self):
        """The pair is farther apart than `prec` licences merging, so a correct method splits."""
        return not self.degenerate and self.ratio > 1.0


def classify_separation(truth, prec, *, bands=SEPARATION_BANDS, rtol=0.25):
    """The tightest pair in `truth`, keyed to `prec`.

    The tightest pair is NOT the one you are dialling. Relative separation on a rectangle
    scales as `|m^2-n^2|/(m^2+n^2)` -- 3/5 for the (2,1)/(1,2) pair but 7/25 for (3,4)/(4,3) --
    so a case placed by the former leaves the latter twice as tight as intended. Hence this
    reads the WHOLE window's truth rather than the pair the case meant to dial.

    `exercisable=False` means the case landed between bands: report the cell of the sweep as
    empty and say why, rather than dropping it or relabelling it. Section 29 claimed
    `prec = 1e-14` could not be exercised at all and section 29.1 retracted that, so an
    unexercisable cell is a reportable measurement, not a defect.
    """
    t = np.sort(np.asarray(truth, dtype=float))
    if t.size < 2:
        return Separation()
    gaps = np.diff(t)
    # The tightest pair is the one with the smallest RELATIVE gap, not the smallest absolute
    # one, because that is the quantity the purity predicate is keyed to
    # (`span > prec * max(|t|, 1)`). The two disagree, and not marginally: on `[10, 520]` the
    # (4,5)/(5,4) pair at lam = 404.7 is relatively 2.7x tighter than (2,1)/(1,2) at
    # lam = 49.3 while being absolutely 3x WIDER, so an absolute argmin names the wrong pair
    # and the sweep silently measures a band nobody asked for. Found by dialling a case to
    # 0.5x and having it come back 1.37x.
    scales = np.maximum(np.abs(t[1:]), 1.0)
    i = int(np.argmin(gaps / scales))
    gap = float(gaps[i])
    scale = float(scales[i])
    rel = gap / scale
    ratio = rel / prec if prec > 0 else float("inf")
    degenerate = gap == 0.0
    band = None
    if not degenerate:
        for b in bands:
            if abs(ratio - b) <= rtol * b:
                band = float(b)
                break
    return Separation(min_gap=gap, min_gap_rel=rel, ratio=ratio, band=band,
                      degenerate=degenerate,
                      exercisable=bool(degenerate or band is not None),
                      at=(float(t[i]), float(t[i + 1])))


# --- the score -----------------------------------------------------------------------------

@dataclass
class BracketScore:
    label: str = ""
    prec: float = 0.0
    n_true: int = 0             # true eigenvalues in the window
    n_levels_prec: int = 0      # levels of S(prec): the COARSE end of the lattice
    n_brackets: int = 0
    covered: int = 0            # true eigenvalues lying in some bracket
    impure: int = 0             # brackets spanning more than prec*lam of true eigenvalues
    spurious: int = 0           # brackets containing no true eigenvalue
    overlapped: int = 0         # true eigenvalues in more than one bracket
    declined: bool = False      # the method stopped, reporting why
    decline_reason: str = ""
    cost: CostReport = field(default_factory=CostReport)
    cost_cap: float = COST_CAP_PER_EIG
    seconds: float = 0.0
    widths: tuple = ()

    @property
    def evaluations(self):
        return self.cost.total

    @property
    def coverage(self):
        return self.covered / self.n_true if self.n_true else 1.0

    @property
    def pure(self):
        return self.impure == 0

    @property
    def per_eigenvalue(self):
        return self.cost.per_eigenvalue(self.n_true)

    @property
    def cost_ok(self):
        """Within the per-eigenvalue budget. NEVER a flat cap (section 29.2)."""
        return self.per_eigenvalue <= self.cost_cap

    @property
    def in_lattice(self):
        """Bracket count within [coarse, fine]. Outside is not itself a failure -- it is
        implied by a purity or spurious failure -- but it localizes what went wrong."""
        return self.n_levels_prec <= self.n_brackets <= self.n_true

    @property
    def ok(self):
        """The filter's verdict: everything covered, nothing impure, cost within budget.

        A DECLINED run is never `ok`, whatever it returned. That is the whole content of
        declining being a third outcome: it may not be laundered into a pass by the partial
        result happening to look good.
        """
        return (not self.declined and self.coverage == 1.0 and self.impure == 0
                and self.cost_ok)

    @property
    def failures(self):
        """EVERY criterion that rejected this set, not just the first.

        More than one can fire, and asserting on a single verdict over-specifies: a fine grid
        fails cost AND purity, because a grid's cells are orders wider than `prec * lam` --
        which is the measured reason a uniform grid "holds coverage and fails purity" and not a
        defect in the grid. So the control suite asks whether the criterion meant for an
        adversary is AMONG its failures.
        """
        out = []
        if self.declined:
            out.append("declined")
        if self.coverage < 1.0:
            out.append("coverage")
        if self.impure:
            out.append("purity")
        if not self.cost_ok:
            out.append("cost")
        return tuple(out)

    @property
    def verdict(self):
        """One word for a table: 'pass', 'declined', or the criterion that rejected it."""
        if self.declined:
            return "declined"
        if self.coverage < 1.0:
            return "coverage"
        if self.impure:
            return "purity"
        if not self.cost_ok:
            return "cost"
        return "pass"


def _levels_at(truth, prec):
    """Levels of `S(prec)`: greedy grouping with each group's DIAMETER <= prec*lam.

    Anchored to the group's first member, which on a sorted array IS the diameter rule and
    matches `search.md` Phase D. `chladni.reference.collapse` does exactly this; it is
    reimplemented here only so the scorer has no dependency on the reference layer's default.
    """
    t = np.sort(np.asarray(truth, dtype=float))
    if t.size == 0:
        return []
    groups, cur = [], [t[0]]
    for x in t[1:]:
        if x - cur[0] <= prec * max(abs(x), 1.0):
            cur.append(x)
        else:
            groups.append(cur)
            cur = [x]
    groups.append(cur)
    return groups


def score_brackets(brackets, truth, prec, *, label="", cost=None, seconds=0.0,
                   declined=False, decline_reason="", cost_cap=COST_CAP_PER_EIG,
                   evaluations=None):
    """Coverage, purity, spurious, fineness and cost for one bracket set at one `prec`.

    `brackets` and `truth` must already be in the same variable -- use `to_lam_brackets`
    first. `cost` is a `CostReport`; `evaluations` is accepted as a shorthand for a
    search-only cost and exists so older callers keep working, but a score built that way
    cannot reject the unevaluated-partition adversary and says so by carrying
    `cost.measured = False`.
    """
    t = np.sort(np.asarray(truth, dtype=float))
    bs = [(float(lo), float(hi)) for lo, hi in brackets]
    groups = _levels_at(t, prec)

    if cost is None:
        cost = CostReport(search_evals=int(evaluations or 0), measured=evaluations is None)

    inside = [np.nonzero((t >= lo) & (t <= hi))[0] for lo, hi in bs]
    hit_count = np.zeros(t.size, dtype=int)
    for ix in inside:
        hit_count[ix] += 1

    impure = 0
    for ix in inside:
        if ix.size > 1:
            span = t[ix[-1]] - t[ix[0]]
            if span > prec * max(abs(t[ix[-1]]), 1.0):
                impure += 1

    return BracketScore(
        label=label, prec=prec, n_true=t.size, n_levels_prec=len(groups),
        n_brackets=len(bs), covered=int(np.sum(hit_count > 0)), impure=impure,
        spurious=sum(1 for ix in inside if ix.size == 0),
        overlapped=int(np.sum(hit_count > 1)),
        declined=bool(declined), decline_reason=str(decline_reason),
        cost=cost, cost_cap=float(cost_cap), seconds=seconds,
        widths=tuple(hi - lo for lo, hi in bs))


# --- the driver ----------------------------------------------------------------------------

def charge_refinement(brackets_lam, prec, *, refine=None, obj=None):
    """Evaluations the refinement of these brackets costs, per bracket.

    With `refine` supplied this RUNS it against the same `Objective`, so cache hits are free
    and the charge is what the handoff actually costs -- which is the only honest way to price
    it. Without one it falls back to `predicted_refine_cost`, and the report is marked
    `measured=False` so a reader knows which they are looking at.
    """
    per = []
    for lo, hi in brackets_lam:
        if refine is not None and obj is not None:
            per.append(int(refine(obj, lo, hi, prec=prec).evaluations))
        else:
            per.append(predicted_refine_cost(hi - lo, 0.5 * (lo + hi), prec))
    return tuple(per)


def run_bracketer(brackets_fn, obj, truth, prec, *, label="", refine=None,
                  cost_cap=COST_CAP_PER_EIG):
    """Drive one bracketer through one `Objective` and score what it returns.

    Cost is `obj.n_evals` -- cache MISSES -- plus the charged refinement. The objective is
    reset first so the measurement is of this bracketer alone; a candidate that inherited a
    warm cache from the run before would be scored on someone else's evaluations.

    `truth` is in `lam`, always. The returned brackets are converted from the objective's
    variable here, once, identically for every method.
    """
    import time

    import inspect

    obj.reset()
    # A bracketer may ask for the `Objective` itself by accepting an `objective` keyword. That
    # gives it the window's variable and `expected_count` without widening the mandatory seam,
    # and it avoids a second cache wrapped around the first -- which would make the charged
    # evaluation count depend on which of the two was consulted.
    kw = {}
    try:
        if "objective" in inspect.signature(brackets_fn).parameters:
            kw["objective"] = obj
    except (TypeError, ValueError):
        pass
    if obj.has_modes:
        try:
            if "modes" in inspect.signature(brackets_fn).parameters:
                kw["modes"] = obj.modes
        except (TypeError, ValueError):
            pass
    t0 = time.perf_counter()
    result = brackets_fn(obj.tensions, obj.lo, obj.hi, prec=prec, **kw)
    seconds = time.perf_counter() - t0

    declined = not bool(result) and hasattr(result, "reason")
    raw = list(result) if not declined else list(result.partial)
    bs = to_lam_brackets(raw, obj.variable)

    search_evals = obj.n_evals
    per = charge_refinement(bs, prec, refine=refine, obj=obj)
    # The refiner shares the cache, so its own misses are already in `obj.n_evals` when it ran
    # for real. Charge the difference rather than double-counting.
    refine_evals = (obj.n_evals - search_evals) if refine is not None else int(sum(per))

    return score_brackets(
        bs, truth, prec, label=label, seconds=seconds,
        declined=declined,
        decline_reason=getattr(result, "reason", ""),
        cost_cap=cost_cap,
        cost=CostReport(search_evals=search_evals, refine_evals=int(refine_evals),
                        per_bracket=per, measured=refine is not None))


def truth_in_lam(truth, variable):
    """A case's truth as `lam`. `harness.Case` states truth in the CASE's variable.

    Bracket scoring is always done in `lam`, because `prec` is relative to `lam` -- it is the
    user's contract and it is stated there. Grouping or comparing in `k` would apply `prec` to
    the wrong quantity and silently halve it.
    """
    t = np.sort(np.asarray(truth, dtype=float))
    return t * t if variable == "k" else t


def objective_for(case, prec):
    """An `Objective` over a `harness.Case`, in that case's own variable.

    Attaches `expected_count` when the case has a domain, per `CHLADNI_V1.md` section 6: it
    comes from the PDE, which is honestly where it comes from, and a search may read it FOR
    SIZING ONLY. A case without a domain -- the surrogate tier -- gets none, and a search that
    needs a count has to measure one for itself.
    """
    from chladni.mps.objective import Objective

    expected = None
    dom = getattr(case, "domain", None)
    if dom is not None:
        from chladni import asymp

        def expected(lo, hi, _d=dom, _v=getattr(case, "variable", "lam")):
            if _v == "k":
                lo, hi = lo * lo, hi * hi
            try:
                return float(asymp.expected_count(lo, hi, _d))
            except (NotImplementedError, ValueError):
                return float("nan")

    # `Case.modes` has existed since the harness was written (`harness.py`: "the normalized
    # candidate eigenFUNCTION at `x`, as a vector sampled on a FIXED point set shared across all
    # `x`") and `harness.run` has always passed it. This scorer dropped it on the floor, so a
    # candidate reaching the objective could not see it. Carried now; nothing reads it yet.
    return Objective(case.tensions, case.a, case.b, prec=prec,
                     variable=getattr(case, "variable", "lam"), expected_count=expected,
                     modes=getattr(case, "modes", None))


def run_case(brackets_fn, case, prec, *, label=None, refine=None,
             cost_cap=COST_CAP_PER_EIG):
    """Score one bracketer on one case at one `prec`. The sweep's unit of work."""
    obj = objective_for(case, prec)
    truth = truth_in_lam(case.truth, obj.variable)
    return run_bracketer(brackets_fn, obj, truth, prec,
                         label=label or case.label, refine=refine, cost_cap=cost_cap)
