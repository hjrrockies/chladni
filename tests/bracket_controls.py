"""Controls: the bracket sets that validate the METRIC rather than any search.

TEST SUPPORT ONLY. `CHLADNI_V1.md` section 5.2. These are not baselines -- they are allowed to
read truth, which no candidate may do -- and they exist because a metric nobody has tried to
break is a metric nobody has any reason to believe. This apparatus found every one of the ~15
defects in the originating repo; inspection found none of them.

# Two halves, and the second is the one that gets forgotten

**Adversaries must FAIL**, each by the criterion meant for it. If `single_bracket` is rejected
by spurious rather than by purity, the metric is rejecting it for the wrong reason and will let
through the next thing that is wrong the same way.

**Oracles must PASS.** They catch a metric too strict to accept a correct answer -- which is the
failure that looks like rigour, and the one an adversary suite alone cannot see. The acceptance
rule is a LATTICE (one bracket per true eigenvalue at the fine end, one per level of `S(prec)`
at the coarse end, anything between), so all five of these are correct answers and a scorer that
rejects any of them is broken.

# The contract

Same seam as a candidate (`docs/design/candidates.md` section 1):

    brackets(tensions, a, b, *, prec, modes=None) -> [(lo, hi), ...] | Declined

returning intervals in the OBJECTIVE'S variable. An oracle factory therefore takes its `truth`
in that same variable -- use `truth_in(truth, variable)` when the case is parameterized by `k`
and truth, as always, arrives in `lam`.
"""
from __future__ import annotations

import numpy as np

from objective import Declined

__all__ = ["truth_in", "whole_window", "grid_partition_unevaluated", "evaluated_fine_grid",
           "oracle_minus_one", "oracle_two_merged", "brackets_in_the_gaps",
           "declines_everything", "oracle_fine", "oracle_coarse", "oracle_mixed",
           "oracle_split_cluster", "oracle_endpoint_exact", "ADVERSARIES", "ORACLES"]


def truth_in(truth, variable):
    """Truth arrives in `lam`; an oracle emitting in `k` needs it there."""
    t = np.sort(np.asarray(truth, dtype=float))
    return np.sqrt(t) if variable == "k" else t


def _pad(x, frac):
    """A bracket of relative half-width `frac` around `x`."""
    h = frac * max(abs(float(x)), 1.0)
    return float(x) - h, float(x) + h


def _levels(truth, prec, variable="lam"):
    """Group truth into levels of `S(prec)`, by diameter, in the emitting variable.

    `prec` is relative to `lam` -- it is the user's contract and it is stated in `lam` -- so
    the grouping is done in `lam` and the groups are converted afterwards. Grouping directly
    in `k` would apply `prec` to the wrong quantity and silently halve it.
    """
    t = np.sort(np.asarray(truth, dtype=float))
    lam = t * t if variable == "k" else t
    if lam.size == 0:
        return []
    groups, cur = [], [0]
    for i in range(1, lam.size):
        if lam[i] - lam[cur[0]] <= prec * max(abs(lam[i]), 1.0):
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    return [t[g] for g in groups]


# --- adversaries: each must fail, and by the criterion meant for it ------------------------

def whole_window():
    """`[(a, b)]`. Must fail PURITY wherever the window holds more than one level.

    It scores coverage 1.0 and spurious 0 while being useless, so purity is the only thing
    that can reject it. Measured in section 29: rejected on 18 of 21 cases, and the three it
    passed are the exact degeneracies, which hold ONE level -- where one bracket is the
    correct answer and passing is correct.
    """
    def brackets(tensions, a, b, *, prec, modes=None):
        tensions(0.5 * (a + b))
        return [(float(a), float(b))]

    return brackets


def grid_partition_unevaluated(max_cells=200_000):
    """A partition fine enough to GUARANTEE purity, returned without evaluating anything.

    The single most instructive adversary, and the reason `CostReport` charges the refinement
    a bracket hands on: this satisfies coverage, purity AND a search-only budget. Nothing but
    a cost model that prices the handoff can reject it.

    Not knowing where any eigenvalue is, a partition can only guarantee purity by making every
    cell narrower than `prec * lam`, which needs `(b - a) / (prec * a)` cells. At `prec = 1e-3`
    over a typical window that is tens of thousands, each charged 2-7 evaluations to reject --
    so it fails on cost by orders. At tight `prec` the required count is not materializable at
    all (about 4e10 cells at `prec = 1e-9`), the partition is truncated at `max_cells`, and it
    is then rejected on PURITY instead. Both are correct rejections and the sweep should record
    which regime it was in rather than pretending one number covers both.
    """
    def brackets(tensions, a, b, *, prec, modes=None):
        a, b = float(a), float(b)
        cell = max(prec * abs(a), np.spacing(b) * 4)
        n = int(min(np.ceil((b - a) / cell), max_cells))
        edges = np.linspace(a, b, max(n, 1) + 1)
        return [(float(edges[i]), float(edges[i + 1])) for i in range(edges.size - 1)]

    return brackets


def evaluated_fine_grid(n=4000):
    """`n` slivers, committed only where the tension is below a fraction of the plateau.

    Honest work, absurd amount of it. Must fail on COST: it pays `n` evaluations before it
    knows anything, and the per-eigenvalue figure is an order or two past a working method's
    ~30. Measured in section 29: rejected by cost on every case.
    """
    def brackets(tensions, a, b, *, prec, modes=None):
        xs = np.linspace(a, b, n + 1)
        v = np.array([tensions(0.5 * (xs[i] + xs[i + 1]))[0] for i in range(n)])
        tau = 0.5 * float(np.median(v))
        return [(float(xs[i]), float(xs[i + 1])) for i in np.nonzero(v < tau)[0]]

    return brackets


def oracle_minus_one(truth, *, variable="lam", drop=0, frac=1e-6):
    """A perfect answer with one target deleted. Must fail COVERAGE.

    The control for the failure that matters most: a lost eigenvalue cannot be recovered
    downstream by any amount of accuracy, so coverage is the hard metric and this is what
    proves it is actually being measured.
    """
    t = truth_in(truth, variable)

    def brackets(tensions, a, b, *, prec, modes=None):
        keep = [x for i, x in enumerate(t) if i != drop % max(t.size, 1)]
        return [_pad(x, frac) for x in keep]

    return brackets


def oracle_two_merged(truth, *, variable="lam"):
    """A perfect answer with the two most widely separated neighbours in one bracket.

    Must fail PURITY. Deliberately the WIDEST adjacent pair rather than the tightest: merging
    the tightest pair may be legal (that is what `prec` licences), so an adversary built on it
    would pass correctly and prove nothing.
    """
    t = truth_in(truth, variable)

    def brackets(tensions, a, b, *, prec, modes=None):
        if t.size < 2:
            return [_pad(x, 1e-6) for x in t]
        i = int(np.argmax(np.diff(t)))
        out = [(float(t[i]) * (1 - 1e-9), float(t[i + 1]) * (1 + 1e-9))]
        out += [_pad(x, 1e-6) for j, x in enumerate(t) if j not in (i, i + 1)]
        return sorted(out)

    return brackets


def brackets_in_the_gaps(truth, *, variable="lam"):
    """One bracket in each gap, none containing anything. Must fail COVERAGE.

    Coverage 0 with a plausible bracket count and plausible widths -- the shape of a method
    whose locator converged to the wrong thing everywhere. Distinguishes "found nothing" from
    "returned nothing", which a count-based metric cannot.
    """
    t = truth_in(truth, variable)

    def brackets(tensions, a, b, *, prec, modes=None):
        mids = 0.5 * (t[:-1] + t[1:])
        out = []
        for m in mids:
            lo, hi = _pad(m, 1e-3)
            if not np.any((t >= lo) & (t <= hi)):
                out.append((lo, hi))
        return out

    return brackets


def declines_everything(reason="constructed decline"):
    """Reports DECLINED always. Must never score as a pass.

    The control for the mechanism that makes declining honest. A method that declines whenever
    it is about to be wrong has a perfect record over what it accepted and has demonstrated
    nothing, so `BracketScore.ok` must be False for a declined run however good the partial
    result looks -- and this returns a PERFECT partial result to prove that it is.
    """
    def brackets(tensions, a, b, *, prec, modes=None):
        tensions(0.5 * (a + b))
        return Declined(reason, partial=[(float(a), float(b))])

    return brackets


# --- oracles: each must pass ---------------------------------------------------------------

def oracle_fine(truth, *, variable="lam", frac=1e-9):
    """One tight bracket per TRUE EIGENVALUE. The fine end of the lattice."""
    t = truth_in(truth, variable)

    def brackets(tensions, a, b, *, prec, modes=None):
        return [_pad(x, frac) for x in np.unique(t)]

    return brackets


def oracle_coarse(truth, *, variable="lam", frac=1e-9):
    """One bracket per LEVEL of `S(prec)`. The coarse end of the lattice.

    A within-`prec` cluster arrives as a single bracket, which is exactly what `prec` licences,
    so this must pass -- and it is the one a scorer keyed to true eigenvalues rather than to
    levels would wrongly reject.
    """
    t = truth_in(truth, variable)

    def brackets(tensions, a, b, *, prec, modes=None):
        out = []
        for g in _levels(t, prec, variable):
            lo, hi = float(np.min(g)), float(np.max(g))
            h = frac * max(abs(hi), 1.0)
            out.append((lo - h, hi + h))
        return out

    return brackets


def oracle_mixed(truth, *, variable="lam", frac=1e-9):
    """Fine on some levels, coarse on others. Anything between the lattice ends is legal."""
    t = truth_in(truth, variable)

    def brackets(tensions, a, b, *, prec, modes=None):
        out = []
        for i, g in enumerate(_levels(t, prec, variable)):
            if i % 2:
                out += [_pad(x, frac) for x in np.unique(g)]
            else:
                lo, hi = float(np.min(g)), float(np.max(g))
                h = frac * max(abs(hi), 1.0)
                out.append((lo - h, hi + h))
        return out

    return brackets


def oracle_split_cluster(truth, *, variable="lam", frac=1e-15):
    """A within-`prec` cluster SPLIT into one bracket per member. Must pass.

    Merging is permitted and never required, so splitting is not a failure: downstream the
    refiner finds both members, Phase D sees two roots inside `tol_merge`, and the answer is
    two levels of multiplicity 1. Penalising this would contradict the multiplicity convention
    -- and a scorer that counts brackets against levels rather than bounding them would.

    Members of an EXACT degeneracy cannot be separated by any interval, so they are emitted as
    one bracket; no interval can separate a level from itself and asking for that is the
    scorer bug this oracle guards against.
    """
    t = truth_in(truth, variable)

    def brackets(tensions, a, b, *, prec, modes=None):
        out = []
        for g in _levels(t, prec, variable):
            members = np.unique(g)
            if members.size == 1:
                out.append(_pad(members[0], max(frac, 1e-15)))
                continue
            for x in members:
                h = min(frac * max(abs(float(x)), 1.0),
                        0.25 * float(np.min(np.diff(members))))
                out.append((float(x) - h, float(x) + h))
        return sorted(out)

    return brackets


def oracle_endpoint_exact(truth, *, variable="lam", frac=1e-9):
    """Brackets whose ENDPOINT lands exactly on an eigenvalue. Must pass.

    Brackets are CLOSED and one containment predicate is shared by coverage and purity, so an
    eigenvalue sitting on an endpoint is contained. This is not a contrived case: every emitted
    endpoint must be a sampled point, and the cover phase's envelope minimiser for a cone is
    its APEX -- so the scan lands exactly ON a root and then splits there, leaving it on the
    shared endpoint of two intervals and in the interior of neither.
    """
    t = truth_in(truth, variable)

    def brackets(tensions, a, b, *, prec, modes=None):
        out = []
        for x in np.unique(t):
            h = frac * max(abs(float(x)), 1.0)
            out.append((float(x), float(x) + h))
        return out

    return brackets


#: Must FAIL, each by the criterion named. `declines_everything` is separate: it must report
#: DECLINED, which is neither.
ADVERSARIES = {
    "whole_window":              (whole_window, "purity"),
    "grid_partition_unevaluated": (grid_partition_unevaluated, "cost"),
    "evaluated_fine_grid":       (evaluated_fine_grid, "cost"),
    "oracle_minus_one":          (oracle_minus_one, "coverage"),
    "oracle_two_merged":         (oracle_two_merged, "purity"),
    "brackets_in_the_gaps":      (brackets_in_the_gaps, "coverage"),
}

#: Must PASS. The half that catches a metric too strict to accept a correct answer.
ORACLES = {
    "oracle_fine":           oracle_fine,
    "oracle_coarse":         oracle_coarse,
    "oracle_mixed":          oracle_mixed,
    "oracle_split_cluster":  oracle_split_cluster,
    "oracle_endpoint_exact": oracle_endpoint_exact,
}
