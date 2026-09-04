"""Scoring a search for eigenvalues, so that different searches can be compared honestly.

TEST SUPPORT ONLY.

The metric is RECALL first. lappy's known failure is missing eigenvalues, and an eigenvalue
that is never found cannot be recovered by any amount of downstream accuracy -- whereas a
located root that is a digit short is merely a located root that is a digit short. So the
headline number is "of the eigenvalues that exist in this interval, how many did you
return", counted WITH multiplicity, and everything else is secondary.

# The interface

A search is any callable

    search(tensions, a, b, *, modes=None) -> [(x, mult), ...]

where `tensions(x)` returns the ascending vector `sigma_1 <= sigma_2 <= ...`, and `modes(x)`
-- when the case supplies one -- returns the normalized candidate eigenFUNCTION at `x`, as a
vector sampled on a FIXED point set shared across all `x`.

`tensions` alone was the whole contract until the Gram angle and the eigenvector swing were
measured, and both need the eigenfunction rather than the tension. Widening rather than
special-casing keeps the layer method-agnostic -- FEM and boundary-integral methods produce
eigenvectors too -- and keeps candidates comparable, which is the reason the restriction
existed. A search that does not want `modes` ignores it; the uniform-grid control and the
lappy baselines do exactly that.

The fixed point set is not an extra requirement on the method: `MPSPencil`'s interior
`Sampling` is set at construction and only `k` changes, so vectors at different `x` already
live on the same points and compare with a plain dot product.

The variable `x` is whatever the caller set up; the harness never assumes `k` or `lam`. Truth
is given in the same variable. `docs/conventions.md` asks only that a parameter's name state
its variable, and here the name is the caller's business.

# Why a uniform grid is included as a baseline

`lipschitz-cone-mps.md` section 3.2 concedes that the honest comparison for its branch and
bound is "a well-chosen uniform grid", not a strawman fine one. So the grid is here, it is
tuned the way someone competent would tune it, and it is scored identically. A proposed
search that cannot beat it has not earned its complexity.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Score", "score", "Case", "run", "uniform_grid_search", "unfold",
           "levels_agree", "WEYL_DEFICIT_TOL"]

#: Above this, the three-term Weyl count says a mode is missing. Calibrated in
#: `chladni.asymp`: a correct consecutive set reaches at worst about +0.32, and deleting one
#: mode reads +1.16. The gap between those two numbers is the whole value of the diagnostic,
#: and 0.5 sits in it.
WEYL_DEFICIT_TOL = 0.5


@dataclass
class Score:
    """What a search achieved on one case. `recall` is the headline."""

    label: str = ""
    n_true: int = 0            # eigenvalues in the interval, with multiplicity
    n_found: int = 0           # returned, with multiplicity
    n_matched: int = 0         # correctly returned, with multiplicity
    n_missed: int = 0
    n_spurious: int = 0        # returned where no eigenvalue exists (or duplicated)
    n_mult_wrong: int = 0      # right location, wrong multiplicity
    n_missed_distinct: int = 0  # missed entirely, as opposed to undercounted
    levels_ok: bool = True     # collapsed values AND multiplicities agree, up to rtol
    weyl_deficit: float = float("nan")   # modes the three-term Weyl count says are missing
    max_rel_err: float = float("nan")
    med_rel_err: float = float("nan")
    evaluations: int = 0
    missed_at: list = field(default_factory=list)

    @property
    def recall(self):
        return self.n_matched / self.n_true if self.n_true else 1.0

    @property
    def precision(self):
        return self.n_matched / self.n_found if self.n_found else 1.0

    @property
    def perfect(self):
        return (self.n_missed == 0 and self.n_spurious == 0
                and self.n_mult_wrong == 0)


def unfold(pairs):
    """`[(x, mult), ...]` -> a flat ascending array with multiplicity repeated.

    A spectrum is a multiset and this is the data structure for one. Both sides of a
    comparison get unfolded, so truth and result are the same kind of object and the match is
    a multiset match rather than a comparison between two different encodings.

    Nothing is lost, because a true degeneracy is BIT-identical: two equal floats mean one
    doubly-degenerate eigenvalue and two unequal floats mean two eigenvalues. `chladni`'s
    `*_spectrum` functions already return this form; a search returns pairs, because
    "eigenvalue at x with multiplicity m" is what a solver actually concludes, and this is
    where the two meet.
    """
    if isinstance(pairs, np.ndarray) and pairs.ndim == 1:
        return np.sort(pairs.astype(float))
    out = []
    for item in pairs:
        try:
            x, m = item
        except (TypeError, ValueError):
            x, m = item, 1
        out.extend([float(x)] * int(m))
    return np.sort(np.asarray(out, dtype=float))


def levels_agree(found, truth, rtol=1e-6):
    """Up to `rtol`, did the search recover the ground truth's VALUES and MULTIPLICITIES?

    Both sides are unfolded, then collapsed at the SAME tolerance, then compared level by
    level. One coarse yes/no, complementary to `score`'s fine-grained diagnosis.

    It is here rather than folded into the matching because collapsing is a comparison-time
    choice, not a property of the spectrum. `chladni.reference` deliberately hands back an
    unfolded multiset with true degeneracies made bit-identical by `snap`; whether two values
    `rtol` apart should be read as one doubly-degenerate eigenvalue is the question this
    function asks, and different callers may answer it differently.

    It also covers the case `snap` cannot: a closed form whose repeated eigenvalues do not
    come out bit-identical at all, or a tabulated spectrum where the degenerate members were
    solved separately and agree only to the table's own precision.
    """
    from chladni.reference import collapse

    a = collapse(unfold(found), rtol)
    b = collapse(unfold(truth), rtol)
    if len(a) != len(b):
        return False
    return all(m1 == m2 and abs(v1 - v2) <= rtol * max(abs(v2), 1e-300)
               for (v1, m1), (v2, m2) in zip(a, b))


def weyl_audit(found_lam, domain):
    """How many eigenvalues the three-term Weyl count says are missing from `found_lam`.

    The only completeness check here that does not depend on the search being right, and
    therefore the only one available on a domain with no closed form. It cuts between distinct
    levels and compares the corner-corrected count against how many values lie below each cut.

    Two limits, both inherited and both worth stating at the call site. It is blind at the TOP
    of the range -- every cut lies below the highest level, so a mode missing there moves no
    cut, and a caller has to ask for more eigenvalues than it consumes. And a Robin or mixed
    boundary has no single perimeter coefficient, so it raises rather than guessing; that is
    reported as `nan` here rather than propagated.
    """
    from chladni.asymp import weyl_deficit

    try:
        return float(weyl_deficit(np.asarray(found_lam, dtype=float), domain))
    except (NotImplementedError, ValueError):
        return float("nan")


def score(found, truth, rtol=1e-6, label="", evaluations=0, case=None):
    """Compare a search's output against the complete truth for an interval.

    Both sides are unfolded to multisets and matched greedily, nearest first, each truth entry
    claimable once. Then the leftovers are DIAGNOSED rather than lumped together, because
    "you missed an eigenvalue" and "you found it but called it simple when it is double" are
    different bugs with different fixes:

    * an unmatched TRUTH entry whose value was matched by some other copy is a
      **multiplicity undercount** -- the search found the level and undercounted it;
    * an unmatched truth entry with nothing of the search's nearby is a **missed eigenvalue**;
    * an unmatched FOUND entry near a matched value is a **multiplicity overcount**;
    * an unmatched found entry near nothing is a **spurious root**.

    `n_missed` counts every unmatched truth entry either way, so `recall` stays the headline
    and stays honest; the split is diagnostic on top of it.
    """
    t = unfold(truth)
    f = unfold(found)
    claimed = np.zeros(t.size, dtype=bool)
    used = np.zeros(f.size, dtype=bool)
    errs = []

    order = sorted(range(f.size), key=lambda i: f[i])
    for i in order:
        best, bd = -1, np.inf
        for j in range(t.size):
            if claimed[j]:
                continue
            d = abs(f[i] - t[j]) / max(abs(t[j]), 1e-300)
            if d < bd and d <= rtol:
                best, bd = j, d
        if best >= 0:
            claimed[best] = True
            used[i] = True
            errs.append(bd)

    def near(vals, x):
        return vals.size > 0 and np.min(np.abs(vals - x)) <= rtol * max(abs(x), 1e-300)

    matched_vals = t[claimed]
    mult_under = missed = 0
    missed_at = []
    for j in np.nonzero(~claimed)[0]:
        if near(matched_vals, t[j]):
            mult_under += 1
        else:
            missed += 1
        missed_at.append(float(t[j]))

    complete = True if case is None else case.complete
    mult_over = spurious = 0
    for i in np.nonzero(~used)[0]:
        if near(matched_vals, f[i]):
            mult_over += 1
        elif complete:
            spurious += 1
        # else: truth is only a known prefix, so an unmatched find may be a real eigenvalue

    return Score(
        label=label,
        n_true=t.size,
        n_found=f.size,
        n_matched=int(claimed.sum()),
        n_missed=int((~claimed).sum()),
        n_spurious=spurious + mult_over,
        n_mult_wrong=mult_under + mult_over,
        n_missed_distinct=missed,
        max_rel_err=max(errs) if errs else float("nan"),
        med_rel_err=float(np.median(errs)) if errs else float("nan"),
        evaluations=evaluations,
        missed_at=missed_at,
        levels_ok=levels_agree(found, truth, rtol) if complete else True,
        weyl_deficit=(weyl_audit(case.to_lam(f), case.domain)
                      if case is not None and case.domain is not None else float("nan")),
    )


@dataclass
class Case:
    """One benchmark: a tension function, an interval, and what is known about the truth.

    `truth` is COMPLETE by default -- every eigenvalue in `[a, b]`, which is what makes recall
    meaningful. Set `complete=False` when only a prefix is known, as it is for every domain
    with no closed form: then a found value with no counterpart in `truth` is not counted
    spurious, because it may well be real.

    `domain` enables the Weyl audit, which is the only completeness check that does not
    depend on the search being right. It is what lets a corner domain be scored at all --
    `L_shape`, `GWW1` and `mushroom` have tabulated first-k spectra and no way to know whether
    an interval holds more.

    `variable` says whether the search works in `k` or `lam`, because Weyl's law is stated in
    `lam` and `conventions.md` asks that a parameter's name state its variable rather than
    leaving it to be inferred.
    """

    label: str
    tensions: object
    a: float
    b: float
    truth: object
    counter: object = None
    domain: object = None
    complete: bool = True
    variable: str = "k"
    modes: object = None       # modes(x) -> the normalized eigenfunction sample, or None

    def to_lam(self, x):
        x = np.asarray(x, dtype=float)
        if self.variable == "k":
            return x**2
        if self.variable == "lam":
            return x
        raise ValueError(f"variable must be 'k' or 'lam', got {self.variable!r}")

    def evaluations(self):
        c = self.counter
        return getattr(c, "calls", 0) if c is not None else 0

    def reset(self):
        if self.counter is not None and hasattr(self.counter, "calls"):
            self.counter.calls = 0


def run(search, cases, rtol=1e-6):
    """Score `search` on every case. Returns a list of `Score`."""
    out = []
    for c in cases:
        c.reset()
        found = search(c.tensions, c.a, c.b, modes=c.modes)
        out.append(score(found, c.truth, rtol=rtol, label=c.label,
                         evaluations=c.evaluations(), case=c))
    return out


def table(scores):
    """A readable summary; recall first, because that is the point."""
    w = max((len(s.label) for s in scores), default=5)
    lines = [f"{'case':<{w}}  {'recall':>7} {'prec':>6} {'miss':>5} {'spur':>5} "
             f"{'multX':>6} {'max err':>9} {'evals':>7} {'weyl':>6}"]
    for s in scores:
        wd = "  --  " if np.isnan(s.weyl_deficit) else f"{s.weyl_deficit:6.2f}"
        flag = "" if np.isnan(s.weyl_deficit) or s.weyl_deficit < WEYL_DEFICIT_TOL else " !"
        lines.append(
            f"{s.label:<{w}}  {s.recall:7.3f} {s.precision:6.3f} {s.n_missed:5d} "
            f"{s.n_spurious:5d} {s.n_mult_wrong:6d} {s.max_rel_err:9.1e} "
            f"{s.evaluations:7d} {wd}{flag}")
    return "\n".join(lines)


# --- the baseline every proposal has to beat ---------------------------------------------


def uniform_grid_search(n_per_well=8, tau=1e-3, refine=True, G=10.0):
    """A competently tuned uniform grid: sample, take discrete local minima, refine, count.

    This is section 3.2's "well-chosen uniform grid", the comparison the Lipschitz document
    says is the honest one. `n_per_well` sets the spacing as a fraction of the expected
    eigenvalue spacing rather than of the dip width -- sampling at the dip-width scale is the
    strawman nobody builds.

    Multiplicity uses both conditions measured to be necessary: (C1) the largest ratio gap in
    the low spectrum, and (C2) that each counted branch is a local minimum, without which a
    simple eigenvalue with a near neighbour is reported as a double.
    """
    from scipy.optimize import minimize_scalar

    def search(tensions, a, b, *, modes=None):
        n = max(int(n_per_well * (b - a) / _spacing(tensions, a, b)), 16)
        xs = np.linspace(a, b, n)
        S = [tensions(x) for x in xs]
        s1 = np.array([v[0] for v in S])
        idx = _local_minima(s1)
        out = []
        for i in idx:
            lo, hi = xs[i - 1], xs[i + 1]
            if refine:
                r = minimize_scalar(lambda x: tensions(x)[0] ** 2, bounds=(lo, hi),
                                    method="bounded", options={"xatol": 1e-13})
                x, f = r.x, np.sqrt(max(r.fun, 0.0))
            else:
                x, f = xs[i], s1[i]
            if f > tau:
                continue
            out.append((x, _multiplicity(tensions, x, lo, hi, G)))
        return out

    return search


def _local_minima(v):
    """Indices of discrete local minima, with ties broken to the LEFT.

    The asymmetry is deliberate. A strict `<` on both sides goes blind whenever a well is
    sampled symmetrically and its two flanking values are exactly equal -- and on the
    surrogate, whose curves are exact parabolas in the square, that happens for 85 of the
    184 grid sizes from 16 to 199. Real pencils are not symmetric about a well (0 of 144
    grid sizes on a disk, smallest neighbour gap 6e-4), so this is a surrogate artifact
    rather than a measured failure of real searches. The fix costs nothing and a detector
    that a tie can blind is wrong regardless, so it is taken.
    """
    return np.nonzero((v[1:-1] <= v[:-2]) & (v[1:-1] < v[2:]))[0] + 1


def _spacing(tensions, a, b):
    """A crude Weyl-free estimate of the eigenvalue spacing: the width of the interval over
    the number of plateau crossings a coarse pass sees. Deliberately simple -- the baseline
    is not supposed to be clever."""
    xs = np.linspace(a, b, 64)
    s1 = np.array([tensions(x)[0] for x in xs])
    lvl = 0.5 * np.median(s1)
    below = s1 < lvl
    n = max(int(np.sum(below[1:] & ~below[:-1])), 1)
    return (b - a) / n


def _multiplicity(tensions, x, lo, hi, G):
    """(C1) largest ratio gap, and (C2) each counted branch is a local minimum here."""
    t = tensions(x)
    tl, tr = tensions(lo), tensions(hi)
    n = min(len(t), len(tl), len(tr), 6)
    t, tl, tr = t[:n], tl[:n], tr[:n]
    if n < 2:
        return 1
    ratios = t[1:] / np.maximum(t[:-1], 1e-300)
    kappa = int(np.argmax(ratios)) + 1
    if ratios[kappa - 1] < G:
        kappa = 1
    while kappa > 1 and not (t[kappa - 1] < tl[kappa - 1] and t[kappa - 1] < tr[kappa - 1]):
        kappa -= 1
    return max(kappa, 1)
