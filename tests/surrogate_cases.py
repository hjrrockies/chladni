"""TIER 1: the surrogate case set, fixed before any measurement.

TEST SUPPORT ONLY. The cheapest of the three tiers (`~2 us` per evaluation against 8-37 ms for
a rectangle pencil and 111 ms for GWW1), and the one a candidate must pass before anything more
expensive is spent on it.

# Fixed in advance, and widened only on its own merits

`CHLADNI_V1.md` section 7 is unambiguous: **every claim measured on the case that motivated it
was overstated, without exception** -- "2.2-2.7x cheaper" became 1.77-1.79x, "coverage 1.0 at
every precision" became lost eigenvalues on 3 of 7 cases -- until a case set fixed in advance
became the gate. Twice the narrowness cost coverage rather than only credibility. So this set is
chosen for what it exercises, not for what makes a result look good, and a case is added here
only for a reason that would have applied before the result that prompted it.

# What tier 1 can and cannot decide

**Can**: seed and cover behaviour, the exclusion ladder, the decline triggers, the refiner's
guards, every scorer control, and the SHAPE of the crossover -- the full six-precision grid runs
essentially free here, which is where the crossover is read before tier 2 pays for it.

**Cannot**: anything reading `sigma_3` and above. The surrogate puts every non-well curve at the
cap, so the higher branches have no structure -- median relative error 1.7e-02 for `sigma_1`,
9.2e-02 for `sigma_2` and 1.7e-01 for `sigma_3`. The (C1) gap test reads exactly that branch, so
**no multiplicity verdict taken here is final** until a real pencil confirms it. It also cannot
test anything about the rank, the basis or the floor, because it has none of them.

# The wells are stated in `k`

`Surrogate` is parameterized by `k`, which is also the canonical search variable, so these cases
carry `variable="k"` and their truth in `k`; `bracket_score.truth_in_lam` converts for scoring.
`c` is therefore a slope in `k` and the measured range is the one to use -- 0.445 to 1.725 at 19
real eigenvalues on four domains. A cone written with a `k`-slope but evaluated in `lam` is wrong
by the factor `2*k`, which is enough to make an honest uniform grid score coverage 0.
"""
from __future__ import annotations

import numpy as np

from harness import Case
from surrogate import Surrogate, Well

__all__ = ["make_case", "isolated", "pair_ladder", "cluster", "degenerate", "rough",
           "all_cases", "adequate", "C_K", "EPS"]

#: Wall slope in `k`. Measured 0.445 to 1.725 at 19 eigenvalues on disk, rect, eq_tri and
#: L-shape; 1.0 sits in that range and is the rectangle's own neighbourhood.
C_K = 1.0

#: Tension floor. A property of THE BASIS, not of the domain, and on a corner domain it varies
#: five orders WITHIN one scan. `1e-15` is a well-resolved plane-wave rectangle: measured
#: `sigma_mu` on real `chladni` pencils runs 3.3e-15 (square) to 6.7e-13 (rect 2x1).
#:
#: **It was 1e-13, and that made the bottom of the `prec` ladder unachievable.** The basis
#: adequacy contract is `prec*lam >= margin * eps/c`, and with `eps = 1e-13` and `c = 1` the
#: resolution limit is `1.4e-12` in `lam` at `lam ~ 49`, against `prec*lam = 4.9e-13` at
#: `prec = 1e-14`. So `prec = 1e-14` asked for something a third of what the objective could
#: deliver, and the searcher was scored as losing coverage on cases that were impossible --
#: 13 of 66 "failures" that were the CASE SET's fault, not the algorithm's. At `1e-15` the same
#: grid is 60/60.
#:
#: The lesson is tier 1's version of `rect_suite.assert_floor`: **the adequacy precondition has
#: to be checked on the cheap tier too**, or the cheap tier reports failures that mean nothing.
#: `adequate()` below is that check.
EPS = 1e-15


def make_case(label, wells, lo, hi, *, plateau=None, roughness=0.0, n=24, seed=0):
    """One tier-1 case: a counted `Surrogate` and its exact truth, in `k`."""
    s = Surrogate(wells, plateau=plateau, n=n, roughness=roughness, seed=seed)
    truth = np.array([w.k for w in s.wells for _ in range(w.mult)], dtype=float)

    def tensions(k):
        return s.tensions(float(k))

    case = Case(label=label, tensions=tensions, a=float(lo), b=float(hi),
                truth=np.sort(truth), complete=True, variable="k")
    case.surrogate = s
    return case


def isolated(n_wells=6, lo=3.0, hi=12.0, eps=EPS, c=C_K):
    """Well-separated wells at an irrational spacing, so nothing is accidentally degenerate.

    The base case: if a method cannot do this it cannot do anything, and its cost here is the
    per-eigenvalue figure everything else is compared against.
    """
    ks = lo + (hi - lo) * (np.arange(1, n_wells + 1) / (n_wells + 1.0)) ** 1.05
    return make_case(f"S:isolated/{n_wells}", [Well(k, eps, c) for k in ks], lo, hi)


def pair_ladder(bands=(0.5, 0.8, 1.3, 3.0, 30.0), prec=1e-8, *, k0=7.0, eps=EPS, c=C_K,
                lo=3.0, hi=12.0):
    """One case per band: a pair placed at `band * prec` RELATIVE, straddling the licence.

    `prec` is relative to `lam`, so a pair at relative separation `s` in `lam` sits at `s/2`
    in `k` -- `dlam/lam = 2*dk/k`. Getting that factor wrong dials every case to twice its
    intended band, which is precisely the class of error that made the rectangle sweep land
    at 0.5x before it was corrected.
    """
    out = []
    for band in bands:
        dk = 0.5 * band * prec * k0
        if k0 + dk == k0:
            continue            # not representable at this band: the cell is empty, not zero
        wells = [Well(3.9, eps, c), Well(k0, eps, c), Well(k0 + dk, eps, c),
                 Well(10.7, eps, c)]
        out.append(make_case(f"S:pair/b={band:g}/p={prec:.0e}", wells, lo, hi))
    return out


def cluster(mult=3, band=1.3, prec=1e-8, *, k0=7.0, eps=EPS, c=C_K, lo=3.0, hi=12.0):
    """A tight group of `mult` wells with UNEQUAL gaps, in ratio 1 : 2.2 : 1.

    Unequal deliberately. A method can suit two eigenvalues and suit a tight group of three or
    four with unequal gaps much less -- measured 3 of 8 on clusters against 6 of 12 on pairs
    for one method -- so an evenly-spaced cluster would be the easier case wearing the harder
    name.
    """
    steps = np.array([1.0, 2.2, 1.0])[: mult - 1]
    dk = 0.5 * band * prec * k0
    ks = k0 + np.concatenate([[0.0], np.cumsum(steps) * dk])
    if len(set(ks)) < len(ks):
        return []
    wells = [Well(3.9, eps, c)] + [Well(k, eps, c) for k in ks] + [Well(10.7, eps, c)]
    return [make_case(f"S:cluster{mult}/b={band:g}/p={prec:.0e}", wells, lo, hi)]


def degenerate(mult=3, *, k0=7.0, eps=EPS, c=C_K, lo=3.0, hi=12.0):
    """An EXACT degeneracy: `mult` coincident wells.

    Swept as a class rather than as one case, because the crossover ladder varies a separation
    and never reaches zero -- and `delta = 0` is the regime a shape optimization crosses
    whenever eigenvalues meet. No interval can separate a level from itself, so one bracket
    here is the correct answer and a scorer demanding one per mode is broken.
    """
    wells = [Well(3.9, eps, c), Well(k0, eps, c, mult=mult), Well(10.7, eps, c)]
    return make_case(f"S:degenerate{mult}", wells, lo, hi)


def rough(level=1e-10, *, lo=3.0, hi=12.0, eps=EPS, c=C_K):
    """Wells on a curve with a deterministic high-frequency perturbation.

    `1e-10` relative is the measured roughness of a real `sigma_1` at a SAFE rank; `1e-2` is
    what an unsafe one gives. A search must not mistake roughness for a well: a true well
    measured across its whole width has prominence `plateau/eps`, worst case 72, while
    roughness minima reach `1 + 2.4e-10` -- ten orders apart, so the discrimination is
    available and a method that fails it is failing something easy.
    """
    ks = [4.1, 6.3, 9.8]
    return make_case(f"S:rough/{level:.0e}", [Well(k, eps, c) for k in ks], lo, hi,
                     roughness=level)


def adequate(case, prec, *, margin=3.0, c=C_K):
    """Can this case's objective support this `prec` at all? The tier-1 precondition.

    `search.md` section 2.2's contract, in the form the BRACKETING stage needs: a bracket of
    width `prec*lam` is usable only while it is wider than the well's flat bottom, whose scale
    is `eps/c`. Below that the curve carries no signal and no method can separate anything.

    Scoring a case that fails this measures the case, not the method. Tier 2 has enforced this
    from the start through `rect_suite.assert_floor`; tier 1 did not, and paid for it -- see
    `EPS`. Cases failing it should be SKIPPED with the reason reported, never counted as
    failures and never silently dropped.
    """
    eps = min(w.eps for w in case.surrogate.wells)
    lam = float(np.median(np.asarray(case.truth, float) ** 2))
    k = float(np.sqrt(lam))
    epsc_lam = (eps / c) * 2.0 * k          # eps/c mapped from k into lam
    return prec * lam >= margin * epsc_lam


def all_cases(prec=1e-8):
    """The fixed tier-1 set at one `prec`."""
    return ([isolated()] + pair_ladder(prec=prec) + cluster(3, prec=prec)
            + cluster(4, prec=prec) + [degenerate(2), degenerate(3)] + [rough()])
