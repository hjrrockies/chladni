"""The multiplicity TOTAL on a tight cluster: where it is wrong, and whether it says so.

Ported from `wip/v1-promotion-and-multiplicity` (plan Phase 3a) onto main, with every table
RE-MEASURED here. The wip tables predate the refiner fixes, and those moved individual cells --
at 1e-6 band 1.6 on the even cluster went from right to wrong, at 1e-10 the same band from wrong
to right -- so a strict xfail list copied across would have recorded a different solver.

A wrong total is the error that moves an index: `lam_3` becomes a different eigenvalue and the
solver reports an accurate but non-consecutive spectrum. So the total is what is asserted, and
the grouping -- which is genuinely arbitrary in part of this range -- is not.

# The grid under the promoted defaults, measured 2026-09-14

Attained merge, the reach rule on the cache slope, the side search, no cross-check. Declared
total; `L` wrong and flagged uncertain, `S` wrong and SILENT, `f` right but flagged, `D` a
reasoned decline:

    N = 50, EVENLY spaced, truth 3
    prec     0.3  0.5  0.8  1.0  1.3  1.6  2.0  2.6  3.0  4.0  6.0
    1e-06     3    3    3    3    3    3    3    3    3    3    3
    1e-08     3    3   2L    3    3    3    3    3    3    3    3
    1e-10     3    3    3    3    3    3    3    3    3    3    3
    1e-12     3    3    3    3    3    3    3    3    3    3    3
    1e-14     3    3    3    3    3    3    3   3f    3    3    3

    N = 65, UNEVENLY spaced (1 : 2.2 : 1), truth 4
    prec     0.3  0.5  0.8  1.0  1.3  1.6  2.0  2.6  3.0  4.0  6.0
    1e-06     4    4   4f    4    D    D    4    4    4    4    4
    1e-08     4    D    4    4    4    4    D    4    4    4    4
    1e-10     4    4    4    4    D    4    4    4    4    4    4
    1e-12     4    4    4    4    4    4    4    4    4    4    4
    1e-14     4    4    4    4    4    4    4    4    4    D    4

One wrong cell, flagged, and none silent -- against 33 wrong (32 flagged, 1 silent) under the
previous defaults, which failed wherever a cluster's width fell between about one and eight merge
licences. That band belonged to the licence merge folding roots already resolved into separate
brackets and to `_collapse_ambiguous` recounting over a synthesised bracket whose radius placed
it; both are out of the defaults now, and `test_the_cross_check_LOSES_a_correct_answer...` below
keeps the evidence under the configuration it describes.

The wrong cell, the even cluster at 1e-8, band 0.8, is 2 of 3: the reach rule declines and the
side search cannot locate the lower member 0.3e-9 relative inside its sub-bracket's end, where the
refiner returns `no_parabola` and gives up. `TODO.md` records the refiner fallback.

# Tiers

The slow tier carries a 20-cell slice across both sides of the licence at 1e-8 and 1e-14; the
`robust` tier carries the whole grid. Every wrong cell is a STRICT xfail: the body runs, and when
a fix makes one pass the suite goes red until its entry comes off.
"""
from __future__ import annotations

import numpy as np
import pytest

import bracket_score as bs
import hard_roster as hr
import rect_suite as rs
from chladni.mps.multiplicity import multiplicity
from chladni.mps.search import solve

BANDS = (0.3, 0.5, 0.8, 1.0, 1.3, 1.6, 2.0, 2.6, 3.0, 4.0, 6.0)
PRECS = (1e-6, 1e-8, 1e-10, 1e-12, 1e-14)
EVEN, UNEVEN = (50, 3), (65, 4)

#: Wrong totals under the promoted defaults, keyed `(N, prec, band)`.
WRONG = {(50, 1e-8, 0.8)}
#: Wrong totals that nothing flags. None on this grid.
SILENT = set()

SLOW_PRECS, SLOW_BANDS = (1e-8, 1e-14), (0.5, 0.8, 1.3, 2.0, 3.0)

_WHY = ("the reach rule declines and the side search cannot locate a member 0.3e-9 relative inside "
        "its sub-bracket's end: the refiner returns no_parabola there. TODO.md, refiner fallback")


def cluster(N, m, band, prec):
    """The split counterpart of the degeneracy at `N*pi^2`, minimum adjacent gap at `band*prec`."""
    (case,) = rs.split_degeneracy_cases(prec, bands=(band,), levels={N: m})
    return case


_SOLVES = {}


def _solve(N, m, band, prec):
    key = (N, band, prec)
    if key not in _SOLVES:
        case = cluster(N, m, band, prec)
        _SOLVES[key] = (case, solve(bs.objective_for(case, prec), prec=prec))
    return _SOLVES[key]


def _cells(precs, bands, marks):
    for N, m in (EVEN, UNEVEN):
        for prec in precs:
            for band in bands:
                xf = ([pytest.mark.xfail(strict=True, reason=_WHY)]
                      if (N, prec, band) in WRONG else [])
                yield pytest.param(N, m, prec, band, marks=marks + xf,
                                   id=f"N{N}-{prec:.0e}-b{band:g}")


def _assert_total(N, m, prec, band):
    case, res = _solve(N, m, band, prec)
    if res.declined:
        assert res.diagnostics.decline_reason.strip(), "a decline must carry its reason"
        return
    assert sum(x.mult for x in res.levels) == m, (
        f"declared {sum(x.mult for x in res.levels)} of {m}; "
        f"n_uncertain={res.diagnostics.n_uncertain}")


@pytest.mark.parametrize("N, m, prec, band",
                         list(_cells(SLOW_PRECS, SLOW_BANDS, [pytest.mark.slow])))
def test_the_cluster_total_SLICE(N, m, prec, band):
    _assert_total(N, m, prec, band)


@pytest.mark.parametrize("N, m, prec, band", list(_cells(PRECS, BANDS, [pytest.mark.robust])))
def test_the_cluster_total_GRID(N, m, prec, band):
    _assert_total(N, m, prec, band)


def _wrong_cells(tier):
    """Wrong cells for `tier`: the slice's own cells for "slow", every one for "robust"."""
    mark = pytest.mark.slow if tier == "slow" else pytest.mark.robust
    for N, prec, band in sorted(WRONG):
        if tier == "slow" and not (prec in SLOW_PRECS and band in SLOW_BANDS):
            continue
        m = dict((EVEN, UNEVEN))[N]
        xf = ([pytest.mark.xfail(strict=True, reason="the one wrong cell nothing flags")]
              if (N, prec, band) in SILENT else [])
        yield pytest.param(N, m, prec, band, marks=[mark] + xf,
                           id=f"N{N}-{prec:.0e}-b{band:g}")


@pytest.mark.parametrize("N, m, prec, band", list(_wrong_cells("slow")))
def test_every_wrong_total_in_the_slice_is_LOUD(N, m, prec, band):
    """The deliverable of the safety net: the total is still wrong, it no longer claims to be
    right. `certified_counts` is the half of clause (3) `met_request` never asked."""
    _, res = _solve(N, m, band, prec)
    assert res.diagnostics.n_uncertain > 0
    assert not res.diagnostics.certified_counts


@pytest.mark.parametrize("N, m, prec, band", list(_wrong_cells("robust")))
def test_every_wrong_total_in_the_grid_is_LOUD(N, m, prec, band):
    _, res = _solve(N, m, band, prec)
    assert res.diagnostics.n_uncertain > 0


@pytest.mark.slow
def test_the_uneven_cluster_really_is_unevenly_spaced():
    """The premise, asserted. If the dial made this cluster even, every uneven cell above would
    silently measure something else."""
    t = np.unique(np.asarray(cluster(*UNEVEN, 1.0, 1e-8).truth, dtype=float))
    gaps = np.diff(t)
    assert t.size == 4
    assert gaps[1] / gaps[0] == pytest.approx(2.2, rel=0.05)
    assert gaps[2] / gaps[0] == pytest.approx(1.0, rel=0.05)
    even = np.diff(np.unique(np.asarray(cluster(*EVEN, 1.0, 1e-8).truth, dtype=float)))
    assert even.size == 2 and even[1] / even[0] == pytest.approx(1.0, rel=0.05)


@pytest.mark.slow
def test_the_MULTIPLICITY_precondition_holds_at_both_ends_of_the_ladder():
    """`20*eps/c/lam`, the resolution at which the basis can tell a pair apart, sits below every
    precision in the grid -- but at 1e-14 only by a factor of 1.3 to 2, which is not the same as
    every CELL being resolvable. What decides a cell is the members' separation against that
    resolution, not `prec`: at 1e-14, band 0.5, the partners are 0.47 licences apart, about
    5e-15 relative, BELOW it -- which is why the reach rule reads that pair as one eigenvalue
    without a flag, and why a benchmark excuses it through `basis_sufficiency`. The bracketing
    criterion would be the wrong bound here either way."""
    worst = 0.0
    for N, m in (EVEN, UNEVEN):
        for prec in (1e-6, 1e-14):
            _, rows = rs.assert_floor(cluster(N, m, 1.0, prec), finest_prec=prec)
            worst = max(worst, max(r["mult_prec"] for r in rows))
    assert worst < 1e-13, f"the multiplicity precondition admits only {worst:.1e}"


@pytest.mark.slow
@pytest.mark.parametrize("band", [
    1.3, 1.6,
    pytest.param(2.0, marks=pytest.mark.xfail(strict=True, reason=(
        "the previous configuration now DECLINES here at stage 1, cross-check on or off -- 'the "
        "slope estimate never settled' -- since the QR route around Accelerate's dgeqrf moved the "
        "objective; the promoted defaults read this cell right (4 of 4)"))),
])
def test_the_cross_check_LOSES_a_correct_answer_on_the_uneven_cluster(band):
    """Measured on main at 1e-8: ablated, four levels of multiplicity one and the right total;
    enabled, the total reads 2 and both levels are flagged. The guard is load-bearing on the even
    triple (`test_solve.py`) and net-negative here, which is why it is retired rather than tuned."""
    case = cluster(*UNEVEN, band, 1e-8)
    old = dict(merge="licence", rule="gap", reach_slope="outside")   # the previous defaults
    off = solve(bs.objective_for(case, 1e-8), prec=1e-8, collapse=False, **old)
    on = solve(bs.objective_for(case, 1e-8), prec=1e-8, collapse=True, **old)
    assert [x.mult for x in off.levels] == [1, 1, 1, 1]
    assert sum(x.mult for x in on.levels) < 4
    assert on.diagnostics.n_uncertain > 0, "and it says so"


@pytest.mark.slow
def test_no_wide_bracket_reach_reads_the_uneven_cluster_correctly():
    """Measured on main at band 1.3: reaches of 0.5 to 4 licences read 1 at the deepest root and
    6 licences reads 2. If any reach read 4 the defect would be a missing cap and the fix a
    number; none does."""
    case = cluster(*UNEVEN, 1.3, 1e-8)
    t = np.unique(np.asarray(case.truth, dtype=float))
    lic = 1e-8 * abs(t[-1])
    obj = bs.objective_for(case, 1e-8)
    res = solve(obj, prec=1e-8, collapse=False)
    rep = min(res.levels, key=lambda x: x.sigma).lam
    for d in (0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
        mu = multiplicity(obj.tensions, rep, prec=1e-8,
                          wide_bracket=(rep - d * lic, rep + d * lic)).mu
        assert mu < 4, f"reach {d} licences read {mu}"


@pytest.mark.slow
@pytest.mark.parametrize("name", ["disk", "eq_tri", "L_shape"])
def test_the_guard_is_QUIET_on_the_roster(name):
    """The roster is what a caller actually solves, and it must come back certified."""
    case = hr.ROSTER[name]()
    res = solve(bs.objective_for(case, 1e-8), prec=1e-8)
    assert res.diagnostics.certified_counts, res.diagnostics.uncertain
