"""The multiplicity TOTAL on a tight cluster: where it is wrong, and whether it says so.

Ported from `wip/v1-promotion-and-multiplicity` (plan Phase 3a) onto main, with every table
RE-MEASURED here. The wip tables predate the refiner fixes, and those moved individual cells --
at 1e-6 band 1.6 on the even cluster went from right to wrong, at 1e-10 the same band from wrong
to right -- so a strict xfail list copied across would have recorded a different solver.

A wrong total is the error that moves an index: `lam_3` becomes a different eigenvalue and the
solver reports an accurate but non-consecutive spectrum. So the total is what is asserted, and
the grouping -- which is genuinely arbitrary in part of this range -- is not.

# The grid under the promoted defaults, measured 2026-09-15

Attained merge, the reach rule on the cache slope, the side search, no cross-check, and the
refiner's `no_parabola` fallback and inward walk. Declared total; `L` wrong and flagged uncertain,
`S` wrong and SILENT, `f` right but flagged, `D` a reasoned decline:

    N = 50, EVENLY spaced, truth 3
    prec     0.3  0.5  0.8  1.0  1.3  1.6  2.0  2.6  3.0  4.0  6.0
    1e-06     3    3    3    3    3    3    3    3    3    3    3
    1e-08     3    3    3    3    3    3    3    3    3    3    3
    1e-10     3    3    3    3    3    3    3    3    3    3    3
    1e-12     3    3    3    3    3    3    3    3    3    3    3     (no longer run)
    1e-14     U    U    3    3    3    3    3   3f    3    3    3     (no longer run)

    N = 65, UNEVENLY spaced (1 : 2.2 : 1), truth 4
    prec     0.3  0.5  0.8  1.0  1.3  1.6  2.0  2.6  3.0  4.0  6.0
    1e-06     4    4    4    4    D    D    4    4    4    4    4
    1e-08     4    D    4    4    4    4    D    4    4    4    4
    1e-10     4    4    4    4    D    4    4    4    4    4    4
    1e-12     4    4    4    4    4    4    4    4    4    4    4     (no longer run)
    1e-14     U    U    4    4    4    4    4    4    4    D    4     (no longer run)

The 1e-12 and 1e-14 rows were dropped in the 2026-10-09 prune (`notes/test_inventory_2026-10-09.md`,
decision 3): they sit below douse's working precision. They are kept above as the measurement.

# `U`: the precondition, which this file scored past for its whole life

`TODO.md` section 8 -- **a case that violates a precondition measures the case, not the method**
-- and this grid was the fourth tier to learn it. At `prec = 1e-14`, bands 0.3 and 0.5 dial the
cluster tighter than the basis can resolve, so no counting rule can be right there and whatever
the cell reported was luck. At 50*pi^2, band 0.3: members 2.99e-15 apart against a multiplicity
resolution of 7.11e-15. Two cells of 55 here, two at 65*pi^2, eight of 55 at 170*pi^2 -- all of
them at 1e-14, and every one previously scored as a pass or a failure on its merits.

This is not bookkeeping. Under the attribution rule the 50*pi^2 1e-14 band-0.3 cell reads 2 of 3
where it used to read 3, and taking that as a regression would have been taking a basis limit for
a solver defect.

No wrong cell -- against 33 wrong (32 flagged, 1 silent) under the previous defaults, which failed
wherever a cluster's width fell between about one and eight merge licences. That band belonged to
the licence merge folding roots already resolved into separate brackets and to
`_collapse_ambiguous` recounting over a synthesised bracket whose radius placed it; both are out of
the defaults now. The tests that pinned the cross-check's own behaviour were retired on
2026-10-09 along with it (`git log -S test_the_cross_check_LOSES` for the evidence). The last wrong cell, the even cluster at 1e-8, band 0.8, was
2 of 3 until the refiner stopped giving up on a start with no parabola.

# Tiers

The slow tier carries a 20-cell slice across both sides of the licence at 1e-8 and 1e-10; the
`robust` tier carries the whole grid. Each cell passes when its total is right, declined with a
reason, or wrong and FLAGGED (`tests/prec_score.py`'s `flagged` verdict); only a silent wrong
total fails. The strict-xfail lists of wrong cells, and the "every wrong total is LOUD" tests over
them, went in the same prune: both lists were empty, so those tests collected nothing, and the
flagged contract now asserts the loudness on every cell.
"""
from __future__ import annotations

import numpy as np
import pytest

import bracket_score as bs
import hard_roster as hr
import rect_suite as rs
from basis_sufficiency import sufficient
from chladni.mps._regularize import _AVOID_GEQRF
from chladni.mps.search import solve

BANDS = (0.3, 0.5, 0.8, 1.0, 1.3, 1.6, 2.0, 2.6, 3.0, 4.0, 6.0)
PRECS = (1e-6, 1e-8, 1e-10)
EVEN, UNEVEN = (50, 3), (65, 4)

SLOW_PRECS, SLOW_BANDS = (1e-8, 1e-10), (0.5, 0.8, 1.3, 2.0, 3.0)


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
                yield pytest.param(N, m, prec, band, marks=marks,
                                   id=f"N{N}-{prec:.0e}-b{band:g}")


def _assert_total(N, m, prec, band):
    # THE PRECONDITION, which this file scored past for its whole life. `TODO.md` section 8: a
    # case that violates a precondition measures the CASE, not the method -- and three tiers
    # learned that separately before this one did. At `prec = 1e-14`, bands 0.3 and 0.5 dial the
    # cluster tighter than the basis can resolve: at 50*pi^2, band 0.3, the members sit 2.99e-15
    # apart against a multiplicity resolution of 7.11e-15, so no counting rule can be right here
    # and whatever this cell reported was luck. Measured unexercisable: 2 cells of 55 at 50*pi^2
    # and at 65*pi^2, 8 of 55 at 170*pi^2, all of them at 1e-14 -- a rung the grid no longer runs,
    # so the guard is now a guard rather than a live exclusion.
    suff = sufficient(cluster(N, m, band, prec), prec)
    if not suff.multiplicity:
        pytest.skip(f"UNEXERCISABLE: the basis cannot resolve this cluster's multiplicity at "
                    f"prec={prec:.0e} -- {'; '.join(suff.reasons)}")
    case, res = _solve(N, m, band, prec)
    if res.declined:
        assert res.diagnostics.decline_reason.strip(), "a decline must carry its reason"
        return
    total = sum(x.mult for x in res.levels)
    flagged = (not res.diagnostics.certified_counts or res.diagnostics.n_ambiguous > 0
               or any(x.count_uncertain or x.grouping_ambiguous for x in res.levels))
    # THE CONTRACT (2026-10-09): right, declined, or FLAGGED. A wrong total the solve says is
    # uncertain is honest; only a silent one fails (tests/prec_score.py, the `flagged` verdict).
    assert total == m or flagged, (
        f"declared {total} of {m}, unflagged; n_uncertain={res.diagnostics.n_uncertain}")


@pytest.mark.parametrize("N, m, prec, band",
                         list(_cells(SLOW_PRECS, SLOW_BANDS, [pytest.mark.slow])))
def test_the_cluster_total_SLICE(N, m, prec, band):
    _assert_total(N, m, prec, band)


@pytest.mark.parametrize("N, m, prec, band", list(_cells(PRECS, BANDS, [pytest.mark.robust])))
def test_the_cluster_total_GRID(N, m, prec, band):
    _assert_total(N, m, prec, band)


# --- the same grid on the pencil that SHIPS ------------------------------------------------
#
# Every cell above is scored on `rect_suite`'s pencil, fixed at `SUITE_RTOL` -- a threshold, and
# under `reg="auto"` an explicit `rtol=` stays one. So the grid above never exercises the default:
# the threshold at its own derived `rtol`, `sqrt(prec*eps)`, at every `prec` (since 2026-10-10;
# before it, the penalty at `prec >= 1e-8`). These rows rebuild each case's pencil with no regularization knob at all, which is what
# `MPSEigensolver` builds. Same cells, same precondition, same scoring.

#: There was once a wrong total on the default pencil, `(50, 1e-12, 0.3)`, 2 of 3 and silent, on
#: the threshold below 1e-8 and under the old default too: the side search's exclusion swallowed a member and the refusal at its inner end
#: read as a rejection (`notes/smooth_filter.md` 10.7). Fixed in `search.solve`, which now reads an
#: inner-end refusal too low for the root's own wall as a well inside the exclusion. Its rung, 1e-12,
#: is no longer run (2026-10-09 prune).
_SOLVES_DEFAULT = {}


def _solve_default(N, m, band, prec):
    from chladni.mps import MPSPencil
    from chladni.mps.objective import Objective

    key = (N, band, prec)
    if key not in _SOLVES_DEFAULT:
        case = cluster(N, m, band, prec)
        q = MPSPencil(case.pencil.bdry, case.pencil.interior, prec=prec)
        base = bs.objective_for(case, prec)
        obj = Objective(lambda x: q.tensions(float(x)), base.lo, base.hi, prec=prec,
                        variable=case.variable, expected_count=base.expected_count)
        _SOLVES_DEFAULT[key] = (case, solve(obj, prec=prec))
    return _SOLVES_DEFAULT[key]


def _default_cells(precs, bands, marks):
    for N, m in (EVEN, UNEVEN):
        for prec in precs:
            for band in bands:
                yield pytest.param(N, m, prec, band, marks=marks,
                                   id=f"N{N}-{prec:.0e}-b{band:g}")


def _assert_total_default(N, m, prec, band):
    suff = sufficient(cluster(N, m, band, prec), prec)
    if not suff.multiplicity:
        pytest.skip(f"UNEXERCISABLE: the basis cannot resolve this cluster's multiplicity at "
                    f"prec={prec:.0e} -- {'; '.join(suff.reasons)}")
    _, res = _solve_default(N, m, band, prec)
    if res.declined:
        assert res.diagnostics.decline_reason.strip(), "a decline must carry its reason"
        return
    total = sum(x.mult for x in res.levels)
    flagged = (not res.diagnostics.certified_counts or res.diagnostics.n_ambiguous > 0
               or any(x.count_uncertain or x.grouping_ambiguous for x in res.levels))
    # THE CONTRACT (2026-10-09): right, declined, or FLAGGED. A wrong total the solve says is
    # uncertain is honest; only a silent one fails (tests/prec_score.py, the `flagged` verdict).
    assert total == m or flagged, (
        f"declared {total} of {m}, unflagged; n_uncertain={res.diagnostics.n_uncertain}")


@pytest.mark.parametrize("N, m, prec, band",
                         list(_default_cells((1e-8, 1e-10), SLOW_BANDS, [pytest.mark.slow])))
def test_the_cluster_total_SLICE_on_the_default_pencil(N, m, prec, band):
    _assert_total_default(N, m, prec, band)


@pytest.mark.parametrize("N, m, prec, band",
                         list(_default_cells(PRECS, BANDS, [pytest.mark.robust])))
def test_the_cluster_total_GRID_on_the_default_pencil(N, m, prec, band):
    _assert_total_default(N, m, prec, band)


#: Whole-solve decline rate this grid is held to. NOT the estimator's number: `mult_suite` scores
#: `decline_rate < 0.10` on the reach rule and measures 2.1%, but those are declines of a COUNT
#: over a bracket, and the ones here are stage 1 refusing to bound a slope -- upstream of the
#: estimator entirely. Measured on this grid: 0 of 53 exercisable cells at `50*pi^2`, 6 of 53 at
#: `65*pi^2`, so 6 of 106 = 5.7%. Since the 2026-10-09 prune the grid stops at 1e-10, and the
#: same measure reads 5 of 66 = 7.6% (61 right, 0 flagged; all five "the slope estimate never
#: settled"): the dropped rungs carried one decline and forty passes. Set at 15% to leave room
#: without letting the rate double unnoticed; a change that pushes it past this is a change to report, not to absorb.
#:
#: The number to beat is not this one. On the held-out `145*pi^2` grid the rate is 23 of 45, and
#: `notes/landscape_findings.md` section 34 shows every one of those is the threshold rank
#: flipping rather than anything about the search.
MAX_DECLINE_RATE = 0.15

#: The wrong-but-FLAGGED rate this grid is held to. Flagging is honest (`_assert_total`), and that
#: is exactly why it needs a cap of its own: a method that flags every count it is unsure of -- or
#: every count -- would pass every cell and demonstrate nothing, the same trap the decline rate
#: guards. Measured 2026-10-09 on this grid at 1e-6..1e-10: 0 of 66. Set at 10%: room for real
#: flags, not for flagging as a strategy.
MAX_FLAG_RATE = 0.10


@pytest.mark.ulp_sensitive  # flips under tests/ulp_jitter.py and across CPUs; notes/jitter_sweep.py
@pytest.mark.robust
def test_the_DECLINE_RATE_is_quoted_beside_the_pass_rate():
    # (and the FLAG rate, since 2026-10-09: both honest outcomes are capped, so neither can
    # become a strategy)
    """`docs/design/candidates.md` section 5: **the decline rate is quoted beside the pass rate
    over the same case set, always.** This grid did neither -- `_assert_total` returns early on a
    decline, asserting only that it carries a reason, and nothing aggregated them. A method that
    declines whenever it is about to be wrong has a perfect record over what it accepted and has
    demonstrated nothing, and this grid could not have told the difference.

    Scored over the exercisable cells only, for the same reason `_assert_total` skips the others:
    a cell whose basis cannot resolve the cluster measures the basis.
    """
    n = declined = right = flagged = 0
    reasons = {}
    for N, m in (EVEN, UNEVEN):
        for prec in PRECS:
            for band in BANDS:
                case = cluster(N, m, band, prec)
                if not sufficient(case, prec).multiplicity:
                    continue
                n += 1
                _, res = _solve(N, m, band, prec)
                if res.declined:
                    declined += 1
                    key = res.diagnostics.decline_reason.split(":")[0]
                    reasons[key] = reasons.get(key, 0) + 1
                elif sum(x.mult for x in res.levels) == m:
                    right += 1
                elif res.diagnostics.n_uncertain > 0 or not res.diagnostics.certified_counts:
                    flagged += 1          # wrong but SAID so: honest, and scored as such
    rate = declined / n if n else 0.0
    print(f"\n  {right} right, {declined} declined, {flagged} wrong-but-flagged of {n} "
          f"exercisable cells (decline rate {rate:.1%})")
    for why, k in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"    {k:>3}  {why}")
    # THE CONTRACT: every cell is right, declined, or flagged. A silent wrong total is the only
    # outcome that fails here; which honest outcome a cell lands on can move with the last bits
    # of the arithmetic (tests/ulp_jitter.py), and is not asserted.
    silent = n - right - declined - flagged
    assert silent == 0, f"{silent} cell(s) wrong and unflagged"
    assert rate <= MAX_DECLINE_RATE, (
        f"decline rate {rate:.1%} over {n} cells, above {MAX_DECLINE_RATE:.0%}: {reasons}")
    flag_rate = flagged / n if n else 0.0
    assert flag_rate <= MAX_FLAG_RATE, (
        f"flag rate {flag_rate:.1%} over {n} cells, above {MAX_FLAG_RATE:.0%}: flagging is "
        f"honest, flagging everything is not an answer")


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
@pytest.mark.parametrize("name", ["disk", "eq_tri", "L_shape"])
def test_the_guard_is_QUIET_on_the_roster(name):
    """The roster is what a caller actually solves, and it must come back certified."""
    case = hr.ROSTER[name]()
    res = solve(bs.objective_for(case, 1e-8), prec=1e-8)
    assert res.diagnostics.certified_counts, res.diagnostics.uncertain
