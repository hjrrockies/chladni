"""The rectangle case set: truth, the dial, and the two traps that make a sweep lie.

TIER 2's case machinery. Everything in the default tier here is pure closed-form arithmetic --
no pencil is built, so nothing costs a GSVD; the tests that construct cases are `slow`.

`CHLADNI_V1.md` section 5.4 names two traps and this file exists to keep both nailed down. Both
of them fired for real while this sweep was being built, which is the argument for the tests
rather than for the comments.
"""
from __future__ import annotations

import numpy as np
import pytest

import bracket_score as bs
import rect_suite as rs
from chladni.reference import rect_eig, rect_spectrum

BANDS = (0.5, 0.8, 1.3, 3.0)


# --- truth is complete, and it is not snapped ----------------------------------------------

@pytest.mark.parametrize("window", sorted(rs.CROSSOVER_WINDOWS))
def test_modes_in_window_is_complete(window):
    """Against `rect_spectrum`, which derives its index bound from `lam_hi` rather than
    guessing it, so it is complete by construction and can serve as the count oracle."""
    a, b = rs.CROSSOVER_WINDOWS[window]
    modes = rs.modes_in_window(a, b)
    ours = np.sort([rect_eig(m, n, 1.0, 1.0) for m, n in modes])
    theirs = np.sort(rect_spectrum(a, b, 1.0, 1.0))
    assert ours.size == theirs.size
    assert np.allclose(ours, theirs, rtol=1e-14)


def test_rect_spectrum_would_MERGE_the_bottom_of_the_ladder():
    """The reason truth comes from `rect_eig` directly. `snap` merges within `1e-13 * lam`,
    and the dialled separation at the bottom of the ladder is `5e-15` relative -- two orders
    below that floor. Snapped truth hands back an EXACT DEGENERACY wearing the split-pair
    label, which is the opposite case with the right name on it."""
    a, b = 10.0, 88.0
    L = 1.0 + rs.delta_for(0.5, 1e-14, a, b)
    direct = np.sort([rect_eig(m, n, L, 1.0) for m, n in rs.modes_in_window(a, b)])
    snapped = np.sort(rect_spectrum(a, b, L, 1.0))

    assert np.unique(direct).size == 4, "rect_eig keeps the pair apart"
    assert np.unique(snapped).size == 3, "rect_spectrum merges it"
    gap = np.diff(direct)[1]
    assert gap / np.spacing(direct[1]) == pytest.approx(34, abs=2), "34 ulps, representable"
    assert np.diff(snapped)[1] == 0.0, "and snapped to bit-identical"


# --- trap 1: the tightest pair is not the one you would dial --------------------------------

def test_the_tightest_level_in_the_wide_window_is_not_the_obvious_one():
    """On `[10, 520]` the obvious handle is (2,1)/(1,2) at r = 1.2, but (4,5)/(5,4) sits at
    r = 0.439 -- it splits at 0.366 times the rate, so at any `delta` its separation is 2.7x
    tighter than intended. A case placed by the obvious pair lands at a band nobody chose."""
    r, N, lam, m2 = rs.tightest_pair(*rs.WIDE_WINDOW)
    assert N == 41 and m2 == (16, 25), "the (4,5)/(5,4) level"
    assert r == pytest.approx(2 * 9 / 41)
    obvious = next(x for x in rs.splitting_rates(*rs.WIDE_WINDOW) if x[1] == 5)
    assert obvious[0] / r == pytest.approx(2.73, rel=0.01), "2.7x tighter than intended"


def test_the_relatively_tightest_pair_is_ABSOLUTELY_wider():
    """Which is why the classifier must key on the relative gap: purity is
    `span > prec * max(|t|, 1)`, so the relative one is the binding pair. An absolute argmin
    names (2,1)/(1,2) and the sweep then measures a different band from the one on its label."""
    delta = 1e-8
    L = 1.0 + delta
    lo, hi = rect_eig(1, 2, L, 1.0), rect_eig(2, 1, L, 1.0)
    clo, chi = rect_eig(4, 5, L, 1.0), rect_eig(5, 4, L, 1.0)
    near_abs, far_abs = abs(hi - lo), abs(chi - clo)
    near_rel, far_rel = near_abs / max(lo, hi), far_abs / max(clo, chi)

    assert far_abs > near_abs, "the (4,5) pair is absolutely WIDER"
    assert far_rel < near_rel, "and relatively TIGHTER"
    sep = bs.classify_separation([lo, hi, clo, chi], 1e-8)
    assert sep.at[0] == pytest.approx(min(clo, chi)), "the classifier picks the relative one"


# --- trap 2: a cluster does not split at its outer pair's rate ------------------------------

def test_a_cluster_splits_at_HALF_its_outer_pair_rate():
    """At `50*pi^2` the modes are (1,7), (5,5), (7,1), i.e. `m^2` in {1, 25, 49}. The outer
    pair spans 48, but (5,5) sits exactly halfway, so the minimum ADJACENT gap grows at 24 --
    half the outer rate. Every cluster case dialled with the pair formula landed at exactly
    0.5x its band, which is how this was found."""
    a, b = rs.CROSSOVER_WINDOWS["cluster"]
    r, N, lam, m2 = rs.tightest_pair(a, b)
    assert N == 50 and m2 == (1, 25, 49)
    assert r == pytest.approx(2 * 24 / 50), "adjacent, not outer"
    assert r == pytest.approx(0.5 * 2 * 48 / 50), "exactly half the outer-pair formula"


def test_a_level_that_cannot_split_is_not_offered_as_a_dial():
    """`m^2` must take at least two distinct values within the level, or the level does not
    separate under `delta` at all and dialling on it would divide by zero."""
    rates = rs.splitting_rates(*rs.WIDE_WINDOW)
    assert all(len(m2) >= 2 for _, _, _, m2 in rates)
    assert all(r > 0 for r, _, _, _ in rates)


# --- the dial lands on its band, at every precision -----------------------------------------

@pytest.mark.parametrize("band", BANDS)
@pytest.mark.parametrize("prec", bs.PREC_LADDER)
@pytest.mark.parametrize("window", sorted(rs.CROSSOVER_WINDOWS))
def test_the_dial_achieves_its_band(window, prec, band):
    """The cross product itself: 6 precisions x 4 bands x 3 windows, each landing where it
    was aimed. This is the test that failed 18 of 72 cells and exposed both traps."""
    a, b = rs.CROSSOVER_WINDOWS[window]
    L = 1.0 + rs.delta_for(band, prec, a, b)
    if L == 1.0:
        pytest.skip("the perturbation is not representable here: the cell is empty")
    truth = np.sort([rect_eig(m, n, L, 1.0) for m, n in rs.modes_in_window(a, b)])
    sep = bs.classify_separation(truth, prec)
    assert sep.band == pytest.approx(band), (
        f"{window}@{prec:.0e} band {band} achieved ratio {sep.ratio:.4g} at {sep.at}")
    assert sep.exercisable
    assert sep.must_split == (band > 1.0)


@pytest.mark.parametrize("window", sorted(rs.CROSSOVER_WINDOWS))
def test_the_quantization_shows_only_at_the_bottom_of_the_ladder(window):
    """~2.7% at 1e-14, and clean above it. The dial degrading is expected and is why cases are
    classified by what they ACHIEVED; it is reported rather than corrected."""
    a, b = rs.CROSSOVER_WINDOWS[window]

    def achieved(prec, band):
        L = 1.0 + rs.delta_for(band, prec, a, b)
        truth = np.sort([rect_eig(m, n, L, 1.0) for m, n in rs.modes_in_window(a, b)])
        return bs.classify_separation(truth, prec).ratio / band

    assert abs(achieved(1e-8, 0.5) - 1.0) < 1e-6, "exact well above the floor"
    assert abs(achieved(1e-14, 0.5) - 1.0) < 0.10, "a few percent at the floor"


def test_delta_for_is_the_inverse_of_the_splitting_rate():
    a, b = rs.CROSSOVER_WINDOWS["pair"]
    r = rs.tightest_pair(a, b)[0]
    assert rs.delta_for(1.3, 1e-8, a, b) == pytest.approx(1.3e-8 / r)


# --- the widest window holds the case the gate was widened for ------------------------------

def test_the_wide_window_holds_the_multiplicity_3_level():
    """`50*pi^2`, where (1,7), (5,5) and (7,1) coincide. The gate must assert its own coverage
    of this, or it can quietly stop covering the case it was widened for."""
    a, b = rs.WIDE_WINDOW
    assert a < 50 * rs.PI2 < b
    at_level = [(m, n) for m, n in rs.modes_in_window(a, b) if m * m + n * n == 50]
    assert sorted(at_level) == [(1, 7), (5, 5), (7, 1)]


def test_the_multiplicity_3_level_does_split_under_delta():
    """Present, splits, and into three distinct values -- not two with one left coincident."""
    a, b = rs.WIDE_WINDOW
    L = 1.0 + rs.delta_for(3.0, 1e-8, a, b)
    lv = np.unique([rect_eig(m, n, L, 1.0) for m, n in [(1, 7), (5, 5), (7, 1)]])
    assert lv.size == 3
    gaps = np.diff(lv)
    assert gaps.min() > 0
    assert gaps.max() / gaps.min() == pytest.approx(1.0, abs=0.01), "equal gaps: 1 : 1"


def test_the_wide_window_is_dense_enough_to_be_a_real_search_problem():
    a, b = rs.WIDE_WINDOW
    assert len(rs.modes_in_window(a, b)) >= 30
    assert len(rs.splitting_rates(a, b)) >= 10, "and holds many splittable levels"


# --- case construction (builds pencils) -----------------------------------------------------

@pytest.mark.slow
def test_a_crossover_case_carries_complete_unsnapped_truth():
    (case,) = rs.crossover_cases(1e-8, window="pair", bands=(1.3,))
    assert case.complete and case.variable == "lam"
    assert case.truth.size == 4
    sep = bs.classify_separation(case.truth, 1e-8)
    assert sep.band == pytest.approx(1.3)


@pytest.mark.slow
def test_split_degeneracy_cases_exercise_multiplets_not_only_pairs():
    cases = rs.split_degeneracy_cases(1e-6, bands=(1.3,))
    assert len(cases) == 3, "one per level of multiplicity 2, 3 and 4"
    for case in cases:
        sep = bs.classify_separation(case.truth, 1e-6)
        assert sep.band == pytest.approx(1.3), case.label
        assert not sep.degenerate, "these are the SPLIT counterparts"


@pytest.mark.slow
def test_the_floor_precondition_holds_on_a_representative_case():
    """`assert_floor` is a HARD PRECONDITION, not a diagnostic -- three separate ways of
    getting the discretization wrong each degraded the objective by orders while looking
    healthy. It has existed unused; this is the test that runs it."""
    (case,) = rs.family_A()
    ok, rows = rs.assert_floor(case)
    worst = max(r["sigma_mu"] for r in rows)
    assert ok, f"worst sigma_mu {worst:.2e} over {len(rows)} levels"


# --- complete, unsnapped truth for the pipeline ladder ------------------------------------------

def _window(L, H, a, b):
    from types import SimpleNamespace
    return SimpleNamespace(dims=(L, H), a=a, b=b)


@pytest.mark.parametrize("L, H, a, b", [(1.0, 1.0 / rs.PHI, 5.0, 300.0), (1.0 + 1e-3, 1.0, 10.0, 88.0),
                                        (1.0, 1.0, 10.0, 520.0), (1.0 + 1e-2, 1.0, 470.0, 520.0)])
def test_complete_truth_has_the_same_COUNT_as_rect_spectrum(L, H, a, b):
    """`rect_spectrum` is complete by construction; the unsnapped enumeration must agree on how
    many eigenvalues the window holds, or one of the two is dropping modes."""
    from chladni.reference import rect_spectrum

    got = rs.complete_truth(_window(L, H, a, b))
    assert got.size == len(rect_spectrum(a, b, L, H))


def test_complete_truth_does_NOT_merge_what_rect_spectrum_snaps():
    """On `rect(1 + 3e-14, 1)` the (2,1)/(1,2) pair is 1.8e-12 apart. `rect_spectrum` snaps it to
    one value; the ladder's truth must not, or a correct split scores as a wrong multiplicity."""
    from chladni.reference import rect_spectrum

    L = 1.0 + 3e-14
    snapped = np.asarray(rect_spectrum(10.0, 88.0, L, 1.0), dtype=float)
    got = rs.complete_truth(_window(L, 1.0, 10.0, 88.0))
    pair_snapped = snapped[(snapped > 49.0) & (snapped < 50.0)]
    pair = got[(got > 49.0) & (got < 50.0)]
    assert np.unique(pair_snapped).size == 1, "the premise: snap merges this pair"
    assert np.unique(pair).size == 2 and np.ptp(pair) > 1e-12


# --- the smoothness precondition must certify what the SEARCHER samples -------------------------

@pytest.mark.slow
@pytest.mark.parametrize("prec", [1e-8, 1e-14])
def test_assert_smooth_reads_the_SEARCHERS_own_seed(prec):
    """`assert_smooth` decides which gate cells are skipped as having no slope bound. It seeded
    its own 16-point grid with 4 doublings while the searcher seeds from Weyl with up to 6, so a
    cell could pass the precondition and still decline in the search, or the reverse -- the
    precondition certified a curve the searcher never looked at. It must be the searcher's own
    bootstrap, on the same objective and view.

    On the WIDE window, not the pair window: there Weyl expects about 4 eigenvalues and the
    searcher seeds `max(16, ceil(1.5*4)) = 16` points -- exactly `assert_smooth`'s 16 -- so the two
    grids coincide and a test on that window passes whatever the precondition does."""
    import bracket_score as bs
    from chladni.mps.bracket import _KView, _seed

    (case,) = rs.crossover_cases(prec, window="wide", bands=(1.3,))
    ok, hist = rs.assert_smooth(case, prec)
    view = _KView(bs.objective_for(case, prec))
    _, _, seed_hist, settled = _seed(view, view.lo, view.hi, 0.5 * prec)
    assert ok == settled
    assert hist == tuple(seed_hist), f"precondition saw {hist}, the searcher saw {seed_hist}"
