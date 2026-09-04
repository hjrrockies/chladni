"""Objective hygiene: the floor precondition, and the rank truncation that must truncate.

`CHLADNI_V1.md` section 6, and step 3 of its build order. Nothing here is about a search --
these are properties the OBJECTIVE must have before any measurement taken on it means anything.

**The floor is a precondition, not a diagnostic.** Three separate ways of getting the
discretization wrong each degraded the objective by orders while looking healthy: a frozen rank
over a wide window, too few Fourier-Bessel terms on a corner domain, and corner-FB on a
rectangle, which has no singular corner at all. A floor check is what caught each of them.

Mostly `slow`: these build real pencils and evaluate them.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import rect_suite as rs
from chladni import asymp
from chladni.geometry import rect
from chladni.reference import rect_eig, rect_spectrum

pytestmark = pytest.mark.filterwarnings("ignore::chladni._core.PrecisionWarning")

A_WINDOW = (5.0, 120.0)


def _truth(L, H, a, b):
    return np.sort([rect_eig(m, n, L, H) for m, n in rs.modes_in_window(a, b)
                    if a <= rect_eig(m, n, L, H) <= b])


def _spurious(pencil, truth, a, b, n=900):
    """Sampled local minima of `sigma_1` not near any true eigenvalue.

    This is what a rank cutoff that fails to truncate actually costs. It does NOT move the
    floor -- which is why a floor check alone never noticed -- it makes `sigma_1` rough
    BETWEEN eigenvalues, which is where a search spends all of its time.
    """
    xs = np.linspace(a, b, n)
    v = np.array([pencil.tension(x) for x in xs])
    idx = np.nonzero((v[1:-1] < v[:-2]) & (v[1:-1] <= v[2:]))[0] + 1
    h = xs[1] - xs[0]
    hit = sum(1 for i in idx if np.min(np.abs(truth - xs[i])) < 3 * h)
    return len(idx) - hit, len(idx)


# --- expected_count, sizing only ------------------------------------------------------------

def test_expected_count_matches_the_true_count_to_a_few_percent():
    """Weyl with the perimeter correction, 1-11% on this family. Good enough to SIZE a seed
    and useless for anything else, which is what its docstring says."""
    d = rect(1.0, 1.0)
    for lo, hi in [(10.0, 520.0), (10.0, 88.0), (50.0, 300.0)]:
        got = float(asymp.expected_count(lo, hi, d))
        true = len(rect_spectrum(lo, hi, 1.0, 1.0))
        assert abs(got - true) / true < 0.11, f"[{lo}, {hi}]: {got:.2f} vs {true}"


def test_expected_count_is_clamped_and_additive():
    d = rect(1.0, 1.0)
    assert float(asymp.expected_count(100.0, 10.0, d)) == 0.0
    whole = float(asymp.expected_count(10.0, 520.0, d))
    halves = (float(asymp.expected_count(10.0, 200.0, d))
              + float(asymp.expected_count(200.0, 520.0, d)))
    assert whole == pytest.approx(halves)


# --- the rank cutoff must actually truncate -------------------------------------------------

@pytest.mark.slow
def test_a_cutoff_below_machine_epsilon_BARELY_TRUNCATES_at_the_top_of_the_window():
    """`rtol = 1e-16` is below `eps = 2.2e-16`, so under a RELATIVE threshold it keeps
    essentially everything the arithmetic can distinguish -- and it does so from the TOP down,
    where basis demand is highest. A floor checked only at the low end proves nothing.

    The claim here was once "retains EVERY column", `rank_at(hi) == n_cols`, and that was true
    only with the oversized collocation this suite used to build: 368 boundary points against
    92 columns left all 92 numerically independent, while the sized 138 rows saturate the rank
    at 82. The phenomenon survives the correction and the exact equality does not, so the test
    asserts the phenomenon -- a cutoff below machine epsilon barely truncates at the top, and
    truncates hard at the bottom.

    What the cutoff COSTS is measured in the next test, and that is the part that matters: not
    the floor, which stays healthy, but hundreds of spurious minima between the eigenvalues.
    """
    a, b = A_WINDOW
    p = rs.pencil_for(1.0, 1.0 / rs.PHI, a, b, rtol=1e-16)
    good = rs.pencil_for(1.0, 1.0 / rs.PHI, a, b, rtol=rs.SUITE_RTOL)
    assert p.rank_at(b) > 0.85 * p.n_cols, "almost nothing is discarded at the top"
    assert p.rank_at(b) > 1.5 * good.rank_at(b), "far more than a sound cutoff keeps"
    assert p.rank_at(a) < 0.6 * p.n_cols, "while the low end truncates normally"


@pytest.mark.slow
def test_the_inert_cutoff_costs_spurious_minima_and_NOT_the_floor():
    """Which is exactly why a floor check never noticed it. The damage is between the
    eigenvalues, where a search lives."""
    a, b = A_WINDOW
    truth = _truth(1.0, 1.0 / rs.PHI, a, b)
    inert = rs.pencil_for(1.0, 1.0 / rs.PHI, a, b, rtol=1e-16)
    good = rs.pencil_for(1.0, 1.0 / rs.PHI, a, b, rtol=rs.SUITE_RTOL)

    assert inert.tension(float(truth[0])) < 1e-13, "the floor is fine at 1e-16"
    n_bad, _ = _spurious(inert, truth, a, b)
    n_good, _ = _spurious(good, truth, a, b)
    assert n_bad > 100, f"but the plateau is rough: {n_bad} spurious minima"
    assert n_good == 0, f"and clean at SUITE_RTOL: {n_good}"


@pytest.mark.slow
def test_the_suite_rtol_is_chosen_by_the_FLOOR_not_the_spurious_count():
    """`1.5e-12` -- the derived default at `prec = 1e-8` -- also reaches zero spurious minima,
    but costs two orders of floor and fails the precondition, which would make the bottom of
    the `prec` ladder unexercisable. The floor is the binding constraint."""
    a, b = A_WINDOW
    case = rs.make_case("A", 1.0, 1.0 / rs.PHI, a, b)

    case.pencil = rs.pencil_for(1.0, 1.0 / rs.PHI, a, b, rtol=1.5e-12)
    loose_ok, loose = rs.assert_floor(case)
    case.pencil = rs.pencil_for(1.0, 1.0 / rs.PHI, a, b, rtol=rs.SUITE_RTOL)
    tight_ok, tight = rs.assert_floor(case)

    assert not loose_ok, "1.5e-12 fails the floor precondition"
    assert max(r["sigma_mu"] for r in loose) > 1e-13
    assert tight_ok and max(r["sigma_mu"] for r in tight) < 1e-14
    assert min(r["headroom"] for r in tight) > 1.0, "and supports prec = 1e-14"


@pytest.mark.slow
def test_the_floor_is_PROPORTIONAL_to_rtol_above_the_basis_own_floor():
    """Betcke's relation, in `chladni`'s RELATIVE convention: the floor inflates in proportion
    to the truncation threshold.

    Measured on `rect(1, 1/phi)` over `[5, 120]`, worst tension at a true eigenvalue:

        rtol      1e-16    1e-15    1e-13    1e-11    1e-09    1e-08
        eps      7.5e-16  1.2e-15  2.9e-14  2.3e-11  1.2e-09  1.1e-08
        eps/rtol   7.52     1.17     0.29     2.26     1.21     1.10

    A log-log slope of 1.13 over `rtol >= 1e-13`, with `eps/rtol` settling near 1.15. So

        eps  ~  max(eps_basis, rtol)

    with `eps_basis = 7.5e-16` here, and that is a PREDICTIVE rule rather than a table:

    * It says why `rtol` is safe to raise up to `prec` -- the floor it induces is about `rtol`,
      so at `rtol = prec` the discarded directions contribute at the precision asked for, which
      is by definition negligible.
    * It says exactly where `SUITE_RTOL` belongs. `sqrt(FINEST_PREC * eps_mach) = 1.49e-15` is
      the CROSSOVER: below it the truncation contributes less than the basis already does and
      buys nothing, above it the floor inflates linearly. That the derived rule lands on the
      crossover is the reason it is the right default and not merely a convenient one.
    """
    a, b = A_WINDOW
    truth = _truth(1.0, 1.0 / rs.PHI, a, b)
    rtols = np.array([1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8])
    eps = np.array([max(rs.pencil_for(1.0, 1.0 / rs.PHI, a, b, rtol=float(r)).tension(float(x))
                        for x in truth) for r in rtols])

    slope = np.polyfit(np.log10(rtols), np.log10(eps), 1)[0]
    assert slope == pytest.approx(1.0, abs=0.25), f"log-log slope {slope:.3f}, expected ~1"
    assert 0.1 < np.median(eps / rtols) < 10.0, "and the constant is order one"

    # Below the basis's own floor the relation saturates: more truncation precision buys nothing.
    tiny = max(rs.pencil_for(1.0, 1.0 / rs.PHI, a, b, rtol=1e-16).tension(float(x))
               for x in truth)
    assert tiny > 0.1 * eps[0] / 100, "saturated, not proportional, at rtol below eps_basis"
    assert tiny < 1e-14, "and the basis's own floor is what remains"


def test_the_suite_rtol_is_the_derived_rule_at_the_finest_prec():
    """Not a fitted constant: `chladni`'s own `sqrt(prec*eps)` evaluated at the tightest
    precision the sweep has to serve. The rule delivers about the floor each `prec` needs --
    4e-13 at 1e-8, 2e-15 at 1e-14 -- rather than a constant, which is what it was derived for."""
    assert rs.SUITE_RTOL == pytest.approx(np.sqrt(rs.FINEST_PREC * np.finfo(float).eps))
    assert rs.FINEST_PREC == min(__import__("bracket_score").PREC_LADDER)


# --- the floor holds across the window, checked from the top down ---------------------------

@pytest.mark.slow
def test_the_floor_holds_across_a_WIDE_window_at_every_level():
    """Checked over the whole window rather than at one end. The failure mode this guards is
    a rank frozen at one `lam`, which starves the high end: measured on this very case, a
    frozen rank reaches 3.5e-02 at the worst level against 2.1e-12 for threshold rank."""
    a, b = A_WINDOW
    case = rs.make_case("A:wide", 1.0, 1.0 / rs.PHI, a, b)
    ok, rows = rs.assert_floor(case)
    assert ok, f"worst sigma_mu {max(r['sigma_mu'] for r in rows):.2e}"
    top = max(rows, key=lambda r: r["lam"])
    assert top["sigma_mu"] <= rs.FLOOR_TARGET, "the TOP of the window, specifically"


@pytest.mark.slow
def test_the_rank_grows_with_lam_which_is_why_it_is_re_derived():
    """A rank frozen at one parameter cannot be right at another. Threshold rank reads the
    spectrum in front of it and cannot make that mistake."""
    a, b = A_WINDOW
    p = rs.pencil_for(1.0, 1.0 / rs.PHI, a, b)
    assert p.rank_at(a) < p.rank_at(b), "more oscillation supports more columns"
    assert not p.is_frozen


@pytest.mark.slow
@pytest.mark.parametrize("window", ["pair", "cluster"])
def test_the_floor_precondition_holds_on_the_crossover_cases(window):
    """The precondition runs on the cases that are actually scored, not on a stand-in."""
    for case in rs.crossover_cases(1e-8, window=window, bands=(1.3,)):
        ok, rows = rs.assert_floor(case)
        assert ok, f"{case.label}: worst {max(r['sigma_mu'] for r in rows):.2e}"
