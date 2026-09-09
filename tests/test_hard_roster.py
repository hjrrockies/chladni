"""TIER 3: the transfer tests. Does any of this survive off the rectangle?

All `slow`. `CHLADNI_V1.md` section 8 separates what should transfer from what was measured on
one family, and section 10 puts a second domain first in future work -- "worth more than any
remaining speedup, and the only item that could invalidate the design rather than tune it".

Every case is gated on the precision its BASIS can support, measured per domain. Asking a
domain for more than that measures the basis, and the roster's whole point is that the answer
differs per domain: the L-shape with pure Fourier-Bessel at its reentrant corner reaches
7.1e-15, while the equilateral triangle with plane waves stops at 2.5e-12.
"""
from __future__ import annotations

import numpy as np
import pytest

import bracket_score as bs
import hard_roster as hr
from chladni import asymp
from harness import weyl_audit
from candidates_v1 import v1_brackets
from chladni.mps.bracket import bracket_interval
from chladni.mps.refine import refine

pytestmark = [pytest.mark.slow]

PRECS = (1e-4, 1e-8, 1e-14)


def _skip_unless_supported(case, prec):
    finest = hr.finest_prec(case)
    if prec < finest:
        pytest.skip(f"UNEXERCISABLE: {case.label}'s basis supports prec >= {finest:.2e}, "
                    f"asked {prec:.0e}. Scoring here would measure the basis.")


# --- the precondition itself, measured per domain -------------------------------------------

def test_each_domain_reports_the_precision_its_basis_supports():
    """The numbers that decide which cells of the sweep exist. Recorded so a basis change has
    to confront them."""
    got = {c.label: hr.finest_prec(c) for c in hr.all_cases()}
    print("\n  finest achievable prec: "
          + ", ".join(f"{k} {v:.2e}" for k, v in got.items()))
    assert got["L_shape"] < 1e-14, (
        "pure Fourier-Bessel at the reentrant corner must reach the bottom of the ladder; "
        f"got {got['L_shape']:.2e}")
    assert got["disk"] < 1e-14
    # And the one that cannot, stated rather than discovered as a failure later.
    assert got["eq_tri"] > 1e-13, "plane waves on eq_tri stop well short of 1e-14"


def test_the_l_shape_really_has_a_reentrant_corner():
    """The property that makes it the transfer test: `pi/omega < 1`, so the gradient blows up
    and the eigenfunction carries an `r**(2/3)` singularity."""
    case = hr.l_shape_case()
    sing = hr.nonanalytic(case.domain)
    assert len(sing) == 1, "one non-analytic corner"
    omega = float(sing[0].omega)
    assert omega > np.pi, f"reentrant: omega = {omega:.4f} > pi"
    assert np.pi / omega == pytest.approx(2 / 3, rel=1e-9), "the r**(2/3) exponent"


# --- coverage, provable where truth is complete ------------------------------------------------

@pytest.mark.parametrize("prec", PRECS)
@pytest.mark.parametrize("name", ["disk", "eq_tri"])
def test_coverage_is_PROVED_where_the_spectrum_is_closed_form(name, prec):
    """Complete truth, so coverage is proved rather than audited. These two also carry genuine
    degeneracies, which is what makes them worth the cost."""
    case = hr.ROSTER[name]()
    _skip_unless_supported(case, prec)
    s = bs.run_case(v1_brackets(), case, prec, refine=refine)
    assert not s.declined, f"DECLINED: {s.decline_reason}"
    assert s.coverage == 1.0, f"missed {s.n_true - s.covered} of {s.n_true}"
    assert s.pure, f"{s.impure} impure of {s.n_brackets}"


@pytest.mark.parametrize("prec", PRECS)
def test_the_l_shape_is_covered_and_AUDITED(prec):
    """Truth here is a prefix, so coverage against it is necessary and not sufficient. The Weyl
    audit is the completeness signal, and on a domain with no closed form it is the only one
    there is -- it is what surfaced the recursion-floor bug that prefix truth did not."""
    case = hr.l_shape_case()
    _skip_unless_supported(case, prec)
    s = bs.run_case(v1_brackets(), case, prec, refine=refine)
    assert not s.declined, f"DECLINED: {s.decline_reason}"
    assert s.coverage == 1.0, f"missed {s.n_true - s.covered} of {s.n_true} tabulated levels"
    assert s.pure

    fn = v1_brackets()
    obj = bs.objective_for(case, prec)
    raw = fn(obj.tensions, obj.lo, obj.hi, prec=prec, objective=obj)
    found = [0.5 * (lo + hi) for lo, hi in bs.to_lam_brackets(list(raw), obj.variable)]
    deficit = weyl_audit(np.sort(found), case.domain)
    print(f"\n  L_shape @ {prec:.0e}: {len(found)} brackets, Weyl deficit {deficit:.2f}")
    if np.isfinite(deficit):
        assert abs(deficit) < 2.0, f"the audit says a level is missing: deficit {deficit:.2f}"


# --- the transferable claims of section 8 ------------------------------------------------------

@pytest.mark.parametrize("name", ["disk", "eq_tri", "L_shape"])
def test_the_slope_spread_in_k_is_measured_on_every_domain(name):
    """V1 section 8: "Do not assume that the slope spread in `k` stays at 5%. A domain with
    corners or a re-entrant angle may behave differently, and if the spread is large the case
    for a LOCAL slope model returns."

    So it is measured and REPORTED here rather than assumed. This test records the number; it
    asserts only that the global model remains usable, i.e. that the search does not decline
    for want of a bound.
    """
    case = hr.ROSTER[name]()
    obj = bs.objective_for(case, 1e-8)
    res = bracket_interval(obj, prec=1e-8)
    d = res.diagnostics
    print(f"\n  {name}: slope spread {d.slope_spread:.2f}, L {d.L:.4g} against max secant "
          f"{d.max_secant:.4g} (margin {d.margin:.2f}), {d.L_growth_after_use} growths")
    assert not res.declined, f"DECLINED: {d.decline_reason}"
    assert np.isfinite(d.slope_spread)


@pytest.mark.parametrize("name", ["disk", "eq_tri", "L_shape"])
def test_the_cost_per_eigenvalue_transfers(name):
    """V1's headline is ~30 evaluations per eigenvalue on one rectangle family. Whether that
    survives off the rectangle is the question this tier exists for, so the figure is printed
    and only the scorer's own cap is asserted."""
    case = hr.ROSTER[name]()
    s = bs.run_case(v1_brackets(), case, 1e-8, refine=refine)
    print(f"\n  {name}: {s.per_eigenvalue:.0f} evaluations per eigenvalue "
          f"({s.cost.search_evals} search + {s.cost.refine_evals} refine over {s.n_true})")
    assert s.cost_ok, f"{s.per_eigenvalue:.0f} per eigenvalue exceeds the cap"


@pytest.mark.parametrize("name", ["disk", "eq_tri"])
def test_degeneracies_on_a_smooth_domain_are_bracketed(name):
    """Both smooth domains carry genuine degeneracies -- geometric on the disk, arithmetic on
    the triangle -- and no interval can separate a level from itself, so one bracket per level
    is the correct answer."""
    case = hr.ROSTER[name]()
    lv, mult = np.unique(case.truth, return_counts=True)
    assert mult.max() >= 2, f"{name} should hold a degenerate level"
    s = bs.run_case(v1_brackets(), case, 1e-8, refine=refine)
    assert s.coverage == 1.0 and s.pure


# --- the floor spread the rectangle cannot show -------------------------------------------------

def test_the_l_shapes_FLOOR_SPREAD_IS_A_BASIS_ARTIFACT():
    """The five-order floor spread attributed to this domain does not survive a good basis.

    `notes/landscape_findings.md` section 20.1 measured the L-shape's tension floor at 3.1e-13
    at `k = 4.443` and 5.5e-08 at `k = 5.649` -- five orders WITHIN one scan -- and explained it
    structurally: whichever modes VANISH at the reentrant corner are analytic there and resolved
    almost exactly, while their neighbours carry the `r**(2/3)` singularity. That claim is the
    stated justification for the per-eigenvalue accuracy clause.

    With the recipe-grade basis -- 80 pure Fourier-Bessel terms at the reentrant corner -- the
    same levels over `[5, 40]` read:

        lam      9.640   15.197   19.739   29.521   31.913
        k        3.105    3.898    4.443    5.433    5.649
        eps    1.9e-15  5.4e-15  2.8e-15  8.2e-15  5.3e-15      spread 4.2x

    **5.3e-15 at `k = 5.649`, against the 5.5e-08 recorded there: seven orders.** So the spread
    is a property of the BASIS that was used, not of the domain, and it collapses from 1e+05 to
    4.2 once the basis is adequate. That is the same correction section 27.1 made about
    `rect81`, where a 9.2e+04 floor spread turned out to be a frozen rank over a wide window.

    What this does and does not touch. It does NOT retire the per-eigenvalue accuracy clause:
    the floor still varies, just by 4x rather than 1e+05, and a domain whose basis cannot be
    made adequate everywhere will still show the large spread -- which is exactly why
    `finest_prec` is measured per case rather than assumed. What it does retire is "the L-shape
    is the domain where the floor varies five orders", which was a statement about a basis.
    """
    case = hr.l_shape_case(a=5.0, b=40.0)
    p = case.pencil
    lv = np.unique(case.truth)
    floors = np.array([p.tension(float(x)) for x in lv])
    spread = floors.max() / floors.min()
    print(f"\n  L_shape [5, 40], {p.n_cols} pure-FB columns: "
          f"floors {['%.1e' % f for f in floors]}, spread {spread:.1f}x, "
          f"finest_prec {hr.finest_prec(case):.2e}")

    assert floors.max() < 1e-13, (
        "with a recipe-grade basis every level is resolved to better than 1e-13; the recorded "
        f"5.5e-08 at k = 5.649 was the basis, not the domain (got {floors.max():.2e})")
    assert spread < 100.0, f"and the spread is small, not five orders: {spread:.1f}x"
    # The level the notes single out is present and is well resolved.
    i = int(np.argmin(np.abs(lv - 31.9126)))
    assert abs(lv[i] - 31.9126) < 1e-3
    assert floors[i] < 1e-13, f"k = 5.649 reads {floors[i]:.2e}, not 5.5e-08"
