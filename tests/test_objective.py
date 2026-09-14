"""The shared candidate seam: cost accounting, the sample history, and the decline outcome.

`docs/design/candidates.md` sections 1, 3 and 5. Everything here is tier 1 -- no pencil, no
GSVD -- so it stays in the default run.

The property that actually matters is the cost one. Every claim the harness will make about a
candidate is denominated in `Objective.n_evals`, so a defect here does not produce a wrong
number in one place, it silently rescales every comparison in the study.
"""
from __future__ import annotations

import numpy as np
import pytest

from chladni.mps.objective import Declined, Objective


def cone(k, at=3.0, eps=1e-9, c=0.27):
    """The measured local model: a cone with a rounded bottom, plus a plateau branch."""
    return [np.sqrt(eps**2 + c**2 * (k - at) ** 2), 0.2]


# --- cost accounting -----------------------------------------------------------------------

def test_repeated_evaluation_costs_one_miss():
    """The reason the cache exists: `MPSPencil` does not memoize the GSVD, so a search that
    re-reads a bracket endpoint would be charged for a number it already had."""
    obj = Objective(cone, 1.0, 5.0)
    for _ in range(10):
        obj.tension(3.0)
    assert obj.n_evals == 1
    assert obj.n_calls == 10


def test_n_evals_matches_a_hand_count():
    obj = Objective(cone, 1.0, 5.0)
    xs = [1.0, 2.0, 3.0, 2.0, 1.0, 4.0]
    for x in xs:
        obj.tension(x)
    assert obj.n_evals == len(set(xs))


def test_the_whole_vector_is_cached_not_just_sigma_1():
    """Reading `sigma_2` is free, and it carries the information that decides clustering. A
    gate asking for it at a point the cover phase already visited must not pay again."""
    obj = Objective(cone, 1.0, 5.0)
    obj.tension(3.0)
    before = obj.n_evals
    assert obj.sigma(3.0, 1) == pytest.approx(0.2)
    assert obj.n_evals == before


def test_reset_clears_cost_but_not_the_callable():
    obj = Objective(cone, 1.0, 5.0)
    obj.tension(3.0)
    obj.reset()
    assert obj.n_evals == 0 and obj.n_calls == 0
    assert obj.tension(3.0) == pytest.approx(1e-9)


def test_eviction_is_off_by_default():
    """A window is ~1e3 evaluations; evicting one silently converts a hit into a charge."""
    obj = Objective(cone, 1.0, 5.0)
    for x in np.linspace(1.0, 5.0, 500):
        obj.tension(x)
    assert obj.n_evals == 500
    obj.tension(1.0)
    assert obj.n_evals == 500

    capped = Objective(cone, 1.0, 5.0, max_cache=16)
    for x in np.linspace(1.0, 5.0, 100):
        capped.tension(x)
    assert capped.n_evals == 16


# --- the sample history --------------------------------------------------------------------

def test_samples_are_ascending_and_carry_sigma_1():
    obj = Objective(cone, 1.0, 5.0)
    for x in (4.0, 1.0, 3.0, 2.0):
        obj.tension(x)
    xs, v = obj.samples
    assert np.all(np.diff(xs) > 0)
    assert v[np.argmin(v)] == pytest.approx(1e-9)
    assert xs[np.argmin(v)] == 3.0


def test_samples_support_a_three_point_descent_certificate():
    """The certificate an emitted endpoint must carry: `f(p) > f(q) < f(r)` on SAMPLED points,
    rigorous under continuity alone. Never an extrapolated root -- two-point secant roots from
    opposite walls of a close pair were measured to CROSS."""
    obj = Objective(cone, 1.0, 5.0)
    for x in (2.9, 3.0, 3.1):
        obj.tension(x)
    xs, v = obj.samples
    assert v[0] > v[1] < v[2]


def test_samples_is_empty_before_anything_is_evaluated():
    xs, v = Objective(cone, 1.0, 5.0).samples
    assert xs.size == 0 and v.size == 0


# --- the variable --------------------------------------------------------------------------

def test_the_k_lam_round_trip_is_the_identity_on_the_value():
    obj = Objective(cone, 1.0, 5.0, variable="k")
    assert obj.in_lam(3.0) == pytest.approx(9.0)
    assert obj.from_lam(obj.in_lam(3.0)) == pytest.approx(3.0)

    lam_obj = Objective(cone, 1.0, 25.0, variable="lam")
    assert lam_obj.in_lam(9.0) == 9.0 and lam_obj.from_lam(9.0) == 9.0


def test_sigma_past_the_retained_rank_reads_as_no_neighbour():
    """`inf`, not IndexError: the rank is re-derived at every parameter so the vector length
    moves, and a gate that asks for `sigma_2` where one direction survived should read "no
    neighbour anywhere near"."""
    obj = Objective(lambda k: [1e-9], 1.0, 5.0)
    assert obj.sigma(3.0, 1) == float("inf")


# --- validation ----------------------------------------------------------------------------

@pytest.mark.parametrize("lo, hi", [(5.0, 1.0), (1.0, 1.0), (np.nan, 2.0), (1.0, np.inf)])
def test_a_degenerate_interval_is_refused(lo, hi):
    with pytest.raises(ValueError):
        Objective(cone, lo, hi)


def test_a_non_callable_objective_is_refused():
    with pytest.raises(TypeError):
        Objective(3.0, 1.0, 5.0)


def test_an_unknown_variable_is_refused():
    with pytest.raises(ValueError):
        Objective(cone, 1.0, 5.0, variable="wavenumber")


# --- Weyl, sizing only ---------------------------------------------------------------------

def test_expected_count_is_absent_unless_supplied():
    assert Objective(cone, 1.0, 5.0).expected_count() is None


def test_expected_count_defaults_to_the_whole_window():
    obj = Objective(cone, 1.0, 5.0, expected_count=lambda a, b: (b - a) / 2)
    assert obj.expected_count() == pytest.approx(2.0)
    assert obj.expected_count(1.0, 3.0) == pytest.approx(1.0)


# --- declining -----------------------------------------------------------------------------

def test_a_decline_is_falsey_and_keeps_its_partial_brackets():
    """Declining is a third outcome, not an exception, and it is only honest if reported --
    so the partial result stays inspectable rather than being thrown away."""
    d = Declined("slope never settled", [(1.0, 2.0), (3.0, 4.0)])
    assert not d
    assert len(d) == 2
    assert list(d) == [(1.0, 2.0), (3.0, 4.0)]
    assert "slope never settled" in repr(d)


def test_a_decline_with_no_partial_brackets_is_still_a_decline():
    d = Declined("no progress")
    assert not d and len(d) == 0 and d.reason == "no progress"


# --- a bracket converted to `lam` must still fence the eigenvalue ---------------------------

@pytest.mark.parametrize("variable", ["k", "lam"])
def test_a_converted_bracket_covers_every_point_it_stood_for(variable):
    """Squaring is not exact, so `[a*a, b*b]` computed in float64 can miss `lam* = k*^2` by an
    ulp even when `[a, b]` correctly fenced `k*`. `bracket.py` documents a 4-ulp widening for
    exactly this and assigns it to the harness; `solve` hands the refiner the squared bracket
    unwidened. The conversion has to carry the widening itself."""
    rng = np.random.default_rng(7)
    obj = Objective(cone, 1.0, 400.0, variable=variable)
    for _ in range(200):
        a = float(rng.uniform(1.0, 400.0))
        b = float(np.nextafter(a, np.inf)) if rng.random() < 0.5 else a * (1 + 1e-12)
        lo, hi = obj.in_lam_bracket(a, b)
        for x in (a, b, 0.5 * (a + b)):
            assert lo <= obj.in_lam(x) <= hi
        from fractions import Fraction
        if variable == "k":
            assert Fraction(lo) <= Fraction(a) ** 2 and Fraction(b) ** 2 <= Fraction(hi), (
                "the EXACT square of each end must be inside, not only its rounding")
        assert hi - lo <= (obj.in_lam(b) - obj.in_lam(a)) + 16 * np.spacing(abs(hi)), (
            "a few ulps, not a fraction of the width")
