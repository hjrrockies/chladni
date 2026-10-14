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


# --- the modes widening: what it costs, and what it must not move --------------------------
#
# `notes/solver_features.md` section 4 argues for widening the seam over post-processing the
# result: post-processing inside `MPSEigensolver` puts Phase D's collapse decision ABOVE the
# scored seam, which makes the decision that most needs benchmarking un-benchmarkable. Nothing
# reads `modes` yet; these pin the currency before a rule depends on it.


def _mode(x):
    """A stand-in eigenfunction on a fixed 4-point set. Direction turns with `x`."""
    v = np.array([np.cos(x), np.sin(x), np.cos(2 * x), np.sin(2 * x)])
    return v / np.linalg.norm(v)


def test_an_objective_with_no_modes_says_so_rather_than_failing():
    """A bare tension function is still a valid objective -- the surrogate tier has nothing
    else -- so a rule that wants eigenfunctions asks and falls back."""
    obj = Objective(cone, 1.0, 5.0)
    assert obj.has_modes is False
    assert obj.modes(3.0) is None


def test_modes_must_be_callable():
    with pytest.raises(TypeError, match="modes"):
        Objective(cone, 1.0, 5.0, modes=object())


def test_a_mode_at_an_ALREADY_EVALUATED_point_is_FREE():
    """The cost decision. One evaluation is one `x` at which the pencil was FACTORIZED, and a
    mode there reuses the cached `RegularizedPencil` -- measured at 5-7% over the tension alone,
    so charging it as a second evaluation would overstate it about twentyfold."""
    obj = Objective(cone, 1.0, 5.0, modes=_mode)
    obj.tension(3.0)
    assert obj.n_evals == 1
    obj.modes(3.0)
    assert obj.n_evals == 1, "a mode where a tension was already paid for is not a second miss"
    assert obj.n_mode_evals == 1


def test_a_mode_at_a_FRESH_point_is_CHARGED():
    """The other half, and the one that keeps the currency honest: without it a candidate could
    scan the whole window through `modes` for nothing."""
    obj = Objective(cone, 1.0, 5.0, modes=_mode)
    obj.modes(3.0)
    assert obj.n_evals == 1
    obj.tension(3.0)
    assert obj.n_evals == 1, "the factorization at this x was already charged"


def test_a_repeated_mode_costs_one_miss():
    obj = Objective(cone, 1.0, 5.0, modes=_mode)
    for _ in range(5):
        obj.modes(3.0)
    assert obj.n_evals == 1 and obj.n_mode_evals == 1
    assert obj.n_mode_calls == 5, "asks are diagnostic and counted separately"


def test_a_MODE_ONLY_point_is_charged_but_is_NOT_a_sample():
    """`samples` feeds the slope readers and `wide_bracket_for`, which want `(x, sigma_1)`. An
    `x` where only a mode was computed has no `sigma_1`, so putting it there would hand them a
    point with no value -- worse than not having it."""
    obj = Objective(cone, 1.0, 5.0, modes=_mode)
    obj.tension(2.0)
    obj.modes(4.0)
    xs, _ = obj.samples
    assert obj.n_evals == 2
    assert list(xs) == [2.0], "4.0 is charged, and is not a sample"


def test_reset_drops_the_MODE_cache_too():
    """`bracket_score.run_bracketer` resets so a candidate is charged for its own evaluations
    and nobody else's. A warm mode cache would break exactly that."""
    obj = Objective(cone, 1.0, 5.0, modes=_mode)
    obj.tension(3.0)
    obj.modes(3.0)
    obj.reset()
    assert (obj.n_evals, obj.n_mode_evals, obj.n_calls, obj.n_mode_calls) == (0, 0, 0, 0)


def test_the_currency_is_UNCHANGED_when_no_modes_are_in_play():
    """The whole commit's claim, at the seam every grid is denominated in: with no modes,
    `n_evals` is what it always was, eviction included."""
    obj = Objective(cone, 1.0, 5.0)
    for x in np.linspace(1.0, 5.0, 40):
        obj.tension(x)
    assert obj.n_evals == 40
    capped = Objective(cone, 1.0, 5.0, max_cache=16, modes=_mode)
    for x in np.linspace(1.0, 5.0, 40):
        capped.tension(x)
    assert capped.n_evals == 16, "eviction still governs the tension cache"


# --- the batch seam ----------------------------------------------------------------------


def _serial_and_batched(xs, max_cache=None):
    calls = []

    def batch(misses):
        calls.append(list(misses))
        return [cone(k) for k in misses]
    a = Objective(cone, 1.0, 5.0, max_cache=max_cache)
    for x in xs:
        a.tension(x)
    b = Objective(cone, 1.0, 5.0, max_cache=max_cache, batch=batch)
    b.tensions_many(xs)
    return a, b, calls


def test_a_batch_leaves_the_cache_and_the_count_as_the_serial_loop_would():
    """The seam's contract: misses are evaluated together and inserted in input order, so
    `n_evals`, `samples` and the cache's contents are bit-identical to a loop of `tension`."""
    xs = [1.5, 2.0, 2.0, 3.5, 1.5, 4.25]
    a, b, calls = _serial_and_batched(xs)
    assert a.n_evals == b.n_evals == 4
    assert calls == [[1.5, 2.0, 3.5, 4.25]]          # deduped, in input order, no hits
    assert list(a._cache) == list(b._cache)
    np.testing.assert_array_equal(a.samples[1], b.samples[1])
    assert b.tensions_many([2.0, 9.9]) and calls[-1] == [9.9]   # a later call batches only misses


def test_a_batch_evicts_exactly_as_the_serial_loop_does():
    xs = [1.0 + 0.5 * i for i in range(8)]
    a, b, _ = _serial_and_batched(xs, max_cache=3)
    assert list(a._cache) == list(b._cache) == xs[-3:]
    assert a.n_evals == b.n_evals == 3


def test_tensions_many_falls_back_to_the_scalar_function_without_a_batch():
    a = Objective(cone, 1.0, 5.0)
    vals = a.tensions_many([2.0, 3.0, 2.0])
    assert len(vals) == 3 and a.n_evals == 2 and vals[0] is vals[2]
