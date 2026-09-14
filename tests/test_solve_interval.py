"""The black-box path, end to end: a `Domain` in, eigenvalues out.

`search.md` section 2.4a is the per-stage contract these cite and section 2.4 the pipeline's
three clauses. This is the first test in the repo that goes all the way from geometry to a
located eigenvalue, and it is the one `CLAUDE.md`'s "ultimate goal" paragraph describes.

**Smooth domains only, and that is not laziness.** Clause (1) is COMPLETENESS, and a prefix
truth table cannot distinguish a missing level from the end of the table -- so the L-shape,
which has only a 14-digit prefix, can audit accuracy but never completeness. The disk and the
equilateral triangle have complete closed-form spectra AND genuine degeneracies, so both
halves of clause (1), the count and the multiplicity, are provable rather than merely
uncontradicted.
"""
import numpy as np
import pytest

from chladni.eigensolver import Spectrum
from chladni.mps import PrecisionWarning
from chladni.geometry import disk, eq_tri
from chladni.reference import disk_spectrum, eq_tri_spectrum

from hard_roster import solver_for

pytestmark = pytest.mark.slow

#: `[a, b]` chosen to sit strictly between levels at both ends, so the truth in the window is
#: unambiguous and an endpoint cannot decide the count.
CASES = [
    ("disk", lambda: disk(1.0), lambda a, b: disk_spectrum(a, b, 1.0), 8.0, 60.0),
    ("eq_tri", lambda: eq_tri(1.0), lambda a, b: eq_tri_spectrum(a, b, 1.0), 60.0, 300.0),
]


def _case(name, dom_fn, truth_fn, a, b, prec=1e-8):
    dom = dom_fn()
    return dom, solver_for(dom, b, label=name, prec=prec), np.sort(truth_fn(a, b)), a, b


@pytest.mark.parametrize("name,dom_fn,truth_fn,a,b", CASES)
def test_it_finds_every_level_in_the_window(name, dom_fn, truth_fn, a, b):
    """Clause (1), completeness and multiplicity, as a multiset. COUNT first: a wrong count is
    a different failure from a wrong value, and asserting values first reports the wrong one."""
    _, solver, truth, a, b = _case(name, dom_fn, truth_fn, a, b)
    spec = solver.solve_interval(a, b)

    assert not spec.declined, spec.reason
    assert spec.count == len(truth), (
        f"{name}: found {spec.count} eigenvalues where truth has {len(truth)} in "
        f"[{a}, {b}] -- got {np.round(spec.unfolded, 6)} against {np.round(truth, 6)}")
    assert np.all(np.diff(spec.lam) > 0)
    assert np.allclose(spec.unfolded, truth, rtol=1e-8, atol=0)


@pytest.mark.parametrize("name,dom_fn,truth_fn,a,b", CASES)
def test_the_multiplicities_are_read_not_inferred(name, dom_fn, truth_fn, a, b):
    """The half of clause (1) the old array contract could not carry. Both domains have
    genuine degeneracies -- the disk's are geometric (sin/cos pairs), the triangle's
    arithmetic (`m^2+mn+n^2` collisions) -- so `mult` must exceed 1 somewhere, and match."""
    _, solver, truth, a, b = _case(name, dom_fn, truth_fn, a, b)
    spec = solver.solve_interval(a, b)

    want = np.unique(truth, return_counts=True)[1]
    assert spec.mult.max() > 1, f"{name}: no degeneracy detected, but truth has {want}"
    assert np.array_equal(spec.mult, want), (
        f"{name}: multiplicities {spec.mult} against truth {want}")


def test_a_tighter_request_costs_more_and_says_what_it_could_not_reach():
    """`search.md` section 2.6 and clause (2), which is an OR and has to be tested as one.

    Clause (2): each level within `prec`, **or** the solver reports that it was not. Asserting
    only the first half would demand the basis serve any request, which is not the contract, so
    the OR is asserted at every precision whichever branch holds. What must never happen is
    missing the request silently, and clause (3) is the assertion for that: the reported bar
    contains the error whether or not the request was met.

    An earlier version said this pencil "floors at about 2e-10 relative however tightly it is
    asked". It does not -- its tension at the true eigenvalues is 1.7e-15 -- and the apparent
    floor was the refiner declaring `converged` early. See `tests/test_refine_regression.py`.

    Section 2.6 is the cost half: a tighter request may cost more and must not cost less.
    """
    import warnings

    a, b = 8.0, 60.0
    truth = np.unique(disk_spectrum(a, b, 1.0))
    costs, lams = [], []
    for prec in (1e-6, 1e-8, 1e-10):
        s = solver_for(disk(1.0), b, label="disk", prec=prec)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")          # the shortfall is the SUBJECT here
            spec = s.solve_interval(a, b)
        assert spec.count == len(disk_spectrum(a, b, 1.0))
        err = np.abs(spec.lam - truth)

        within = err <= prec * truth
        assert np.all(within | spec.unmet), (
            f"prec={prec:g}: level(s) missed the request without being reported -- "
            f"err/prec = {np.round(err / (prec * truth), 2)}")
        assert np.all(err <= spec.bound), (
            f"prec={prec:g}: the reported bar does not contain the error")

        costs.append(s.search_interval(a, b).diagnostics.evaluations)
        lams.append(spec.lam)

    assert costs == sorted(costs), f"cost not monotone in prec: {costs}"
    assert np.allclose(lams[0], lams[-1], rtol=1e-6, atol=0), "the answers must agree"


def test_a_request_the_basis_cannot_serve_is_REPORTED_not_met_silently():
    """Clause (3) at the point it bites. A deliberately STARVED plane-wave basis on the disk --
    22 columns, `pad = 0.35`, at the test's own `prec` so its `rtol` is 1.5e-14 -- cannot serve
    `prec = 1e-12` at every level it locates, and the caller must be told rather than handed a
    confident wrong answer.

    Measured over `pad` at this `prec`, levels found of 6 and levels reported unmet:

        pad    columns   floor at worst level   found   unmet
        0.30      18           4.0e-03            1       1
        0.35      22           1.1e-05            2       1
        0.40      26           1.6e-08            3       0
        0.45      28           3.7e-10            5       0

    So starving a basis does not mainly produce reported shortfalls: it LOSES the levels it
    resolves worst, silently. That is a separate defect, pinned by
    `test_a_starved_basis_does_not_DROP_levels_silently` below.

    It used the reference-grade pencil and the claim that it "floors at about 2e-10". That was
    the refiner stopping early; with it fixed the same pencil meets `1e-12`, and this test has
    to starve the basis to keep testing what its name says.

    This is the case that caught `delta_res` reporting a bar three orders tighter than its own
    error: before that fix this test's `achieved` read True.
    """
    a, b = 8.0, 60.0
    truth = np.unique(disk_spectrum(a, b, 1.0))
    s = solver_for(disk(1.0), b, label="disk-starved", pad=0.35, prec=1e-12)

    with pytest.warns(PrecisionWarning, match="short of the requested"):
        spec = s.solve_interval(a, b)

    assert spec.unmet.any(), "a request beyond the basis must be reported as unmet"
    assert not spec.achieved
    err = np.array([np.min(np.abs(truth - x)) for x in spec.lam])
    assert np.all(err <= spec.bound), "and the bar must still hold"


def test_multiplicity_is_read_from_the_cache_and_spends_nothing():
    """`candidates.md` section 1: stage 3 spends no evaluations. Asserted rather than assumed,
    because a multiplicity read that quietly costs a GSVD per level changes the cost model the
    whole comparison is scored under."""
    s = solver_for(disk(1.0), 60.0, label="disk", prec=1e-8)
    res = s.search_interval(8.0, 60.0)
    assert res.diagnostics.multiplicity_evals == 0


def test_the_cost_is_reported_and_is_not_absurd():
    """Not a benchmark -- a guard. The gate measures cost properly; this only catches a wiring
    change that turns a 25-evaluation solve into a 2500-evaluation one."""
    s = solver_for(disk(1.0), 60.0, label="disk", prec=1e-8)
    res = s.search_interval(8.0, 60.0)
    per_eig = res.diagnostics.evaluations / max(res.diagnostics.n_brackets, 1)
    assert per_eig < 60, f"{per_eig:.0f} evaluations per bracket"


def test_solve_n_reaches_the_first_n_through_the_weyl_window():
    """`Eigensolver.solve` over the real search, not the `ClosedForm` double: the window, the
    widening and the truncation, with an actual MPS solve underneath."""
    s = solver_for(disk(1.0), 60.0, label="disk", prec=1e-8)
    got = s.solve(6)
    assert np.allclose(got, np.sort(disk_spectrum(0.0, 40.0, 1.0))[:6], rtol=1e-8, atol=0)


def test_solve_interval_returns_a_Spectrum_and_solve_returns_an_array():
    """The two contracts, side by side. `solve_interval` answers the rich one that carries
    multiplicity and the bar; `solve` answers `Eigensolver`'s array."""
    s = solver_for(disk(1.0), 60.0, label="disk", prec=1e-8)
    spec = s.solve_interval(8.0, 60.0)
    assert isinstance(spec, Spectrum)
    assert isinstance(s.solve(4), np.ndarray)


@pytest.mark.xfail(strict=True, reason=(
    "measured 2026-09-14: on a starved disk basis solve_interval returns 2 of 6 levels at "
    "pad=0.35 and 5 of 6 at pad=0.45, with no decline, no unmet level for the lost ones and no "
    "warning. Clause (1)'s completeness diagnostics do not fire. TODO.md records it."))
@pytest.mark.parametrize("pad", [0.35, 0.45])
def test_a_starved_basis_does_not_DROP_levels_silently(pad):
    """`search.md` section 2.4 clause (3): the completeness diagnostics fire whenever clause (1)
    fails. A basis too poor to locate a level must say so -- a short count with no warning is
    what `Eigensolver.solve(n)` turns into a non-consecutive spectrum by widening the window."""
    import warnings

    a, b = 8.0, 60.0
    truth = np.unique(disk_spectrum(a, b, 1.0))
    s = solver_for(disk(1.0), b, label=f"disk-starved-{pad}", pad=pad, prec=1e-12)
    with warnings.catch_warnings():
        # Any warning is not enough: a PrecisionWarning about a level that WAS found says
        # nothing about the ones that were not. No warning names a missing level today, so the
        # acceptable outcomes are the full count or a decline.
        warnings.simplefilter("ignore")
        try:
            spec = s.solve_interval(a, b)
        except Exception:
            return                                   # refusing loudly is an acceptable answer
    assert spec.count == truth.size or spec.declined, (
        f"returned {spec.count} of {truth.size} levels without declining")

