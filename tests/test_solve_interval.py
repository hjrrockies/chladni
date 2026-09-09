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
    only the first half would demand the basis serve any request, which is not the contract and
    is not true -- this disk pencil floors at about 2e-10 relative however tightly it is asked,
    so at `prec = 1e-10` one level is genuinely out of reach. What must never happen is
    reaching it silently, and clause (3) is the assertion for that: the reported bar contains
    the error whether or not the request was met.

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
    """Clause (3) at the point it bites. This pencil floors at about 2e-10 relative, so
    `prec = 1e-12` is a request it cannot serve -- and the whole apparatus exists so that the
    caller is told rather than handed a confident wrong answer.

    This is the case that caught `delta_res` reporting a bar three orders tighter than its own
    error: before that fix this test's `achieved` read True.
    """
    a, b = 8.0, 60.0
    truth = np.unique(disk_spectrum(a, b, 1.0))
    s = solver_for(disk(1.0), b, label="disk", prec=1e-12)

    with pytest.warns(PrecisionWarning, match="short of the requested"):
        spec = s.solve_interval(a, b)

    assert spec.unmet.any(), "a request beyond the basis must be reported as unmet"
    assert not spec.achieved
    assert np.all(np.abs(spec.lam - truth) <= spec.bound), "and the bar must still hold"


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
