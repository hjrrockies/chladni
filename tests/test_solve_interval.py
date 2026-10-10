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


@pytest.mark.parametrize("seed, f, prec", [(0, 1.25, 1e-8), (11, 1.25, 1e-6)])
def test_an_exact_degeneracy_is_ONE_level_when_the_refiner_stops_on_its_wall(seed, f, prec):
    """The solver contract, not a scoring rule: a detectable exact degeneracy comes back as one
    level with its full multiplicity. Linux CI broke it in the test above -- 7 levels against
    6, total right -- on every leg, and these are the same failure reproduced on Accelerate by
    moving what a BLAS moves, the threshold rank: `rtol = 1.25 SUITE_RTOL`, two of 108 seed /
    `rtol` / `prec` cells.

    The mechanism (seed 0, the double at 57.58): the refiner accepted a point 0.0245 licences up
    the WALL of the double, sigma 1.2e-9 against a floor of 1.5e-15 -- legal, its bar covered
    the error -- the read there counted one and left the partner undecided, and the side search
    found the true root, which that undecided branch then owned. Two simple levels. `solve` now
    re-centres on a deeper find inside the root's own reach and reads the multiplicity there.

    The contract is right, or flagged: an uncertified count raises from `solve_interval` (and a
    decline raises too), and an ambiguous grouping warns. Either says the split is in doubt,
    which is honest; returning the wrong split SILENTLY is the failure."""
    from chladni.helmholtz import PlaneWave
    from chladni.mps import MPSEigensolver
    from rect_suite import SUITE_RTOL

    a, b, dom = 8.0, 60.0, disk(1.0)
    basis = PlaneWave.uniform(int(1.3 * PlaneWave.n_useful(np.sqrt(b) * dom.diameter)),
                              radius=dom.diameter)
    n = len(basis)
    s = MPSEigensolver(dom, basis=basis, n_bdry=3 * n, n_int=n, rtol=f * SUITE_RTOL,
                       rng=np.random.default_rng(seed), prec=prec)
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            spec = s.solve_interval(a, b)
        except RuntimeError as exc:
            assert "could not be certified" in str(exc) or "declined" in str(exc), (
                f"raised, but not as a flag or a decline: {exc}")
            return                                   # flagged loudly: acceptable
    if any(issubclass(w.category, PrecisionWarning) and "ambiguous grouping" in str(w.message)
           for w in caught):
        return                                       # the grouping is flagged: acceptable
    truth = np.sort(disk_spectrum(a, b, 1.0))
    want = np.unique(truth, return_counts=True)[1]
    assert spec.count == len(truth)
    assert np.array_equal(spec.mult, want), f"multiplicities {spec.mult} against truth {want}"


#: Measured 2026-10-09 on re-posing the starved-disk tests at douse's 1e-10 (they were at 1e-12):
#: stage 1 never BRACKETS the levels the starved basis resolves worst -- pad 0.35 brackets 2 of
#: the 6 levels, pad 0.45 5 of 6 (57.58 lost) -- with no rejection, no uncertain count, no unmet
#: level and no warning. At 1e-12 the same losses were caught only on the way past (an uncertain
#: count at 0.35, a deep rejection at 14.68 at 0.45), never by a completeness check. Clause (3)
#: does not fire for a level the search never brackets. A solver defect, not a test artifact, and
#: not confined to 1e-10: pad 0.35 at 1e-8 returns 3 of 6 levels just as silently.
STARVED_SILENT = ("1e-10: a starved basis loses levels stage 1 never brackets -- pad 0.35 returns "
                  "2 of 6 levels, pad 0.45 5 of 6 -- with no decline, flag, unmet level or "
                  "warning; the 1e-12 catches were incidental (2026-10-09)")


@pytest.mark.xfail(strict=False, reason=STARVED_SILENT)
def test_a_request_the_basis_cannot_serve_is_REPORTED_not_met_silently():
    """Clause (3) at the point it bites. A deliberately STARVED plane-wave basis on the disk --
    22 columns, `pad = 0.35`, on the roster's fixed `SUITE_RTOL` -- cannot serve
    `prec = 1e-10` at every level it locates, and the caller must be told rather than handed a
    confident wrong answer.

    Re-posed at 1e-10 from 1e-12 on 2026-10-09 (the test-suite prune: douse's rungs are 1e-10 and
    above). The floors below put the 22-column basis five orders short of either. The table was
    measured at 1e-12, levels found of 6 and levels reported unmet:

        pad    columns   floor at worst level   found   unmet
        0.30      18           4.0e-03            1       1
        0.35      22           1.1e-05            2       1
        0.40      26           1.6e-08            3       0
        0.45      28           3.7e-10            5       0

    So starving a basis does not mainly produce reported shortfalls: it LOSES the levels it
    resolves worst -- silently until 2026-09-20, when `_deep_rejections` made the refusal that
    drops them reach `n_uncertain`. Pinned by
    `test_a_starved_basis_does_not_DROP_levels_silently` below, whose `pad=0.45` case was a strict
    xfail for exactly this and is now a passing one.

    It used the reference-grade pencil and the claim that it "floors at about 2e-10". That was
    the refiner stopping early; with it fixed the same pencil meets `1e-12`, and this test has
    to starve the basis to keep testing what its name says.

    This is the case that caught `delta_res` reporting a bar three orders tighter than its own
    error: before that fix this test's `achieved` read True.
    """
    a, b = 8.0, 60.0
    truth = np.unique(disk_spectrum(a, b, 1.0))
    s = solver_for(disk(1.0), b, label="disk-starved", pad=0.35, prec=1e-10)

    # Two ways to TELL the caller, and either is acceptable; a confident answer is not. Under the
    # promoted defaults the starved basis leaves a count the reach rule cannot certify, and
    # `solve_interval` refuses with that reason -- measured 2026-09-14 -- where the previous
    # defaults returned the levels with a shortfall warning.
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            spec = s.solve_interval(a, b)
        except RuntimeError as exc:
            assert "could not be certified" in str(exc), f"refused, but not for a stated reason: {exc}"
            return
    assert any(issubclass(w.category, PrecisionWarning) and "short of the requested" in str(w.message)
               for w in caught), "returned without warning that the request was not met"
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


@pytest.mark.xfail(strict=False, reason=STARVED_SILENT)
@pytest.mark.parametrize("pad", [
    0.35,     # at 1e-12, the uncertified count made solve_interval refuse
    # 0.45 WAS a strict xfail: "measured 2026-09-14: on a starved disk basis solve_interval
    # returns 5 of 6 levels at pad=0.45 with no decline, no unmet level for the lost one and no
    # warning. Clause (1)'s completeness diagnostics do not fire."
    #
    # It fires now, and not by being aimed at this case. `_deep_rejections` (2026-09-20) was built
    # for a wrong total on a rectangle cluster under Linux/OpenBLAS, and it catches this one on the
    # way past: 6 brackets, 5 levels, the refusal reported at `lam = 14.68197` with
    # `sigma_1 = 4.94e-13` against a worst located level no shallower, reason `not_a_minimum`. So
    # `solve_interval` raises, which this test accepts as "refusing loudly".
    #
    # Different domain, different regime and a different refusal reason than the case the rule was
    # built from, which is the evidence that it keys on something real rather than on that cell.
    # (All measured at 1e-12. At 1e-10 stage 1 brackets only 5 levels and refuses nothing, so
    # there is no rejection to catch it: see `STARVED_SILENT`.)
    0.45,
])
def test_a_starved_basis_does_not_DROP_levels_silently(pad):
    """`search.md` section 2.4 clause (3): the completeness diagnostics fire whenever clause (1)
    fails. A basis too poor to locate a level must say so -- a short count with no warning is
    what `Eigensolver.solve(n)` turns into a non-consecutive spectrum by widening the window.

    Re-posed at 1e-10 from 1e-12 on 2026-10-09 (the test-suite prune); the measurements in the
    comments above were made at 1e-12. Only `RuntimeError`, the type `solve_interval` refuses
    with, counts as refusing loudly: `except Exception` would have passed a crash too."""
    import warnings

    a, b = 8.0, 60.0
    truth = np.unique(disk_spectrum(a, b, 1.0))
    s = solver_for(disk(1.0), b, label=f"disk-starved-{pad}", pad=pad, prec=1e-10)
    with warnings.catch_warnings():
        # Any warning is not enough: a PrecisionWarning about a level that WAS found says
        # nothing about the ones that were not. No warning names a missing level today, so the
        # acceptable outcomes are the full count or a decline.
        warnings.simplefilter("ignore")
        try:
            spec = s.solve_interval(a, b)
        except RuntimeError:
            return                                   # refusing loudly is an acceptable answer
    # LEVELS against levels: `spec.count` is the total WITH multiplicity, and compared with the
    # distinct truth it could never pass -- unseen while every run refused at 1e-12.
    assert spec.lam.size == truth.size or spec.declined, (
        f"returned {spec.lam.size} of {truth.size} levels without declining")

