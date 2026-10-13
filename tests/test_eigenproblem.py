"""Phase 5: the black-box path, `Eigenproblem(dom).solve(n, prec)`.  (2026-10-12)

`docs/design/douse_requirements.md`: a Dirichlet polygon in, the first `n` eigenvalues out --
sorted, levels whole at `n`, each with its bound and flags -- with the basis planned by
`chladni.plan`. The planner climbs `LADDER` until every level up to `n + HEADROOM` meets `prec`
and the count audit is ok. Otherwise it returns its best rung FLAGGED, or raises
`PrecisionShortfall` under `on_shortfall="raise"`. Tested against ground truth: the L-shape's
14-digit table and closed forms.
"""
import numpy as np
import pytest

from chladni.eigenproblem import Eigenproblem
from chladni.eigensolver import Spectrum
from chladni.geometry import L_shape, Polygon, eq_tri, rect
from chladni.mps import PrecisionShortfall
from chladni.reference import L_shape_eigs, eq_tri_eigs, rect_eigs

PREC = 1e-10

# Fourier-Bessel's per-column error bound is very pessimistic (`notes/hard_cases.md`, part 3).
pytestmark = pytest.mark.filterwarnings(
    "ignore:FourierBessel can deliver:chladni._core.PrecisionWarning")


def _check_against(spec, truth, n, prec):
    """Every returned value within its bound of the truth, and the first n right with
    multiplicity, the level at n whole."""
    e = spec.unfolded
    assert e.size >= n
    for x, m, b in zip(spec.lam, spec.mult, spec.bound):
        j = np.argmin(np.abs(truth - x))
        assert abs(truth[j] - x) <= b + 1e-14 * x, (x, truth[j], b)
        assert m == int(np.sum(np.abs(truth - truth[j]) <= 1e-12 * truth[j])), (x, m)


@pytest.mark.slow
def test_the_L_shape_to_1e10_against_its_table():
    spec = Eigenproblem(L_shape()).solve(10, prec=PREC)
    assert isinstance(spec, Spectrum) and spec.plan is not None
    _check_against(spec, L_shape_eigs(25), 10, PREC)
    assert spec.achieved and not spec.flags, spec.flags
    assert spec.count_audit is not None and spec.count_audit.verdict == "ok"
    assert len(spec.levels) == len(spec.lam)
    assert all(L.met for L in spec.levels)


@pytest.mark.slow
def test_a_level_straddling_n_comes_back_whole():
    """The square: 2, 5, 5, 8, 10, 10, ... x pi^2. `solve(5)` returns six values -- the double at
    10 pi^2 whole -- never a level split at n."""
    spec = Eigenproblem(rect(1.0, 1.0)).solve(5, prec=PREC)
    assert spec.count == 6 and spec.mult[-1] == 2
    _check_against(spec, np.asarray(rect_eigs(12, 1.0, 1.0), float), 5, PREC)


@pytest.mark.slow
def test_the_equilateral_triangle_counts_its_doubles():
    spec = Eigenproblem(eq_tri()).solve(8, prec=PREC)
    _check_against(spec, np.asarray(eq_tri_eigs(20), float), 8, PREC)


@pytest.mark.slow
def test_a_plan_is_reused_on_a_nearby_shape():
    """douse: plan once, solve every nearby iterate with it."""
    ep = Eigenproblem(L_shape())
    plan = ep.plan(8, prec=PREC)
    moved = Polygon([0, 1.002j, -1 + 1j, -1.001 - 1j, 1 - 0.999j, 1.0], bc="dir")
    spec = Eigenproblem(moved).solve(8, prec=PREC, plan=plan)
    assert spec.plan is plan and spec.count >= 8


@pytest.mark.slow
def test_a_tighter_request_than_the_plan_is_flagged_not_enlarged():
    """Changing prec is an explicit re-plan: the solve reports, it does not enrich silently."""
    ep = Eigenproblem(L_shape())
    plan = ep.plan(8, prec=1e-8)
    spec = ep.solve(8, prec=1e-12, plan=plan)
    assert spec.plan is plan
    assert any("plan" in f for f in spec.flags), spec.flags


@pytest.mark.robust          # three rungs, twice: minutes
def test_a_request_no_rung_accepts_is_flagged_or_raised():
    """1e-14 on the L-shape: the hybrid top rung meets it on the first six levels, but not on
    the HEADROOM above them that the count audit relies on, so no rung is accepted. Default: the
    best rung, FLAGGED -- the six values may be fine, and the flag says the audit could not be
    trusted to show they are complete. On request: `PrecisionShortfall`."""
    ep = Eigenproblem(L_shape())
    spec = ep.solve(6, prec=1e-14)
    assert any(f.startswith("planner:") for f in spec.flags), spec.flags
    with pytest.raises(PrecisionShortfall):
        ep.solve(6, prec=1e-14, on_shortfall="raise")


def test_the_planner_refuses_what_it_does_not_cover():
    from chladni.geometry import disk
    with pytest.raises(NotImplementedError):
        Eigenproblem(disk(1.0)).solve(3)


def test_the_old_spectrum_still_builds():
    s = Spectrum.from_unfolded([1.0, 2.0, 2.0])
    assert s.count == 3 and s.levels == () and s.flags == () and s.count_audit is None


@pytest.mark.slow
def test_a_long_thin_rectangle_reaches_1e10():
    """4:1. No corner is singular, so the plan has to reach its plane-wave rung."""
    spec = Eigenproblem(rect(1.6, 0.4)).solve(3, prec=PREC)
    assert not spec.flags, spec.flags
    _check_against(spec, np.asarray(rect_eigs(12, 1.6, 0.4), float), 3, PREC)
