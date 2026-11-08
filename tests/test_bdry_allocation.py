"""Basis-aware boundary collocation: every side gets the points the basis needs there.  (2026-11-07)

A Fourier-Bessel column at a corner satisfies the boundary condition EXACTLY on that corner's
two sides, so collocation points there carry no information about its coefficient. Only the
other sides constrain it. Splitting the boundary points by side length alone starves a short
side opposite a heavily loaded corner. On the isosceles triangle with apex 0.05 and 100 terms
per corner, the base gets 24 of 900 points for the 100 apex columns, and the tension is a flat,
spurious ~1e-16 floor over the whole window. lappy guarded this with `pts_per_seg`
(`lappy/geometry.py`).

Two claims are defended. A corner asks for `COLLOCATION_MULT` points per column on its far
sides, split by the ANGLE each side subtends from it, not by length. Its columns are
`sin(mu theta)`, orthogonal in theta, so a face-on side constrains them well-conditionedly and an
edge-on (radial) side barely does: at equal length and 2N points, singular-value ratios
1e-6..1e-17 edge-on against 0.85-0.99 face-on (`notes/collocation_allocation.md`). And the
default changes nothing where the length split already covers the demand, which is every shape
the planner was measured on ("collocation density was never binding", `plan.py`).
"""
import numpy as np
import pytest

from chladni.geometry import L_shape, Polygon, eq_tri, rect
from chladni.geometry._adaptive import allocate
from chladni.helmholtz import FourierBessel, Fundamental
from chladni.helmholtz.fourier_bessel import COLLOCATION_MULT
from chladni.mps import MPSEigensolver, bdry_counts
from chladni.plan import Plan, plan_for


def isosceles(omega):
    """Apex `omega` at the origin, unit legs: segments 0 -> 1 (a leg), 1 -> e^{i omega} (the
    base), e^{i omega} -> 0 (a leg)."""
    return Polygon([0.0, 1.0, np.exp(1j * omega)])


def at(dom, i, n):
    return FourierBessel.at_corner(dom.corners[i], n=n, domain=dom)


def lengths(dom):
    return np.array([s.length for b in dom.boundaries for s in b.segments])


# --- what one corner asks for -----------------------------------------------------------


def test_a_corner_asks_nothing_of_its_own_sides():
    dom = isosceles(0.3)
    d = at(dom, 0, 10).collocation_demand(dom)
    assert d.shape == (3,)
    assert d[0] == pytest.approx(0.0, abs=1e-9) and d[2] == pytest.approx(0.0, abs=1e-9)
    assert d[1] == pytest.approx(COLLOCATION_MULT * 10)


def test_the_demand_is_split_by_subtended_angle_not_length():
    """Corner at the origin; far sides AB (length 3, subtending atan 3) and BC (length 2.24,
    subtending pi/2 - atan 3). A length split would give 1.34:1; the angle split gives 3.9:1."""
    dom = Polygon([0.0, 1.0, 1.0 + 3j, 1j])
    d = at(dom, 0, 20).collocation_demand(dom)
    a_ab, a_bc = np.arctan(3.0), np.pi / 2 - np.arctan(3.0)
    assert d[0] == pytest.approx(0.0, abs=1e-9) and d[3] == pytest.approx(0.0, abs=1e-9)
    assert d[1] / d[2] == pytest.approx(a_ab / a_bc, rel=1e-6)
    assert d.sum() == pytest.approx(COLLOCATION_MULT * 20)


def test_a_square_corner_splits_its_demand_evenly():
    dom = rect(1.0, 1.0)
    d = at(dom, 0, 8).collocation_demand(dom)
    far = d[d > 1e-9]
    assert len(far) == 2 and far == pytest.approx([COLLOCATION_MULT * 4] * 2)


def test_a_basis_sums_its_families_and_sources_ask_nothing():
    dom = isosceles(0.3)
    src = Fundamental(np.array([2 + 2j, -1 - 1j]), radius=1.0)
    assert np.all(src.collocation_demand(dom) == 0.0)
    fb0, fb1 = at(dom, 0, 5), at(dom, 1, 7)
    b = fb0 + fb1 + src
    np.testing.assert_allclose(b.collocation_demand(dom),
                               fb0.collocation_demand(dom) + fb1.collocation_demand(dom))


# --- the per-side counts ----------------------------------------------------------------


@pytest.mark.parametrize("make", [L_shape, eq_tri, lambda: rect(1.0, 1.0)],
                         ids=["L", "equilateral", "square"])
def test_ordinary_shapes_keep_the_length_split(make):
    """Where the length share covers every side's demand, the counts are today's exactly."""
    dom = make()
    p = plan_for(dom, 10, 1e-10)
    basis, n_bdry, _ = p.realize(dom)
    np.testing.assert_array_equal(bdry_counts(dom, basis, n_bdry, k=p.k_max),
                                  allocate(n_bdry, lengths(dom)))


def test_a_short_side_opposite_a_loaded_corner_gets_its_demand():
    dom = isosceles(0.05)
    basis, n_bdry, _ = Plan(prec=1e-14, k_max=np.nan, corner_orders=(100,) * 3).realize(dom)
    share = allocate(n_bdry, lengths(dom))
    assert share[1] < 100                           # the starved base, as it was
    counts = bdry_counts(dom, basis, n_bdry)
    assert counts[1] >= COLLOCATION_MULT * 100
    assert counts[0] == share[0] and counts[2] == share[2]
    assert counts.sum() > n_bdry


def test_the_wavenumber_floor_resolves_every_side():
    dom = rect(4.0, 1.0)
    basis = at(dom, 0, 3)
    k = 40.0
    counts = bdry_counts(dom, basis, 8, k=k)
    assert np.all(counts >= np.ceil(k * lengths(dom) / np.pi))
    np.testing.assert_array_equal(bdry_counts(dom, basis, 8, k=np.nan),
                                  bdry_counts(dom, basis, 8))


def test_the_domain_quadrature_takes_counts_per_segment():
    dom = isosceles(0.3)
    q = dom.bdry_quad(n_per_seg=[7, 11, 5])
    assert np.bincount(q.seg_idx).tolist() == [7, 11, 5]
    with pytest.raises(ValueError, match="segment"):
        dom.bdry_quad(n_per_seg=[7, 11])


# --- the solver and the plan use it -----------------------------------------------------


def test_a_budgeted_solver_places_its_points_by_the_basis():
    dom = isosceles(0.05)
    basis = at(dom, 0, 60) + at(dom, 1, 4) + at(dom, 2, 4)
    s = MPSEigensolver(dom, basis=basis, n_bdry=3 * len(basis), n_int=len(basis))
    assert s.n_bdry == int(bdry_counts(dom, basis, 3 * len(basis)).sum())


def test_a_plan_freezes_its_counts_per_segment():
    dom = L_shape()
    p = plan_for(dom, 10, 1e-10)
    basis, n_bdry, _ = p.realize(dom)
    assert p.bdry_counts == tuple(int(c) for c in bdry_counts(dom, basis, n_bdry, k=p.k_max))
    moved = Polygon([0, 1.03j, -1 + 1j, -1.02 - 1j, 1 - 0.98j, 1.01], bc="dir")
    assert p.solver(moved).n_bdry == sum(p.bdry_counts)


def test_a_plan_without_frozen_counts_computes_them_on_the_shape():
    dom = isosceles(0.05)
    p = Plan(prec=1e-10, k_max=np.nan, corner_orders=(60, 4, 4))
    basis, n_bdry, _ = p.realize(dom)
    assert p.bdry_counts == ()
    assert p.solver(dom).n_bdry == int(bdry_counts(dom, basis, n_bdry).sum())


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:FourierBessel at the corner:chladni._core.PrecisionWarning")
def test_the_thin_isosceles_triangle_finds_a_real_well():
    """The case that exposed it: apex 0.2, 100 terms per corner, prec 1e-14. On the length
    split the base got 83 points, the tension sat on a flat ~1e-14 floor and the search
    declined; with the base at its demand it locates lam_1 with a Moler-Payne bound near 1e-6
    (`notes/collocation_allocation.md`). The basis is deliberately oversized -- the apex's
    100th column has order ~1570 and evaluates to ~1e-11, short of 1e-14 -- so its
    PrecisionWarning is expected here and silenced."""
    dom = Polygon([0.0, 1.0, np.exp(0.2j)], prec=1e-14)
    s = Plan(prec=1e-14, k_max=np.nan, corner_orders=(100,) * 3).solver(dom)
    t = s.tension(np.linspace(300.0, 600.0, 40))
    assert np.median(t) > 1e-6                      # no spurious floor
    res = s.search_interval(300.0, 600.0)
    lv = min(res.levels, key=lambda L: L.lam)
    assert lv.lam == pytest.approx(435.062, rel=1e-5)
    assert lv.mp_bound / lv.lam < 1e-4
