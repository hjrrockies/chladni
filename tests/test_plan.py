"""Phase 5: the frozen, shape-relative `Plan`.  (2026-10-12)

`docs/design/douse_requirements.md`, "Plans: plan once, realize per iterate". A `Plan` holds the
basis choices relative to a shape -- Fourier-Bessel orders per corner (by `dom.corners` index),
collocation ratios, the precision and wavenumber it was sized for -- so douse can plan once and
`realize` it on every nearby iterate with the same corner structure. Pickling reproduces it
exactly.

The sizing is lappy's per-corner rule with a PRECISION term (`notes/planner_recipes.md`): orders
up to `k r + t(prec) (k r)^(1/3) + 3`, `t` where `J_nu(k r)` has decayed to `prec` past its
turning point, times the ladder rung's multiplier. Rungs above the first are HYBRID: Fourier-
Bessel at the singular corners plus a boundary-offset source layer (`notes/hybrid_bases.py`).
"""
import dataclasses
import pickle

import numpy as np
import pytest

from chladni.geometry import L_shape, Polygon, disk, rect
from chladni.plan import LADDER, Plan, plan_for

N, PREC = 10, 1e-10


@pytest.fixture(scope="module")
def dom():
    return L_shape()


def test_a_plan_is_frozen_and_sized_per_corner(dom):
    p = plan_for(dom, N, PREC)
    assert p.rung == 0 and p.prec == PREC and p.k_max > 0
    assert len(p.corner_orders) == len(dom.corners)
    assert all(isinstance(n, int) and n >= 3 for n in p.corner_orders)
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.rung = 1


def test_rung_2_enlarges_the_hybrid(dom):
    """Rungs are different RECIPES, not one growing basis: the hybrid rung 1 can have fewer
    columns than rung 0, since it skips reflection-closed corners. Rung 2 is the same hybrid, grown."""
    p1, p2 = plan_for(dom, N, PREC, rung=1), plan_for(dom, N, PREC, rung=2)
    assert sum(p2.corner_orders) > sum(p1.corner_orders) and p2.n_sources == p1.n_sources


def test_the_sizing_carries_the_precision(dom):
    """A looser request is planned smaller: the precision term, not a fixed constant."""
    assert plan_for(dom, N, 1e-8).n_cols < plan_for(dom, N, 1e-10).n_cols


# Fourier-Bessel's per-column error bound reads ~1e-9 here; measured against 30-digit mpmath the
# columns are good to ~1e-14 (`notes/hard_cases.md`, part 3), so the warning is not substantive.
@pytest.mark.filterwarnings("ignore:FourierBessel can deliver:chladni._core.PrecisionWarning")
def test_a_pickled_plan_reproduces_exactly(dom):
    p = plan_for(dom, N, PREC, rung=1)
    q = pickle.loads(pickle.dumps(p))
    assert q == p
    (b1, nb1, ni1), (b2, nb2, ni2) = p.realize(dom), q.realize(dom)
    assert (len(b1), nb1, ni1) == (len(b2), nb2, ni2)
    quad = dom.bdry_quad(200)
    np.testing.assert_array_equal(b1.at(quad).values(7.0), b2.at(quad).values(7.0))


def test_realize_sizes_the_collocation_from_the_plan(dom):
    p = plan_for(dom, N, PREC)
    b, n_bdry, n_int = p.realize(dom)
    assert len(b) == p.n_cols
    assert n_bdry == int(np.ceil(p.bdry_ratio * len(b))) and n_int == int(np.ceil(p.int_ratio * len(b)))


def test_realize_on_a_nearby_shape_keeps_the_size(dom):
    """douse's case: the same plan on the next iterate, same corner structure."""
    p = plan_for(dom, N, PREC)
    moved = Polygon([0, 1.03j, -1 + 1j, -1.02 - 1j, 1 - 0.98j, 1.01], bc="dir")
    b, _, _ = p.realize(moved)
    assert len(b) == p.n_cols


def test_realize_refuses_a_different_corner_structure(dom):
    p = plan_for(dom, N, PREC)
    with pytest.raises(ValueError, match="corner"):
        p.realize(rect(1.0, 1.0))


@pytest.mark.parametrize("make", [lambda: disk(1.0), lambda: rect(1.0, 0.7, bc="neu")])
def test_the_planner_covers_dirichlet_polygons_only(make):
    with pytest.raises(NotImplementedError):
        plan_for(make(), N, PREC)


def test_a_plan_builds_its_solver(dom):
    p = plan_for(dom, N, PREC)
    s = p.solver(dom)
    assert s.prec == PREC and s.n_cols == p.n_cols


def test_a_hybrid_rung_skips_regular_corners_and_adds_sources(dom):
    """Rung 1: Fourier-Bessel only where the corner is singular (the L-shape's re-entrant one; its
    right angles are reflection-closed), plus a fixed number of offset sources."""
    p = plan_for(dom, N, PREC, rung=1)
    singular = [not c.is_reflection_closed for c in dom.corners]
    assert all((n > 0) == sing for n, sing in zip(p.corner_orders, singular))
    assert p.n_sources > 0 and p.source_offset > 0
    b, _, _ = p.realize(dom)
    assert len(b) == p.n_cols == sum(p.corner_orders) + p.n_sources


def test_a_hybrid_plan_realizes_to_the_same_size_on_a_nearby_shape(dom):
    p = plan_for(dom, N, PREC, rung=1)
    moved = Polygon([0, 1.03j, -1 + 1j, -1.02 - 1j, 1 - 0.98j, 1.01], bc="dir")
    b, _, _ = p.realize(moved)
    assert len(b) == p.n_cols


def test_the_top_rung_adds_corner_clustered_sources(dom):
    """The last rung: Fourier-Bessel at the singular corners plus lightning poles there, as lappy
    set them up (sigma 1, reaching half the diameter, order 2, a 50/50 column split). It solved a
    thin triangle the offset layer could not (`notes/hybrid_bases.py`, round 3)."""
    p = plan_for(dom, N, PREC, rung=len(LADDER) - 1)
    assert p.lightning_n > 0 and p.n_sources == 0
    b, _, _ = p.realize(dom)
    assert len(b) == p.n_cols
    moved = Polygon([0, 1.03j, -1 + 1j, -1.02 - 1j, 1 - 0.98j, 1.01], bc="dir")
    assert len(p.realize(moved)[0]) == p.n_cols


def test_a_rung_with_nothing_to_place_is_empty_and_skipped():
    """Every corner of a rectangle is reflection-closed, so the singular-corner lightning rung
    places nothing. The plan is empty, and the planner must skip it, not crash (T2.8-kink and
    T1.5-r2 on the benchmark, 2026-10-12)."""
    from chladni.eigenproblem import _usable
    p = plan_for(rect(1.0, 0.7), N, PREC, rung=len(LADDER) - 1)
    assert p.n_cols == 0 and not _usable(p)
    assert _usable(plan_for(rect(1.0, 0.7), N, PREC, rung=0))
