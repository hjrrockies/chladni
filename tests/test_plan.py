"""Phase 5: the frozen, shape-relative `Plan`.  (2026-10-12)

`docs/design/douse_requirements.md`, "Plans: plan once, realize per iterate". A `Plan` holds the
basis choices relative to a shape -- Fourier-Bessel orders per corner (by `dom.corners` index),
collocation ratios, the precision and wavenumber it was sized for -- so douse can plan once and
`realize` it on every nearby iterate with the same corner structure. Pickling reproduces it
exactly.

The sizing is lappy's per-corner rule with a PRECISION term (`notes/planner_recipes.md`): orders
up to `k r + t(prec) (k r)^(1/3) + 3`, `t` where `J_nu(k r)` has decayed to `prec` past its
turning point, times the ladder rung's multiplier.
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


def test_higher_rungs_are_larger(dom):
    sizes = [plan_for(dom, N, PREC, rung=r).n_cols for r in range(len(LADDER))]
    assert sizes == sorted(sizes) and sizes[-1] > 2 * sizes[0]


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
