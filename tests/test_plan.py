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


def test_the_lightning_rung_adds_corner_clustered_sources(dom):
    """The last rung: Fourier-Bessel at the singular corners plus lightning poles there, as lappy
    set them up (sigma 1, reaching half the diameter, order 2, a 50/50 column split). It solved a
    thin triangle the offset layer could not (`notes/hybrid_bases.py`, round 3)."""
    rung = next(i for i, r in enumerate(LADDER) if r.lightning)
    p = plan_for(dom, N, PREC, rung=rung)
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
    rung = next(i for i, r in enumerate(LADDER) if r.lightning)
    p = plan_for(rect(1.0, 0.7), N, PREC, rung=rung)
    assert p.n_cols == 0 and not _usable(p)
    assert _usable(plan_for(rect(1.0, 0.7), N, PREC, rung=0))


def test_the_plane_wave_rung_covers_shapes_with_no_singular_corner():
    """A rectangle's corners are all reflection-closed, so the singular-corner rungs place no
    Fourier-Bessel term, and corner terms alone do not converge on a long thin rectangle. The
    last rung adds plane waves (chladni's test roster's basis for such shapes, 2026-10-13: a
    4:1 rectangle reached no rung at 1e-10 before it)."""
    p = plan_for(rect(1.6, 0.4), N, PREC, rung=len(LADDER) - 1)
    assert p.n_plane_waves > 0 and all(n == 0 for n in p.corner_orders)
    b, _, _ = p.realize(rect(1.6, 0.4))
    assert len(b) == p.n_cols == 2 * p.n_plane_waves


# --- short edges get a local source patch (2026-10-13) -----------------------------------------
# douse's N = 6, k = 2 ratio run (seed 1) drives one hexagon edge toward zero: 0.042 of a 1.44
# diameter. The offset layer sits 0.05 D = 0.072 out, farther than the edge is long, and every
# rung but 2 missed 1e-10 there; rung 2 met it at margin 2.0 exactly, so the run stalled on
# refusals. Moving the whole layer closer made other cases climb a rung; a small dense patch of
# sources around each short edge (offset e/3, spacing e/2, over 2e either side of its midpoint)
# fixed the stall shape and changed no other douse benchmark case for the worse
# (`notes/short_edge_sources.py`).

STALL = Polygon([0.0417, 1.1091 + 0.5169j, 0.9097 + 1.1191j, 0.2018 + 1.2364j,
                 -0.2318 + 0.6575j, 0.0], bc="dir")
#: Its first four eigenvalues: two independent bases (rung 2 with and without the patch) agree
#: to 3e-13.
STALL_LAMS = np.array([18.8459218531, 40.6304854566, 53.2319678511, 74.2587486871])


def _short_edge(dom):
    segs = [s for b in dom.boundaries for s in b.segments]
    return int(np.argmin([abs(s.p1 - s.p0) for s in segs]))


def test_ordinary_shapes_get_no_patch(dom):
    from chladni.geometry import reg_ngon
    for d in (dom, reg_ngon(6), rect(1.0, 0.7)):
        for rung in range(len(LADDER)):
            assert plan_for(d, N, PREC, rung=rung).edge_patches == ()


def test_a_short_edge_gets_a_patch_on_the_offset_rungs_only():
    i = _short_edge(STALL)
    for rung in range(len(LADDER)):
        p = plan_for(STALL, 4, PREC, rung=rung)
        if LADDER[rung].source_offset > 0:
            assert p.edge_patches == ((i, 9),), p.edge_patches
        else:
            assert p.edge_patches == ()


def test_a_patch_realizes_at_its_planned_size_here_and_nearby():
    p = plan_for(STALL, 4, PREC, rung=2)
    b, _, _ = p.realize(STALL)
    assert len(b) == p.n_cols == sum(p.corner_orders) + p.n_sources + 9
    v = np.array([c.z for c in STALL.corners])
    moved = Polygon(v + 2e-3 * np.exp(1j * np.arange(len(v))), bc="dir")
    assert len(p.realize(moved)[0]) == p.n_cols


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:FourierBessel can deliver:chladni._core.PrecisionWarning")
def test_the_stalled_hexagon_now_plans_cleanly():
    from chladni.eigenproblem import Eigenproblem
    spec = Eigenproblem(STALL).solve(4, prec=PREC)
    assert not spec.flags, spec.flags
    assert all(L.margin >= 4 for L in spec.levels), [L.margin for L in spec.levels]
    np.testing.assert_allclose(spec.unfolded[:4], STALL_LAMS, rtol=PREC)


def test_the_plan_defaults_to_the_domains_own_prec():
    """`Domain.prec` is where the conventions say precision is set, and the black-box path used
    to ignore it: `plan_for(rect(prec=1e-12), 3).prec` read 1e-8 (2026-10-14)."""
    from chladni.eigenproblem import Eigenproblem
    d = rect(1.0, 1.0, prec=1e-12)
    assert plan_for(d, 3).prec == 1e-12
    assert plan_for(d, 3, prec=1e-8).prec == 1e-8
    assert Eigenproblem(d)._prec(None) == 1e-12


@pytest.mark.filterwarnings("ignore:geometry supports about:chladni._core.PrecisionWarning")
def test_a_shape_the_geometry_cannot_support_is_declined_before_any_solve():
    """A cheap "no": an edge of 1e-13 on a unit square puts `achievable_prec` far above 1e-10,
    and a notch a thousandth of the domain wide puts a corner on a near self-intersection.
    Both decline in milliseconds with a `declined:` flag; neither costs a solve."""
    import time
    from chladni.eigenproblem import Eigenproblem
    from chladni.plan import shape_refusal
    sliver = Polygon([0, 1, 1 + 1j, 1e-13 + 1j, 1j], bc="dir")
    assert "precision" in shape_refusal(sliver, 1e-10)
    notch = Polygon([0, 2, 2 + 2j, 1.0005 + 2j, 1.0005 + 0.05j, 0.9995 + 0.05j, 0.9995 + 2j, 2j],
                    bc="dir")
    assert "self-intersection" in shape_refusal(notch, 1e-10)
    assert shape_refusal(rect(1.2, 0.8), 1e-10) is None
    assert shape_refusal(reg_ngon_with_short_edge(), 1e-10) is None
    t0 = time.perf_counter()
    rec = Eigenproblem(notch).solve(3, prec=1e-10)
    assert rec.declined and rec.flags[0].startswith("declined:") and rec.count == 0
    assert time.perf_counter() - t0 < 1.0
    plan, rec2 = Eigenproblem(sliver).plan_and_solve(3, prec=1e-10)
    assert rec2.declined and plan.rung == 0


def reg_ngon_with_short_edge():
    """A hexagon with one edge at douse's floor, perimeter/(10 N): short, not wild."""
    v = 4.0 / 6 / (2 * np.sin(np.pi / 6)) * np.exp(2j * np.pi * np.arange(6) / 6)
    v[1] = v[0] + (v[1] - v[0]) * (4.0 / 60) / abs(v[1] - v[0])
    return Polygon(v, bc="dir")
