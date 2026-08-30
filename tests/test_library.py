"""The domain library: every shipped shape must match its closed form.

These double as smoke tests for the whole layer -- a library domain exercises
construction, validation, orientation, corners and metrics in one call.
"""
import numpy as np
import pytest

from chladni.geometry import (
    GWW1,
    GWW2,
    L_shape,
    disk,
    eq_tri,
    plus_shape,
    polyomino,
    rect,
    reg_ngon,
)

from conftest import EXACT, PREC

# --- polygons with closed forms ---------------------------------------------------------


@pytest.mark.parametrize("L,H", [(1, 1), (2, 1), (5, 0.25), (1, 3)])
def test_rect(L, H):
    r = rect(L, H)
    assert r.area == pytest.approx(L * H, abs=EXACT)
    assert r.perimeter == pytest.approx(2 * (L + H), abs=EXACT)
    assert r.diameter == pytest.approx(np.hypot(L, H), abs=PREC)
    assert r.inradius == pytest.approx(min(L, H) / 2, abs=PREC)
    assert len(r.corners) == 4
    assert np.allclose([c.theta for c in r.corners], np.pi / 2, atol=EXACT)


def test_l_shape():
    """A 2x2 square with a 1x1 corner bite: area 3, and the bite leaves perimeter at 8."""
    d = L_shape()
    assert d.area == pytest.approx(3.0, abs=EXACT)
    assert d.perimeter == pytest.approx(8.0, abs=EXACT)
    reentrant = [c for c in d.corners if c.is_reentrant]
    assert len(reentrant) == 1
    assert reentrant[0].theta == pytest.approx(3 * np.pi / 2, abs=EXACT)


def test_eq_tri():
    for l in (1.0, 2.5):
        t = eq_tri(l)
        assert t.area == pytest.approx(np.sqrt(3) / 4 * l**2, abs=PREC)
        assert t.perimeter == pytest.approx(3 * l, abs=PREC)
        assert np.allclose([c.theta for c in t.corners], np.pi / 3, atol=PREC)


@pytest.mark.parametrize("n", [3, 4, 5, 7, 12])
def test_reg_ngon(n):
    p = reg_ngon(n)
    assert p.area == pytest.approx(n / 2 * np.sin(2 * np.pi / n), abs=EXACT)
    assert p.perimeter == pytest.approx(2 * n * np.sin(np.pi / n), abs=EXACT)
    assert len(p.corners) == n


# --- curved -----------------------------------------------------------------------------


@pytest.mark.parametrize("r", [0.5, 1.0, 3.0])
def test_disk(r):
    d = disk(r)
    assert d.area == pytest.approx(np.pi * r**2, rel=PREC)
    assert d.perimeter == pytest.approx(2 * np.pi * r, rel=PREC)
    assert d.inradius == pytest.approx(r, rel=PREC)
    assert len(d.corners) == 0


def test_disk_has_no_corners_at_the_seam():
    """The parameterization closes at t=0; that seam is smooth and must not register."""
    assert disk().corners == ()


# --- the isospectral pair ---------------------------------------------------------------


def test_gww_pair_shares_its_weyl_invariants():
    """GWW1 and GWW2 are isospectral, so area and perimeter must agree exactly -- these
    are the first two terms of the Weyl asymptotic. A construction slip in either shape
    shows up here before it ever reaches the solver."""
    a, b = GWW1(), GWW2()
    assert a.area == pytest.approx(14.0, abs=EXACT)
    assert b.area == pytest.approx(14.0, abs=EXACT)
    assert a.perimeter == pytest.approx(12 + 6 * np.sqrt(2), abs=PREC)
    assert b.perimeter == pytest.approx(a.perimeter, abs=PREC)


def test_gww_pair_shares_its_corner_angles():
    """The two shapes are built from the same seven triangles, so the angle multiset
    matches even though the shapes differ."""
    a = sorted(np.round(c.theta / np.pi, 10) for c in GWW1().corners)
    b = sorted(np.round(c.theta / np.pi, 10) for c in GWW2().corners)
    assert a == b == [0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.5, 1.5]


def test_gww_domains_are_not_congruent():
    """Same invariants, different shapes: guard against one being a copy of the other."""
    assert not np.allclose(
        np.sort_complex(GWW1().outer.sample(200)), np.sort_complex(GWW2().outer.sample(200))
    )


# --- polyominoes ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cells",
    [
        [(0, 0)],
        [(0, 0), (1, 0)],
        [(0, 0), (1, 0), (0, 1)],
        [(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)],
    ],
)
def test_polyomino_area_is_the_cell_count(cells):
    assert polyomino(cells).area == pytest.approx(len(cells), abs=EXACT)


def test_polyomino_merges_collinear_edges():
    """A straight run of three cells must arrive as ONE segment, not three joined by
    smooth pi junctions."""
    strip = polyomino([(0, 0), (1, 0), (2, 0)])
    assert len(strip.outer.segments) == 4
    assert len(strip.corners) == 4


def test_plus_shape_has_four_reentrant_corners():
    p = plus_shape()
    assert p.area == pytest.approx(5.0, abs=EXACT)
    assert len([c for c in p.corners if c.is_reentrant]) == 4
    assert len(p.corners) == 12


def test_diagonally_joined_cells_rejected():
    with pytest.raises(ValueError):
        polyomino([(0, 0), (1, 1)])


def test_polyomino_with_an_enclosed_hole_rejected():
    ring = [(i, j) for i in range(3) for j in range(3) if (i, j) != (1, 1)]
    with pytest.raises(ValueError):
        polyomino(ring)


# --- shared contract across the whole library -------------------------------------------


@pytest.mark.parametrize(
    "factory",
    [rect, L_shape, disk, eq_tri, GWW1, GWW2, plus_shape, lambda: reg_ngon(6)],
    ids=["rect", "L_shape", "disk", "eq_tri", "GWW1", "GWW2", "plus_shape", "hexagon"],
)
def test_every_library_domain_is_well_formed(factory):
    d = factory() if factory is not rect else rect(2, 1)
    assert d.is_valid
    assert d.outer.orientation == 1
    assert d.area > 0
    assert d.is_simply_connected
    q = d.bdry_quad(600)
    assert q.weight_family == "arclength"
    assert q.wts.sum() == pytest.approx(d.perimeter, rel=1e-6)
    assert (q.radial(0j) * q.wts).sum() == pytest.approx(2 * d.area, rel=1e-6)


@pytest.mark.parametrize("bc", ["dir", "neu", 0.25])
def test_library_accepts_a_boundary_condition(bc):
    d = L_shape(bc=bc)
    assert all(s.bc == (0.0 if bc == "dir" else 1.0 if bc == "neu" else bc) for s in d.outer.segments)
