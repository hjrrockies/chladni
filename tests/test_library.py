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
    assert np.allclose([c.omega for c in r.corners], np.pi / 2, atol=EXACT)


def test_l_shape():
    """A 2x2 square with a 1x1 corner bite: area 3, and the bite leaves perimeter at 8."""
    d = L_shape()
    assert d.area == pytest.approx(3.0, abs=EXACT)
    assert d.perimeter == pytest.approx(8.0, abs=EXACT)
    reentrant = [c for c in d.corners if c.is_reentrant]
    assert len(reentrant) == 1
    assert reentrant[0].omega == pytest.approx(3 * np.pi / 2, abs=EXACT)


def test_eq_tri():
    for l in (1.0, 2.5):
        t = eq_tri(l)
        assert t.area == pytest.approx(np.sqrt(3) / 4 * l**2, abs=PREC)
        assert t.perimeter == pytest.approx(3 * l, abs=PREC)
        assert np.allclose([c.omega for c in t.corners], np.pi / 3, atol=PREC)


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
    a = sorted(np.round(c.omega / np.pi, 10) for c in GWW1().corners)
    b = sorted(np.round(c.omega / np.pi, 10) for c in GWW2().corners)
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


# =========================================================================================
# Factories ported from lappy. Placement is load-bearing: `chladni.reference`'s closed-form
# eigenfunctions each assume the placement of the factory they name, and their exact L2 norms
# are wrong if the two disagree. So the placement assertions below are not decoration -- they
# are the other half of `tests/test_eigenfunctions.py`.
# =========================================================================================

import numpy as _np  # noqa: E402  (the imports below extend the module header above)

from chladni.geometry import (  # noqa: E402
    H_shape, chevron, cut_square, disk_sector, ellipse, iso_right_tri, iso_tri, mushroom,
    parallelogram, right_trapezoid, stadium,
)

PORTED = [
    ("iso_tri", lambda: iso_tri(2.0)),
    ("right_trapezoid", lambda: right_trapezoid(1.0, 2.0)),
    ("parallelogram", lambda: parallelogram(1.0, 1.0, _np.pi / 3)),
    ("chevron", lambda: chevron(1.0, 2.0)),
    ("H_shape", H_shape),
    ("ellipse", lambda: ellipse(2.0, 1.0)),
    ("disk_sector", lambda: disk_sector(1.0, _np.pi / 3)),
    ("cut_square", lambda: cut_square(0.5)),
    ("mushroom", lambda: mushroom(1.0, 1.0, 1.5)),
    ("stadium", lambda: stadium(1.0, 1.0)),
]


@pytest.mark.parametrize("name,make", PORTED, ids=[n for n, _ in PORTED])
def test_every_ported_factory_builds_and_validates(name, make):
    d = make()
    assert d.is_valid and d.area > 0 and d.perimeter > 0


# --- areas against closed forms ------------------------------------------------------------


def test_the_new_polygons_have_the_areas_the_formulas_say():
    assert iso_tri(2.0).area == pytest.approx(2.0, rel=1e-12)          # base 2, height h
    assert right_trapezoid(1.0, 2.0).area == pytest.approx(1.5, rel=1e-12)   # (h1+h2)/2
    assert parallelogram(1.0, 1.0, np.pi / 3).area == pytest.approx(1.0, rel=1e-12)  # b*h
    assert chevron(1.0, 2.0).area == pytest.approx(1.0, rel=1e-12)     # h2 - h1
    assert H_shape().area == pytest.approx(7.0, rel=1e-12)


def test_the_curved_domains_have_the_areas_the_formulas_say():
    assert ellipse(2.0, 1.0).area == pytest.approx(2 * np.pi, rel=1e-8)          # pi*a*b
    assert disk_sector(1.0, np.pi / 3).area == pytest.approx(np.pi / 6, rel=1e-8)  # r^2*th/2
    # the cut removes the QUARTER DISK of radius r about (1,1), not the corner sliver
    assert cut_square(0.5).area == pytest.approx(1 - np.pi * 0.25 / 4, rel=1e-8)
    assert mushroom(1.0, 1.0, 1.5).area == pytest.approx(
        np.pi * 1.5**2 / 2 + 1.0, rel=1e-8)                             # half disk + stem
    assert stadium(1.0, 1.0).area == pytest.approx(1.0 + np.pi / 4, rel=1e-8)  # L*H + pi R^2


def test_the_stadium_perimeter_is_two_sides_and_a_circle():
    """Two straight sides of length L, and two semicircular caps that together make one full
    circle of radius H/2 -- so `2L + 2*pi*(H/2)`, not `2L + pi*(H/2)`."""
    assert stadium(2.0, 1.0).perimeter == pytest.approx(2 * 2.0 + 2 * np.pi * 0.5, rel=1e-8)


# --- placement, which the reference eigenfunctions depend on ------------------------------


def test_rect_has_its_corner_at_the_origin():
    v = rect(2.0, 1.0).vertices
    assert np.min(np.abs(v)) == 0.0
    assert np.allclose(sorted(v.real), [0, 0, 2, 2])


def test_iso_right_tri_has_its_right_angle_at_the_origin():
    d = iso_right_tri(1.0)
    apex = [c for c in d.corners if abs(c.z) < 1e-14]
    assert len(apex) == 1
    assert apex[0].omega == pytest.approx(np.pi / 2, abs=1e-12)


def test_disk_sector_has_its_apex_at_the_origin_and_opens_from_the_real_axis():
    omega = np.pi / 3
    d = disk_sector(1.0, omega)
    apex = [c for c in d.corners if abs(c.z) < 1e-12]
    assert len(apex) == 1
    assert apex[0].omega == pytest.approx(omega, abs=1e-10)
    assert d.contains(np.array([0.5 * np.exp(1j * omega / 2)]))[0]


def test_a_reentrant_sector_apex_is_reentrant():
    d = disk_sector(1.0, 3 * np.pi / 2)
    apex = [c for c in d.corners if abs(c.z) < 1e-12][0]
    assert apex.is_reentrant


def test_the_plus_shape_sits_on_the_integer_grid():
    """The polyomino eigenfunction `sin(m pi x) sin(n pi y)` is exact only there."""
    v = plus_shape().vertices
    assert np.allclose(v.real, np.round(v.real))
    assert np.allclose(v.imag, np.round(v.imag))


def test_the_stadium_top_edge_honours_the_boundary_condition():
    """lappy builds the stadium's third segment without passing `bc`, so its top straight
    edge silently ignores the requested condition and stays Dirichlet on a Neumann domain.
    Fixed here, and pinned so it cannot come back."""
    d = stadium(1.0, 1.0, bc="neu")
    assert d.bc_type == "neu"
    q = d.bdry_quad(400)
    assert np.all(q.bc == 1.0)


def test_factories_reject_impossible_parameters():
    with pytest.raises(ValueError, match="h1 must be less"):
        chevron(2.0, 1.0)
    with pytest.raises(ValueError, match="theta"):
        disk_sector(1.0, 3 * np.pi)
    with pytest.raises(ValueError, match="r must lie"):
        cut_square(1.5)
    with pytest.raises(ValueError, match="cap radius"):
        mushroom(1.0, 3.0, 1.0)
    with pytest.raises(ValueError, match="semi-axes"):
        ellipse(-1.0, 1.0)
