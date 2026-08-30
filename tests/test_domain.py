"""Domain: metric quantities, containment, multiple connectivity, normalization."""
import numpy as np
import pytest

from chladni.geometry import ArcSegment, Boundary, Domain, LineSegment, Polygon

from conftest import (
    EQ_TRI_AREA,
    EQ_TRI_INRADIUS,
    EXACT,
    L_SHAPE_AREA,
    L_SHAPE_DIAMETER,
    L_SHAPE_PERIMETER,
    PREC,
    ngon_vertices,
)

# --- metric quantities ------------------------------------------------------------------


def test_square_metrics(square):
    assert square.area == pytest.approx(1.0, abs=EXACT)
    assert square.perimeter == pytest.approx(4.0, abs=EXACT)
    assert square.diameter == pytest.approx(np.sqrt(2), abs=PREC)
    assert square.inradius == pytest.approx(0.5, abs=PREC)
    assert square.centroid == pytest.approx(0.5 + 0.5j, abs=PREC)


def test_l_shape_metrics(l_shape):
    assert l_shape.area == pytest.approx(L_SHAPE_AREA, abs=EXACT)
    assert l_shape.perimeter == pytest.approx(L_SHAPE_PERIMETER, abs=EXACT)
    assert l_shape.diameter == pytest.approx(L_SHAPE_DIAMETER, abs=PREC)


def test_eq_tri_metrics(eq_tri):
    assert eq_tri.area == pytest.approx(EQ_TRI_AREA, abs=EXACT)
    assert eq_tri.perimeter == pytest.approx(3.0, abs=EXACT)
    assert eq_tri.inradius == pytest.approx(EQ_TRI_INRADIUS, abs=PREC)


def test_disk_metrics(unit_disk):
    assert unit_disk.area == pytest.approx(np.pi, rel=PREC)
    assert unit_disk.perimeter == pytest.approx(2 * np.pi, rel=PREC)
    assert unit_disk.diameter == pytest.approx(2.0, rel=PREC)
    assert unit_disk.inradius == pytest.approx(1.0, rel=PREC)
    assert unit_disk.centroid == pytest.approx(0j, abs=PREC)


def test_half_disk_area(half_disk):
    assert half_disk.area == pytest.approx(np.pi / 2, rel=PREC)
    assert half_disk.perimeter == pytest.approx(2 + np.pi, rel=PREC)


@pytest.mark.parametrize("n", [3, 4, 5, 6, 8, 20])
def test_regular_ngon_closed_forms(n):
    poly = Polygon(ngon_vertices(n))
    assert poly.area == pytest.approx(n / 2 * np.sin(2 * np.pi / n), abs=EXACT)
    assert poly.perimeter == pytest.approx(2 * n * np.sin(np.pi / n), abs=EXACT)
    assert poly.inradius == pytest.approx(np.cos(np.pi / n), abs=PREC)


def test_bbox(l_shape):
    lo, hi = l_shape.bbox
    assert lo == pytest.approx(0j, abs=EXACT)
    assert hi == pytest.approx(2 + 2j, abs=EXACT)


# --- containment ------------------------------------------------------------------------


def test_contains_is_vectorized(square):
    pts = np.array([0.5 + 0.5j, 1.5 + 0.5j, -1 + 0j, 0.1 + 0.9j])
    out = square.contains(pts)
    assert out.shape == pts.shape
    assert out.dtype == bool
    assert list(out) == [True, False, False, True]


def test_contains_the_bite_of_the_l_shape(l_shape):
    """(1.5, 1.5) is inside the bounding square but outside the L."""
    assert not l_shape.contains(np.array([1.5 + 1.5j]))[0]
    assert l_shape.contains(np.array([0.5 + 1.5j]))[0]
    assert l_shape.contains(np.array([1.5 + 0.5j]))[0]


def test_contains_agrees_with_a_lattice_area_estimate(l_shape):
    g = np.linspace(-0.5, 2.5, 400)
    z = g[:, None] + 1j * g[None, :]
    frac = l_shape.contains(z.ravel()).mean()
    assert frac * 9.0 == pytest.approx(l_shape.area, rel=2e-2)


# --- multiple connectivity --------------------------------------------------------------


def test_annulus_area_and_perimeter(annulus):
    assert annulus.area == pytest.approx(3 * np.pi, rel=PREC)
    assert annulus.perimeter == pytest.approx(6 * np.pi, rel=PREC)
    assert not annulus.is_simply_connected
    assert len(annulus.holes) == 1


def test_annulus_containment(annulus):
    pts = np.array([1.5 + 0j, 0.5 + 0j, 3 + 0j, 0j, 1.5j])
    assert list(annulus.contains(pts)) == [True, False, False, False, True]


def test_square_with_square_hole(square_with_square_hole):
    d = square_with_square_hole
    assert d.area == pytest.approx(8.0, abs=EXACT)
    assert d.perimeter == pytest.approx(16.0, abs=EXACT)
    assert list(d.contains(np.array([0.5 + 0.5j, 1.5 + 1.5j, 2.5 + 2.5j]))) == [
        True,
        False,
        True,
    ]


def test_hole_orientation_is_clockwise(annulus, square_with_square_hole):
    """The convention that makes N = -1j*T outward on every boundary without special
    casing, and makes signed areas simply add."""
    for d in (annulus, square_with_square_hole):
        assert d.outer.orientation == 1
        assert all(h.orientation == -1 for h in d.holes)
        assert d.area == pytest.approx(sum(b.signed_area for b in d.boundaries), abs=PREC)


def test_corners_span_every_boundary(square_with_square_hole):
    assert len(square_with_square_hole.corners) == 8


def test_simply_connected_domain_has_no_holes(square):
    assert square.is_simply_connected
    assert square.holes == ()


# --- normalization ----------------------------------------------------------------------


@pytest.mark.parametrize("by", ["diameter", "area", "inradius"])
def test_normalized_sets_the_named_quantity_to_one(l_shape, by):
    d = l_shape.normalized(by=by)
    assert getattr(d, by) == pytest.approx(1.0, rel=PREC)


def test_normalized_preserves_shape(l_shape):
    d = l_shape.normalized(by="diameter")
    s = l_shape.diameter
    assert d.area == pytest.approx(l_shape.area / s**2, rel=PREC)
    assert d.perimeter == pytest.approx(l_shape.perimeter / s, rel=PREC)
    assert np.allclose(
        sorted(c.theta for c in d.corners), sorted(c.theta for c in l_shape.corners), atol=PREC
    )


def test_normalized_returns_a_new_domain(l_shape):
    d = l_shape.normalized()
    assert d is not l_shape
    assert l_shape.diameter == pytest.approx(L_SHAPE_DIAMETER, abs=PREC)


# --- boundary condition summary ---------------------------------------------------------


def test_bc_type(square):
    assert square.bc_type == "dir"
    assert square.with_bc("neu").bc_type == "neu"


def test_mixed_and_robin_bc_types(split_edge_bc_junction):
    assert split_edge_bc_junction.bc_type == "mixed"
    robin = Domain(
        Boundary(
            [
                LineSegment(0j, 1 + 0j, bc=0.5),
                LineSegment(1 + 0j, 1 + 1j),
                LineSegment(1 + 1j, 1j),
                LineSegment(1j, 0j),
            ]
        )
    )
    assert robin.bc_type == "rob"


# --- interior points --------------------------------------------------------------------


@pytest.mark.parametrize("method", ["random", "grid"])
def test_interior_points_are_interior(l_shape, method):
    pts = l_shape.interior_pts(500, method=method, rng=0)
    assert len(pts) == 500
    assert np.all(l_shape.contains(pts))


def test_interior_points_avoid_the_hole(annulus):
    pts = annulus.interior_pts(400, rng=0)
    assert np.all(np.abs(pts) > 1.0)
    assert np.all(np.abs(pts) < 2.0)


def test_interior_points_are_reproducible(l_shape):
    a = l_shape.interior_pts(100, rng=1234)
    b = l_shape.interior_pts(100, rng=1234)
    assert np.array_equal(a, b)


def test_interior_points_spread_over_the_domain(l_shape):
    """A sanity check against a degenerate sampler that clusters in one cell."""
    pts = l_shape.interior_pts(2000, rng=0)
    assert pts.real.min() < 0.3 and pts.real.max() > 1.7
    assert pts.imag.min() < 0.3 and pts.imag.max() > 1.7
