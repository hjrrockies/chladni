"""Validation policy: cheap invariants always, expensive ones opt-out, orientation fixed
rather than refused.

lappy carries three flags (val_simple, val_closed, val_orientation) whose defaults
disagree between ParametricSegment and MultiSegment, and it raises on a clockwise
boundary. Here there is one flag, split by cost, and a clockwise boundary is simply
reversed -- douse trials wild polygons and should not have to pre-sort vertices.
"""
import numpy as np
import pytest

from chladni.geometry import Boundary, Domain, LineSegment, Polygon

from conftest import EXACT, L_SHAPE, PREC, SQUARE

BOWTIE = np.array([0, 1 + 1j, 1, 1j], dtype=complex)      # edges 0-1 and 2-3 cross
DEGENERATE = np.array([0, 1, 2], dtype=complex)            # collinear: zero area

# --- always checked, regardless of the flag ---------------------------------------------


def test_non_contiguous_always_rejected():
    segs = [LineSegment(0j, 1 + 0j), LineSegment(5j, 6j)]
    for validate in (True, False):
        with pytest.raises(ValueError, match="contiguous"):
            Boundary(segs, validate=validate)


def test_unclosed_boundary_rejected_by_domain():
    b = Boundary.from_vertices(SQUARE, close=False)
    for validate in (True, False):
        with pytest.raises(ValueError, match="closed"):
            Domain(b, validate=validate)


def test_zero_length_segment_rejected():
    with pytest.raises(ValueError):
        LineSegment(1 + 1j, 1 + 1j)


def test_degenerate_zero_area_rejected():
    for validate in (True, False):
        with pytest.raises(ValueError, match="degenerate|area"):
            Polygon(DEGENERATE, validate=validate)


# --- checked only under validate=True ---------------------------------------------------


def test_self_intersecting_polygon_rejected_by_default():
    with pytest.raises(ValueError, match="simple"):
        Polygon(BOWTIE)


def test_self_intersection_skipped_when_validation_is_off():
    """douse opts out. Construction must succeed and stay silent."""
    poly = Polygon(BOWTIE, validate=False)
    assert poly is not None


def test_is_valid_still_reports_after_opting_out():
    """Opting out of the check is not opting out of the answer."""
    assert not Polygon(BOWTIE, validate=False).is_valid
    assert Polygon(SQUARE, validate=False).is_valid


def test_hole_outside_the_outer_boundary_rejected():
    outer = Boundary.from_vertices(SQUARE)
    stray = Boundary.from_vertices(SQUARE + 10 + 10j).reversed()
    with pytest.raises(ValueError, match="hole"):
        Domain(outer, holes=(stray,))
    Domain(outer, holes=(stray,), validate=False)  # opt out, no raise


def test_overlapping_holes_rejected():
    outer = Boundary.from_vertices(np.array([0, 10, 10 + 10j, 10j], dtype=complex))
    h1 = Boundary.from_vertices(np.array([1 + 1j, 5 + 1j, 5 + 5j, 1 + 5j], dtype=complex))
    h2 = Boundary.from_vertices(np.array([4 + 4j, 8 + 4j, 8 + 8j, 4 + 8j], dtype=complex))
    with pytest.raises(ValueError, match="hole|disjoint"):
        Domain(outer, holes=(h1.reversed(), h2.reversed()))


# --- orientation is repaired, not refused -----------------------------------------------


def test_clockwise_vertices_are_accepted_and_reoriented():
    poly = Polygon(SQUARE[::-1])
    assert poly.outer.orientation == 1
    assert poly.area == pytest.approx(1.0, abs=EXACT)


def test_reorientation_preserves_the_shape():
    cw = Polygon(L_SHAPE[::-1])
    ccw = Polygon(L_SHAPE)
    assert cw.area == pytest.approx(ccw.area, abs=EXACT)
    assert sorted(np.round(c.theta, 10) for c in cw.corners) == sorted(
        np.round(c.theta, 10) for c in ccw.corners
    )


def test_a_clockwise_hole_stays_clockwise():
    outer = Boundary.from_vertices(np.array([0, 3, 3 + 3j, 3j], dtype=complex))
    hole_ccw = Boundary.from_vertices(np.array([1 + 1j, 2 + 1j, 2 + 2j, 1 + 2j], dtype=complex))
    d = Domain(outer, holes=(hole_ccw,))
    assert d.holes[0].orientation == -1
    assert d.area == pytest.approx(8.0, abs=EXACT)


def test_outward_normals_survive_reorientation():
    """The whole point of the convention: after repair, -1j*T points out of the domain."""
    poly = Polygon(SQUARE[::-1])
    q = poly.bdry_quad(64)
    inward = q.pts - 1e-6 * q.normals
    assert np.all(poly.contains(inward))
    assert not np.any(poly.contains(q.pts + 1e-6 * q.normals))
