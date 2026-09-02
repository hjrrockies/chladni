"""Corner detection, angles, and the two facts lappy gets wrong.

Two regressions are pinned here deliberately:

* the half-disk, where three segments produce two corners because the radial pieces are
  collinear. lappy carried corner-indexed and segment-indexed arrays side by side and they
  desynced on exactly this shape.
* a Dirichlet/Neumann change across a straight edge, which carries a genuine singularity
  while having interior angle exactly pi. lappy's purely geometric test
  (`abs(turning_angle) > 1e-10`) reports no corner at all.
"""
import dataclasses

import numpy as np
import pytest

from chladni.geometry import Boundary, Corner, LineSegment, Polygon
from chladni.helmholtz import CornerModes

from conftest import COLLINEAR, EQ_TRI, EXACT, L_SHAPE, L_SHAPE_ANGLES, PREC, SQUARE

# --- angles -----------------------------------------------------------------------------


def test_square_corners(square):
    corners = square.corners
    assert len(corners) == 4
    assert np.allclose([c.omega for c in corners], np.pi / 2, atol=EXACT)


def test_eq_tri_corners(eq_tri):
    assert np.allclose([c.omega for c in eq_tri.corners], np.pi / 3, atol=EXACT)


def test_l_shape_corners(l_shape):
    corners = l_shape.corners
    assert len(corners) == 6
    assert np.allclose([c.omega for c in corners], L_SHAPE_ANGLES, atol=EXACT)


def test_exactly_one_reentrant_corner_on_the_l_shape(l_shape):
    reentrant = [c for c in l_shape.corners if c.is_reentrant]
    assert len(reentrant) == 1
    assert reentrant[0].z == pytest.approx(1 + 1j, abs=EXACT)
    assert reentrant[0].omega == pytest.approx(3 * np.pi / 2, abs=EXACT)


@pytest.mark.parametrize("n", [3, 4, 5, 6, 8, 12])
def test_regular_ngon_angles(n):
    from conftest import ngon_vertices

    poly = Polygon(ngon_vertices(n))
    assert np.allclose([c.omega for c in poly.corners], (n - 2) * np.pi / n, atol=EXACT)


@pytest.mark.parametrize("verts", [SQUARE, L_SHAPE, EQ_TRI, COLLINEAR])
def test_interior_angles_sum_to_n_minus_two_pi(verts):
    """Every junction, corner or not, contributes; the sum is a hard identity."""
    poly = Polygon(verts, validate=False)
    total = sum(c.omega for c in poly.corners)
    smooth = len(verts) - len(poly.corners)
    assert total + smooth * np.pi == pytest.approx((len(verts) - 2) * np.pi, abs=EXACT)


# --- smooth junctions are not corners ---------------------------------------------------


def test_collinear_vertex_is_not_a_corner(collinear_poly):
    """Five vertices, four corners: the collinear one has omega == pi."""
    assert len(collinear_poly.corners) == 4
    assert not any(c.z == pytest.approx(1 + 0j, abs=EXACT) for c in collinear_poly.corners)


def test_half_disk_has_three_segments_and_two_corners(half_disk):
    """lappy's desync case. Both corners are pi/2, at the ends of the diameter."""
    assert len(half_disk.outer.segments) == 3
    corners = half_disk.corners
    assert len(corners) == 2
    assert np.allclose([c.omega for c in corners], np.pi / 2, atol=PREC)
    assert np.allclose(sorted(c.z.real for c in corners), [-1.0, 1.0], atol=PREC)


def test_corner_indices_point_at_their_own_segment(half_disk):
    """`seg_out` must index the segment whose p0 IS the corner. This is the invariant
    whose absence made lappy's parallel arrays unmaskable."""
    segs = half_disk.outer.segments
    for c in half_disk.corners:
        assert segs[c.seg_out].p0 == pytest.approx(c.z, abs=PREC)
        assert segs[c.seg_in].p1 == pytest.approx(c.z, abs=PREC)
        assert c.seg_in == (c.seg_out - 1) % len(segs)


def test_corners_follow_traversal_order(l_shape):
    idx = [c.seg_out for c in l_shape.corners]
    assert idx == sorted(idx)


# --- boundary-condition junctions -------------------------------------------------------


def test_bc_change_across_a_straight_edge_is_a_singular_corner(split_edge_bc_junction):
    """omega is exactly pi, so it is not geometric, and the wedge IS reflection-closed --
    yet the eigenfunction is singular there. Geometry reports the junction; only
    `CornerModes` calls it singular."""
    corners = split_edge_bc_junction.corners
    # dir->neu at 0.5 and neu->dir at 1: two changes, but only one is on a straight edge
    assert len([c for c in corners if c.is_bc_junction]) == 2
    straight = [c for c in corners if c.is_bc_junction and not c.is_geometric]
    assert len(straight) == 1
    c = straight[0]
    assert c.z == pytest.approx(0.5 + 0j, abs=EXACT)
    assert c.omega == pytest.approx(np.pi, abs=EXACT)
    assert not c.is_geometric
    assert c.is_reflection_closed
    assert CornerModes.from_corner(c).is_singular
    assert (c.bc_in, c.bc_out) == (0.0, 1.0)


# --- which corners the angle alone can vouch for ------------------------------------------


def _wedge_corner(omega, bc_out=0.0, bc_in=0.0):
    """A bare `Corner` with a prescribed angle and BC pair.

    The predicates under test read `omega`, `bc_in` and `bc_out` and nothing else, so a wedge
    that no closed polygon realises -- omega = 0.7, say -- can still be tested. The tangents
    are kept consistent with `omega = angle(-t_in/t_out)` so the object is not a lie.
    """
    return Corner(z=0j, seg_out=0, seg_in=1, omega=omega, t_in=-np.exp(1j * omega),
                  t_out=1 + 0j, bc_in=bc_in, bc_out=bc_out, clearance=1.0,
                  branch_angle=float("nan"))


def test_reflection_closed_is_exactly_pi_over_an_integer():
    for omega, closed in [(np.pi, True), (np.pi / 2, True), (np.pi / 3, True),
                          (np.pi / 4, True), (2 * np.pi / 3, False), (0.7, False),
                          (3 * np.pi / 2, False)]:
        assert _wedge_corner(omega).is_reflection_closed is closed, omega


def test_reflection_closure_ignores_the_boundary_conditions():
    """The angle is the whole test. A right-angled Dirichlet/Neumann junction is closed even
    though its BC pair is what would decide the offset -- which is not geometry's to know."""
    assert _wedge_corner(np.pi / 2, bc_out=1.0, bc_in=0.0).is_reflection_closed


def test_a_reentrant_corner_is_never_reflection_closed(l_shape):
    """omega > pi puts pi/omega in (1/2, 1), which is never an integer."""
    assert not any(c.is_reflection_closed for c in l_shape.corners if c.is_reentrant)
    assert all(c.is_reflection_closed for c in l_shape.corners if not c.is_reentrant)


def test_right_angles_are_corners_and_the_eigenfunction_is_analytic_there(square):
    """A square's corners are geometric, reflection-closed, and not singular. This is the
    distinction `is_geometric` alone cannot make, and the reason it is not geometry's to
    make either: the answer needs the BC pair as well as the angle."""
    assert all(c.is_geometric and c.is_reflection_closed for c in square.corners)
    assert not any(CornerModes.from_corner(c).is_singular for c in square.corners)


def test_geometry_never_misses_a_singular_corner(l_shape, square, half_disk,
                                                 split_edge_bc_junction, rect_4x1):
    """The contract `grading="corner"` rests on. Geometry cannot name the singular set --
    the offset is a Helmholtz fact -- but the set it grades must contain it, or the
    quadrature is under-resolved exactly where resolution matters."""
    for dom in (l_shape, square, half_disk, split_edge_bc_junction, rect_4x1):
        for c in dom.corners:
            if CornerModes.from_corner(c).is_singular:
                assert not c.is_reflection_closed or c.is_bc_junction


def test_the_containment_is_strict_at_a_right_angled_zaremba_corner(split_edge_bc_junction):
    """The one shape of corner geometry grades and need not: orders 1, 3, 5 are integral,
    so the eigenfunction is analytic, but only the offset says so."""
    c = next(c for c in split_edge_bc_junction.corners
             if c.z == pytest.approx(1 + 0j, abs=EXACT))
    assert c.is_bc_junction and c.is_reflection_closed
    assert not CornerModes.from_corner(c).is_singular


def test_corner_carries_the_bc_pair_not_just_the_angle():
    """The singularity exponent depends on the BC pair, so both sides must be reachable
    from the corner without walking back into the boundary."""
    bdry = Boundary(
        [
            LineSegment(0j, 1 + 0j, bc="neu"),
            LineSegment(1 + 0j, 1 + 1j, bc="dir"),
            LineSegment(1 + 1j, 1j, bc="dir"),
            LineSegment(1j, 0j, bc=0.3),
        ]
    )
    by_z = {complex(np.round(c.z, 12)): c for c in bdry.corners}
    assert (by_z[1 + 0j].bc_in, by_z[1 + 0j].bc_out) == (1.0, 0.0)
    assert (by_z[0j].bc_in, by_z[0j].bc_out) == (0.3, 1.0)
    assert by_z[0j].is_bc_junction
    assert not by_z[1 + 1j].is_bc_junction


# --- clearance --------------------------------------------------------------------------


def test_clearance_ignores_the_adjacent_segments(rect_4x1):
    """Distance to the nearest NON-adjacent boundary piece. Counting the adjacent edges
    would give 0; taking the far edge would give 4. The answer is the top edge, at 1."""
    origin = next(c for c in rect_4x1.corners if c.z == pytest.approx(0j, abs=EXACT))
    assert origin.clearance == pytest.approx(1.0, abs=PREC)


def test_clearance_on_the_reentrant_corner(l_shape):
    reentrant = next(c for c in l_shape.corners if c.is_reentrant)
    assert reentrant.clearance == pytest.approx(1.0, abs=PREC)


def test_clearance_is_positive_everywhere(l_shape, half_disk, square_with_square_hole):
    for dom in (l_shape, half_disk, square_with_square_hole):
        assert all(c.clearance > 0 for c in dom.corners)


# --- branch cut rays --------------------------------------------------------------------


def test_branch_ray_leaves_a_convex_corner_along_the_exterior_bisector(square):
    origin = next(c for c in square.corners if c.z == pytest.approx(0j, abs=EXACT))
    assert origin.branch_angle == pytest.approx(5 * np.pi / 4, abs=PREC)


def test_branch_ray_at_a_reentrant_corner(l_shape):
    """The missing quadrant at (1,1) spans directions (0, pi/2); its bisector is pi/4."""
    reentrant = next(c for c in l_shape.corners if c.is_reentrant)
    assert reentrant.branch_angle == pytest.approx(np.pi / 4, abs=PREC)


def test_branch_ray_stays_outside_the_domain(l_shape, square, half_disk):
    """The property that actually matters: the ray must not re-enter the domain."""
    for dom in (l_shape, square, half_disk):
        for c in dom.corners:
            assert np.isfinite(c.branch_angle)
            eps = np.geomspace(1e-6, 10 * dom.diameter, 60)
            ray = c.z + eps * np.exp(1j * c.branch_angle)
            assert not np.any(dom.contains(ray))


# --- the object itself ------------------------------------------------------------------


def test_corner_is_frozen(square):
    c = square.corners[0]
    assert dataclasses.is_dataclass(c)
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.omega = 1.0


def test_angle_tolerance_is_exposed_not_buried(square):
    assert square.outer.angle_atol == pytest.approx(max(square.outer.prec, 1e-12))
