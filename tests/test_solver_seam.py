"""What the layers above geometry depend on.

`particular_solutions` and `mps` will be written against this file. Everything here is a
contract that, if missing, forces the layer above to reach into geometry's internals -- which
is the seam lappy ended up with, where `assemble_panels` lives in eigfun_integrals and its
own comments record a downstream package having to redo the affine panel map by hand.
"""
import dataclasses

import numpy as np
import pytest

from chladni.geometry import (
    Boundary,
    Corner,
    Domain,
    L_shape,
    Panel,
    Polygon,
    annulus,
    legendre_nodes,
)

from conftest import EXACT, PREC


@pytest.fixture
def holed():
    """3x3 square with a unit hole: the shape where every multiply-connected bug showed."""
    outer = Boundary.from_vertices(np.array([0, 3, 3 + 3j, 3j], dtype=complex))
    hole = Boundary.from_vertices(
        np.array([1 + 1j, 2 + 1j, 2 + 2j, 1 + 2j], dtype=complex)
    ).reversed()
    return Domain(outer, holes=(hole,))


# --- the panel-plan seam ----------------------------------------------------------------


def test_a_caller_supplied_plan_reproduces_the_built_in_rule(l_shape):
    """One panel of Gauss-Legendre per segment is what geometry does by default, so a plan
    saying exactly that must agree to the last bit."""
    n_seg = len(l_shape.outer.segments)
    plan = [Panel(0, i, 0.0, 1.0, legendre_nodes(24)) for i in range(n_seg)]
    a = l_shape.outer.quad(panels=plan)
    # n_per_seg, not a total: the built-in splits a total by ARCLENGTH, and the L-shape's
    # edges are 2,1,1,1,1,2, so n=144 is not 24 apiece
    b = l_shape.outer.quad(n_per_seg=[24] * n_seg)
    assert np.array_equal(a.seg_idx, b.seg_idx)
    assert np.allclose(a.pts, b.pts)
    assert np.allclose(a.wts, b.wts)
    assert np.allclose(a.tau, b.tau)


def test_a_plan_gets_the_full_quadrature_contract(l_shape):
    """Frames, arclength weights, provenance and BC tagging all come from geometry; the
    plan supplies only node placement and weights."""
    plan = [Panel(0, i, 0.0, 1.0, legendre_nodes(16)) for i in range(len(l_shape.outer.segments))]
    q = l_shape.bdry_quad(panels=plan)
    assert q.wts.sum() == pytest.approx(l_shape.perimeter, rel=PREC)
    assert (q.radial(0j) * q.wts).sum() == pytest.approx(2 * l_shape.area, rel=1e-8)
    assert np.allclose(q.normals, -1j * q.tangents)
    assert set(np.unique(q.seg_idx)) == set(range(len(l_shape.outer.segments)))
    assert np.all((q.tau >= 0) & (q.tau <= 1))
    for i in np.unique(q.seg_idx):
        m = q.seg_idx == i
        assert np.allclose(l_shape.outer.segments[i].p(q.tau[m]), q.pts[m], atol=PREC)
        assert np.all(q.bc[m] == l_shape.outer.segments[i].bc)


def test_a_plan_may_split_a_segment_into_many_panels(l_shape):
    """Corner-anchored panels are the point: a short panel at the corner, a long one after."""
    plan = []
    for i in range(len(l_shape.outer.segments)):
        plan += [Panel(0, i, 0.0, 0.05, legendre_nodes(12)),
                 Panel(0, i, 0.05, 1.0, legendre_nodes(20))]
    q = l_shape.bdry_quad(panels=plan)
    assert q.wts.sum() == pytest.approx(l_shape.perimeter, rel=PREC)
    assert len(np.unique(q.panel_id)) == 2 * len(l_shape.outer.segments)
    for pid in np.unique(q.panel_id):
        assert len(np.unique(q.seg_idx[q.panel_id == pid])) == 1


def test_a_plan_declares_its_weight_family(l_shape):
    """A singular rule must not come back looking like an arclength one."""
    plan = [Panel(0, i, 0.0, 1.0, legendre_nodes(16)) for i in range(len(l_shape.outer.segments))]
    assert l_shape.bdry_quad(panels=plan).weight_family == "arclength"
    q = l_shape.bdry_quad(panels=plan, weight_family="even")
    assert q.weight_family == "even"


def test_a_plan_spans_holes(holed):
    plan = [Panel(bi, si, 0.0, 1.0, legendre_nodes(20))
            for bi, b in enumerate(holed.boundaries) for si in range(len(b.segments))]
    q = holed.bdry_quad(panels=plan)
    assert set(np.unique(q.bdry_idx)) == {0, 1}
    assert q.wts.sum() == pytest.approx(holed.perimeter, rel=PREC)
    assert (q.radial(0j) * q.wts).sum() == pytest.approx(2 * holed.area, rel=1e-8)


def test_weights_come_back_in_tau_and_geometry_only_converts_to_ds(l_shape):
    """The homogeneity of a singular weight under panel rescaling belongs to whoever chose
    it, so geometry multiplies by length and does nothing else."""
    seg = l_shape.outer.segments[0]
    marker = lambda t0, t1: (np.array([0.5 * (t0 + t1)]), np.array([t1 - t0]))
    q = seg.quad(panels=[Panel(0, 0, 0.0, 1.0, marker)])
    assert q.wts[0] == pytest.approx(seg.length, rel=1e-14)


# --- corners carry enough to be used -----------------------------------------------------


def test_corner_boundary_index_disambiguates(holed):
    """Without bdry_idx the flattened tuple is ambiguous: corner 0 of the outer boundary
    and corner 0 of the hole both report seg_out == 0."""
    keys = {(c.bdry_idx, c.seg_out) for c in holed.corners}
    assert len(keys) == len(holed.corners) == 8
    assert (0, 0) in keys and (1, 0) in keys


def test_corner_indices_are_the_join_key_into_a_quadrature(holed):
    """Selecting a corner's own nodes has to work by integer comparison."""
    q = holed.bdry_quad(800)
    for c in holed.corners:
        m = (q.bdry_idx == c.bdry_idx) & (q.seg_idx == c.seg_out)
        assert m.any()
        seg = holed.boundaries[c.bdry_idx].segments[c.seg_out]
        assert np.allclose(seg.p(q.tau[m]), q.pts[m], atol=PREC)


def test_corner_segment_references_resolve(holed):
    for c in holed.corners:
        assert c.segment_out is holed.boundaries[c.bdry_idx].segments[c.seg_out]
        assert c.segment_in is holed.boundaries[c.bdry_idx].segments[c.seg_in]
        assert c.segment_out.p0 == pytest.approx(c.z, abs=PREC)
        # the immediate use: clearance as a fraction of the adjacent edge
        assert 0 < c.clearance / c.segment_out.length


def test_corner_references_point_at_leaves_only(holed):
    """Corner -> Segment is safe; Corner -> Boundary or Domain would close the reference
    cycle that lappy's cache.py exists to undo."""
    for c in holed.corners:
        for f in dataclasses.fields(c):
            assert not isinstance(getattr(c, f.name), (Boundary, Domain))


def test_segment_references_stay_out_of_equality_and_repr(square):
    c = square.corners[0]
    assert "LineSegment" not in repr(c)
    twin = dataclasses.replace(c, segment_out=None, segment_in=None)
    assert twin == c  # equality is by value on the numbers, not by segment identity


# --- clearance across boundaries ---------------------------------------------------------


def test_clearance_accounts_for_holes(holed):
    """The radius of the largest disk about the corner INSIDE Omega. Boundary-local
    clearance reported 3.0 here; the hole's nearest corner is at sqrt(2)."""
    c = next(c for c in holed.corners if abs(c.z) < 1e-12)
    assert c.clearance == pytest.approx(np.sqrt(2), abs=PREC)


def test_clearance_is_a_true_inscribed_radius(holed):
    """Whatever it reports, a disk of that radius about the corner must not meet another
    boundary."""
    for c in holed.corners:
        others = [b for k, b in enumerate(holed.boundaries) if k != c.bdry_idx]
        for b in others:
            assert float(b.dist(c.z)) >= c.clearance - PREC


def test_boundary_local_clearance_is_an_upper_bound(holed):
    """A lone Boundary cannot see a hole it knows nothing about, so its clearance is an
    over-estimate -- which is why Domain.corners tightens it."""
    local = next(c for c in holed.outer.corners if abs(c.z) < 1e-12)
    full = next(c for c in holed.corners if abs(c.z) < 1e-12)
    assert local.clearance >= full.clearance
    assert local.clearance == pytest.approx(3.0, abs=PREC)


# --- the branch-cut obstruction on holes -------------------------------------------------


def test_only_outer_corners_admit_an_exterior_branch_cut(holed):
    """A loop in Omega encircling the hole winds around every point of that hole's
    boundary, so omega measured from a hole corner gains 2*pi on it. Killing that loop
    needs a cut joining the hole to the outer boundary, which crosses Omega."""
    assert [c.on_hole for c in holed.corners] == [False] * 4 + [True] * 4
    assert all(c.admits_exterior_branch_cut for c in holed.corners if not c.on_hole)
    assert not any(c.admits_exterior_branch_cut for c in holed.corners if c.on_hole)


def test_every_corner_of_a_simply_connected_domain_admits_a_cut(l_shape, square, eq_tri):
    for d in (l_shape, square, eq_tri):
        assert all(c.admits_exterior_branch_cut for c in d.corners)
        assert not any(c.on_hole for c in d.corners)


def test_hole_branch_angle_still_points_into_the_hole(holed):
    """It is not a valid cut, but it IS a genuine direction into the adjacent exterior
    component -- which is what you want for placing a fundamental solution's source.

    The escape test that outer corners use is meaningless here: every ray crosses a bounded
    hole and leaves, so it fired on the correct direction and the fallback returned one
    pointing INTO Omega -- 1.25*pi at (1,1) where the true bisector is 0.25*pi.
    """
    for c in holed.corners:
        if not c.on_hole:
            continue
        assert np.isfinite(c.branch_angle)
        ray = c.z + np.geomspace(1e-6, 0.4, 24) * np.exp(1j * c.branch_angle)
        assert not np.any(holed.contains(ray))


def test_hole_corner_angles_are_measured_inside_the_domain(holed):
    """omega is the wedge of Omega, not of the hole: a convex hole corner is reentrant
    from the domain's side."""
    for c in holed.corners:
        if c.on_hole:
            assert c.omega == pytest.approx(3 * np.pi / 2, abs=PREC)
            assert c.is_reentrant
