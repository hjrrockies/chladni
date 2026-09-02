"""Quadrature: weights, provenance, and the integral identities they must satisfy.

lappy's `Segment.pts(n, kind='legendre')` computes `tau, _wts = quadfunc(n)` and discards
the weights; `MultiSegment.pts` then concatenates the results by abusing `+` under
`np.sum`. Rellich normalization integrates (x - x0).n against |grad u|^2 over the boundary,
so the weights have to survive into the object the solver is handed. Everything below is a
consequence of that.

The integral identities here -- weights summing to the perimeter, oint n ds == 0, the
divergence theorem -- hold for the ARCLENGTH family only, i.e. a rule whose weight function
is 1. A corner-adapted Gauss-Jacobi rule folds a singular weight into its own weights, so
its weights sum to oint w ds instead. Geometry only ever builds arclength rules, but
`Quadrature` is also the type the MPS layer returns, so every identity test below asserts
the family first rather than assuming it.
"""
import dataclasses

import numpy as np
import pytest

from chladni.geometry import ArcSegment, Boundary, LineSegment, SplineSegment

from conftest import EXACT, PREC, SQUARE

# --- the basic contract -----------------------------------------------------------------


def test_geometry_only_builds_arclength_rules(square, l_shape, unit_disk, annulus):
    """Geometry does not know nu, so it cannot build a singular rule and must not claim to."""
    for dom in (square, l_shape, unit_disk, annulus):
        assert dom.bdry_quad(200).weight_family == "arclength"


def test_arclength_weights_sum_to_length(square, l_shape, unit_disk, annulus):
    for dom in (square, l_shape, unit_disk, annulus):
        q = dom.bdry_quad(600)
        assert q.weight_family == "arclength"
        assert q.wts.sum() == pytest.approx(dom.perimeter, rel=PREC)


def test_integrating_one_gives_the_perimeter(l_shape):
    q = l_shape.bdry_quad(400)
    assert q.integrate(lambda z: np.ones_like(z, dtype=float)) == pytest.approx(
        l_shape.perimeter, rel=PREC
    )


def test_weights_are_positive(l_shape, unit_disk):
    for dom in (l_shape, unit_disk):
        assert np.all(dom.bdry_quad(300).wts > 0)


def test_arrays_are_aligned(l_shape):
    q = l_shape.bdry_quad(257)
    n = len(q.pts)
    for name in ("normals", "tangents", "wts", "tau", "seg_idx", "bdry_idx", "bc", "panel_id"):
        assert len(getattr(q, name)) == n


def test_nodes_lie_on_the_boundary(l_shape):
    q = l_shape.bdry_quad(200)
    assert np.allclose([l_shape.outer.dist(p) for p in q.pts], 0.0, atol=PREC)


def test_frames_match_the_segments(unit_disk):
    q = unit_disk.bdry_quad(128)
    assert np.allclose(q.normals, -1j * q.tangents)
    assert np.allclose(np.abs(q.normals), 1.0, rtol=PREC)
    assert np.allclose(q.normals, q.pts, atol=PREC)  # outward radial on the unit circle


# --- exact integral identities ----------------------------------------------------------


def test_circle_integrals_exactly(unit_disk):
    q = unit_disk.bdry_quad(200)
    assert q.integrate(lambda z: z.real**2) == pytest.approx(np.pi, rel=PREC)
    assert q.integrate(lambda z: z.real) == pytest.approx(0.0, abs=PREC)
    assert q.integrate(lambda z: z.real * z.imag) == pytest.approx(0.0, abs=PREC)


def test_normals_integrate_to_zero(square, l_shape, unit_disk, annulus):
    """oint n ds = 0 for any closed boundary. Catches a flipped hole normal."""
    for dom in (square, l_shape, unit_disk, annulus):
        q = dom.bdry_quad(600)
        assert (q.normals * q.wts).sum() == pytest.approx(0j, abs=PREC * dom.perimeter)


@pytest.mark.parametrize("x0", [0j, 1 + 1j, -5 + 3j])
def test_divergence_theorem(square, l_shape, unit_disk, eq_tri, x0):
    """oint (x - x0).n ds == 2 * area, for ANY base point.

    This is the single strongest check available at this layer: it ties pts, normals and
    wts together against an independently known area, and it is exactly the integrand
    shape the Rellich identity uses. Arclength family only -- a singular rule integrates
    against w, not ds, and this identity says nothing about it.
    """
    for dom in (square, l_shape, unit_disk, eq_tri):
        q = dom.bdry_quad(800)
        assert q.weight_family == "arclength"
        assert (q.radial(x0) * q.wts).sum() == pytest.approx(2 * dom.area, rel=1e-6)


@pytest.mark.parametrize("x0", [0j, 1.4 + 0.2j])
def test_divergence_theorem_with_holes(annulus, square_with_square_hole, x0):
    """The same identity across a multiply-connected boundary. Fails outright if hole
    orientation or normal direction is wrong."""
    for dom in (annulus, square_with_square_hole):
        q = dom.bdry_quad(1200)
        assert (q.radial(x0) * q.wts).sum() == pytest.approx(2 * dom.area, rel=1e-6)


def test_radial_is_the_real_dot_product(square):
    q = square.bdry_quad(64)
    x0 = 0.3 - 0.2j
    manual = (q.pts.real - x0.real) * q.normals.real + (q.pts.imag - x0.imag) * q.normals.imag
    assert np.allclose(q.radial(x0), manual)


def test_radial_defaults_to_the_origin(square):
    q = square.bdry_quad(64)
    assert np.allclose(q.radial(), q.radial(0j))


# --- convergence ------------------------------------------------------------------------


def test_smooth_integrand_converges_spectrally(unit_disk):
    """On an analytic boundary the error must fall far faster than any power of n."""
    exact = 2 * np.pi * float(np.i0(1.0))  # oint exp(x) ds on the unit circle
    errs = [
        abs(unit_disk.bdry_quad(n).integrate(lambda z: np.exp(z.real)) - exact)
        for n in (8, 16, 32)
    ]
    assert errs[-1] < 1e-12
    assert errs[-1] < errs[0] / 1e6
    # measured: 1.7e-04, 1.7e-11, 2.1e-14. The rule is at machine precision by n=32, so a
    # range starting at 16 could only be asking for sub-machine accuracy.


def test_panels_break_at_corners_so_a_polygon_is_exact(square):
    """A panel spanning a vertex would destroy convergence; per-segment panels make a
    piecewise-polynomial integrand exact at low order."""
    q = square.bdry_quad(32)
    assert q.integrate(lambda z: z.real**3) == pytest.approx(1.5, abs=1e-12)


def test_spline_boundary_converges(unit_disk):
    """Knot-aligned panels are the point of break_taus; without them a Gauss rule over a
    knot loses spectral convergence no matter how fine the parameterization."""
    pts = np.exp(2j * np.pi * np.arange(12) / 12)
    seg = SplineSegment.interp_from_pts(np.append(pts, pts[0]))
    bdry = Boundary([seg], validate=False)
    coarse = bdry.quad(120).integrate(lambda z: np.ones_like(z, dtype=float))
    fine = bdry.quad(480).integrate(lambda z: np.ones_like(z, dtype=float))
    assert abs(fine - bdry.length) < abs(coarse - bdry.length) + 1e-14
    assert abs(fine - bdry.length) < 1e-10


# --- provenance -------------------------------------------------------------------------


def test_tau_is_within_segment_normalized_arclength(l_shape):
    q = l_shape.bdry_quad(300)
    assert np.all((q.tau >= 0) & (q.tau <= 1))
    for i in range(len(l_shape.outer.segments)):
        m = q.seg_idx == i
        assert np.allclose(l_shape.outer.segments[i].p(q.tau[m]), q.pts[m], atol=PREC)


def test_bdry_idx_separates_outer_from_holes(annulus):
    q = annulus.bdry_quad(400)
    assert set(np.unique(q.bdry_idx)) == {0, 1}
    assert np.all(np.abs(q.pts[q.bdry_idx == 0]) == pytest.approx(2.0, abs=PREC))
    assert np.all(np.abs(q.pts[q.bdry_idx == 1]) == pytest.approx(1.0, abs=PREC))


def test_a_panel_never_straddles_two_segments(l_shape):
    q = l_shape.bdry_quad(300)
    for pid in np.unique(q.panel_id):
        assert len(np.unique(q.seg_idx[q.panel_id == pid])) == 1


def test_bc_is_carried_per_node(split_edge_bc_junction):
    q = split_edge_bc_junction.bdry_quad(400)
    assert set(np.unique(q.bc)) == {0.0, 1.0}
    assert np.all(q.bc[q.seg_idx == 1] == 1.0)
    assert np.array_equal(q.neu_mask, q.bc == 1.0)
    assert np.array_equal(q.dir_mask, q.bc == 0.0)
    # the Neumann piece is the half-edge from 0.5 to 1
    assert q.wts[q.neu_mask].sum() == pytest.approx(0.5, rel=PREC)


# --- grading ----------------------------------------------------------------------------


def test_grading_clusters_nodes_toward_corners(l_shape):
    corner = next(c for c in l_shape.corners if c.is_reentrant).z
    plain = l_shape.bdry_quad(400)
    graded = l_shape.bdry_quad(400, grading="corner")
    assert np.abs(graded.pts - corner).min() < np.abs(plain.pts - corner).min()


def test_grading_clusters_at_bc_junctions_including_the_one_it_need_not(
        split_edge_bc_junction):
    """Geometry grades where the ANGLE cannot rule out a singularity, plus every BC junction.
    That reaches the flat dir/neu junction at 0.5, which is genuinely singular, and also the
    right angle at 1, which is not -- geometry does not own the offset that would tell them
    apart, and over-grading is the safe direction."""
    plain = split_edge_bc_junction.bdry_quad(400)
    graded = split_edge_bc_junction.bdry_quad(400, grading="corner")
    for z in (0.5 + 0j, 1 + 0j):
        assert np.abs(graded.pts - z).min() < np.abs(plain.pts - z).min()
    assert graded.wts.sum() == pytest.approx(split_edge_bc_junction.perimeter, rel=PREC)


def test_grading_leaves_a_square_alone(square):
    """Every corner is reflection-closed and no boundary condition changes, so there is
    nothing to resolve and grading must not move a node."""
    plain = square.bdry_quad(200)
    graded = square.bdry_quad(200, grading="corner")
    assert np.allclose(graded.pts, plain.pts, atol=EXACT)


def test_grading_relocates_nodes_without_changing_the_measure(l_shape):
    """Grading is a node-placement choice, not a change of weight function. It stays in the
    arclength family, so the perimeter identity must survive it."""
    q = l_shape.bdry_quad(400, grading="corner")
    assert q.weight_family == "arclength"
    assert q.wts.sum() == pytest.approx(l_shape.perimeter, rel=PREC)
    assert np.all(q.wts > 0)
    assert (q.radial(0j) * q.wts).sum() == pytest.approx(2 * l_shape.area, rel=1e-6)


# --- singular families ------------------------------------------------------------------


def test_a_singular_family_is_representable(square):
    """The MPS layer builds Gauss-Jacobi rules around r^(2 nu - 2) and returns them as
    Quadrature. The type must carry that fact rather than silently look like an arclength
    rule -- lappy's weight_family exists because callers could not see it any other way."""
    plain = square.bdry_quad(64)
    singular = dataclasses.replace(plain, wts=plain.wts * 0.5, weight_family="even")
    assert singular.weight_family == "even"
    assert plain.weight_family == "arclength"
    assert singular.wts.sum() != pytest.approx(square.perimeter, rel=1e-3)


def test_integrate_is_literal_regardless_of_family(square):
    """integrate(f) is always sum(f(pts) * wts). It equals oint f ds only in the arclength
    family; for a singular rule it is oint f w ds, and the caller supplies f = integrand/w."""
    plain = square.bdry_quad(64)
    singular = dataclasses.replace(plain, wts=plain.wts * 0.5, weight_family="even")
    one = lambda z: np.ones_like(z, dtype=float)
    assert plain.integrate(one) == pytest.approx(plain.wts.sum(), rel=1e-12)
    assert singular.integrate(one) == pytest.approx(singular.wts.sum(), rel=1e-12)
    assert singular.integrate(one) == pytest.approx(0.5 * square.perimeter, rel=PREC)


def test_provenance_survives_a_change_of_family(square):
    """seg_idx/tau/normals are geometric and must not be disturbed by reweighting."""
    plain = square.bdry_quad(64)
    singular = dataclasses.replace(plain, wts=plain.wts * 0.5, weight_family="even")
    assert np.array_equal(singular.seg_idx, plain.seg_idx)
    assert np.allclose(singular.tau, plain.tau)
    assert np.allclose(singular.normals, plain.normals)
    assert np.allclose(singular.radial(0j), plain.radial(0j))


# --- immutability -----------------------------------------------------------------------


def test_quadrature_is_frozen(square):
    import dataclasses

    q = square.bdry_quad(32)
    assert dataclasses.is_dataclass(q)
    with pytest.raises(dataclasses.FrozenInstanceError):
        q.wts = np.zeros(1)
