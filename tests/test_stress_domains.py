"""Domain classes the rest of the suite does not reach.

The other files test shapes chosen to have clean closed forms. That leaves whole classes
untested, and two of them turned out to matter: a shape optimizer generates polygons with
hundreds of vertices, and corner construction was O(n^2) there -- 534 ms for a 400-gon
against 6 ms for everything else in the lifecycle. The rest of this file pins the classes
that were merely unexamined: mixed line/arc boundaries, near-slit and sliver corners, high
aspect ratio, and domains far from the origin or at extreme scale.
"""
import time
import warnings

import numpy as np
import pytest

from chladni.geometry import (
    ArcSegment,
    Boundary,
    Domain,
    LineSegment,
    Polygon,
    L_shape,
    rect,
    reg_ngon,
)

from conftest import EXACT, PREC

# --- mixed line and arc boundaries ------------------------------------------------------


def stadium(L=2.0, H=1.0):
    return Domain(Boundary([
        LineSegment(-L / 2 - 1j * H / 2, L / 2 - 1j * H / 2),
        ArcSegment(L / 2 + 0j, H / 2, -np.pi / 2, np.pi / 2),
        LineSegment(L / 2 + 1j * H / 2, -L / 2 + 1j * H / 2),
        ArcSegment(-L / 2 + 0j, H / 2, np.pi / 2, 3 * np.pi / 2),
    ]))


def disk_sector(r=1.0, omega=np.pi / 2):
    return Domain(Boundary([
        LineSegment(0j, r + 0j),
        ArcSegment(0j, r, 0.0, omega),
        LineSegment(r * np.exp(1j * omega), 0j),
    ]))


def test_stadium_is_exact():
    L, H = 2.0, 1.0
    s = stadium(L, H)
    assert s.area == pytest.approx(L * H + np.pi * (H / 2) ** 2, abs=PREC)
    assert s.perimeter == pytest.approx(2 * L + np.pi * H, abs=PREC)
    assert s.inradius == pytest.approx(H / 2, abs=PREC)


def test_stadium_has_no_corners():
    """Every line/arc junction is tangential. A corner test keyed on the segment TYPE
    changing, rather than on the turning angle, would report four corners here."""
    assert stadium().corners == ()


def test_disk_sector_corners():
    """The mirror case: line/arc junctions that are genuine corners."""
    s = disk_sector(1.0, np.pi / 2)
    assert s.area == pytest.approx(np.pi / 4, abs=PREC)
    assert s.perimeter == pytest.approx(2 + np.pi / 2, abs=PREC)
    assert len(s.corners) == 3
    assert np.allclose([c.omega for c in s.corners], np.pi / 2, atol=PREC)


def test_half_disk_sector_has_a_smooth_pi_junction():
    """omega = pi puts the two radii in line: three segments, two corners. This is the
    shape whose corner arrays desynced in lappy."""
    s = disk_sector(1.0, np.pi)
    assert len(s.outer.segments) == 3
    assert len(s.corners) == 2


# --- near-slit and sliver corners -------------------------------------------------------


def wedge(frac):
    """Four points on the unit circle spanning 2*pi*(1-frac): a reentrant corner at the
    origin of angle 2*pi*(1-frac), approaching the slit limit as frac -> 0. Area is
    exactly sin(omega/2)."""
    omega = 2 * np.pi * (1 - frac)
    return Polygon(np.array([0, 1, np.exp(1j * omega / 2), np.exp(1j * omega)]))


@pytest.mark.parametrize("frac", [0.25, 0.10, 0.02])
def test_near_slit_corner(frac):
    omega = 2 * np.pi * (1 - frac)
    d = wedge(frac)
    assert d.area == pytest.approx(np.sin(omega / 2), abs=PREC)
    reentrant = [c for c in d.corners if c.is_reentrant]
    assert len(reentrant) == 1
    assert reentrant[0].omega == pytest.approx(omega, abs=PREC)
    assert reentrant[0].clearance > 0
    assert np.isfinite(reentrant[0].branch_angle)


def test_near_slit_branch_ray_still_escapes():
    """nu = pi/omega is 0.51 at frac=0.02, right at lappy's admissibility threshold. The
    exterior wedge is only 0.04*pi wide, so the branch ray has very little room."""
    d = wedge(0.02)
    for c in d.corners:
        eps = np.geomspace(1e-9, 10 * d.diameter, 80)
        assert not np.any(d.contains(c.z + eps * np.exp(1j * c.branch_angle)))


@pytest.mark.parametrize("eps", [0.05, 0.005])
def test_sliver_triangle(eps):
    """The opposite extreme: an interior angle near zero, where nu = pi/eps is enormous."""
    d = Polygon(np.array([0, 1, np.exp(1j * eps)]))
    assert d.area == pytest.approx(0.5 * np.sin(eps), abs=PREC)
    assert min(c.omega for c in d.corners) == pytest.approx(eps, abs=PREC)
    assert sum(c.omega for c in d.corners) == pytest.approx(np.pi, abs=PREC)


# --- high aspect ratio ------------------------------------------------------------------


@pytest.mark.parametrize("ar", [10, 100, 1000])
def test_thin_rectangles(ar):
    """Rejection sampling draws from the bounding box, so a thin domain is where an
    acceptance-rate assumption would break."""
    r = rect(ar, 1.0)
    assert r.area == pytest.approx(ar, abs=EXACT)
    assert r.inradius == pytest.approx(0.5, abs=PREC)
    assert r.diameter == pytest.approx(np.hypot(ar, 1.0), abs=PREC)
    pts = r.interior_pts(400, rng=0)
    assert len(pts) == 400 and np.all(r.contains(pts))
    q = r.bdry_quad(2000)
    assert (q.radial(0j) * q.wts).sum() == pytest.approx(2 * r.area, rel=1e-8)


def test_l_shape_inradius_is_set_by_the_reentrant_corner():
    """2 - sqrt(2), not 1/2. The largest inscribed disk is not the one that fits the arm:
    it sits on the diagonal, pinned by the reentrant corner."""
    assert L_shape().inradius == pytest.approx(2 - np.sqrt(2), abs=PREC)


# --- many vertices, which is what a shape optimizer generates ----------------------------


@pytest.mark.parametrize("n", [50, 200, 500])
def test_large_polygons_stay_exact(n):
    p = reg_ngon(n)
    assert p.area == pytest.approx(n / 2 * np.sin(2 * np.pi / n), abs=PREC)
    assert p.perimeter == pytest.approx(2 * n * np.sin(np.pi / n), abs=PREC)
    assert p.inradius == pytest.approx(np.cos(np.pi / n), abs=PREC)
    assert len(p.corners) == n
    q = p.bdry_quad(4 * n)
    assert (q.radial(0j) * q.wts).sum() == pytest.approx(2 * p.area, rel=1e-8)


@pytest.mark.slow
def test_corner_construction_is_not_quadratic():
    """Regression guard with wide margin. Per-corner clearance and blocked-arc scans put a
    400-gon at 534 ms; batched it is ~7 ms. The budget catches a return to the old
    behaviour without being sensitive to machine load."""
    n = 400
    v = np.exp(2j * np.pi * np.arange(n) / n) * (1 + 0.15 * np.cos(10 * np.pi * np.arange(n) / n))
    d = Polygon(v)
    t = time.perf_counter()
    corners = d.corners
    elapsed = time.perf_counter() - t
    assert len(corners) == n
    assert elapsed < 0.15, f"corners took {elapsed*1e3:.0f} ms for a {n}-gon"


# --- position and scale -----------------------------------------------------------------


@pytest.mark.parametrize("offset", [0, 1e3 + 1e3j, 1e6])
def test_metrics_are_invariant_under_translation(offset):
    """Tolerances are derived from prec and a length scale, so they have to travel. A suite
    that only ever works near the origin cannot see a hard-coded one."""
    base = L_shape()
    d = base.translated(offset)
    assert d.area == pytest.approx(base.area, rel=1e-10)
    assert d.perimeter == pytest.approx(base.perimeter, rel=1e-10)
    assert d.diameter == pytest.approx(base.diameter, rel=1e-10)
    assert d.inradius == pytest.approx(base.inradius, rel=1e-8)
    assert np.allclose(sorted(c.omega for c in d.corners),
                       sorted(c.omega for c in base.corners), atol=1e-9)


@pytest.mark.parametrize("scale", [1e-6, 1e-4, 1.0, 1e4, 1e6])
def test_metrics_are_invariant_under_scale(scale):
    base = L_shape()
    d = base.scaled(scale)
    assert d.area == pytest.approx(base.area * scale**2, rel=1e-10)
    assert d.perimeter == pytest.approx(base.perimeter * scale, rel=1e-10)
    assert d.diameter == pytest.approx(base.diameter * scale, rel=1e-10)
    assert d.inradius == pytest.approx(base.inradius * scale, rel=1e-8)
    assert len(d.corners) == len(base.corners)


@pytest.mark.parametrize("offset,scale", [(1e3, 1e-3), (1e6, 1e-4), (1e6, 1.0)])
def test_small_domain_far_from_the_origin_degrades_predictably(offset, scale):
    """The one place prec cannot be delivered, and it is arithmetic, not implementation.

    A feature of size `scale` at coordinate `offset` is resolved to about
    eps * offset / scale in relative terms -- at offset 1e6 and scale 1e-4 that is 8e-7, so
    a 1e-8 promise is not available in float64 however the code is written. Measured:
    1.2e-11 at (1e3, 1e-3) and 2.7e-7 at (1e6, 1e-4), both inside the bound.

    There is no post-hoc repair. The bits go when the vertices are ROUNDED into their
    far-from-origin coordinates, so translating the finished domain back to the origin
    recovers nothing, and `normalized()` -- which scales about the origin -- leaves the
    offset-to-size ratio untouched and so does not help either. The fix belongs to the
    caller and belongs upstream: build the domain about its own centre and carry the
    offset separately.
    """
    from chladni.geometry import PrecisionWarning

    base = L_shape()
    with warnings.catch_warnings():
        # the domain announces this shortfall itself; that behaviour is tested separately
        warnings.simplefilter("ignore", PrecisionWarning)
        d = base.scaled(scale).translated(offset)
    limit = 10 * np.finfo(float).eps * abs(offset) / (scale * base.diameter)
    tol = max(1e-10, limit)
    assert d.perimeter == pytest.approx(base.perimeter * scale, rel=tol)
    assert d.area == pytest.approx(base.area * scale**2, rel=tol)
    assert len(d.corners) == len(base.corners)
    # building at the origin and scaling there keeps every digit; the offset is what costs
    assert base.scaled(scale).perimeter == pytest.approx(base.perimeter * scale, rel=1e-12)


@pytest.mark.parametrize("scale", [1e-4, 1e4])
def test_quadrature_identities_hold_at_extreme_scale(scale):
    d = L_shape().scaled(scale)
    q = d.bdry_quad(600)
    assert q.wts.sum() == pytest.approx(d.perimeter, rel=PREC)
    assert (q.radial(0j) * q.wts).sum() == pytest.approx(2 * d.area, rel=1e-8)


# --- boundary conditions across the harder shapes ---------------------------------------


@pytest.mark.parametrize("bc", ["dir", "neu", 0.3])
def test_robin_and_mixed_conditions_survive_a_reentrant_corner(bc):
    d = L_shape(bc=bc)
    q = d.bdry_quad(400)
    assert np.all(q.bc == (0.0 if bc == "dir" else 1.0 if bc == "neu" else bc))
    assert len([c for c in d.corners if c.is_reentrant]) == 1


# --- honesty about precision ------------------------------------------------------------


def test_shipped_domains_claim_no_precision_they_lack():
    """No false positives: everything in the library must clear the default prec easily."""
    from chladni.geometry import GWW1, annulus, disk, plus_shape

    for d in (L_shape(), GWW1(), plus_shape(), disk(), annulus(), reg_ngon(500), rect(1000, 1)):
        assert d.achievable_prec <= d.prec
        assert d.achievable_prec < 1e-12


@pytest.mark.parametrize("offset,scale", [(1e3, 1e-3), (1e6, 1e-4)])
def test_achievable_prec_bounds_the_error_that_actually_occurs(offset, scale):
    """The estimate has to be an upper bound, or it is worse than no estimate at all."""
    from chladni.geometry import PrecisionWarning

    base = L_shape()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PrecisionWarning)
        d = base.scaled(scale).translated(offset)
    err = abs(d.perimeter - base.perimeter * scale) / (base.perimeter * scale)
    assert err <= d.achievable_prec


def test_far_offset_domain_warns():
    from chladni.geometry import PrecisionWarning

    with pytest.warns(PrecisionWarning, match="from the origin"):
        L_shape().scaled(1e-4).translated(1e6)


def test_near_degenerate_edge_warns():
    """The case a shape optimizer produces: a well-placed domain with one tiny edge."""
    from chladni.geometry import PrecisionWarning

    v = np.array([0, 1, 1 + 1e-11j, 1 + 1j, 1j], dtype=complex)
    with pytest.warns(PrecisionWarning, match="shortest segment"):
        Polygon(v)


def test_validate_false_does_not_suppress_the_precision_warning():
    """`validate` skips the O(n^2) checks; it is not a licence to misreport precision.
    Gating on it would also have let `translated()` -- the operation that creates the
    problem -- rebuild silently."""
    from chladni.geometry import PrecisionWarning

    v = np.array([0, 1, 1 + 1e-11j, 1 + 1j, 1j], dtype=complex)
    with pytest.warns(PrecisionWarning):
        Polygon(v, validate=False)


def test_precision_warning_is_filterable_on_its_own():
    """douse trials thousands of shapes and needs to silence this specifically, not with a
    blanket filter that would hide everything else."""
    import warnings

    from chladni.geometry import PrecisionWarning

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.simplefilter("ignore", PrecisionWarning)
        L_shape().scaled(1e-4).translated(1e6)
    assert caught == []


def test_a_loose_prec_request_is_honoured_without_a_warning():
    """The warning is about the gap, not about the geometry: ask for less and it is quiet."""
    import warnings

    v = np.array([0, 1, 1 + 1e-11j, 1 + 1j, 1j], dtype=complex)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        d = Polygon(v, prec=1e-3)
    assert caught == []
    assert d.achievable_prec <= 1e-3


# --- branch cuts for surrounded corners --------------------------------------------------


def cut_stays_outside(d, c, per_leg=2000):
    """Sample the whole cut -- every leg of the polyline plus a long final ray."""
    path = c.branch_path(20 * d.diameter)
    u = np.linspace(0.0, 1.0, per_leg)[1:-1]
    z = np.concatenate([a + u * (b - a) for a, b in zip(path[:-1], path[1:])])
    return not bool(d.contains(z).any())


def test_spiral_corners_all_get_a_branch_cut():
    """Half the spiral's corners are surrounded: their exterior is a channel that turns,
    so no straight ray escapes. Without a polyline cut they would carry none at all."""
    from chladni.geometry import spiral

    d = spiral()
    assert all(np.isfinite(c.branch_angle) for c in d.corners)
    assert all(c.admits_exterior_branch_cut for c in d.corners)
    walled = [c for c in d.corners if c.branch_vertices]
    assert len(walled) > 0, "the spiral is supposed to need polyline cuts"


def test_every_spiral_cut_stays_out_of_the_domain():
    """The property that matters: a cut that re-enters Omega makes the expansion it
    supports discontinuous inside the domain."""
    from chladni.geometry import spiral

    d = spiral()
    for c in d.corners:
        assert cut_stays_outside(d, c), f"cut from {c.z} re-enters the domain"


def test_simple_domains_need_no_polyline(l_shape, square, eq_tri):
    """The walk is a fallback and must stay one: a straight ray works everywhere here, so
    no wall-following should be attempted."""
    from chladni.geometry import plus_shape

    for d in (l_shape, square, eq_tri, plus_shape()):
        assert all(c.branch_vertices == () for c in d.corners)


def test_branch_path_shape():
    from chladni.geometry import spiral

    d = spiral()
    for c in d.corners:
        path = c.branch_path(1.0)
        assert len(path) == len(c.branch_vertices) + 2
        assert path[0] == pytest.approx(c.z, abs=EXACT)
        if c.branch_vertices:
            assert path[1] == pytest.approx(c.branch_vertices[0], abs=EXACT)


def test_branch_path_refuses_when_there_is_no_cut():
    from chladni.geometry import Corner

    c = Corner(z=0j, seg_out=0, seg_in=1, omega=1.0, t_in=1 + 0j, t_out=1j,
               bc_in=0.0, bc_out=0.0, clearance=1.0, branch_angle=float("nan"))
    with pytest.raises(ValueError, match="no branch cut"):
        c.branch_path(1.0)


def test_hole_corners_do_not_attempt_a_walk(square_with_square_hole):
    """A hole corner has no valid cut for topological reasons, so there is nothing to
    wall-follow towards; the bisector is reported as a local direction and nothing else."""
    for c in square_with_square_hole.corners:
        if c.on_hole:
            assert c.branch_vertices == ()
            assert not c.admits_exterior_branch_cut


def test_spiral_is_a_well_formed_domain():
    from chladni.geometry import spiral

    d = spiral()
    assert d.is_valid and d.outer.orientation == 1 and d.area > 0
    q = d.bdry_quad(2000)
    assert (q.radial(0j) * q.wts).sum() == pytest.approx(2 * d.area, rel=1e-8)
