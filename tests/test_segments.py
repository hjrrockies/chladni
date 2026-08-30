"""Segment contract: arclength parameterization, endpoints, frames, panel breaks.

The headline test here is `test_constant_speed`. Normalized-arclength parameterization
means |dp(tau)| == length identically, and that property is what every boundary quadrature
downstream rests on. lappy reached it only after replacing an interpolated inverse arclength
map with a Newton solve; the interpolant left |dp/dtau| - len at 8.6e-3 and capped ellipse
boundary integrals at ~7e-5 regardless of quadrature order.
"""
import numpy as np
import pytest

from chladni.geometry import ArcSegment, LineSegment, ParametricSegment, SplineSegment

from conftest import EXACT, PREC, ellipse_arclength, ellipse_perimeter

# --- the segments under test ------------------------------------------------------------

A, B = 2.0, 1.0  # ellipse semi-axes used throughout


def ellipse_seg(t0=0.0, t1=np.pi / 2, prec=PREC):
    return ParametricSegment(
        lambda t: A * np.cos(t) + 1j * B * np.sin(t),
        lambda t: -A * np.sin(t) + 1j * B * np.cos(t),
        t0,
        t1,
        prec=prec,
    )


def all_segments():
    return [
        LineSegment(0j, 3 + 4j),
        ArcSegment(1 + 1j, 2.0, 0.0, np.pi / 2),
        ArcSegment(0j, 1.0, np.pi / 3, -np.pi / 4),  # decreasing sweep
        ellipse_seg(),
        ellipse_seg(0.3, 2.9),
    ]


@pytest.fixture(params=all_segments(), ids=["line", "arc", "arc_cw", "ellipse", "ellipse_off"])
def seg(request):
    return request.param


# --- closed-form lengths ----------------------------------------------------------------


def test_line_length_exact():
    assert LineSegment(0j, 3 + 4j).length == pytest.approx(5.0, abs=EXACT)


@pytest.mark.parametrize("r,t0,t1", [(2.0, 0.0, np.pi / 2), (0.5, 1.0, 4.0), (3.0, 1.0, -2.0)])
def test_arc_length_exact(r, t0, t1):
    """An arc has a closed form; it must not go through the adaptive machinery."""
    assert ArcSegment(0j, r, t0, t1).length == pytest.approx(r * abs(t1 - t0), abs=EXACT)


def test_parametric_length_matches_elliptic_integral():
    """Quarter ellipse against mpmath, and the full ellipse against scipy's E."""
    quarter = ellipse_seg(0.0, np.pi / 2)
    assert quarter.length == pytest.approx(ellipse_arclength(A, B, 0, np.pi / 2), rel=PREC)

    full = ellipse_seg(0.0, 2 * np.pi)
    assert full.length == pytest.approx(ellipse_perimeter(A, B), rel=PREC)


# --- the arclength parameterization -----------------------------------------------------


def test_constant_speed(seg):
    """|dp(tau)| == length everywhere. The defining property of the parameterization."""
    tau = np.linspace(0, 1, 97)
    assert np.allclose(np.abs(seg.dp(tau)), seg.length, rtol=seg.prec)


def test_endpoints(seg):
    assert seg.p(0.0) == pytest.approx(seg.p0, abs=EXACT)
    assert seg.p(1.0) == pytest.approx(seg.p1, abs=EXACT)
    assert seg.T(0.0) == pytest.approx(seg.T0, abs=EXACT)
    assert seg.T(1.0) == pytest.approx(seg.T1, abs=EXACT)


def test_tau_is_arclength_not_just_monotone():
    """p(tau) sits at arclength tau*L along the curve, checked against mpmath.

    Constant speed plus the right endpoints already implies this, but an implementation
    can satisfy both while mapping tau to the wrong native t, so pin the point itself.
    """
    import mpmath as mp

    seg = ellipse_seg(0.0, np.pi / 2)
    L = ellipse_arclength(A, B, 0, np.pi / 2)
    for tau in (0.25, 0.5, 0.75):
        t_star = float(
            mp.findroot(lambda t: ellipse_arclength(A, B, 0, float(t)) - tau * L, tau * np.pi / 2)
        )
        expected = A * np.cos(t_star) + 1j * B * np.sin(t_star)
        assert seg.p(tau) == pytest.approx(expected, abs=PREC * L)


def test_tighter_prec_is_more_accurate():
    """prec is a real knob, not decoration: 1e-12 must beat 1e-4 on the same curve."""
    exact = ellipse_arclength(A, B, 0, np.pi / 2)
    loose = abs(ellipse_seg(0.0, np.pi / 2, prec=1e-4).length - exact)
    tight = abs(ellipse_seg(0.0, np.pi / 2, prec=1e-12).length - exact)
    assert tight <= loose
    assert tight < 1e-11


def test_vectorized(seg):
    """No Python-level loop over points: array in, array out, matching shape."""
    tau = np.linspace(0, 1, 11)
    for f in (seg.p, seg.dp, seg.T, seg.N):
        out = f(tau)
        assert out.shape == tau.shape
        assert np.iscomplexobj(out)


# --- frames -----------------------------------------------------------------------------


def test_normal_is_tangent_rotated(seg):
    tau = np.linspace(0, 1, 33)
    assert np.allclose(seg.N(tau), -1j * seg.T(tau))


def test_tangent_and_normal_are_unit(seg):
    tau = np.linspace(0, 1, 33)
    assert np.allclose(np.abs(seg.T(tau)), 1.0, rtol=seg.prec)
    assert np.allclose(np.abs(seg.N(tau)), 1.0, rtol=seg.prec)


def test_line_frame_exact():
    seg = LineSegment(0j, 3 + 4j)
    assert seg.T(0.5) == pytest.approx((3 + 4j) / 5, abs=EXACT)
    assert seg.N(0.5) == pytest.approx((4 - 3j) / 5, abs=EXACT)


def test_arc_normal_points_away_from_centre():
    """A CCW arc bounds the disk on its left, so -1j*T points radially outward."""
    seg = ArcSegment(1 + 1j, 2.0, 0.0, np.pi / 2)
    tau = np.linspace(0, 1, 17)
    radial = (seg.p(tau) - (1 + 1j)) / 2.0
    assert np.allclose(seg.N(tau), radial, atol=PREC)


# --- panel breaks -----------------------------------------------------------------------


def test_smooth_segments_have_no_breaks(seg):
    assert len(seg.break_taus) == 0


def test_spline_breaks_at_its_knots():
    """A degree-k spline is only C^(k-1) at its knots, so quadrature panels must break
    there. lappy measured 8.3e-11 at 96 knot-aligned nodes against 1.3e-06 at 128 with a
    single global panel."""
    pts = np.array([0, 1 + 0.4j, 2 - 0.3j, 3 + 0.6j, 4 + 0j], dtype=complex)
    seg = SplineSegment.interp_from_pts(pts)
    assert len(seg.break_taus) > 0
    assert np.all(seg.break_taus > 0) and np.all(seg.break_taus < 1)
    assert np.all(np.diff(seg.break_taus) > 0)  # sorted, unique


# --- queries ----------------------------------------------------------------------------


def test_dist_to_line():
    seg = LineSegment(0j, 2 + 0j)
    assert seg.dist(1 + 1j) == pytest.approx(1.0, abs=EXACT)   # perpendicular foot interior
    assert seg.dist(3 + 0j) == pytest.approx(1.0, abs=EXACT)   # past the end, clamps to p1
    assert seg.dist(-1j) == pytest.approx(1.0, abs=EXACT)      # past the start, clamps to p0


def test_dist_to_arc():
    seg = ArcSegment(0j, 1.0, 0.0, np.pi / 2)
    assert seg.dist(0j) == pytest.approx(1.0, abs=PREC)
    assert seg.dist(2 + 0j) == pytest.approx(1.0, abs=PREC)


def test_intersection_of_crossing_lines():
    a = LineSegment(-1 + 0j, 1 + 0j)
    b = LineSegment(-1j, 1j)
    assert a.intersects(b)
    hits = a.intersection(b)
    assert len(hits) == 1
    assert hits[0] == pytest.approx(0j, abs=EXACT)


def test_disjoint_segments_do_not_intersect():
    a = LineSegment(0j, 1 + 0j)
    b = LineSegment(2j, 1 + 2j)
    assert not a.intersects(b)
    assert len(a.intersection(b)) == 0


def test_touching_at_an_endpoint_counts_as_intersecting():
    a = LineSegment(0j, 1 + 0j)
    b = LineSegment(1 + 0j, 1 + 1j)
    assert a.intersects(b)
    assert a.intersection(b)[0] == pytest.approx(1 + 0j, abs=EXACT)


def test_simple_segments_are_simple(seg):
    assert seg.is_simple


def test_self_crossing_parametric_is_not_simple():
    """A lemniscate crosses itself at the origin."""
    seg = ParametricSegment(
        lambda t: np.sin(t) + 1j * np.sin(t) * np.cos(t),
        lambda t: np.cos(t) + 1j * (np.cos(2 * t)),
        0.0,
        2 * np.pi,
    )
    assert not seg.is_simple


# --- boundary conditions ----------------------------------------------------------------


def test_bc_encoding():
    assert LineSegment(0j, 1 + 0j, bc="dir").bc == 0.0
    assert LineSegment(0j, 1 + 0j, bc="neu").bc == 1.0
    assert LineSegment(0j, 1 + 0j, bc=0.25).bc == 0.25


def test_bc_defaults_to_dirichlet():
    assert LineSegment(0j, 1 + 0j).bc == 0.0


@pytest.mark.parametrize("bad", ["robin", -0.5, 1.5, 1j, None])
def test_bad_bc_rejected(bad):
    with pytest.raises((ValueError, TypeError)):
        LineSegment(0j, 1 + 0j, bc=bad)
