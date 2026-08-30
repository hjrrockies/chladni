"""Fundamental solutions: `Y_m(k*r)` multipoles at exterior source points.

Two things here are not really about the mathematics.

`Y_0` and `Y_0' = -Y_1` get a fast path, because lappy measured `yv(0, .)` at 42x `y0` and
`yvp(0, .)` at 85x `y1` -- 1.98 s of a 5.88 s L-shape solve, for 36 of 159 columns. Speed is
the only difference allowed, and this file checks that.

Sources must be OUTSIDE Omega, and that is an exception rather than a filter. lappy's
docstring is worth citing rather than re-deriving: an interior pole breaks the MPS premise,
voids the Moler-Payne certificate, and makes the tension BETTER rather than worse. Measured
there, 24 of 240 sources inside took the tension background from 5e-2 to 3e-7, which made the
acceptance test swallow everything, produced a spurious eigenvalue and hid a real one.
Silence is the failure mode, so this raises.
"""
import numpy as np
import pytest

from conftest import EXACT, L_SHAPE, mp_bessely, mp_bessely_prime
from chladni.geometry import Polygon
from chladni.helmholtz import Fundamental, SourcesInsideDomain

K = 3.7


def _residual(fam, k, pts, h=1e-3):
    def u(z):
        return fam.values(k, z)

    lap = np.zeros_like(u(pts))
    for step in (h, 1j * h):
        lap = lap + (-u(pts + 2 * step) + 16 * u(pts + step) - 30 * u(pts)
                     + 16 * u(pts - step) - u(pts - 2 * step)) / (12 * h**2)
    return np.abs(lap + k**2 * u(pts)).max() / (k**2 * np.abs(u(pts)).max())


# --- the functions ------------------------------------------------------------------------

def test_a_monopole_is_y_zero():
    src = np.array([2.0 + 0.5j])
    fam = Fundamental(src, radius=1.0)
    pts = np.array([0.0 + 0j, 0.3 + 0.2j, -0.5 + 0.1j])
    got = fam.values(K, pts)[:, 0] * np.exp(fam.log_scale(K)[0])
    for i, p in enumerate(pts):
        assert got[i] == pytest.approx(mp_bessely(0, K * abs(p - src[0])), rel=1e-12)


def test_a_multipole_is_y_m_times_the_angular_factor():
    src = np.array([2.0 + 0.5j])
    fam = Fundamental(src, order=3, radius=1.0)
    assert len(fam) == 5                      # Y0, Y1cos, Y1sin, Y2cos, Y2sin
    p = np.array([0.1 + 0.4j])
    d = p - src[0]
    r, th = float(abs(d[0])), float(np.angle(d[0]))
    got = fam.values(K, p)[0] * np.exp(fam.log_scale(K))
    want = [mp_bessely(0, K * r),
            mp_bessely(1, K * r) * np.cos(th), mp_bessely(1, K * r) * np.sin(th),
            mp_bessely(2, K * r) * np.cos(2 * th), mp_bessely(2, K * r) * np.sin(2 * th)]
    assert got == pytest.approx(want, rel=1e-11)


@pytest.mark.parametrize("order", [1, 2, 4])
def test_every_column_solves_the_helmholtz_equation(order):
    fam = Fundamental(np.array([2.0 + 0.5j, -1.5 + 1.0j]), order=order, radius=1.0)
    pts = np.array([0.0 + 0j, 0.3 + 0.2j, -0.4 + 0.35j, 0.2 - 0.3j])
    assert _residual(fam, K, pts) < 5e-9


def test_the_gradient_matches_fourth_order_central_differences():
    fam = Fundamental(np.array([2.0 + 0.5j, -1.5 + 1.0j]), order=3, radius=1.0)
    pts = np.array([0.0 + 0j, 0.3 + 0.2j, -0.4 + 0.35j])
    h = 1e-4
    g = fam.grad(K, pts)
    for axis, step in ((0, h), (1, 1j * h)):
        fd = (-fam.values(K, pts + 2 * step) + 8 * fam.values(K, pts + step)
              - 8 * fam.values(K, pts - step) + fam.values(K, pts - 2 * step)) / (12 * h)
        got = g.real if axis == 0 else g.imag
        assert np.max(np.abs(got - fd)) < 1e-10 * np.abs(g).max()


def test_the_derivative_of_y_zero_is_minus_y_one():
    """`Y_0' = -Y_1` exactly, which is what makes the order-zero fast path a fast path."""
    fam = Fundamental(np.array([0j]), radius=1.0)
    r = np.array([0.3, 1.0, 4.0])
    g = fam.grad(K, r + 0j)[:, 0] * np.exp(fam.log_scale(K)[0])
    for i, ri in enumerate(r):
        assert g[i].real == pytest.approx(K * mp_bessely_prime(0, K * ri), rel=1e-11)


def test_the_order_zero_fast_path_agrees_with_the_general_order_path():
    """Dispatch is on the order VALUES, not on the `order` attribute, so a family assembled
    some other way still takes it. Checked by comparing an order-1 family against the m = 0
    column of an order-2 one, which cannot use the fast path."""
    src = np.array([2.0 + 0.5j, -1.0 + 2.0j])
    fast = Fundamental(src, order=1, radius=1.0)
    slow = Fundamental(src, order=2, radius=1.0)
    pts = np.array([0.0 + 0j, 0.3 + 0.2j, -0.4 + 0.35j])
    a = fast.values(K, pts) * np.exp(fast.log_scale(K))[None, :]
    b = slow.values(K, pts, cols=[0, 3]) * np.exp(slow.log_scale(K, cols=[0, 3]))[None, :]
    assert np.allclose(a, b, rtol=0.0, atol=1e-13 * np.abs(a).max())


def test_multipole_columns_are_scaled_to_a_sane_range():
    """Y_m blows up at small argument the way J_mu underflows at large order; the standoff
    radius is what sets the scale, and it is a geometric constant, not a point-set quantity."""
    fam = Fundamental(np.array([2.0 + 0j]), order=12, radius=1.0)
    pts = 2.0 + np.exp(1j * np.linspace(0, 2 * np.pi, 400)) * np.linspace(1.0, 3.0, 400)
    cmax = np.abs(fam.values(K, pts)).max(axis=0)
    assert cmax.max() / cmax.min() < 60.0


# --- placement ------------------------------------------------------------------------------

def test_offset_puts_sources_along_the_outward_normal():
    poly = Polygon(L_SHAPE)
    q = poly.bdry_quad(120)
    fam = Fundamental.offset(q, 0.15)
    assert len(fam.origins) == len(q)
    assert np.allclose(fam.origins, q.pts + 0.15 * q.normals, rtol=0, atol=EXACT)
    assert fam.radius == pytest.approx(0.15, abs=EXACT)


SLIT = np.array([0, 4, 4 + 3j, 2.1 + 3j, 2.1 + 1j, 1.9 + 1j, 1.9 + 3j, 3j], dtype=complex)


def test_offset_refuses_sources_that_landed_inside_the_domain():
    """A 0.2-wide slit cut down from the top. The outward normal on a slit wall points across
    the slit, so any offset past its width lands in Omega on the far side -- 134 of 600
    sources at offset 0.3, and none at 0.15.

    lappy dropped such sources silently. Measured there: 24 of 240 inside took the tension
    background from 5e-2 to 3e-7, which made every candidate pass, produced a spurious
    eigenvalue and hid a real one. Silence is the failure mode."""
    poly = Polygon(SLIT)
    q = poly.bdry_quad(600)
    Fundamental.offset(q, 0.15, domain=poly)          # safe, and stays silent
    with pytest.raises(SourcesInsideDomain, match="inside the domain"):
        Fundamental.offset(q, 0.3, domain=poly)


def test_check_exterior_is_available_on_a_hand_built_family():
    poly = Polygon(L_SHAPE)
    outside = Fundamental(np.array([-1.0 - 1.0j, 3.0 + 3.0j]), radius=1.0)
    outside.check_exterior(poly)
    with pytest.raises(SourcesInsideDomain, match="inside"):
        Fundamental(np.array([0.5 + 0.5j]), radius=1.0).check_exterior(poly)


# --- lightning clustering -----------------------------------------------------------------------

def test_lightning_spacings_match_the_published_formula():
    """`s_j = L*exp(-sigma*(sqrt(n) - sqrt(j)))`, j = 1..n, along the exterior bisector."""
    poly = Polygon(L_SHAPE)
    c = [c for c in poly.corners if c.is_reentrant][0]
    n, sig = 16, 4.0
    fam = Fundamental.lightning(c, n, sigma=sig, L=0.5)
    d = np.abs(fam.origins - c.z)
    j = np.arange(1, n + 1)
    assert np.allclose(np.sort(d), np.sort(0.5 * np.exp(-sig * (np.sqrt(n) - np.sqrt(j)))),
                       rtol=1e-13)


def test_lightning_defaults_to_sigma_four_not_lappys_one():
    """lappy's formula matches the published Gopal-Trefethen form exactly and its DEFAULTS do
    not: `sigma=1` against `sigma ~ 4`. Measured at n = 25 and L = 1, the innermost pole sits
    at 1.1e-07 with sigma = 4 and at 1.8e-02 with sigma = 1 -- five orders of magnitude of
    reach toward the corner, which is the entire point of clustering."""
    poly = Polygon(L_SHAPE)
    c = [c for c in poly.corners if c.is_reentrant][0]
    n = 25
    innermost = np.abs(Fundamental.lightning(c, n, L=1.0).origins - c.z).min()
    assert innermost == pytest.approx(np.exp(-4.0 * (np.sqrt(n) - 1)), rel=1e-12)
    slow = np.abs(Fundamental.lightning(c, n, sigma=1.0, L=1.0).origins - c.z).min()
    # exp(-16) = 1.1e-07 against exp(-4) = 1.8e-02: five orders of magnitude of reach.
    assert innermost == pytest.approx(1.125e-7, rel=1e-3)
    assert slow == pytest.approx(1.832e-2, rel=1e-3)


def test_lightning_points_into_the_exterior():
    poly = Polygon(L_SHAPE)
    for c in poly.corners:
        fam = Fundamental.lightning(c, 10, L=0.5 * c.clearance)
        assert not poly.contains(fam.origins).any()


def test_the_taper_drops_the_outer_end_of_the_cluster():
    """Gopal-Trefethen taper the OUTER end, where the cluster would otherwise duplicate the
    ambient source layer and put near-identical columns into the pencil. How much to taper is
    a coordination question between this cluster and that layer, so it is a parameter here
    and a decision for the (deferred) planner."""
    poly = Polygon(L_SHAPE)
    c = [c for c in poly.corners if c.is_reentrant][0]
    full = Fundamental.lightning(c, 20, L=0.5)
    cut = Fundamental.lightning(c, 20, L=0.5, taper=0.25)
    assert len(cut.origins) == 15
    d_full, d_cut = np.abs(full.origins - c.z), np.abs(cut.origins - c.z)
    assert d_cut.max() < d_full.max()          # the OUTER poles went
    assert d_cut.min() == pytest.approx(d_full.min(), rel=1e-13)


def test_lightning_sources_carry_their_own_standoff():
    """The scale needs the distance from the source to Omega, not to the corner -- on the
    exterior bisector those differ by sin of half the exterior angle."""
    poly = Polygon(L_SHAPE)
    c = [c for c in poly.corners if c.is_reentrant][0]
    fam = Fundamental.lightning(c, 8, L=0.4)
    d = np.abs(fam.origins - c.z)
    assert np.allclose(fam.radius, d * np.sin((2 * np.pi - c.theta) / 2), rtol=1e-13)


# --- shapes and columns -----------------------------------------------------------------------

def test_the_matrix_is_points_by_columns():
    fam = Fundamental(np.array([2 + 0j, 3 + 1j, -2 + 0j]), order=2, radius=1.0)
    assert len(fam) == 9
    pts = np.linspace(-0.5, 0.5, 7) + 0.1j
    assert fam.values(K, pts).shape == (7, 9)
    assert fam.grad(K, pts).shape == (7, 9)


def test_cols_preserves_the_requested_order():
    fam = Fundamental(np.array([2 + 0j, 3 + 1j]), order=2, radius=1.0)
    pts = np.linspace(-0.5, 0.5, 4) + 0.1j
    want = np.array([5, 1, 0])
    assert np.array_equal(fam.values(K, pts, cols=want), fam.values(K, pts)[:, want])


def test_singular_exponents_report_the_sources_as_regular_in_omega():
    """A fundamental solution is singular at its pole, which is OUTSIDE Omega -- so from the
    point of view of a corner-graded boundary rule it is regular everywhere that matters."""
    fam = Fundamental(np.array([2 + 0j, 3 + 1j]), order=2, radius=1.0)
    s = fam.singular_exponents()
    assert np.all(np.isnan(s.origin))
    assert np.all(s.exponent == 0.0)
