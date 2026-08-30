"""Corner-adapted Fourier-Bessel functions.

What this file defends, in order of how much it costs to get wrong:

1. The angle lift is anchored to the branch cut, so a point on the phi = 0 ray gives
   STRUCTURALLY exact 0.0 for the Dirichlet family. The obvious
   `np.angle((z - c.z)/c.t_out) % (2*np.pi)` is wrong -- a point that should sit at phi = 0
   comes out of np.angle as -1e-17 and wraps to 6.283 -- and the symptom is a Dirichlet
   basis function evaluating to 0.11 on the ray where it must vanish.
2. Every column solves the Helmholtz equation exactly. That is the entire premise of MPS;
   a column that does not is not a particular solution and voids any certificate computed
   from it.
3. The branch discontinuity lies on the cut geometry hands us, and nowhere in Omega.
4. The gradient is the analytic one, not a plausible rearrangement of it.

Tolerances are measured. Complex-step differentiation does NOT work here (relative error
8.5e3): `scipy.special.jv` does not deliver the imaginary part of a 1e-20 perturbation to
relative accuracy, and `arctan2` does not complexify. Fourth-order central differences at
h = 1e-4 reach 4.3e-13, which is what this file uses.
"""
import numpy as np
import pytest

from conftest import EXACT, L_SHAPE
from chladni.geometry import Polygon
from chladni.helmholtz import FourierBessel
from chladni.helmholtz.corners import CornerModes

K = 7.3


def wedge(theta=1.4 * np.pi, bc_out=0.0, bc_in=0.0, n=5, radius=1.0):
    """A bare corner at the origin opening along the positive real axis."""
    return FourierBessel(
        origin=0j, t_out=1 + 0j, radius=radius,
        modes=CornerModes.from_wedge(theta, bc_out, bc_in),
        degrees=CornerModes.from_wedge(theta, bc_out, bc_in).degrees(n),
    )


def unscaled(fam, k, pts, cols=None):
    """Columns with the per-column scale put back, for tests that want the mathematical
    function rather than the conditioned one. Only safe where exp(sigma) is representable."""
    return fam.values(k, pts) * np.exp(fam.log_scale(k))


# --- the boundary conditions, which are the whole point ---------------------------------------

def test_the_dirichlet_family_is_exactly_zero_on_the_outgoing_ray():
    """`sin(mu*0)` and nothing else. If this is not `== 0.0` the lift is wrapping."""
    fam = wedge(bc_out=0.0, bc_in=0.0)
    r = np.linspace(0.01, 0.9, 40)
    assert np.all(fam.values(K, r + 0j) == 0.0)


def test_the_dirichlet_family_vanishes_on_the_incoming_ray_to_roundoff():
    """Only to roundoff, and that asymmetry is real: `sin(mu*theta)` carries theta's own
    error, while `sin(mu*0)` cannot. A test demanding exact zero must target bc_out's ray."""
    theta = 1.4 * np.pi
    fam = wedge(theta, 0.0, 0.0)
    r = np.linspace(0.01, 0.9, 40)
    pts = r * np.exp(1j * theta)
    got = np.abs(fam.values(K, pts))
    scale = np.abs(fam.values(K, r * np.exp(1j * theta / 2))).max()
    assert got.max() < 1e-15 * scale
    assert got.max() > 0.0


def test_the_neumann_family_has_zero_normal_derivative_on_both_rays():
    theta = 1.4 * np.pi
    fam = wedge(theta, 1.0, 1.0)
    r = np.linspace(0.05, 0.9, 40)
    for phi, nvec in ((0.0, -1j), (theta, 1j * np.exp(1j * theta))):
        pts = r * np.exp(1j * phi)
        dn = fam.deriv(K, pts, np.full(len(r), nvec))
        assert np.abs(dn).max() < 1e-13 * max(1.0, np.abs(fam.values(K, pts)).max())


@pytest.mark.parametrize("bc_out,bc_in", [(0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0)])
def test_every_zaremba_combination_satisfies_its_own_rays(bc_out, bc_in):
    """The four-row table, exercised through the real evaluator rather than through the
    angular factor alone -- so the lift, the exponents and the derivative all have to agree."""
    theta = 1.4 * np.pi
    fam = wedge(theta, bc_out, bc_in)
    r = np.linspace(0.05, 0.9, 30)
    ref = np.abs(fam.values(K, r * np.exp(1j * theta / 2))).max()
    for phi, bc, nvec in ((0.0, bc_out, -1j), (theta, bc_in, 1j * np.exp(1j * theta))):
        pts = r * np.exp(1j * phi)
        got = fam.values(K, pts) if bc == 0.0 else fam.deriv(K, pts, np.full(len(r), nvec))
        assert np.abs(got).max() < 1e-13 * ref, f"phi={phi} bc={bc}"


# --- Helmholtz exactness ------------------------------------------------------------------------

def _helmholtz_residual(fam, k, pts, h):
    """`(laplacian u + k**2 u)`, relative to `k**2 * max|u|`, by a fourth-order five-point
    stencil in each direction.

    The normalization is against `k**2 * max|u|` and not against `max|u|`, because that is
    the size of the two terms being cancelled -- dividing by `|u|` alone reports a residual
    `k**2` times too pessimistic and says nothing about how exact the solution is.

    Measured h-sweep, worst over four wedge angles: 2.9e-3 at h = 1e-1, 3.1e-7 at 1e-2,
    5.7e-10 at 1e-3, then back up to 9.9e-8 at 1e-4 as roundoff takes over. Clean fourth
    order until it stops, with the minimum at h = 1e-3. The floor is the differencing and
    not the code, which is why the assertions here are 5e-9 rather than 1e-14 -- the
    analytic check that reaches machine level is `test_the_radial_factor_solves_bessels_equation`.
    """
    def u(z):
        return fam.values(k, z)

    lap = np.zeros_like(u(pts))
    for step in (h, 1j * h):
        lap = lap + (-u(pts + 2 * step) + 16 * u(pts + step) - 30 * u(pts)
                     + 16 * u(pts - step) - u(pts - 2 * step)) / (12 * h**2)
    return np.abs(lap + k**2 * u(pts)).max() / (k**2 * np.abs(u(pts)).max())


@pytest.mark.parametrize("theta", [np.pi / 2, np.pi, 1.4 * np.pi, 1.9 * np.pi])
def test_every_column_solves_the_helmholtz_equation(theta):
    fam = wedge(theta, 0.0, 0.0, n=4)
    phi = np.linspace(0.15 * theta, 0.85 * theta, 7)
    pts = np.concatenate([r * np.exp(1j * phi) for r in (0.3, 0.6, 0.9)])
    assert _helmholtz_residual(fam, K, pts, 1e-3) < 5e-9


def test_the_finite_difference_check_is_limited_by_the_differencing():
    """Recorded so nobody tightens the tolerance above and then chases the code. The
    residual has a minimum in h and it is not at the smallest h."""
    fam = wedge(1.4 * np.pi, 0.0, 0.0, n=3)
    pts = np.array([0.5 * np.exp(1j * 0.7), 0.8 * np.exp(1j * 2.0)])
    errs = [_helmholtz_residual(fam, K, pts, h) for h in (1e-1, 1e-2, 1e-3, 1e-4)]
    assert errs[1] < errs[0] / 1e3           # fourth order, while it lasts
    assert errs[2] < errs[1] and errs[2] < errs[3]
    assert errs[2] < 5e-9


# --- the gradient ---------------------------------------------------------------------------

def test_the_gradient_matches_fourth_order_central_differences():
    fam = wedge(1.4 * np.pi, 0.0, 0.0, n=5)
    pts = np.array([0.4 * np.exp(1j * 0.9), 0.7 * np.exp(1j * 2.5), 0.2 * np.exp(1j * 3.5)])
    h = 1e-4
    g = fam.grad(K, pts)
    for axis, step in ((0, h), (1, 1j * h)):
        fd = (-fam.values(K, pts + 2 * step) + 8 * fam.values(K, pts + step)
              - 8 * fam.values(K, pts - step) + fam.values(K, pts - 2 * step)) / (12 * h)
        got = g.real if axis == 0 else g.imag
        scale = np.abs(g).max()
        assert np.max(np.abs(got - fd)) < 1e-10 * scale


def test_complex_step_differentiation_does_not_work_here():
    """Recorded so it is not tried again. `jv` does not carry the imaginary part of a 1e-20
    perturbation to relative accuracy, and `arctan2` does not complexify at all."""
    from scipy.special import jv

    nu, x, h = 3.5, 2.0, 1e-20
    cs = np.imag(jv(nu, complex(x, h))) / h
    exact = 0.5 * (jv(nu - 1, x) - jv(nu + 1, x))
    assert abs(cs - exact) / abs(exact) > 1.0


def test_values_and_grad_agrees_with_the_two_taken_separately():
    """One transcendental pass instead of two. lappy paid for `_bessel` twice at the same
    parameter because `_eval_pointset` and `_grad_pointset` had no memo between them."""
    fam = wedge(1.4 * np.pi, 0.0, 0.0, n=6)
    pts = np.array([0.3 + 0.2j, -0.4 + 0.5j, 0.6 - 0.1j])
    v, g = fam.at(pts).values_and_grad(K)
    assert np.array_equal(v, fam.values(K, pts))
    assert np.array_equal(g, fam.grad(K, pts))


# --- the branch cut -----------------------------------------------------------------------------

def test_the_angle_is_continuous_everywhere_except_on_the_cut():
    """Sampled on a circle about the corner: exactly one jump, and it sits at beta."""
    theta = 1.4 * np.pi
    fam = wedge(theta, 0.0, 0.0, n=1)
    a = np.linspace(0, 2 * np.pi, 4001, endpoint=False)
    phi = fam.at(0.5 * np.exp(1j * a)).phi
    jumps = np.nonzero(np.abs(np.diff(phi)) > np.pi)[0]
    assert len(jumps) == 1
    assert a[jumps[0]] == pytest.approx(fam.modes.beta, abs=2e-3)


def test_the_interior_wedge_maps_to_zero_through_theta():
    theta = 1.4 * np.pi
    fam = wedge(theta, 0.0, 0.0, n=1)
    a = np.linspace(1e-6, theta - 1e-6, 500)
    phi = fam.at(0.5 * np.exp(1j * a)).phi
    assert np.allclose(phi, a, atol=1e-12)


def test_a_square_corner_needs_no_cut_at_all():
    """nu = 2 is integral, so every order is an integer and the family is single-valued.
    The lift still runs and phi still jumps by 2*pi somewhere; the point is that the VALUES
    do not notice, because sin of an integer multiple of 2*pi is periodic.

    Tested as a wrap-around: the step from the last sample back to the first has to be the
    same size as every other step. Comparing consecutive differences to a small number
    instead would just measure the sampling density."""
    fam = wedge(np.pi / 2, 0.0, 0.0, n=4)
    assert fam.modes.is_single_valued
    n = 2000
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    v = fam.values(K, 0.5 * np.exp(1j * a))
    steps = np.abs(np.diff(v, axis=0)).max(axis=1)
    wrap = np.abs(v[0] - v[-1]).max()
    assert wrap < 3 * steps.max()
    assert steps.max() < 10 * np.median(steps)


def test_the_lift_is_not_a_modulo():
    """The bug this file exists for. `angle(z/t_out) % (2*pi)` sends a point at phi = 0 to
    6.283 whenever np.angle returns -1e-17, and the Dirichlet family then evaluates to
    something of order 0.1 on the ray where it must vanish."""
    theta = 1.4 * np.pi
    fam = wedge(theta, 0.0, 0.0, n=3)
    z = np.array([0.5 * np.exp(-1e-17j), 0.5 + 0j])
    assert np.all(fam.values(K, z) == 0.0)


# --- from geometry ------------------------------------------------------------------------------

def test_at_corner_derives_everything_from_a_corner():
    l = Polygon(L_SHAPE)
    reentrant = [c for c in l.corners if c.is_reentrant]
    assert len(reentrant) == 1
    fam = FourierBessel.at_corner(reentrant[0], n=6)
    assert fam.modes.nu == pytest.approx(2 / 3, abs=EXACT)
    assert fam.radius == pytest.approx(reentrant[0].clearance, abs=EXACT)
    assert len(fam) == 6


def test_a_corner_family_vanishes_on_its_own_two_edges():
    """The join key into a quadrature is (bdry_idx, seg_out/seg_in), and the property that
    makes a corner family worth having is that it already satisfies the BC on both."""
    l = Polygon(L_SHAPE)
    q = l.bdry_quad(400)
    for c in l.corners:
        fam = FourierBessel.at_corner(c, n=4)
        for seg in (c.seg_out, c.seg_in):
            m = (q.bdry_idx == c.bdry_idx) & (q.seg_idx == seg)
            assert m.any()
            v = fam.values(1.0, q.pts[m])
            assert np.abs(v).max() < 1e-12


def test_a_hole_corner_is_refused_by_at_corner():
    from chladni.geometry import Boundary, Domain, LineSegment

    mk = lambda v: Boundary([LineSegment(v[i], v[(i + 1) % len(v)]) for i in range(len(v))])
    dom = Domain(mk(np.array([0, 6, 6 + 6j, 6j], dtype=complex)),
                 holes=[mk(np.array([2 + 2j, 2 + 4j, 4 + 4j, 4 + 2j], dtype=complex))])
    for c in [c for c in dom.corners if c.on_hole]:
        with pytest.raises(ValueError, match="hole"):
            FourierBessel.at_corner(c, n=3)


# --- the polyline cut ---------------------------------------------------------------------------

def _dist_to_path(path, z):
    a, b = path[:-1], path[1:]
    d = b - a
    t = np.clip(((z[:, None] - a).real * d.real
                 + (z[:, None] - a).imag * d.imag) / np.abs(d) ** 2, 0.0, 1.0)
    return np.abs(z[:, None] - (a + t * d)).min(axis=1)


def test_a_wall_following_cut_puts_the_branch_on_the_cut_geometry_supplied():
    """The spiral's surrounded corners have no straight ray to infinity, so geometry hands
    back a polyline. The family must put its branch there and not on the bisector.

    Stated as: every 2*pi jump of phi, on a circle about the corner, lies ON
    `Corner.branch_path`. That is the property that matters and it is directly checkable.
    Continuity in Euclidean distance is NOT the right statement and was tried first: in a
    spiral, two points a hundredth apart can be on opposite sides of a wall, with the cut
    running legitimately through the wall between them.

    This also happens to be the test that caught the frame bug -- winding a corner-local
    polygon against un-rotated points, which is invisible on any corner whose t_out is 1.
    """
    from chladni.geometry import spiral

    d = spiral()
    walled = [c for c in d.corners if c.branch_vertices]
    assert walled, "the spiral is supposed to need polyline cuts"
    n = 4000
    for c in walled:
        fam = FourierBessel.at_corner(c, n=1)
        path = c.branch_path(50 * d.diameter)
        for rho in (0.05, 0.2, 0.5, 1.0):
            a = np.linspace(0, 2 * np.pi, n, endpoint=False)
            z = c.z + rho * np.exp(1j * a)
            phi = fam.at(z).phi
            j = np.nonzero(np.abs(np.diff(phi)) > np.pi)[0]
            assert len(j) > 0
            mid = 0.5 * (z[j] + z[j + 1])
            # within one angular sample of the cut; 2*pi/n * rho is the spacing.
            assert _dist_to_path(path, mid).max() < 2 * rho * 2 * np.pi / n


def test_the_ray_case_pays_nothing_for_the_polyline_machinery():
    fam = wedge(1.4 * np.pi, 0.0, 0.0, n=2)
    assert fam.branch_vertices == ()
    assert fam.at(np.array([0.3 + 0.1j])).winding is None


# --- shapes, columns, immutability ----------------------------------------------------------------

def test_the_matrix_is_points_by_columns():
    fam = wedge(n=7)
    pts = np.linspace(0.1, 0.9, 11) * np.exp(0.6j)
    assert fam.values(K, pts).shape == (11, 7)
    assert fam.grad(K, pts).shape == (11, 7)
    assert fam.grad(K, pts).dtype == complex
    assert fam.values(K, pts).dtype == float


def test_cols_selects_and_preserves_the_requested_order():
    fam = wedge(n=6)
    pts = np.linspace(0.1, 0.9, 5) * np.exp(0.6j)
    full = fam.values(K, pts)
    want = np.array([4, 0, 3])
    assert np.allclose(fam.values(K, pts, cols=want), full[:, want], rtol=0, atol=0)


def test_a_family_is_immutable():
    fam = wedge()
    with pytest.raises(AttributeError):
        fam.origin = 1 + 0j


def test_a_sampling_is_reusable_across_k():
    """The whole caching design: MPS builds one Sampling per point set and calls it for
    hundreds of k. Nothing may mutate."""
    fam = wedge(n=4)
    pts = np.linspace(0.1, 0.9, 9) * np.exp(0.6j)
    s = fam.at(pts)
    first = s.values(3.0).copy()
    s.values(11.0)
    assert np.array_equal(s.values(3.0), first)


def test_the_radial_factor_solves_bessels_equation_at_machine_level():
    """The analytic companion to the finite-difference check, with no differencing floor.

    Bessel's equation is what makes the whole family Helmholtz-exact:
    `J'' + J'/x + (1 - mu**2/x**2) J = 0`, and everything else is the chain rule. Checked on
    the scaled kernel directly, so it holds in the underflow regime too -- where the
    finite-difference test cannot run at all, because every value there is zero in float64.
    """
    from chladni.helmholtz._bessel import log_j

    mu = np.array([0.5, 2.0, 37.0, 220.0, 400.0])[None, :]
    x = np.array([0.05, 0.8, 6.0, 25.0, 48.0])[:, None]
    s = log_j(mu, x)
    # J'' from the equation is not independent, so differentiate the series relation instead:
    # x**2 J'' = (mu**2 - x**2) J - x J'. Check that against a Richardson-free identity,
    # the recurrence J_{mu-1} + J_{mu+1} = 2*mu*J_mu/x, which uses no derivative at all.
    lo, hi = log_j(mu - 1, x), log_j(mu + 1, x)
    lhs = (lo.S * np.exp(lo.sigma - s.sigma) + hi.S * np.exp(hi.sigma - s.sigma))
    assert np.allclose(lhs, 2 * mu * s.S / x, rtol=1e-11, atol=0.0)
    # and the derivative relation 2*J' = J_{mu-1} - J_{mu+1}
    d = (lo.S * np.exp(lo.sigma - s.sigma) - hi.S * np.exp(hi.sigma - s.sigma))
    assert np.allclose(2 * s.Sp, d, rtol=1e-11, atol=0.0)
