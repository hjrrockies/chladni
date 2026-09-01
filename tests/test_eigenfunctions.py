"""Closed-form eigenfunctions: they solve the equation, they vanish, and their norms are right.

Three properties, and none of them is checked against another chladni function.

An eigenfunction has to satisfy `laplacian(u) + lam*u = 0` -- verified by finite differences
in the interior, which is independent of everything in the package. It has to vanish on a
Dirichlet boundary. And its `norm2` has to match a numerical quadrature, because `norm2` is
what makes these usable as ground truth for the Rellich work: an independently-normalized
eigenfunction is something a quadrature can be checked AGAINST, rather than something a
quadrature defines.

Placement is the fourth property and the easiest to get wrong. Every function assumes the
placement of the factory it names; if the two ever drift apart the norms are simply wrong, so
the boundary test doubles as a placement test.
"""
import numpy as np
import pytest

from chladni import reference as ref
from chladni.geometry import disk, disk_sector, iso_right_tri, plus_shape, rect


def laplacian(u, z, h=1e-5):
    """Five-point finite-difference Laplacian, independent of the package."""
    return ((u(z + h) + u(z - h) + u(z + 1j * h) + u(z - 1j * h) - 4 * u(z)) / h**2)


def quad_norm2(u, dom, n=400_000, seed=0):
    """Monte-Carlo integral of |u|^2 over the domain. Crude but wholly independent."""
    pts = dom.interior_pts(n, rng=np.random.default_rng(seed))
    return float(np.mean(np.abs(u(pts)) ** 2) * dom.area)


CASES = [
    ("rect", lambda: ref.rect_eigfun(2, 3, 2.0, 1.0), ref.rect_eig(2, 3, 2.0, 1.0),
     lambda: rect(2.0, 1.0)),
    ("iso_right_tri", lambda: ref.iso_right_tri_eigfun(3, 1, 1.0),
     ref.iso_right_tri_eig(3, 1, 1.0), lambda: iso_right_tri(1.0)),
    ("disk", lambda: ref.disk_eigfun(2, 1, 1.0), ref.disk_eig(2, 1, 1.0), lambda: disk(1.0)),
    ("sector", lambda: ref.sector_eigfun(1, 1, 1.0, 2 * np.pi / 3),
     ref.sector_eig(1, 1, 1.0, 2 * np.pi / 3), lambda: disk_sector(1.0, 2 * np.pi / 3)),
    ("polyomino", lambda: ref.polyomino_eigfun(1, 2, 5), ref.polyomino_eig(1, 2),
     plus_shape),
]


@pytest.mark.parametrize("name,make,lam,dom_fn", CASES, ids=[c[0] for c in CASES])
def test_it_solves_the_helmholtz_equation(name, make, lam, dom_fn):
    u, _ = make()
    dom = dom_fn()
    z = dom.interior_pts(60, rng=np.random.default_rng(1))
    # stay clear of the boundary, where a difference stencil leaves the domain
    z = z[dom.boundary_distance(z) > 1e-2]
    resid = laplacian(u, z) + lam * u(z)
    assert np.max(np.abs(resid)) < 1e-4 * max(np.max(np.abs(u(z))), 1.0)


@pytest.mark.parametrize("name,make,lam,dom_fn", CASES, ids=[c[0] for c in CASES])
def test_it_vanishes_on_the_dirichlet_boundary(name, make, lam, dom_fn):
    """Also the placement test: a formula built for a differently-placed domain does not
    vanish on this one."""
    u, _ = make()
    q = dom_fn().bdry_quad(400)
    assert np.max(np.abs(u(q.pts))) < 1e-9


@pytest.mark.parametrize("name,make,lam,dom_fn", CASES, ids=[c[0] for c in CASES])
def test_the_exact_norm_matches_a_quadrature(name, make, lam, dom_fn):
    """`norm2` is the reason these exist. Monte-Carlo is crude, so the tolerance is loose --
    the point is that the exact value is right, not that the estimator is."""
    u, norm2 = make()
    assert quad_norm2(u, dom_fn()) == pytest.approx(norm2, rel=0.02)


def test_normalized_eigfun_has_unit_norm():
    un = ref.normalized_eigfun(ref.rect_eigfun(2, 3, 2.0, 1.0))
    assert quad_norm2(un, rect(2.0, 1.0)) == pytest.approx(1.0, rel=0.02)


def test_a_polyomino_mode_is_exact_on_any_polyomino():
    """The family's whole point: `sin(m pi x) sin(n pi y)` vanishes on the ENTIRE integer
    grid, so it is an eigenfunction of every polyomino, and `norm2` is `n_cells/4` whatever
    the shape. Note what that costs -- it is smooth at the reentrant corners and carries zero
    singular amplitude, so it is a control, not a corner test."""
    u, norm2 = ref.polyomino_eigfun(1, 2, 5)
    assert norm2 == pytest.approx(5 / 4)
    q = plus_shape().bdry_quad(600)
    assert np.max(np.abs(u(q.pts))) < 1e-12


def test_the_neumann_rectangle_mode_has_zero_normal_derivative():
    u, norm2 = ref.rect_eigfun(1, 0, 2.0, 1.0, bc="neu")
    assert norm2 == pytest.approx((2.0 / 2) * 1.0)
    q = rect(2.0, 1.0, bc="neu").bdry_quad(400)
    h = 1e-6
    dn = (u(q.pts - h * q.normals) - u(q.pts - 3 * h * q.normals)) / (2 * h)
    assert np.max(np.abs(dn)) < 1e-4


# --- gradients ----------------------------------------------------------------------------


def test_the_polyomino_gradient_matches_a_difference():
    u, _ = ref.polyomino_eigfun(1, 2, 5)
    g = ref.polyomino_eigfun_grad(1, 2)
    z = plus_shape().interior_pts(40, rng=np.random.default_rng(2))
    h = 1e-6
    fd = (u(z + h) - u(z - h)) / (2 * h) + 1j * (u(z + 1j * h) - u(z - 1j * h)) / (2 * h)
    assert np.allclose(g(z), fd, rtol=1e-5, atol=1e-7)


def test_the_sector_gradient_is_analytic_because_a_difference_cannot_be():
    """Away from the apex it must match a difference. AT the apex it must not be regularized:
    for `nu < 1` it blows up like `rho**(nu-1)`, which is the entire point of testing there,
    and a difference step near the corner crosses the branch of `angle(z)` and returns
    garbage silently."""
    omega = 3 * np.pi / 2                      # nu = 2/3 < 1, singular apex
    u, _ = ref.sector_eigfun(1, 1, 1.0, omega)
    g = ref.sector_eigfun_grad(1, 1, 1.0, omega)

    z = 0.5 * np.exp(1j * np.linspace(0.2, omega - 0.2, 12))
    h = 1e-6
    fd = (u(z + h) - u(z - h)) / (2 * h) + 1j * (u(z + 1j * h) - u(z - 1j * h)) / (2 * h)
    assert np.allclose(g(z), fd, rtol=1e-4, atol=1e-6)

    near = np.array([1e-8 * np.exp(1j * 0.7)])
    assert np.abs(g(near))[0] > np.abs(g(z)).max(), "the apex singularity was smoothed away"


def test_sin_parity_is_rejected_for_the_radial_disk_mode():
    with pytest.raises(ValueError, match="identically zero"):
        ref.disk_eigfun(0, 1, 1.0, parity="sin")
