"""Closed-form eigenfunctions, with their exact L2 norms.

Each `*_eigfun` returns `(u, norm2)`, where `u(z)` is vectorized over complex `z = x + 1j*y`
and `norm2` is the EXACT value of the integral of `|u|**2` over the domain.

Un-normalized on purpose. An independently-normalized eigenfunction is ground truth that a
quadrature can be checked AGAINST, rather than something a quadrature defines; keeping the
norm separate is what makes that possible. `normalized_eigfun` divides by `sqrt(norm2)` when
that is what is wanted.

Gradients come back as `du_dx + 1j*du_dy` in a single complex array, which is
`conventions.md`'s gradient convention, so a directional derivative is `dot(vec, grad)` with
no reshaping.

**Placement is load-bearing.** Every function here assumes the placement of the factory it
names, and its `norm2` is simply wrong if the two disagree:

    rect(L, H)             corner at the origin
    iso_right_tri(l)       right angle at the origin, legs on +x and +y
    disk(r)                centred at the origin
    disk_sector(r, omega)  apex at the origin, angular extent [0, omega]
    polyomino(cells)       on the integer grid
"""
from __future__ import annotations

import numpy as np
from scipy.special import jv

from ._core import exact
from .closed_form import bessel_zero

__all__ = ["rect_eigfun", "iso_right_tri_eigfun", "polyomino_eigfun",
           "polyomino_eigfun_grad", "disk_eigfun", "sector_eigfun", "sector_eigfun_grad",
           "normalized_eigfun"]


def normalized_eigfun(eigfun_result):
    """`u / sqrt(norm2)` from an `(u, norm2)` pair."""
    u, norm2 = eigfun_result
    s = np.sqrt(float(norm2))
    return lambda z: u(z) / s


@exact("closed form on rect(L, H), corner at the origin")
def rect_eigfun(m, n, L=1.0, H=1.0, bc="dir"):
    """`(u, norm2)` on `[0, L] x [0, H]`. Dirichlet `m, n >= 1`; Neumann `m, n >= 0`."""
    m, n, L, H = int(m), int(n), float(L), float(H)
    if bc == "dir":
        if m < 1 or n < 1:
            raise ValueError(f"Dirichlet needs m, n >= 1, got ({m}, {n})")
        u = lambda z: np.sin(m * np.pi * z.real / L) * np.sin(n * np.pi * z.imag / H)
        return u, (L / 2) * (H / 2)
    if bc == "neu":
        if m < 0 or n < 0:
            raise ValueError(f"Neumann needs m, n >= 0, got ({m}, {n})")
        u = lambda z: np.cos(m * np.pi * z.real / L) * np.cos(n * np.pi * z.imag / H)
        return u, (L if m == 0 else L / 2) * (H if n == 0 else H / 2)
    raise ValueError(f"bc must be 'dir' or 'neu', got {bc!r}")


@exact("closed form on iso_right_tri(l), right angle at the origin")
def iso_right_tri_eigfun(m, n, l=1.0):
    """`(u, norm2)` on the triangle with vertices `0, l, 1j*l`. Needs `m > n >= 1`.

    The square's mode antisymmetrized under `(x, y) -> (y, x)`; the square gives `l**2/2` and
    the antisymmetry halves it.
    """
    m, n, l = int(m), int(n), float(l)
    if not m > n >= 1:
        raise ValueError(f"need m > n >= 1, got ({m}, {n})")

    def u(z):
        x, y = z.real, z.imag
        return (np.sin(m * np.pi * x / l) * np.sin(n * np.pi * y / l)
                - np.sin(n * np.pi * x / l) * np.sin(m * np.pi * y / l))

    return u, l**2 / 4


@exact("closed form on any polyomino of unit cells, on the integer grid")
def polyomino_eigfun(m, n, n_cells):
    """`(u, norm2)` for `sin(m*pi*x) sin(n*pi*y)`, exact on ANY polyomino of `n_cells` cells.

    Each cell contributes `1/2 * 1/2`, so `norm2 = n_cells / 4` regardless of shape.

    The caveat that comes with it: a closed-form eigenfunction on a nonconvex domain is
    necessarily SMOOTH at the reentrant corners -- which is exactly why the L-shape has none.
    This family exercises geometry, panel splitting and assembly across several reentrant
    corners at once with exact truth, and carries ZERO singular amplitude. A control, not a
    corner test.
    """
    m, n = int(m), int(n)
    if m < 1 or n < 1:
        raise ValueError(f"need m, n >= 1, got ({m}, {n})")
    u = lambda z: np.sin(m * np.pi * z.real) * np.sin(n * np.pi * z.imag)
    return u, float(n_cells) / 4.0


@exact("closed form on any polyomino of unit cells")
def polyomino_eigfun_grad(m, n):
    """`g(z) = du_dx + 1j*du_dy` for `polyomino_eigfun`. A bare callable, not a pair."""
    m, n = int(m), int(n)

    def g(z):
        x, y = z.real, z.imag
        sx, cx = np.sin(m * np.pi * x), np.cos(m * np.pi * x)
        sy, cy = np.sin(n * np.pi * y), np.cos(n * np.pi * y)
        return m * np.pi * cx * sy + 1j * (n * np.pi * sx * cy)

    return g


@exact("closed form on disk(r), centred at the origin")
def disk_eigfun(m, n, r=1.0, parity="cos"):
    """`(u, norm2)` for `J_m(k rho) * trig(m theta)` with `k = j_{m,n}/r`. Dirichlet only.

    `parity='sin'` is rejected for `m = 0`, where it is identically zero.
    """
    m, n, r = int(m), int(n), float(r)
    if m < 0 or n < 1:
        raise ValueError(f"need m >= 0, n >= 1, got ({m}, {n})")
    if parity not in ("cos", "sin"):
        raise ValueError(f"parity must be 'cos' or 'sin', got {parity!r}")
    if m == 0 and parity == "sin":
        raise ValueError("parity='sin' is identically zero for m = 0")
    z0 = bessel_zero(m, n)
    k = z0 / r
    trig = np.cos if parity == "cos" else np.sin

    def u(zz):
        return jv(m, k * np.abs(zz)) * trig(m * np.angle(zz))

    angular = 2 * np.pi if m == 0 else np.pi
    return u, angular * (r**2 / 2) * jv(m + 1, z0) ** 2


@exact("closed form on disk_sector(r, omega), apex at the origin")
def sector_eigfun(l, n, r=1.0, omega=np.pi / 2):
    """`(u, norm2)` for `J_nu(k rho) sin(nu theta)` with `nu = l*pi/omega`, `k = j_{nu,n}/r`.

    `theta` is `angle(z) mod 2*pi`, which matters once `omega > pi`.

    This is the sharpest reference in the suite: `u` is EXACTLY one corner-adapted
    Fourier-Bessel column, so a one-column basis reproduces it to machine precision.
    """
    l, n, r, omega = int(l), int(n), float(r), float(omega)
    if l < 1 or n < 1:
        raise ValueError(f"need l, n >= 1, got ({l}, {n})")
    nu = l * np.pi / omega
    z0 = bessel_zero(nu, n)
    k = z0 / r

    def u(zz):
        return jv(nu, k * np.abs(zz)) * np.sin(nu * (np.angle(zz) % (2 * np.pi)))

    return u, (omega / 2) * (r**2 / 2) * jv(nu + 1, z0) ** 2


@exact("closed form on disk_sector(r, omega)")
def sector_eigfun_grad(l, n, r=1.0, omega=np.pi / 2):
    """`g(z) = du_dx + 1j*du_dy` for `sector_eigfun`. A bare callable.

    Analytic rather than differenced, and the reason is specific: the graded quadratures this
    is meant to check place nodes within 1e-6 of the apex, where a difference step of
    comparable size crosses the corner and `angle(z)` jumps by `2*pi` -- returning garbage
    silently, and negative squared norms downstream.

    For `nu < 1` the gradient blows up like `rho**(nu-1)` at the apex, which is the entire
    point of testing there. It is returned as-is -- inf or nan exactly at the apex -- rather
    than regularized.
    """
    l, n, r, omega = int(l), int(n), float(r), float(omega)
    nu = l * np.pi / omega
    z0 = bessel_zero(nu, n)
    k = z0 / r

    def g(zz):
        rho = np.abs(zz)
        th = np.angle(zz) % (2 * np.pi)
        with np.errstate(divide="ignore", invalid="ignore"):
            jp = 0.5 * (jv(nu - 1, k * rho) - jv(nu + 1, k * rho))
            u_r = k * jp * np.sin(nu * th)
            u_t = jv(nu, k * rho) * nu * np.cos(nu * th) / rho
            return (u_r + 1j * u_t) * np.exp(1j * th)

    return g
