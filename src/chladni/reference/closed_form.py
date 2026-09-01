"""Eigenvalues that are known in closed form.

Every function here returns `lam`, ascending, with multiplicity repeated -- `lam` because
`conventions.md` makes it the parameter a user talks about, and with multiplicity because a
list of distinct levels silently understates a degenerate spectrum.

Two shapes of query, and the second is not in lappy:

    *_eigs(k, ...)                     the first k
    *_spectrum(lam_lo, lam_hi, ...)    EVERY eigenvalue in the interval

The second is what recall testing needs and it cannot be got from the first by truncation:
"the first ten" and "all of them below 40" are different questions, and only the second can
tell you whether a search missed one.

**Both return the same shape: a flat ascending array with multiplicity expressed by
REPETITION.** A spectrum is a multiset, and that is the data structure for one. It also keeps
degeneracy information exactly, because a true degeneracy is bit-identical -- two equal floats
mean one doubly-degenerate eigenvalue, two unequal ones mean two eigenvalues, and no
`(value, count)` bookkeeping is needed to tell them apart. `collapse()` is available as a
VIEW for a caller who wants counts, but nothing here applies it: see its docstring for why a
tolerance must not live in ground truth.
"""
from __future__ import annotations

import numpy as np
from scipy.special import jn_zeros, jnp_zeros

from ._core import _check_bc, _check_k, _take_k, exact, snap

__all__ = [
    "rect_eig", "rect_eigs", "rect_spectrum",
    "disk_eig", "disk_eigs", "disk_spectrum",
    "sector_eig", "sector_eigs", "sector_spectrum",
    "eq_tri_eig", "eq_tri_eigs", "eq_tri_spectrum",
    "iso_right_tri_eig", "iso_right_tri_eigs", "iso_right_tri_spectrum",
    "polyomino_eig", "polyomino_spectrum",
    "bessel_zero",
]


def bessel_zero(nu, n, derivative=False):
    """The n-th positive zero of `J_nu`, or of `J_nu'` when `derivative=True`.

    One place where the policy lives, because the two backends are good at different things
    and mixing them ad hoc is how a reference ends up quietly worse than the solver checking
    against it.

    **Integer order goes to scipy**, which is at or below one ulp: measured against mpmath at
    50 dps over 32 cases spanning order 0 to 400 and zero index 1 to 100, the worst relative
    error was 1.73e-16 for `J` and 1.76e-16 for `J'`, against a machine epsilon of 2.22e-16.
    There is nothing to gain by replacing it.

    **Fractional order goes to mpmath**, because scipy does not offer it at all and the
    obvious substitute is worse than it looks. lappy's note is worth keeping: a `brentq` scan
    lost two to three digits at fractional order -- returning zeros where `|J_nu(z)|` was
    1.1e-13 rather than ~1e-16 -- which put sector eigenvalues out by up to 1.4e-13 relative
    and once made a solver appear to violate its own error bound when the reference was the
    thing that was wrong.

    These values define the reference spectra for disks and sectors, so they have to be more
    accurate than anything checked against them. Note that an eigenvalue is the zero SQUARED,
    which doubles the relative error: about 3.5e-16, or 15.5 digits.

    **Indexing at `m = 0` with `derivative=True`.** `J_0' = -J_1` and `J_1(0) = 0`, so mpmath
    counts `x = 0` as the first zero while scipy skips it. Every integer order goes to scipy,
    so the convention here is scipy's -- the trivial zero is never returned, and the Neumann
    disk's constant mode is prepended explicitly by `disk_eigs` instead of arriving this way.
    """
    nu = float(nu)
    n = int(n)
    if nu.is_integer() and nu >= 0:
        m = int(nu)
        return float((jnp_zeros(m, n) if derivative else jn_zeros(m, n))[-1])
    import mpmath as mp

    with mp.workdps(40):
        return float(mp.besseljzero(mp.mpf(nu), n, derivative=1 if derivative else 0))


# --- rectangle ---------------------------------------------------------------------------


@exact("closed form: lam = pi**2 * (m**2/L**2 + n**2/H**2)")
def rect_eig(m, n, L=1.0, H=1.0):
    """`lam` of mode `(m, n)`. Broadcasts. Dirichlet needs `m, n >= 1`; Neumann `>= 0`."""
    m, n = np.asarray(m, dtype=float), np.asarray(n, dtype=float)
    return np.pi**2 * (m**2 / float(L) ** 2 + n**2 / float(H) ** 2)


@exact("closed form: lam = pi**2 * (m**2/L**2 + n**2/H**2)")
def rect_eigs(k, L=1.0, H=1.0, bc_type="dir"):
    """The first `k` eigenvalues of the `L x H` rectangle, ascending, with multiplicity.

    Neumann includes the zero mode.
    """
    k, bc_type = _check_k(k), _check_bc(bc_type)
    start = 1 if bc_type == "dir" else 0
    top = start + int(np.ceil(np.sqrt(k))) + k
    idx = np.arange(start, top)
    return _take_k(rect_eig(idx[:, None], idx[None, :], L, H).ravel(), k)


@exact("closed form: lam = pi**2 * (m**2/L**2 + n**2/H**2)")
def rect_spectrum(lam_lo, lam_hi, L=1.0, H=1.0, bc_type="dir"):
    """EVERY eigenvalue in `[lam_lo, lam_hi]`, ascending, with multiplicity repeated.

    Complete by construction: the index bound is derived from `lam_hi`, not guessed.
    """
    bc_type = _check_bc(bc_type)
    start = 1 if bc_type == "dir" else 0
    L, H = float(L), float(H)
    m_max = int(np.floor(L * np.sqrt(max(lam_hi, 0.0)) / np.pi)) + 1
    n_max = int(np.floor(H * np.sqrt(max(lam_hi, 0.0)) / np.pi)) + 1
    m = np.arange(start, m_max + 1)[:, None]
    n = np.arange(start, n_max + 1)[None, :]
    lam = rect_eig(m, n, L, H).ravel()
    return snap(lam[(lam >= lam_lo) & (lam <= lam_hi)])


# --- disk --------------------------------------------------------------------------------


@exact("closed form: lam = (j_{m,n}/r)**2, Dirichlet; (j'_{m,n}/r)**2, Neumann")
def disk_eig(m, n, r=1.0, bc_type="dir"):
    """`lam` of the mode with angular index `m >= 0` and radial index `n >= 1`.

    Multiplicity is 1 for `m = 0` and 2 for `m >= 1`. The Neumann constant mode `lam = 0` is
    not returned here; `disk_eigs` prepends it.
    """
    bc_type = _check_bc(bc_type)
    z = bessel_zero(int(m), int(n), derivative=(bc_type == "neu"))
    return (z / float(r)) ** 2


def _disk_pairs(lam_hi, r, bc_type):
    """(lam, mult) for every disk mode with `lam <= lam_hi`. Complete: `j_{m,1}` grows with
    `m`, so the angular index is exhausted as soon as the first radial zero exceeds the cut."""
    out, m = [], 0
    lim = np.sqrt(max(lam_hi, 0.0)) * float(r)
    while True:
        n_max = 1
        while True:
            zz = jn_zeros(m, n_max) if bc_type == "dir" else jnp_zeros(m, n_max)
            if zz[-1] > lim:
                break
            n_max *= 2
            if n_max > 4096:
                raise ValueError("radial index overflow; lam_hi is implausibly large")
        zz = zz[zz <= lim]
        if zz.size == 0 and m > 0:
            break
        for z in zz:
            out.append(((z / float(r)) ** 2, 1 if m == 0 else 2))
        m += 1
        if m > 4096:
            raise ValueError("angular index overflow")
    return out


@exact("closed form: lam = (j_{m,n}/r)**2, Dirichlet; (j'_{m,n}/r)**2, Neumann")
def disk_eigs(k, r=1.0, bc_type="dir"):
    """The first `k` eigenvalues of the disk, ascending, with multiplicity.

    Every `m >= 1` mode is doubly degenerate (the sine and cosine pair) and appears twice.
    Neumann prepends the zero mode.
    """
    k, bc_type = _check_k(k), _check_bc(bc_type)
    lam = [0.0] if bc_type == "neu" else []
    hi = (np.pi * (k + 2) / float(r)) ** 2
    for value, mult in _disk_pairs(hi, r, bc_type):
        lam.extend([value] * mult)
    return _take_k(lam, k)


@exact("closed form: lam = (j_{m,n}/r)**2, Dirichlet; (j'_{m,n}/r)**2, Neumann")
def disk_spectrum(lam_lo, lam_hi, r=1.0, bc_type="dir"):
    """EVERY eigenvalue in `[lam_lo, lam_hi]`, ascending, with multiplicity repeated.

    Each `m >= 1` mode appears twice: the sine and cosine pair are the same real number, so
    they are the same float, and the repetition IS the degeneracy.
    """
    bc_type = _check_bc(bc_type)
    lam = [0.0] if (bc_type == "neu" and lam_lo <= 0.0 <= lam_hi) else []
    for value, mult in _disk_pairs(lam_hi, r, bc_type):
        if lam_lo <= value <= lam_hi:
            lam.extend([value] * mult)
    return snap(np.asarray(lam, dtype=float))


# --- circular sector ---------------------------------------------------------------------


@exact("closed form: lam = (j_{l*pi/omega, n}/r)**2")
def sector_eig(l, n, r=1.0, omega=np.pi / 2):
    """`lam` of the sector mode `(l, n)`, `l >= 1` angular and `n >= 1` radial.

    Every mode is simple: the angular factor is `sin(l*nu*theta)` with `nu = pi/omega`. This
    is the sharpest reference available, because that eigenfunction is EXACTLY one
    corner-adapted Fourier-Bessel column.
    """
    nu = np.pi / float(omega)
    return (bessel_zero(int(l) * nu, int(n)) / float(r)) ** 2


def _sector_values(lam_hi, r, omega):
    nu = np.pi / float(omega)
    lim = np.sqrt(max(lam_hi, 0.0)) * float(r)
    out, l = [], 1
    while True:
        if bessel_zero(l * nu, 1) > lim:
            break
        n = 1
        while True:
            z = bessel_zero(l * nu, n)
            if z > lim:
                break
            out.append((z / float(r)) ** 2)
            n += 1
            if n > 4096:
                raise ValueError("radial index overflow")
        l += 1
        if l > 4096:
            raise ValueError("angular index overflow")
    return out


@exact("closed form: lam = (j_{l*pi/omega, n}/r)**2")
def sector_eigs(k, r=1.0, omega=np.pi / 2):
    """The first `k` Dirichlet eigenvalues of the sector of opening `omega`."""
    k = _check_k(k)
    hi = (np.pi * (k + 2) / float(r)) ** 2 * max(1.0, (np.pi / float(omega)) ** 2)
    return _take_k(_sector_values(hi, r, omega), k)


@exact("closed form: lam = (j_{l*pi/omega, n}/r)**2")
def sector_spectrum(lam_lo, lam_hi, r=1.0, omega=np.pi / 2):
    """EVERY Dirichlet eigenvalue in `[lam_lo, lam_hi]`, ascending. All modes are simple."""
    v = [x for x in _sector_values(lam_hi, r, omega) if lam_lo <= x <= lam_hi]
    return snap(np.asarray(v, dtype=float))


# --- equilateral triangle ----------------------------------------------------------------


@exact("Lame closed form: lam = (16*pi**2/(9*l**2)) * (m**2 + m*n + n**2)")
def eq_tri_eig(m, n, l=1.0):
    """`lam` of the equilateral triangle of side `l`, from the Lame solution.

    `m, n >= 1`. Multiplicity is arithmetic rather than geometric: `m**2 + m*n + n**2`
    collides for distinct index pairs, on top of the `(m,n)`/`(n,m)` pairing.
    """
    m, n = np.asarray(m, dtype=float), np.asarray(n, dtype=float)
    return (16 * np.pi**2 / (9 * float(l) ** 2)) * (m**2 + m * n + n**2)


def _eq_tri_values(lam_hi, l):
    cap = int(np.ceil(np.sqrt(max(lam_hi, 0.0) * 9 * float(l) ** 2 / (16 * np.pi**2)))) + 2
    m = np.arange(1, cap + 1)[:, None]
    n = np.arange(1, cap + 1)[None, :]
    return eq_tri_eig(m, n, l).ravel()


@exact("Lame closed form: lam = (16*pi**2/(9*l**2)) * (m**2 + m*n + n**2)")
def eq_tri_eigs(k, l=1.0):
    """The first `k` Dirichlet eigenvalues, ascending, with multiplicity."""
    k = _check_k(k)
    hi = (16 * np.pi**2 / (9 * float(l) ** 2)) * (3 * (k + 2) ** 2)
    return _take_k(_eq_tri_values(hi, l), k)


@exact("Lame closed form: lam = (16*pi**2/(9*l**2)) * (m**2 + m*n + n**2)")
def eq_tri_spectrum(lam_lo, lam_hi, l=1.0):
    """EVERY Dirichlet eigenvalue in `[lam_lo, lam_hi]`, ascending, multiplicity repeated."""
    v = _eq_tri_values(lam_hi, l)
    return snap(v[(v >= lam_lo) & (v <= lam_hi)])


# --- isosceles right triangle ------------------------------------------------------------


@exact("closed form: lam = pi**2 (m**2 + n**2)/l**2, m > n >= 1 (square modes, antisymmetric)")
def iso_right_tri_eig(m, n, l=1.0):
    """`lam` of the right isosceles triangle with legs `l`. Valid only for `m > n >= 1`.

    The triangle is the square folded on its diagonal, so its modes are the square's
    antisymmetric combinations; `m == n` gives the zero function and is not a mode.
    """
    if not (int(m) > int(n) >= 1):
        raise ValueError(f"need m > n >= 1, got ({m}, {n})")
    return np.pi**2 * (int(m) ** 2 + int(n) ** 2) / float(l) ** 2


def _iso_right_values(lam_hi, l):
    cap = int(np.floor(float(l) * np.sqrt(max(lam_hi, 0.0)) / np.pi)) + 2
    return np.array([np.pi**2 * (m * m + n * n) / float(l) ** 2
                     for m in range(2, cap + 1) for n in range(1, m)])


@exact("closed form: lam = pi**2 (m**2 + n**2)/l**2, m > n >= 1")
def iso_right_tri_eigs(k, l=1.0):
    """The first `k` Dirichlet eigenvalues, ascending, with multiplicity."""
    k = _check_k(k)
    hi = np.pi**2 * 2 * (k + 3) ** 2 / float(l) ** 2
    return _take_k(_iso_right_values(hi, l), k)


@exact("closed form: lam = pi**2 (m**2 + n**2)/l**2, m > n >= 1")
def iso_right_tri_spectrum(lam_lo, lam_hi, l=1.0):
    """EVERY Dirichlet eigenvalue in `[lam_lo, lam_hi]`, ascending, multiplicity repeated."""
    v = _iso_right_values(lam_hi, l)
    return snap(v[(v >= lam_lo) & (v <= lam_hi)])


# --- polyomino ---------------------------------------------------------------------------


@exact("closed form: lam = pi**2 (m**2 + n**2), independent of which cells")
def polyomino_eig(m, n):
    """`lam` of any polyomino of unit cells, for `m, n >= 1`.

    `sin(m*pi*x) sin(n*pi*y)` vanishes on the WHOLE integer grid, so it is an exact Dirichlet
    eigenfunction of every polyomino regardless of shape. Note what that costs: a closed-form
    eigenfunction on a nonconvex domain is necessarily SMOOTH at the reentrant corners, which
    is exactly why the L-shape has none. Use this family as a control on geometry, panel
    splitting and assembly -- it carries zero singular amplitude and cannot test a corner.
    """
    m, n = np.asarray(m, dtype=float), np.asarray(n, dtype=float)
    return np.pi**2 * (m**2 + n**2)


@exact("closed form: lam = pi**2 (m**2 + n**2)")
def polyomino_spectrum(lam_lo, lam_hi):
    """Every `pi**2 (m**2+n**2)` in the interval, ascending, with multiplicity repeated.

    The multiplicity counted here is that of the GRID modes. A given polyomino may not
    support all of them as distinct eigenfunctions, so treat this as an upper bound on
    multiplicity unless the shape is a full rectangle.
    """
    cap = int(np.floor(np.sqrt(max(lam_hi, 0.0)) / np.pi)) + 2
    m = np.arange(1, cap + 1)[:, None]
    n = np.arange(1, cap + 1)[None, :]
    v = polyomino_eig(m, n).ravel()
    return snap(v[(v >= lam_lo) & (v <= lam_hi)])
