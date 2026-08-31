"""Shared fixtures and ground truth for the geometry test suite.

Every reference value here is closed-form, or computed from a source independent of
`chladni` (scipy's elliptic integrals, mpmath quadrature). Nothing in this file may be
regenerated from `chladni` output -- once the solver is trustworthy that becomes an option
for features with no closed form, but not before, and not for these.

The suite is written against docs/design/geometry.md before the implementation exists, so
collection fails until `chladni.geometry` is importable. That is the intended red state.
"""
import numpy as np
import pytest

from chladni.geometry import (
    ArcSegment,
    Boundary,
    Domain,
    LineSegment,
    Polygon,
)

# --- tolerances -------------------------------------------------------------------------

PREC = 1e-8      # the library default working precision
EXACT = 1e-13    # closed-form quantities must land at machine level, not at prec

# --- vertex sets, all counter-clockwise -------------------------------------------------

SQUARE = np.array([0, 1, 1 + 1j, 1j], dtype=complex)
RECT_4x1 = np.array([0, 4, 4 + 1j, 1j], dtype=complex)
L_SHAPE = np.array([0, 2, 2 + 1j, 1 + 1j, 1 + 2j, 2j], dtype=complex)
EQ_TRI = np.array([0, 1, 0.5 + np.sqrt(3) / 2 * 1j], dtype=complex)

# vertex 1 is collinear with its neighbours: a pi junction that is NOT a corner
COLLINEAR = np.array([0, 1, 2, 2 + 1j, 1j], dtype=complex)


def ngon_vertices(n, r=1.0):
    """Regular n-gon, circumradius r, CCW, first vertex on the positive real axis."""
    return r * np.exp(2j * np.pi * np.arange(n) / n)


# --- closed-form ground truth -----------------------------------------------------------

# L_SHAPE is a 2x2 square with a 1x1 bite out of the top-right corner.
L_SHAPE_AREA = 3.0
L_SHAPE_PERIMETER = 8.0          # 2+1+1+1+1+2; a corner bite leaves the perimeter unchanged
L_SHAPE_DIAMETER = 2 * np.sqrt(2)          # |(2,0) - (0,2)|
L_SHAPE_ANGLES = np.array([0.5, 0.5, 0.5, 1.5, 0.5, 0.5]) * np.pi

EQ_TRI_AREA = np.sqrt(3) / 4
EQ_TRI_INRADIUS = 1 / (2 * np.sqrt(3))

# --- polygon fixtures -------------------------------------------------------------------


@pytest.fixture
def square():
    return Polygon(SQUARE)


@pytest.fixture
def rect_4x1():
    return Polygon(RECT_4x1)


@pytest.fixture
def l_shape():
    return Polygon(L_SHAPE)


@pytest.fixture
def eq_tri():
    return Polygon(EQ_TRI)


@pytest.fixture
def collinear_poly():
    return Polygon(COLLINEAR)


# --- curved and mixed fixtures ----------------------------------------------------------


@pytest.fixture
def unit_disk():
    """Full circle as a single closed ArcSegment."""
    arc = ArcSegment(0j, 1.0, 0.0, 2 * np.pi)
    return Domain(Boundary([arc]))


@pytest.fixture
def half_disk():
    """The lappy regression case: THREE segments, TWO corners.

    The two radial pieces are collinear, so the junction at the origin is a smooth pi
    junction. lappy's corner arrays desynced here -- `corners` had 2 entries while the
    angle arrays had 3 -- because one indexing was corner-based and the other
    segment-based. Corner angles at +-1 are pi/2.
    """
    return Domain(
        Boundary(
            [
                LineSegment(-1 + 0j, 0j),
                LineSegment(0j, 1 + 0j),
                ArcSegment(0j, 1.0, 0.0, np.pi),
            ]
        )
    )


@pytest.fixture
def annulus():
    """Outer radius 2, hole radius 1. Area 3*pi, perimeter 6*pi."""
    outer = Boundary([ArcSegment(0j, 2.0, 0.0, 2 * np.pi)])
    hole = Boundary([ArcSegment(0j, 1.0, 0.0, 2 * np.pi)]).reversed()
    return Domain(outer, holes=(hole,))


@pytest.fixture
def plus_shape_domain():
    from chladni.geometry import plus_shape

    return plus_shape()


@pytest.fixture
def square_with_square_hole():
    """[0,3]^2 with [1,2]^2 removed. Area 8, perimeter 12 + 4 = 16."""
    outer = Boundary.from_vertices(np.array([0, 3, 3 + 3j, 3j], dtype=complex))
    hole = Boundary.from_vertices(
        np.array([1 + 1j, 2 + 1j, 2 + 2j, 1 + 2j], dtype=complex)
    ).reversed()
    return Domain(outer, holes=(hole,))


@pytest.fixture
def split_edge_bc_junction():
    """A unit square whose bottom edge is split into two COLLINEAR segments with
    different boundary conditions.

    The junction at 0.5 has interior angle exactly pi, so it is not a geometric corner,
    but the Dirichlet/Neumann change makes it genuinely singular. lappy's detection test
    -- `abs(turning_angle) > 1e-10` -- classifies it as no corner at all.
    """
    return Domain(
        Boundary(
            [
                LineSegment(0j, 0.5 + 0j, bc="dir"),
                LineSegment(0.5 + 0j, 1 + 0j, bc="neu"),
                LineSegment(1 + 0j, 1 + 1j),
                LineSegment(1 + 1j, 1j),
                LineSegment(1j, 0j),
            ]
        )
    )


# --- independent reference for curved arclength -----------------------------------------


def ellipse_perimeter(a, b):
    """Exact ellipse perimeter, 4*a*E(m) with m = 1 - b^2/a^2, from scipy."""
    from scipy.special import ellipe

    a, b = max(a, b), min(a, b)
    return 4 * a * ellipe(1 - (b / a) ** 2)


def ellipse_arclength(a, b, t0, t1):
    """Arclength of (a*cos t, b*sin t) over [t0, t1], by mpmath quadrature."""
    import mpmath as mp

    with mp.workdps(30):
        speed = lambda t: mp.sqrt((a * mp.sin(t)) ** 2 + (b * mp.cos(t)) ** 2)
        return float(mp.quad(speed, [t0, t1]))


# --- mpmath references for the helmholtz layer ------------------------------------------
#
# 50 decimal digits, which is ~37 digits of headroom over float64. These are slow (a few ms
# per call, one Python call per point), so callers keep the grids small and mark the sweeps
# `slow`. Never vectorized -- mpmath has no array path, and pretending otherwise here would
# hide the cost from the test that pays it.

MP_DPS = 50


def mp_besselj(nu, x):
    """J_nu(x) as a float64, computed at 50 dps. Reference, not a fallback."""
    import mpmath as mp

    with mp.workdps(MP_DPS):
        return float(mp.besselj(mp.mpf(float(nu)), mp.mpf(float(x))))


def mp_besselj_prime(nu, x):
    """J_nu'(x) at 50 dps, via mpmath's own derivative form."""
    import mpmath as mp

    with mp.workdps(MP_DPS):
        nu, x = mp.mpf(float(nu)), mp.mpf(float(x))
        return float(0.5 * (mp.besselj(nu - 1, x) - mp.besselj(nu + 1, x)))


def mp_bessely(m, x):
    """Y_m(x) at 50 dps."""
    import mpmath as mp

    with mp.workdps(MP_DPS):
        return float(mp.bessely(mp.mpf(float(m)), mp.mpf(float(x))))


def mp_bessely_prime(m, x):
    """Y_m'(x) at 50 dps."""
    import mpmath as mp

    with mp.workdps(MP_DPS):
        m, x = mp.mpf(float(m)), mp.mpf(float(x))
        return float(0.5 * (mp.bessely(m - 1, x) - mp.bessely(m + 1, x)))


def mp_log_besselj(nu, x):
    """(sign, log|J_nu(x)|) at 50 dps -- the reference in the regime where float64 cannot
    hold the value at all, which is the whole point of the scaled kernel."""
    import mpmath as mp

    with mp.workdps(MP_DPS):
        v = mp.besselj(mp.mpf(float(nu)), mp.mpf(float(x)))
        return (1 if v > 0 else -1 if v < 0 else 0), float(mp.log(abs(v))) if v != 0 else -mp.inf


def mp_log_bessely(m, x):
    """(sign, log|Y_m(x)|) at 50 dps."""
    import mpmath as mp

    with mp.workdps(MP_DPS):
        v = mp.bessely(mp.mpf(float(m)), mp.mpf(float(x)))
        return (1 if v > 0 else -1 if v < 0 else 0), float(mp.log(abs(v))) if v != 0 else -mp.inf


# --- eigenvalue ground truth ------------------------------------------------------------
#
# Closed forms only, as the rest of this file. These are the references the `mps` layer is
# tested against, and `docs/design/particular_solutions.md` lists them as "the mps layer's
# first tests". lappy's tabulated tables are deliberately NOT ported: its own docs/todo.md
# records `ellipse_eigs(a=2)[0]` being wrong in the 9th digit, and a table nobody can
# rederive is not ground truth.


def rect_eigs(L=1.0, H=1.0, n=10):
    """The n smallest Dirichlet eigenvalues of the L x H rectangle, ascending.

    `lam_{pq} = pi**2 * (p**2/L**2 + q**2/H**2)`, p, q >= 1.
    """
    p = np.arange(1, n + 1)[:, None]
    q = np.arange(1, n + 1)[None, :]
    lam = np.pi**2 * (p**2 / L**2 + q**2 / H**2)
    return np.sort(lam.ravel())[:n]


def disk_eigs(r=1.0, n=10):
    """The n smallest Dirichlet eigenvalues of the disk of radius r, ascending.

    `lam_{mj} = (j_{m,j} / r)**2` with `j_{m,j}` the j-th positive zero of `J_m`. Every
    `m >= 1` is doubly degenerate (the sine and cosine modes), which is why they are
    repeated here -- a multiplicity the deferred estimator will have to see.
    """
    from scipy.special import jn_zeros

    lam = []
    for m in range(n + 1):
        for z in jn_zeros(m, n):
            lam.append((z / r) ** 2)
            if m > 0:
                lam.append((z / r) ** 2)  # doubly degenerate
    return np.sort(np.array(lam))[:n]


def sector_eigs(omega, r=1.0, n=10):
    """The n smallest Dirichlet eigenvalues of the circular sector of opening `omega`.

    `lam_{lj} = (j_{l*nu, j} / r)**2` with `nu = pi/omega` and `l >= 1`. Simple for
    `omega <= pi`, and the sharpest reference in the suite: the eigenfunction is
    `J_{l*nu}(k r) sin(l*nu*theta)`, which is EXACTLY one Fourier-Bessel column of the
    corner family at the apex. A single-column basis reproduces it to machine precision.
    """
    import mpmath as mp

    nu = np.pi / omega
    with mp.workdps(MP_DPS):
        # mpmath, not scipy: `jn_zeros` takes integer orders only, and `l*nu` is fractional
        # for every sector that is not a half-disk.
        lam = [float(mp.besseljzero(mp.mpf(l * nu), j)) ** 2 / r**2
               for l in range(1, n + 1) for j in range(1, n + 1)]
    return np.sort(np.array(lam))[:n]


def eq_tri_eigs(l=1.0, n=10):
    """The n smallest Dirichlet eigenvalues of the equilateral triangle of side `l`.

    Lame's closed form: `lam_{mn} = (16*pi**2 / (9*l**2)) * (m**2 + m*n + n**2)` for
    integers `m, n >= 1`. The multiplicity structure is arithmetic, not geometric --
    `m**2 + m*n + n**2` takes some values more than once (e.g. 49 = 7**2 at (3,5) and
    (5,3)), on top of the (m,n)/(n,m) pairing.
    """
    m = np.arange(1, 2 * n + 1)[:, None]
    q = np.arange(1, 2 * n + 1)[None, :]
    lam = (16 * np.pi**2 / (9 * l**2)) * (m**2 + m * q + q**2)
    return np.sort(lam.ravel())[:n]


#: The L-shape's first Dirichlet eigenvalue, the standard corner-singularity benchmark.
#: Quoted from the literature to 20 digits; the domain is `L_SHAPE`, a 2x2 square with a
#: 1x1 bite, whose reentrant corner has `omega = 3*pi/2` and `nu = 2/3`.
L_SHAPE_LAM1 = 9.6397238440219410527


@pytest.fixture
def sector():
    """A circular sector of opening 2*pi/3, radius 1, Dirichlet: the exact-eigenfunction
    reference. Apex at the origin, first edge along the positive real axis."""
    from chladni.geometry import ArcSegment

    omega = 2 * np.pi / 3
    return Domain(
        Boundary(
            [
                LineSegment(0, 1),
                ArcSegment(0j, 1.0, 0.0, omega),
                LineSegment(np.exp(1j * omega), 0),
            ]
        )
    )
