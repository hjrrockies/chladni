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
