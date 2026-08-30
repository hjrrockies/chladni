"""Deriving the local expansion at a corner -- this layer's first job.

`geometry` stops at `omega` and the boundary-condition pair and never computes `nu = pi/omega`
(boundary.py:43). That line is load-bearing: lappy crossed it with
`geometry.pts_per_seg(domain, fb_basis)`, and the whole `Corner` design exists so that
function has nowhere to come back. This file defends the crossing happening here and only
here, and defends the exponent table being right.

The angular conditions are checked against the mathematics directly -- `sin(mu*theta)` on a ray
is a two-line assertion, not something that needs the rest of the module -- so a wrong table
fails here rather than three files later as a mysteriously poor eigenvalue.
"""
import numpy as np
import pytest

from conftest import EXACT
from chladni.geometry import Polygon
from chladni.helmholtz.corners import CornerModes


def _modes(omega, bc_out, bc_in):
    """A CornerModes for a bare wedge, no geometry object involved. The design principle is
    that a family is constructible from its own parameters and does not need ghost geometry;
    this is the same rule one level down."""
    return CornerModes.from_wedge(omega, bc_out=bc_out, bc_in=bc_in)


# --- nu ------------------------------------------------------------------------------------

@pytest.mark.parametrize("omega", [np.pi / 2, np.pi, 2 * np.pi / 3, 3 * np.pi / 2, 1.9 * np.pi])
def test_nu_is_pi_over_theta(omega):
    assert _modes(omega, 0.0, 0.0).nu == pytest.approx(np.pi / omega, abs=EXACT)


def test_a_right_angle_has_integer_nu():
    """nu = 2 at a square corner, which is what makes a square's FB family single-valued
    with no branch cut at all."""
    assert _modes(np.pi / 2, 0.0, 0.0).nu == pytest.approx(2.0, abs=EXACT)


def test_the_l_shape_reentrant_corner_has_nu_two_thirds():
    """omega = 3*pi/2, nu = 2/3 -- the exponent that costs the L-shape about seven digits
    when it is missing from the basis."""
    assert _modes(3 * np.pi / 2, 0.0, 0.0).nu == pytest.approx(2 / 3, abs=EXACT)


# --- the four-row table --------------------------------------------------------------------

TABLE = [
    # bc_out (the theta = 0 ray), bc_in (the theta = omega ray), kind, offset, first degree
    (0.0, 0.0, "sin", 0.0, 1),
    (1.0, 1.0, "cos", 0.0, 0),
    (0.0, 1.0, "sin", 0.5, 0),
    (1.0, 0.0, "cos", 0.5, 0),
]


@pytest.mark.parametrize("bc_out,bc_in,kind,offset,first", TABLE)
def test_the_exponent_family_decouples(bc_out, bc_in, kind, offset, first):
    """The BC on the theta = 0 ray picks sin against cos; the BC on the theta = omega ray picks
    the half-integer offset. Two independent booleans, one code path -- which is why the
    mixed case costs nothing over the unmixed one."""
    m = _modes(1.3 * np.pi, bc_out, bc_in)
    assert m.kind == kind
    assert m.offset == offset
    assert m.first_degree == first


def test_dirichlet_dirichlet_skips_the_zero_degree():
    """l = 0 gives sin(0), the zero function. Including it would put a column of exact zeros
    into the pencil, which the GSVD's rank truncation would then have to discard."""
    m = _modes(np.pi / 2, 0.0, 0.0)
    assert m.first_degree == 1
    assert m.degrees(3).tolist() == [1, 2, 3]


def test_neumann_neumann_keeps_the_zero_degree():
    """l = 0 gives cos(0) = 1, i.e. the constant times J_0 -- a legitimate and, at a Neumann
    corner, the leading mode."""
    m = _modes(np.pi / 2, 1.0, 1.0)
    assert m.first_degree == 0
    assert m.degrees(3).tolist() == [0, 1, 2]


@pytest.mark.parametrize("bc_out,bc_in,kind,offset,first", TABLE)
def test_the_modes_satisfy_their_own_boundary_conditions(bc_out, bc_in, kind, offset, first):
    """Checked against the angular factor directly, with no part of this module involved
    beyond the exponents themselves.

    On either ray the outward normal is +-(1/r)*d_dtheta, so a Neumann condition is
    `d_dtheta(angular) = 0` and a Dirichlet condition is `angular = 0`.
    """
    omega = 1.9 * np.pi
    m = _modes(omega, bc_out, bc_in)
    ang = np.sin if m.kind == "sin" else np.cos
    dang = (lambda t: np.cos(t)) if m.kind == "sin" else (lambda t: -np.sin(t))
    for mu in m.orders(m.degrees(4)):
        for theta, bc in ((0.0, bc_out), (omega, bc_in)):
            got = ang(mu * theta) if bc == 0.0 else dang(mu * theta)
            assert abs(got) < 1e-14, f"mu={mu} theta={theta} bc={bc}"


def test_the_first_ray_is_satisfied_structurally_and_the_second_only_numerically():
    """An asymmetry worth knowing before writing an assertion against it: `sin(mu*0)` is
    exactly 0.0 by construction, while `sin(mu*omega)` is 0 only up to roundoff in omega.
    A test demanding exact zero must target the theta = 0 ray."""
    omega = 1.9 * np.pi
    m = _modes(omega, 0.0, 0.0)
    mus = m.orders(m.degrees(4))
    assert np.all(np.sin(mus * 0.0) == 0.0)
    assert np.max(np.abs(np.sin(mus * omega))) < 1e-14
    assert np.max(np.abs(np.sin(mus * omega))) > 0.0


# --- Zaremba -------------------------------------------------------------------------------

def test_a_flat_bc_junction_is_the_classic_square_root_singularity():
    """omega = pi, so nu = 1 and the leading mixed exponent is 1/2. This is a STRONGER
    singularity than any convex Dirichlet corner, at a junction `Corner.is_geometric` does
    not see at all -- lappy detected it three layers up, as a reason to bail out."""
    m = _modes(np.pi, 0.0, 1.0)
    assert m.nu == pytest.approx(1.0, abs=EXACT)
    assert m.orders(m.degrees(1))[0] == pytest.approx(0.5, abs=EXACT)


def test_the_mixed_leading_exponent_is_half_the_unmixed_one():
    for omega in (np.pi / 2, np.pi, 3 * np.pi / 2):
        plain = _modes(omega, 0.0, 0.0)
        mixed = _modes(omega, 0.0, 1.0)
        assert mixed.orders(mixed.degrees(1))[0] == pytest.approx(
            0.5 * plain.orders(plain.degrees(1))[0], abs=EXACT)


# --- Robin ---------------------------------------------------------------------------------

def test_a_robin_corner_takes_the_neumann_family():
    """There is no closed-form separable local solution at a Robin corner: the condition
    reads f'(0)/r = h*f(0), and the 1/r is structural. As r -> 0 it dominates h*u, so Robin
    degenerates to Neumann in the leading asymptotics (Grisvard). The family is therefore
    SINGULARITY-adapted, not BC-adapted -- still Helmholtz-exact, which is all MPS needs."""
    m = _modes(1.3 * np.pi, 0.35, 0.35)
    assert m.kind == "cos"
    assert m.offset == 0.0
    assert m.bc_adapted is False


def test_a_robin_dirichlet_junction_takes_the_mixed_family():
    m = _modes(1.3 * np.pi, 0.35, 0.0)
    assert (m.kind, m.offset) == ("cos", 0.5)
    assert m.bc_adapted is False


def test_two_different_robin_parameters_are_not_a_junction():
    """Both sides degenerate to Neumann, so the exponent family is the same on both rays --
    the Robin parameters differ but the SINGULARITY does not."""
    m = _modes(1.3 * np.pi, 0.2, 0.8)
    assert (m.kind, m.offset) == ("cos", 0.0)


def test_a_pure_dirichlet_corner_is_bc_adapted():
    assert _modes(1.3 * np.pi, 0.0, 0.0).bc_adapted is True
    assert _modes(1.3 * np.pi, 0.0, 1.0).bc_adapted is True


# --- corners from geometry -------------------------------------------------------------------

def test_from_corner_reads_theta_and_the_bc_pair_off_the_geometry_object():
    sq = Polygon(np.array([0, 1, 1 + 1j, 1j], dtype=complex))
    for c in sq.corners:
        m = CornerModes.from_corner(c)
        assert m.nu == pytest.approx(np.pi / c.omega, abs=EXACT)
        assert m.kind == "sin"


def test_beta_lands_in_the_exterior_wedge():
    """The branch cut has to leave through the exterior, or it cuts Omega. Measured in the
    corner-local frame that means omega < beta < 2*pi."""
    from conftest import L_SHAPE

    for c in Polygon(L_SHAPE).corners:
        m = CornerModes.from_corner(c)
        assert c.omega < m.beta < 2 * np.pi


def test_beta_is_the_exterior_bisector_for_a_polygon():
    """Geometry sets `branch_angle` to the exterior bisector where a ray suffices, so in
    the corner-local frame beta is exactly halfway across the exterior wedge. Checking the
    relative measure is what catches a frame error that an absolute angle would hide."""
    from conftest import L_SHAPE

    for c in Polygon(L_SHAPE).corners:
        m = CornerModes.from_corner(c)
        assert m.beta == pytest.approx(c.omega + (2 * np.pi - c.omega) / 2, abs=1e-12)


# --- the hole obstruction ----------------------------------------------------------------------

def _square_with_hole():
    from chladni.geometry import Boundary, LineSegment

    outer = np.array([0, 6, 6 + 6j, 6j], dtype=complex)
    hole = np.array([2 + 2j, 2 + 4j, 4 + 4j, 4 + 2j], dtype=complex)   # CW
    mk = lambda v: Boundary([LineSegment(v[i], v[(i + 1) % len(v)]) for i in range(len(v))])
    from chladni.geometry import Domain

    return Domain(mk(outer), holes=[mk(hole)])


def test_a_hole_corner_with_integer_nu_is_accepted():
    """A square hole corner has omega = 3*pi/2 measured inside Omega, so nu = 2/3 -- NOT
    integral, and rejected. This test pins the acceptance path instead, on a hole corner
    whose interior angle is pi/2 so that nu = 2."""
    m = CornerModes.from_wedge(np.pi / 2, 0.0, 0.0, admits_branch_cut=False)
    assert m.nu == pytest.approx(2.0, abs=EXACT)
    assert np.all(m.orders(m.degrees(4)) % 1.0 == 0.0)


def test_a_hole_corner_with_fractional_nu_is_refused():
    """The obstruction is topological, not a matter of searching harder for a direction: a
    loop in Omega encircling the hole winds around every point of that hole's boundary, so
    no cut confined to the hole kills it and no cut reaching the outer boundary avoids Omega.
    A fractional mu there would be multivalued. Refuse, with the reason."""
    with pytest.raises(ValueError, match="hole"):
        CornerModes.from_wedge(3 * np.pi / 2, 0.0, 0.0, admits_branch_cut=False)


def test_the_square_hole_corners_of_a_real_domain_are_refused():
    dom = _square_with_hole()
    holes = [c for c in dom.corners if c.on_hole]
    assert len(holes) == 4
    for c in holes:
        assert c.omega == pytest.approx(3 * np.pi / 2, abs=1e-12)
        assert not c.admits_exterior_branch_cut
        with pytest.raises(ValueError, match="hole"):
            CornerModes.from_corner(c)


def test_a_mixed_bc_hole_corner_needs_an_even_nu():
    """The mixed family carries mu = (l + 1/2)*nu, so single-valuedness needs nu/2 integral,
    not just nu. A right-angled hole corner (nu = 2) passes; nu = 1 does not."""
    CornerModes.from_wedge(np.pi / 2, 0.0, 1.0, admits_branch_cut=False)
    with pytest.raises(ValueError, match="hole"):
        CornerModes.from_wedge(np.pi, 0.0, 1.0, admits_branch_cut=False)


# --- immutability, as everywhere else in the project ------------------------------------------

def test_corner_modes_are_frozen():
    m = _modes(np.pi / 2, 0.0, 0.0)
    with pytest.raises(Exception):
        m.nu = 1.0


def test_an_exponent_a_hair_below_an_integer_still_counts_as_one():
    """`nu % 1 <= atol` alone would refuse 1.9999999999999998, whose residue is 0.99999...,
    and pi/(pi/2) is exactly the kind of expression that lands there."""
    from chladni.helmholtz.corners import CornerModes as CM

    m = CM(nu=1.9999999999999998, kind="sin", offset=0.0, omega=np.pi / 2)
    assert m.is_single_valued
    m = CM(nu=2.0000000000000004, kind="sin", offset=0.5, omega=np.pi / 2)
    assert m.is_single_valued
