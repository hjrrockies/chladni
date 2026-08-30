"""The local expansion at a corner: the exponent family, the angular kind, the branch cut.

This is where `nu = pi/omega` gets computed, and it is the only place. `geometry` stops at
`omega` and the boundary-condition pair on purpose (boundary.py:43) -- lappy crossed that
line with `geometry.pts_per_seg(domain, fb_basis)`, a layer-1 signature taking a layer-2
object, and the whole `Corner` design exists so that function has nowhere to come back.

The corner-local frame, fixed once and used by everything downstream:

    theta = 0       the OUTGOING ray, direction `t_out`
    theta = omega   the incoming ray, direction `-t_in`, reached counter-clockwise
    theta = beta    the branch cut, somewhere in the exterior wedge (omega, 2*pi)

`Boundary.corners` computes `omega = angle(-t_in/t_out) % (2*pi)` (boundary.py:319), so
`arg((z - c.z)/t_out)` lying in `(0, omega)` IS the interior wedge, by construction and with
no hole branch: `Domain` reverses holes to clockwise, which puts Omega on the left of travel
on every boundary, so `-1j*T` is outward everywhere and one convention covers both.

The quantization decouples, which is the piece lappy never had. `sin(mu*0) = 0` and
`cos'(mu*0) = 0` hold for free, so the condition on the theta = 0 ray only chooses sin against
cos; the theta = omega ray then forces `sin(mu*omega) = 0` or `cos(mu*omega) = 0`, i.e. an
integer or half-integer multiple of pi:

    bc at theta=0   bc at theta=omega   angular       mu                     leading
    dir           dir               sin(mu*theta)   l*nu,  l = 1,2,...     nu
    neu           neu               cos(mu*theta)   l*nu,  l = 0,1,...     0
    dir           neu               sin(mu*theta)   (l + 1/2)*nu           nu/2
    neu           dir               cos(mu*theta)   (l + 1/2)*nu           nu/2

Two booleans, one code path. Keeping theta = 0 on `t_out` in all four cases -- rather than
rotating the frame onto "the Dirichlet ray" in the mixed case -- is what makes that true.

The last two rows are Zaremba corners. At a FLAT junction, omega = pi, the leading exponent
is mu = 1/2: the classic square-root singularity, stronger than any convex Dirichlet corner,
at a junction `Corner.is_geometric` cannot see. `chladni` detects it one layer down through
`is_bc_junction`, which lappy did not -- its `eigfun_integrals.corner_specs` meets the same
corner three layers up and declares it unhandled.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: How close to an integer an exponent must be before a hole corner will accept it. The
#: comparison is on `mu` itself rather than on a ratio, because `nu = pi/omega` inherits
#: omega's own roundoff and an exact-equality test would refuse a legitimate right angle.
_INTEGRAL_ATOL = 1e-12


def _near_integer(v):
    """Within `_INTEGRAL_ATOL` of an integer, from either side.

    `v % 1.0 <= atol` alone is wrong: `pi/(pi/2)` can land at 1.9999999999999998, whose
    residue is 0.9999999999999998 and which is an integer for every purpose here.
    """
    frac = float(v) % 1.0
    return min(frac, 1.0 - frac) <= _INTEGRAL_ATOL


def _singularity_family(bc):
    """Which local family a boundary condition puts a corner in: 'dir' or 'neu'.

    Robin maps to 'neu', and that is a real approximation rather than a naming convenience.
    There is no closed-form separable local solution at a Robin corner: applying
    `-(1/r)*d_dtheta(u) + h*u = 0` to `u = J_mu(k*r)*f(theta)` gives `f'(0)/r = h*f(0)`, an
    r-dependent condition no single `(mu, f)` satisfies, and the 1/r is structural. But as
    r -> 0 that term dominates `h*u`, so the leading local asymptotics are Neumann's
    (Grisvard). The resulting family is SINGULARITY-adapted, not BC-adapted: it carries the
    right leading exponent and is Helmholtz-exact, which is all MPS requires, since the
    boundary condition is imposed by least squares and not built into the basis.
    """
    return "dir" if bc == 0.0 else "neu"


@dataclass(frozen=True)
class CornerModes:
    """The corner-adapted expansion at one corner, before any point set is involved.

    `orders(degrees)` turns integer degrees into Bessel orders. `beta` is the branch cut in
    the corner-local frame, NaN when the corner needs none (integral `mu`) or has none.
    """

    nu: float
    kind: str            # 'sin' | 'cos'
    offset: float        # 0.0, or 0.5 at a mixed-BC (Zaremba) corner
    omega: float
    beta: float = float("nan")
    #: False when a Robin condition was mapped onto the Neumann family. The functions are
    #: still exact solutions and still carry the right singularity; they simply do not
    #: satisfy the stated boundary condition termwise, and a docstring that claimed
    #: otherwise would be the kind of quiet overstatement this project tries not to make.
    bc_adapted: bool = True

    @property
    def first_degree(self):
        """1 for the dir/dir family, 0 otherwise.

        `sin(0*theta)` is the zero function, so degree 0 there would put a column of exact
        zeros into the pencil for the GSVD's rank truncation to find and discard.
        """
        return 1 if (self.kind == "sin" and self.offset == 0.0) else 0

    def degrees(self, n):
        """The first `n` degrees of this family."""
        return np.arange(self.first_degree, self.first_degree + int(n))

    def orders(self, degrees):
        """`mu = (l + offset)*nu` -- the Bessel order of each column."""
        return (np.asarray(degrees, dtype=float) + self.offset) * self.nu

    @property
    def leading_exponent(self):
        """The smallest `mu` this family carries: the corner's own singularity strength."""
        return float(self.orders([self.first_degree])[0])

    @property
    def is_single_valued(self):
        """Whether every order is an integer, so no branch cut is needed at all.

        True at a right-angled Dirichlet corner (nu = 2), which is why a square needs no cut.
        """
        return _near_integer(self.nu) and _near_integer(self.offset * self.nu)

    @classmethod
    def from_wedge(cls, omega, bc_out, bc_in, beta=None, admits_branch_cut=True):
        """From the bare parameters, with no geometry object involved.

        The project's rule is that a family is constructible from what defines it, so that a
        bespoke basis never has to be routed through ghost geometry. `beta` defaults to the
        exterior bisector, which is what `Boundary._branch_angles` chooses whenever a
        straight ray clears Omega.
        """
        omega = float(omega)
        if not 0.0 < omega < 2 * np.pi:
            raise ValueError(f"interior angle must lie in (0, 2*pi), got {omega!r}")
        out, inn = _singularity_family(bc_out), _singularity_family(bc_in)
        kind = "sin" if out == "dir" else "cos"
        offset = 0.5 if out != inn else 0.0
        beta = omega + 0.5 * (2 * np.pi - omega) if beta is None else float(beta)
        modes = cls(
            nu=np.pi / omega,
            kind=kind,
            offset=offset,
            omega=omega,
            beta=beta,
            bc_adapted=(bc_out in (0.0, 1.0) and bc_in in (0.0, 1.0)),
        )
        if not admits_branch_cut and not modes.is_single_valued:
            raise ValueError(
                f"a corner on a hole admits no branch cut, and this family is multivalued "
                f"without one: omega = {omega:.6g} gives nu = {modes.nu:.6g} and orders "
                f"{modes.offset:g} + l times that, which are not all integers. The "
                f"obstruction is topological -- a loop in Omega encircling the hole winds "
                f"around every point of that hole's boundary, so a cut confined to the hole "
                f"blocks nothing and a cut reaching the outer boundary crosses Omega. Only "
                f"omega = pi/m (and, for a mixed-BC corner, an even nu) is usable here; "
                f"place a fundamental solution inside the hole instead."
            )
        return modes

    @classmethod
    def from_corner(cls, corner):
        """From a `geometry.Corner`. The branch cut is read in the corner-local frame.

        `Corner.branch_angle` is an absolute direction; `beta` is that direction measured
        from `t_out`, which is the frame every downstream angle lives in. Converting here
        rather than at each use is what keeps a frame error from becoming six of them.
        """
        beta = None
        if np.isfinite(corner.branch_angle):
            beta = (corner.branch_angle - np.angle(corner.t_out)) % (2 * np.pi)
        return cls.from_wedge(
            corner.omega,
            bc_out=corner.bc_out,
            bc_in=corner.bc_in,
            beta=beta,
            admits_branch_cut=corner.admits_exterior_branch_cut,
        )
