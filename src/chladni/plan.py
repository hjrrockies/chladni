"""The basis PLAN: what to solve a Dirichlet polygon with, relative to its shape.

`docs/design/douse_requirements.md`, "Plans: plan once, realize per iterate". A `Plan` is
frozen and picklable. It holds the choices, not the basis, so `realize(dom)` builds the same size
of basis on any domain with the same corner structure: douse plans once and realizes on every
nearby iterate.

**What a plan contains** was measured, not assumed (`notes/planner_recipes.md`, the douse
benchmark at 1e-10 and 1e-8): Fourier-Bessel terms at every corner, and nothing else.
- The SIZE of the corner expansions is what decides whether a shape reaches `prec`.
- A source layer barely raised the fraction of levels met for the columns it cost.
- Collocation density was never binding (`notes/hard_cases.md`).

**Sizing.** lappy's per-corner rule takes orders up to `k r + 2 (k r)^(1/3) + 3`, with `r` the
distance to the farthest vertex. It has no precision in it. This takes orders up to
`k r + t (k r)^(1/3) + 3`, where `t = 2^(-1/3) (1.5 ln(1/prec))^(2/3)` is where `J_nu(k r)` has
decayed to `prec` past its turning point (Airy asymptotics): 8.4 at 1e-10, 7.3 at 1e-8. That is
`n = ceil(scale * nu_max / alpha)` terms per corner, `alpha = pi/omega`, at the ladder rung's
`scale`. At scale 1 it passed as many benchmark cases as lappy's rule at 2x, with 20% fewer
columns.

**The ladder** (`LADDER`): the planner climbs it and stops at the first rung that meets the
request. It does not hill-climb on column count, because enrichment is not monotone (on the
L-shape 40 terms per corner beat 80). Rung 3 (4x) reaches about the 6x lappy size that the
slow-converging douse shapes need. Sharp, thin-wedged shapes cap near 1e-7 at any size
(`notes/hard_cases.md`), and the planner ends those flagged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._core import PREC, check_prec

__all__ = ["Plan", "plan_for", "corner_order", "LADDER", "HEADROOM"]

#: Rung multipliers on the precision-sized corner orders.
LADDER = (1.0, 2.0, 3.0, 4.0)
#: Levels solved past the n a caller asked for: the count audit's headroom
#: (`notes/count_audit.md`: a drop with four levels above it is caught ~94% of the time).
HEADROOM = 4
#: Padding on the Weyl estimate of the top of the window.
_TOP_PAD = 0.15


def _t(prec):
    """Where `J_nu(x)` has decayed to `prec` past `nu = x`, in units of `x^(1/3)`."""
    return 2.0 ** (-1.0 / 3.0) * (1.5 * np.log(1.0 / prec)) ** (2.0 / 3.0)


def corner_order(k, r, omega, prec, scale=1.0):
    """Fourier-Bessel terms at a corner of angle `omega`, reaching radius `r` at wavenumber `k`."""
    kr = float(k) * float(r)
    nu_max = kr + _t(prec) * kr ** (1.0 / 3.0) + 3.0
    return max(int(np.ceil(scale * nu_max / (np.pi / float(omega)))), 3)


@dataclass(frozen=True)
class Plan:
    """What to solve a Dirichlet polygon with, relative to its shape. See the module docstring.

    `corner_orders[i]` is the number of Fourier-Bessel terms at `dom.corners[i]`. `k_max` and
    `prec` are what the orders were sized for. A solve at a higher wavenumber or a tighter `prec`
    reports its shortfall rather than enlarging the plan: changing either is a re-plan.
    """
    prec: float
    k_max: float
    corner_orders: tuple
    rung: int = 0
    bdry_ratio: float = 3.0
    int_ratio: float = 1.0

    @property
    def n_cols(self):
        return int(sum(self.corner_orders))

    def realize(self, dom):
        """`(basis, n_bdry, n_int)` on `dom`, which must have the plan's corner structure."""
        from .helmholtz import FourierBessel

        corners = dom.corners
        if len(corners) != len(self.corner_orders):
            raise ValueError(f"the plan was made for {len(self.corner_orders)} corners and this "
                             f"domain has {len(corners)}: a different corner structure is a "
                             f"re-plan, not a realization")
        basis = None
        for c, n in zip(corners, self.corner_orders):
            f = FourierBessel.at_corner(c, n=int(n), domain=dom, prec=self.prec)
            basis = f if basis is None else basis + f
        m = len(basis)
        return basis, int(np.ceil(self.bdry_ratio * m)), int(np.ceil(self.int_ratio * m))

    def solver(self, dom, *, rng=0):
        """An `MPSEigensolver` for `dom` built from this plan."""
        from .mps import MPSEigensolver

        basis, n_bdry, n_int = self.realize(dom)
        return MPSEigensolver(dom, basis=basis, n_bdry=n_bdry, n_int=n_int, prec=self.prec,
                              rng=rng)


def _require_dirichlet_polygon(dom):
    from .geometry import Polygon

    if not isinstance(dom, Polygon):
        raise NotImplementedError(
            f"the planner covers Dirichlet polygons only, and this is a {type(dom).__name__}: "
            f"build an MPSEigensolver with an explicit basis")
    if dom.bc_type != "dir":
        raise NotImplementedError(
            f"the planner covers Dirichlet polygons only, and this boundary is "
            f"{dom.bc_type!r}: build an MPSEigensolver with an explicit basis")


def plan_for(dom, n, prec=PREC, rung=0):
    """The plan at ladder rung `rung` for the first `n` levels of `dom` (plus `HEADROOM`)."""
    from . import asymp

    _require_dirichlet_polygon(dom)
    prec = check_prec(prec)
    if not 0 <= int(rung) < len(LADDER):
        raise ValueError(f"rung must lie in [0, {len(LADDER) - 1}], got {rung}")
    lam_top = float(asymp.weyl_est_poly(int(n) + HEADROOM, dom)) * (1.0 + _TOP_PAD)
    k_max = float(np.sqrt(lam_top))
    zs = np.array([c.z for c in dom.corners])
    orders = tuple(corner_order(k_max, float(np.max(np.abs(zs - c.z))), c.omega, prec,
                                LADDER[int(rung)]) for c in dom.corners)
    return Plan(prec=prec, k_max=k_max, corner_orders=orders, rung=int(rung))
