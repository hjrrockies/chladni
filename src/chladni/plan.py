"""The basis PLAN: what to solve a Dirichlet polygon with, relative to its shape.

`docs/design/douse_requirements.md`, "Plans: plan once, realize per iterate". A `Plan` is
frozen and picklable. It holds the choices, not the basis, so `realize(dom)` builds the same size
of basis on any domain with the same corner structure: douse plans once and realizes on every
nearby iterate.

**What a plan contains** was measured, not assumed (`notes/planner_recipes.md`,
`notes/hard_cases.md`, `notes/hybrid_bases.py`; the douse benchmark at 1e-10).
- The SIZE of the corner expansions decides most shapes. Rung 0 is Fourier-Bessel at every
  corner and nothing else.
- The shapes it cannot reach -- slow-converging pentagons, thin triangles -- need a HYBRID:
  Fourier-Bessel at the SINGULAR corners only, plus a boundary-offset source layer (6 per
  wavelength at 5% of the diameter). That reached 100% of levels on a pentagon that corner terms
  left at 27%, and on a thin triangle left at 10%. Rungs 1-2 are that hybrid.
- Corner-clustered "lightning" sources were left out. With chladni's tight defaults they
  declined everywhere, and clustering produces very ill-conditioned columns.
- Collocation density was never binding.

**Sizing.** lappy's per-corner rule takes orders up to `k r + 2 (k r)^(1/3) + 3`, with `r` the
distance to the farthest vertex. It has no precision in it. This takes orders up to
`k r + t (k r)^(1/3) + 3`, where `t = 2^(-1/3) (1.5 ln(1/prec))^(2/3)` is where `J_nu(k r)` has
decayed to `prec` past its turning point (Airy asymptotics): 8.4 at 1e-10, 7.3 at 1e-8. That is
`n = ceil(scale * nu_max / alpha)` terms per corner, `alpha = pi/omega`, at the ladder rung's
`scale`. At scale 1 it passed as many benchmark cases as lappy's rule at 2x, with 20% fewer
columns.

**The ladder** (`LADDER`): the planner climbs it and stops at the first rung that meets the
request. It does not hill-climb on column count, because enrichment is not monotone (on the
L-shape 40 terms per corner beat 80). Some sharp, thin-wedged shapes reach no rung, and the
planner ends those FLAGGED: honest, not solved.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._core import PREC, check_prec

__all__ = ["Plan", "Recipe", "plan_for", "corner_order", "LADDER", "HEADROOM"]

@dataclass(frozen=True)
class Recipe:
    """One ladder rung: Fourier-Bessel at `corners` ("all" or "singular") with orders `scale` x
    the precision rule, plus sources at `source_density` per wavelength, offset by
    `source_offset` x the diameter (0 = no sources)."""
    scale: float
    corners: str = "all"
    source_density: float = 0.0
    source_offset: float = 0.0


#: The rungs, cheapest first (decided 2026-10-12, `notes/planner_recipes.md`,
#: `notes/hybrid_bases.py`). Rung 2 is rung 1 with larger corner expansions; it was not measured
#: on its own before shipping.
LADDER = (
    Recipe(1.0),
    Recipe(2.0, "singular", 6.0, 0.05),
    Recipe(3.0, "singular", 6.0, 0.05),
)
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
    corner_orders: tuple              # 0 = no Fourier-Bessel at that corner
    rung: int = 0
    n_sources: int = 0                # a COUNT, so the size is identical on nearby shapes
    source_offset: float = 0.0        # x the diameter
    bdry_ratio: float = 3.0
    int_ratio: float = 1.0

    @property
    def n_cols(self):
        return int(sum(self.corner_orders)) + int(self.n_sources)

    def realize(self, dom):
        """`(basis, n_bdry, n_int)` on `dom`, which must have the plan's corner structure."""
        from .helmholtz import FourierBessel

        corners = dom.corners
        if len(corners) != len(self.corner_orders):
            raise ValueError(f"the plan was made for {len(self.corner_orders)} corners and this "
                             f"domain has {len(corners)}: a different corner structure is a "
                             f"re-plan, not a realization")
        parts = [FourierBessel.at_corner(c, n=int(n), domain=dom, prec=self.prec)
                 for c, n in zip(corners, self.corner_orders) if n > 0]
        if self.n_sources:
            parts.append(_offset_sources(dom, self.n_sources, self.source_offset, self.prec))
        if not parts:
            raise ValueError("the plan has neither corner terms nor sources")
        basis = parts[0]
        for f in parts[1:]:
            basis = basis + f
        m = len(basis)
        return basis, int(np.ceil(self.bdry_ratio * m)), int(np.ceil(self.int_ratio * m))

    def solver(self, dom, *, prec=None, rng=0):
        """An `MPSEigensolver` for `dom` built from this plan, at `prec` (default: the plan's).
        A tighter `prec` than the plan's is solved on the plan's basis and reports its
        shortfall; it does not enlarge the basis."""
        from .mps import MPSEigensolver

        basis, n_bdry, n_int = self.realize(dom)
        return MPSEigensolver(dom, basis=basis, n_bdry=n_bdry, n_int=n_int,
                              prec=self.prec if prec is None else prec, rng=rng)


def _offset_sources(dom, n, offset, prec):
    """`n` fundamental-solution sources at `offset` x the diameter outside the boundary, moved
    closer (halving the offset, twice) where a thin domain would put them inside, as the
    benchmark's recipe B does."""
    from .helmholtz import Fundamental

    err = None
    for f in (offset, offset / 2, offset / 4):
        try:
            return Fundamental.offset(dom.bdry_quad(int(n)), f * dom.diameter, domain=dom,
                                      prec=prec)
        except Exception as e:  # noqa: BLE001 -- SourcesInsideDomain on a thin domain
            err = e
    raise ValueError(f"no source offset down to {offset / 4:g} x the diameter keeps the sources "
                     f"outside this domain: {err}")


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
    rc = LADDER[int(rung)]
    lam_top = float(asymp.weyl_est_poly(int(n) + HEADROOM, dom)) * (1.0 + _TOP_PAD)
    k_max = float(np.sqrt(lam_top))
    zs = np.array([c.z for c in dom.corners])
    orders = tuple(
        0 if (rc.corners == "singular" and c.is_reflection_closed)
        else corner_order(k_max, float(np.max(np.abs(zs - c.z))), c.omega, prec, rc.scale)
        for c in dom.corners)
    n_src = (int(np.ceil(rc.source_density * k_max * dom.perimeter / (2 * np.pi)))
             if rc.source_density > 0 else 0)
    return Plan(prec=prec, k_max=k_max, corner_orders=orders, rung=int(rung), n_sources=n_src,
                source_offset=rc.source_offset)
