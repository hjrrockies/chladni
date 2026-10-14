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

__all__ = ["Plan", "Recipe", "plan_for", "corner_order", "LADDER", "HEADROOM", "shape_refusal",
           "CLEARANCE_FLOOR"]

@dataclass(frozen=True)
class Recipe:
    """One ladder rung: Fourier-Bessel at `corners` ("all" or "singular") with orders `scale` x
    the precision rule, plus sources at `source_density` per wavelength, offset by
    `source_offset` x the diameter (0 = no sources)."""
    scale: float
    corners: str = "all"
    source_density: float = 0.0
    source_offset: float = 0.0
    lightning: bool = False
    plane_waves: float = 0.0          # x PlaneWave.n_useful(k_max * diameter); 0 = none


#: lappy's lightning setup (`lappy/bases.py`, `FundamentalBasis.by_corners`;
#: `jupyter_notebooks/corner_basis_tests.ipynb`): poles at `L exp(-sigma (sqrt(n) - sqrt(j)))` on
#: the exterior bisector, order 2 (3 columns each), as many columns as the corner terms. With
#: chladni's own defaults (sigma 4, L = the corner's clearance, order 1) the poles pack ~1e-7 L
#: from the vertex and the solve declined on almost every case.
LIGHTNING_SIGMA = 1.0
LIGHTNING_REACH = 0.5                 # x the diameter
LIGHTNING_ORDER = 2


#: The rungs (decided 2026-10-12, `notes/planner_recipes.md`, `notes/hybrid_bases.py`). Rung 2 is
#: rung 1 with larger corner expansions; it was not measured on its own before shipping. Rung 3,
#: lightning, comes LAST although it is cheap: it solved a thin triangle that the offset layer
#: could not (T2.6-N3k5-L6, 100% met at margin 25), and failed one the layer solved. As the last
#: resort it changes no case that an earlier rung already accepts. Rung 4 adds PLANE WAVES to the
#: singular-corner terms, for shapes whose corners are all reflection-closed (rectangles): the
#: singular-corner rungs place nothing there, and a 4:1 rectangle reached no other rung at 1e-10
#: (2026-10-13). Plane waves at 1.3x `n_useful` are `tests/hard_roster.py`'s basis for them.
LADDER = (
    Recipe(1.0),
    Recipe(2.0, "singular", 6.0, 0.05),
    Recipe(3.0, "singular", 6.0, 0.05),
    Recipe(1.0, "singular", lightning=True),
    Recipe(1.0, "singular", plane_waves=1.3),
)
#: Short edges get a local source PATCH on the rungs with an offset layer (2026-10-13,
#: `notes/short_edge_sources.py`). The layer sits `source_offset` x the diameter out, which on
#: douse's collapsing-edge hexagon (an edge 0.042 of a 1.44 diameter) is farther than the edge is
#: long: every rung but 2 missed 1e-10 there, and rung 2 met it at margin 2.0 exactly. Moving the
#: whole layer closer made other cases climb a rung; this patch -- `PATCH_N` sources spread over
#: `PATCH_REACH` x e of boundary either side of the edge's midpoint, `PATCH_OFFSET` x e out --
#: gave margins 5-27 there, took T2.1-rerun from rung 2 to rung 1, and changed no other douse
#: benchmark case. An edge is short below `SHORT_EDGE` x the diameter.
SHORT_EDGE = 0.15
PATCH_REACH = 2.0
PATCH_OFFSET = 1.0 / 3.0
PATCH_N = 9                           # spacing e/2 over 4e of boundary, both ends included

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
    lightning_n: int = 0              # poles per singular corner (order 2: 3 columns each)
    n_plane_waves: int = 0            # directions (2 columns each), a COUNT like n_sources
    edge_patches: tuple = ()          # ((segment index, sources), ...) at short edges
    bdry_ratio: float = 3.0
    int_ratio: float = 1.0

    @property
    def n_cols(self):
        n_light = sum(1 for n in self.corner_orders if n > 0) * int(self.lightning_n)
        return (int(sum(self.corner_orders)) + int(self.n_sources)
                + (2 * LIGHTNING_ORDER - 1) * n_light + 2 * int(self.n_plane_waves)
                + sum(int(m) for _, m in self.edge_patches))

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
        for i, m in self.edge_patches:
            parts.append(_edge_patch(dom, int(i), int(m), self.prec))
        if self.lightning_n:
            from .helmholtz import Fundamental
            for c, n in zip(corners, self.corner_orders):
                if n > 0:
                    parts.append(Fundamental.lightning(
                        c, int(self.lightning_n), sigma=LIGHTNING_SIGMA,
                        L=LIGHTNING_REACH * dom.diameter, order=LIGHTNING_ORDER, prec=self.prec))
        if self.n_plane_waves:
            from .helmholtz import PlaneWave
            parts.append(PlaneWave.uniform(int(self.n_plane_waves), radius=dom.diameter,
                                           prec=self.prec))
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


def _segments(dom):
    return [s for b in dom.boundaries for s in b.segments]


def _short_edges(dom):
    """Indices (in `_segments` order) of the edges shorter than `SHORT_EDGE` x the diameter."""
    return [i for i, s in enumerate(_segments(dom)) if s.length < SHORT_EDGE * dom.diameter]


def _edge_patch(dom, i, m, prec):
    """`m` sources over `PATCH_REACH` x e of boundary either side of edge `i`'s midpoint (e its
    length), each `PATCH_OFFSET` x e out along its own segment's outward normal. Placed by
    arclength from the shape, so the count -- not the positions -- is what the plan fixes."""
    from .helmholtz import Fundamental

    segs = [s for s in dom.boundaries[0].segments]
    if len(segs) != len(_segments(dom)):
        raise ValueError("edge patches are placed on the outer boundary only")
    e = segs[i].length
    lengths = np.array([s.length for s in segs])
    starts = np.concatenate([[0.0], np.cumsum(lengths)[:-1]])
    total = lengths.sum()
    origins = []
    for t in np.linspace(-PATCH_REACH * e, PATCH_REACH * e, int(m)):
        u = (starts[i] + 0.5 * e + t) % total
        j = int(np.searchsorted(starts, u, side="right") - 1)
        tau = (u - starts[j]) / lengths[j]
        origins.append(segs[j].p(tau) + PATCH_OFFSET * e * segs[j].N(tau))
    fam = Fundamental(np.array(origins, dtype=complex), radius=PATCH_OFFSET * e,
                      extent=dom.diameter + e, prec=prec)
    try:
        return fam.check_exterior(dom)
    except Exception as err:  # noqa: BLE001 -- a reflex neighbour pulled a source inside
        raise ValueError(f"the source patch at edge {i} falls inside this domain: {err}") from err


#: A corner whose clearance (distance to the nearest NON-adjacent boundary) is below this
#: fraction of the diameter sits on a near self-intersection: a slit, a pinched neck, a notch a
#: thousandth of the domain wide. No rung resolves such a shape and every rung costs a full
#: solve to find out -- one such trial cost 583 s (chladni/TODO.md, 2026-10-14). Conservative on
#: purpose: a merely SHORT edge (douse's edge floor is perimeter/(10 N)) clears it by an order.
CLEARANCE_FLOOR = 1e-3


def shape_refusal(dom, prec):
    """Why no plan could meet `prec` on `dom`, from its geometry alone, or None. Microseconds:
    `achievable_prec` (float spacing against the bounding box and the shortest edge) and the
    corners' clearance. The planner declines on it before any solve."""
    ap = float(dom.achievable_prec)
    if ap > prec:
        return (f"the geometry supports about {ap:.1e} relative precision in float64 "
                f"(offset or feature contrast), looser than the requested {prec:.0e}")
    D = float(dom.diameter)
    for c in dom.corners:
        clear = float(getattr(c, "clearance", np.inf))
        if np.isfinite(clear) and D > 0 and clear < CLEARANCE_FLOOR * D:
            return (f"a corner at {c.z:.6g} sits {clear:.2e} from a non-adjacent boundary, "
                    f"under {CLEARANCE_FLOOR:g} x the diameter {D:.3g}: a near self-intersection")
    return None


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


def plan_for(dom, n, prec=None, rung=0):
    """The plan at ladder rung `rung` for the first `n` levels of `dom` (plus `HEADROOM`).
    `prec=None` is the domain's own `prec`."""
    from . import asymp

    _require_dirichlet_polygon(dom)
    prec = check_prec(dom.prec if prec is None else prec)
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
    n_sing = sum(1 for o in orders if o > 0)
    n_light = (max(int(round(sum(orders) / ((2 * LIGHTNING_ORDER - 1) * n_sing))), 2)
               if rc.lightning and n_sing else 0)
    n_pw = 0
    if rc.plane_waves > 0:
        from .helmholtz import PlaneWave
        n_pw = int(np.ceil(rc.plane_waves * PlaneWave.n_useful(k_max * dom.diameter, prec)))
    patches = (tuple((i, PATCH_N) for i in _short_edges(dom)) if rc.source_offset > 0 else ())
    return Plan(prec=prec, k_max=k_max, corner_orders=orders, rung=int(rung), n_sources=n_src,
                source_offset=rc.source_offset, lightning_n=n_light, n_plane_waves=n_pw,
                edge_patches=patches)
