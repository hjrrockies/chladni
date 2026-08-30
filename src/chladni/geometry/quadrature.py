"""Boundary quadrature: nodes, weights, and provenance.

lappy's `Segment.pts(n, kind='legendre')` computes `tau, _wts = quadfunc(n)` and throws the
weights away; `MultiSegment.pts` then reassembles per-segment results with
`np.sum([PointSet, ...])`, using `+` as concatenation. The Rellich identity integrates
(x - x0).n against |grad u|^2 over the boundary, so the weights have to reach the solver.
They do, here, in the same object as the nodes.

Fields mirror lappy's `BoundaryQuad`, which its own comment describes as "geometry-only
boundary quadrature: no domain reference, no lam dependence, no basis" -- a layer-1 object
that had ended up in eigfun_integrals.py. `seg_idx` and `tau` are each node's provenance,
carried for the reason lappy gives: without them a downstream shape package has to reach
into private helpers and redo the affine panel map by hand.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._adaptive import allocate, gauss_legendre

#: Corner grading: geometric panels toward a graded endpoint.
_GRADE_LAYERS = 4
_GRADE_RATIO = 0.25

#: Largest Gauss order used on a single panel; above this, subdivide instead.
_MAX_PANEL_ORDER = 32


@dataclass(frozen=True)
class Quadrature:
    """A boundary quadrature rule, with every node's provenance.

    `integrate(f)` is always ``sum(f(pts) * wts)``. What that equals depends on the rule:

    * ``weight_family == 'arclength'`` -- the weight function is 1, so ``integrate(f)`` is
      ``oint f ds`` and ``wts.sum()`` is the boundary length. Everything the geometry layer
      builds is in this family, grading included: grading relocates nodes, it does not
      change the measure.
    * any other family -- a singular weight ``w`` is folded into the weights, so
      ``integrate(f)`` is ``oint f w ds`` and ``wts.sum()`` is ``oint w ds``, NOT the
      perimeter. The caller supplies ``f = integrand / w``.

    The second case is what the Rellich boundary integrals need: near a singular corner the
    integrand carries ``r^(2 nu - 2)``, and only a Gauss-Jacobi rule built around that
    exponent resolves it. Such rules need ``nu``, so they are built one layer up -- but they
    come back as a `Quadrature`, which is why `weight_family` lives here rather than in the
    layer that sets it. lappy reached the same conclusion; its values ``'even'``
    (``sub = nu``, matched to the eigenfunction's own exponent family) and ``'integer'``
    (``sub = 1/2``, exact on the integer powers a corner-moving shape velocity supplies)
    carry over unchanged, and its comment records that the two "are not interchangeable in
    either direction".
    """

    pts: np.ndarray
    normals: np.ndarray
    tangents: np.ndarray
    wts: np.ndarray
    tau: np.ndarray
    seg_idx: np.ndarray
    bdry_idx: np.ndarray
    bc: np.ndarray
    panel_id: np.ndarray
    panels: tuple = ()
    weight_family: str = "arclength"

    def __len__(self):
        return len(self.pts)

    @property
    def dir_mask(self):
        return self.bc == 0.0

    @property
    def neu_mask(self):
        return self.bc == 1.0

    @property
    def rob_mask(self):
        return (self.bc != 0.0) & (self.bc != 1.0)

    def integrate(self, f):
        """``sum(f(pts) * wts)``. See the class docstring for what that equals."""
        return np.sum(np.asarray(f(self.pts)) * self.wts)

    def radial(self, x0=0j):
        """``(x - x0) . n`` at every node.

        A method rather than a stored field: the Rellich base point is a solver choice,
        while the arithmetic is pure geometry.
        """
        d = self.pts - x0
        return d.real * self.normals.real + d.imag * self.normals.imag

    @classmethod
    def concat(cls, parts):
        """Join per-boundary rules, renumbering panel ids so they stay unique."""
        parts = [p for p in parts if len(p)]
        if not parts:
            raise ValueError("nothing to concatenate")
        families = {p.weight_family for p in parts}
        if len(families) > 1:
            raise ValueError(f"cannot concatenate mixed weight families: {sorted(families)}")
        offset = 0
        panel_ids = []
        for p in parts:
            panel_ids.append(p.panel_id + offset)
            offset += int(p.panel_id.max()) + 1 if len(p.panel_id) else 0
        cat = lambda name: np.concatenate([getattr(p, name) for p in parts])
        return cls(
            pts=cat("pts"),
            normals=cat("normals"),
            tangents=cat("tangents"),
            wts=cat("wts"),
            tau=cat("tau"),
            seg_idx=cat("seg_idx"),
            bdry_idx=cat("bdry_idx"),
            bc=cat("bc"),
            panel_id=np.concatenate(panel_ids),
            panels=tuple(pan for p in parts for pan in p.panels),
            weight_family=parts[0].weight_family,
        )


def _graded_edges(grade_start, grade_end, layers=_GRADE_LAYERS, ratio=_GRADE_RATIO):
    """Panel edges on [0, 1], geometrically refined toward the graded ends."""
    if not grade_start and not grade_end:
        return [0.0, 1.0]
    geom = [ratio**k for k in range(layers, 0, -1)]  # e.g. 1/256, 1/64, 1/16, 1/4
    if grade_start and grade_end:
        left = [0.5 * g for g in geom]
        right = [1.0 - 0.5 * g for g in reversed(geom)]
        return [0.0] + left + [0.5] + right + [1.0]
    if grade_start:
        return [0.0] + geom + [1.0]
    return [0.0] + [1.0 - g for g in reversed(geom)] + [1.0]


def panel_edges(seg, grade_start=False, grade_end=False):
    """Panel boundaries in the segment's tau, honouring its derivative breaks.

    A panel spanning a break sees an integrand that is not analytic across it, so a Gauss
    rule over the whole panel loses spectral convergence no matter how accurate the
    parameterization is. lappy measured 8.3e-11 at 96 knot-aligned nodes against 1.3e-06 at
    128 with one global panel.
    """
    edges = _graded_edges(grade_start, grade_end)
    brk = np.asarray(seg.break_taus, dtype=float)
    if len(brk):
        edges = np.concatenate([edges, brk])
    edges = np.unique(np.asarray(edges, dtype=float))
    return edges[(edges >= 0.0) & (edges <= 1.0)]


def segment_nodes(seg, n, rule="legendre", grade_start=False, grade_end=False, panel0=0):
    """Per-node arrays for one segment: tau, points, frames, arclength weights, panel ids."""
    if rule != "legendre":
        raise ValueError(f"unknown quadrature rule {rule!r}")
    edges = panel_edges(seg, grade_start, grade_end)
    widths = np.diff(edges)
    # Cap the order per panel. np.polynomial.legendre.leggauss is O(order^2): one call at
    # order 1000 costs 39 ms, and a douse loop perturbing vertices changes the per-segment
    # split every iterate, so a cache keyed on order misses every time. Composite panels
    # draw from a handful of small orders instead, which the cache holds for the whole run.
    target = max(len(widths), int(np.ceil(int(n) / _MAX_PANEL_ORDER)))
    if target > len(widths):
        splits = allocate(target, widths, min_each=1)
        edges = np.unique(np.concatenate(
            [np.linspace(lo, hi, r + 1) for lo, hi, r in zip(edges[:-1], edges[1:], splits)]))
        widths = np.diff(edges)
    counts = allocate(n, widths) if len(widths) > 1 else np.array([int(n)])
    x, w = None, None
    taus, wts, pids = [], [], []
    for k, (lo, hi, cnt) in enumerate(zip(edges[:-1], edges[1:], counts)):
        x, w = gauss_legendre(int(cnt))
        taus.append(lo + (hi - lo) * x)
        wts.append((hi - lo) * w * seg.length)
        pids.append(np.full(int(cnt), panel0 + k, dtype=int))
    tau = np.concatenate(taus)
    return {
        "tau": tau,
        "pts": seg.p(tau),
        "tangents": seg.T(tau),
        "normals": seg.N(tau),
        "wts": np.concatenate(wts),
        "panel_id": np.concatenate(pids),
        "n_panels": len(widths),
        "panels": tuple((lo, hi) for lo, hi in zip(edges[:-1], edges[1:])),
    }
