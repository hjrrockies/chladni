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

from dataclasses import dataclass, field
from typing import Callable

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

    `integrate(f)` is always ``sum(f(pts) * wts)``, and in EVERY family it approximates
    ``oint f ds``: the measure is always arclength. What the family names is the class of
    integrands the rule is EXACT on.

    * ``weight_family == 'arclength'`` -- smooth integrands. Everything the geometry layer
      builds is in this family, grading included: grading relocates nodes, it does not
      change the class. ``wts.sum()`` is the boundary length.
    * ``'even'`` and ``'integer'`` -- corner rules built one layer up, in
      `chladni.integrals`, around the singular integrands at a corner: ``r^(2 nu - 2)`` times
      a series in ``r^nu`` and ``r^2`` (``'even'``, substitution exponent ``nu``, for the
      Rellich and Gram integrals), or times integer powers of ``r`` (``'integer'``,
      substitution exponent ``1/2``, for a Hadamard weight ``V.n = O(r)`` at a moving corner).
      Their weights already carry the substitution's Jacobian and the Jacobi weight divided
      OUT, so the caller passes the raw integrand. They are NOT exact on constants, so
      ``wts.sum()`` differs from the length -- by design, and ``wts.sum() - length`` per corner
      panel is a self-diagnostic, never something to renormalize away.

    lappy reached the same split, and its comment records that the two corner families "are
    not interchangeable in either direction": ``'even'`` loses ~10 digits on ``r^1``, and
    ``'integer'`` ~10 on the eigenfunction's own series. The family is chosen by the WEIGHT's
    parity, not by the eigenfunction.
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
    #: Corner-local frames. A node on an ANCHORED panel (`Panel.anchor`) carries the corner
    #: it is anchored to and its exact offset from it, `pts == anchor_pt + offset` to
    #: rounding; elsewhere both are NaN. Near a corner `c`, `pts - c` formed from rounded
    #: `pts` is only good to `eps * |c|` absolute, which on a re-entrant corner's
    #: `r^(2 nu - 2)` integrand costs whole digits (`notes/hadamard_accuracy.md`, addendum 2).
    anchor_pt: np.ndarray = None
    offset: np.ndarray = None

    def __post_init__(self):
        nan = np.full(len(self.pts), np.nan + 1j * np.nan)
        if self.anchor_pt is None:
            object.__setattr__(self, "anchor_pt", nan)
        if self.offset is None:
            object.__setattr__(self, "offset", nan.copy())

    @property
    def anchored(self):
        """Mask of the nodes that carry a corner-local frame."""
        return ~np.isnan(self.anchor_pt.real)

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
        parts = list(parts)
        if not parts:
            raise ValueError("nothing to concatenate")
        if any(len(p) == 0 for p in parts):
            # An empty part is a boundary with no nodes -- a hole that would vanish from every
            # integral. Refused rather than dropped (2026-10-14).
            raise ValueError("every part of a concatenated rule needs at least one node; part(s) "
                             f"{[i for i, p in enumerate(parts) if len(p) == 0]} are empty")
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
            anchor_pt=cat("anchor_pt"),
            offset=cat("offset"),
        )


@dataclass(frozen=True)
class Panel:
    """One panel of an externally supplied quadrature plan.

    `nodes(tau0, tau1)` returns `(tau, w)`: node positions inside `[tau0, tau1]`, and
    weights **in the tau variable**, so that `sum(w * f(tau))` approximates
    `integral f(tau) * weight(tau) dtau` over the panel. Geometry multiplies by the
    segment's length to convert dtau into ds, and does nothing else to them.

    That split is deliberate. Rescaling a singular weight onto a shorter panel introduces a
    power of the panel width set by the weight's homogeneity -- `(1-x)^a` picks up
    `h^(1+a)` -- and only the layer that chose the weight knows the exponent. Geometry
    owning that arithmetic would be a silent factor-of-h bug waiting for the first
    corner-adapted rule.

    An ANCHORED panel (`anchor` is `"start"` or `"end"`) touches that end of its segment and
    is laid out in the distance `d` from it: `nodes(d0, d1)` is called with the panel's span
    in `d` (`d = tau` from the start, `1 - tau` from the end) and returns nodes in `d`. The
    node positions are then `p_end + seg.offset(d, end)`, and the rule records each node's
    anchor and offset. Without an anchor, `1 - d` rounds to the nearest representable tau and
    a node `1e-12` from a corner loses its last four digits of distance before anything else
    happens.
    """

    bdry_idx: int
    seg_idx: int
    tau0: float
    tau1: float
    nodes: Callable = field(compare=False, repr=False, default=None)
    anchor: str = None


def legendre_nodes(order):
    """A `Panel.nodes` callable for plain Gauss-Legendre -- the arclength family."""

    def build(tau0, tau1):
        x, w = gauss_legendre(int(order))
        return tau0 + (tau1 - tau0) * x, (tau1 - tau0) * w

    return build


def nodes_from_plan(seg, panels, panel0=0):
    """Materialize a caller's panel plan on one segment: frames, weights, provenance."""
    taus, wts, pids, spans, pts, anchors, offsets = [], [], [], [], [], [], []
    for k, pan in enumerate(panels):
        t0, t1 = float(pan.tau0), float(pan.tau1)
        if pan.anchor is None:
            tau, w = pan.nodes(t0, t1)
            tau = np.asarray(tau, dtype=float)
            pts.append(seg.p(tau))
            anchors.append(np.full(len(tau), np.nan + 1j * np.nan))
            offsets.append(np.full(len(tau), np.nan + 1j * np.nan))
        elif pan.anchor in ("start", "end"):
            at_start = pan.anchor == "start"
            if (t0 if at_start else t1) != (0.0 if at_start else 1.0):
                raise ValueError(f"a panel anchored at the {pan.anchor} must touch it; "
                                 f"got [{t0}, {t1}]")
            d, w = pan.nodes(*((t0, t1) if at_start else (1.0 - t1, 1.0 - t0)))
            d, w = np.asarray(d, dtype=float), np.asarray(w, dtype=float)
            if not at_start:                         # keep tau ascending within the panel
                d, w = d[::-1], w[::-1]
            tau = d if at_start else 1.0 - d
            c = seg.p0 if at_start else seg.p1
            off = np.asarray(seg.offset(d, pan.anchor), dtype=complex)
            pts.append(c + off)
            anchors.append(np.full(len(d), c, dtype=complex))
            offsets.append(off)
        else:
            raise ValueError(f"anchor must be None, 'start' or 'end', got {pan.anchor!r}")
        taus.append(tau)
        wts.append(np.asarray(w, dtype=float) * seg.length)
        pids.append(np.full(len(tau), panel0 + k, dtype=int))
        spans.append((t0, t1))
    tau = np.concatenate(taus)
    return {
        "tau": tau, "pts": np.concatenate(pts), "tangents": seg.T(tau), "normals": seg.N(tau),
        "wts": np.concatenate(wts), "panel_id": np.concatenate(pids),
        "n_panels": len(panels), "panels": tuple(spans),
        "anchor_pt": np.concatenate(anchors), "offset": np.concatenate(offsets),
    }


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
