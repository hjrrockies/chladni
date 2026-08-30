"""Planar domains, simply connected or with holes.

Outer boundaries run counter-clockwise and holes clockwise. That one convention is why holes
cost so little: the outward normal is `-1j * T` on every boundary with no special-casing, and
signed areas simply add, because a hole's clockwise loop contributes negatively.
"""
from __future__ import annotations

from functools import cached_property

import numpy as np
from scipy.optimize import minimize

from ._adaptive import allocate, refine
from ._core import PREC, Frozen, _set, bc_kind
from .boundary import Boundary
from .quadrature import Quadrature
from .segments import LineSegment
from .validate import check_closed, check_holes, check_nondegenerate, check_simple


class Domain(Frozen):
    """A planar domain: one outer boundary, and any number of holes."""

    def __init__(self, outer, holes=(), validate=True):
        if not isinstance(outer, Boundary):
            raise TypeError(f"outer must be a Boundary, got {type(outer).__name__}")
        holes = tuple(holes)
        check_closed(outer, "outer boundary")
        for k, h in enumerate(holes):
            check_closed(h, f"hole {k}")
        check_nondegenerate(outer, "domain")
        if outer.orientation < 0:
            outer = outer.reversed()
        holes = tuple(h.reversed() if h.orientation > 0 else h for h in holes)
        _set(self, outer=outer, holes=holes,
             prec=min([outer.prec] + [h.prec for h in holes]), _validated=bool(validate))
        if validate:
            check_simple(outer, "outer boundary")
            for k, h in enumerate(holes):
                check_simple(h, f"hole {k}")
            check_holes(outer, holes)

    # --- structure ----------------------------------------------------------------------

    @property
    def boundaries(self):
        return (self.outer,) + self.holes

    @property
    def is_simply_connected(self):
        return len(self.holes) == 0

    @cached_property
    def corners(self):
        return tuple(c for b in self.boundaries for c in b.corners)

    @cached_property
    def is_valid(self):
        """The answer to the validation question, whether or not it was asked at build time."""
        try:
            check_simple(self.outer, "outer boundary")
            for k, h in enumerate(self.holes):
                check_simple(h, f"hole {k}")
            check_holes(self.outer, self.holes)
        except ValueError:
            return False
        return True

    @property
    def bc_type(self):
        kinds = {bc_kind(s.bc) for b in self.boundaries for s in b.segments}
        if kinds == {"dir"}:
            return "dir"
        if kinds == {"neu"}:
            return "neu"
        if kinds <= {"dir", "neu"}:
            return "mixed"
        return "rob"

    # --- metric quantities --------------------------------------------------------------

    @cached_property
    def area(self):
        return float(sum(b.signed_area for b in self.boundaries))

    @cached_property
    def perimeter(self):
        return float(sum(b.length for b in self.boundaries))

    @cached_property
    def bbox(self):
        pts = np.concatenate([b.polyline_pts for b in self.boundaries])
        return (complex(pts.real.min(), pts.imag.min()), complex(pts.real.max(), pts.imag.max()))

    @cached_property
    def centroid(self):
        """From the boundary: A*Cx = oint (x^2/2) n_x ds, and likewise for y."""

        def moments(n):
            q = self.bdry_quad(n)
            cx = 0.5 * float((q.pts.real**2 * q.normals.real * q.wts).sum())
            cy = 0.5 * float((q.pts.imag**2 * q.normals.imag * q.wts).sum())
            return cx + 1j * cy

        n0 = 8 * sum(len(b.segments) for b in self.boundaries)
        scale = abs(self.area) * self.diameter
        return complex(refine(moments, n0, self.prec, scale=scale) / self.area)

    def _boundary_samples(self, per_segment=64):
        pts, prov = [], []
        for bi, b in enumerate(self.boundaries):
            for si, seg in enumerate(b.segments):
                tau = np.linspace(0.0, 1.0, per_segment)
                pts.append(seg.p(tau))
                prov.extend((bi, si, t) for t in tau)
        return np.concatenate(pts), prov

    @cached_property
    def diameter(self):
        """Max distance between boundary points: coarse pair, then refined on the true curves.

        A polyline maximum is not enough on its own. For a unit circle at prec=1e-8 the
        adaptive polyline spaces vertices ~7e-4 apart, so the longest chord between vertices
        falls short of the true diameter by ~6e-8 -- a relative error above the working
        precision it was supposed to deliver.
        """
        pts, prov = self._boundary_samples()
        d = np.abs(pts[:, None] - pts[None, :])
        i, j = np.unravel_index(np.argmax(d), d.shape)
        best = float(d[i, j])
        (bi, si, ti), (bj, sj, tj) = prov[i], prov[j]
        sa = self.boundaries[bi].segments[si]
        sb = self.boundaries[bj].segments[sj]
        res = minimize(lambda x: -abs(complex(sa.p(x[0])) - complex(sb.p(x[1]))),
                       np.array([ti, tj]), bounds=[(0.0, 1.0)] * 2, method="L-BFGS-B",
                       options={"ftol": 1e-16, "gtol": 1e-14})
        return max(best, float(-res.fun))

    def boundary_distance(self, pt):
        return np.min([b.dist(pt) for b in self.boundaries], axis=0)

    @cached_property
    def inradius(self):
        """Radius of the largest inscribed disk: grid search, then Nelder-Mead."""
        lo, hi = self.bbox
        m = 60
        gx = np.linspace(lo.real, hi.real, m)
        gy = np.linspace(lo.imag, hi.imag, m)
        z = (gx[:, None] + 1j * gy[None, :]).ravel()
        z = z[self.contains(z)]
        if not len(z):
            return 0.0
        d = self.boundary_distance(z)
        start = z[int(np.argmax(d))]

        def neg(x):
            w = complex(x[0], x[1])
            if not bool(self.contains(np.array([w]))[0]):
                return 0.0
            return -float(self.boundary_distance(np.array([w]))[0])

        res = minimize(neg, np.array([start.real, start.imag]), method="Nelder-Mead",
                       options={"xatol": 1e-13, "fatol": 1e-15, "maxiter": 4000, "maxfev": 4000})
        return max(float(d.max()), float(-res.fun))

    # --- containment --------------------------------------------------------------------

    def contains(self, pts):
        """Ray-crossing parity over every boundary at once.

        Exact for line and arc segments rather than polyline-based: an inscribed polyline
        makes a hole slightly too small, so points in its chord gaps would be reported as
        interior. On the annulus that is the difference between `abs(z) > 1` holding and
        holding to within 6e-8.
        """
        z = np.asarray(pts, dtype=complex)
        total = sum(b.crossings(z) for b in self.boundaries)
        return (np.asarray(total) % 2).astype(bool)

    # --- quadrature and sampling --------------------------------------------------------

    def bdry_quad(self, n=None, *, n_per_bdry=None, rule="legendre", grading=None):
        bs = self.boundaries
        if n_per_bdry is not None:
            counts = np.asarray(n_per_bdry, dtype=int)
        elif n is not None:
            counts = (allocate(int(n), np.array([b.length for b in bs]))
                      if len(bs) > 1 else np.array([int(n)]))
        else:
            raise ValueError("give either n or n_per_bdry")
        return Quadrature.concat([
            b.quad(int(counts[k]), rule=rule, grading=grading, bdry_idx=k) for k, b in enumerate(bs)
        ])

    def interior_pts(self, n, *, method="random", rng=None, oversample=4.0):
        n = int(n)
        lo, hi = self.bbox
        if method == "random":
            gen = np.random.default_rng(rng)
            out = []
            have = 0
            box = (hi.real - lo.real) * (hi.imag - lo.imag)
            factor = max(oversample * box / max(self.area, 1e-300), 2.0)
            for _ in range(40):
                if have >= n:
                    break
                m = int(np.ceil(max(n - have, 16) * factor))
                x = gen.uniform(lo.real, hi.real, m)
                y = gen.uniform(lo.imag, hi.imag, m)
                z = x + 1j * y
                keep = z[self.contains(z)]
                out.append(keep)
                have += len(keep)
            else:
                raise RuntimeError("interior_pts: rejection sampling did not converge")
            return np.concatenate(out)[:n]
        if method == "grid":
            side = int(np.ceil(np.sqrt(n * oversample)))
            for _ in range(12):
                gx = np.linspace(lo.real, hi.real, side + 2)[1:-1]
                gy = np.linspace(lo.imag, hi.imag, side + 2)[1:-1]
                z = (gx[:, None] + 1j * gy[None, :]).ravel()
                z = z[self.contains(z)]
                if len(z) >= n:
                    idx = np.linspace(0, len(z) - 1, n).astype(int)
                    return z[idx]
                side = int(side * 1.5) + 1
            raise RuntimeError("interior_pts: grid did not reach the requested count")
        raise ValueError(f"unknown method {method!r}")

    # --- transforms ---------------------------------------------------------------------

    def _rebuilt(self, outer, holes):
        return Domain(outer, holes, validate=False)

    def translated(self, dz):
        return self._rebuilt(self.outer.translated(dz), tuple(h.translated(dz) for h in self.holes))

    def scaled(self, s, about=0j):
        return self._rebuilt(self.outer.scaled(s, about=about),
                             tuple(h.scaled(s, about=about) for h in self.holes))

    def rotated(self, theta, about=0j):
        return self.scaled(np.exp(1j * float(theta)), about=about)

    def normalized(self, by="diameter"):
        """Rescale so `by` becomes 1. Eigenvalues then transform as lam -> lam * s^2."""
        if by == "area":
            s = 1.0 / np.sqrt(self.area)
        elif by in ("diameter", "inradius"):
            s = 1.0 / getattr(self, by)
        else:
            raise ValueError(f"unknown normalization {by!r}")
        return self.scaled(s)

    def with_bc(self, bc):
        return self._rebuilt(self.outer.with_bc(bc), tuple(h.with_bc(bc) for h in self.holes))

    def with_prec(self, prec):
        return self._rebuilt(self.outer.with_prec(prec), tuple(h.with_prec(prec) for h in self.holes))

    def __add__(self, dz):
        return self.translated(dz)

    __radd__ = __add__

    def __mul__(self, s):
        return self.scaled(s)

    __rmul__ = __mul__

    def __repr__(self):
        holes = f", {len(self.holes)} holes" if self.holes else ""
        return f"{type(self).__name__}(area={self.area:.6g}{holes})"


class Polygon(Domain):
    """A domain whose every segment is straight, with closed forms where they exist."""

    def __init__(self, vertices=None, bdry=None, bc="dir", holes=(), validate=True, prec=PREC):
        if bdry is None:
            if vertices is None:
                raise ValueError("give either vertices or bdry")
            bdry = Boundary.from_vertices(np.asarray(vertices, dtype=complex), bc=bc,
                                          validate=False, prec=prec)
        super().__init__(bdry, holes, validate=validate)

    @cached_property
    def vertices(self):
        return np.array([s.p0 for s in self.outer.segments])

    @property
    def n_vertices(self):
        return len(self.outer.segments)

    n_sides = n_vertices

    @cached_property
    def edge_lengths(self):
        return self.outer.seg_lengths

    @cached_property
    def interior_angles(self):
        """Every junction, corner or not -- segment-indexed, unlike `corners`."""
        segs = self.outer.segments
        t_out = np.array([s.T0 for s in segs])
        t_in = np.roll(np.array([s.T1 for s in segs]), 1)
        return np.angle(-t_in / t_out) % (2 * np.pi)

    @cached_property
    def area(self):
        """Shoelace, exact -- no quadrature and no refinement loop."""
        if self.holes:
            return float(super().area)
        v = self.vertices
        return float(0.5 * np.imag(np.sum(np.conj(v) * np.roll(v, -1))))

    @cached_property
    def diameter(self):
        if any(not isinstance(s, LineSegment) for b in self.boundaries for s in b.segments):
            return super().diameter
        v = np.concatenate([np.array([s.p0 for s in b.segments]) for b in self.boundaries])
        return float(np.abs(v[:, None] - v[None, :]).max())
