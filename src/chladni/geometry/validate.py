"""Validation policy: cheap invariants always, expensive ones opt-out, orientation repaired.

lappy carries three flags -- `val_simple`, `val_closed`, `val_orientation` -- whose defaults
disagree between `ParametricSegment` (val_simple=False) and `MultiSegment` (val_simple=True),
and it raises `"boundary must be in CCW (positive) orientation"` rather than reorienting.
Here there is one flag, split by cost rather than by kind, and a clockwise boundary is simply
reversed: douse trials wild polygons and should not have to pre-sort vertices.
"""
from __future__ import annotations

import numpy as np

from ._core import as_xy


def check_closed(boundary, what="boundary"):
    if not boundary.is_closed:
        raise ValueError(f"{what} must be a closed curve")


def check_nondegenerate(boundary, what="boundary"):
    """Reject a boundary with no two-dimensional extent.

    Deliberately a bounding-box test rather than a zero-area test. A bowtie also has zero
    signed area, but it encloses real area with cancelling orientation and must survive
    `validate=False` so an optimizer can trial it; a collinear vertex list encloses nothing
    and is degenerate under any flag.
    """
    pts = boundary.polyline_pts
    w = float(pts.real.max() - pts.real.min())
    h = float(pts.imag.max() - pts.imag.min())
    scale = max(w, h, 1.0)
    tol = max(boundary.prec, 1e-12) * scale
    if w <= tol or h <= tol:
        raise ValueError(f"degenerate {what}: zero enclosed area (extent {w:.3g} x {h:.3g})")


def check_simple(boundary, what="boundary"):
    if not boundary.is_simple:
        raise ValueError(f"{what} must be a simple curve")


def inside(boundary, pts):
    """Odd/even ray-crossing parity against a single closed boundary."""
    return (boundary.crossings(pts) % 2).astype(bool)


def check_holes(outer, holes):
    """Each hole strictly inside the outer boundary, and holes pairwise disjoint."""
    from shapely import LinearRing, STRtree

    for k, h in enumerate(holes):
        if not np.all(inside(outer, h.polyline_pts)):
            raise ValueError(f"hole {k} is not contained in the outer boundary")
    if len(holes) < 2:
        return
    rings = [LinearRing(as_xy(np.append(h.polyline_pts, h.polyline_pts[:1]))) for h in holes]
    tree = STRtree(rings)
    for k, ring in enumerate(rings):
        for j in tree.query(ring):
            if int(j) != k and ring.intersects(rings[int(j)]):
                raise ValueError(f"holes {min(k, int(j))} and {max(k, int(j))} are not disjoint")
