"""Geometry: segments, boundaries, domains, and their boundary quadrature.

Points are complex128 throughout; objects are immutable, and every transform returns a new
one. Outer boundaries run counter-clockwise and holes clockwise, so the outward normal is
``-1j * T`` everywhere and signed areas add.
"""
from ._core import PREC, PrecisionWarning, as_xy, from_xy
from .boundary import Boundary, Corner
from .domain import Domain, Polygon
from .library import (
    GWW1, GWW2, L_shape, annulus, disk, eq_tri, iso_right_tri, plus_shape, polyomino,
    rect, reg_ngon,
)
from .quadrature import Panel, Quadrature, legendre_nodes
from .segments import ArcSegment, LineSegment, ParametricSegment, Segment, SplineSegment

__all__ = [
    "PREC", "PrecisionWarning", "as_xy", "from_xy",
    "Segment", "LineSegment", "ArcSegment", "ParametricSegment", "SplineSegment",
    "Boundary", "Corner", "Domain", "Polygon", "Quadrature", "Panel", "legendre_nodes",
    "rect", "L_shape", "eq_tri", "iso_right_tri", "reg_ngon", "disk", "annulus",
    "GWW1", "GWW2", "polyomino", "plus_shape",
]
