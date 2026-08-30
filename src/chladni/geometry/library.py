"""Shipped domains. Each is a smoke test of the whole layer in one call."""
from __future__ import annotations

import numpy as np

from ._core import PREC
from .boundary import Boundary
from .domain import Domain, Polygon
from .segments import ArcSegment


def rect(L=1.0, H=1.0, bc="dir", prec=PREC):
    return Polygon([0, L, L + 1j * H, 1j * H], bc=bc, prec=prec)


def L_shape(bc="dir", prec=PREC):
    """A 2x2 square with a 1x1 corner bite: area 3, and the bite leaves the perimeter at 8."""
    return Polygon([0, 1j, -1 + 1j, -1 - 1j, 1 - 1j, 1], bc=bc, prec=prec)


def eq_tri(l=1.0, bc="dir", prec=PREC):
    return Polygon([0, l, 0.5 * l + 1j * np.sqrt(3) / 2 * l], bc=bc, prec=prec)


def iso_right_tri(l=1.0, bc="dir", prec=PREC):
    return Polygon([0, l, 1j * l], bc=bc, prec=prec)


def reg_ngon(n, r=1.0, bc="dir", prec=PREC):
    return Polygon(r * np.exp(2j * np.pi * np.arange(int(n)) / int(n)), bc=bc, prec=prec)


def disk(r=1.0, center=0j, bc="dir", prec=PREC):
    return Domain(Boundary([ArcSegment(center, r, 0.0, 2 * np.pi, bc=bc, prec=prec)]))


def annulus(r_outer=2.0, r_inner=1.0, center=0j, bc="dir", prec=PREC):
    outer = Boundary([ArcSegment(center, r_outer, 0.0, 2 * np.pi, bc=bc, prec=prec)])
    hole = Boundary([ArcSegment(center, r_inner, 0.0, 2 * np.pi, bc=bc, prec=prec)]).reversed()
    return Domain(outer, holes=(hole,))


def GWW1(bc="dir", prec=PREC):
    """First of the Gordon-Webb-Wolpert isospectral pair. Area 14, perimeter 12 + 6*sqrt(2)."""
    vx = np.array([1, 3, 3, -1, -1, -3, -1, 1])
    vy = np.array([-3, -1, 1, 1, 3, 1, -1, -1])
    return Polygon(vx + 1j * vy, bc=bc, prec=prec)


def GWW2(bc="dir", prec=PREC):
    """Second GWW domain. Shares its area, perimeter and corner-angle multiset with GWW1."""
    vx = np.array([1, 1, 3, 1, -1, -1, -3, -3])
    vy = np.array([-3, -1, -1, 1, 1, 3, 3, 1])
    return Polygon(vx + 1j * vy, bc=bc, prec=prec)


def polyomino(cells, bc="dir", validate=True, prec=PREC):
    """Boundary polygon of the union of unit grid cells, each given as (i, j).

    Every cell contributes its four edges counter-clockwise; an edge shared by two cells
    appears twice with opposite orientation and cancels, so what survives is already
    correctly oriented. Consecutive collinear edges are then merged, so a straight run of k
    cells becomes one segment rather than k joined by smooth pi junctions.

    The region must be simply connected and edge-connected. A diagonal-only join leaves a
    vertex with two outgoing edges; an enclosed hole leaves more than one loop. Both are
    rejected, since neither yields a simple boundary polygon.
    """
    cells = [(int(i), int(j)) for i, j in cells]
    if not cells:
        raise ValueError("polyomino needs at least one cell")
    if len(set(cells)) != len(cells):
        raise ValueError("duplicate cells")

    edges = {}
    for i, j in cells:
        corners = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
        for k in range(4):
            a, b = corners[k], corners[(k + 1) % 4]
            if (b, a) in edges:
                del edges[(b, a)]
            else:
                edges[(a, b)] = True

    nxt = {}
    for a, b in edges:
        if a in nxt:
            raise ValueError(
                "cells are joined only at a corner; the boundary would not be a simple polygon"
            )
        nxt[a] = b
    start = next(iter(nxt))
    loop = [start]
    cur = nxt[start]
    while cur != start:
        loop.append(cur)
        cur = nxt[cur]
    if len(loop) != len(edges):
        raise ValueError(
            "cells must form a simply-connected, edge-connected region "
            f"({len(edges)} boundary edges span more than one loop)"
        )

    v = np.array([complex(x, y) for x, y in loop])
    n = len(v)
    keep = [k for k in range(n)
            if abs(np.imag(np.conj(v[k] - v[k - 1]) * (v[(k + 1) % n] - v[k]))) > 0]
    return Polygon(v[keep], bc=bc, validate=validate, prec=prec)


def plus_shape(bc="dir", prec=PREC):
    """The 5-cell plus: four reentrant corners, twelve corners in all."""
    return polyomino([(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)], bc=bc, prec=prec)


def spiral(turns=1.5, pitch=1.0, width=0.35, n=12, r0=0.6, bc="dir", prec=PREC):
    """A polygonal spiral strip, counter-clockwise.

    The shape that motivates polyline branch cuts: its inner coils surround corners whose
    exterior is a channel that turns, so no straight ray escapes to infinity. At the
    default settings half of its 24 corners are in that position, and with ray-only cuts
    every one of them would carry no branch cut at all -- quietly barring a fractional-order
    expansion there. See `Corner.branch_vertices`.
    """
    phi = np.linspace(0, 2 * np.pi * turns, int(n))
    centre = (r0 + pitch * phi / (2 * np.pi)) * np.exp(1j * phi)
    tangent = np.gradient(centre)
    tangent /= np.abs(tangent)
    normal = 1j * tangent
    v = np.concatenate([centre + width / 2 * normal, (centre - width / 2 * normal)[::-1]])
    if 0.5 * np.imag(np.sum(np.conj(v) * np.roll(v, -1))) < 0:
        v = v[::-1]
    return Polygon(v, bc=bc, prec=prec)
