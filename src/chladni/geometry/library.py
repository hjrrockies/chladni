"""Shipped domains. Each is a smoke test of the whole layer in one call."""
from __future__ import annotations

import numpy as np

from ._core import PREC
from .boundary import Boundary
from .domain import Domain, Polygon
from .segments import ArcSegment, LineSegment, ParametricSegment


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
    theta = np.linspace(0, 2 * np.pi * turns, int(n))
    centre = (r0 + pitch * theta / (2 * np.pi)) * np.exp(1j * theta)
    tangent = np.gradient(centre)
    tangent /= np.abs(tangent)
    normal = 1j * tangent
    v = np.concatenate([centre + width / 2 * normal, (centre - width / 2 * normal)[::-1]])
    if 0.5 * np.imag(np.sum(np.conj(v) * np.roll(v, -1))) < 0:
        v = v[::-1]
    return Polygon(v, bc=bc, prec=prec)


# --- ported from lappy, with its placement preserved ------------------------------------
#
# Placement is load-bearing: `chladni.reference`'s closed-form eigenfunctions assume the
# placement of the factory they name, and their exact L2 norms are wrong if the two disagree.
# Every shape below sits exactly where lappy puts it, so the formulas transfer unchanged.
#
# Circular arcs use `ArcSegment`, which is closed form. lappy builds them as general
# `ParametricSegment`s and pays the adaptive machinery for something with an exact answer;
# only `ellipse` genuinely needs the parametric form.


def iso_tri(h=1.0, bc="dir", prec=PREC):
    """Isosceles triangle, base 2 on the real axis, apex at `1j*h`. Area `h`.

    The aspect family behind `reference.iso_tri_eigs`: as `h` grows the apex angle closes,
    and the reference tables run out to `h = 20`.
    """
    return Polygon([1, 1j * float(h), -1], bc=bc, prec=prec)


def right_trapezoid(h1=1.0, h2=2.0, bc="dir", prec=PREC):
    """Right trapezoid on the unit base `[0, 1]`, left side `h1`, right side `h2`."""
    return Polygon([1j * float(h1), 0, 1, 1 + 1j * float(h2)], bc=bc, prec=prec)


def parallelogram(b=1.0, h=1.0, alpha=np.pi / 3, bc="dir", prec=PREC):
    """Base `b` on the positive real axis, height `h`, shear angle `alpha` at the origin."""
    if not 0.0 < alpha < np.pi:
        raise ValueError(f"alpha must lie in (0, pi), got {alpha!r}")
    d = h / np.tan(alpha)
    return Polygon([0, b, b + d + 1j * h, d + 1j * h], bc=bc, prec=prec)


def chevron(h1=1.0, h2=2.0, bc="dir", prec=PREC):
    """A dart symmetric about the imaginary axis: wingtips at -1 and 1, notch at `1j*h1`,
    apex at `1j*h2`. Area `h2 - h1`.

    The reentrant notch is at `1j*h1`, and the wingtip angle closes as `h2/h1 -> 1` -- down
    to about 11 degrees for the sharpest tabulated pair, which is why
    `reference.chevron_eigs` is only good to 3-4 digits there.
    """
    h1, h2 = float(h1), float(h2)
    if h1 < 0 or h2 < 0:
        raise ValueError(f"h1 and h2 must be non-negative, got ({h1}, {h2})")
    if h1 >= h2:
        raise ValueError(f"h1 must be less than h2, got ({h1}, {h2})")
    return Polygon([-1, 1j * h1, 1, 1j * h2], bc=bc, prec=prec)


def H_shape(bc="dir", prec=PREC):
    """The fixed 12-vertex H, four reentrant corners. Spans x in [-1, 2], y in [-2, 1]."""
    vx = np.array([-1, 0, 0, 1, 1, 2, 2, 1, 1, 0, 0, -1])
    vy = np.array([-2, -2, -1, -1, -2, -2, 1, 1, 0, 0, 1, 1])
    return Polygon(vx + 1j * vy, bc=bc, prec=prec)


def ellipse(a=2.0, b=1.0, bc="dir", prec=PREC):
    """Centred at the origin, semi-axis `a` along x and `b` along y.

    The one shape here that genuinely needs `ParametricSegment`: an ellipse is not an arc of
    anything closed form.
    """
    a, b = float(a), float(b)
    if a <= 0 or b <= 0:
        raise ValueError(f"semi-axes must be positive, got ({a}, {b})")
    seg = ParametricSegment(lambda t: a * np.cos(t) + 1j * b * np.sin(t),
                            lambda t: -a * np.sin(t) + 1j * b * np.cos(t),
                            0.0, 2 * np.pi, bc=bc, prec=prec)
    return Domain(Boundary([seg]))


def disk_sector(r=1.0, theta=np.pi / 2, bc="dir", prec=PREC):
    """Circular sector, apex at the origin, angular extent `[0, theta]`.

    The sharpest reference in the suite. A Dirichlet sector's eigenfunctions are
    `J_(l*nu)(k r) sin(l*nu*theta)` with `nu = pi/theta`, which is EXACTLY one corner-adapted
    Fourier-Bessel column -- so a one-column basis reproduces an eigenfunction to machine
    precision and the tension at the true eigenvalue is limited only by the Bessel kernel.
    `theta > pi` puts a reentrant corner at the apex.
    """
    theta = float(theta)
    if not 0.0 < theta < 2 * np.pi:
        raise ValueError(f"theta must lie in (0, 2*pi), got {theta!r}")
    return Domain(Boundary([
        LineSegment(0, r, bc=bc, prec=prec),
        ArcSegment(0j, r, 0.0, theta, bc=bc, prec=prec),
        LineSegment(r * np.exp(1j * theta), 0, bc=bc, prec=prec),
    ]))


def cut_square(r=0.5, bc="dir", prec=PREC):
    """The unit square with the corner at `1+1j` replaced by an arc of radius `r`.

    The arc is centred ON the cut corner, so it bites into the domain. The other three
    corners stay right angles, which is what lets `reference.cut_square_eigs` be solved with
    a pure Fourier-Bessel basis.
    """
    r = float(r)
    if not 0.0 < r < 1.0:
        raise ValueError(f"r must lie in (0, 1), got {r!r}")
    return Domain(Boundary([
        LineSegment(0, 1, bc=bc, prec=prec),
        LineSegment(1, 1 + (1 - r) * 1j, bc=bc, prec=prec),
        ArcSegment(1 + 1j, r, -np.pi / 2, -np.pi, bc=bc, prec=prec),
        LineSegment((1 - r) + 1j, 1j, bc=bc, prec=prec),
        LineSegment(1j, 0, bc=bc, prec=prec),
    ]))


def mushroom(a=1.0, b=1.0, r=1.5, bc="dir", prec=PREC):
    """A half-disk cap of radius `r` on a rectangular stem of width `b` and depth `a`.

    Two reentrant 270-degree corners at the junction, at `-b/2` and `+b/2`.
    """
    a, b, r = float(a), float(b), float(r)
    if r <= b / 2:
        raise ValueError(f"cap radius r must exceed half the stem width, got r={r}, b={b}")
    v = [-r, -b / 2, -b / 2 - 1j * a, b / 2 - 1j * a, b / 2, r]
    segs = [LineSegment(v[i], v[i + 1], bc=bc, prec=prec) for i in range(len(v) - 1)]
    segs.append(ArcSegment(0j, r, 0.0, np.pi, bc=bc, prec=prec))
    return Domain(Boundary(segs))


def stadium(L=1.0, H=1.0, bc="dir", prec=PREC):
    """The Bunimovich stadium: straight sides of length `L`, semicircular caps of radius H/2.

    Vertically centred on the real axis; the LEFT CAP'S CENTRE sits at the origin, not the
    leftmost point, so the domain spans x in [-H/2, L + H/2].

    lappy's version omits `bc` on the top straight edge, which therefore silently ignores the
    requested boundary condition. Fixed here; `tests/test_library.py` pins it.
    """
    L, H = float(L), float(H)
    if L <= 0 or H <= 0:
        raise ValueError(f"L and H must be positive, got ({L}, {H})")
    R = H / 2
    return Domain(Boundary([
        LineSegment(-1j * R, L - 1j * R, bc=bc, prec=prec),
        ArcSegment(L, R, -np.pi / 2, np.pi / 2, bc=bc, prec=prec),
        LineSegment(L + 1j * R, 1j * R, bc=bc, prec=prec),
        ArcSegment(0j, R, np.pi / 2, 3 * np.pi / 2, bc=bc, prec=prec),
    ]))
