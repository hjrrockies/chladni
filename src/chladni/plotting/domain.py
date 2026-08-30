"""Plotting for the geometry layer.

Structure mirrors the geometry it draws: a segment, a boundary, a domain. `plot_diagnostic`
is the working tool -- colour-coded, index-labelled segments and annotated corners, with
optional normals, branch cuts, quadrature nodes and interior points.

Nothing here reaches for `polyline_pts`. That is adaptive at `prec` -- 16385 points for a
unit circle -- and a measured effort went into keeping a solve from materializing it. A plot
samples instead, at a resolution chosen for pixels rather than for precision.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry._core import bc_kind
from ._axes import BC_LINESTYLE, bc_label, merge, prepare

#: Points per segment when drawing. On a unit circle this is a chord error of ~1e-5 of the
#: radius, comfortably under a pixel at any sane figure size.
DRAW_N = 200


@dataclass
class DomainArtists:
    ax: object
    lines: list = field(default_factory=list)
    patch: object = None
    legend: object = None


@dataclass
class DiagnosticArtists:
    ax: object
    lines: list = field(default_factory=list)
    texts: list = field(default_factory=list)
    corner_markers: list = field(default_factory=list)
    quivers: list = field(default_factory=list)
    cuts: list = field(default_factory=list)
    nodes: object = None
    interior: object = None


def _sample(seg, n):
    return seg.p(np.linspace(0.0, 1.0, int(n)))


def _boundary_xy(boundary, n):
    """One closed polyline for a whole boundary, for filling."""
    parts = [_sample(s, n) for s in boundary.segments]
    z = np.concatenate([parts[0]] + [p[1:] for p in parts[1:]])
    return np.column_stack([z.real, z.imag])


# --- primitives ---------------------------------------------------------------------------


def plot_segment(segment, *, ax=None, n=DRAW_N, show_bc=False, **kwargs):
    """Draw one segment. Returns the list matplotlib's `plot` returns."""
    _, ax = prepare(ax)
    z = _sample(segment, n)
    style = dict(kwargs)
    if show_bc:
        style.setdefault("linestyle", BC_LINESTYLE[bc_kind(segment.bc)])
    return ax.plot(z.real, z.imag, **merge({"color": "k"}, style))


def plot_boundary(boundary, *, ax=None, n=DRAW_N, show_bc=False, **kwargs):
    """Draw every segment of one boundary. One Line2D per segment, always -- so that
    per-segment styling (a boundary condition, a diagnostic colour) is always available."""
    _, ax = prepare(ax)
    return [plot_segment(s, ax=ax, n=n, show_bc=show_bc, **kwargs)[0] for s in boundary.segments]


def plot_domain(domain, *, ax=None, fill=None, show_bc=False, n=DRAW_N,
                fill_kw=None, legend=False, **kwargs):
    """Draw a domain: its boundaries, and optionally the region itself.

    `fill=None` shades exactly when the domain has holes -- which is exactly when an outline
    alone is ambiguous about which side is the domain. `True`/`False` force it.

    The shading is one compound `Path` over all boundaries. Outer boundaries run
    counter-clockwise and holes clockwise, so matplotlib's fill rule punches the holes out
    with no special-casing on our part: a dividend from the orientation convention.

    `show_bc` puts the boundary condition into the linestyle -- solid Dirichlet, dashed
    Neumann, dash-dot Robin -- and `legend=True` names the conditions actually present.
    """
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    _, ax = prepare(ax)
    art = DomainArtists(ax=ax)

    if fill is None:
        fill = bool(domain.holes)
    if fill:
        paths = [Path(_boundary_xy(b, n), closed=True) for b in domain.boundaries]
        patch = PathPatch(
            Path.make_compound_path(*paths),
            **merge({"facecolor": "0.85", "edgecolor": "none", "alpha": 0.6, "zorder": 0},
                    fill_kw),
        )
        ax.add_patch(patch)
        art.patch = patch

    for b in domain.boundaries:
        art.lines.extend(plot_boundary(b, ax=ax, n=n, show_bc=show_bc, **kwargs))

    if legend:
        art.legend = _bc_legend(ax, domain, art.lines, show_bc)
    ax.autoscale_view()
    return art


def _bc_legend(ax, domain, lines, show_bc):
    """One entry per condition actually present, in the order first met."""
    from matplotlib.lines import Line2D

    seen, handles = [], []
    for line, seg in zip(lines, [s for b in domain.boundaries for s in b.segments]):
        label = bc_label(seg.bc)
        if label in seen:
            continue
        seen.append(label)
        handles.append(Line2D([], [], color=line.get_color(), label=label,
                              linestyle=BC_LINESTYLE[bc_kind(seg.bc)] if show_bc else "-"))
    return ax.legend(handles=handles)


# --- vectors on the boundary ---------------------------------------------------------------


def uniform_boundary_points(domain, n):
    """`n` points spread evenly in ARCLENGTH over every boundary, with their frames.

    Deliberately not `bdry_quad`. Quadrature nodes are Gauss-Legendre, which cluster towards
    panel ends -- correct for integration, wrong for a picture: on a disk, a single panel put
    most of the arrows in a clump at the seam. `tau` is normalized arclength, so a linspace
    in `tau` is what "evenly spaced" actually means here. Endpoints are dropped so adjacent
    segments do not double up at their shared junction.
    """
    segs = [(b, s) for b in domain.boundaries for s in b.segments]
    lengths = np.array([s.length for _, s in segs])
    counts = np.maximum(1, np.round(int(n) * lengths / lengths.sum()).astype(int))
    pts, nrm, tan = [], [], []
    for (_, seg), k in zip(segs, counts):
        tau = np.linspace(0.0, 1.0, int(k), endpoint=False)
        pts.append(seg.p(tau))
        nrm.append(seg.N(tau))
        tan.append(seg.T(tau))
    return np.concatenate(pts), np.concatenate(nrm), np.concatenate(tan)


def _boundary_vectors(domain, n, which):
    """Points and a unit vector at each.

    Written as "a vector per boundary node" rather than specifically as normals, because
    Cauchy data (du/dn scaled along the normal) is the same picture with a different length.
    """
    pts, nrm, tan = uniform_boundary_points(domain, n)
    return pts, (nrm if which == "normal" else tan)


def plot_normals(domain, *, n=64, ax=None, **kwargs):
    """Outward unit normals as a quiver."""
    _, ax = prepare(ax)
    pts, vec = _boundary_vectors(domain, n, "normal")
    return ax.quiver(pts.real, pts.imag, vec.real, vec.imag, angles="xy", **kwargs)


def plot_tangents(domain, *, n=64, ax=None, **kwargs):
    """Unit tangents as a quiver, in the direction of traversal."""
    _, ax = prepare(ax)
    pts, vec = _boundary_vectors(domain, n, "tangent")
    return ax.quiver(pts.real, pts.imag, vec.real, vec.imag, angles="xy", **kwargs)


# --- the diagnostic ------------------------------------------------------------------------


def plot_diagnostic(domain, *, ax=None, segments=True, corners=True, normals=False,
                    branch_cuts=False, quad=None, interior=0, n=DRAW_N, rng=0,
                    labels=True, fontsize=7, **kwargs):
    """Colour-coded, index-labelled view of a domain.

    Answers "which segment is index 3" at a glance: each segment gets its own colour from a
    qualitative cycle and its index printed at its midpoint, pushed outward along the normal
    so the label clears the line. Corners are marked and annotated with their index and
    interior angle.

    `labels=False` drops the angle annotation and keeps bare indices, which is what a
    24-corner spiral wants. Corners on holes are drawn open rather than filled. They cannot carry a fractional-order
    expansion -- `admits_exterior_branch_cut` is False there for topological reasons -- and
    that is exactly the sort of thing a diagnostic should make visible.
    """
    import matplotlib.pyplot as plt

    _, ax = prepare(ax)
    art = DiagnosticArtists(ax=ax)
    scale = domain.diameter
    all_segs = [(bi, si, s) for bi, b in enumerate(domain.boundaries)
                for si, s in enumerate(b.segments)]
    cmap = plt.get_cmap("tab20" if len(all_segs) > 10 else "tab10")
    ncol = 20 if len(all_segs) > 10 else 10

    for k, (_, si, seg) in enumerate(all_segs):
        colour = cmap(k % ncol)
        if segments:
            art.lines.extend(plot_segment(seg, ax=ax, n=n, color=colour, **kwargs))
            mid = complex(seg.p(0.5))
            off = mid + 0.035 * scale * complex(seg.N(0.5))
            art.texts.append(ax.text(off.real, off.imag, str(si), color=colour, ha="center",
                                     va="center", fontsize=fontsize + 1, fontweight="bold"))

    if corners:
        for ci, c in enumerate(domain.corners):
            face = "none" if c.on_hole else "crimson"
            art.corner_markers.extend(
                ax.plot(c.z.real, c.z.imag, "o", markerfacecolor=face,
                        markeredgecolor="crimson", markersize=6, zorder=5)
            )
            # along the interior bisector: always points into Omega, so labels sit inside
            # the shape instead of wherever the branch cut happened to go
            inward = np.exp(1j * (np.angle(c.t_out) + c.omega / 2))
            tip = c.z + 0.05 * scale * inward
            art.texts.append(ax.text(tip.real, tip.imag,
                                     f"{ci}\n{c.omega / np.pi:.2f}π" if labels else str(ci),
                                     color="crimson", ha="center", va="center",
                                     fontsize=fontsize))

    if normals:
        art.quivers.append(plot_normals(domain, n=48, ax=ax, color="0.4", width=0.003))

    if branch_cuts:
        for c in domain.corners:
            if not np.isfinite(c.branch_angle):
                continue
            path = c.branch_path(0.35 * scale)
            art.cuts.extend(ax.plot(path.real, path.imag, ":", color="darkorange",
                                    linewidth=1.2, zorder=4))

    if quad is not None:
        art.nodes = ax.scatter(quad.pts.real, quad.pts.imag, s=4, color="tab:blue", zorder=6)

    if interior:
        pts = domain.interior_pts(int(interior), rng=rng)
        art.interior = ax.scatter(pts.real, pts.imag, s=4, color="0.5", zorder=3)

    ax.autoscale_view()
    return art


# --- the primitive every field plot will need ----------------------------------------------


def field_grid(domain, n=200, pad=0.0):
    """`(X, Y, mask)` over the domain's bounding box, masked to the domain.

    The primitive behind every eigenfunction colormap and surface plot to come: evaluate on
    `X + 1j*Y`, then `np.where(mask, u, np.nan)` for `contourf`/`pcolormesh`, or feed the
    masked array straight to a 3-D surface. Holes fall out of `contains` with no extra work.
    """
    lo, hi = domain.bbox
    dx, dy = (hi.real - lo.real) * pad, (hi.imag - lo.imag) * pad
    x = np.linspace(lo.real - dx, hi.real + dx, int(n))
    y = np.linspace(lo.imag - dy, hi.imag + dy, int(n))
    X, Y = np.meshgrid(x, y, indexing="xy")
    mask = domain.contains((X + 1j * Y).ravel()).reshape(X.shape)
    return X, Y, mask
