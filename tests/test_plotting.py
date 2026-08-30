"""Plotting: structure, not pixels.

Image comparison is too brittle to earn its keep here, so every assertion is about the
artists produced -- how many, of what type, with which linestyle -- and about the two
architectural properties the module exists to protect: geometry stays matplotlib-free, and
a plot never forces the prec-resolution polyline into existence.
"""
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.patches import PathPatch
from matplotlib.path import Path

from chladni.geometry import (
    GWW1,
    Boundary,
    Domain,
    L_shape,
    LineSegment,
    Polygon,
    annulus as annulus_lib,
    disk,
    eq_tri,
    plus_shape,
    rect,
    reg_ngon,
    spiral,
)
from chladni.plotting import (
    field_grid,
    plot_boundary,
    plot_diagnostic,
    plot_domain,
    plot_normals,
    plot_segment,
    plot_tangents,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def mixed_bc():
    """One edge of each kind, so the linestyle table is fully exercised."""
    return Domain(Boundary([
        LineSegment(0j, 1 + 0j, bc="dir"),
        LineSegment(1 + 0j, 1 + 1j, bc="neu"),
        LineSegment(1 + 1j, 1j, bc=0.4),
        LineSegment(1j, 0j, bc="dir"),
    ]))


# --- axes handling ------------------------------------------------------------------------


def test_plots_onto_a_supplied_axes_without_creating_a_figure(l_shape):
    fig, ax = plt.subplots()
    before = plt.get_fignums()
    art = plot_domain(l_shape, ax=ax)
    assert art.ax is ax
    assert plt.get_fignums() == before
    assert len(ax.lines) == len(l_shape.outer.segments)


def test_creates_an_equal_aspect_axes_when_none_given(l_shape):
    art = plot_domain(l_shape)
    assert art.ax.get_aspect() == 1.0
    assert len(plt.get_fignums()) == 1


def test_a_supplied_axes_keeps_whatever_aspect_it_had(l_shape):
    fig, ax = plt.subplots()
    ax.set_aspect("auto")
    plot_domain(l_shape, ax=ax)
    assert ax.get_aspect() == "auto"


# --- boundary lines -----------------------------------------------------------------------


def test_one_line_per_segment(l_shape, square, eq_tri):
    for d in (l_shape, square, eq_tri):
        art = plot_domain(d)
        assert len(art.lines) == sum(len(b.segments) for b in d.boundaries)


def test_lines_trace_the_boundary(unit_disk):
    art = plot_domain(unit_disk, n=400)
    xy = art.lines[0].get_xydata()
    assert np.allclose(np.hypot(xy[:, 0], xy[:, 1]), 1.0, atol=1e-4)


def test_kwargs_reach_the_lines(l_shape):
    art = plot_domain(l_shape, color="red", linewidth=3.0)
    assert all(ln.get_color() == "red" for ln in art.lines)
    assert all(ln.get_linewidth() == 3.0 for ln in art.lines)


def test_holes_are_drawn_too(annulus):
    art = plot_domain(annulus)
    assert len(art.lines) == 2


# --- boundary conditions ------------------------------------------------------------------


def test_show_bc_maps_conditions_to_linestyles(mixed_bc):
    art = plot_domain(mixed_bc, show_bc=True)
    styles = [ln.get_linestyle() for ln in art.lines]
    assert styles[0] == "-"     # Dirichlet
    assert styles[1] == "--"    # Neumann
    assert styles[2] == "-."    # Robin
    assert styles[3] == "-"


def test_bc_styling_is_off_by_default(mixed_bc):
    art = plot_domain(mixed_bc)
    assert len({ln.get_linestyle() for ln in art.lines}) == 1


def test_legend_names_only_the_conditions_present(mixed_bc, l_shape):
    art = plot_domain(mixed_bc, show_bc=True, legend=True)
    labels = [t.get_text() for t in art.legend.get_texts()]
    assert any("Dirichlet" in x for x in labels)
    assert any("Neumann" in x for x in labels)
    assert any("0.4" in x for x in labels)  # Robin carries its parameter

    plain = plot_domain(l_shape, show_bc=True, legend=True)
    assert [t.get_text() for t in plain.legend.get_texts()] == ["Dirichlet"]


# --- filling, and the multiply-connected question -----------------------------------------


def test_fill_defaults_to_on_for_holed_domains_and_off_otherwise(annulus, l_shape):
    assert plot_domain(annulus).patch is not None
    assert plot_domain(l_shape).patch is None


def test_fill_can_be_forced_either_way(annulus, l_shape):
    assert plot_domain(l_shape, fill=True).patch is not None
    assert plot_domain(annulus, fill=False).patch is None


def test_the_fill_patch_has_one_subpath_per_boundary(square_with_square_hole):
    art = plot_domain(square_with_square_hole, fill=True)
    assert isinstance(art.patch, PathPatch)
    codes = art.patch.get_path().codes
    assert int(np.sum(codes == Path.MOVETO)) == len(square_with_square_hole.boundaries)


def test_the_fill_excludes_the_hole(square_with_square_hole):
    """The point of the whole exercise: the shaded region must be Omega, not its outline.

    Checked by rasterizing three points rather than by `Path.contains_point`, which is not
    the fill predicate -- it reports True inside a hole even for a hand-built compound path
    with correctly opposed winding. Agg's nonzero rule is what actually draws, so that is
    what has to be asked. This is a semantic check on three pixels, not image comparison.
    """
    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    plot_domain(square_with_square_hole, ax=ax, fill=True,
                fill_kw={"facecolor": "blue", "alpha": 1.0, "edgecolor": "none"})
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_position([0, 0, 1, 1])
    ax.axis("off")
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())
    h, w = img.shape[:2]
    at = lambda x, y: tuple(img[int((1 - y / 3) * h) - 1, int((x / 3) * w)][:3])
    assert at(0.5, 0.5) == (0, 0, 255)        # in the domain: filled
    assert at(1.5, 1.5) == (255, 255, 255)    # in the hole: punched out


def test_opposed_winding_is_what_makes_the_punch_work(square_with_square_hole):
    """The structural reason the above holds, asserted directly so a regression names itself."""
    d = square_with_square_hole
    assert d.outer.orientation == 1
    assert all(h.orientation == -1 for h in d.holes)


def test_fill_kwargs_are_separate_from_line_kwargs(annulus):
    art = plot_domain(annulus, color="red", fill_kw={"facecolor": "blue"})
    assert all(ln.get_color() == "red" for ln in art.lines)
    assert art.patch.get_facecolor()[2] == 1.0  # blue channel


# --- vectors ------------------------------------------------------------------------------


def test_normals_and_tangents_are_quivers(l_shape):
    fig, ax = plt.subplots()
    qn = plot_normals(l_shape, n=40, ax=ax)
    qt = plot_tangents(l_shape, n=40, ax=ax)
    assert qn.N == qt.N == 40
    assert len(ax.collections) == 2


def test_normals_point_out_of_the_domain(square):
    q = plot_normals(square, n=32)
    base = q.X + 1j * q.Y
    vec = q.U + 1j * q.V
    outward = base + 1e-6 * vec / np.abs(vec)
    assert not np.any(square.contains(outward))


# --- diagnostics --------------------------------------------------------------------------


def test_diagnostic_labels_every_segment_and_corner(l_shape):
    art = plot_diagnostic(l_shape)
    n_seg = len(l_shape.outer.segments)
    n_cor = len(l_shape.corners)
    assert len(art.lines) == n_seg
    assert len(art.texts) == n_seg + n_cor


def test_diagnostic_gives_each_segment_its_own_colour(plus_shape_domain):
    art = plot_diagnostic(plus_shape_domain)
    colours = {tuple(np.round(ln.get_color(), 6)) if not isinstance(ln.get_color(), str)
               else ln.get_color() for ln in art.lines}
    assert len(colours) >= 10  # a qualitative cycle, not one flat colour


def test_diagnostic_segment_labels_carry_the_index(l_shape):
    art = plot_diagnostic(l_shape)
    labels = {t.get_text() for t in art.texts}
    for i in range(len(l_shape.outer.segments)):
        assert str(i) in labels


def test_diagnostic_marks_hole_corners_differently(square_with_square_hole):
    """A hole corner cannot carry a fractional-order expansion, which is worth seeing."""
    art = plot_diagnostic(square_with_square_hole)
    fills = {m.get_markerfacecolor() for m in art.corner_markers}
    assert len(fills) == 2  # filled for outer, open for hole


def test_diagnostic_overlays_are_opt_in(l_shape):
    plain = plot_diagnostic(l_shape)
    assert plain.quivers == [] and plain.cuts == []
    rich = plot_diagnostic(l_shape, normals=True, branch_cuts=True)
    assert len(rich.quivers) == 1
    assert len(rich.cuts) == len(l_shape.corners)


def test_diagnostic_draws_polyline_branch_cuts():
    """A spiral needs wall-following cuts; each must be drawn with all of its legs."""
    d = spiral()
    art = plot_diagnostic(d, branch_cuts=True)
    assert len(art.cuts) == len(d.corners)
    walked = [c for c in d.corners if c.branch_vertices]
    assert walked, "the spiral is supposed to need polyline cuts"
    for c, line in zip(d.corners, art.cuts):
        assert len(line.get_xydata()) == len(c.branch_vertices) + 2


def test_diagnostic_can_overlay_a_quadrature_and_interior_points(l_shape):
    q = l_shape.bdry_quad(120)
    art = plot_diagnostic(l_shape, quad=q, interior=50)
    assert art.nodes is not None and len(art.nodes.get_offsets()) == len(q)
    assert art.interior is not None and len(art.interior.get_offsets()) == 50


# --- the field-plot primitive -------------------------------------------------------------


def test_field_grid_masks_to_the_domain(l_shape):
    X, Y, mask = field_grid(l_shape, n=120)
    assert X.shape == Y.shape == mask.shape == (120, 120)
    assert mask.dtype == bool
    assert mask[~mask].size > 0 and mask.any()
    z = X[mask] + 1j * Y[mask]
    assert np.all(l_shape.contains(z))


def test_field_grid_excludes_holes(annulus):
    X, Y, mask = field_grid(annulus, n=100)
    r = np.hypot(X[mask], Y[mask])
    assert r.min() > 1.0 and r.max() < 2.0


def test_field_grid_covers_the_bounding_box(l_shape):
    X, Y, _ = field_grid(l_shape, n=50)
    lo, hi = l_shape.bbox
    assert X.min() == pytest.approx(lo.real) and X.max() == pytest.approx(hi.real)


# --- convenience methods ------------------------------------------------------------------


def test_geometry_objects_carry_a_plot_method(l_shape):
    """Domain-level plots return an artist container; the smaller ones return plain artist
    lists, matching what matplotlib's own `plot` gives back."""
    assert l_shape.plot().ax is not None
    assert l_shape.plot_diagnostic().ax is not None
    lines = l_shape.outer.plot()
    assert len(lines) == len(l_shape.outer.segments)
    assert len(l_shape.outer.segments[0].plot()) == 1


def test_plot_boundary_and_plot_segment_work_standalone(l_shape):
    fig, ax = plt.subplots()
    plot_boundary(l_shape.outer, ax=ax)
    plot_segment(l_shape.outer.segments[0], ax=ax)
    assert len(ax.lines) == len(l_shape.outer.segments) + 1


# --- architectural guards ------------------------------------------------------------------


def test_geometry_never_imports_matplotlib():
    """Layer 1 must not depend on a plotting library. lappy's geometry.py carries a
    top-level `import matplotlib.pyplot as plt`; the convenience methods here import
    inside the function body precisely so this stays true."""
    code = (
        "import sys, chladni.geometry\n"
        "leaked = sorted(m for m in sys.modules if m.split('.')[0] == 'matplotlib')\n"
        "assert not leaked, leaked\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)


def test_plotting_does_not_materialize_the_prec_polyline():
    """polyline_pts is adaptive at prec -- 16385 points for a unit circle -- and a
    solve-shaped lifecycle deliberately never builds it. A plot must not be what does."""
    d = disk()
    plot_domain(d, n=300)
    plot_diagnostic(d)
    assert "polyline_pts" not in d.outer.segments[0].__dict__


# --- smoke over the library ----------------------------------------------------------------


@pytest.mark.parametrize(
    "factory",
    [lambda: rect(2, 1), L_shape, disk, eq_tri, GWW1, plus_shape, spiral, annulus_lib,
     lambda: reg_ngon(7)],
    ids=["rect", "L_shape", "disk", "eq_tri", "GWW1", "plus_shape", "spiral", "annulus", "ngon"],
)
def test_every_library_domain_plots(factory):
    d = factory()
    art = plot_domain(d, show_bc=True, legend=True)
    assert len(art.lines) == sum(len(b.segments) for b in d.boundaries)
    diag = plot_diagnostic(d, normals=True, branch_cuts=True)
    assert len(diag.lines) == sum(len(b.segments) for b in d.boundaries)


def test_boundary_vectors_are_evenly_spaced_not_clustered(unit_disk):
    """Quadrature nodes are Gauss-Legendre and bunch at panel ends -- right for integrating,
    wrong for a picture. On a disk that put most of the arrows in a clump at the seam."""
    from chladni.plotting import uniform_boundary_points

    pts, _, _ = uniform_boundary_points(unit_disk, 64)
    gaps = np.abs(np.diff(np.append(pts, pts[0])))
    assert gaps.max() / gaps.min() < 1.2

    quad_pts = unit_disk.bdry_quad(64).pts
    qgaps = np.abs(np.diff(np.append(quad_pts, quad_pts[0])))
    assert qgaps.max() / qgaps.min() > 5  # the clustering this avoids


def test_diagnostic_labels_can_be_reduced_for_busy_domains():
    """A 24-corner spiral cannot fit an angle annotation at every corner."""
    d = spiral()
    full = plot_diagnostic(d)
    bare = plot_diagnostic(d, labels=False)
    assert all("π" in t.get_text() for t in full.texts if "\n" in t.get_text())
    assert not any("π" in t.get_text() for t in bare.texts)
