"""Plotting for chladni.

Kept in its own layer so that `chladni.geometry` never imports matplotlib. The convenience
`.plot()` methods on geometry objects import this package inside the method body for the
same reason -- lappy's `geometry.py` carries a top-level `import matplotlib.pyplot as plt`,
which makes layer 1 depend on a plotting library.
"""
from ._axes import BC_LINESTYLE, prepare
from .domain import (
    DiagnosticArtists,
    DomainArtists,
    field_grid,
    plot_boundary,
    plot_diagnostic,
    plot_domain,
    plot_normals,
    plot_segment,
    plot_tangents,
    uniform_boundary_points,
)

__all__ = [
    "BC_LINESTYLE", "prepare", "DomainArtists", "DiagnosticArtists",
    "plot_segment", "plot_boundary", "plot_domain",
    "plot_normals", "plot_tangents", "plot_diagnostic", "field_grid", "uniform_boundary_points",
]
