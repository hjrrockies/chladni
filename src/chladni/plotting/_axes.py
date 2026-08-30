"""Shared plumbing: axes preparation and the boundary-condition style table.

Every plot in this package -- including the eigenfunction and residual plots still to come --
goes through `prepare`, so the "use my axes or make me one" contract is stated once.
"""
from __future__ import annotations

#: Boundary condition -> linestyle. Matches lappy's mapping so figures from the two
#: codebases read alike: solid Dirichlet, dashed Neumann, dash-dot Robin.
BC_LINESTYLE = {"dir": "-", "neu": "--", "rob": "-."}


def prepare(ax=None, aspect="equal"):
    """Return `(fig, ax)`, creating a figure only when none was supplied.

    A supplied axes is used as-is -- aspect included. A caller who has arranged a subplot
    grid has already made those choices, and silently overriding them is the kind of
    helpfulness that ruins a composed figure.
    """
    import matplotlib.pyplot as plt

    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots()
    ax.set_aspect(aspect)
    return fig, ax


def merge(defaults, overrides):
    """Caller kwargs win over defaults, and `color` beats a defaulted `color`."""
    out = dict(defaults)
    out.update(overrides or {})
    if ("color" in (overrides or {}) or "c" in (overrides or {})) and "color" in defaults:
        out.pop("color", None)
        out.update(overrides)
    return out


def bc_label(bc):
    """Human-readable name for a normalized BC parameter."""
    if bc == 0.0:
        return "Dirichlet"
    if bc == 1.0:
        return "Neumann"
    return f"Robin ({bc:g})"
