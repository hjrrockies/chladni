"""Eigenvalues and eigenfunctions that are already known.

Three kinds of thing, kept in separate modules because they are trusted differently:

    closed_form.py     exact formulas -- the rectangle, disk, sector, triangles, polyominoes
    tables.py          another solver's numbers, from 14 digits down to under 3
    eigenfunctions.py  closed-form eigenfunctions with their exact L2 norms

Accuracy is DATA, not prose. Every function carries `digits`, `source` and `notes`, and
`catalog()` lists them, because a docstring cannot be read by the test that has to pick a
tolerance. See `docs/design/reference.md`.

    >>> from chladni import reference
    >>> reference.L_shape_eigs(3)
    >>> reference.L_shape_eigs.digits
    14.0
    >>> reference.ellipse_eigs.notes[(2.0, 1.0)][0]        # a value known to be wrong
"""
from ._core import (REGISTRY, SNAP_RTOL, catalog, collapse, exact, recipe, snap,
                    tabulated)
from .closed_form import (
    bessel_zero,
    disk_eig, disk_eigs, disk_spectrum,
    eq_tri_eig, eq_tri_eigs, eq_tri_spectrum,
    iso_right_tri_eig, iso_right_tri_eigs, iso_right_tri_spectrum,
    polyomino_eig, polyomino_spectrum,
    rect_eig, rect_eigs, rect_spectrum,
    sector_eig, sector_eigs, sector_spectrum,
)
from .eigenfunctions import (
    disk_eigfun, iso_right_tri_eigfun, normalized_eigfun, polyomino_eigfun,
    polyomino_eigfun_grad, rect_eigfun, sector_eigfun, sector_eigfun_grad,
)
from .tables import (
    H_shape_eigs, L_shape_eigs, chevron_eigs, cut_square_eigs, ellipse_eigs,
    gww_eigs, iso_tri_eigs, mushroom_eigs, reg_ngon_eigs,
)

__all__ = [
    "catalog", "recipe", "collapse", "snap", "SNAP_RTOL", "exact", "tabulated",
    "REGISTRY",
    "bessel_zero",
    "rect_eig", "rect_eigs", "rect_spectrum",
    "disk_eig", "disk_eigs", "disk_spectrum",
    "sector_eig", "sector_eigs", "sector_spectrum",
    "eq_tri_eig", "eq_tri_eigs", "eq_tri_spectrum",
    "iso_right_tri_eig", "iso_right_tri_eigs", "iso_right_tri_spectrum",
    "polyomino_eig", "polyomino_spectrum",
    "gww_eigs", "L_shape_eigs", "ellipse_eigs", "reg_ngon_eigs", "cut_square_eigs",
    "mushroom_eigs", "H_shape_eigs", "chevron_eigs", "iso_tri_eigs",
    "rect_eigfun", "iso_right_tri_eigfun", "polyomino_eigfun", "polyomino_eigfun_grad",
    "disk_eigfun", "sector_eigfun", "sector_eigfun_grad", "normalized_eigfun",
]
