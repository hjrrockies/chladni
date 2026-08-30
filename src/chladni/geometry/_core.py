"""Primitives for the geometry layer: the BC encoding and the complex/(N,2) seam.

Immutability and `prec` are shared with every layer above, so they live in `chladni._core`
and are re-exported here -- geometry was their only consumer until `helmholtz` arrived, and
a second `Frozen` would be a second answer to the same question.

Points are complex128 everywhere. `.real`/`.imag` are free views, rotation is multiplication
by exp(1j*theta), and a corner angle is one call to np.angle. Conversion to the (N,2) form
that shapely and matplotlib expect happens only at their boundary, through as_xy/from_xy.
"""
from __future__ import annotations

import numpy as np

from .._core import PREC, Frozen, PrecisionWarning, _set, check_prec

__all__ = ["PREC", "PrecisionWarning", "Frozen", "_set", "check_prec",
           "normalize_bc", "bc_kind", "as_xy", "from_xy", "complexify"]


def normalize_bc(bc):
    """'dir' -> 0.0, 'neu' -> 1.0, and any real in [0, 1] is the Robin parameter of
    (1 - bc)*u + bc*du_dN = 0.

    Kept from lappy, but resolved once at construction instead of through a mutable
    property setter -- these objects do not have setters.
    """
    if isinstance(bc, str):
        if bc == "dir":
            return 0.0
        if bc == "neu":
            return 1.0
        raise ValueError(f"boundary condition must be 'dir', 'neu', or a real in [0, 1], got {bc!r}")
    if isinstance(bc, bool) or bc is None:
        raise ValueError(f"boundary condition must be 'dir', 'neu', or a real in [0, 1], got {bc!r}")
    if isinstance(bc, complex) and bc.imag != 0.0:
        raise ValueError(f"boundary condition must be real, got {bc!r}")
    try:
        value = float(np.real(bc)) if isinstance(bc, complex) else float(bc)
    except (TypeError, ValueError):
        raise ValueError(f"boundary condition must be 'dir', 'neu', or a real in [0, 1], got {bc!r}") from None
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"Robin parameter must lie in [0, 1], got {value!r}")
    return value


def bc_kind(bc):
    """'dir', 'neu' or 'rob' for a normalized BC parameter."""
    if bc == 0.0:
        return "dir"
    if bc == 1.0:
        return "neu"
    return "rob"


def as_xy(z):
    """Complex points -> (..., 2) real array, for shapely and matplotlib."""
    z = np.asarray(z, dtype=complex)
    return np.stack([z.real, z.imag], axis=-1)


def from_xy(a):
    """(..., 2) real array -> complex points."""
    a = np.asarray(a, dtype=float)
    return a[..., 0] + 1j * a[..., 1]


def complexify(z):
    """Coerce to a complex array without copying when it is already one."""
    return np.asarray(z, dtype=complex)
