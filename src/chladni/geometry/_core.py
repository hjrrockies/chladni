"""Primitives shared across the geometry layer: immutability, the BC encoding, prec, and
the complex/(N,2) seam.

Points are complex128 everywhere. `.real`/`.imag` are free views, rotation is multiplication
by exp(1j*theta), and a corner angle is one call to np.angle. Conversion to the (N,2) form
that shapely and matplotlib expect happens only at their boundary, through as_xy/from_xy.
"""
from __future__ import annotations

import numpy as np

#: Default working relative precision. Valid down to about 1e-15.
PREC = 1e-8


class PrecisionWarning(UserWarning):
    """The requested `prec` exceeds what this geometry can support in float64.

    Its own category so a caller can act on it in one line -- `douse` grids trial thousands
    of shapes and want `simplefilter("ignore", PrecisionWarning)`, not a blanket -W ignore
    that would also hide everything else.
    """


class Frozen:
    """Base for the immutable geometry objects.

    Immutability is what lets `prec` be a plain attribute instead of a cache key: a fixed
    object's approximations can never be stale, so `functools.cached_property` is correct,
    and correct under threads -- which matters because douse runs solves in a thread pool
    over algorithmically generated domains.

    `cached_property` writes straight into `instance.__dict__`, bypassing `__setattr__`, so
    blocking assignment here does not block caching. Construction uses `_set`.
    """

    __hash__ = object.__hash__

    def __setattr__(self, name, value):
        raise AttributeError(
            f"{type(self).__name__} is immutable; use a transform "
            f"(translated/rotated/scaled/reversed/with_bc/with_prec) to get a new object"
        )

    def __delattr__(self, name):
        raise AttributeError(f"{type(self).__name__} is immutable")


def _set(obj, **kwargs):
    """Assign attributes during construction, past the immutability guard."""
    for key, value in kwargs.items():
        object.__setattr__(obj, key, value)


def check_prec(prec):
    """Working relative precision: a real number strictly inside (0, 1)."""
    try:
        value = float(prec)
    except (TypeError, ValueError):
        raise ValueError(f"prec must be a real number in (0, 1), got {prec!r}") from None
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"prec must lie strictly in (0, 1), got {value!r}")
    return value


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
