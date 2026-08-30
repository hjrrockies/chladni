"""Primitives shared by every layer: immutability and the working precision.

These started in `geometry/_core.py`, which is where the conventions doc still points for
the reasoning. They moved up when `helmholtz` arrived and needed the same `Frozen` contract:
a second copy would be a second `__setattr__` message and a second answer to "is
`cached_property` safe here", and those have to be one answer. `geometry._core` re-exports
them, so nothing below changed.
"""
from __future__ import annotations

import numpy as np

#: Default working relative precision. Valid down to about 1e-15.
PREC = 1e-8


class PrecisionWarning(UserWarning):
    """The requested `prec` exceeds what this configuration can support in float64.

    Its own category so a caller can act on it in one line -- `douse` grids trial thousands
    of shapes and want `simplefilter("ignore", PrecisionWarning)`, not a blanket -W ignore
    that would also hide everything else.
    """


class Frozen:
    """Base for the immutable objects.

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
