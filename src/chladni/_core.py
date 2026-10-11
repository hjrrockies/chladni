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

#: How close to an integer `pi/omega`, or a corner exponent, must be before it counts as one.
#: The comparison is on the quantity itself rather than on a ratio, because `pi/omega`
#: inherits omega's roundoff and an exact-equality test would refuse a legitimate right angle.
#:
#: Two layers share it, asking different questions: `Corner.is_reflection_closed` classifies
#: the ANGLE, `CornerModes.is_single_valued` decides whether the EXPANSION is smooth. They
#: share the constant because geometry's answer must never be the LOOSER of the two -- it
#: grades quadrature on the strength of it, and a corner geometry waved through as regular
#: while helmholtz calls it singular is under-resolved, which is the one failure this
#: arrangement may not have. That rules out the tempting `abs(omega - pi/m) <= angle_atol`:
#: `angle_atol` is `max(prec, 1e-12)`, which is 1e-8 at the default `prec`, so a corner at
#: `omega = pi/2 + 1e-9` would read closed here and singular there.
INTEGRAL_ATOL = 1e-12


def near_integer(v):
    """Within `INTEGRAL_ATOL` of an integer, from either side.

    `v % 1.0 <= atol` alone is wrong: `pi/(pi/2)` can land at 1.9999999999999998, whose
    residue is 0.9999999999999998 and which is an integer for every purpose here.
    """
    frac = float(v) % 1.0
    return min(frac, 1.0 - frac) <= INTEGRAL_ATOL


class PrecisionWarning(UserWarning):
    """The requested `prec` exceeds what this configuration can support in float64.

    Its own category so a caller can act on it in one line -- `douse` grids trial thousands
    of shapes and want `simplefilter("ignore", PrecisionWarning)`, not a blanket -W ignore
    that would also hide everything else.
    """


class PrecisionShortfall(RuntimeError):
    """A solve could not deliver what was asked, raised on request (`on_shortfall="raise"`).

    The input to a RE-PLAN, not just the fact of failure (`docs/design/douse_requirements.md`,
    "Precision per level"): `result` is the whole result as it would have been returned, `level`
    the first level that fell short (None when no level carries it, as for a dropped
    eigenvalue), `margin` how far short it was (`Level.margin`; NaN without a level), and
    `audit` the count audit (`asymp.CountAudit`, or None when none was taken). The retry is a
    richer basis at the same `prec`, not a tighter request: the same basis fails the same way.

    Raised EARLY -- at the first refined level that misses `prec`, before the window is finished
    -- `result` and `audit` are None, `level`'s multiplicity has not been read (it says 1), and
    `located` holds the values refined so far.
    """

    def __init__(self, message, *, result=None, level=None, margin=float("nan"), audit=None,
                 located=()):
        super().__init__(message)
        self.result, self.level, self.margin, self.audit = result, level, margin, audit
        self.located = tuple(located)


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
