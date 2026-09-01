"""The method-agnostic eigensolver contract, and the spectral parameter.

This module sits above `geometry` and below every method package -- the same position
`helmholtz` occupies. Deliberately NOT in `chladni/mps/`: a later `chladni/fem/` must not
import its base class out of the MPS package. And deliberately not in `chladni/_core.py`,
which is dependency-free by design, `geometry` importing from it rather than the other way.

What is on `Eigensolver` is thin, and the thinness is the point. MPS scans a scalar tension
across `lam` and minimizes it. FEM assembles a sparse symmetric pencil and solves it ONCE for
many eigenvalues, with no scan and nothing to minimize. A boundary-integral method has its own
residual with its own shape. The only surface all three share is "given a domain and an
interval, return the eigenvalues in it" -- which is exactly what lappy's `BaseEigensolver`
settled on, and it was right.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ._core import PREC, PrecisionWarning, _set, check_prec

__all__ = ["Eigensolver", "lam_to_k", "k_to_lam", "weyl_window", "PREC", "PrecisionWarning"]

#: Relative slack below the Faber-Krahn bound. Faber-Krahn is an EQUALITY for the disk, so an
#: unpadded lower endpoint lands exactly ON lam_1 -- and a grid search that ignores its
#: endpoints then misses the first mode entirely. lappy hit this: "The unit disk returned
#: modes 2..k+1: every value correct to ~14 digits, every certificate valid, and nothing in
#: the pipeline able to detect the omission."
_WINDOW_PAD = 1e-3

#: Relative slack above the Weyl estimate. Weyl's law is asymptotic and under-counts at small
#: n, so the window is padded and `solve` widens further when even that came up short. Kept
#: modest because widening is cheap and an over-wide first window is not.
_WEYL_PAD = 0.15

#: How many times `solve` may widen before giving up. A bound, not a tuning knob: without one
#: a domain whose search is broken loops until the machine dies, which is the failure lappy
#: recorded when an unbounded recursion "drove a 16GB machine to a 59.8GB footprint".
_MAX_WIDEN = 8


def lam_to_k(lam):
    """`k = sqrt(lam)`, validated.

    The validation is the reason this is a function and not an expression. A line search can
    hand `lam` a negative iterate, and `sqrt` of that is a complex number arriving three
    functions later as a confusing error about array dtypes -- the same argument `check_k`
    makes one layer down, at the other end of the same conversion.
    """
    value = np.asarray(lam, dtype=float)
    if not np.all(np.isfinite(value)) or np.any(value <= 0.0):
        raise ValueError(
            f"lam must be finite and strictly positive, got {lam!r}. The Helmholtz layer "
            f"is parameterized by k = sqrt(lam) and cannot accept a non-positive iterate."
        )
    return np.sqrt(value) if value.ndim else float(np.sqrt(value))


def k_to_lam(k):
    """`lam = k**2`. The other direction needs no validation: every real k gives a real lam."""
    value = np.asarray(k, dtype=float)
    return value**2 if value.ndim else float(value**2)


def weyl_window(domain, n, pad=_WEYL_PAD):
    """A `lam` interval expected to contain the first `n` eigenvalues, as `(a, b)`.

    The upper end is `asymp.weyl_est`, the two-term Weyl law inverted; the lower end is
    `bounds.faber_krahn`, padded by `_WINDOW_PAD` for the reason recorded there. Both used to
    be inlined here, along with a hard-coded first zero of `J_0`.

    For a boundary that is not purely Dirichlet, Faber-Krahn does not apply -- the Neumann
    spectrum starts at `lam = 0`, whose constant eigenfunction no MPS basis represents and
    whose `k = 0` the helmholtz layer rejects by construction -- so the window starts just
    above zero and the zero mode is simply not found.
    """
    from . import asymp, bounds

    b = float(asymp.weyl_est(n, domain)) * (1.0 + pad)
    if domain.bc_type == "dir":
        a = float(bounds.faber_krahn(domain)) * (1.0 - _WINDOW_PAD)
    else:
        a = b * 1e-6
    return a, b


class Eigensolver(ABC):
    """Anything that computes Laplacian eigenvalues of a planar domain.

    Subclasses implement `solve_interval`; `solve` comes free. There is no tension here, no
    basis and no collocation -- see the module docstring for why none of it survives contact
    with a second method.
    """

    def __init__(self, domain, prec=None):
        _set(self, domain=domain,
             prec=check_prec(domain.prec if prec is None else prec))

    @property
    def bc_type(self):
        return self.domain.bc_type

    @abstractmethod
    def solve_interval(self, a, b):
        """Every eigenvalue in `[a, b]`, ascending, with multiplicity repeated."""

    def solve(self, n):
        """The `n` smallest eigenvalues, ascending, with multiplicity repeated.

        A Weyl window, then `solve_interval`, widening upward if the window came up short.
        The lower endpoint never moves: it is a true bound, and lowering it would only
        invite spurious roots below the spectrum.
        """
        a, b = weyl_window(self.domain, n)
        for _ in range(_MAX_WIDEN):
            found = np.asarray(self.solve_interval(a, b), dtype=float)
            if len(found) >= n:
                return np.sort(found)[:n]
            b = a + 2.0 * (b - a)
        raise RuntimeError(
            f"found {len(found)} eigenvalues, short of the {n} requested, after "
            f"{_MAX_WIDEN} widenings out to lam = {b:.6g}. Either the domain's spectrum is "
            f"sparser than Weyl's law predicts, or solve_interval is missing roots."
        )

    def __repr__(self):
        return f"{type(self).__name__}({self.domain!r}, prec={self.prec:.0e})"
