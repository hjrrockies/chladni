"""The GSVD call: tension out, coefficients out.

Private module name because the public function `tensions` and the wrapped library are both
called things that would collide at `chladni.mps.gsvd`.

`gsvd4py` returns `c` non-increasing and `s` non-decreasing with `c**2 + s**2 == 1`, so the
generalized singular values `c/s` come back DESCENDING and the tension -- the smallest of them
-- is the last entry. Reversing once here is why nothing downstream has to remember that.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as la
from gsvd4py import gsvd as _gsvd
from gsvd4py import gsvdvals as _gsvdvals

from ._regularize import regularize

__all__ = ["tensions", "coef", "tensions_of", "coef_of"]


def tensions_of(r):
    """The ascending tension vector of an already-regularized pencil."""
    c, s = _gsvdvals(r.B, r.I)
    # s == 0 is an infinite generalized singular value: a direction the interior block
    # annihilates. +inf is the honest answer and it sorts to the far end, where it belongs.
    sigma = np.divide(c, s, out=np.full(c.shape, np.inf), where=(s != 0))
    return sigma[::-1].copy()


def coef_of(r, mult=1):
    """The `mult` coefficient vectors attaining the smallest generalized singular values.

    `mode="separate"` gives `Q` and `R` directly, so the minimizer comes out of one triangular
    solve rather than by forming `X` and inverting it. The price is that separate mode is
    UNSORTED -- `gsvd4py` sorts only in its full and econ modes -- which is why the index
    search below is explicit. lappy's path, and it is the right one.
    """
    D1, D2, R, Q = _gsvd(r.B, r.I, mode="separate", compute_u=False, compute_v=False)[:-2]
    c, s = np.max(D1, axis=0), np.max(D2, axis=0)
    if mult > len(c):
        raise ValueError(f"asked for {mult} coefficient vectors but the pencil has {len(c)}")
    # ratio c/s ascending == c descending, s ascending: lexsort's last key is primary
    idx = np.lexsort((s, -c))[-mult:][::-1]
    Er = np.eye(R.shape[0])[:, idx]
    Xr = Q[:, -R.shape[1]:] @ la.solve_triangular(R, Er)
    return r.back(Xr)


def tensions(A_B, A_I, reg="svd", rank=None):
    """Regularize and return the ascending tension vector. Raw arrays in, array out."""
    return tensions_of(regularize(A_B, A_I, reg=reg, rank=rank))


def coef(A_B, A_I, mult=1, reg="svd", rank=None):
    """Regularize and return the `mult` minimizing coefficient vectors, `(n_cols, mult)`."""
    return coef_of(regularize(A_B, A_I, reg=reg, rank=rank), mult)
