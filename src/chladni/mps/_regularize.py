"""Rank truncation before the GSVD, and the one map back out of it.

Private module name, public functions: `regularize` the function and `regularize` the module
cannot both live at `chladni.mps.regularize` without `import chladni.mps.regularize` becoming
ambiguous, and the function is the one callers want.

Two decisions are recorded here.

**The rank is an integer, not a tolerance -- for reproducibility and cost, not accuracy.**
lappy re-derives the rank at every parameter, cutting at `(s >= rtol).sum()`. The original
argument against that was a correctness one: the retained rank becomes a step function of `k`,
so the tension should jump, and near an eigenvalue truncation might delete the eigenfunction.
That was measured and it is wrong. `||A_I c||` is of order one for any minimizer -- that is
what the tension's denominator is for -- so the minimizer has healthy norm in the STACKED
pencil, which is what the cutoff acts on. On the L-shape at `lam_1`, tension 1.7e-12, the
minimizer carries all of its energy in the retained subspace and the threshold rank does not
move at all. See `docs/design/mps.md`, "As built".

What survives is narrower. The rank is a property of the solver rather than of `k`, so
`tension` is a function of `k` alone: two visits to a parameter agree and a re-solve does not
drift for unrelated reasons. And it skips one `svdvals` of the stack per evaluation, 11.3 ms
against 13.4 ms per `k` on a 132-column pencil. `tests/test_regularization.py` pins both,
and pins the agreement measurement too, so the next person to ask can find the answer.

**There is one inverse.** `regularize` returns `back`, the map from reduced coordinates to
basis coordinates, built by whichever mode ran. lappy's `nullspace_coef` instead branches on
`reg_type` a second time at the recovery site, so two places have to agree about a
transformation only one of them computed.
"""
from __future__ import annotations

import warnings
from typing import Callable, NamedTuple, Optional

import numpy as np
import scipy.linalg as la

__all__ = ["RegularizedPencil", "regularize", "choose_rank"]

#: How close `s[r]/s[r-1]` may come to 1 before the frozen rank is reported as straddling a
#: near-degenerate pair. At a true tie the retained subspace is not well defined and the
#: continuity argument for a fixed index does not apply -- the one configuration where this
#: design has nothing to offer, so it says so rather than pretending.
_DEGENERACY_ATOL = 1e-8


class RegularizedPencil(NamedTuple):
    """The reduced pair, plus everything needed to get back out of it.

    `B` and `I` are the reduced blocks; `back(y)` maps reduced coordinates to basis
    coordinates; `rank` is what was retained; `s` are the singular values the choice was made
    from, or None for a mode that computes none. `tola`/`tolb` are passed through to the GSVD
    by the `"implicit"` mode and are None otherwise.
    """

    B: np.ndarray
    I: np.ndarray
    back: Callable[[np.ndarray], np.ndarray]
    rank: int
    s: Optional[np.ndarray] = None
    tola: Optional[float] = None
    tolb: Optional[float] = None


def _svd_gesdd_then_gesvd(a):
    """`la.svd`, falling back to the slower, sturdier `gesvd` driver on non-convergence.

    Kept from lappy, warning and all. Its docstring records why: "one shape out of roughly
    1500 solves raised here ... it took down the whole sweep and lost six of ten starts,
    about an hour of compute." The warning is deliberate -- falling back silently would hide
    the fact that a domain is numerically difficult.
    """
    try:
        return la.svd(a, full_matrices=False)
    except np.linalg.LinAlgError:
        warnings.warn(
            "the default SVD driver (gesdd) failed to converge on this pencil; retrying "
            "with gesvd. This is a signal that the domain is numerically difficult, not a "
            "routine event.",
            UserWarning,
            stacklevel=3,
        )
        return la.svd(a, full_matrices=False, lapack_driver="gesvd")


def choose_rank(A_B, A_I, rtol):
    """The numerical rank of the stacked pencil at a RELATIVE cutoff `s >= rtol * s[0]`.

    Relative, unlike lappy, whose `rtol` was an absolute cutoff on `s` despite the name --
    so its retained rank moved when the pencil was scaled, and the column equilibration one
    module over was load-bearing for a reason it never stated.

    This is called ONCE, at a reference parameter, and the integer it returns is then frozen.
    Calling it per-parameter is what the module docstring exists to argue against.
    """
    s = la.svdvals(np.vstack([np.asarray(A_B), np.asarray(A_I)]))
    if s.size == 0 or s[0] == 0.0:
        return 0
    return int((s >= rtol * s[0]).sum())


def _warn_if_degenerate(s, rank):
    if s is None or rank <= 0 or rank >= len(s):
        return
    hi, lo = s[rank - 1], s[rank]
    if hi > 0 and lo / hi > 1.0 - _DEGENERACY_ATOL:
        warnings.warn(
            f"the frozen rank {rank} cuts a near-degenerate pair "
            f"(s[{rank - 1}] = {hi:.6e}, s[{rank}] = {lo:.6e}). The retained subspace is "
            f"ill-defined there, so the tension can still move discontinuously in k -- the "
            f"one case a fixed index does not fix. Move the rank off the tie.",
            UserWarning,
            stacklevel=3,
        )


def regularize(A_B, A_I, reg="svd", rank=None, tola=None, tolb=None):
    """Reduce the pencil `(A_B, A_I)` to `rank` columns, returning the map back.

    `reg` selects the mode:

    * `"svd"` -- economic QR of the stack, SVD of the triangular factor, keep the leading
      `rank` left singular directions. The default, and the best conditioned.
    * `"qrp"` -- column-pivoted QR, keep the leading `rank` pivoted columns. Cheaper, and it
      names actual basis columns, which makes it the readable one when debugging.
    * `"implicit"` -- no pre-truncation; `tola`/`tolb` are handed to the GSVD so LAPACK makes
      the rank decision. **Diagnostic only.** LAPACK's decision is threshold-based, which
      reintroduces exactly the discontinuity in `k` that the fixed rank removes.
    * `None` -- no regularization at all.

    `rank=None` means no truncation: the full numerical rank is kept.
    """
    A_B, A_I = np.asarray(A_B, dtype=float), np.asarray(A_I, dtype=float)
    m, n = A_B.shape[0], A_B.shape[1]

    if reg is None:
        r = n if rank is None else int(rank)
        return RegularizedPencil(A_B[:, :r], A_I[:, :r], lambda y: np.asarray(y), r)

    if reg == "implicit":
        r = n if rank is None else int(rank)
        return RegularizedPencil(A_B[:, :r], A_I[:, :r], lambda y: np.asarray(y), r,
                                 tola=tola, tolb=tolb)

    stack = np.vstack([A_B, A_I])

    if reg == "svd":
        Q, R = la.qr(stack, mode="economic")
        Z, s, Yt = _svd_gesdd_then_gesvd(R)
        r = len(s) if rank is None else int(rank)
        _warn_if_degenerate(s, r)
        Z1, s1 = Z[:, :r], s[:r]
        Y1 = Yt[:r].T                                 # (n, r)
        reduced = Q @ Z1

        def back(y, _Y1=Y1, _s1=s1):
            # A c = Q Z diag(s) Yt c, and c = Y1 diag(1/s1) y sends Yt c to [y; 0], so
            # A c = (Q Z1) y exactly -- the statement that makes this an inverse.
            y = np.asarray(y, dtype=float)
            return _Y1 @ (y / _s1[:, None] if y.ndim == 2 else y / _s1)

        return RegularizedPencil(reduced[:m], reduced[m:], back, r, s)

    if reg == "qrp":
        Q, R, P = la.qr(stack, mode="economic", pivoting=True)
        d = np.abs(np.diag(R))
        r = len(d) if rank is None else int(rank)
        _warn_if_degenerate(d, r)
        reduced, R11 = Q[:, :r], R[:r, :r]

        def back(y, _R11=R11, _P=P[:r], _n=n, _r=r):
            # stack[:, P] = Q R, so a vector supported on the first r pivoted columns has
            # R11 c_perm = y, and the pivots say where those columns live in basis space.
            y = np.asarray(y, dtype=float)
            flat = y.ndim == 1
            y2 = y[:, None] if flat else y
            c = np.zeros((_n, y2.shape[1]))
            c[_P] = la.solve_triangular(_R11, y2)
            return c[:, 0] if flat else c

        return RegularizedPencil(reduced[:m], reduced[m:], back, r, d)

    raise ValueError(
        f"unknown regularization {reg!r}; expected 'svd', 'qrp', 'implicit' or None"
    )
