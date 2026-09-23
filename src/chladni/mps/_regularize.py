"""Regularization before the GSVD -- truncation or a penalty -- and the one map back out of it.

Private module name, public functions: `regularize` the function and `regularize` the module
cannot both live at `chladni.mps.regularize` without `import chladni.mps.regularize` becoming
ambiguous, and the function is the one callers want.

Three decisions are recorded here.

**The rank is an integer, not a tolerance -- for reproducibility and cost, not accuracy.**
lappy re-derives the rank at every parameter, cutting at `(s >= rtol).sum()`. The original
argument against that was a correctness one: the retained rank becomes a step function of `k`,
so the tension should jump, and near an eigenvalue truncation might delete the eigenfunction.
Measured AT THE ROOT, that is wrong. `||A_I c||` is of order one for any minimizer -- that is
what the tension's denominator is for -- so the minimizer has healthy norm in the STACKED
pencil, which is what the cutoff acts on. On the L-shape at `lam_1`, tension 1.7e-12, the
minimizer carries all of its energy in the retained subspace and the threshold rank does not
move at all. See `docs/design/mps.md`, "As built".

**It is not wrong on the FLANK, and that is where it has cost a real answer.**
`notes/session_2026-09-19.md` (the 09-21 trace) observes the threshold rank flip 85 -> 86 at
one bracket end on Linux/OpenBLAS, where the first dropped singular value sits at 0.89x the
cutoff -- rounding noise, decided differently by a different BLAS. The flip lowers `sigma_1`
there by 67%, the refiner steps toward that end, and a true root is refused. A flip barely
moves `sigma_1` at a root (median ratio 0.986) and moves it by up to 3.5x on the flanks, which
is exactly where a refiner compares ends. The penalty modes below exist because of this.

What survives of the integer-rank argument is narrower. The rank is a property of the solver
rather than of `k`, so `tension` is a function of `k` alone: two visits to a parameter agree and
a re-solve does not drift for unrelated reasons. And it skips one `svdvals` of the stack per
evaluation, 11.3 ms against 13.4 ms per `k` on a 132-column pencil.
`tests/test_regularization.py` pins both, and pins the agreement measurement too, so the next
person to ask can find the answer.

**There is one inverse.** `regularize` returns `back`, the map from reduced coordinates to
basis coordinates, built by whichever mode ran. lappy's `nullspace_coef` instead branches on
`reg_type` a second time at the recovery site, so two places have to agree about a
transformation only one of them computed.

**A penalty is an alternative to a rank, and only the QR form of it removes the rank.** The
`"tikhonov"` modes make no truncation decision at all. They minimize

    sigma_tau = min_c  sqrt(||A_B c||^2 + tau^2 ||c||^2) / ||A_I c||,

the GSVD of `([A_B; tau*I], A_I)`, which is continuous in the matrix for every `tau > 0` where a
truncation is not. `tau = rtol * ||[A_B; A_I]||_F`: relative, like `rtol` everywhere else here,
and on an equilibrated stack -- every column of unit norm -- exactly `rtol * sqrt(n_cols)`, a
constant for the pencil's life. (A column that underflowed to exact zeros keeps norm 0, so the
norm counts nonzero columns; the equilibrated matrix is already discontinuous at such an event.)
The penalty acts in EQUILIBRATED coordinates, which is the point: raw Fourier-Bessel columns
span about 1149 nats of scale, and `||c||` there would penalize nothing but the largest few.

`"tikhonov"` hands the GSVD `([A_B; tau*I], A_I)` directly. `"tikhonov_qr"` first takes an
unpivoted QR of `[A_B; tau*I; A_I]` and hands it the two row blocks of `Q`. In exact arithmetic
the two are identical -- `R` is invertible, and the GSVD is blind to an invertible right
factor. Through LAPACK they are not, and the difference is the whole point. `tau*I` makes the
STACK full rank but does nothing for the INTERIOR block, and LAPACK's `?ggsvp3` decides that
block's numerical rank with its own tolerance. So `"tikhonov"` does not remove the rank
decision, it moves it inside LAPACK: on `rect(1, 1/phi)`'s 126-column pencil its count of
finite tensions was 33, 43, 48 at `lam = 35.7, 150, 261`, at every `tau` tried. `"tikhonov_qr"`
hands LAPACK an orthonormal stack, and its count was `n_int = 63` at all three.
`"tikhonov_qr"` is the mode to use; `"tikhonov"` is kept as the literal formulation and as an
oracle the QR form must agree with.

Both need `tau` above the rounding noise. At the `prec`-derived default `rtol` the two agree
to 1e-7 or better; at `rtol = 1.5e-15` they part by 1e-3.

**What limits the penalty at tight `prec` is NOISE, not the floor.** The literal form's floor
saturates near 9e-14 -- that is LAPACK's internal cut, and it is where the claim that "a plain
penalty cannot go below 1e-12" came from. The QR form's floor tracks `tau` -- 1.4x `tau` down to
`tau = 1.7e-14`, 1.9x at the bottom -- to 2.1e-15, within 1.3x of the threshold's. But its noise
grows as `tau` shrinks, multiplicatively along the whole curve: 1e-2 relative at the `tau` a
1e-14 floor needs, against the threshold's 1e-3. An order-1 penalty charges the eigenfunction
itself `tau * ||c||`, so floor and noise are both set by `tau` against the rounding level and
cannot be traded apart. `notes/smooth_filter.md` sections 2, 7.4, 9 and 10 carry the
measurements.
"""
from __future__ import annotations

import os
import warnings
from typing import Callable, NamedTuple, Optional

import numpy as np
import scipy.linalg as la
from scipy.linalg import lapack

__all__ = ["RegularizedPencil", "regularize", "choose_rank"]

#: How close `s[r]/s[r-1]` may come to 1 before the frozen rank is reported as straddling a
#: near-degenerate pair. At a true tie the retained subspace is not well defined and the
#: continuity argument for a fixed index does not apply -- the one configuration where this
#: design has nothing to offer, so it says so rather than pretending.
_DEGENERACY_ATOL = 1e-8

#: The penalty modes: no rank decision, so no `rank` to freeze and no `lam_ref` to choose one.
_PENALTY = ("tikhonov", "tikhonov_qr")

#: Every mode `regularize` accepts, so a caller can refuse a bad one at construction rather
#: than on first use.
_MODES = ("svd", "qrp", None) + _PENALTY


def _lapack_is_accelerate():
    try:
        import scipy
        name = scipy.show_config(mode="dicts")["Build Dependencies"]["lapack"]["name"]
        return str(name).lower() == "accelerate"
    except Exception:
        return False


#: Route the stack's QR around Accelerate's blocked `dgeqrf`. Set
#: `CHLADNI_ACCELERATE_GEQRF=1` to call it anyway -- to re-test once Apple ships a fix.
_AVOID_GEQRF = _lapack_is_accelerate() and os.environ.get("CHLADNI_ACCELERATE_GEQRF") != "1"


def _qr_economic(a):
    """Economic `(Q, R)` of `a`: `la.qr`, except on Accelerate, where it is `dgeqrt`.

    Accelerate's blocked Householder QR `dgeqrf` corrupts the process heap, intermittently, on
    macOS 26.5 / M4 (scipy 1.18.1, numpy 2.5.2). Measured on a random 548 x 274 matrix -- the
    size of the rectangle suite's 170*pi^2 stack -- with no chladni code involved: raw
    `dgeqrf$NEWLAPACK` via ctypes kills most processes within 3000 calls, and so do
    `scipy.linalg.qr` and `numpy.linalg.qr`. A canary found no write within 65536 elements of
    A, TAU or WORK, so the damage is inside Accelerate, and it surfaces LATER, wherever the
    heap is next touched: a dict insert, `abc.__instancecheck__`, garbage collection at
    shutdown. Tensions computed before the crash were also wrong -- levels moved from run to
    run. It is not threading (`BLASSetThreading` single-threaded still crashes) and not
    the workspace size (8x `lwork` still crashes). Unblocked `dgeqr2`, `dorgqr`, `dgeqrt` +
    `dgemqrt`, square `gesdd`, the GSVD, and `dgemm`/`dtrmm` at `dlarfb`'s shapes all
    survive. scipy/scipy#26145 reports the same class of fault for Accelerate's `dpotrf`.

    `dgeqrt` is the same Householder QR in compact-WY form, through scipy's public LAPACK
    wrappers, at no measurable cost (4.06 ms against 3.99 ms on the 548 x 274 stack). Every
    other LAPACK keeps `la.qr` and its bits.
    """
    if not _AVOID_GEQRF:
        return la.qr(a, mode="economic")
    a = np.asarray_chkfinite(np.asarray(a, dtype=float))
    m, n = a.shape
    k = min(m, n)
    if k == 0:
        return la.qr(a, mode="economic")
    f, t, info = lapack.dgeqrt(min(32, k), a)
    if info != 0:
        raise np.linalg.LinAlgError(f"dgeqrt failed (info={info})")
    q, info = lapack.dgemqrt(f[:, :k], t, np.eye(m, k, order="F"))
    if info != 0:
        raise np.linalg.LinAlgError(f"dgemqrt failed (info={info})")
    return q, np.triu(f[:k])


class RegularizedPencil(NamedTuple):
    """The reduced pair, plus everything needed to get back out of it.

    `B` and `I` are the reduced blocks; `back(y)` maps reduced coordinates to basis
    coordinates; `rank` is what was retained; `s` are the singular values the choice was made
    from, or None for a mode that computes none.

    For a penalty mode `B` is the AUGMENTED numerator block -- `[A_B; tau*I]`, or its rows of
    `Q` -- so it has `n_cols` more rows than `A_B`, and `rank` is `n_cols` because nothing is
    dropped. `tau` is the absolute penalty weight used, or None for a truncating mode.
    """

    B: np.ndarray
    I: np.ndarray
    back: Callable[[np.ndarray], np.ndarray]
    rank: int
    s: Optional[np.ndarray] = None
    tau: Optional[float] = None


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


def _rank_from(diag, rank, rtol):
    """The rank to truncate at: the caller's, or derived from this factorization's own
    diagonal at the relative cutoff `rtol`.

    For `"svd"` the diagonal is the singular values of the stacked pencil; for `"qrp"` it is
    `|diag R|`, which is only a WEAK rank revealer -- on a Kahan matrix at n = 90 it
    overestimates the smallest singular value by 3.4e4. Measured on MPS pencils it is far
    tamer: across 240 evaluations on four domains it retained at most ten extra columns and
    never put the cut below 2.2e-13, three decades clear of where the tension goes rough.
    The failure mode is real and has not been observed here, which is the honest summary.
    """
    if rank is not None:
        return int(rank)
    if diag.size == 0 or diag[0] <= 0.0:
        return 0
    return int((diag >= rtol * diag[0]).sum())


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


def regularize(A_B, A_I, reg="svd", rank=None, rtol=np.finfo(float).eps):
    """Regularize the pencil `(A_B, A_I)` for the GSVD, returning the map back.

    `reg` selects the mode:

    * `"svd"` -- economic QR of the stack, SVD of the triangular factor, keep the leading
      `rank` left singular directions. The default, and the best conditioned.
    * `"qrp"` -- column-pivoted QR, keep the leading `rank` pivoted columns. Cheaper, and it
      names actual basis columns, which makes it the readable one when debugging.
    * `None` -- no regularization at all.
    * `"tikhonov_qr"` -- a PENALTY instead of a truncation: unpivoted QR of
      `[A_B; tau*I; A_I]`, and the two row blocks of `Q` go to the GSVD. No rank is chosen,
      so none can change with `lam`. `back` is `R^-1`.
    * `"tikhonov"` -- the same penalty handed to the GSVD directly as `([A_B; tau*I], A_I)`.
      Identical in exact arithmetic; NOT free of rank decisions in practice, because LAPACK
      then decides the interior block's rank itself. `back` is the identity.

    For the penalty modes `rtol` sets the weight, `tau = rtol * ||[A_B; A_I]||_F`, and `rank`
    must be None. The module docstring records why the two differ and what they cannot do.

    There was a fourth, `"implicit"`, which passed `tola`/`tolb` to `gsvd4py` and let
    LAPACK's `?ggsvp3` pick the rank. `gsvd4py` 0.3.0 removed those parameters, so the mode
    went with them. It was never the default and the measurements are kept in
    `docs/design/mps.md` -- it truncated two to four columns harder on a different criterion
    and was only 1.16x cheaper at matched rank, so nothing here is lost.

    `rank=None` means **derive the rank here**, from this mode's own factorization at the
    relative cutoff `rtol`. That is not a convenience: the `"svd"` mode's SVD of `R` has the
    same singular values as the stacked pencil (`Q` has orthonormal columns), so asking
    `choose_rank` first computed the identical decomposition twice. Measured on a 134-column
    L-shape pencil, that redundant `svdvals` was 2.13 ms of an 8.19 ms evaluation -- 26% --
    and deriving the rank in place deletes it. `"qrp"` likewise reads its rank off `|diag R|`
    from the factorization it already performed, which is the whole point of pivoting.

    `choose_rank` remains available as a free function, for diagnostics and for choosing a
    rank to FREEZE, where the caller genuinely wants the number before the reduction.
    """
    A_B, A_I = np.asarray(A_B, dtype=float), np.asarray(A_I, dtype=float)
    m, n = A_B.shape[0], A_B.shape[1]

    if reg is None:
        r = n if rank is None else int(rank)
        return RegularizedPencil(A_B[:, :r], A_I[:, :r], lambda y: np.asarray(y), r)

    stack = np.vstack([A_B, A_I])

    if reg == "svd":
        Q, R = _qr_economic(stack)
        Z, s, Yt = _svd_gesdd_then_gesvd(R)
        r = _rank_from(s, rank, rtol)
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
        r = _rank_from(d, rank, rtol)
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

    if reg in _PENALTY:
        tau = _penalty_weight(stack, rank, rtol, reg)
        numer = np.vstack([A_B, tau * np.eye(n)])            # (m + n, n)

        if reg == "tikhonov":
            return RegularizedPencil(numer, A_I, lambda y: np.asarray(y, dtype=float), n,
                                     tau=tau)

        # `_qr_economic`, not `la.qr`: this is a tall QR of exactly the kind Accelerate's
        # `dgeqrf` corrupts the heap on. Unpivoted, so `back` needs no permutation.
        Q, R = _qr_economic(np.vstack([numer, A_I]))

        def back(y, _R=R):
            # [numer; A_I] = Q R, so a reduced vector y names c = R^-1 y: numer c = Q_1 y and
            # A_I c = Q_2 y, the statement that makes this an inverse.
            y = np.asarray(y, dtype=float)
            flat = y.ndim == 1
            c = la.solve_triangular(_R, y[:, None] if flat else y)
            return c[:, 0] if flat else c

        return RegularizedPencil(Q[:m + n], Q[m + n:], back, n, tau=tau)

    raise ValueError(f"unknown regularization {reg!r}; expected one of {_MODES}")


def _penalty_weight(stack, rank, rtol, reg):
    """`tau = rtol * ||stack||_F`, refusing the inputs that would make it meaningless.

    Relative, so scaling the pencil scales `tau` with it and leaves the tensions alone. On an
    equilibrated stack this is `rtol * sqrt(n_cols)`. Zero is refused rather than returned:
    `tau = 0` is `reg=None` with a singular `R` waiting in the QR form's `back`.
    """
    if rank is not None:
        raise ValueError(
            f"reg={reg!r} is a penalty and chooses no rank, so rank={rank!r} has nothing to "
            f"freeze. Freezing a rank is a truncating mode's option ('svd' or 'qrp')."
        )
    rtol = float(rtol)
    if not rtol > 0.0:
        raise ValueError(f"reg={reg!r} needs rtol > 0 to set its penalty, got {rtol!r}")
    tau = rtol * float(np.linalg.norm(stack))
    if not tau > 0.0:
        raise ValueError(f"reg={reg!r} cannot penalize an all-zero pencil")
    return tau
