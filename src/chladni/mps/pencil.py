"""`MPSPencil`: a basis, two point sets, and everything up to the tension.

Domain-free, and speaks `lam` -- like `MPSEigensolver`, and unlike everything below it.

**The lam/k seam is the mps/helmholtz boundary, and nothing finer.** Every `mps` object takes
`lam`; every `helmholtz` object takes `k`. That line is where the meaning genuinely changes:
a Fourier-Bessel column IS a function of `k*r`, and the Bessel accuracy regimes are stated in
`(nu, k*r)`, so `k` is not a parametrisation choice down there but what the functions are. A
tension is not such a thing, so it does not inherit the variable.

This class used to take `k`, which put a second seam INSIDE one layer: two sibling classes,
same layer, different variables, and a user descending from solver to pencil crossed it
without any change in what the numbers meant. `conventions.md` asks that a parameter's name
state its variable; it does not ask a reader to remember which of two siblings is which.

It is also the variable the refinement wants. Fitting a quadratic to `sigma_1**2` over
identical sample points, the residual is smaller in `lam` in 17 of 17 cases across four
domains -- median 5.1x -- so a search built on this object works in `lam` throughout.

This is also the answer to `docs/design/mps.md`'s second design question -- how to expose a bare-bones solver for bespoke work -- and it is a second object
rather than a better-severed first one.

lappy tried to cut ONE `MPSEigensolver` free of the domain and failed twice: on boundary
conditions, where `make_bdry_vander` branched on `bc_param` and could not express a mixed
boundary except as a degenerate Robin blend; and on Rellich orthonormalization, which needs
quadrature weights the severed object did not carry.

Neither failure is available here. `Sampling.trace(k)` assembles `(1-bc)*u + bc*du_dn` from
the `Quadrature`'s own per-node normals and bc, so this module contains no boundary-condition
logic at all and never needs a `Domain` to ask what the condition is. And a `Quadrature`
carries its weights, so the deferred Rellich work has what it needs from the same object. The
dependency that survives is on a quadrature, which is the thing actually needed.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as la

import warnings

from .._core import PREC, Frozen, PrecisionWarning, _set, check_prec
from ..eigensolver import lam_to_k
from ..helmholtz import check_k
from . import _gsvd
from ._regularize import (_MODES, _PENALTY, _REACH_FLOOR, _REACH_ROUNDING, C_PEN,
                          PENALTY_PREC_MIN, choose_rank, regularize)

__all__ = ["MPSPencil", "assemble", "equilibrate", "RankDiagnostics"]

#: How many parameters' worth of reduced pencil to keep. Bounded because a scan visits
#: hundreds of k and each entry costs O((m_B + m_I) * rank); NOT an eviction policy in
#: lappy's sense -- there is no lock, no byte ledger and no weakref finalizer, because the
#: object is immutable and nothing cached can go stale.
_CACHE_MAX = 64

#: How many wavenumbers to probe when the rank is chosen over a RANGE rather than at a
#: point. The pencil's numerical rank grows with k -- measured on a 54-column disk at the
#: default rtol: 31 at k = 2.3 rising to 47 at k = 5.7 -- so a rank frozen at one k is too
#: large at the low end of a scan. Nine points is enough to find the minimum without
#: assuming the growth is monotone, which is observed but not proved.
_RANK_SAMPLES = 9

#: Below this relative singular value at the cut, the retained direction is within a couple
#: of decades of machine epsilon and is no longer determined. Measured on the disk: at
#: s_r/s_0 = 1e-13 the tension has exactly one local minimum in a one-eigenvalue window and
#: no wrong-way steps; at 1e-16 it has ninety, with 39% of the steps on the descending wall
#: going the wrong way. The transition is sharp and sits at eps, so this leaves three
#: decades of margin.
_UNSAFE_RATIO = 1e-13


class RankDiagnostics(tuple):
    """What the frozen rank was chosen from: `rank` and the descending singular values `s`."""

    __slots__ = ()

    def __new__(cls, rank, s):
        return super().__new__(cls, (rank, s))

    rank = property(lambda self: self[0])
    s = property(lambda self: self[1])


def _warn_if_out_of_reach(tau, prec, n_cols):
    """Warn when a penalty's `tau` is outside the band that can deliver `prec`.

    Two jaws (`_regularize._REACH_FLOOR`, `_REACH_ROUNDING`): a floor too high to separate
    members `prec` apart, or a `tau` so near rounding that the tension is noise. Warned rather
    than refused -- a penalty below its reach is what the caller asked for, and `reg="auto"`
    never builds one. The constants are one rectangle family's, and the message says so.
    """
    eps = np.finfo(float).eps
    if tau > _REACH_FLOOR * prec:
        why = (f"its floor, about 1.4 tau = {1.4 * tau:.1e}, is too high to separate eigenvalues "
               f"prec apart (tau above {_REACH_FLOOR} prec lost cluster members)")
    elif tau < _REACH_ROUNDING * eps * np.sqrt(n_cols):
        why = (f"tau = {tau:.1e} is within {_REACH_ROUNDING:g}x of rounding "
               f"(eps sqrt(n) = {eps * np.sqrt(n_cols):.1e}), where the tension is noise the "
               f"search cannot read slopes from")
    else:
        return
    warnings.warn(
        f"a penalty regularization at prec={prec:.0e} is outside its measured reach: {why}. "
        f"No penalty tau serves prec much below 150 eps sqrt(n); use reg='svd' or 'auto' there. "
        f"(Reach measured on one rectangle family: notes/penalty_prec_limit_explained.md.)",
        PrecisionWarning,
        stacklevel=3,
    )


def _as_lam_ref(lam_ref):
    """Normalize `lam_ref` to None, a single lam, or a validated `(lo, hi)` pair."""
    if lam_ref is None:
        return None
    arr = np.atleast_1d(np.asarray(lam_ref, dtype=float))
    if arr.size == 1:
        return float(lam_to_k(arr[0])) ** 2
    if arr.size != 2:
        raise ValueError(f"lam_ref must be an eigenvalue or a (lo, hi) pair, got {lam_ref!r}")
    lo, hi = float(lam_to_k(arr[0])) ** 2, float(lam_to_k(arr[1])) ** 2
    if not lo < hi:
        raise ValueError(f"lam_ref range must satisfy lo < hi, got ({lo}, {hi})")
    return (lo, hi)


def _rank_probe_lams(lam_ref):
    """The eigenvalues the rank is chosen over: one, or `_RANK_SAMPLES` across the range."""
    if isinstance(lam_ref, tuple):
        return np.linspace(lam_ref[0], lam_ref[1], _RANK_SAMPLES)
    return (lam_ref,)


def assemble(bdry, interior, k):
    """The collocation matrices `(A_B, A_I)` at wavenumber `k`.

    Takes `k`, not `lam`: this is the call into `helmholtz`, and below the mps boundary the
    wavenumber is what the basis functions are made of rather than a parametrisation choice.
    `MPSPencil.matrices(lam)` is the `lam`-facing wrapper.

    The whole of it. `trace` is the boundary condition, applied per node one layer down, and
    that is why this function has no branches -- compare lappy's `make_bdry_vander`, whose
    three-way branch on `np.all(bc_param == ...)` sends a mixed Dirichlet/Neumann boundary
    down the Robin path.
    """
    k = check_k(k)
    return bdry.trace(k), interior.values(k)


def equilibrate(A_B, A_I):
    """Scale each column to unit 2-norm over the STACKED pencil. Returns `(B, I, log_norms)`.

    Over the stack, not over either block separately: the GSVD sees the pair, and normalizing
    the blocks independently would change the very ratio the tension measures.

    This is `mps` owning pencil conditioning, which `conventions.md` separates from
    `helmholtz` owning entry-wise accuracy. The two compose because the log scales ADD, and
    `log_norms` is what gets added.

    A column that underflowed to exact zeros keeps scale 1 rather than dividing by zero, and
    is left for the rank truncation to drop -- lappy pruned such columns by hand instead.
    """
    A_B, A_I = np.asarray(A_B, dtype=float), np.asarray(A_I, dtype=float)
    norms = np.linalg.norm(np.vstack([A_B, A_I]), axis=0)
    safe = np.where(norms > 0.0, norms, 1.0)
    log_norms = np.where(norms > 0.0, np.log(safe), 0.0)
    return A_B / safe, A_I / safe, log_norms


class MPSPencil(Frozen):
    """A Helmholtz basis sampled at boundary and interior collocation points.

    `bdry` and `interior` are `helmholtz.Sampling` objects over the same basis. `bdry` must
    come from a `Quadrature`, since that is what carries the normals and boundary conditions
    `trace` needs.

    The rank is re-derived per `lam` at the cutoff `rtol` (THRESHOLD), or frozen at
    construction by `rank=` or `lam_ref=`; see `_regularize` for the trade. `rtol` defaults to
    `sqrt(prec * eps)`, the geometric mean of what was asked for and what the arithmetic can
    do. At the default `prec = 1e-8` that is 1.5e-12, which is where lappy's `rtol` landed after
    measuring twelve domains -- derived here rather than fitted. It also moves in the right
    direction, which is easy to invert: a TIGHTER `prec` resolves more independent columns, so
    the cutoff falls and the rank rises.

    `reg="tikhonov_qr"` (or `"tikhonov"`) replaces the truncation with a penalty of weight
    `tau = rtol * sqrt(n_cols)`, and chooses no rank at all, so `rank=` and `lam_ref=` are refused
    with it. Its `rtol` defaults to `C_PEN * prec / sqrt(n_cols)`, so `tau = C_PEN * prec`: a
    penalty's floor IS its `tau`, and the truncation's `sqrt(prec * eps)` put it at 2.4 prec by
    1e-14. An explicit penalty outside its measured reach warns (`PrecisionWarning`) and runs.

    **The default, `reg="auto"`, is the threshold at every `prec`** (since 2026-10-10; it was
    `"tikhonov_qr"` at `prec >= 1e-8` from 2026-09-24). A threshold rank leaves a decision to
    rounding (`notes/session_2026-09-19.md`), but on the douse benchmark the penalty lost to it at
    1e-6, 1e-8 and 1e-10 -- more silent losses, a looser bound, more wall time
    (`notes/regularization_1e10.md`). Passing `rtol=`, `rank=` or `lam_ref=` asks for a
    truncation and gets one; a penalty is asked for by name.
    See `_regularize` for both.
    """

    def __init__(self, bdry, interior, *, reg="auto", rank=None, rtol=None, lam_ref=None,
                 prec=PREC):
        if getattr(bdry, "normals", None) is None or getattr(bdry, "bc", None) is None:
            raise ValueError(
                "the boundary Sampling must be built from a Quadrature, which carries the "
                "normals and boundary conditions the trace needs: use "
                "basis.at(domain.bdry_quad(n)), not basis.at(bare_points)"
            )
        if bdry.family is not interior.family:
            raise ValueError(
                "the boundary and interior Samplings must come from the same basis -- they "
                "index the same column space, and a pencil whose two blocks mean different "
                "columns is not a pencil"
            )
        prec = check_prec(prec)
        if reg not in _MODES + ("auto",):
            raise ValueError(f"unknown regularization {reg!r}; expected one of "
                             f"{_MODES + ('auto',)}")
        if reg == "auto":
            # The threshold at every `prec` since 2026-10-10 (`PENALTY_PREC_MIN` is infinite).
            explicit = rtol is not None or rank is not None or lam_ref is not None
            reg = ("tikhonov_qr" if not explicit and prec >= PENALTY_PREC_MIN else "svd")
        n_cols = len(bdry.family)
        if rtol is None:
            # A penalty's floor is its `tau`, so `tau` scales with `prec`; a truncation's floor is
            # the basis's, and its cutoff sits between `prec` and the arithmetic. `_regularize`
            # forms `tau = rtol * ||A||_F`, which is `rtol * sqrt(n_cols)` once equilibrated.
            rtol = (C_PEN * prec / np.sqrt(n_cols) if reg in _PENALTY
                    else float(np.sqrt(prec * np.finfo(float).eps)))
        rtol = float(rtol)
        if reg in _PENALTY and (rank is not None or lam_ref is not None):
            # Refused here, before `lam_ref` would go and choose a rank for a mode with none.
            raise ValueError(
                f"reg={reg!r} is a penalty and chooses no rank, so rank= and lam_ref= have "
                f"nothing to freeze. They belong to the truncating modes 'svd' and 'qrp'."
            )
        if reg in _PENALTY and not rtol > 0.0:
            raise ValueError(f"reg={reg!r} needs rtol > 0 to set its penalty, got {rtol!r}")
        if reg in _PENALTY:
            _warn_if_out_of_reach(rtol * np.sqrt(n_cols), prec, n_cols)

        lam_ref = _as_lam_ref(lam_ref)
        if rank is None and lam_ref is not None:
            # frozen, chosen over the range: the minimum, because the rank grows with lam
            rank = min(choose_rank(*equilibrate(*assemble(bdry, interior,
                                                          lam_to_k(lam)))[:2], rtol)
                       for lam in _rank_probe_lams(lam_ref))
        if rank is not None:
            rank = int(rank)
            if not 0 < rank <= len(bdry.family):
                raise ValueError(f"rank must lie in [1, {len(bdry.family)}], got {rank}")

        _set(self, bdry=bdry, interior=interior, reg=reg, rank=rank, rtol=rtol,
             lam_ref=lam_ref, prec=prec,
             _cache={})

    def with_rtol(self, rtol):
        """The same pencil at a different rank-truncation threshold.

        Follows the `with_prec` transform protocol: a new object, sharing both `Sampling`s, so
        the assembly is not repeated -- only `regularize`'s cutoff moves. That is what makes a
        five-step calibration of the threshold affordable (`mps.calibrate`).

        The cache is NOT shared. It holds regularized factorizations, which are exactly what
        `rtol` changes; carrying them over would return the old rank under the new threshold.
        """
        return type(self)(self.bdry, self.interior, reg=self.reg,
                          rank=self.rank if self.is_frozen else None,
                          rtol=float(rtol), lam_ref=None, prec=self.prec)

    @classmethod
    def from_basis(cls, basis, bdry_quad, int_pts, **kw):
        """Bind the basis to both point sets, then build. Sugar, and the common spelling."""
        return cls(basis.at(bdry_quad), basis.at(int_pts), **kw)

    # --- what it is made of ---------------------------------------------------------------

    @property
    def basis(self):
        return self.bdry.family

    @property
    def n_cols(self):
        return len(self.basis)

    @property
    def n_bdry(self):
        return len(self.bdry)

    @property
    def n_int(self):
        return len(self.interior)

    @property
    def bdry_pts(self):
        return self.bdry.pts

    @property
    def int_pts(self):
        return self.interior.pts

    # --- the pipeline ----------------------------------------------------------------------

    def matrices(self, lam):
        """`(A_B, A_I)`: the raw collocation matrices, before equilibration.

        The one place `k` appears, because `assemble` calls into `helmholtz` and that is the
        layer where the wavenumber is what the functions are made of.
        """
        return assemble(self.bdry, self.interior, lam_to_k(lam))

    def _equilibrated(self, lam):
        key = ("eq", float(lam))
        hit = self._cache.get(key)
        if hit is None:
            hit = equilibrate(*self.matrices(lam))
            self._store(key, hit)
        return hit

    def pencil(self, lam):
        """`(B, I)`: the equilibrated pencil, which is what the GSVD actually sees."""
        B, I, _ = self._equilibrated(lam)
        return B, I

    def log_scale(self, lam):
        """Total per-column log scale: the family's own, plus this pencil's equilibration.

        `conventions.md`: "the log scales simply add, and a coefficient vector means the same
        function on every point set." A column of the equilibrated pencil is
        `psi_j * exp(-log_scale(k)[j])`, and unscaled coefficients are never materialized --
        at a `pi/8` corner the family's half of this spans about 1149 nats.
        """
        return self.basis.log_scale(lam_to_k(lam)) + self._equilibrated(lam)[2]

    @property
    def is_frozen(self):
        """True when the rank is fixed for the life of the pencil, False for threshold."""
        return self.rank is not None

    def rank_at(self, lam):
        """How many directions the pencil actually yields at `lam`.

        In threshold mode this is re-derived from the spectrum, so it varies with `lam` --
        which is the point: the pencil's numerical rank GROWS with `lam`, and a rank fixed
        at one end of a scan is wrong at the other.

        The contract is the same for every mode: `len(tensions(lam)) == rank_at(lam)`, and
        `coef(lam, mult=rank_at(lam))` is always a valid call. A pre-truncating mode gets
        that for free, because it hands the GSVD a reduced stack of full column rank; the
        `None` mode does not, and the branch below is what makes it hold there too.
        """
        if self.reg is None:
            # No pre-truncation, so LAPACK sees a pencil that may be rank deficient and
            # applies its OWN criterion, reporting `k + l`. That number is not recoverable
            # here: on the L-shape fixture it runs 44, 48, 50 at lam = 10.89, 25, 60, while
            # `n_cols` is 50 throughout and `choose_rank` gives 42, 45, 48 at this rtol and
            # 50, 50, 50 at eps. It agrees with neither, and nothing short of the GSVD
            # predicts it -- so read it off the tension vector, whose length IS `k + l`.
            # Cached, so this is not a second factorization.
            #
            # This holds whether or not a rank was frozen: `None` truncates to the FIRST
            # `rank` columns without regard to conditioning, so LAPACK can still find that
            # stack deficient. The frozen shortcut below would report the wrong number.
            return len(self.tensions(lam))
        if self.reg in _PENALTY:
            # Also read off the tension vector, for the same reason: nothing is truncated, so
            # the count is LAPACK's `k + l`. The penalty makes the stack full rank, so this is
            # `n_cols` -- and `tests/test_regularization.py` checks that across a scan rather
            # than assuming it, because a count that moved would be a rank change by the back
            # door, which is the one thing these modes exist to rule out.
            return len(self.tensions(lam))
        if self.rank is not None:
            return self.rank
        # A pre-truncating mode derives its rank from its own factorization, so read it back
        # off the (cached) reduction rather than recomputing one here. Calling `choose_rank`
        # would repeat the decomposition `regularize` has already performed.
        return self.regularized(lam).rank

    def regularized(self, lam):
        """The `RegularizedPencil` at `lam`: truncated to `rank_at(lam)`, or penalized."""
        key = ("reg", float(lam))
        hit = self._cache.get(key)
        if hit is None:
            B, I = self.pencil(lam)
            hit = regularize(B, I, reg=self.reg, rank=self.rank, rtol=self.rtol)
            self._store(key, hit)
        return hit

    def rank_diagnostics(self, lam):
        """The frozen rank and the singular values at `lam`, so the choice can be audited.

        The frozen rank is right almost everywhere and this is how you check the exception:
        the numerical rank grows with lam, so a rank chosen too high sits at the noise
        floor at the LOW end of a scan, which is where the tension goes rough.
        """
        B, I = self.pencil(lam)
        s = la.svdvals(np.vstack([B, I]))
        if self.rank is not None:
            self._warn_if_unsafe(s, lam)
        return RankDiagnostics(self.rank_at(lam), s)

    def _warn_if_unsafe(self, s, lam):
        """Warn when the frozen rank has reached the noise floor at this `lam`.

        A reported precision is never a claim contradicted elsewhere in the same object
        (`conventions.md`), and this is the contradiction that matters here: a pencil whose
        cut sits at machine epsilon returns a tension with dozens of spurious local minima
        per true one, which reads as an eigenvalue and is not.

        The message quotes the ratio rounded to a power of ten, on the same reasoning as
        `Sampling._check_accuracy`: Python's default warning filter keys on message text,
        and a k-dependent float would fire on every evaluation of a scan.
        """
        if s.size == 0 or s[0] <= 0.0:
            return
        ratio = s[self.rank - 1] / s[0]
        if ratio >= _UNSAFE_RATIO:
            return
        decade = int(np.floor(np.log10(ratio))) if ratio > 0 else -np.inf
        warnings.warn(
            f"the frozen rank {self.rank} sits at a relative singular value of about "
            f"1e{decade} at this lam, below the safe level {_UNSAFE_RATIO:.0e}. The retained "
            f"directions are no longer determined and the tension acquires spurious local "
            f"minima. The numerical rank grows with lam, so this usually means the rank was "
            f"chosen at a single k above the range in use: pass lam_ref=(lo, hi) instead.",
            PrecisionWarning,
            stacklevel=3,
        )

    def tensions(self, lam):
        """Every generalized singular value of the reduced pencil, ascending."""
        return _gsvd.tensions_of(self.regularized(lam))

    def tension(self, lam):
        """The smallest generalized singular value: the quantity the search minimizes."""
        return float(self.tensions(lam)[0])

    def coef(self, lam, mult=1):
        """The `mult` minimizing coefficient vectors, `(n_cols, mult)`, in the SCALED space.

        The function they name is `sum_j c_j * exp(-log_scale(lam)[j]) * psi_j`. Recovering the
        unscaled vector is not offered because it does not exist in float64.

        ONE EXCEPTION to the shape, and it is `reg=None` with a frozen `rank`: the rows are
        `rank`, not `n_cols`. That mode keeps the first `rank` basis columns and its `back` is
        the identity on them, so the vector is expressed in the columns retained rather than
        in the full basis. Padding it with zeros to `n_cols` names the identical function --
        checked, the tension agrees with the reported one -- but it is NOT the prefix of the
        vector the untruncated solve would give, which is a different problem over a larger
        space and points elsewhere entirely.

        `"svd"` and `"qrp"` both scatter back to `n_cols`, and `None` in threshold mode
        retains every column, so the other five mode/rank combinations return `n_cols` rows.
        The penalty modes drop nothing and always return `n_cols` rows.
        Padding here too would be a one-line change to `regularize`; it is not made because
        nothing needs it and a silent shape change is worse than a documented one.
        """
        c = _gsvd.coef_of(self.regularized(lam), mult)
        return c if c.ndim == 2 else c[:, None]

    def family_coef(self, lam, x):
        """`x` from `coef(lam)` in the coordinates of the FAMILY's own columns.

        A column of the equilibrated pencil is the family's column divided by this pencil's
        per-column norm over its collocation points, so `basis.at(pts).values(k) @
        family_coef(lam, x)` is the function `x` names, on ANY points -- what a consumer needs
        to evaluate an eigenfunction off the collocation set, as the boundary integrals do on
        their own quadrature nodes. Only the pencil's own scale is divided out: the family's
        `log_scale` is already out of its columns, and adding it again is the double count the
        scaled-column convention exists to prevent (`conventions.md`).

        `x` is `(n, )` or `(n, m)`. `coef`'s one shape exception -- `reg=None` with a frozen
        rank, `rank` rows in the retained columns -- is scattered back to `n_cols` rows here,
        so the result always matches the family.
        """
        x = np.asarray(x)
        own = self._equilibrated(float(lam))[2]
        if x.shape[0] != own.size:
            full = np.zeros((own.size,) + x.shape[1:], dtype=x.dtype)
            full[:x.shape[0]] = x
            x = full
        scale = np.exp(-own)
        return x * (scale if x.ndim == 1 else scale[:, None])

    def interior_image(self, lam):
        """The candidate eigenfunction at `lam`, sampled on the interior points. `(n_int,)`, unit.

        `search.md` section 2.4 clause (1)'s `u_i = I(lam_i) @ coef(lam_i, 1)` -- EQUILIBRATED and
        deliberately not unscaled, because the unscaled vector does not exist in float64 and the
        comparison the angle makes does not need it.

        **The length is `n_int` at every `lam`, and that is the whole reason the seam is cheap.**
        The coefficient vector's width moves with `lam` under a threshold rank, so a caller
        reasoning from `coef` alone would get vectors it cannot compare. `I` has `n_int` ROWS --
        `interior` is a `Sampling` fixed at construction and only `k` changes inside `assemble` --
        so the product lands on the same points for every `lam`. That fixed shared point set is
        exactly what the candidate seam promises, and it holds by construction rather than by
        agreement between callers.

        **It is informative only ACROSS `lam`.** Within one `lam` the GSVD gives `A_I x_j = s_j v_j`
        with `V` orthogonal, so the interior images of distinct generalized singular vectors are
        orthogonal BY CONSTRUCTION -- measured 4.6e-16 -- and an angle between two of them says
        nothing about the eigenfunctions. `search.md` section 2.4 records this; it is repeated here
        because this method is where someone would reach for the wrong comparison.

        Costs one GSVD in `separate` mode. At an `lam` the search has already tensioned, the
        `RegularizedPencil` is cached, so no assembly, equilibration or factorization is
        repeated -- but the `separate`-mode call is not free: measured 1.16 ms against 1.81 ms
        for a complete fresh evaluation on a `family_B` pencil, so about 0.6 of one. At a cold
        `lam` it pays the factorization too and costs a full evaluation (1.83 ms).

        The cache that makes the cheap case cheap is `_cache`, FIFO with `_CACHE_MAX = 64`,
        while a solve samples far more points than that -- so this only holds because
        `search.md` section 2.6 asks for modes at the REPORTED ROOTS at the END, and those are
        the freshest entries. Measured on the same case: 4 of 4 roots still cached after an
        89-evaluation solve. A caller that asks mid-scan, or for a root found long before, pays
        the full price and should expect to.
        """
        lam = float(lam)
        _, I = self.pencil(lam)
        c = self.coef(lam, 1)[:, 0]
        if c.size != I.shape[1]:
            # `reg=None` with a frozen rank expresses the vector in the RETAINED columns rather
            # than in the full basis -- `coef`'s one documented shape exception. Scatter it back
            # rather than let the product fail or, worse, broadcast.
            full = np.zeros(I.shape[1], dtype=c.dtype)
            full[:c.size] = c
            c = full
        u = I @ c
        n = float(np.linalg.norm(u))
        return u / n if n > 0 else u

    # --- the cache -------------------------------------------------------------------------

    def _store(self, key, value):
        cache = self._cache
        if len(cache) >= _CACHE_MAX:
            cache.pop(next(iter(cache)))
        cache[key] = value

    def __repr__(self):
        if self.reg in _PENALTY:
            tail = f"rtol={self.rtol:.3g}"
        else:
            tail = f"rank={self.rank if self.rank is not None else 'threshold'}"
        return (f"MPSPencil(n_cols={self.n_cols}, n_bdry={self.n_bdry}, "
                f"n_int={self.n_int}, reg={self.reg!r}, {tail})")
