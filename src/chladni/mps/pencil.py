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
                          choose_rank, penalty_tensions, reduce_penalty, regularize,
                          svd_factors)
from .policy import Policy

__all__ = ["MPSPencil", "assemble", "equilibrate", "RankDiagnostics"]

#: How many VECTORS to keep -- tensions, equilibration scales, ranks and coefficient vectors,
#: keyed by parameter. Never matrices: an equilibrated block or a reduced pencil is
#: O((m_B + m_I) * n_cols), 10-20 MB on a 562-column basis, and caching 64 of them held ~0.5 GB
#: per solver while the douse backend kept its last solver alive (2026-10-13: N = 6 runs at
#: 3-4.4 GB). A search almost never asks for the same parameter's MATRICES twice; what it
#: reuses is a tension (absorbed here and by the objective's own cache) and, at a found level,
#: the scales `family_coef` needs right after `coef` (`tests/test_pencil_cache.py`). A vector
#: entry is a few kB, so this bound is about entry count, not memory. NOT an eviction policy in
#: lappy's sense -- there is no lock, no byte ledger and no weakref finalizer, because the
#: object is immutable and nothing cached can go stale.
_CACHE_MAX = 256

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
            # The threshold at every `prec`, decided 2026-10-10 (`notes/regularization_1e10.md`):
            # on the douse benchmark truncation beat the penalty on every axis. The penalty is
            # `reg="tikhonov_qr"`, explicitly.
            reg = "svd"
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
        """`(B, I, scales)`, assembled fresh. Only the per-column SCALES are kept, so a
        `family_coef` or `log_scale` after it does not assemble again."""
        out = equilibrate(*self.matrices(lam))
        self._store(("scale", float(lam)), out[2])
        return out

    def _scales(self, lam):
        """This pencil's per-column equilibration log scales at `lam`: cached, a vector."""
        hit = self._cache.get(("scale", float(lam)))
        return hit if hit is not None else self._equilibrated(lam)[2]

    def pencil(self, lam):
        """`(B, I)`: the equilibrated pencil, which is what the GSVD actually sees. Not
        cached -- matrices never are."""
        B, I, _ = self._equilibrated(lam)
        return B, I

    def log_scale(self, lam):
        """Total per-column log scale: the family's own, plus this pencil's equilibration.

        `conventions.md`: "the log scales simply add, and a coefficient vector means the same
        function on every point set." A column of the equilibrated pencil is
        `psi_j * exp(-log_scale(k)[j])`, and unscaled coefficients are never materialized --
        at a `pi/8` corner the family's half of this spans about 1149 nats.
        """
        return self.basis.log_scale(lam_to_k(lam)) + self._scales(lam)

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
            # The tension vector is cached, so this is not a second factorization.
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
        # off the reduction (its rank is cached, a scalar) rather than calling `choose_rank`,
        # which would repeat the decomposition `regularize` performs.
        hit = self._cache.get(("rank", float(lam)))
        return hit if hit is not None else self.regularized(lam).rank

    def regularized(self, lam):
        """The `RegularizedPencil` at `lam`: truncated to `rank_at(lam)`, or penalized.

        Built fresh on every call: it holds the reduced blocks, and matrices are never cached.
        Its rank is kept (a scalar), and so are the scales from the equilibration under it."""
        B, I = self.pencil(lam)
        out = regularize(B, I, reg=self.reg, rank=self.rank, rtol=self.rtol)
        self._store(("rank", float(lam)), out.rank)
        self._keep_spectrum(lam, out)
        return out

    def _regularized_for(self, lam, policy):
        """The `RegularizedPencil` at `lam` under `policy` (normalized; None = the pencil's own)."""
        if policy is None:
            return self.regularized(lam)
        if policy.kind == "rank":
            return self._regularized_at(lam, policy.rank)
        if self.reg != "svd":
            raise ValueError(f"the coarse penalty is reduced from the 'svd' mode's factorization; "
                             f"this pencil is reg={self.reg!r}")
        B, I = self.pencil(lam)
        f = svd_factors(B, I)
        out = reduce_penalty(f, policy.tau_rel * float(f.s[0]))
        self._keep_spectrum(lam, out)
        return out

    def _penalty_tensions(self, lam, policy):
        """The coarse penalty's tensions at `lam`, by the CS route (`penalty_tensions`), keeping
        the spectrum the same factorization computed."""
        if self.reg != "svd":
            raise ValueError(f"the coarse penalty is reduced from the 'svd' mode's factorization; "
                             f"this pencil is reg={self.reg!r}")
        B, I = self.pencil(lam)
        f = svd_factors(B, I)
        s = np.asarray(f.s, dtype=float)
        s.setflags(write=False)
        self._store(("spec", float(lam)), s)
        return penalty_tensions(f, policy.tau_rel * float(f.s[0]))

    def _regularized_at(self, lam, rank):
        """The `RegularizedPencil` at `lam` truncated to `rank` directions, or the pencil's own
        when `rank` is None. A frozen rank is a truncation, so a penalty pencil refuses it."""
        if rank is None:
            return self.regularized(lam)
        if self.reg in _PENALTY:
            raise ValueError(f"reg={self.reg!r} is a penalty and chooses no rank, so it cannot be "
                             f"read at a frozen rank")
        B, I = self.pencil(lam)
        out = regularize(B, I, reg=self.reg, rank=int(rank), rtol=self.rtol)
        self._keep_spectrum(lam, out)
        return out

    def _keep_spectrum(self, lam, out):
        """Keep the stack's singular values when the factorization just computed them: `"svd"`
        takes the SVD of R, so they are free; the other modes compute no singular values."""
        if self.reg == "svd" and out.s is not None:
            s = np.asarray(out.s, dtype=float)
            s.setflags(write=False)
            self._store(("spec", float(lam)), s)

    def spectrum(self, lam):
        """The singular values of the equilibrated stack `[B; I]` at `lam`, descending. A VECTOR.

        The stable rank (`notes/triangle_apex_planner.md` sections 9-12) is read from two of these
        at nearby `lam`, so it needs no matrix kept. Under `reg="svd"` the evaluation that made a
        tension already computed it, and it is cached beside the tension; under another mode it
        costs an assembly and an SVD of the stack."""
        key = ("spec", float(lam))
        hit = self._cache.get(key)
        if hit is None:
            if self.reg == "svd":
                self._regularized_at(lam, self.rank)
                hit = self._cache.get(key)
            if hit is None:                         # another mode, or evicted under threads
                B, I = self.pencil(lam)
                hit = la.svdvals(np.vstack([B, I]))
                hit.setflags(write=False)
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

    def tensions(self, lam, policy=None):
        """Every generalized singular value of the reduced pencil, ascending. Cached: a vector.

        `policy` (`chladni.mps.policy.Policy`) names the rank to read at: None or
        `Policy.pencil()` is the pencil's own, keyed exactly as before policies existed; a frozen
        rank is keyed apart, so two policies at one `lam` never overwrite each other.
        Returned read-only, because the cached array is shared by every caller."""
        policy = Policy.normalize(policy)
        key = ("tens", float(lam)) if policy is None else ("tens", float(lam), policy.key)
        hit = self._cache.get(key)
        if hit is None:
            if policy is not None and policy.kind == "penalty":
                hit = self._penalty_tensions(lam, policy)
            else:
                hit = _gsvd.tensions_of(self._regularized_for(lam, policy))
            hit.setflags(write=False)
            self._store(key, hit)
        return hit

    def tension(self, lam, policy=None):
        """The smallest generalized singular value: the quantity the search minimizes."""
        return float(self.tensions(lam, policy)[0])

    def coef(self, lam, mult=1, rank=None):
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

        `rank` truncates at that many directions instead of the pencil's own: certification at
        the rank a level was refined at (`Level.rank`). None is the pencil's own.
        """
        key = (("coef", float(lam), int(mult)) if rank is None
               else ("coef", float(lam), int(mult), int(rank)))
        hit = self._cache.get(key)
        if hit is None:
            c = _gsvd.coef_of(self._regularized_at(lam, rank), mult)
            hit = c if c.ndim == 2 else c[:, None]
            hit.setflags(write=False)
            self._store(key, hit)
        return hit

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
        own = self._scales(float(lam))
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

        Costs one assembly and one factorization, plus the `separate`-mode GSVD for the
        coefficient: no matrices are cached (`_CACHE_MAX`), so there is no cheap case. The
        objective caches the returned VECTOR, which is what a search re-reads.
        """
        lam = float(lam)
        B, I = self.pencil(lam)
        reg = regularize(B, I, reg=self.reg, rank=self.rank, rtol=self.rtol)
        c = _gsvd.coef_of(reg, 1)
        c = (c if c.ndim == 1 else c[:, 0])
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
        # Tolerant of concurrent writers: `MPSEigensolver.tensions(array)` maps over a thread
        # pool, and a 186-evaluation solve writes ~560 entries against `_CACHE_MAX`, so eviction
        # runs under threads. Two threads may pick the same oldest key (`pop` with a default), or
        # the dict may change between `iter` and `next` (`RuntimeError`, retried once). The
        # cache is a memo, so a lost or duplicate eviction costs nothing (2026-10-14).
        cache = self._cache
        if len(cache) >= _CACHE_MAX:
            for _ in range(2):
                try:
                    cache.pop(next(iter(cache), None), None)
                    break
                except RuntimeError:            # dictionary changed size during iteration
                    continue
        cache[key] = value

    def __repr__(self):
        if self.reg in _PENALTY:
            tail = f"rtol={self.rtol:.3g}"
        else:
            tail = f"rank={self.rank if self.rank is not None else 'threshold'}"
        return (f"MPSPencil(n_cols={self.n_cols}, n_bdry={self.n_bdry}, "
                f"n_int={self.n_int}, reg={self.reg!r}, {tail})")
