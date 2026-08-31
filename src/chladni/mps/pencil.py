"""`MPSPencil`: a basis, two point sets, and everything up to the tension.

Domain-free, and speaks `k`. This is the answer to `docs/design/mps.md`'s second design
question -- how to expose a bare-bones solver for bespoke work -- and it is a second object
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
from ..helmholtz import check_k
from . import _gsvd
from ._regularize import choose_rank, regularize

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


def _as_k_ref(k_ref):
    """Normalize `k_ref` to None, a single k, or a validated `(lo, hi)` pair."""
    if k_ref is None:
        return None
    arr = np.atleast_1d(np.asarray(k_ref, dtype=float))
    if arr.size == 1:
        return check_k(arr[0])
    if arr.size != 2:
        raise ValueError(f"k_ref must be a wavenumber or a (lo, hi) pair, got {k_ref!r}")
    lo, hi = check_k(arr[0]), check_k(arr[1])
    if not lo < hi:
        raise ValueError(f"k_ref range must satisfy lo < hi, got ({lo}, {hi})")
    return (lo, hi)


def _rank_probe_ks(k_ref):
    """The wavenumbers the rank is chosen over: one, or `_RANK_SAMPLES` across the range."""
    if isinstance(k_ref, tuple):
        return np.linspace(k_ref[0], k_ref[1], _RANK_SAMPLES)
    return (k_ref,)


def assemble(bdry, interior, k):
    """The collocation matrices `(A_B, A_I)` at wavenumber `k`.

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

    The rank is frozen at construction; see `_regularize` for why it is an integer rather
    than a tolerance. `rtol` defaults to `sqrt(prec * eps)`, the geometric mean of what was
    asked for and what the arithmetic can do. At the default `prec = 1e-8` that is 1.5e-12,
    which is where lappy's `rtol` landed after measuring twelve domains -- derived here
    rather than fitted. It also moves in the right direction, which is easy to invert: a
    TIGHTER `prec` resolves more independent columns, so the cutoff falls and the rank rises.
    """

    def __init__(self, bdry, interior, *, reg="svd", rank=None, rtol=None, k_ref=None,
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
        rtol = float(np.sqrt(prec * np.finfo(float).eps)) if rtol is None else float(rtol)

        k_ref = _as_k_ref(k_ref)
        if rank is None and k_ref is not None:
            # frozen, chosen over the range: the minimum, because the rank grows with k
            rank = min(choose_rank(*equilibrate(*assemble(bdry, interior, k))[:2], rtol)
                       for k in _rank_probe_ks(k_ref))
        if rank is not None:
            rank = int(rank)
            if not 0 < rank <= len(bdry.family):
                raise ValueError(f"rank must lie in [1, {len(bdry.family)}], got {rank}")

        _set(self, bdry=bdry, interior=interior, reg=reg, rank=rank, rtol=rtol,
             k_ref=k_ref, prec=prec,
             _cache={})

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

    def matrices(self, k):
        """`(A_B, A_I)`: the raw collocation matrices, before equilibration."""
        return assemble(self.bdry, self.interior, k)

    def _equilibrated(self, k):
        key = ("eq", float(check_k(k)))
        hit = self._cache.get(key)
        if hit is None:
            hit = equilibrate(*self.matrices(k))
            self._store(key, hit)
        return hit

    def pencil(self, k):
        """`(B, I)`: the equilibrated pencil, which is what the GSVD actually sees."""
        B, I, _ = self._equilibrated(k)
        return B, I

    def log_scale(self, k):
        """Total per-column log scale: the family's own, plus this pencil's equilibration.

        `conventions.md`: "the log scales simply add, and a coefficient vector means the same
        function on every point set." A column of the equilibrated pencil is
        `psi_j * exp(-log_scale(k)[j])`, and unscaled coefficients are never materialized --
        at a `pi/8` corner the family's half of this spans about 1149 nats.
        """
        k = check_k(k)
        return self.basis.log_scale(k) + self._equilibrated(k)[2]

    @property
    def is_frozen(self):
        """True when the rank is fixed for the life of the pencil, False for threshold."""
        return self.rank is not None

    def rank_at(self, k):
        """The rank actually used at `k`.

        In threshold mode this is re-derived from the spectrum, so it varies with `k` --
        which is the point: the pencil's numerical rank GROWS with `k`, and a rank fixed at
        one end of a scan is wrong at the other.
        """
        if self.rank is not None:
            return self.rank
        if self.reg == "implicit":
            # LAPACK decides inside ?ggsvp3 and does not report it until the GSVD returns,
            # so the length of the tension vector IS the rank. Cached, so this is not a
            # second factorization.
            return len(self.tensions(k))
        # Every other mode derives its rank from its own factorization, so read it back off
        # the (cached) reduction rather than recomputing one here. Calling `choose_rank`
        # would repeat the decomposition `regularize` has already performed.
        return self.regularized(k).rank

    def regularized(self, k):
        """The `RegularizedPencil` at `k`, truncated to `rank_at(k)`."""
        key = ("reg", float(check_k(k)))
        hit = self._cache.get(key)
        if hit is None:
            B, I = self.pencil(k)
            hit = regularize(B, I, reg=self.reg, rank=self.rank, rtol=self.rtol)
            self._store(key, hit)
        return hit

    def rank_diagnostics(self, k):
        """The frozen rank and the singular values at `k`, so the choice can be audited.

        The frozen rank is right almost everywhere and this is how you check the exception:
        the numerical rank grows with k, so a rank chosen too high sits at the noise floor
        at the LOW end of a scan, which is where the tension goes rough.
        """
        B, I = self.pencil(k)
        s = la.svdvals(np.vstack([B, I]))
        if self.rank is not None:
            self._warn_if_unsafe(s, k)
        return RankDiagnostics(self.rank_at(k), s)

    def _warn_if_unsafe(self, s, k):
        """Warn when the frozen rank has reached the noise floor at this `k`.

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
            f"1e{decade} at this k, below the safe level {_UNSAFE_RATIO:.0e}. The retained "
            f"directions are no longer determined and the tension acquires spurious local "
            f"minima. The numerical rank grows with k, so this usually means the rank was "
            f"chosen at a single k above the range in use: pass k_ref=(lo, hi) instead.",
            PrecisionWarning,
            stacklevel=3,
        )

    def tensions(self, k):
        """Every generalized singular value of the reduced pencil, ascending."""
        return _gsvd.tensions_of(self.regularized(k))

    def tension(self, k):
        """The smallest generalized singular value: the quantity the search minimizes."""
        return float(self.tensions(k)[0])

    def coef(self, k, mult=1):
        """The `mult` minimizing coefficient vectors, `(n_cols, mult)`, in the SCALED space.

        The function they name is `sum_j c_j * exp(-log_scale(k)[j]) * psi_j`. Recovering the
        unscaled vector is not offered because it does not exist in float64.
        """
        c = _gsvd.coef_of(self.regularized(k), mult)
        return c if c.ndim == 2 else c[:, None]

    # --- the cache -------------------------------------------------------------------------

    def _store(self, key, value):
        cache = self._cache
        if len(cache) >= _CACHE_MAX:
            cache.pop(next(iter(cache)))
        cache[key] = value

    def __repr__(self):
        rank = self.rank if self.rank is not None else "threshold"
        return (f"MPSPencil(n_cols={self.n_cols}, n_bdry={self.n_bdry}, "
                f"n_int={self.n_int}, reg={self.reg!r}, rank={rank})")
