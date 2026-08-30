"""The evaluation contract: what a family is, and what a family sampled at points is.

Two objects, and the split between them is the whole performance design.

A `HelmholtzFamily` is k-independent and geometry-independent -- it knows what its columns
ARE. A `Sampling` is a family bound to a fixed set of points, holding everything that does
not depend on k: radii, angles, angular factors, the pieces of the chain rule. The solver
builds one `Sampling` at construction and then calls `sampling.values(k)` for hundreds of k.

lappy did the same caching with `instance_cache`: 370 lines with a lock, a global byte
ledger, weakref finalizers and an eviction policy, keyed on a value-hashed `PointSet`. Here
the object IS the cache. It is immutable, so it is thread-safe with no lock; it is owned by
the caller, so there is nothing to evict; and `family.values(k, pts)` remains available as
sugar for a one-off, documented as sugar.

`trace(k)` is the seam the solver actually wants. A `Sampling` built from a `Quadrature`
carries `normals` and `bc`, so it can assemble `(1 - bc)*u + bc*du_dn` itself -- which
deletes lappy's `make_bdry_vander` boundary-condition branch (mps.py:257-289) and lets a
Robin or mixed-BC boundary share one Bessel pass between the value and the derivative
instead of paying for two.

`cols` is one index space, the global column index, at every level. lappy has two inside
`FourierBesselBasis` with the translation duplicated across two methods, and
`MultiBasis._dispatch_cols` silently reorders output for an unsorted `cols` while promising
"in the given order". Here `cols` is honoured as given, and the test passes shuffled indices.
"""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import NamedTuple

import numpy as np

from .._core import PREC, Frozen, PrecisionWarning, _set, check_prec

__all__ = ["HelmholtzFamily", "Sampling", "SingularStructure", "check_k",
           "PREC", "PrecisionWarning"]


def check_k(k):
    """The wavenumber: a finite real, strictly positive.

    One rule, which is the practical argument for carrying `k` rather than `lam` down here:
    a solver's line search can hand `lam` a negative iterate, and `sqrt` of that is a complex
    number arriving three functions later as a confusing error about array dtypes.
    """
    try:
        value = float(k)
    except (TypeError, ValueError):
        raise ValueError(f"k must be a positive real wavenumber, got {k!r}") from None
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"k must be finite and strictly positive, got {value!r}")
    return value


class SingularStructure(NamedTuple):
    """Per-column corner-singularity structure, for a corner-graded quadrature.

    `origin[j]` is where column j is singular, NaN where it is regular everywhere (a
    fundamental solution or a plane wave); `exponent[j]` is its leading radial exponent
    there, 0 where regular.

    This exists because geometry's `Panel` seam exists: a Gauss-Jacobi rule graded to
    `r**(2*p - 2)` needs `p` per column, and the alternative is the solver re-deriving it
    from the domain, which is where lappy put it. It ships with its consumer's test -- lappy's
    analogue `corner_terms` carries a detailed contract and has no production consumer at all.
    """

    origin: np.ndarray
    exponent: np.ndarray


class HelmholtzFamily(Frozen, ABC):
    """A set of exact solutions of `laplacian(u) + k**2 u = 0`, as columns of a matrix.

    Subclasses set their fields with `_set` in `__init__` and implement `n_cols`, `_sample`,
    `log_scale` and `singular_exponents`.

    Columns arrive PRE-SCALED: column j is `psi_j * exp(-log_scale(k)[j])`, and the scale
    depends on `(family, column, k)` and never on the evaluation points. That is what lets a
    downstream unit-norm rescale compose without double-counting -- the log scales add -- and
    it is what lets a column at a sharp corner exist at all rather than underflowing to zeros.
    """

    def __init__(self, prec=PREC):
        _set(self, prec=check_prec(prec))

    # --- what a subclass must provide ---------------------------------------------------

    @property
    @abstractmethod
    def n_cols(self):
        """How many columns this family contributes."""

    @abstractmethod
    def _sample(self, pts, normals, bc):
        """Bind to a point set: everything k-independent, computed once."""

    @abstractmethod
    def log_scale(self, k, cols=None):
        """`sigma`: the per-column log scale already divided out of the returned columns."""

    @abstractmethod
    def singular_exponents(self):
        """A `SingularStructure` describing where each column is singular."""

    # --- the shared surface ---------------------------------------------------------------

    def __len__(self):
        return self.n_cols

    def at(self, pts):
        """Bind to points. Accepts a complex array, or a `Quadrature`, which additionally
        supplies the normals and boundary conditions that `trace` needs."""
        if hasattr(pts, "pts") and hasattr(pts, "normals") and hasattr(pts, "bc"):
            return self._sample(np.asarray(pts.pts, dtype=complex),
                                np.asarray(pts.normals, dtype=complex),
                                np.asarray(pts.bc, dtype=float))
        return self._sample(np.atleast_1d(np.asarray(pts, dtype=complex)), None, None)

    def values(self, k, pts, cols=None):
        """Sugar for `at(pts).values(k, cols)`. Builds a throwaway `Sampling`, so it is for
        a one-off and not for a loop over k -- that is what `at` is for."""
        return self.at(pts).values(k, cols)

    def grad(self, k, pts, cols=None):
        return self.at(pts).grad(k, cols)

    def deriv(self, k, pts, vecs, cols=None):
        return self.at(pts).deriv(k, vecs, cols)

    def __add__(self, other):
        from .basis import Basis

        return Basis.join(self, other)

    def __repr__(self):
        return f"{type(self).__name__}(n_cols={self.n_cols})"


class Sampling(Frozen):
    """A family bound to a fixed point set, holding every k-independent quantity.

    Immutable and reusable: the solver hands the same object hundreds of k values and
    nothing inside it changes.
    """

    @property
    @abstractmethod
    def family(self):
        """The family this came from."""

    @abstractmethod
    def _eval(self, k, cols, want_grad):
        """`(values, grad_or_None)` at `k`. One transcendental pass, both outputs."""

    @abstractmethod
    def accuracy(self, k, cols=None):
        """Per-column bound on the relative error of the returned entries."""

    def __len__(self):
        return len(self.pts)

    def values(self, k, cols=None):
        return self._eval(check_k(k), _as_cols(cols), False)[0]

    def grad(self, k, cols=None):
        return self._eval(check_k(k), _as_cols(cols), True)[1]

    def values_and_grad(self, k, cols=None):
        return self._eval(check_k(k), _as_cols(cols), True)

    def deriv(self, k, vecs, cols=None):
        """Directional derivative along complex unit `vecs`, one per point."""
        g = self.grad(k, cols)
        v = np.asarray(vecs, dtype=complex)
        return g.real * v.real[:, None] + g.imag * v.imag[:, None]

    def trace(self, k, cols=None):
        """`(1 - bc)*u + bc*du_dn` at every node -- the boundary matrix, assembled here.

        Needs the normals and boundary conditions, so it wants a `Sampling` built from a
        `Quadrature`. The Dirichlet and Neumann cases skip the half they do not need; the
        Robin and mixed cases take one pass for both, which is the reason `values_and_grad`
        exists.
        """
        if self.normals is None or self.bc is None:
            raise ValueError(
                "trace needs normals and boundary conditions; build this Sampling from a "
                "Quadrature (family.at(domain.bdry_quad(n))) rather than from bare points"
            )
        bc = self.bc
        if np.all(bc == 0.0):
            return self.values(k, cols)
        if np.all(bc == 1.0):
            return self.deriv(k, self.normals, cols)
        v, g = self.values_and_grad(k, cols)
        n = self.normals
        dn = g.real * n.real[:, None] + g.imag * n.imag[:, None]
        return (1.0 - bc)[:, None] * v + bc[:, None] * dn

    def log_scale(self, k, cols=None):
        return self.family.log_scale(check_k(k), _as_cols(cols))

    def _check_accuracy(self, rel_err):
        """Warn when the entries cannot be delivered to the family's own `prec`.

        The bound is free -- it falls out of the series loop in `_bessel` -- so this costs one
        reduction over an array the caller already paid for. Same pattern as
        `Domain.achievable_prec` two layers down, and the same rule: a reported precision is
        never a claim contradicted elsewhere in the same object.

        The message quotes the shortfall rounded to a power of ten rather than its exact
        value. That is deliberate: Python's default warning filter keys on the message text,
        so a message carrying a k-dependent float would fire on every one of the hundreds of
        evaluations in a solve instead of once.
        """
        worst = float(np.max(rel_err))
        if worst <= self.family.prec:
            return
        decade = int(np.ceil(np.log10(worst)))
        warnings.warn(
            f"{type(self.family).__name__} can deliver about 1e{decade} relative accuracy "
            f"on these points, short of the requested prec={self.family.prec:.0e}. The "
            f"limit is the Bessel kernel's own scale, not the point set: see "
            f"Sampling.accuracy(k) for the per-column bound.",
            PrecisionWarning,
            stacklevel=4,
        )

    def __repr__(self):
        return f"{type(self).__name__}({self.family!r}, n_pts={len(self)})"


def _as_cols(cols):
    """Normalize a column selection to an int index array, or None for all of them.

    Honoured as given: no sort, no dedup. The caller asked for an order.
    """
    if cols is None:
        return None
    cols = np.asarray(cols)
    if cols.dtype == bool:
        return np.nonzero(cols)[0]
    return cols.astype(int, copy=False)


def _take(a, cols):
    return a if cols is None else a[..., cols]
