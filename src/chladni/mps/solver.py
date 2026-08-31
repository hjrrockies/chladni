"""`MPSEigensolver`: the domain-dependent half, speaking `lam`.

The split from `MPSPencil` is not decoration. The pencil is a basis and two point sets, which
is `helmholtz`'s world, where the parameter is `k`; the solver is a domain and a user, where
the parameter is `lam`. So the `lam = k**2` conversion sits at the boundary between them
because that is where it falls, not because a class was built to hold it -- `conventions.md`
asks only that a parameter's NAME state its variable, and deliberately no longer names a
single seam, so that whether `k` or `lam` suits a given step stays a measurement.

Basis planning is deferred (`docs/design/particular_solutions.md`, "Still deferred"), so
`basis` is required. A default here would become load-bearing the way lappy's did, and lappy's
`default_basis_for` still raises `NotImplementedError` for any non-polyline boundary.
"""
from __future__ import annotations

import concurrent.futures as cf

import numpy as np

from .._core import Frozen, _set
from ..eigensolver import Eigensolver, lam_to_k, weyl_window
from .pencil import MPSPencil

__all__ = ["MPSEigensolver"]

#: How many eigenvalues the default reference window is sized for, when `lam_ref` is not
#: given. Only sets where the rank is chosen, so it wants to be a typical scan rather than a
#: precise one.
_REF_MODES = 10


class MPSEigensolver(Eigensolver, Frozen):
    """The Method of Particular Solutions on a `Domain`.

    Collocation may be FIXED or BUDGETED: pass `bdry_quad`/`int_pts` to say exactly where the
    points go, or `n_bdry`/`n_int` to have them built from the domain. Both reach the same
    pencil; the first is for bespoke work and the second is what a planner will eventually
    fill in.

    `rng` seeds the interior points. It is not optional in spirit: lappy's unseeded default
    made `iso_right_tri` span 2.5 to 5.8 certified digits across draws, and a solver whose
    answer depends on an unnamed global is not reproducible.
    """

    def __init__(self, domain, *, basis, bdry_quad=None, int_pts=None, n_bdry=None,
                 n_int=None, reg="svd", rank=None, rtol=None, lam_ref=None, prec=None,
                 rng=None):
        super().__init__(domain, prec=prec)

        if bdry_quad is None:
            if n_bdry is None:
                raise ValueError("pass either bdry_quad or n_bdry")
            bdry_quad = domain.bdry_quad(n_bdry)
        if int_pts is None:
            if n_int is None:
                raise ValueError("pass either int_pts or n_int")
            int_pts = domain.interior_pts(n_int, rng=np.random.default_rng(rng))

        if lam_ref is None and rank is None:
            # The pencil refuses to guess a reference, and is right to: it has no geometry to
            # guess from. A solver does, so it may -- the midpoint of a typical scan window.
            a, b = weyl_window(domain, _REF_MODES)
            lam_ref = float(np.sqrt(a * b))

        _set(self, _pencil=MPSPencil.from_basis(
            basis, bdry_quad, int_pts, reg=reg, rank=rank, rtol=rtol,
            k_ref=None if lam_ref is None else lam_to_k(lam_ref), prec=self.prec))
        _set(self, lam_ref=lam_ref)

    # --- delegation, and the one conversion ------------------------------------------------

    @property
    def pencil(self):
        return self._pencil

    basis = property(lambda self: self._pencil.basis)
    n_cols = property(lambda self: self._pencil.n_cols)
    n_bdry = property(lambda self: self._pencil.n_bdry)
    n_int = property(lambda self: self._pencil.n_int)
    rank = property(lambda self: self._pencil.rank)
    rtol = property(lambda self: self._pencil.rtol)
    reg = property(lambda self: self._pencil.reg)

    def tensions(self, lam):
        """The ascending tension vector at `lam`, or one row per entry of an array `lam`.

        Arrays are evaluated on a thread pool: the work is LAPACK, which releases the GIL,
        and `douse` runs whole solves across threads on top of this.
        """
        arr = np.asarray(lam, dtype=float)
        if arr.ndim == 0:
            return self._pencil.tensions(lam_to_k(float(arr)))
        ks = lam_to_k(arr)
        with cf.ThreadPoolExecutor() as ex:
            return np.array(list(ex.map(self._pencil.tensions, ks)))

    def tension(self, lam):
        """The smallest generalized singular value at `lam`: the quantity to be minimized."""
        arr = np.asarray(lam, dtype=float)
        if arr.ndim == 0:
            return self._pencil.tension(lam_to_k(float(arr)))
        return self.tensions(arr)[:, 0]

    def coef(self, lam, mult=1):
        """The `mult` minimizing coefficient vectors at `lam`, in the scaled column space."""
        return self._pencil.coef(lam_to_k(lam), mult)

    def log_scale(self, lam):
        """Total per-column log scale at `lam`; the recovery factor for `coef`."""
        return self._pencil.log_scale(lam_to_k(lam))

    def rank_diagnostics(self, lam):
        return self._pencil.rank_diagnostics(lam_to_k(lam))

    # --- the deferred layer ------------------------------------------------------------------

    def solve_interval(self, a, b):
        raise NotImplementedError(
            "the eigenvalue search is not built yet: this layer assembles the collocation "
            "matrices, regularizes the pencil, calls the GSVD and reports the tension, and "
            "stops there. Minimizing tension(lam) over an interval and estimating "
            "multiplicities is the next layer -- see docs/design/mps.md. Until then, scan "
            "tension(lam) yourself."
        )

    def __repr__(self):
        return (f"MPSEigensolver({self.domain!r}, n_cols={self.n_cols}, "
                f"n_bdry={self.n_bdry}, n_int={self.n_int}, rank={self.rank})")
