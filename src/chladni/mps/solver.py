"""`MPSEigensolver`: the domain-dependent half, speaking `lam`.

This is a facade over `MPSPencil`, by composition rather than inheritance: it HOLDS one as
`_pencil` and delegates. What it adds is the domain -- and therefore Weyl windows, boundary
conditions, and a place for the eventual basis planner to live.

**Both speak `lam`, and so does everything else in `mps`.** An earlier version had the pencil
take `k` while this took `lam`, which put the conversion seam INSIDE one layer: two classes a
user descends between -- `solver.pencil` is a documented property -- with the variable
changing across the hop and nothing about the numbers changing to justify it. The seam now
sits at `mps`/`helmholtz`, which is where the meaning actually changes: a Fourier-Bessel
column IS a function of `k*r`, and a tension is not such a thing.

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

        # No lam_ref by default: the rank is re-derived per lam (threshold), which is the
        # only rule that cannot pick an unsafe rank. Pass lam_ref=(lo, hi) to freeze it.

        _set(self, _pencil=MPSPencil.from_basis(
            basis, bdry_quad, int_pts, reg=reg, rank=rank, rtol=rtol,
            lam_ref=lam_ref, prec=self.prec))
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
    is_frozen = property(lambda self: self._pencil.is_frozen)
    rtol = property(lambda self: self._pencil.rtol)
    reg = property(lambda self: self._pencil.reg)

    def tensions(self, lam):
        """The ascending tension vector at `lam`, or one row per entry of an array `lam`.

        Arrays are evaluated on a thread pool: the work is LAPACK, which releases the GIL,
        and `douse` runs whole solves across threads on top of this.

        Under the threshold rank the vector's LENGTH varies with `lam`, since the pencil's
        numerical rank grows with `lam`. Rows are padded to a common width with `+inf`, which
        is the honest value: a direction the pencil does not resolve has no finite tension,
        and `+inf` sorts to the end exactly as the zero-denominator case already does.
        """
        arr = np.asarray(lam, dtype=float)
        if arr.ndim == 0:
            return self._pencil.tensions(float(arr))
        with cf.ThreadPoolExecutor() as ex:
            rows = list(ex.map(self._pencil.tensions, arr))
        w = max(len(r) for r in rows)
        out = np.full((len(rows), w), np.inf)
        for i, r in enumerate(rows):
            out[i, :len(r)] = r
        return out

    def tension(self, lam):
        """The smallest generalized singular value at `lam`: the quantity to be minimized."""
        arr = np.asarray(lam, dtype=float)
        if arr.ndim == 0:
            return self._pencil.tension(float(arr))
        return self.tensions(arr)[:, 0]

    def coef(self, lam, mult=1):
        """The `mult` minimizing coefficient vectors at `lam`, in the scaled column space."""
        return self._pencil.coef(lam, mult)

    def log_scale(self, lam):
        """Total per-column log scale at `lam`; the recovery factor for `coef`."""
        return self._pencil.log_scale(lam)

    def rank_diagnostics(self, lam):
        return self._pencil.rank_diagnostics(lam)

    # --- the deferred layer ------------------------------------------------------------------

    def findability(self, a, b, *, dk, calibrate=False, n=None):
        """Can a scan of step `dk` SEE every dip in `[a, b]`? Cached per window and step.

        The check `chladni.mps.calibrate` exists for, exposed where a caller has both a domain
        and a window. It answers the question the tension at a located eigenvalue cannot: not
        "how well is the basis resolving what I found" but "could I have found it at all".

        Costs about 40 evaluations and is worth them. The failure it guards against returned a
        NON-CONSECUTIVE spectrum -- every value accurate, three levels missing -- with an
        excellent tension at every value returned, no exception and no warning, and a Weyl
        deficit that read the wrong sign. `calibrate=True` additionally tries to raise the
        plateau by loosening the rank truncation; see `calibrate.calibrate_rtol` for why that
        lever is measured INERT on `chladni` pencils and the scan step is the real remedy.
        """
        from . import calibrate as _cal

        key = (float(a), float(b), float(dk), bool(calibrate))
        cache = getattr(self, "_findability", None)
        if cache is None:
            _set(self, _findability={})
            cache = self._findability
        if key in cache:
            return cache[key]

        kw = {} if n is None else {"n": n}
        if calibrate:
            cal = _cal.calibrate_rtol(self._pencil, self.domain, a, b, dk=dk, **kw)
        else:
            sb = _cal.sigma_base(self._pencil, a, b, **kw)
            target = _cal.findability_target(self.domain, dk)
            cal = _cal.RtolCalibration(
                rtol=float(self.rtol), sigma_base=sb, target=target, dk=float(dk),
                achieved=bool(sb >= target), calibrated=False,
                required_dk=_cal.required_dk(self.domain, sb),
                evaluations=int(kw.get("n", _cal._N_PROBE)),
                history=((float(self.rtol), sb, float("nan")),))
        _cal.warn_if_unfindable(cal, span_k=lam_to_k(b) - lam_to_k(a))
        cache[key] = cal
        return cal

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
                f"n_bdry={self.n_bdry}, n_int={self.n_int}, "
                f"rank={self.rank if self.rank is not None else 'threshold'})")
