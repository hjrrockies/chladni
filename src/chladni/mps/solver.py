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

import warnings

from .._core import Frozen, PrecisionWarning, _set, check_prec
from ..eigensolver import Eigensolver, Spectrum, lam_to_k, weyl_window
from .pencil import MPSPencil

__all__ = ["MPSEigensolver", "DEFAULT_RNG"]

#: How many eigenvalues the default reference window is sized for, when `lam_ref` is not
#: given. Only sets where the rank is chosen, so it wants to be a typical scan rather than a
#: precise one.
_REF_MODES = 10

#: Seed for the interior fill when the caller names none. A FIXED CONSTANT, not `None`.
#:
#: `np.random.default_rng(None)` draws fresh OS entropy, so the default made the located
#: eigenvalue depend on an unnamed global. Measured on the isolated (1,1) mode of `rect(1.01, 1)`
#: over 12 seeds (`notes/landscape_findings.md` section 17): relative error median 2.91e-10, min
#: 1.79e-11, max 1.97e-09 -- a **110x run-to-run spread**, against 1.38e-12 for a deterministic
#: grid fill. That spread goes into a `douse` shape objective as noise, and a shape optimization
#: cannot descend on an objective that moves when nothing moved.
#:
#: The value is arbitrary and must stay arbitrary-looking: `calibrate.py` records that changing
#: only the seed moved `sigma_base` by 14x at fixed basis and that "the DEFAULT seed was the lucky
#: one on the shape that happened to solve", so a seed chosen because it scored well would be
#: tuning. 0 is chosen because every suite in `tests/` already passes 0 explicitly.
#:
#: This fixes DETERMINISM and nothing else. Section 17's own recommendation is to switch the fill
#: to a deterministic low-discrepancy set, which is a different change: it was measured on one
#: domain, and section 17 names the L-shape's thin-protrusion rejection case as the test to run
#: before the METHOD changes. `method="random"` therefore stands.
DEFAULT_RNG = 0


class MPSEigensolver(Eigensolver, Frozen):
    """The Method of Particular Solutions on a `Domain`.

    Collocation may be FIXED or BUDGETED: pass `bdry_quad`/`int_pts` to say exactly where the
    points go, or `n_bdry`/`n_int` to have them built from the domain. Both reach the same
    pencil; the first is for bespoke work and the second is what a planner will eventually
    fill in.

    `rng` seeds the interior points, and defaults to `DEFAULT_RNG` rather than to `None`. It was
    not optional in spirit and is no longer optional in fact: lappy's unseeded default made
    `iso_right_tri` span 2.5 to 5.8 certified digits across draws, here it spread the located
    eigenvalue 110x, and a solver whose answer depends on an unnamed global is not reproducible.
    Pass `rng=` to choose a draw; pass a `Generator` to supply one.
    """

    def __init__(self, domain, *, basis, bdry_quad=None, int_pts=None, n_bdry=None,
                 n_int=None, reg="auto", rank=None, rtol=None, lam_ref=None, prec=None,
                 rng=DEFAULT_RNG, certify=True):
        super().__init__(domain, prec=prec)
        # `certify`: attach a Moler-Payne bound to every level `search_interval` returns
        # (`certify_levels`). On by default -- the bound is what makes `Spectrum.bound` honest
        # under the penalty, whose floor overstates the error by two to three orders.
        _set(self, certify=bool(certify))

        if bdry_quad is None:
            if n_bdry is None:
                raise ValueError("pass either bdry_quad or n_bdry")
            bdry_quad = domain.bdry_quad(n_bdry)
        if int_pts is None:
            if n_int is None:
                raise ValueError("pass either int_pts or n_int")
            int_pts = domain.interior_pts(n_int, rng=np.random.default_rng(rng))

        # `reg="auto"` by default: the penalty at `prec >= 1e-8`, the per-lam threshold below
        # (`MPSPencil`). Passing rtol=, rank= or lam_ref= asks for a truncation; lam_ref=(lo, hi)
        # freezes its rank.

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

    # --- the search ----------------------------------------------------------------------

    def objective(self, a, b, *, prec=None):
        """The counted, memoized tension function on `[a, b]`, in `lam`. Cached per window.

        Public because the evaluation cache is the expensive thing here: a caller can run one
        search, inspect what it paid for, and drive another stage against the same samples
        instead of buying them twice. Stage 3 depends on exactly that -- it reads multiplicity
        out of this cache and spends nothing of its own.

        **`lam`, not `k`.** `bracket_interval` covers in `k` internally, wrapping a non-`k`
        objective in its own view and converting brackets back to the variable it was handed,
        so handing it `lam` gives `lam` brackets out and the refiner gets the variable it
        wants. A second conversion here would be the footgun `bracket.py` records.

        Weyl reaches the searcher through `expected_count`, for SIZING only.
        """
        from .. import asymp
        from .objective import Objective

        prec = self.prec if prec is None else check_prec(prec)
        key = (float(a), float(b), float(prec))
        cache = getattr(self, "_objectives", None)
        if cache is None:
            _set(self, _objectives={})
            cache = self._objectives
        if key not in cache:
            cache[key] = Objective.from_pencil(
                self._pencil, float(a), float(b), variable="lam", prec=prec,
                expected_count=lambda lo, hi: asymp.expected_count(lo, hi, self.domain))
        return cache[key]

    def search_interval(self, a, b, *, prec=None, objective=None):
        """Every level in `[a, b]` as a `SolveResult`: the MPS-rich answer, diagnostics and all.

        This is the seam to reach for. `solve_interval` narrows what comes back to the
        method-agnostic `Spectrum` that `Eigensolver` contracts for, which necessarily drops
        the tension, the brackets and the gap ratios; here they survive.

        A decline comes back as a VALUE with `diagnostics.declined` set, per
        `docs/design/candidates.md` section 5. `solve_interval` is where it becomes an error,
        because that is where an empty answer would otherwise be indistinguishable from one.
        """
        from .search import solve

        prec = self.prec if prec is None else check_prec(prec)
        obj = self.objective(a, b, prec=prec) if objective is None else objective
        res = solve(obj, prec=prec)
        if not self.certify or not res.levels:
            return res
        return self.certify_levels(res, lam_max=float(b), prec=prec)

    def certify_levels(self, res, *, lam_max, prec):
        """`res` with a Moler-Payne bound on every level (`chladni.integrals.certify`), and its
        precision summary recomputed from the tighter of each level's two bars.

        Per level: the `mult` minimizing vectors at the level's `lam`, their Cauchy data on an
        integral rule sized for `lam_max` and `prec`, a Rellich norm, and the bound per vector;
        the level carries the WORST of them. Columns are certified separately, as lappy did --
        whether a cluster's subspace admits a sharper bound is an open question
        (`docs/design/boundary_integrals.md`). Off a Dirichlet boundary the bound does not
        apply, and each level says so in `bound_reason` and keeps `delta_res`.
        """
        from dataclasses import replace

        from ..integrals import cauchy, certify, integral_quad
        from .search import _precision_summary

        p = self._pencil
        if np.any(np.asarray(p.bdry.bc) != 0.0):
            why = "Moler-Payne needs a Dirichlet condition on the whole boundary"
            levels = tuple(replace(L, bound_reason=why) for L in res.levels)
        else:
            key = (float(lam_max), float(prec))
            cache = getattr(self, "_integral_quads", None)
            if cache is None:
                _set(self, _integral_quads={})
                cache = self._integral_quads
            if key not in cache:
                cache[key] = integral_quad(self.domain, lam_max, prec=prec)
            iq = cache[key]
            origins = certify.x0_probes(self.domain, iq.x0)
            levels = []
            for L in res.levels:
                fc = p.family_coef(L.lam, p.coef(L.lam, int(L.mult)))
                cd = cauchy.cauchy_data(p.basis, fc, L.lam, iq.quad)
                k = lam_to_k(L.lam)

                def evaluate(z, fc=fc, k=k):
                    V = p.basis.values(k, z)
                    return V @ fc, np.abs(V) @ np.abs(fc)

                mp = certify.moler_payne(cd, evaluate, self.domain, origins)
                levels.append(replace(L, mp_bound=float(np.max(mp.abs)),
                                      mp_eps=float(np.max(mp.eps)),
                                      cancellation=float(np.max(mp.cancellation)),
                                      bound_reason=mp.reason))
            levels = tuple(levels)
        unmet, attained = _precision_summary(levels, prec)
        return replace(res, levels=levels,
                       diagnostics=replace(res.diagnostics, unmet=unmet,
                                           attained_prec=attained))

    def solve_interval(self, a, b):
        """The `Spectrum` of `[a, b]` -- `Eigensolver`'s contract, over `search_interval`.

        Two things are reported rather than swallowed, and they are different in kind.

        A **decline** raises. It must not come back as an empty `Spectrum`, because an empty
        `Spectrum` already means "there are none in this interval" -- and `Eigensolver.solve`
        answers a short count by widening and asking again eight times before blaming the
        spectrum, which is precisely the accurate-but-incomplete failure this layer exists to
        prevent.

        A **precision shortfall** warns. The values are right to the bar they attained; it is
        the pencil that binds, not the caller's request, so the numbers are still returned and
        the bar is carried on the `Spectrum` for the caller to check.

        An **ambiguous grouping** warns too. `search.md` section 2.4a stage 3 guarantees the
        TOTAL over a clear window even where the grouping inside it is arbitrary, so the count
        is right and the split may be coarser than `prec` asked for. That is the in-scope half
        of Phase D; deciding the grouping needs the interior Gram angle and is not built.
        """
        res = self.search_interval(a, b)
        d = res.diagnostics
        if d.n_uncertain:
            # An uncertain COUNT raises, where a precision shortfall only warns: the value is
            # located to `prec`, what is in doubt is how many eigenvalues sit at it, and an array
            # with multiplicity repeated cannot say "and possibly one more here". A short array
            # is worse than an exception, because `Eigensolver.solve` widens and refills.
            lam0, mult0, alt0, why0 = d.uncertain[0]
            raise RuntimeError(
                f"the multiplicity of {d.n_uncertain} level(s) in [{float(a):.6g}, "
                f"{float(b):.6g}] could not be certified. First at lam = {lam0:.10g}: read "
                f"{mult0}, alternative reading {alt0} -- {why0}. Ways on: call search_interval "
                f"to inspect the levels; move prec (tighter resolves the cluster into separate "
                f"levels, looser merges it); or improve the basis.")
        if d.declined:
            raise RuntimeError(
                f"the search declined on lam in [{float(a):.6g}, {float(b):.6g}]: "
                f"{d.decline_reason}. It found {len(res.levels)} level(s) before stopping; "
                f"call search_interval to inspect them."
            )
        if d.n_ambiguous:
            warnings.warn(
                f"{d.n_ambiguous} level(s) in [{float(a):.6g}, {float(b):.6g}] have an "
                f"ambiguous grouping and were collapsed. The TOTAL count is right; the split "
                f"between them is coarser than prec={self.prec:.0e} asked for, because "
                f"deciding it needs the interior Gram angle (search.md Phase D).",
                PrecisionWarning, stacklevel=2,
            )
        if len(d.unmet):
            warnings.warn(
                f"{len(d.unmet)} level(s) in [{float(a):.6g}, {float(b):.6g}] reached only "
                f"{d.attained_prec:.1e} relative, short of the requested "
                f"{self.prec:.0e}. The values are correct to the bar reported on the "
                f"Spectrum; the pencil is what binds here, not the request.",
                PrecisionWarning, stacklevel=2,
            )
        lam = np.array([x.lam for x in res.levels], dtype=float)
        mult = np.array([x.mult for x in res.levels], dtype=int)
        bound = np.array([x.bound for x in res.levels], dtype=float)
        order = np.argsort(lam)
        return Spectrum(lam[order], mult[order], bound[order], prec=self.prec)

    def __repr__(self):
        return (f"MPSEigensolver({self.domain!r}, n_cols={self.n_cols}, "
                f"n_bdry={self.n_bdry}, n_int={self.n_int}, "
                f"rank={self.rank if self.rank is not None else 'threshold'})")
