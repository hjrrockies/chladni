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

`basis` is required here: this is the hand-built path. The black-box path is
`chladni.eigenproblem.Eigenproblem`, which plans a basis (`chladni.plan`) and builds this solver
from it. Keeping the planner out of this class means no default basis becomes load-bearing the way
lappy's `default_basis_for` did.
"""
from __future__ import annotations

import concurrent.futures as cf
from dataclasses import dataclass

import numpy as np

import warnings

from .._core import Frozen, PrecisionShortfall, PrecisionWarning, _set, check_prec
from ..eigensolver import Eigensolver, Spectrum, lam_to_k, weyl_window
from .pencil import MPSPencil

__all__ = ["MPSEigensolver", "Eigenfunctions", "DEFAULT_RNG"]


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

#: `delta_res` is admitted into `Level.bound` only when Moler-Payne roughly agrees:
#: `mp_bound <= MP_CROSSCHECK * prec * lam`, besides `delta_res <= prec * lam`. `delta_res`
#: measures the WELL, and a basis can put a clean, deep well in the wrong place: T2.8-N5s2 on a
#: Fourier-Bessel-only basis sat 7e-10 off at margin R = 56, under-covered at any admission
#: margin, while Moler-Payne covered it (`notes/admission_bound.md`). On the douse benchmark,
#: 20 removed that met-but-wrong level and the Linux edge case (MP 67000x the request), at a cost
#: of 2-5 met levels per ~1200; honest levels have MP within ~2-5x of the request.
MP_CROSSCHECK = 20.0

#: A tracked window wider than this many local mean Weyl spacings is searched, not refined: it
#: may hold more than one well, and the refiner answers "no well" on such a window at the cost
#: of its evaluations plus a cold solve (`track_levels`).
WIDE_WINDOW = 1.0


@dataclass(frozen=True)
class Eigenfunctions:
    """A level's eigenfunctions: `coef`, `(n_cols, mult)` in the FAMILY's coordinates, so
    `basis.values(lam_to_k(lam), pts) @ coef` evaluates them anywhere; L2(Omega)-orthonormal
    when `orthonormal`. `rule` is the `IntegralQuad` the Gram was formed on -- the node set a
    Hadamard derivative will want -- and `gram_residual` the orthonormality defect at a second
    Rellich origin."""

    lam: float
    coef: np.ndarray
    rule: object
    orthonormal: bool
    gram_residual: float
    reason: str = ""


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

        # `reg="auto"` by default: the per-lam threshold at every `prec` (`MPSPencil`); the
        # penalty is `reg="tikhonov_qr"`, explicitly. Passing rtol=, rank= or lam_ref= asks for a truncation; lam_ref=(lo, hi)
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

    def search_interval(self, a, b, *, prec=None, objective=None, on_shortfall="flag"):
        """Every level in `[a, b]` as a `SolveResult`: the MPS-rich answer, diagnostics and all.

        This is the seam to reach for. `solve_interval` narrows what comes back to the
        method-agnostic `Spectrum` that `Eigensolver` contracts for, which necessarily drops
        the tension, the brackets and the gap ratios; here they survive.

        A decline comes back as a VALUE with `diagnostics.declined` set, per
        `docs/design/candidates.md` section 5. `solve_interval` is where it becomes an error,
        because that is where an empty answer would otherwise be indistinguishable from one.

        `on_shortfall="raise"` raises `PrecisionShortfall` instead of returning a result that
        fell short. A level that misses `prec` raises EARLY, the moment it is refined, with that
        level, its `margin` and the values located so far -- a median 57% of a failing solve's
        wall time saved on the douse benchmark (`notes/early_shortfall.md`). A count audit of
        `short` or `excess`, an uncertified count, or a decline is known only at the end and
        raises after the solve, with the whole result and the audit. `"flag"` (the default)
        reports the same things on the result and returns it.
        """
        from .search import solve

        if on_shortfall not in ("flag", "raise"):
            raise ValueError(f"on_shortfall must be 'flag' or 'raise', got {on_shortfall!r}")
        prec = self.prec if prec is None else check_prec(prec)
        obj = self.objective(a, b, prec=prec) if objective is None else objective
        res = solve(obj, prec=prec, abort_unmet=(on_shortfall == "raise"))
        if self.certify and res.levels:
            res = self.certify_levels(res, lam_max=float(b), prec=prec)
        res = self.audit_count(res, a, b)
        if on_shortfall == "raise":
            _raise_on_shortfall(res, a, b, prec)
        return res

    def track_levels(self, prev, h, *, mults=None, prec=None):
        """Re-locate known levels in small windows around their previous values: `(result,
        None)`, or `(None, why)` when a window does not hold what it should.

        THE CALLER SETS THE WINDOWS. `prev` are the previous level values (ascending), `mults`
        their multiplicities, `h` the half-width each may have moved by -- which a shape
        optimizer knows before it steps, from the level's own shape derivative and its step
        (douse: `h_i = c |g_i . d|`). A window with one previous level is REFINED, not searched:
        the bracket refiner locates the well (`mps.refine`, ~15-20 evaluations), and the
        multiplicity is the previous one, verified on the tension vector there -- an m-fold
        level has m small tensions and a gap of `multiplicity.GAP_RATIO` above them. Windows that
        overlap are merged and SEARCHED (`search_interval`): that is a cluster, and the search's
        multiplicity machinery decides it. Measured need (`notes/warm_trace_2026-10-14.md`): the
        prefix warm solve spent ~90 evaluations per solve re-covering the whole range below
        `lam_n`; 65-70% of a douse run.

        Completeness is the caller's: nothing between the windows is looked at. The return is a
        `SolveResult` like `search_interval`'s (certified when `certify`), with the count audit
        marked unaudited.
        """
        from dataclasses import replace as _replace

        from .. import asymp
        from .multiplicity import GAP_RATIO
        from .refine import refine
        from .search import Level, SolveDiagnostics, SolveResult

        prec = self.prec if prec is None else check_prec(prec)
        prev = np.atleast_1d(np.asarray(prev, dtype=float))
        h = np.broadcast_to(np.asarray(h, dtype=float), prev.shape)
        mults = (np.ones(prev.shape, dtype=int) if mults is None
                 else np.broadcast_to(np.asarray(mults, dtype=int), prev.shape))
        if prev.size == 0:
            return SolveResult(), None
        if np.any(h <= 0) or np.any(prev - h <= 0):
            return None, "windows: every half-width must be positive and below its value"
        # merge overlapping windows, in ascending order
        windows = []
        for i in np.argsort(prev):
            lo, hi = float(prev[i] - h[i]), float(prev[i] + h[i])
            if windows and lo <= windows[-1][1]:
                windows[-1] = (windows[-1][0], max(windows[-1][1], hi), windows[-1][2] + [int(i)])
            else:
                windows.append((lo, hi, [int(i)]))
        levels, n_evals, refine_evals = [], 0, 0
        for lo, hi, members in windows:
            # A window holding several previous levels, or one wide enough to hold several
            # wells (wider than `WIDE_WINDOW` x the local mean Weyl spacing: an early, large
            # optimizer step), is SEARCHED; the refiner is the tool for one well (measured: ten
            # of 28 cold fallbacks over three seeds were refines on wide windows that found "no
            # well", `notes/warm_trace_2026-10-14.md`).
            if len(members) > 1 or (hi - lo) > WIDE_WINDOW * self._mean_spacing(0.5 * (lo + hi)):
                res_w = self.search_interval(lo, hi, prec=prec)
                if res_w.declined:
                    return None, f"search: the merged window [{lo:.6g}, {hi:.6g}] declined"
                levels.extend(res_w.levels)
                n_evals += res_w.diagnostics.evaluations
                continue
            i = members[0]
            obj = self.objective(lo, hi, prec=prec)
            r = refine(obj, lo, hi, prec=prec, interior=(lo, hi))
            refine_evals += r.evaluations
            n_evals += r.evaluations
            if not r.accepted:
                return None, (f"refine: no well in [{lo:.6g}, {hi:.6g}] around {prev[i]:.10g} "
                              f"({r.reason})")
            m = int(mults[i])
            t = np.asarray(obj.tensions(r.lam), dtype=float)
            if t.size < m or (m > 1 and t[m - 1] > GAP_RATIO * t[0]):
                return None, (f"multiplicity: {prev[i]:.10g} was {m}-fold and the tensions at "
                              f"{r.lam:.10g} no longer show {m} small values")
            if t.size > m and t[m] < GAP_RATIO * t[m - 1]:
                return None, (f"multiplicity: a neighbour has joined {prev[i]:.10g} "
                              f"(sigma_{m + 1}/sigma_{m} = {t[m] / t[m - 1]:.2g})")
            levels.append(Level(lam=float(r.lam), mult=m, sigma=float(r.sigma),
                                delta_res=float(r.delta_res), bracket=(lo, hi), prec=prec,
                                count_decided_by="tracked"))
        levels = tuple(sorted(levels, key=lambda L: L.lam))
        audit = asymp.CountAudit("unaudited", reason="tracked windows are not a prefix")
        res = SolveResult(levels=levels,
                          diagnostics=SolveDiagnostics(refine_evals=refine_evals,
                                                       evaluations=n_evals,
                                                       prec_requested=prec, count_audit=audit))
        if self.certify and res.levels:
            res = self.certify_levels(res, lam_max=float(windows[-1][1]), prec=prec)
            res = _replace(res, diagnostics=_replace(res.diagnostics, count_audit=audit))
        return res, None

    def _mean_spacing(self, lam):
        """The local mean Weyl spacing at `lam` (the gap between consecutive Weyl estimates)."""
        from .. import asymp
        k = max(float(asymp.weyl_count_poly(lam, self.domain)), 0.5)
        return float(asymp.weyl_est_poly(k + 1, self.domain) - asymp.weyl_est_poly(k, self.domain))

    def audit_count(self, res, a, b):
        """`res` with `diagnostics.count_audit`: the returned count against Weyl's law
        (`asymp.count_audit`). The one check for a DROPPED eigenvalue, which no level can flag.

        Free -- it reads values already in hand -- and attached to an EMPTY result too, which is
        the case it catches most plainly: a basis that found nothing in `[a, b]`. `b` is a coarse
        top only (an arbitrary `b` need not sit in a gap); a caller that knows its window ends
        mid-gap calls `asymp.count_audit` with `top_is_gap=True`. Non-prefix windows and
        non-Dirichlet boundaries come back `"unaudited"`."""
        from dataclasses import replace

        from ..asymp import count_audit

        audit = count_audit(res.unfolded, self.domain, lo=float(a), top=float(b))
        return replace(res, diagnostics=replace(res.diagnostics, count_audit=audit))

    def certify_levels(self, res, *, lam_max, prec):
        """`res` with a Moler-Payne bound on every level (`chladni.integrals.certify`), and its
        precision summary recomputed from each level's reported bar (`Level.bound`: the tighter
        of the two, except that a `delta_res` short of `prec` is not admitted).

        Per level: the `mult` minimizing vectors at the level's `lam`, their Cauchy data on an
        integral rule sized for `lam_max` and `prec`, a Rellich norm, and the bound per vector;
        the level carries the WORST of them. Columns are certified separately, as lappy did --
        whether a cluster's subspace admits a sharper bound is an open question
        (`docs/design/boundary_integrals.md`). Off a Dirichlet boundary the bound does not
        apply, and each level says so in `bound_reason` and keeps `delta_res`.
        """
        from dataclasses import replace

        from ..integrals import cauchy, certify
        from .search import _precision_summary

        p = self._pencil
        if np.any(np.asarray(p.bdry.bc) != 0.0):
            why = "Moler-Payne needs a Dirichlet condition on the whole boundary"
            levels = tuple(replace(L, bound_reason=why) for L in res.levels)
        else:
            iq = self._integral_quad(lam_max, prec)
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
                                      bound_reason=mp.reason,
                                      delta_res_admitted=bool(
                                          L.delta_res <= prec * abs(L.lam)
                                          and not (np.max(mp.abs)
                                                   > MP_CROSSCHECK * prec * abs(L.lam)))))
            levels = tuple(levels)
        unmet, attained = _precision_summary(levels, prec)
        return replace(res, levels=levels,
                       diagnostics=replace(res.diagnostics, unmet=unmet,
                                           attained_prec=attained))

    def _integral_quad(self, lam_max, prec):
        """The integral rule for `lam` up to `lam_max` at `prec`, built once per pair. A rule is
        a function of geometry, `lam_max` and `prec` alone, so it serves every level of a solve."""
        from ..integrals import integral_quad

        key = (float(lam_max), float(prec))
        cache = getattr(self, "_integral_quads", None)
        if cache is None:
            _set(self, _integral_quads={})
            cache = self._integral_quads
        if key not in cache:
            cache[key] = integral_quad(self.domain, lam_max, prec=prec)
        return cache[key]

    def eigenfunctions(self, level, *, prec=None):
        """The level's eigenfunctions, L2(Omega)-orthonormal, as `Eigenfunctions`.

        The `mult` minimizing vectors at the level's ONE `lam` -- a merged level's members are
        never combined across different `lam`, where a Gram would need Green's identity and divide
        by their difference -- are put in the family's coordinates, their Cauchy data evaluated
        on an integral rule (evaluate first, sandwich never), their Gram formed from the full
        Rellich identity, and the columns Löwdin-orthonormalized. Only the span is promised, not
        which orthonormal basis of it.

        A DEFICIENT Gram -- columns nearly dependent, so the multiplicity is likely wrong -- is
        not orthonormalized: the columns come back normalized one by one, `orthonormal` is False
        and `reason` says why. `gram_residual` is `max |G - I|` recomputed at a SECOND Rellich
        origin, the check a caller can make without an interior oracle.
        """
        from ..integrals import cauchy, certify, identities

        prec = self.prec if prec is None else check_prec(prec)
        lam, m = float(level.lam), int(level.mult)
        p = self._pencil
        fc = p.family_coef(lam, p.coef(lam, m))
        iq = self._integral_quad(lam, prec)
        cd = cauchy.cauchy_data(p.basis, fc, lam, iq.quad)
        G = identities.gram(cd, iq.x0)
        try:
            out, ok, why = fc @ identities.lowdin(G), True, ""
        except ValueError as err:
            out, ok, why = fc / np.sqrt(np.abs(np.diag(G))), False, str(err)
        check = certify.x0_probes(self.domain, iq.x0, n=1)[1]
        G2 = identities.gram(cauchy.cauchy_data(p.basis, out, lam, iq.quad), check)
        resid = float(np.abs(G2 - np.eye(m)).max())
        return Eigenfunctions(lam=lam, coef=out, rule=iq, orthonormal=ok, gram_residual=resid,
                              reason=why)

    def hadamard(self, level, V, *, prec=None):
        """The Hadamard shape derivative of a Dirichlet level, or of several: `(n_fields, m, m)`.

        `level` may be a SEQUENCE of levels: a nearly degenerate cluster the solver resolved into
        distinct levels. The block then spans all their members, cross terms included -- the
        model a cluster optimizer needs on its way into a degeneracy. Each level is
        orthonormalized on its own, and members of different levels are orthogonal as
        eigenfunctions of different eigenvalues.

        `H[f, i, j] = - oint d_n u_i d_n u_j V_f ds` over the level's eigenfunctions,
        L2-orthonormalized (`eigenfunctions`); its eigenvalues are the derivatives of the level's
        members under the normal velocity `V_f`. `V(curve, t) -> (n_fields, len(t))`, with `curve`
        the boundary curve's index (segments in `domain.boundaries` order, flattened) and `t` in
        [0, 1] its parameter, so `V` may be discontinuous at a corner, as a moving vertex's normal
        velocity is (`docs/design/douse_requirements.md`, "Hadamard integrals"). Integrated on the
        `'integer'` rule (`integrals.integral_quad`), whose corner panels are fitted to the
        velocity's integer powers of `r` on top of the corner family. Dirichlet only.
        """
        from ..integrals import cauchy, identities, integral_quad

        if self.domain.bc_type != "dir":
            raise NotImplementedError("Hadamard derivatives are built for Dirichlet boundaries "
                                      "only (the Neumann and mixed formulas differ)")
        prec = self.prec if prec is None else check_prec(prec)
        levels = [level] if hasattr(level, "lam") else list(level)
        efs = []
        for L in levels:
            ef = self.eigenfunctions(L, prec=prec)
            if not ef.orthonormal:
                raise ValueError(f"the level at lam = {float(L.lam):.10g} could not be "
                                 f"orthonormalized ({ef.reason}); its Hadamard matrix would mix "
                                 f"a wrong count")
            efs.append(ef)
        lam = max(float(L.lam) for L in levels)       # the rule is sized for the largest
        key = ("hadamard", lam, float(prec))
        cache = getattr(self, "_integral_quads", None)
        if cache is None:
            _set(self, _integral_quads={})
            cache = self._integral_quads
        if key not in cache:
            cache[key] = integral_quad(self.domain, lam, prec=prec, family="integer")
        q = cache[key].quad
        cds = [cauchy.cauchy_data(self._pencil.basis, ef.coef, float(L.lam), q)
               for L, ef in zip(levels, efs)]
        cd = cds[0] if len(cds) == 1 else cauchy.CauchyData(
            q, lam, np.hstack([c.u for c in cds]), np.hstack([c.u_n for c in cds]),
            np.hstack([c.u_t for c in cds]))
        offsets = np.cumsum([0] + [len(b.segments) for b in self.domain.boundaries])
        curve = offsets[np.asarray(q.bdry_idx, int)] + np.asarray(q.seg_idx, int)
        vals = None
        for c in np.unique(curve):
            mask = curve == c
            v = np.atleast_2d(np.asarray(V(int(c), np.asarray(q.tau)[mask]), dtype=float))
            if vals is None:
                vals = np.zeros((v.shape[0], len(curve)))
            vals[:, mask] = v
        return np.stack([-identities.weighted_integral(cd, "NN", vf) for vf in vals])

    def solve_interval(self, a, b, *, on_shortfall="flag"):
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
        res = self.search_interval(a, b, on_shortfall=on_shortfall)
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
        from ..eigensolver import Level
        return Spectrum(tuple(Level.from_search(x, self.prec)
                              for x in sorted(res.levels, key=lambda x: x.lam)),
                        prec=self.prec)

    def __repr__(self):
        return (f"MPSEigensolver({self.domain!r}, n_cols={self.n_cols}, "
                f"n_bdry={self.n_bdry}, n_int={self.n_int}, "
                f"rank={self.rank if self.rank is not None else 'threshold'})")


def _raise_on_shortfall(res, a, b, prec):
    """`PrecisionShortfall` if `res` fell short of what was asked; nothing otherwise."""
    d = res.diagnostics
    audit = d.count_audit
    short = sorted((L for L in res.levels if L.count_shortfall), key=lambda L: L.lam)
    reasons = []
    if short:
        reasons.append(f"{len(short)} level(s) unmet, the first at lam = {short[0].lam:.10g} "
                       f"(margin {short[0].margin:.3g}: the request spans that many local floor "
                       f"scales; 2 is needed)")
    if audit is not None and audit.verdict in ("short", "excess"):
        reasons.append(f"the count audit says {audit.verdict} (mean Weyl deficit "
                       f"{audit.statistic:.3g}, at the window's top {audit.top_deficit:.3g}"
                       + (f", from lam ~ {audit.step_at:.6g}" if np.isfinite(audit.step_at)
                          else "") + ")")
    if d.n_uncertain:
        reasons.append(f"{d.n_uncertain} count(s) could not be certified")
    if d.declined:
        reasons.append(f"the search declined: {d.decline_reason}")
    if not reasons:
        return
    first = short[0] if short else None
    raise PrecisionShortfall(
        f"the solve on lam in [{float(a):.6g}, {float(b):.6g}] fell short of prec={prec:.0e}: "
        + "; ".join(reasons) + ". The retry is a richer basis at the same prec.",
        result=res, level=first, margin=(first.margin if first is not None else float("nan")),
        audit=audit)
