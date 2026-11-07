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
from dataclasses import dataclass, replace

import numpy as np

import warnings

from .._core import Frozen, PrecisionShortfall, PrecisionWarning, _set, check_prec
from ..eigensolver import Eigensolver, Spectrum, lam_to_k, weyl_window
from .pencil import MPSPencil

__all__ = ["MPSEigensolver", "Eigenfunctions", "ClusterEigenfunctions", "DEFAULT_RNG"]


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
    Rellich origin. `orthonormal` is False, with a `reason`, when the Gram is deficient, or when
    `gram_residual` plus the level's Moler-Payne eps exceeds the bar `hadamard` refuses at
    (2026-11-05)."""

    lam: float
    coef: np.ndarray
    rule: object
    orthonormal: bool
    gram_residual: float
    reason: str = ""


@dataclass(frozen=True)
class ClusterEigenfunctions:
    """Eigenfunctions of a cluster's members, orthonormalized TOGETHER (2026-11-05;
    `SolveRecord.cluster_eigenfunctions`, `SolveRecord.eigenfunctions` for a member).

    Members sit at DIFFERENT `lam`, and a coefficient vector means a function only at its own: a
    family's columns change with `k`, and the same coefficients read at a neighbour's `lam` were
    1e-4 .. 2e-2 off on well-separated pairs. So the orthonormal functions are kept as the raw
    columns plus a mixing matrix: function `a` is `sum_j (basis at lams[j]) @ coef[:, j] *
    mix[j, a]`. `mix` is near the identity -- off by about the cluster's overlap -- and only the
    joint span is promised (`eigensolver.Cluster`).

    `lams` per raw column and `gram` their joint Gram (`identities.gram(..., lams=)`); `Lam` is
    the eigenvalue matrix in the orthonormal basis, `mix^T sym(gram diag(lams)) mix`: the raw
    column `j` is an eigenfunction at `lams[j]`, so `<u_i, A u_j> = lams[j] gram[i, j]`. It is
    `diag(lams)` to within `overlap * spread * lam`.
    `rule`, `orthonormal`, `gram_residual`, `reason` as for `Eigenfunctions`, the residual taken
    over the whole joint Gram, cross-level entries included. `levels` the level indices covered."""

    lams: np.ndarray
    coef: np.ndarray
    mix: np.ndarray
    gram: np.ndarray
    rule: object
    orthonormal: bool
    gram_residual: float
    levels: tuple = ()
    reason: str = ""

    @property
    def Lam(self):
        GD = self.gram * self.lams[None, :]
        return self.mix.T @ (0.5 * (GD + GD.T)) @ self.mix

    def select(self, cols, levels):
        """The same functions restricted to columns `cols` of `mix` (one member's)."""
        return replace(self, mix=self.mix[:, list(cols)], levels=tuple(levels))

    def cauchy_data(self, basis, quad):
        """`CauchyData` of the orthonormal functions on `quad`, each raw column at its own lam."""
        from ..integrals import cauchy
        u, un, ut = 0.0, 0.0, 0.0
        for lam in np.unique(self.lams):
            j = np.flatnonzero(self.lams == lam)
            cd = cauchy.cauchy_data(basis, self.coef[:, j] @ self.mix[j], float(lam), quad)
            u, un, ut = u + cd.u, un + cd.u_n, ut + cd.u_t
        return cauchy.CauchyData(quad, float(self.lams.max()), u, un, ut)

    def values(self, basis, pts):
        """The orthonormal functions at `pts`, `(len(pts), n_functions)`."""
        return sum(basis.values(lam_to_k(float(lam)), pts)
                   @ (self.coef[:, self.lams == lam] @ self.mix[self.lams == lam])
                   for lam in np.unique(self.lams))


#: `HadamardReport.attained` is at least this many times the worse of the exact-identity residual
#: and the Moler-Payne eigenfunction bound. Calibrated 2026-11-05 on 327 simple levels of the 51
#: douse shapes at prec 1e-10 against 1e-12 (`notes/accuracy_studies.py` grad): the vertex-gradient
#: error over max(identity, eps/alpha) had median 0.29, p99 1.41, worst 2.77
#: (`notes/hadamard_accuracy.md`, "Calibration").
HADAMARD_ESTIMATE_K = 3.0

#: How far above `prec` a level's `gram_residual` may sit before `hadamard` refuses it. The
#: derivative scales with `1/||u||^2`, so a normalization off by `g` puts `g` into every entry;
#: past this the matrix is not a derivative worth returning. Floored at 1e-8 so a loose `prec`
#: still refuses a residual of order one (`notes/hadamard_accuracy.md`: a flat triangle at prec
#: 1e-12 came back with `gram_residual` 0.16 and a dilation identity off by 0.43).
HADAMARD_GRAM_REFUSE = 1e3
#: Past this model error of the certification rule, a Moler-Payne bound -- off by about that
#: much RELATIVE -- is flagged (`eigenproblem._record`). Inf when a singular corner's rule was
#: demoted. Ordinary rules sit at `integral_prec(prec)`, far below.
CERTIFY_RULE_FLAG = 1e-3

#: The prec the CERTIFICATION rule is sized for (`integral_quad` sizes to a tenth of it, 1e-6),
#: unless the solve's own prec is looser. The Moler-Payne bound takes only the NORM from the
#: rule -- the boundary sup has its own dense sampling (`certify.boundary_sup`) -- and a norm
#: error delta moves the bound by delta relative, so a few digits are all it needs. Sizing it at
#: 0.1*prec bought 1e-11 nodes at prec 1e-10 for a 1e-3 question
#: (`accuracy_audit_2026-11-03.md` 2.2). `eigenfunctions` and `hadamard` keep the tight rule.
CERTIFY_NORM_PREC = 1e-5
HADAMARD_GRAM_FLOOR = 1e-8

#: The corner rule (`integrals.integral_quad(..., corners=)`) every boundary integral here uses,
#: 2026-11-03: graded, in corner-local frames (`notes/hadamard_accuracy.md`, "The Gram
#: pipeline"). On sector-apex norms it reaches 1e-15 .. 2e-14 to 1.9 pi, at the origin or
#: shifted, where the fitted rule gave 6e-12 .. 1e-8. On the flat triangle it fitted one edge 30%
#: wrong while claiming 1e-8, which was the 0.16 `gram_residual`. Certification switched only
#: after old-vs-new benchmarks: `notes/bound_coverage.py` (717 levels: same lam, coverage, met;
#: bounds within 0.89-1.1x) and the douse gate at 1e-10 (same tallies, flags, rungs; bounds
#: within 1.8e-5 relative).
INTEGRAL_CORNERS = "graded"

#: Rellich origins `eigenfunctions` checks its Gram at, besides the one it was formed at.
GRAM_PROBES = 3


@dataclass(frozen=True)
class HadamardReport:
    """What a Hadamard matrix is good for, beside the matrix (`hadamard(..., report=True)`).

    `sizing_prec` and `shortfalls` are the integral rule's own (`IntegralQuad`): the worst MODEL
    bound over its panels and every panel short of target. `gram_residual` is each level's
    orthonormality defect (`Eigenfunctions.gram_residual`). `mp_eps` is the worst Moler-Payne
    relative residual of the levels' eigenfunctions (nan where uncertified).

    `attained` is an ESTIMATE of H's relative error, calibrated rather than proven:

        max( max(sizing, norm_rule, gram_residual) + mp_eps,
             HADAMARD_ESTIMATE_K * max(identity_residual, eigenfunction_bound) ).

    The first term is what the rules and the normalization can see. The second carries the two
    errors they cannot: `identity_residual` measures H's own error on the four exact identities,
    and `eigenfunction_bound` is Moler-Payne's bound on how far the eigenfunctions sit from the
    true eigenspace, which grows as a neighbour comes close. Each covers what the other misses:
    on well-separated levels the identities catch a boundary `u_n` error of 5e-8 that eps/alpha
    reads as 3e-10; on near-pairs eps/alpha reads 5e-6 .. 8e-4 where the identities read 3e-10.
    Calibrated 2026-11-05 on 327 simple levels of the 51 douse shapes, the vertex gradients at prec
    1e-10 against 1e-12 (`notes/accuracy_studies.py` grad): error / max(identity, eps/alpha) had
    median 0.29, p99 1.41, worst 2.77, so `HADAMARD_ESTIMATE_K = 3` covered every one. Silent
    misses (error above 1e-10, estimate below) fell from 147 of 178 short levels under the old
    `attained` to none; the price is 90-odd warnings on levels that were not short. Not a bound:
    Moler-Payne bounds the eigenfunction in L2(Omega), H reads `u_n` on the boundary.

    For a cluster (`groups`), the bound uses its members' worst eps and the cluster's own gap to
    the outside: the joint span is what is determined. Where a level carries no `gap` (one not
    from `Eigenproblem.solve`), `eigenfunction_bound` is nan and only the identities enter.
    Before 2026-11-05 `attained` was the first term alone, and under-reported on 488 of 850
    levels (`notes/accuracy_audit_2026-11-03.md`)."""

    prec: float
    sizing_prec: float
    shortfalls: tuple
    gram_residual: tuple
    attained: float
    #: The worst model bound of the rules the levels were NORMALIZED on (`Eigenfunctions.rule`).
    #: H scales as 1/||u||^2, so it enters `attained` too; before 2026-11-03 only the Hadamard
    #: rule's own sizing did (`notes/accuracy_audit_2026-11-03.md`, 1.5).
    norm_rule_prec: float = float("nan")
    mp_eps: float = float("nan")
    #: The residuals of the four exact identities on the same nodes and eigenfunctions:
    #: translation x, y (H = 0), rotation (H = 0) and dilation (H = -2 lam I), the last two about
    #: the boundary's centroid and divided by its radius so every field has unit size; each
    #: `max |entry| / lam`. They measure H's own error with no reference solve.
    #: `identity_residual` is their max. (2026-11-05, `notes/hadamard_accuracy.md`.)
    identity_fields: tuple = ()
    identity_residual: float = float("nan")
    #: Moler-Payne's eigenfunction bound (Betcke-Trefethen 2005, Thm 8.1, eq. 8.2), worst over the
    #: block: `x sqrt(1 + x^2)`, `x = mp_eps / gap`, a level's or a cluster's (`Level.gap`).
    eigenfunction_bound: float = float("nan")

    @property
    def met(self):
        return self.attained <= self.prec


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

        # The families check their own evaluation against their own `prec` (`Sampling.
        # _check_accuracy`), so they must check against the solve's: a 1e-14 solve on a basis
        # built at 1e-8 would never warn, and a 1e-8 solve on a 1e-12 basis would warn for
        # nothing. Only the claim changes; the columns are the same.
        if getattr(basis, "prec", self.prec) != self.prec:
            basis = basis.with_prec(self.prec)
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
        from .objective import default_workers
        with cf.ThreadPoolExecutor(max_workers=default_workers()) as ex:
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

        key = (float(a), float(b), float(dk), bool(calibrate), n)
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

    def search_interval(self, a, b, *, prec=None, objective=None, on_shortfall="flag",
                        stable_rank=False, coarse_bracket=False):
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

        `stable_rank=True` re-refines each root whose cell has a noise tail at the stable rank,
        frozen in that cell; `coarse_bracket=True` brackets on the coarse penalty and refines each
        bracket at its stable rank (`chladni.mps.search.solve`; stage-policy steps 4 and 5, off by
        default).
        """
        from .search import solve

        if on_shortfall not in ("flag", "raise"):
            raise ValueError(f"on_shortfall must be 'flag' or 'raise', got {on_shortfall!r}")
        prec = self.prec if prec is None else check_prec(prec)
        if objective is not None and getattr(objective, "pencil", None) is not self._pencil:
            # Certification reads THIS solver's pencil (`certify_levels`), so a search driven
            # through another pencil's objective -- or a bare callable -- would certify levels it
            # did not find there (2026-10-27).
            raise ValueError("objective= must be this solver's own (self.objective(a, b)): "
                             "certification reads this solver's pencil")
        obj = self.objective(a, b, prec=prec) if objective is None else objective
        res = solve(obj, prec=prec, abort_unmet=(on_shortfall == "raise"),
                    stable_rank=stable_rank, coarse_bracket=coarse_bracket)
        if self.certify and res.levels:
            res = self.certify_levels(res, lam_max=float(b), prec=prec,
                                      objective=obj if coarse_bracket else None)
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
        from .search import Level, SolveDiagnostics, SolveResult, _precision_summary

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
        # What a merged window's search could not certify travels with the result: it used to
        # be dropped here, and the warm path accepted uncertain counts and deep rejections (a
        # possible lost eigenvalue) unflagged (`notes/accuracy_audit_2026-11-03.md`, 1.2).
        doubts = dict(n_uncertain=0, uncertain=(), rejected_deep=(), n_ambiguous=0)
        searched = []
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
                searched.extend(res_w.levels)      # certified by search_interval already
                dw = res_w.diagnostics
                n_evals += dw.evaluations
                doubts["n_uncertain"] += int(dw.n_uncertain)
                doubts["uncertain"] += tuple(dw.uncertain)
                doubts["rejected_deep"] += tuple(dw.rejected_deep)
                doubts["n_ambiguous"] += int(dw.n_ambiguous)
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
        # Only the REFINED levels are certified here. A merged window's levels were certified by
        # its own `search_interval`; certifying them again was one redundant Moler-Payne pass per
        # cluster window on every warm step (`accuracy_audit_2026-11-03.md` 2.3).
        if self.certify and levels:
            levels = list(self.certify_levels(SolveResult(levels=tuple(levels)),
                                              lam_max=float(windows[-1][1]), prec=prec).levels)
        levels = tuple(sorted(levels + searched, key=lambda L: L.lam))
        audit = asymp.CountAudit("unaudited", reason="tracked windows are not a prefix")
        res = SolveResult(levels=levels,
                          diagnostics=SolveDiagnostics(refine_evals=refine_evals,
                                                       evaluations=n_evals,
                                                       prec_requested=prec, count_audit=audit,
                                                       **doubts))
        if self.certify:
            unmet, attained = _precision_summary(levels, prec)
            res = _replace(res, diagnostics=_replace(res.diagnostics, unmet=unmet,
                                                     attained_prec=attained))
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

    def certify_levels(self, res, *, lam_max, prec, objective=None):
        """`res` with a Moler-Payne bound on every level (`chladni.integrals.certify`), and its
        precision summary recomputed from each level's reported bar (`Level.bound`: the tighter
        of the two, except that a `delta_res` short of `prec` is not admitted).

        Per level: the `mult` minimizing vectors at the level's `lam`, their Cauchy data on an
        integral rule sized for `lam_max` and `prec`, a Rellich norm, and the bound per vector;
        the level carries the WORST of them. Columns are certified separately, as lappy did --
        whether a cluster's subspace admits a sharper bound is an open question
        (`docs/design/boundary_integrals.md`). Off a Dirichlet boundary the bound does not
        apply, and each level says so in `bound_reason` and keeps `delta_res`.

        With `objective` (the search's, stage-policy step 6, 2026-10-27), each level is certified
        at its own rank AND at the stable rank its recorded spectra give (`search._stable_rank_near`),
        and keeps the SMALLER bound: every trial function gives a valid Moler-Payne bound, and the
        threshold rank's flutter makes the coefficient vector -- and the bound -- noisy (apex
        (0.3, 0.25) rung 3: 4.5e-11 in one run, 3.3e-9 in another, roots 1e-11 apart).
        """
        from dataclasses import replace

        from ..integrals import cauchy, certify
        from .search import _precision_summary

        p = self._pencil
        if np.any(np.asarray(p.bdry.bc) != 0.0):
            why = "Moler-Payne needs a Dirichlet condition on the whole boundary"
            levels = tuple(replace(L, bound_reason=why) for L in res.levels)
        else:
            iq = self._integral_quad(lam_max, max(float(prec), CERTIFY_NORM_PREC))
            origins = certify.x0_probes(self.domain, iq.x0)
            levels = []
            for L in res.levels:
                # At the rank the level was refined at, when a stage froze one (`Level.rank`),
                # and -- given the search's objective -- at the stable rank there too.
                ranks = [getattr(L, "rank", None)]
                if objective is not None and hasattr(objective, "spectra_near"):
                    from .search import _stable_rank_near
                    alt = _stable_rank_near(objective, float(L.lam))[0]
                    if alt is not None and alt not in ranks and alt >= int(L.mult):
                        ranks.append(alt)
                mp = None
                for rk in ranks:
                    fc = p.family_coef(L.lam, p.coef(L.lam, int(L.mult), rank=rk))
                    cd = cauchy.cauchy_data(p.basis, fc, L.lam, iq.quad)
                    k = lam_to_k(L.lam)

                    def evaluate(z, fc=fc, k=k):
                        V = p.basis.values(k, z)
                        return V @ fc, np.abs(V) @ np.abs(fc)

                    this = certify.moler_payne(cd, evaluate, self.domain, origins)
                    if mp is None or np.max(this.abs) < np.max(mp.abs):
                        mp = this
                levels.append(replace(L, mp_bound=float(np.max(mp.abs)),
                                      mp_eps=float(np.max(mp.eps)),
                                      cancellation=float(np.max(mp.cancellation)),
                                      mp_drift=float(np.max(mp.drift)),
                                      mp_norm_spread=float(np.max(mp.norm_spread)),
                                      bound_reason=mp.reason,
                                      rule_prec=float(iq.sizing_prec),
                                      delta_res_admitted=bool(
                                          L.delta_res <= prec * abs(L.lam)
                                          and not (np.max(mp.abs)
                                                   > MP_CROSSCHECK * prec * abs(L.lam)))))
            levels = tuple(levels)
        unmet, attained = _precision_summary(levels, prec)
        return replace(res, levels=levels,
                       diagnostics=replace(res.diagnostics, unmet=unmet,
                                           attained_prec=attained))

    def _integral_quad(self, lam_max, prec, corners=INTEGRAL_CORNERS):
        """The integral rule for `lam` up to `lam_max` at `prec`, built once per triple. A rule
        is a function of geometry, `lam_max`, `prec` and the corner rule alone, so it serves
        every level of a solve."""
        from ..integrals import integral_quad

        key = (float(lam_max), float(prec), corners)
        cache = getattr(self, "_integral_quads", None)
        if cache is None:
            _set(self, _integral_quads={})
            cache = self._integral_quads
        if key not in cache:
            cache[key] = integral_quad(self.domain, lam_max, prec=prec, corners=corners)
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
        and `reason` says why. `gram_residual` is `max |G - I|` recomputed at `GRAM_PROBES`
        further Rellich origins, worst taken: a DETECTOR a caller can read without an interior
        oracle, not an error bar. One origin was not enough: at the Rellich origin a corner's own
        error vanishes (`notes/hadamard_accuracy.md`, "The Gram pipeline").
        """
        from ..integrals import cauchy, certify, identities

        prec = self.prec if prec is None else check_prec(prec)
        lam, m = float(level.lam), int(level.mult)
        p = self._pencil
        fc = p.family_coef(lam, p.coef(lam, m, rank=getattr(level, "rank", None)))
        iq = self._integral_quad(lam, prec)
        cd = cauchy.cauchy_data(p.basis, fc, lam, iq.quad)
        G = identities.gram(cd, iq.x0)
        try:
            out, ok, why = fc @ identities.lowdin(G), True, ""
        except ValueError as err:
            out, ok, why = fc / np.sqrt(np.abs(np.diag(G))), False, str(err)
        cd2 = cauchy.cauchy_data(p.basis, out, lam, iq.quad)
        resid = max(float(np.abs(identities.gram(cd2, x0) - np.eye(m)).max())
                    for x0 in certify.x0_probes(self.domain, iq.x0, n=GRAM_PROBES)[1:])
        # `gram_residual` alone misses a starved basis: there it read 1e-15 .. 3e-9 against a true
        # Gram defect of 6e-8 .. 4e-5, which the level's Moler-Payne eps covered every time
        # (`notes/accuracy_audit_2026-11-03.md`, "ARC accuracy studies"). Past the bar `hadamard`
        # refuses at, the columns are not called orthonormal.
        eps = float(getattr(level, "mp_eps", np.nan))
        bar = max(HADAMARD_GRAM_REFUSE * prec, HADAMARD_GRAM_FLOOR)
        if ok and np.isfinite(eps) and not resid + eps <= bar:
            ok, why = False, (f"normalized only to about gram_residual + Moler-Payne eps = "
                              f"{resid + eps:.1e}, above {bar:.0e}: the eigenfunctions' boundary "
                              f"residual, not the Gram")
        return Eigenfunctions(lam=lam, coef=out, rule=iq, orthonormal=ok, gram_residual=resid,
                              reason=why)

    def _joint_cauchy(self, levels, efs, prec):
        """`(cd, lams, iq)`: several levels' columns, each at its OWN lam, on one integral rule
        sized for the largest."""
        from ..integrals import cauchy

        lam_max = max(float(L.lam) for L in levels)
        iq = self._integral_quad(lam_max, prec)
        basis = self._pencil.basis
        cds = [cauchy.cauchy_data(basis, ef.coef, float(L.lam), iq.quad)
               for L, ef in zip(levels, efs)]
        cd = cauchy.CauchyData(iq.quad, lam_max, np.hstack([c.u for c in cds]),
                               np.hstack([c.u_n for c in cds]), np.hstack([c.u_t for c in cds]))
        lams = np.concatenate([[float(L.lam)] * int(L.mult) for L in levels])
        return cd, lams, iq

    @staticmethod
    def _overlap_origins(iq):
        """The rule's Rellich origin, the boundary's centroid, and the centroid moved by the
        boundary's radius in x and in y: far enough apart that the gap's bias, affine in the
        origin, shows (the ARC overlap study's origins, `notes/accuracy_studies.py` overlap)."""
        q = iq.quad
        c = complex(np.sum(q.wts * q.pts) / np.sum(q.wts))
        R = float(np.abs(q.pts - c).max())
        return (complex(iq.x0), c, c + R, c + 1j * R)

    def cross_gram(self, A, B, *, prec=None):
        """`max |G_AB|` of the cross-level Gram block of levels `A` and `B` at each of four Rellich
        origins (`_overlap_origins`), as an array. Real overlap of the two levels' eigenfunctions
        reads the same at every origin; the gap's own bias, `(lam - mu)/2 int (v r.grad u - u
        r.grad v)` (`identities.gram`), moves with it. `eigenproblem.link_clusters` reads both."""
        from ..integrals import identities

        prec = self.prec if prec is None else check_prec(prec)
        efs = [self.eigenfunctions(L, prec=prec) for L in (A, B)]
        cd, lams, iq = self._joint_cauchy((A, B), efs, prec)
        m = int(A.mult)
        return np.array([float(np.abs(identities.gram(cd, x0, lams=lams)[:m, m:]).max())
                         for x0 in self._overlap_origins(iq)])

    def joint_eigenfunctions(self, levels, *, prec=None):
        """Several levels' eigenfunctions orthonormalized TOGETHER, as `ClusterEigenfunctions`:
        a cluster's (`eigensolver.Cluster`). Each level's own columns first (`eigenfunctions`),
        then the joint Gram over all of them, each at its own lam (`identities.gram(...,
        lams=)`), Löwdin-orthonormalized. `gram_residual` is the joint `max |G - I|` at
        `GRAM_PROBES` further origins: within a cluster it also carries the gap's bias, which is
        the limit boundary data can orthonormalize the members to."""
        from ..integrals import certify, identities

        prec = self.prec if prec is None else check_prec(prec)
        efs = [self.eigenfunctions(L, prec=prec) for L in levels]
        cd, lams, iq = self._joint_cauchy(levels, efs, prec)
        coef = np.hstack([ef.coef for ef in efs])
        ok = all(ef.orthonormal for ef in efs)
        why = "; ".join(ef.reason for ef in efs if ef.reason)
        G = identities.gram(cd, iq.x0, lams=lams)
        try:
            mix = identities.lowdin(G)
        except ValueError as err:
            mix, ok, why = np.diag(np.abs(np.diag(G)) ** -0.5), False, str(err)
        resid = max(float(np.abs(mix.T @ identities.gram(cd, x0, lams=lams) @ mix
                                  - np.eye(len(lams))).max())
                    for x0 in certify.x0_probes(self.domain, iq.x0, n=GRAM_PROBES)[1:])
        return ClusterEigenfunctions(lams=lams, coef=coef, mix=mix, gram=G, rule=iq,
                                     orthonormal=ok, gram_residual=resid, reason=why)

    def hadamard(self, level, V, *, prec=None, report=False, groups=()):
        """The Hadamard shape derivative of a Dirichlet level, or of several: `(n_fields, m, m)`.

        `level` may be a SEQUENCE of levels, giving one block over all their members, cross terms
        included -- the model a cluster optimizer needs on its way into a degeneracy. `groups`
        names positions in that sequence (contiguous tuples) whose levels form a CLUSTER
        (`eigensolver.Cluster`): their eigenfunctions are orthonormalized together
        (`joint_eigenfunctions`). Every other level is orthonormalized on its own. Without the
        grouping, a cluster's per-level eigenfunctions overlap and the block is wrong at the size
        of the overlap: 7.8e-4 of lam on a hard douse shape at prec 1e-10, with every identity at
        2e-10 (2026-11-05, `notes/hadamard_accuracy.md`, "Clusters"). `SolveRecord.hadamard`
        groups from the spectrum's clusters and refuses a list that cuts through one.

        `H[f, i, j] = - oint d_n u_i d_n u_j V_f ds` over the level's eigenfunctions,
        L2-orthonormalized (`eigenfunctions`); its eigenvalues are the derivatives of the level's
        members under the normal velocity `V_f`. `V(curve, t) -> (n_fields, len(t))`, with `curve`
        the boundary curve's index (segments in `domain.boundaries` order, flattened) and `t` in
        [0, 1] its parameter, so `V` may be discontinuous at a corner, as a moving vertex's normal
        velocity is (`docs/design/douse_requirements.md`, "Hadamard integrals"). Integrated on the
        `'integer'` family with GRADED corner panels (`integrals.integral_quad(...,
        corners="graded")`), evaluated in each corner's own frame. The fitted interpolatory rule
        this replaced stalled at 5e-8 .. 9e-6 on the velocity's integer powers of `r`
        (`notes/hadamard_accuracy.md`). Dirichlet only.

        THE RULE AND THE NORMALIZATION ARE REPORTED, NOT ASSUMED (2026-11-03,
        `notes/hadamard_accuracy.md`). A `PrecisionWarning` is issued when `attained` exceeds
        `prec` (`HadamardReport`), and `report=True` returns `(H, HadamardReport)`. A level whose
        `gram_residual` exceeds `max(HADAMARD_GRAM_REFUSE * prec, HADAMARD_GRAM_FLOOR)` is refused
        with `ValueError`, the same refusal as a level that could not be orthonormalized at all.
        """
        from ..integrals import cauchy, identities, integral_quad

        if self.domain.bc_type != "dir":
            raise NotImplementedError("Hadamard derivatives are built for Dirichlet boundaries "
                                      "only (the Neumann and mixed formulas differ)")
        prec = self.prec if prec is None else check_prec(prec)
        levels = [level] if hasattr(level, "lam") else list(level)
        groups = [tuple(int(j) for j in g) for g in groups if len(g) > 1]
        grouped = [j for g in groups for j in g]
        if (len(set(grouped)) != len(grouped) or any(not 0 <= j < len(levels) for j in grouped)
                or any(g != tuple(range(g[0], g[0] + len(g))) for g in groups)):
            raise ValueError(f"groups must be disjoint runs of positions in the levels, got "
                             f"{groups}")
        units = sorted(groups + [(j,) for j in range(len(levels)) if j not in grouped])
        refuse = max(HADAMARD_GRAM_REFUSE * prec, HADAMARD_GRAM_FLOOR)
        efs = []
        for u in units:
            Ls = [levels[j] for j in u]
            ef = (self.eigenfunctions(Ls[0], prec=prec) if len(u) == 1
                  else self.joint_eigenfunctions(Ls, prec=prec))
            at = "lam = " + ", ".join(f"{float(L.lam):.10g}" for L in Ls)
            if not ef.orthonormal:
                raise ValueError(f"the level at {at} could not be orthonormalized ({ef.reason}); "
                                 f"its Hadamard matrix would mix a wrong count")
            if not ef.gram_residual <= refuse:
                raise ValueError(f"the level at {at} is normalized only to gram_residual = "
                                 f"{ef.gram_residual:.1e} (refused above {refuse:.0e} at prec = "
                                 f"{prec:.0e}); its Hadamard matrix would carry that error in "
                                 f"every entry")
            efs.append((Ls, ef))
        levels = [L for Ls, _ in efs for L in Ls]
        lam = max(float(L.lam) for L in levels)       # the rule is sized for the largest
        key = ("hadamard", lam, float(prec))
        cache = getattr(self, "_integral_quads", None)
        if cache is None:
            _set(self, _integral_quads={})
            cache = self._integral_quads
        if key not in cache:
            cache[key] = integral_quad(self.domain, lam, prec=prec, family="integer",
                                       corners=INTEGRAL_CORNERS)
        q = cache[key].quad
        basis = self._pencil.basis
        cds = [ef.cauchy_data(basis, q) if isinstance(ef, ClusterEigenfunctions)
               else cauchy.cauchy_data(basis, ef.coef, float(Ls[0].lam), q) for Ls, ef in efs]
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
        H = np.stack([-identities.weighted_integral(cd, "NN", vf) for vf in vals])
        center = complex(np.sum(q.wts * q.pts) / np.sum(q.wts))
        rig, R = identities.rigid_weights(q, center)
        Hid = [-identities.weighted_integral(cd, "NN", w) for w in rig]
        # dilation: -2 Lam, Lam = lam I for a level and the joint eigenvalue matrix for a cluster
        Lams = [ef.Lam if isinstance(ef, ClusterEigenfunctions)
                else float(Ls[0].lam) * np.eye(int(Ls[0].mult)) for Ls, ef in efs]
        cut = np.cumsum([0] + [len(M) for M in Lams])
        Lam = np.zeros((cut[-1], cut[-1]))
        for M, a, b in zip(Lams, cut[:-1], cut[1:]):
            Lam[a:b, a:b] = M
        Hid[3] = Hid[3] + 2 * Lam / R                     # field scaled by 1/R
        # the identities hold within a level (or cluster) only: between lam != mu the Rellich and
        # translation forms keep a (lam - mu) term, so the cross entries are content, not error
        ident = tuple(max(float(np.abs(h[a:b, a:b]).max()) for a, b in zip(cut[:-1], cut[1:]))
                      / lam for h in Hid)
        iq = cache[key]
        resid = tuple(float(ef.gram_residual) for _, ef in efs)
        norm_rule = max(float(ef.rule.sizing_prec) for _, ef in efs)
        eps = [float(getattr(L, "mp_eps", np.nan)) for L in levels]
        mp_eps = max((e for e in eps if np.isfinite(e)), default=float("nan"))
        # Moler-Payne's eigenfunction bound per level or cluster: eps/alpha sqrt(1 + (eps/alpha)^2)
        bounds = []
        for Ls, _ in efs:
            e = max(float(getattr(L, "mp_eps", np.nan)) for L in Ls)
            a = float(getattr(Ls[0], "gap", np.nan))
            x = np.nan if not np.isfinite(a) else (e / a if a > 0 else np.inf)
            bounds.append(x * np.sqrt(1.0 + x * x))
        eig_bound = max(bounds) if np.all(np.isfinite(bounds) | np.isinf(bounds)) else float("nan")
        base = (max([float(iq.sizing_prec), norm_rule] + list(resid))
                + (mp_eps if np.isfinite(mp_eps) else 0.0))
        calib = HADAMARD_ESTIMATE_K * max(max(ident),
                                          eig_bound if not np.isnan(eig_bound) else 0.0)
        rep = HadamardReport(prec=float(prec), sizing_prec=float(iq.sizing_prec),
                             shortfalls=tuple(iq.shortfalls), gram_residual=resid,
                             attained=max(base, calib), norm_rule_prec=norm_rule, mp_eps=mp_eps,
                             identity_fields=ident, identity_residual=max(ident),
                             eigenfunction_bound=float(eig_bound))
        if not rep.met:
            if calib > base:
                where = ("the eigenfunctions' Moler-Payne bound eps/alpha: a near neighbour"
                         if eig_bound > max(ident) else "the exact-identity residual")
            else:
                where = (", ".join(w for w, *_ in rep.shortfalls)
                         or ("the eigenfunctions' Moler-Payne residual"
                             if rep.mp_eps > rep.prec / 2 else "the normalization"))
            warnings.warn(
                f"the Hadamard matrix at lam = {lam:.10g} is good to {rep.attained:.1e} "
                f"relative (estimate), short of prec = {prec:.0e}: {where}. "
                f"report=True returns the details.", PrecisionWarning, stacklevel=2)
        return (H, rep) if report else H

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
                + (f"{d.attained_prec:.1e} relative" if np.isfinite(d.attained_prec)
                   else "no finite bar") + f", short of the requested "
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
                       f"(margin {short[0].margin:.3g}: prec*lam over the reported bar, times "
                       f"2; 2 is needed" + (f"; {short[0].bound_reason}" if short[0].bound_reason
                                            else "") + ")")
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
