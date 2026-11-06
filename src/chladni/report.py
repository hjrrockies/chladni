"""`SolveReport`: what a solve computed along the way, collected in one flat record (2026-11-06).

A `SolveRecord` answers the question asked -- the levels, their bounds, the flags. The search,
the planner and the certificate compute much more on the way there, and most of it was dropped.
This module gathers it, so a solve can be studied after the fact: which stage limited the
accuracy, how close the pencil came to cutting a real direction, how flat the objective was at
each root. Nothing here changes a solve. It is optional, and reached through
`SolveRecord.report(tier)`.

Tiers, by cost:

* 0 -- already on the record: the spectrum, its levels and clusters, the count audit, the plan,
  the pencil's size and truncation, the search's diagnostics.
* 1 -- computed during the solve and, before this module, thrown away: the search's own per-level
  record (tension floor, `delta_res`, how the count was decided, which term set the bound,
  Moler-Payne's drift and norm spread), the planner's ladder (each rung tried, its outcome and
  wall time), the integral rules' shortfalls, and the collocation points near each corner.
  Keeping them costs nothing; reading them costs microseconds.
* 2 -- cheap extra work, per level: the slope of the tension at the root and the floor it implies,
  one SVD of the pencil at the root, the eigenfunctions' Gram residual, and a Hadamard report on
  the exact identity fields (translation, rotation, dilation), which needs no reference. Roughly
  a few tension evaluations, one SVD and one integral per level.

Every value is a float, int, bool, str, None, or a list or dict of those, so a report goes into
a JSON ledger as it is (`SolveReport.as_dict`). Fields whose source is absent (a declined solve,
a warm solve without a ladder, a record without its solver) are left out, not filled with NaN.
"""
import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ["SolveReport", "solve_report", "TENSION_STEP"]

#: The tension slope at a root is read at `lam (1 +- TENSION_STEP)`, or a quarter of the gap to
#: the nearest neighbour when that is smaller. Far above the floor (the floor sits near `prec`)
#: and far below the spacing, so `sigma(lam +- h) ~ c h` and `c` is the slope.
TENSION_STEP = 1e-6


@dataclass(frozen=True)
class SolveReport:
    """`run` (one dict for the solve) and `levels` (one dict per kept level, in the spectrum's
    order). `tier` is the deepest tier collected. See the module docstring."""
    tier: int
    run: dict = field(default_factory=dict)
    levels: tuple = ()

    def as_dict(self):
        return dict(tier=self.tier, run=_clean(self.run), levels=[_clean(L) for L in self.levels])

    def __getitem__(self, key):
        return self.run[key]


def solve_report(record, tier=1):
    """The `SolveReport` of `record` (a `SolveRecord`) up to `tier` (0, 1 or 2)."""
    tier = int(tier)
    if tier not in (0, 1, 2):
        raise ValueError(f"tier must be 0, 1 or 2, got {tier}")
    run = _run_tier0(record)
    levels = [_level_tier0(record, i) for i in range(len(record.spectrum.levels))]
    if tier >= 1:
        run.update(_run_tier1(record))
        for i, L in enumerate(levels):
            L.update(_level_tier1(record, i))
    if tier >= 2 and record.solver is not None and not record.spectrum.declined:
        _levels_tier2(record, levels)
    return SolveReport(tier=tier, run=run, levels=tuple(levels))


# --- tier 0 ---------------------------------------------------------------------------------

def _run_tier0(record):
    spec = record.spectrum
    dom = None if record.solver is None else record.solver.domain
    run = dict(prec=spec.prec, n_levels=len(spec.levels), n_values=int(sum(spec.mult)),
               declined=bool(spec.declined), reason=spec.reason, flags=list(spec.flags),
               warm=bool(record.warm), warm_depth=int(record.warm_depth),
               warm_note=None if record.warm_note is None else str(record.warm_note))
    if record.seconds is not None:
        run["seconds"] = record.seconds
    if dom is not None:
        run["geometry"] = dict(achievable_prec=_f(dom.achievable_prec), area=_f(dom.area),
                               perimeter=_f(dom.perimeter), diameter=_f(dom.diameter),
                               corners=[dict(z=[c.z.real, c.z.imag], omega=_f(c.omega),
                                             clearance=_f(getattr(c, "clearance", math.inf)))
                                        for c in dom.corners])
    if spec.count_audit is not None:
        run["count_audit"] = _scalars(spec.count_audit)
    run["clusters"] = [_scalars(c) for c in spec.clusters]
    if record.plan is not None:
        run["plan"] = _scalars(record.plan)
        run["plan"]["n_cols"] = int(record.plan.n_cols)
    s = record.solver
    if s is not None:
        run["pencil"] = dict(n_cols=int(s.n_cols), n_bdry=int(s.n_bdry), n_int=int(s.n_int),
                             rank=s.rank, rtol=_f(s.rtol), reg=str(s.reg))
    d = record.diagnostics
    if d is not None:
        run["diagnostics"] = _scalars(d)
        run["diagnostics"]["n_rejected_deep"] = len(getattr(d, "rejected_deep", ()) or ())
        if getattr(d, "search", None) is not None:
            run["search"] = _scalars(d.search)
    return run


def _level_tier0(record, i):
    L = record.spectrum.levels[i]
    out = _scalars(L)
    out["index"] = i
    return out


# --- tier 1 ---------------------------------------------------------------------------------

def _run_tier1(record):
    run = {}
    if record.ladder:
        run["ladder"] = [dict(r) for r in record.ladder]
    s = record.solver
    if s is not None:
        quads = getattr(s, "_integral_quads", {}) or {}
        run["integral_rules"] = [
            dict(sizing_prec=_f(getattr(q, "sizing_prec", math.nan)),
                 shortfalls=[[str(w), _f(a), str(r)] for w, a, r in getattr(q, "shortfalls", ())])
            for q in quads.values()]
        if hasattr(s, "_pencil"):
            run["corner_points"] = _corner_points(s.domain, s._pencil)
    return run


def _corner_points(dom, pencil):
    """Collocation near each corner: boundary and interior points within half the corner's
    clearance (or half its shorter edge, if smaller). A thin wedge whose interior holds no point
    is a candidate cause of the 1e-7 cap on sharp shapes (`notes/hard_cases.md`)."""
    bp = np.asarray(pencil.bdry_pts, dtype=complex)
    ip = np.asarray(pencil.int_pts, dtype=complex)
    zs = np.array([c.z for c in dom.corners])
    out = []
    for j, c in enumerate(dom.corners):
        edges = [abs(c.z - zs[j - 1]), abs(c.z - zs[(j + 1) % len(zs)])]
        r = 0.5 * min(min(edges), float(getattr(c, "clearance", math.inf)))
        out.append(dict(omega=_f(c.omega), radius=_f(r),
                        n_bdry=int(np.sum(np.abs(bp - c.z) < r)),
                        n_int=int(np.sum(np.abs(ip - c.z) < r))))
    return out


def _level_tier1(record, i):
    sl = record.search_levels
    if not sl or i >= len(sl):
        return {}
    L = sl[i]
    out = {f"search_{k}": v for k, v in _scalars(L).items()
           if k not in ("lam", "mult", "prec")}
    for name in ("bracket", "wide_bracket", "spacings"):
        v = getattr(L, name, ())
        if v:
            out[f"search_{name}"] = [_f(x) for x in np.ravel(v)]
    return out


# --- tier 2 ---------------------------------------------------------------------------------

def _levels_tier2(record, levels):
    spec, s = record.spectrum, record.solver
    lam = np.asarray(spec.lam, dtype=float)
    done_clusters = {}
    for i, row in enumerate(levels):
        L = spec.levels[i]
        lam_i = float(L.lam)
        gaps = np.abs(np.delete(lam, i) - lam_i)
        h = min(TENSION_STEP * lam_i, 0.25 * float(gaps.min())) if gaps.size else TENSION_STEP * lam_i
        try:
            s0 = float(s.tension(lam_i))
            sp, sm = float(s.tension(lam_i + h)), float(s.tension(lam_i - h))
            c = 0.5 * (sp + sm) / h
            row.update(tension_root=s0, tension_step=h, tension_slope=c,
                       tension_floor_rel=s0 / (c * lam_i) if c > 0 else math.inf)
        except Exception as err:                      # noqa: BLE001 -- a report never raises
            row["tension_error"] = f"{type(err).__name__}: {err}"
        try:
            rd = s.rank_diagnostics(lam_i)
            sv = np.asarray(rd.singular_values if hasattr(rd, "singular_values") else rd[1])
            rk = int(rd.rank if hasattr(rd, "rank") else rd[0])
            if sv.size and sv[0] > 0:
                row.update(svd_rank=rk, svd_cut_rel=_f(sv[rk - 1] / sv[0]) if rk >= 1 else None,
                           svd_dropped_rel=_f(sv[rk] / sv[0]) if rk < sv.size else None,
                           svd_min_rel=_f(sv[-1] / sv[0]))
        except Exception as err:                      # noqa: BLE001
            row["svd_error"] = f"{type(err).__name__}: {err}"
        try:
            ef = record.eigenfunctions(i)
            row.update(ef_orthonormal=bool(ef.orthonormal), ef_gram_residual=_f(ef.gram_residual))
        except Exception as err:                      # noqa: BLE001
            row["ef_error"] = f"{type(err).__name__}: {err}"
        unit = i if L.cluster is None else tuple(spec.clusters[L.cluster].levels)
        if unit not in done_clusters:
            done_clusters[unit] = _identity_report(record, unit)
        row.update(done_clusters[unit])


def _identity_report(record, unit):
    """`HadamardReport` fields for level `unit` (or a cluster's levels, together), on a zero
    velocity field: the identity residuals and the calibrated `attained` are formed inside
    `hadamard` from the exact identity fields whatever `V` is, so the user field adds nothing."""
    import warnings

    from ._core import PrecisionWarning

    def zero(curve, t):
        return np.zeros((1, len(t)))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PrecisionWarning)
            _, rep = record.hadamard(list(unit) if isinstance(unit, tuple) else unit, zero,
                                     report=True)
    except Exception as err:                          # noqa: BLE001
        return {"hadamard_error": f"{type(err).__name__}: {err}"}
    out = {}
    for k, v in _scalars(rep).items():
        out[f"hadamard_{k}"] = v
    fields_ = getattr(rep, "identity_fields", None)
    if fields_ is not None:
        out["hadamard_identity_fields"] = [_f(x) for x in np.ravel(fields_)]
    return out


# --- plumbing -------------------------------------------------------------------------------

def _f(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x


def _scalars(obj):
    """The scalar fields of a dataclass (and its scalar properties named in `__dataclass_fields__`
    only): numbers, strings, bools, None, and flat tuples of numbers."""
    out = {}
    if not dataclasses.is_dataclass(obj):
        return out
    for f_ in dataclasses.fields(obj):
        v = getattr(obj, f_.name)
        if isinstance(v, (bool, np.bool_)):
            out[f_.name] = bool(v)
        elif isinstance(v, (int, np.integer)):
            out[f_.name] = int(v)
        elif isinstance(v, (float, np.floating)):
            out[f_.name] = float(v)
        elif isinstance(v, str) or v is None:
            out[f_.name] = v
        elif isinstance(v, (tuple, list)) and all(isinstance(x, (int, float, np.integer,
                                                                 np.floating)) for x in v):
            out[f_.name] = [float(x) if isinstance(x, (float, np.floating)) else int(x)
                            for x in v]
        elif isinstance(v, np.ndarray) and v.ndim <= 1 and v.dtype.kind in "fiub":
            out[f_.name] = [float(x) for x in v]
    for name in ("margin", "resolution", "bound", "width", "attained_prec", "count_shortfall"):
        if hasattr(type(obj), name) and isinstance(getattr(type(obj), name), property):
            try:
                v = getattr(obj, name)
            except Exception:                         # noqa: BLE001
                continue
            out[name] = bool(v) if isinstance(v, (bool, np.bool_)) else _f(v)
    return out


def _clean(x):
    """JSON-safe: non-finite floats become strings ("inf", "-inf", "nan") so a strict parser
    accepts the report."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    return x
