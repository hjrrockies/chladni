"""Accuracy studies for the 2026-11-03 audit fixes, as one Slurm array.  (2026-11-03)

Follow-ups to `notes/accuracy_audit_2026-11-03.md`, sized for ARC rather than a laptop:

  bessel   Does every Bessel branch's reported `rel_err` bound the truth, ON LINUX's scipy build?
           The scipy coefficient in `_j_scipy` was fitted on macOS. J dispatcher over dense
           (nu, x) bands against 30-digit mpmath; the Y (`log_y`) branches, never re-measured;
           the Olver/Debye branch on its own, whose 2e-12 is a flat claim nobody measured.
           Values AND derivatives (the derivative carries no bound of its own: recorded only).

  eigs     Per douse reference shape (`benchmarks/douse/references.json`, certified to ~1e-11)
           and per hard ratio shape (`hadamard_hard_shapes_cases.json`), at prec 1e-10 and
           1e-12, on two checkouts: OLD = 5a02def (before the series-branch fix) and NEW =
           9de388a (with it), so nothing but the kernel differs. Per level: lam against the
           reference, the bar and `met`; the Hadamard identity error (translation, rotation: H
           = 0; dilation: eig H = -2 lam) against `HadamardReport.attained` and `mp_eps`. A
           solve that raises is a row with its reason -- the census of the 315-degree notch's
           "no source offset keeps the sources outside" failure.

  starved  Is `gram_residual` honest where it is LARGE, where `hadamard()` refuses on it? The
           plan's basis is shrunk (corner orders and sources scaled by a factor) until levels
           degrade, and each level's `gram_residual` is set beside its Gram matrix from an
           independent interior rule (ear-clipped triangles, Duffy-mapped Gauss).

  grad     (2026-11-05, its own array: `--table grad`) Does the exact-identity residual of H
           bound the error of the VERTEX gradients? H at 1e-10 against H at 1e-12 on every
           shape, per vertex hat field. See "grad" below.

Run:
  python notes/accuracy_studies.py run <study> <args...>        one unit, JSON to stdout
  python notes/accuracy_studies.py task <i> --tip SRC --old SRC --new SRC --out DIR
  python notes/accuracy_studies.py task <i> --tip SRC --table grad --out DIR
  python notes/accuracy_studies.py analyze DIR
  python notes/accuracy_studies.py analyze_grad DIR
  python notes/accuracy_studies.py ntasks [grad]
"""
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EPS = np.finfo(float).eps
MP_DPS = 30

# --- the task list ------------------------------------------------------------------------------

J_BANDS, Y_BANDS, DEBYE_BANDS = 24, 4, 4
PRECS = ("1e-10", "1e-12")
STARVED_SHAPES = ["T1.1-good", "T1.4-hex", "T2.2", "T2.6-N5k4-L7-ref0", "T2.8-N5s2",
                  "T2.9-ngon5", "T2.7-median", "T2.4-N5-L5"]
STARVE_FACTORS = (1.0, 0.6, 0.4, 0.25, 0.15, 0.1, 0.06)
GRAD_PRECS = (1e-10, 1e-12)


def _ref_cases():
    return json.load(open(ROOT / "benchmarks" / "douse" / "references.json"))["cases"]


def _shape_cases():
    geo = {c["id"]: c for c in json.load(open(ROOT / "benchmarks" / "douse" / "cases.json"))["cases"]}
    out = []
    for r in _ref_cases():
        out.append(dict(id=r["id"], verts=geo[r["id"]]["vertices"], refs=r["levels"]))
    for c in json.load(open(HERE / "hadamard_hard_shapes_cases.json"))["cases"]:
        out.append(dict(id="hard:" + c["source"], verts=c["verts"], refs=None, n=len(c["lams"])))
    return out


def tasks():
    out = [("tip", "bessel", "J", b) for b in range(J_BANDS)]
    out += [("tip", "bessel", "Y", b) for b in range(Y_BANDS)]
    out += [("tip", "bessel", "debye", b) for b in range(DEBYE_BANDS)]
    out += [("tip", "starved", s, None) for s in STARVED_SHAPES]
    out += [(impl, "eigs", ci, p) for ci in range(len(_shape_cases())) for p in PRECS
            for impl in ("old", "new")]
    return out


# --- bessel -------------------------------------------------------------------------------------

def _mp_logabs(v):
    import mpmath as mp
    return float(mp.log(abs(v))) if v != 0 else -np.inf


def _compare(nus, xs, fn, ref, dref, branch_of=None):
    """Worst true-error / bound over a grid, for the value and (informational) the derivative."""
    NU, X = np.meshgrid(nus, xs, indexing="ij")
    s = fn(NU, X)
    R = np.array([[ref(a, b) for b in xs] for a in nus], dtype=object)
    logref = np.vectorize(_mp_logabs, otypes=[float])(R)
    got = np.log(np.abs(s.S)) + s.sigma
    err = np.abs(got - logref)
    bound = s.rel_err + 2 * EPS * np.abs(logref)          # + the check's own rounding
    ratio = np.where(np.isfinite(logref), err / bound, np.nan)
    dR = np.array([[dref(a, b) for b in xs] for a in nus], dtype=object)
    dlog = np.vectorize(_mp_logabs, otypes=[float])(dR)
    dgot = np.log(np.abs(s.Sp)) + s.sigma
    dratio = np.where(np.isfinite(dlog), np.abs(dgot - dlog) / bound, np.nan)
    flat = np.argsort(np.nan_to_num(ratio, nan=-1).ravel())[::-1][:12]
    worst = []
    for idx in flat:
        i, j = np.unravel_index(idx, ratio.shape)
        worst.append(dict(nu=float(NU[i, j]), x=float(X[i, j]), err=float(err[i, j]),
                          rel_err=float(s.rel_err[i, j]), ratio=float(ratio[i, j]),
                          log_abs=float(logref[i, j]),
                          branch=branch_of(NU[i, j], X[i, j]) if branch_of else None))
    return dict(n=int(np.isfinite(ratio).sum()), max_ratio=float(np.nanmax(ratio)),
                n_over=int(np.nansum(ratio > 1)), p99_ratio=float(np.nanpercentile(ratio, 99)),
                max_rel_err_true=float(np.nanmax(np.where(np.isfinite(logref), err, np.nan))),
                deriv_max_ratio=float(np.nanmax(dratio)),
                deriv_p99_ratio=float(np.nanpercentile(dratio, 99)), worst=worst)


def _j_branch(nu, x):
    """Which branch the dispatcher used for one cell, by matching its output exactly."""
    from chladni.helmholtz import _bessel as B
    d = B.log_j(nu, x, False)
    for name, f in (("series", B._j_series), ("scipy", B._j_scipy), ("debye", B._j_debye)):
        try:
            o = f(np.array([nu]), np.array([x]))
            if np.asarray(o.S).ravel()[0] == d.S:
                return name
        except Exception:                                   # noqa: BLE001
            pass
    return "?"


def run_bessel(kind, band):
    import mpmath as mp
    from chladni.helmholtz import _bessel as B
    mp.mp.dps = MP_DPS
    rng = np.random.default_rng(100 + band)
    if kind == "J":
        lo, hi = 400.0 * band / J_BANDS, 400.0 * (band + 1) / J_BANDS
        nus = np.sort(rng.uniform(lo, hi, 15))
        xs = np.unique(np.concatenate([np.geomspace(0.05, 600, 70), rng.uniform(0.5, 600, 50)]))
        r = _compare(nus, xs, lambda n, x: B.log_j(n, x, True),
                     lambda a, b: mp.besselj(a, b),
                     lambda a, b: mp.besselj(a, b, derivative=1), _j_branch)
    elif kind == "Y":
        ms = np.arange(20 * band, 20 * (band + 1), dtype=float)
        xs = np.unique(np.concatenate([np.geomspace(1e-3, 300, 70), rng.uniform(0.1, 300, 30)]))
        r = _compare(ms, xs, lambda m, x: B.log_y(m, x, True),
                     lambda a, b: mp.bessely(a, b),
                     lambda a, b: mp.bessely(a, b, derivative=1))
    else:                                                   # where the dispatcher uses Debye
        # Branch C is self-selecting: the cells where scipy's `jv` underflows to 0 (or gives a
        # non-finite value). Check it there, through the dispatcher -- elsewhere it is never used,
        # and near the turning point it is not accurate, by design.
        from scipy.special import jv
        lo, hi = 200.0 + 450.0 * band, 200.0 + 450.0 * (band + 1)
        nus = np.sort(rng.uniform(lo, hi, 8))
        rows = []
        for nu in nus:
            xs = nu * np.geomspace(1e-4, 0.98, 400)
            xs = xs[(jv(nu, xs) == 0.0) | ~np.isfinite(jv(nu, xs))][-40:]   # the hardest 40
            if xs.size:
                rows.append(_compare(np.array([nu]), xs, lambda n, x: B.log_j(n, x, True),
                                     lambda a, b: mp.besselj(a, b),
                                     lambda a, b: mp.besselj(a, b, derivative=1), _j_branch))
        r = dict(n=sum(q["n"] for q in rows), max_ratio=max(q["max_ratio"] for q in rows),
                 n_over=sum(q["n_over"] for q in rows),
                 p99_ratio=max(q["p99_ratio"] for q in rows),
                 max_rel_err_true=max(q["max_rel_err_true"] for q in rows),
                 deriv_max_ratio=max(q["deriv_max_ratio"] for q in rows),
                 deriv_p99_ratio=max(q["deriv_p99_ratio"] for q in rows),
                 worst=sorted((w for q in rows for w in q["worst"]),
                              key=lambda w: -np.nan_to_num(w["ratio"]))[:12])
    import scipy
    r.update(kind=kind, band=band, scipy=scipy.__version__, numpy=np.__version__,
             platform=sys.platform)
    return r


# --- eigs ---------------------------------------------------------------------------------------

def run_eigs(ci, prec):
    sys.path.insert(0, str(HERE))
    from hadamard_hard_shapes import fields
    from chladni.eigenproblem import Eigenproblem
    from chladni.geometry import Polygon
    c = _shape_cases()[ci]
    prec = float(prec)
    verts = np.array([complex(*v) for v in c["verts"]])
    refs = c["refs"] or []
    n = min(sum(int(r["mult"]) for r in refs), 10) if refs else c["n"]
    t0 = time.time()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            rec = Eigenproblem(Polygon(verts, bc="dir")).solve(n, prec=prec)
        except Exception as e:                              # noqa: BLE001 -- a failure is a row
            return dict(id=c["id"], prec=prec, failed=f"{type(e).__name__}: {str(e)[:300]}",
                        t_solve=time.time() - t0)
    t_solve = time.time() - t0
    # some solve paths (the old tree's, at least) return a record with no solver: the levels
    # are still a row, only the Hadamard columns go missing
    V = fields(rec.solver.domain, verts) if rec.solver is not None else None
    levels = []
    for i, L in enumerate(rec.spectrum.levels):
        lam = float(L.lam)
        row = dict(i=i, lam=lam, mult=int(L.mult), bound=float(L.bound), met=bool(L.met),
                   mp_eps=float(getattr(L, "mp_eps", np.nan)))
        good = [r for r in refs if r.get("status") != "provisional"
                and np.isfinite(r.get("bound", np.nan))]
        if good:
            r = min(good, key=lambda r: abs(r["lam"] - lam))
            if abs(r["lam"] - lam) < 1e-4 * lam:
                row.update(ref=float(r["lam"]), ref_bound=float(r["bound"]),
                           err=abs(lam - float(r["lam"])))
        try:
            if V is None:
                raise ValueError("the solve returned no solver")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                H, rep = rec.hadamard(i, V, report=True)
            H = np.asarray(H)
            row.update(ident=float(max(np.abs(H[f]).max() for f in range(3)) / lam),
                       dil=float(np.abs(np.linalg.eigvalsh(H[3]) + 2 * lam).max() / lam),
                       attained=float(rep.attained), h_mp_eps=float(rep.mp_eps),
                       gram=[float(g) for g in np.atleast_1d(rep.gram_residual)],
                       sizing=float(rep.sizing_prec))
        except Exception as e:                              # noqa: BLE001
            row.update(h_error=f"{type(e).__name__}: {str(e)[:200]}")
        levels.append(row)
    return dict(id=c["id"], prec=prec, t_solve=t_solve, flags=list(rec.flags),
                warnings=sorted({str(x.message)[:160] for x in w}), levels=levels)


# --- starved ------------------------------------------------------------------------------------

def _ear_clip(v):
    """Triangles of a simple polygon (counter-clockwise or not), by ear clipping."""
    v = list(v)
    area = sum((a.conjugate() * b).imag for a, b in zip(v, v[1:] + v[:1]))
    if area < 0:
        v = v[::-1]
    idx = list(range(len(v)))
    tris = []

    def cross(a, b, c):
        return ((b - a).conjugate() * (c - a)).imag

    while len(idx) > 3:
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = v[i0], v[i1], v[i2]
            if cross(a, b, c) <= 0:
                continue
            if any(cross(a, b, v[j]) > 0 and cross(b, c, v[j]) > 0 and cross(c, a, v[j]) > 0
                   for j in idx if j not in (i0, i1, i2)):
                continue
            tris.append((a, b, c))
            idx.pop(k)
            break
        else:
            raise ValueError("ear clipping found no ear: the polygon is not simple")
    tris.append(tuple(v[j] for j in idx))
    return tris


def _interior_rule(verts, n=40):
    x, w = np.polynomial.legendre.leggauss(n)
    x, w = 0.5 * (x + 1), 0.5 * w
    a, b = np.meshgrid(x, x, indexing="ij")
    wa = np.outer(w, w)
    pts, wts = [], []
    for p, q, r in _ear_clip(verts):
        det = abs(((q - p).conjugate() * (r - p)).imag)
        z = p + a * (q - p) + a * b * (r - q)              # Duffy, Jacobian a * det
        pts.append(z.ravel())
        wts.append((det * wa * a).ravel())
    return np.concatenate(pts), np.concatenate(wts)


def run_starved(case_id):
    from dataclasses import replace
    from chladni.eigensolver import lam_to_k
    from chladni.geometry import Polygon
    from chladni.plan import plan_for
    geo = {c["id"]: c for c in json.load(open(ROOT / "benchmarks" / "douse" / "cases.json"))["cases"]}
    ref = {r["id"]: r for r in _ref_cases()}[case_id]
    verts = np.array([complex(*v) for v in geo[case_id]["vertices"]])
    dom = Polygon(verts, bc="dir")
    lo, hi = ref["window"]
    n = sum(int(r["mult"]) for r in ref["levels"])
    prec = 1e-10
    plan = plan_for(dom, n, prec)
    pts, wts = _interior_rule(verts)
    out = []
    for f in STARVE_FACTORS:
        p = replace(plan, corner_orders=tuple(max(int(round(f * m)), 1) if m else 0
                                              for m in plan.corner_orders),
                    n_sources=int(round(f * plan.n_sources)))
        row = dict(factor=f, n_cols=int(p.n_cols), n_ref=n, levels=[])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                s = p.solver(dom, prec=prec)
                res = s.search_interval(lo, hi)
                for L in res.levels:
                    ef = s.eigenfunctions(L)
                    U = s.basis.values(lam_to_k(ef.lam), pts) @ ef.coef
                    G = (U * wts[:, None]).T @ U
                    r = min(ref["levels"], key=lambda r: abs(r["lam"] - L.lam))
                    row["levels"].append(dict(
                        lam=float(L.lam), mult=int(L.mult), ref=float(r["lam"]),
                        lam_err=abs(float(L.lam) - float(r["lam"])) / float(r["lam"]),
                        met=bool(L.met(prec)), mp_eps=float(L.mp_eps),
                        gram_residual=float(ef.gram_residual), orthonormal=bool(ef.orthonormal),
                        oracle_defect=float(np.abs(G - np.eye(len(G))).max())))
        except Exception as e:                              # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        out.append(row)
    return dict(id=case_id, prec=prec, rows=out)


# --- grad (2026-11-05) --------------------------------------------------------------------------
#
# Does the exact-identity residual bound the error of the VERTEX gradients? Per shape, H at 1e-10
# and at 1e-12 (the reference, where its own identities read <= 1e-11; Part A's FD checks it on
# the hard shapes) for every vertex hat field, plus translation x, y and rotation and dilation
# about the vertex centroid. Per level: each hat field's tr H and, for a multiple level, the
# eigenvalues of its block; the identity residuals, rotation and dilation divided by the radius
# R = max |v - centre| so every field has unit size; attained, gram, mp_eps.

def _grad_levels(rec, V, R):
    out = []
    for i, L in enumerate(rec.spectrum.levels):
        lam = float(L.lam)
        row = dict(i=i, lam=lam, mult=int(L.mult), met=bool(L.met), bound=float(L.bound),
                   mp_eps=float(getattr(L, "mp_eps", np.nan)))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                H, rep = rec.hadamard(i, V, report=True)
            H = np.asarray(H)
            I2 = 2 * lam * np.eye(H.shape[1])
            row.update(ident=[float(np.abs(H[0]).max() / lam), float(np.abs(H[1]).max() / lam),
                              float(np.abs(H[2]).max() / (lam * R)),
                              float(np.abs(H[3] + I2).max() / (lam * R))],
                       tr=[float(np.trace(H[f])) for f in range(4, H.shape[0])],
                       eig=[np.linalg.eigvalsh(H[f]).tolist() for f in range(4, H.shape[0])],
                       attained=float(rep.attained), gram=max(np.atleast_1d(rep.gram_residual)),
                       sizing=float(rep.sizing_prec), norm_rule=float(rep.norm_rule_prec))
        except Exception as e:                              # noqa: BLE001 -- a refusal is a row
            row.update(h_error=f"{type(e).__name__}: {str(e)[:200]}")
        out.append(row)
    return out


def run_grad(ci):
    sys.path.insert(0, str(HERE))
    from hadamard_hard_shapes import fields
    from chladni.eigenproblem import Eigenproblem
    from chladni.geometry import Polygon
    c = _shape_cases()[ci]
    verts = np.array([complex(*v) for v in c["verts"]])
    refs = c["refs"] or []
    n = min(sum(int(r["mult"]) for r in refs), 10) if refs else c["n"]
    center = complex(np.mean(verts))
    R = float(np.abs(verts - center).max())
    out = dict(id=c["id"], n_verts=len(verts), radius=R)
    for prec in GRAD_PRECS:
        t0 = time.time()
        try:
            rec = Eigenproblem(Polygon(verts, bc="dir")).solve(n, prec=prec)
            V = fields(rec.solver.domain, verts, center)
            out[f"{prec:.0e}"] = dict(levels=_grad_levels(rec, V, R), flags=list(rec.flags),
                                      t=time.time() - t0)
        except Exception as e:                              # noqa: BLE001
            out[f"{prec:.0e}"] = dict(failed=f"{type(e).__name__}: {str(e)[:300]}")
    return out


def tasks_grad():
    return [("tip", "grad", ci, None) for ci in range(len(_shape_cases()))]


def _grad_pairs(rows, ref_ident=1e-11, ref_rel=0.1):
    """(test level, reference level) at 1e-10 against 1e-12, matched by lam and mult, where the
    reference's own identity residual is at most `ref_ident`, or at most `ref_rel` times the test
    level's. The relative clause is needed: on the smoke triangle (T1.1-good) the 1e-12 solve's
    y-translation identity reads up to 1.9e-9 at an eigenvalue bar of 1.6e-14, so an absolute
    1e-11 keeps only the easy levels."""
    pairs, skipped = [], 0
    for r in rows:
        a, b = r.get("1e-10", {}), r.get("1e-12", {})
        if "levels" not in a or "levels" not in b:
            continue
        for L in a["levels"]:
            if "tr" not in L:
                continue
            M = [K for K in b["levels"] if "tr" in K and K["mult"] == L["mult"]
                 and abs(K["lam"] - L["lam"]) < 1e-7 * L["lam"]]
            if len(M) != 1 or max(M[0]["ident"]) > max(ref_ident, ref_rel * max(L["ident"])):
                skipped += 1
                continue
            K = M[0]
            ev = max(max(abs(x - y) for x, y in zip(e1, e2))
                     for e1, e2 in zip(L["eig"], K["eig"])) / L["lam"]
            pairs.append(dict(id=r["id"], lam=L["lam"], mult=L["mult"], e_v=ev,
                              r_id=max(L["ident"]), attained=L["attained"], gram=L["gram"],
                              mp_eps=L["mp_eps"], ref_id=max(K["ident"])))
    return pairs, skipped


def analyze_grad(d):
    rows = [json.load(open(f)) for f in sorted(Path(d).glob("grad_*.json"))]
    bad = [r for r in rows if "failed" in r]
    for r in bad:
        print("  CRASH", r.get("a"), r["failed"][-300:].replace("\n", " | "))
    for p in GRAD_PRECS:
        fails = [r["id"] for r in rows if "failed" in r.get(f"{p:.0e}", {})]
        print(f"prec {p:.0e}: {len(rows)} shapes, {len(fails)} failed solves {fails[:6]}")
    P, skipped = _grad_pairs(rows)
    print(f"\n{len(P)} levels compared at 1e-10 against 1e-12 ({skipped} skipped: no unique "
          f"match, or the reference's identities above max(1e-11, 0.1 x the test's)); "
          f"{len(_grad_pairs(rows, ref_rel=0)[0])} with the absolute 1e-11 alone")
    if not P:
        return
    e = np.array([x["e_v"] for x in P])
    for name, est in (("r_id", [x["r_id"] for x in P]), ("attained", [x["attained"] for x in P]),
                      ("max(r_id, attained)", [max(x["r_id"], x["attained"]) for x in P]),
                      ("max(r_id, mp_eps)", [max(x["r_id"], x["mp_eps"]) for x in P])):
        q = e / np.array(est)
        print(f"  e_v / {name:20s} median {np.median(q):6.2f}  p99 {np.percentile(q, 99):6.2f}  "
              f"max {q.max():6.2f}  k covering 99%: {np.percentile(q, 99):.2f}; "
              f"over 1: {int((q > 1).sum())}")
    print("  silent at 1e-10 (e_v > 1e-10, estimate <= 1e-10):",
          {name: int(((e > 1e-10) & (np.array(est) <= 1e-10)).sum())
           for name, est in (("r_id", [x["r_id"] for x in P]),
                             ("attained", [x["attained"] for x in P]))})
    q = e / np.array([x["r_id"] for x in P])
    print("  worst e_v / r_id:")
    for k in np.argsort(-q)[:8]:
        x = P[k]
        print(f"    {x['id']:30s} lam {x['lam']:9.3f} m {x['mult']} e_v {x['e_v']:.1e} "
              f"r_id {x['r_id']:.1e} attained {x['attained']:.1e} mp {x['mp_eps']:.1e} "
              f"ref_id {x['ref_id']:.1e}")


# --- plumbing -----------------------------------------------------------------------------------

def run(study, *args):
    import chladni
    if study == "bessel":
        r = run_bessel(args[0], int(args[1]))
    elif study == "eigs":
        r = run_eigs(int(args[0]), args[1])
    elif study == "starved":
        r = run_starved(args[0])
    elif study == "grad":
        r = run_grad(int(args[0]))
    else:
        raise SystemExit(f"unknown study {study!r}")
    r.update(study=study, chladni=chladni.__file__)
    return r


def task(i, srcs, out, table=None):
    impl, study, a, b = (table or tasks)()[i]
    rest = [q for q in os.environ.get("PYTHONPATH", "").split(os.pathsep) if q]
    src = Path(srcs[impl]).resolve()
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(src)] + rest))
    args = [sys.executable, __file__, "run", study, str(a)] + ([str(b)] if b is not None else [])
    p = subprocess.run(args, env=env, capture_output=True, text=True)
    Path(out).mkdir(parents=True, exist_ok=True)
    name = Path(out) / f"{study}_{impl}_{a}_{b}.json".replace(":", "_").replace("/", "_")
    if p.returncode != 0:
        name.write_text(json.dumps(dict(impl=impl, study=study, a=a, b=b,
                                        failed=p.stderr[-3000:])))
        sys.exit(p.returncode)
    r = json.loads(p.stdout.strip().splitlines()[-1])
    r.update(impl=impl, task=i)
    if not Path(r["chladni"]).resolve().is_relative_to(src):
        raise RuntimeError(f"task {i} imported chladni from {r['chladni']}, not {src}")
    name.write_text(json.dumps(r))


def analyze(d):
    rows = [json.load(open(f)) for f in sorted(Path(d).glob("*.json"))]
    crashed = [r for r in rows if "failed" in r and "study" in r and "levels" not in r
               and r.get("study") != "eigs"]
    print(f"{len(rows)} results, {len(crashed)} crashed tasks")
    for r in crashed:
        print("  CRASH", r["study"], r.get("a"), r.get("b"), r["failed"][-300:].replace("\n", " | "))

    print("\n== bessel: true error / reported bound (> 1 means the bound failed)")
    for kind in ("J", "Y", "debye"):
        b = [r for r in rows if r.get("study") == "bessel" and r.get("kind") == kind]
        if not b:
            continue
        worst = max(b, key=lambda r: r["max_ratio"])
        print(f"  {kind:6s} {sum(r['n'] for r in b):7d} cells: max ratio {worst['max_ratio']:.2f}, "
              f"{sum(r['n_over'] for r in b)} over 1; worst true rel err "
              f"{max(r['max_rel_err_true'] for r in b):.1e}; derivative max ratio "
              f"{max(r['deriv_max_ratio'] for r in b):.1f} (p99 "
              f"{max(r['deriv_p99_ratio'] for r in b):.1f}); scipy {worst['scipy']} {worst['platform']}")
        for w in worst["worst"][:4]:
            print(f"      nu {w['nu']:8.2f} x {w['x']:8.2f} err {w['err']:.1e} rel_err "
                  f"{w['rel_err']:.1e} ratio {w['ratio']:.2f} log|J| {w['log_abs']:.1f} "
                  f"{w.get('branch') or ''}")

    E = [r for r in rows if r.get("study") == "eigs"]
    if E:
        print("\n== eigs: old (5a02def) vs new (9de388a)")
        by = {}
        for r in E:
            by.setdefault((r["id"], r["prec"]), {})[r["impl"]] = r
        for prec in (1e-10, 1e-12):
            for impl in ("old", "new"):
                rs = [v[impl] for (i, p), v in by.items() if p == prec and impl in v]
                fail = [r for r in rs if "failed" in r]
                lv = [L for r in rs if "levels" in r for L in r["levels"]]
                refd = [L for L in lv if "err" in L]
                over = [L for L in refd if L["err"] > max(L["bound"], prec * L["lam"]) + L["ref_bound"]]
                metlie = [L for L in refd if L["met"] and L["err"] > prec * L["lam"] + L["ref_bound"]]
                print(f"  prec {prec:.0e} {impl}: {len(rs)} shapes, {len(fail)} failed solves, "
                      f"{len(lv)} levels ({len(refd)} with a reference): bar exceeded on "
                      f"{len(over)}, met-but-wrong {len(metlie)}, met {sum(L['met'] for L in lv)}")
            both = [(v["old"], v["new"]) for (i, p), v in by.items() if p == prec
                    and "levels" in v.get("old", {}) and "levels" in v.get("new", {})]
            shifts = [abs(a["lam"] - b["lam"]) / b["lam"] for o, n in both
                      for a, b in zip(o["levels"], n["levels"])]
            if shifts:
                print(f"    old -> new lam shift: median {np.median(shifts):.1e}, "
                      f"max {max(shifts):.1e} over {len(shifts)} levels")
        H = [L for r in E if r["impl"] == "new" and "levels" in r for L in r["levels"]
             if "ident" in L]
        if H:
            true = [max(L["ident"], L["dil"]) for L in H]
            att = [L["attained"] for L in H]
            rat = np.array(true) / np.array(att)
            print(f"\n== hadamard (new): {len(H)} levels; attained under-reports on "
                  f"{int((rat > 1).sum())}, worst x{rat.max():.2f}; p90 x{np.percentile(rat, 90):.2f}")
            for k in (1, 2, 3, 5):
                print(f"    k*attained covers {int((rat <= k).sum())}/{len(H)}")
        fails = {}
        for r in E:
            if "failed" in r:
                fails.setdefault(r["failed"][:70], []).append((r["id"], r["prec"], r["impl"]))
        if fails:
            print("\n== failed solves (the notch census)")
            for why, who in fails.items():
                print(f"  {len(who):3d} x {why}")
                print("       ", sorted({(i, p) for i, p, _ in who})[:8])

    S = [r for r in rows if r.get("study") == "starved"]
    if S:
        print("\n== starved: gram_residual against the interior oracle")
        pairs = [(L["gram_residual"], L["oracle_defect"], L["lam_err"]) for r in S
                 for row in r["rows"] for L in row.get("levels", [])]
        g, o, e = map(np.array, zip(*pairs))
        big = o > 1e-8
        print(f"  {len(pairs)} levels; oracle defect > 1e-8 on {int(big.sum())}; there, "
              f"oracle / gram_residual: median {np.median(o[big] / g[big]) if big.any() else np.nan:.2g}, "
              f"max {np.max(o[big] / g[big]) if big.any() else np.nan:.2g}")
        for r in S:
            print(f"  {r['id']:22s}", " ".join(
                f"{row['factor']}:{row['n_cols']}:" + (
                    f"g{max([L['gram_residual'] for L in row['levels']] or [np.nan]):.0e}/"
                    f"o{max([L['oracle_defect'] for L in row['levels']] or [np.nan]):.0e}"
                    if not row.get("error") else "ERR") for row in r["rows"]))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        print(json.dumps(run(*sys.argv[2:])))
    elif cmd == "task":
        a = sys.argv[3:]
        opts = {a[k].lstrip("-"): a[k + 1] for k in range(0, len(a), 2)}
        table = tasks_grad if opts.get("table") == "grad" else None
        task(int(sys.argv[2]), dict(tip=opts["tip"], old=opts.get("old", opts["tip"]),
                                    new=opts.get("new", opts["tip"])), opts["out"], table)
    elif cmd == "analyze":
        analyze(sys.argv[2])
    elif cmd == "analyze_grad":
        analyze_grad(sys.argv[2])
    elif cmd == "ntasks":
        print(len(tasks_grad() if sys.argv[2:] == ["grad"] else tasks()))
    else:
        raise SystemExit(__doc__)
