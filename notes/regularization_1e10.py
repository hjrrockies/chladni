"""Phase 2: is truncation good enough at prec = 1e-10, or does a penalty earn the default?  (2026-10-10)

Three regularizations, everything else equal:

    truncation   reg="svd", the threshold rank at its derived rtol -- what `reg="auto"` gives at
                 1e-10 today (PENALTY_PREC_MIN = 1e-8)
    penalty      reg="tikhonov_qr" at the auto rule's tau = C_PEN * prec, i.e. `auto` extended to
                 1e-10; searched on the penalized tensions sigma_tau
    posterior    the same pencil at a FIXED tau = 30 eps sqrt(n) (`notes/posterior_penalty.md`'s
                 C = 30), searched on the unpenalized Rayleigh-Ritz tensions on the 8 lowest
                 penalized vectors (`notes/posterior_penalty.py`), certified as usual

Experiments (each writes notes/regularization_1e10_<exp>.json):

    e1   the douse benchmark (benchmarks/douse), basis "D", at 1e-10, on the reference window and
         four jittered ones (`scoreboard.JITTERS`): count right / flagged / silent, stability,
         bound coverage and tightness (bound / true error), wall time
    e2   the 50/65 pi^2 cluster grids (`notes/posterior_penalty.py`'s) plus the 170 pi^2 band-0.3
         cluster, at 1e-10, under ulp-jitter seeds 0..S (`tests/ulp_jitter.py`): right /
         declined / flagged / silent
    e3   census of the truncation hazard along real 1e-10 solves on douse shapes: the ratio of
         the first DROPPED singular value to the cutoff at every evaluated point

The decision rule, written down before running (plan, 2026-10-09): switch the 1e-10 default to a
penalty mode only if it improves count correctness or stability on the benchmark by a clear
margin, at no more than 2x wall time, with no new silent losses. Otherwise keep truncation.

    .venv/bin/python notes/regularization_1e10.py e1 [WORKERS]
    .venv/bin/python notes/regularization_1e10.py e2 [SEEDS] [WORKERS]
    .venv/bin/python notes/regularization_1e10.py e3
"""
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "douse"))
warnings.simplefilter("ignore")

# CHLADNI_REG_PREC / CHLADNI_REG_VARIANTS rerun e1 at another rung (env, so pool workers see it).
PREC = float(os.environ.get("CHLADNI_REG_PREC", "1e-10"))
VARIANTS = tuple(os.environ.get("CHLADNI_REG_VARIANTS", "truncation,penalty,posterior").split(","))
_SUFFIX = "" if PREC == 1e-10 else f"_prec{PREC:g}"
EPS = np.finfo(float).eps


def posterior_tensions(q, lam, k=8):
    """`notes/posterior_penalty.py`'s, verbatim in effect."""
    import scipy.linalg as la
    lam = float(lam)
    pen = q.tensions(lam)
    k = min(k, len(pen))
    B, I = q.pencil(lam)
    X = q.coef(lam, k)
    _, R = la.qr(I @ X, mode="economic")
    post = np.sort(la.svdvals(la.solve_triangular(R, (B @ X).T, trans="T").T))
    return np.concatenate([post, pen[k:]])


def make_solver(dom, basis, variant, prec=PREC, rng=0):
    from chladni.mps import MPSEigensolver
    n = len(basis)
    kw = dict(basis=basis, n_bdry=3 * n, n_int=n, prec=prec, rng=rng)
    if variant == "truncation":
        return MPSEigensolver(dom, reg="svd", **kw)
    if variant == "penalty":
        return MPSEigensolver(dom, reg="tikhonov_qr", **kw)
    return MPSEigensolver(dom, reg="tikhonov_qr", rtol=30 * EPS, **kw)   # tau = 30 eps sqrt(n)


def search(s, lo, hi, variant, prec=PREC):
    """`search_interval`, with the posterior variant reading posterior tensions."""
    if variant != "posterior":
        return s.search_interval(lo, hi)
    from chladni import asymp
    from chladni.mps.objective import Objective
    q = s._pencil
    obj = Objective(lambda x: posterior_tensions(q, x), float(lo), float(hi), variable="lam",
                    prec=prec, expected_count=lambda a, b: asymp.expected_count(a, b, s.domain))
    return s.search_interval(lo, hi, objective=obj)


# --- e1: the douse benchmark --------------------------------------------------------------------

def e1_run(args):
    import make_references as MR
    import scoreboard  # noqa: F401 -- registers basis "D" (1.0x lappy's per-corner sizing)
    case, window, variant, jit = args
    lo, hi = window[0] * (1 + jit[0]), window[1] * (1 + jit[1])
    dom = MR.polygon(case["vertices"])
    b = MR.basis_for(dom, "D", np.sqrt(hi))
    t0 = time.perf_counter()
    try:
        s = make_solver(dom, b, variant)
        res = search(s, lo, hi, variant)
    except Exception as e:  # noqa: BLE001 -- a crash is a scored outcome
        return dict(id=case["id"], variant=variant, jit=jit, error=repr(e)[:300],
                    wall=time.perf_counter() - t0)
    d = res.diagnostics
    return dict(id=case["id"], variant=variant, jit=list(jit), wall=time.perf_counter() - t0,
                declined=bool(d.declined), unmet=len(d.unmet), n_uncertain=int(d.n_uncertain),
                levels=[dict(lam=float(L.lam), mult=int(L.mult), bound=float(L.bound),
                             count_flag=bool(L.count_uncertain or L.grouping_ambiguous),
                             met=bool(L.met(PREC))) for L in res.levels])


def e1(workers=8):
    import make_references as MR
    import scoreboard as SB
    cases = {c["id"]: c for c in MR.load_cases()}
    refs = {r["id"]: r for r in json.load(open(os.path.join(ROOT, "benchmarks", "douse",
                                                            "references.json")))["cases"]}
    jobs = [(cases[i], refs[i]["window"], v, j) for i in refs for v in VARIANTS
            for j in SB.JITTERS]
    with ProcessPoolExecutor(workers) as ex:
        runs = list(ex.map(e1_run, jobs))
    json.dump(dict(prec=PREC, runs=runs), open(os.path.join(HERE, f"regularization_1e10_e1{_SUFFIX}.json"),
                                                "w"), indent=1)
    summarize_e1(runs, refs)


def summarize_e1(runs, refs):
    import scoreboard as SB
    print(f"e1: douse benchmark at prec {PREC:g}, basis D, {len(SB.JITTERS)} windows per case\n")
    print(f"{'variant':11s} {'scored':>6s} {'ok':>4s} {'flagged':>7s} {'SILENT':>6s} "
          f"{'declined':>8s} {'crash':>5s} {'stable':>9s} {'uncov':>5s} {'tight(med)':>10s} "
          f"{'wall med':>8s} {'wall tot':>8s}")
    for v in VARIANTS:
        rows = [r for r in runs if r["variant"] == v]
        per_case = {}
        for r in rows:
            per_case.setdefault(r["id"], []).append(r)
        ok = flagged = silent = declined = crash = scored = stable = unc = 0
        tight = []
        for cid, rr in per_case.items():
            ref = refs[cid]
            outs = []
            for r in rr:
                if "error" in r:
                    crash += 1
                    outs.append("crash")
                    continue
                sc = SB.score_chladni(r, ref)
                outs.append(str(sc.get("sig", sc["outcome"])))
                if sc["outcome"] == "unscored":
                    continue
                scored += 1
                if sc["outcome"] == "declined":
                    declined += 1
                elif sc["outcome"] == "ok":
                    ok += 1
                elif sc.get("silent"):
                    silent += 1
                else:
                    flagged += 1
                unc += sc.get("uncovered", 0)
                tight += _tightness(r, ref)
            stable += len(set(outs)) == 1
        walls = [r["wall"] for r in rows]
        print(f"{v:11s} {scored:6d} {ok:4d} {flagged:7d} {silent:6d} {declined:8d} {crash:5d} "
              f"{stable:4d}/{len(per_case):<4d} {unc:5d} "
              f"{np.median(tight) if tight else float('nan'):10.1f} "
              f"{np.median(walls):7.1f}s {sum(walls) / 60:7.1f}m")


def _tightness(run, ref):
    """bound / true error for levels matching a trusted reference level (error > 0)."""
    trusted = [L for L in ref["levels"] if L["status"] != "provisional"]
    out = []
    for L in run["levels"]:
        if not trusted:
            break
        T = min(trusted, key=lambda T: abs(T["lam"] - L["lam"]))
        err = abs(T["lam"] - L["lam"])
        if err <= 1e-6 * T["lam"] and err > 0 and np.isfinite(L["bound"]):
            out.append(L["bound"] / err)
    return out


# --- e2: cluster grids under ulp jitter ---------------------------------------------------------

CLUSTERS = ((50, 3), (65, 4))
BANDS = (0.3, 0.5, 0.8, 1.0, 1.3, 1.6, 2.0, 2.6, 3.0, 4.0, 6.0)


def e2_run(args):
    seed, variant, N, m, band = args
    import ulp_jitter
    # Workers are reused across jobs: a seed installed by one job must not leak into the next.
    if seed:
        ulp_jitter.install(seed, 4.0)
    else:
        ulp_jitter.uninstall()
    import bracket_score as bs
    import rect_suite as rs
    from basis_sufficiency import sufficient
    from chladni.mps import MPSPencil
    from chladni.mps.objective import Objective
    from chladni.mps.search import solve
    (case,) = rs.split_degeneracy_cases(PREC, bands=(band,), levels={N: m})
    if not sufficient(case, PREC).multiplicity:
        return dict(seed=seed, variant=variant, N=N, band=band, outcome="unexercisable")
    p = case.pencil
    if variant == "truncation":
        q = MPSPencil(p.bdry, p.interior, reg="svd", prec=PREC)
    elif variant == "penalty":
        q = MPSPencil(p.bdry, p.interior, reg="tikhonov_qr", prec=PREC)
    else:
        q = MPSPencil(p.bdry, p.interior, reg="tikhonov_qr", prec=PREC, rtol=30 * EPS)
    f = ((lambda x: posterior_tensions(q, x)) if variant == "posterior"
         else (lambda x: q.tensions(float(x))))
    base = bs.objective_for(case, PREC)
    obj = Objective(f, base.lo, base.hi, prec=PREC, variable=case.variable,
                    expected_count=getattr(base, "expected_count", None))
    t0 = time.perf_counter()
    try:
        res = solve(obj, prec=PREC)
    except Exception as e:  # noqa: BLE001
        return dict(seed=seed, variant=variant, N=N, band=band, outcome="crash", detail=repr(e))
    total = sum(x.mult for x in res.levels)
    m = int(len(case.truth))        # the cluster's own count (170 pi^2 is 8-fold, not the m asked)
    flagged = (not res.diagnostics.certified_counts or res.diagnostics.n_ambiguous > 0
               or any(x.count_uncertain or x.grouping_ambiguous for x in res.levels))
    out = ("declined" if res.declined else "right" if total == m
           else "flagged" if flagged else "SILENT")
    return dict(seed=seed, variant=variant, N=N, band=band, outcome=out, total=total, m=m,
                evals=obj.n_evals, wall=time.perf_counter() - t0)


def e2(seeds=4, workers=8):
    cells = [(N, m, b) for N, m in CLUSTERS for b in BANDS] + [(170, 4, 0.3)]
    jobs = [(s, v, N, m, b) for s in range(seeds + 1) for v in VARIANTS for N, m, b in cells]
    with ProcessPoolExecutor(workers) as ex:
        runs = list(ex.map(e2_run, jobs))
    json.dump(dict(prec=PREC, seeds=seeds, runs=runs),
              open(os.path.join(HERE, f"regularization_1e10_e2{_SUFFIX}.json"), "w"), indent=1)
    print(f"e2: cluster grids + 170 pi^2 band 0.3 at prec {PREC:g}, seeds 0..{seeds}\n")
    for v in VARIANTS:
        rr = [r for r in runs if r["variant"] == v and r["outcome"] != "unexercisable"]
        tally = {k: sum(r["outcome"] == k for r in rr) for k in
                 ("right", "flagged", "SILENT", "declined", "crash")}
        evals = [r["evals"] for r in rr if "evals" in r]
        walls = [r["wall"] for r in rr if "wall" in r]
        c170 = [r["outcome"] for r in rr if r["N"] == 170]
        print(f"{v:11s} {len(rr):4d} cells x seeds: {tally}  median evals "
              f"{np.median(evals) if evals else 0:.0f}  wall {sum(walls) / 60:.1f}m  "
              f"170pi^2: {c170}")
        silent = [(r["seed"], r["N"], r["band"], r.get("total"), r.get("m")) for r in rr
                  if r["outcome"] in ("SILENT", "crash")]
        if silent:
            print(f"            silent/crash: {silent}")


# --- e3: census of the truncation hazard --------------------------------------------------------

def e3():
    import make_references as MR
    import scoreboard  # noqa: F401 -- registers basis "D"
    import scipy.linalg as la
    cases = {c["id"]: c for c in MR.load_cases()}
    refs = {r["id"]: r for r in json.load(open(os.path.join(ROOT, "benchmarks", "douse",
                                                            "references.json")))["cases"]}
    ratios, near = [], {}
    for cid in list(refs)[::4]:                 # every 4th case: ~10 shapes
        lo, hi = refs[cid]["window"]
        dom = MR.polygon(cases[cid]["vertices"])
        b = MR.basis_for(dom, "D", np.sqrt(hi))
        s = make_solver(dom, b, "truncation")
        res = s.search_interval(lo, hi)
        obj = s.objective(lo, hi)
        lams = np.asarray(obj.samples[0], dtype=float)
        p = s._pencil
        rr = []
        for lam in lams[:: max(1, len(lams) // 60)]:
            B, I = p.pencil(float(lam))
            sv = la.svdvals(np.vstack([B, I]))
            cut = p.rtol * sv[0]
            dropped = sv[sv <= cut]
            if dropped.size:
                rr.append(float(dropped[0] / cut))
        ratios += rr
        near[cid] = (len(rr), int(np.sum(np.asarray(rr) >= 0.5)), int(np.sum(np.asarray(rr) >= 0.9)))
        print(f"  {cid:22s} {len(rr):3d} points; s_(r+1)/cutoff >= 0.5 at {near[cid][1]}, "
              f">= 0.9 at {near[cid][2]}", flush=True)
    r = np.asarray(ratios)
    print(f"\ne3: {r.size} evaluated points; first dropped singular value / cutoff: median "
          f"{np.median(r):.2f}; >= 0.5 at {np.mean(r >= 0.5):.0%}; >= 0.9 at {np.mean(r >= 0.9):.0%}")
    json.dump(dict(prec=PREC, ratios=ratios, per_case=near),
              open(os.path.join(HERE, f"regularization_1e10_e3{_SUFFIX}.json"), "w"), indent=1)


if __name__ == "__main__":
    exp = sys.argv[1]
    if exp == "e1":
        e1(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
    elif exp == "e2":
        e2(int(sys.argv[2]) if len(sys.argv) > 2 else 4, int(sys.argv[3]) if len(sys.argv) > 3 else 8)
    elif exp == "e3":
        e3()
