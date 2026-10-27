"""Stage-policy step 4 gate on the douse benchmark (2026-10-27): basis "D", the five jittered
windows per case (`benchmarks/douse/scoreboard.py`), each case solved with the shipped threshold
path and with `stable_rank=True`. The same runner and scoring as the baseline session's
`notes/stage_policy_baseline.py douse_e1` (branch claude/fervent-wiles-d26d63), kept self-contained
here so neither writes into the other's folder.

usage: python notes/stage_policy_step4_douse.py PREC [WORKERS]
Writes notes/stage_policy_step4_douse_<prec>.json and prints the tally per mode.
"""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, "tests"), os.path.join(ROOT, "benchmarks", "douse"), HERE]


def run(args):
    import warnings
    warnings.filterwarnings("ignore")
    import make_references as MR
    import scoreboard  # noqa: F401 -- registers basis "D"
    from chladni.mps import MPSEigensolver
    case, window, jit, prec, stable = args
    lo, hi = window[0] * (1 + jit[0]), window[1] * (1 + jit[1])
    dom = MR.polygon(case["vertices"])
    b = MR.basis_for(dom, "D", np.sqrt(hi))
    n = len(b)
    t0 = time.perf_counter()
    try:
        s = MPSEigensolver(dom, basis=b, n_bdry=3 * n, n_int=n, prec=prec, rng=0)
        res = s.search_interval(lo, hi, stable_rank=stable)
    except Exception as e:  # noqa: BLE001 -- a crash is a scored outcome
        return dict(id=case["id"], jit=list(jit), stable=stable, error=repr(e)[:300],
                    wall=time.perf_counter() - t0)
    d = res.diagnostics
    return dict(id=case["id"], jit=list(jit), stable=stable, wall=time.perf_counter() - t0,
                declined=bool(d.declined), unmet=len(d.unmet), n_uncertain=int(d.n_uncertain),
                evals=int(d.evaluations), rerank_evals=int(d.rerank_evals),
                n_reranked=int(d.n_reranked), n_rerank_refused=int(d.n_rerank_refused),
                levels=[dict(lam=float(L.lam), mult=int(L.mult), bound=float(L.bound),
                             rank=L.rank, count_flag=bool(L.count_uncertain or L.grouping_ambiguous),
                             met=bool(L.met(prec))) for L in res.levels])


def report(runs, prec, refs):
    import regularization_1e10 as R
    import scoreboard as SB
    R.PREC = prec
    for stable in (False, True):
        rr = [r for r in runs if r["stable"] == stable]
        tally = dict(ok=0, flagged=0, silent=0, declined=0, crash=0, unscored=0)
        rel, tight, sig = [], [], {}
        for r in rr:
            if "error" in r:
                tally["crash"] += 1
                sig.setdefault(r["id"], []).append("x")
                continue
            sc = SB.score_chladni(r, refs[r["id"]])
            o = sc["outcome"]
            key = ("unscored" if o == "unscored" else "declined" if o == "declined" else "ok"
                   if o == "ok" else "silent" if sc.get("silent") else "flagged")
            tally[key] += 1
            sig.setdefault(r["id"], []).append(key[0])
            if o != "unscored":
                tight += R._tightness(r, refs[r["id"]])
                rel += [L["bound"] / L["lam"] for L in r["levels"] if np.isfinite(L["bound"])]
        ev = [r["evals"] for r in rr if "evals" in r]
        rer = [r.get("rerank_evals", 0) for r in rr if "evals" in r]
        roots = [r.get("n_reranked", 0) for r in rr if "evals" in r]
        refused = [r.get("n_rerank_refused", 0) for r in rr if "evals" in r]
        stab = sum(len(set(v)) == 1 for v in sig.values())
        print(f"{'stable' if stable else 'threshold':9s} prec {prec:g}: {tally}  stable-across-jitter "
              f"{stab}/{len(sig)}; bound/err median {np.median(tight):.1f}; rel bound median "
              f"{np.median(rel):.2e} p90 {np.percentile(rel, 90):.2e}; evals total {sum(ev)} "
              f"(rerank {sum(rer)} over {sum(roots)} roots, {sum(refused)} refused)")
        for cid, v in sorted(sig.items()):
            if set(v) != {"o"}:
                print(f"     {cid:26s} {''.join(v)}")


if __name__ == "__main__":
    prec = float(sys.argv[1])
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    import make_references as MR
    import scoreboard as SB
    cases = {c["id"]: c for c in MR.load_cases()}
    refs = {r["id"]: r for r in json.load(open(os.path.join(ROOT, "benchmarks", "douse",
                                                            "references.json")))["cases"]}
    jobs = [(cases[i], refs[i]["window"], j, prec, st) for st in (False, True)
            for i in refs for j in SB.JITTERS]
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        runs = list(ex.map(run, jobs))
    json.dump(dict(date="2026-10-27", prec=prec, wall_total=time.time() - t0, runs=runs),
              open(os.path.join(HERE, f"stage_policy_step4_douse_{prec:g}.json"), "w"), indent=1)
    report(runs, prec, refs)
