"""Phase 4 step 4: is reporting a shortfall EARLY worth building?  (2026-10-11)

Step 3 raises `PrecisionShortfall` after the solve. Two ways to say it sooner were proposed
(`docs/design/douse_requirements.md`, "Precision per level"); this measures what each could buy
before either is built.

    early abort   stop at the first refined level that misses `prec`. The most it can save is
                  the work done after that moment: the remaining refines and stage 3, and the
                  Moler-Payne certification.
    pre-scan      predict unmet regions from stage 1's samples, before any refine. Doubtful in
                  principle: a sample away from a well bounds the floor from ABOVE, and a
                  shortfall needs a bound from below.

Instrumented from outside (`src/` untouched): `search.refine` is wrapped to log, per call, the
evaluation count and time when it returns and whether the level it located misses `prec`
(`delta_res > prec*lam`, i.e. margin < 2); at its FIRST call it snapshots stage 1's samples.
`search.solve` is wrapped to time the end of the search, before certification.

Cost is counted in tension evaluations (deterministic; they dominate) and in wall time.
Douse benchmark, bases D and A, prec 1e-10, reference windows.

    .venv/bin/python notes/early_shortfall.py [WORKERS]
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
sys.path[:0] = [os.path.join(ROOT, "tests"), os.path.join(ROOT, "benchmarks", "douse"), HERE]
warnings.simplefilter("ignore")
PREC = 1e-10
OUT = os.path.join(HERE, "early_shortfall.json")


def run_one(args):
    import make_references as MR
    import scoreboard  # noqa: F401 -- registers basis "D"
    from chladni.mps import MPSEigensolver
    from chladni.mps import search as S

    case, window, basis = args
    log = dict(refines=[], stage1=None)
    t0 = time.perf_counter()
    real_refine, real_solve = S.refine, S.solve

    def refine(obj, lo, hi, **kw):
        if log["stage1"] is None:
            xs, sig = obj.samples
            log["stage1"] = dict(xs=list(map(float, xs)), sig=list(map(float, sig)),
                                 evals=obj.n_evals, t=time.perf_counter() - t0)
        r = real_refine(obj, lo, hi, **kw)
        short = bool(r.accepted and np.isfinite(r.delta_res) and r.delta_res > PREC * abs(r.lam))
        log["refines"].append(dict(evals=obj.n_evals, t=time.perf_counter() - t0, short=short,
                                   lam=float(r.lam), accepted=bool(r.accepted)))
        return r

    def solve(obj, **kw):
        res = real_solve(obj, **kw)
        log["solve_end"] = dict(evals=obj.n_evals, t=time.perf_counter() - t0)
        return res

    S.refine, S.solve = refine, solve
    try:
        dom = MR.polygon(case["vertices"])
        b = MR.basis_for(dom, basis, np.sqrt(window[1]))
        n = len(b)
        s = MPSEigensolver(dom, basis=b, n_bdry=3 * n, n_int=n, prec=PREC, rng=0, reg="svd")
        res = s.search_interval(*window)
    except Exception as e:  # noqa: BLE001
        return dict(id=case["id"], basis=basis, error=repr(e)[:200])
    finally:
        S.refine, S.solve = real_refine, real_solve
    d = res.diagnostics
    return dict(id=case["id"], basis=basis, wall=time.perf_counter() - t0, log=log,
                levels=[dict(lam=float(L.lam), margin=float(L.margin), met=bool(L.met(PREC)),
                             bracket=[float(x) for x in L.bracket]) for L in res.levels],
                audit=d.count_audit.verdict if d.count_audit is not None else None,
                n_uncertain=int(d.n_uncertain), declined=bool(d.declined))


def report(runs):
    saved_e, saved_t, cert_t, n_short = [], [], [], 0
    for r in runs:
        if "error" in r or r["log"].get("solve_end") is None:
            continue
        lg = r["log"]
        cert_t.append(1 - lg["solve_end"]["t"] / r["wall"])
        first = next((x for x in lg["refines"] if x["short"]), None)
        if first is None:
            continue
        n_short += 1
        saved_e.append(1 - first["evals"] / lg["solve_end"]["evals"])
        saved_t.append(1 - first["t"] / r["wall"])
    saved_e, saved_t = np.array(saved_e), np.array(saved_t)
    print(f"{len(runs)} solves; {n_short} have a refined level that misses prec")
    print(f"  certification's share of wall time, all solves: median {np.median(cert_t):.0%}")
    print(f"  EARLY ABORT at the first such level would save, of the solve:")
    print(f"    evaluations: median {np.median(saved_e):.0%}, quartiles "
          f"{np.percentile(saved_e, 25):.0%}-{np.percentile(saved_e, 75):.0%}")
    print(f"    wall time (certification included): median {np.median(saved_t):.0%}, quartiles "
          f"{np.percentile(saved_t, 25):.0%}-{np.percentile(saved_t, 75):.0%}")
    walls = np.array([r["wall"] for r in runs if "error" not in r])
    print(f"  wall per solve: median {np.median(walls):.1f} s, 90th {np.percentile(walls, 90):.1f} s")

    # Pre-scan: does stage 1's lowest sample inside a level's bracket rank levels by margin?
    from scipy.stats import spearmanr
    pred, marg, unmet = [], [], []
    for r in runs:
        if "error" in r or r["log"]["stage1"] is None:
            continue
        xs = np.array(r["log"]["stage1"]["xs"])
        sg = np.array(r["log"]["stage1"]["sig"])
        for L in r["levels"]:
            if len(L["bracket"]) != 2 or not np.isfinite(L["margin"]):
                continue
            inside = (xs >= L["bracket"][0]) & (xs <= L["bracket"][1])
            if not inside.any():
                continue
            pred.append(sg[inside].min())
            marg.append(L["margin"])
            unmet.append(not L["met"])
    pred, marg, unmet = np.array(pred), np.array(marg), np.array(unmet)
    rho = spearmanr(pred, marg).correlation
    # AUC of "larger lowest sample => unmet"
    pos, neg = pred[unmet], pred[~unmet]
    auc = np.mean([[p > q for q in neg] for p in pos]) if len(pos) and len(neg) else float("nan")
    print(f"\nPRE-SCAN: {len(pred)} levels with stage-1 samples inside their bracket "
          f"({unmet.sum()} unmet)")
    print(f"  Spearman(lowest stage-1 sample, final margin) = {rho:.2f}")
    print(f"  AUC, lowest sample predicting 'unmet': {auc:.2f}  (0.5 = no information)")
    if len(pos) and len(neg):
        for q in (0.5, 0.9):
            th = np.quantile(neg, q)
            print(f"  flag where the lowest sample exceeds the met levels' {q:.0%} quantile "
                  f"({th:.2e}): catches {np.mean(pos > th):.0%} of unmet levels, "
                  f"false flags {np.mean(neg > th):.0%} of met")


def main(workers=8):
    import make_references as MR
    cases = {c["id"]: c for c in MR.load_cases()}
    refs = json.load(open(os.path.join(ROOT, "benchmarks", "douse", "references.json")))["cases"]
    jobs = [(cases[r["id"]], r["window"], b) for r in refs for b in ("D", "A")]
    with ProcessPoolExecutor(workers) as ex:
        runs = list(ex.map(run_one, jobs))
    for r in runs:
        if "log" in r:
            r["log"]["solve_end"] = r["log"].get("solve_end")
    json.dump(dict(prec=PREC, runs=runs), open(OUT, "w"), indent=1)
    report(runs)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
