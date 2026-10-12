"""Phase 5 step 4: the black-box path on the douse benchmark.  (2026-10-12)

`Eigenproblem(dom).solve(n, prec)` on every douse benchmark case, with nothing about the basis
supplied. `n` is the number of trusted reference eigenvalues (counted with multiplicity), at prec
1e-10 and 1e-8. Each solve is scored against the references:

    correct   every returned level below the trusted top matches the references, by Phase 3's
              resolution groups (`notes/phase3_rescore.py`), and no MET level is off a TIGHT
              reference value by more than prec*lam
    flagged   `spec.flags` is non-empty

giving four outcomes: CLEAN (correct, unflagged), FLAGGED-OK (correct, flagged), FLAGGED-WRONG
(wrong, flagged -- honest), and SILENT (wrong, unflagged -- the failure that must be zero).
Also recorded: the rung the planner stopped at, columns, and wall time with planning included.

    .venv/bin/python notes/phase5_benchmark.py [WORKERS]
"""
import json
import os
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, "benchmarks", "douse"), HERE]
warnings.simplefilter("ignore")

PRECS = (1e-10, 1e-8)
OUT = os.path.join(HERE, "phase5_benchmark.json")


def run_one(args):
    import make_references as MR
    import phase3_rescore as P3
    from chladni.eigenproblem import Eigenproblem
    case, ref, prec = args
    trusted, top = P3.trusted_and_top(ref)
    if not trusted:
        return dict(id=case["id"], prec=prec, skipped="no trusted reference levels")
    n = sum(T["mult"] for T in trusted)
    dom = MR.polygon(case["vertices"])
    t0 = time.perf_counter()
    try:
        spec = Eigenproblem(dom).solve(n, prec=prec)
    except Exception as e:  # noqa: BLE001
        return dict(id=case["id"], prec=prec, n=n, error=repr(e)[:300],
                    wall=time.perf_counter() - t0)
    return dict(id=case["id"], prec=prec, n=n, wall=time.perf_counter() - t0,
                rung=spec.plan.rung if spec.plan is not None else None,
                n_cols=spec.plan.n_cols if spec.plan is not None else None,
                flags=list(spec.flags), declined=bool(spec.declined),
                audit=spec.count_audit.verdict if spec.count_audit is not None else None,
                levels=[dict(lam=L.lam, mult=L.mult, bound=L.bound, met=L.met,
                             count_flag=L.count_uncertain) for L in spec.levels])


def score(r, ref):
    import phase3_rescore as P3
    trusted, top = P3.trusted_and_top(ref)
    run = dict(prec=r["prec"], declined=r["declined"], levels=r["levels"], audit=r["audit"],
               n_uncertain=0)
    outcome, drops = P3.new_score(run, ref)
    tl = np.array([T["lam"] for T in trusted])
    tb = np.array([T["bound"] if np.isfinite(T.get("bound", np.nan)) else np.inf for T in trusted])
    harm = 0
    for L in r["levels"]:
        if not L["met"] or L["lam"] >= top:
            continue
        j = int(np.argmin(np.abs(tl - L["lam"])))
        err = abs(tl[j] - L["lam"])
        if tb[j] <= 0.1 * r["prec"] * L["lam"] and err <= 1e-4 * L["lam"] \
                and err > r["prec"] * L["lam"] + tb[j]:
            harm += 1
    correct = outcome in ("ok",) and not harm
    flagged = bool(r["flags"])
    return ("CLEAN" if correct and not flagged else "FLAGGED-OK" if correct
            else "FLAGGED-WRONG" if flagged else "SILENT"), outcome, drops, harm


def main(workers=8):
    import make_references as MR
    cases = {c["id"]: c for c in MR.load_cases()}
    refs = {r["id"]: r for r in json.load(open(os.path.join(ROOT, "benchmarks", "douse",
                                                            "references.json")))["cases"]}
    jobs = [(cases[i], refs[i], p) for i in refs for p in PRECS]
    with ProcessPoolExecutor(workers) as ex:
        runs = list(ex.map(run_one, jobs))
    json.dump(dict(runs=runs), open(OUT, "w"), indent=1)
    report(runs, refs)


def report(runs=None, refs=None):
    if runs is None:
        runs = json.load(open(OUT))["runs"]
        refs = {r["id"]: r for r in json.load(open(os.path.join(ROOT, "benchmarks", "douse",
                                                                "references.json")))["cases"]}
    for p in PRECS:
        rr = [r for r in runs if r["prec"] == p and "skipped" not in r]
        tally, rungs, silent = Counter(), Counter(), []
        for r in rr:
            if "error" in r:
                tally["ERROR"] += 1
                silent.append((r["id"], "ERROR", r["error"][:120]))
                continue
            o, outcome, drops, harm = score(r, refs[r["id"]])
            tally[o] += 1
            rungs[r["rung"]] += 1
            if o == "SILENT" or o == "FLAGGED-WRONG":
                silent.append((r["id"], o, outcome, drops, harm, r["flags"][:2]))
        walls = [r["wall"] for r in rr if "wall" in r]
        print(f"\n=== prec {p:g}: {len(rr)} cases ===")
        print("  " + "  ".join(f"{k} {v}" for k, v in sorted(tally.items())))
        print(f"  rung reached: {dict(sorted(rungs.items()))}")
        print(f"  wall (planning + solve): median {np.median(walls):.0f} s, "
              f"90th {np.percentile(walls, 90):.0f} s, max {max(walls):.0f} s")
        for x in silent:
            print("   ", x)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report()
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
