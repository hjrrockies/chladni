"""Would a margin rule catch truncation's wrong counts on the douse benchmark?  (2026-10-11)

Phase 3 of the 1e-10 plan proposes counting at `prec` only where the margin

    R = prec * lam / (eps / c)

is generous, and flagging the level (over a coarser resolution) where it is not. This script asks
whether such a threshold T exists BEFORE the rule is designed: re-solve the benchmark with
truncation (basis D, the five `scoreboard.JITTERS` windows), record each level's R, and attribute
every count defect to a level.

    R is read from the refiner's `delta_res = DELTA_SAFETY * eps / c` (`mps/refine.py`), so
    R = DELTA_SAFETY * prec * lam / delta_res. The plan wants eps/c from the cached wide bracket;
    `delta_res` is the proxy that exists today.

A DEFECT is a trusted reference level whose count is wrong in a solve (members dropped, or
too many), or a solved level matching no reference level (extra). Each defect is attributed to a
solved level: the one assigned to it, or the nearest one for a level dropped outright. A rule
"flag where R < T" CATCHES a defect when its attributed level has R < T, and FLAGS NEEDLESSLY a
solved level whose count is right and R < T.

    .venv/bin/python notes/margin_rule_offline.py run [WORKERS]     # both rungs, writes json
    .venv/bin/python notes/margin_rule_offline.py report
    CHLADNI_MARGIN_BASIS=A ...                                      # the references' basis
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
sys.path.insert(0, HERE)
warnings.simplefilter("ignore")

PRECS = (1e-10, 1e-8)
#: "D" is the scoreboard's stand-in (1.0x lappy's per-corner sizing); "A" is the references'
#: recipe (1.5x). CHLADNI_MARGIN_BASIS selects it (env, so pool workers see it).
BASIS = os.environ.get("CHLADNI_MARGIN_BASIS", "D")
OUT = os.path.join(HERE, "margin_rule_offline.json" if BASIS == "D"
                   else f"margin_rule_offline_{BASIS}.json")
THRESHOLDS = (3, 10, 30, 100, 300, 1000)


def run_one(args):
    import make_references as MR
    import scoreboard  # noqa: F401 -- registers basis "D"
    from chladni.mps.refine import DELTA_SAFETY
    from regularization_1e10 import make_solver
    case, window, prec, jit = args
    lo, hi = window[0] * (1 + jit[0]), window[1] * (1 + jit[1])
    dom = MR.polygon(case["vertices"])
    b = MR.basis_for(dom, BASIS, np.sqrt(hi))
    t0 = time.perf_counter()
    try:
        res = make_solver(dom, b, "truncation", prec=prec).search_interval(lo, hi)
    except Exception as e:  # noqa: BLE001
        return dict(id=case["id"], prec=prec, jit=list(jit), error=repr(e)[:300])
    d = res.diagnostics
    levels = []
    for L in res.levels:
        R = DELTA_SAFETY * prec * abs(L.lam) / L.delta_res if L.delta_res > 0 else float("inf")
        levels.append(dict(lam=float(L.lam), mult=int(L.mult), bound=float(L.bound),
                           delta_res=float(L.delta_res), sigma=float(L.sigma), R=float(R),
                           count_flag=bool(L.count_uncertain or L.grouping_ambiguous),
                           met=bool(L.met(prec))))
    return dict(id=case["id"], prec=prec, jit=list(jit), wall=time.perf_counter() - t0,
                declined=bool(d.declined), unmet=len(d.unmet), levels=levels)


def run(workers=8):
    import make_references as MR
    import scoreboard as SB
    cases = {c["id"]: c for c in MR.load_cases()}
    refs = json.load(open(os.path.join(ROOT, "benchmarks", "douse", "references.json")))["cases"]
    jobs = [(cases[r["id"]], r["window"], p, j) for r in refs for p in PRECS for j in SB.JITTERS]
    with ProcessPoolExecutor(workers) as ex:
        runs = list(ex.map(run_one, jobs))
    json.dump(dict(runs=runs), open(OUT, "w"), indent=1)


def attribute(run, ref):
    """`(defects, good)`: R of the level each defect is attributed to, and R of each solved level
    whose count is right. Scored over the trusted reference levels, as `score_chladni` does."""
    import scoreboard as SB
    trusted = []
    for L in ref["levels"]:
        if L["status"] == "provisional":
            break
        trusted.append(L)
    if not trusted or run.get("declined") or "error" in run:
        return None
    top = trusted[-1]["lam"] * (1 + 1e-4)
    if len(trusted) < len(ref["levels"]):
        top = np.sqrt(trusted[-1]["lam"] * ref["levels"][len(trusted)]["lam"])
    levels = [L for L in run["levels"] if L["lam"] < top]
    if not levels:
        return None
    tl = np.array([T["lam"] for T in trusted])
    assigned = {j: [] for j in range(len(trusted))}
    defects, good = [], []
    for L in levels:
        j = int(np.argmin(np.abs(tl - L["lam"])))
        if abs(tl[j] - L["lam"]) <= SB.MATCH * tl[j]:
            assigned[j].append(L)
        else:
            defects.append(dict(kind="extra", lam=L["lam"], R=L["R"], flagged=L["count_flag"]))
    ll = np.array([L["lam"] for L in levels])
    for j, T in enumerate(trusted):
        got = assigned[j]
        n = sum(L["mult"] for L in got)
        if n == T["mult"]:
            good += [dict(lam=L["lam"], R=L["R"], flagged=L["count_flag"]) for L in got]
            continue
        if got:
            R = min(L["R"] for L in got)
            flagged = any(L["count_flag"] for L in got)
        else:                       # dropped outright: the nearest solved level
            near = levels[int(np.argmin(np.abs(ll - T["lam"])))]
            R, flagged = near["R"], near["count_flag"]
        defects.append(dict(kind="dropped" if n < T["mult"] else "too-many", lam=T["lam"],
                            want=T["mult"], got=n, R=R, flagged=flagged))
    return defects, good


def report():
    runs = json.load(open(OUT))["runs"]
    refs = {r["id"]: r for r in json.load(open(os.path.join(ROOT, "benchmarks", "douse",
                                                            "references.json")))["cases"]}
    for p in PRECS:
        D, G, where = [], [], []
        for r in runs:
            if r["prec"] != p:
                continue
            a = attribute(r, refs[r["id"]])
            if a is None:
                continue
            d, g = a
            D += d
            G += g
            where += [(r["id"], tuple(r["jit"]), x) for x in d]
        Rd = np.array([x["R"] for x in D])
        Rg = np.array([x["R"] for x in G])
        silent = np.array([not x["flagged"] for x in D])
        print(f"\nprec {p:g}: {len(D)} defects ({silent.sum()} unflagged by the solve), "
              f"{len(G)} right-count levels")
        print(f"  R of defects:     min {Rd.min():.3g}  median {np.median(Rd):.3g}  "
              f"max {Rd.max():.3g}" if len(Rd) else "  no defects")
        print(f"  R of good levels: min {Rg.min():.3g}  p5 {np.percentile(Rg, 5):.3g}  "
              f"median {np.median(Rg):.3g}")
        print(f"  {'T':>6s} {'caught':>9s} {'caught unflagged':>17s} {'needless flags':>16s}")
        for T in THRESHOLDS:
            print(f"  {T:6g} {np.sum(Rd < T):4d}/{len(Rd):<4d} "
                  f"{np.sum((Rd < T) & silent):8d}/{silent.sum():<8d} "
                  f"{np.sum(Rg < T):7d}/{len(Rg):<7d} ({np.mean(Rg < T):.1%})")
        print("  defects:")
        for cid, jit, x in sorted(where, key=lambda t: t[2]["R"]):
            print(f"    {cid:22s} {str(jit):24s} {x['kind']:8s} lam {x['lam']:<12.6g} "
                  f"R {x['R']:<9.3g} {'flagged' if x['flagged'] else 'UNFLAGGED'}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
    report()
