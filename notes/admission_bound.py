"""Which bar should `Level.bound` report: `delta_res` (admitted at a margin) or Moler-Payne?  (2026-10-11)

`delta_res` (the refiner's `DELTA_SAFETY * eps/c`) measures the WELL; Moler-Payne bounds the
distance to an eigenvalue. They part where the basis is BIASED: the well is clean and deep but in
the wrong place. T2.8-N5s2 on the Fourier-Bessel-only basis A sits 7e-10 relative from the true
eigenvalue at margin R = 56 (1e-8), 1.9x outside `delta_res`, while its Moler-Payne bound covers it
(`notes/phase3_rescore.md`). This re-solves the douse benchmark recording BOTH bars per level, so
each admission policy can be priced: how many levels it leaves under-covered, and how many met
levels it turns unmet.

    .venv/bin/python notes/admission_bound.py run [WORKERS]      # writes notes/admission_bound.json
    .venv/bin/python notes/admission_bound.py report
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

OUT = os.path.join(HERE, "admission_bound.json")
PRECS = (1e-10, 1e-8)
BASES = ("D", "A")


def run_one(args):
    import make_references as MR
    import scoreboard  # noqa: F401 -- registers basis "D"
    from chladni.mps import MPSEigensolver
    case, window, basis, prec, jit = args
    lo, hi = window[0] * (1 + jit[0]), window[1] * (1 + jit[1])
    dom = MR.polygon(case["vertices"])
    b = MR.basis_for(dom, basis, np.sqrt(hi))
    n = len(b)
    t0 = time.perf_counter()
    try:
        s = MPSEigensolver(dom, basis=b, n_bdry=3 * n, n_int=n, prec=prec, rng=0, reg="svd")
        res = s.search_interval(lo, hi)
    except Exception as e:  # noqa: BLE001
        return dict(id=case["id"], basis=basis, prec=prec, jit=list(jit), error=repr(e)[:300])
    return dict(id=case["id"], basis=basis, prec=prec, jit=list(jit),
                wall=time.perf_counter() - t0, declined=bool(res.diagnostics.declined),
                levels=[dict(lam=float(L.lam), mult=int(L.mult), delta_res=float(L.delta_res),
                             mp_bound=float(L.mp_bound), bound=float(L.bound),
                             count_flag=bool(L.count_uncertain or L.grouping_ambiguous))
                        for L in res.levels])


def run(workers=8):
    import make_references as MR
    import scoreboard as SB
    cases = {c["id"]: c for c in MR.load_cases()}
    refs = json.load(open(os.path.join(ROOT, "benchmarks", "douse", "references.json")))["cases"]
    jobs = [(cases[r["id"]], r["window"], b, p, j) for r in refs for b in BASES for p in PRECS
            for j in SB.JITTERS]
    with ProcessPoolExecutor(workers) as ex:
        runs = list(ex.map(run_one, jobs))
    json.dump(dict(runs=runs), open(OUT, "w"), indent=1)


def policies(L, prec):
    """The reported bar under each admission policy."""
    dr, mp, lam = L["delta_res"], L["mp_bound"], abs(L["lam"])
    have_mp = np.isfinite(mp)
    out = {}
    for M in (1, 2, 5, 10):
        adm = dr <= prec * lam / M
        out[f"margin M={M}"] = (min(dr, mp) if have_mp else dr) if adm else (mp if have_mp else dr)
    out["Moler-Payne only"] = mp if have_mp else dr
    return out


def report():
    runs = json.load(open(OUT))["runs"]
    refs = {r["id"]: r for r in json.load(open(os.path.join(ROOT, "benchmarks", "douse",
                                                            "references.json")))["cases"]}
    for basis in BASES:
        for p in PRECS:
            tally = {}
            n_judged = 0
            for r in runs:
                if r["basis"] != basis or r["prec"] != p or "levels" not in r:
                    continue
                ref = [T for T in refs[r["id"]]["levels"]
                       if T["status"] != "provisional" and np.isfinite(T.get("bound", np.nan))]
                tl = np.array([T["lam"] for T in ref])
                tb = np.array([T["bound"] for T in ref])
                for L in r["levels"]:
                    pol = policies(L, p)
                    judge = False
                    if len(tl):
                        j = int(np.argmin(np.abs(tl - L["lam"])))
                        err = abs(tl[j] - L["lam"])
                        # judge coverage only against a reference much tighter than the level
                        tight = min(v for v in pol.values() if np.isfinite(v))
                        judge = tb[j] <= 0.1 * tight and err <= 1e-4 * L["lam"]
                    n_judged += judge
                    for k, b in pol.items():
                        t = tally.setdefault(k, [0, 0, 0])
                        t[0] += b <= p * abs(L["lam"])                 # met
                        if judge:
                            t[1] += b < err                            # under-covered
                            t[2] += (b < err) and (b <= p * abs(L["lam"]))  # met AND under
            print(f"basis {basis}, prec {p:g}  ({n_judged} levels judgeable against a tight reference)")
            for k, (met, und, met_und) in tally.items():
                print(f"  {k:18s} met {met:5d}   under-covered {und:3d}   met & under-covered {met_und:3d}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
    report()
