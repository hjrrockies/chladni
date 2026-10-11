"""Do rank flips at the truncation cutoff move outcomes on douse shapes?  (2026-10-11)

`notes/regularization_1e10.md` E3 found the first dropped singular value within 10% of the cutoff
at 15% of evaluated points. This asks whether that matters: re-solve the benchmark (basis D,
truncation, prec 1e-10, the five `scoreboard.JITTERS` windows) with `rtol` scaled by 0.9 and 1.1,
and compare each solve's count structure and outcome with E1's truncation solve at 1.0x
(`notes/regularization_1e10_e1.json`). A case whose outcome moves under a 10% `rtol` nudge is
rank-flip sensitive; one that is wrong at all three is not a rank-flip defect.

    .venv/bin/python notes/rtol_nudge.py [WORKERS]
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
FACTORS = (0.9, 1.1)
OUT = os.path.join(HERE, "rtol_nudge.json")


def run_one(args):
    import make_references as MR
    import scoreboard  # noqa: F401 -- registers basis "D"
    from chladni.mps import MPSEigensolver
    case, window, f, jit = args
    lo, hi = window[0] * (1 + jit[0]), window[1] * (1 + jit[1])
    dom = MR.polygon(case["vertices"])
    b = MR.basis_for(dom, "D", np.sqrt(hi))
    n = len(b)
    rtol = f * float(np.sqrt(PREC * np.finfo(float).eps))
    t0 = time.perf_counter()
    try:
        s = MPSEigensolver(dom, basis=b, n_bdry=3 * n, n_int=n, prec=PREC, rng=0, reg="svd",
                           rtol=rtol)
        res = s.search_interval(lo, hi)
    except Exception as e:  # noqa: BLE001
        return dict(id=case["id"], factor=f, jit=list(jit), error=repr(e)[:300])
    d = res.diagnostics
    return dict(id=case["id"], factor=f, jit=list(jit), wall=time.perf_counter() - t0,
                declined=bool(d.declined), unmet=len(d.unmet), n_uncertain=int(d.n_uncertain),
                levels=[dict(lam=float(L.lam), mult=int(L.mult), bound=float(L.bound),
                             count_flag=bool(L.count_uncertain or L.grouping_ambiguous),
                             met=bool(L.met(PREC))) for L in res.levels])


def outcome(run, ref):
    import scoreboard as SB
    sc = SB.score_chladni(run, ref)
    o = sc["outcome"]
    if o == "wrong":
        o = "SILENT" if sc.get("silent") else "flagged"
    return o, str(sc.get("sig"))


def main(workers=8):
    import make_references as MR
    import scoreboard as SB
    cases = {c["id"]: c for c in MR.load_cases()}
    refs = {r["id"]: r for r in json.load(open(os.path.join(ROOT, "benchmarks", "douse",
                                                            "references.json")))["cases"]}
    if os.path.exists(OUT) and "--rescore" in sys.argv:
        runs = json.load(open(OUT))["runs"]
    else:
        jobs = [(cases[i], refs[i]["window"], f, j) for i in refs for f in FACTORS
                for j in SB.JITTERS]
        with ProcessPoolExecutor(workers) as ex:
            runs = list(ex.map(run_one, jobs))
        json.dump(dict(prec=PREC, runs=runs), open(OUT, "w"), indent=1)
    base = [dict(r, factor=1.0) for r in
            json.load(open(os.path.join(HERE, "regularization_1e10_e1.json")))["runs"]
            if r["variant"] == "truncation"]
    table = {}
    for r in base + runs:
        if "error" in r:
            o, sig = "crash", "crash"
        else:
            o, sig = outcome(r, refs[r["id"]])
        table.setdefault(r["id"], {}).setdefault(tuple(r["jit"]), {})[r["factor"]] = (o, sig)
    moved_sig = moved_out = n = 0
    print(f"rtol x {FACTORS} vs 1.0 at prec {PREC:g}, basis D (o ok, f flagged, S silent, "
          f"d declined, u unscored)\n")
    for cid, by_jit in sorted(table.items()):
        line = []
        for jit, by_f in by_jit.items():
            if len(by_f) < 3:
                continue
            n += 1
            outs = [by_f[f][0] for f in (0.9, 1.0, 1.1)]
            sigs = {by_f[f][1] for f in (0.9, 1.0, 1.1)}
            moved_sig += len(sigs) > 1
            moved_out += len(set(outs)) > 1
            line.append("".join(o[0] if o != "SILENT" else "S" for o in outs))
        if any(len(set(x)) > 1 or "S" in x for x in line):
            print(f"  {cid:24s} " + " ".join(line))
    print(f"\n{n} (case, window) solves: count structure moved under the nudge at {moved_sig}, "
          f"outcome moved at {moved_out}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8)
