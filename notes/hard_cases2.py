"""Phase 5 step 0c: why do the sharp-cornered douse shapes stagnate?  (2026-10-12)

`notes/planner_recipes.md`: 11 of 35 douse cases pass no rung at 1e-10. Their worst margins vs
corner-expansion size (2x .. 4x lappy's sizing) split them in two:

    group A  converging steadily; extrapolates to passing at 4.1-4.7x. T2.3 (3 cases),
             T2.6-N5k4-L7 (2), T2.1-rerun. Near-right angles plus one slightly re-entrant corner.
    group B  stagnant: T1.3 (11- and 16-degree corners), T2.2 (15 degrees), T2.6-N3k4-L8 (18 and
             20 degrees), T2.6-N3k5-L6 (40 and 55 degrees, slow). Very SHARP corners.

Two hypotheses for group B, plus a check on group A, at prec 1e-10 on the reference window:

    per-corner   base 3x; then each corner in turn at 9x (3x its base). Does one corner unlock it?
    accurate     base 3x with every Fourier-Bessel column whose own relative-error bound
                 (`Sampling.accuracy(k)` over the boundary points, at the window's top) exceeds
                 1e-11 removed. Do inaccurate columns cap the floor?
    group A      5x, to confirm the extrapolation.

    .venv/bin/python notes/hard_cases2.py [WORKERS]
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
GROUP_A = ("T2.3-pre-ref0", "T2.6-N5k4-L7-ref0", "T2.1-rerun")
GROUP_B = ("T2.2", "T1.3", "T2.6-N3k4-L8-ref0", "T2.6-N3k5-L6-ref0")
ACC_CUT = 1e-11
OUT = os.path.join(HERE, "hard_cases2.json")


def orders(dom, k, scales):
    zs = np.array([c.z for c in dom.corners])
    out = []
    for c, s in zip(dom.corners, scales):
        r = float(np.max(np.abs(zs - c.z)))
        out.append(max(int(np.ceil(s * (k * r + 2 * (k * r) ** (1 / 3) + 3) / (np.pi / c.omega))), 3))
    return out


def build(dom, k, scales, acc_cut=None):
    from chladni.helmholtz import FourierBessel
    quad = dom.bdry_quad(600)
    b, dropped = None, 0
    for c, n in zip(dom.corners, orders(dom, k, scales)):
        f = FourierBessel.at_corner(c, n=n, domain=dom)
        if acc_cut is not None:
            acc = np.asarray(f.at(quad).accuracy(k), float)
            keep = np.flatnonzero(acc <= acc_cut)
            dropped += n - len(keep)
            if len(keep) == 0:
                continue
            f = FourierBessel.at_corner(c, degrees=np.asarray(f.degrees)[keep], domain=dom)
        b = f if b is None else b + f
    return b, dropped


def run_one(args):
    import make_references as MR
    from chladni.mps import MPSEigensolver
    case, window, label, scales, acc_cut = args
    dom = MR.polygon(case["vertices"])
    k = float(np.sqrt(window[1]))
    t0 = time.perf_counter()
    try:
        b, dropped = build(dom, k, scales, acc_cut)
        n = len(b)
        res = MPSEigensolver(dom, basis=b, n_bdry=3 * n, n_int=n, prec=PREC, rng=0,
                             reg="svd").search_interval(*window)
    except Exception as e:  # noqa: BLE001
        return dict(id=case["id"], label=label, error=repr(e)[:200])
    lv = res.levels
    return dict(id=case["id"], label=label, n_cols=n, dropped=dropped,
                wall=time.perf_counter() - t0, declined=bool(res.diagnostics.declined),
                met=[bool(L.met(PREC)) for L in lv], margin=[float(L.margin) for L in lv],
                floor=[float(L.sigma) for L in lv])


def main(workers=8):
    import make_references as MR
    cases = {c["id"]: c for c in MR.load_cases()}
    refs = {r["id"]: r for r in json.load(open(os.path.join(ROOT, "benchmarks", "douse",
                                                            "references.json")))["cases"]}
    jobs = []
    for cid in GROUP_B:
        m = len(MR.polygon(cases[cid]["vertices"]).corners)
        jobs.append((cases[cid], refs[cid]["window"], "base 3x", [3.0] * m, None))
        jobs.append((cases[cid], refs[cid]["window"], f"3x, inaccurate cols out (>{ACC_CUT:g})",
                     [3.0] * m, ACC_CUT))
        for i in range(m):
            s = [3.0] * m
            s[i] = 9.0
            jobs.append((cases[cid], refs[cid]["window"], f"3x, corner {i} at 9x", s, None))
    for cid in GROUP_A:
        m = len(MR.polygon(cases[cid]["vertices"]).corners)
        jobs.append((cases[cid], refs[cid]["window"], "5x", [5.0] * m, None))
    with ProcessPoolExecutor(workers) as ex:
        runs = list(ex.map(run_one, jobs))
    json.dump(dict(runs=runs), open(OUT, "w"), indent=1)
    for cid in GROUP_B + GROUP_A:
        dom = MR.polygon(cases[cid]["vertices"])
        print(f"\n{cid}: corners (index: omega/pi) "
              + ", ".join(f"{i}: {c.omega / np.pi:.3f}" for i, c in enumerate(dom.corners)))
        for r in runs:
            if r["id"] != cid:
                continue
            if "error" in r:
                print(f"  {r['label']:34s} ERROR {r['error']}")
                continue
            m = np.array(r["margin"]) if r["margin"] else np.array([np.nan])
            print(f"  {r['label']:34s} cols {r['n_cols']:4d} (dropped {r['dropped']:3d}) "
                  f"{r['wall']:5.1f}s  met {np.mean(r['met']) if r['met'] else 0:5.0%}  margin "
                  f"min {np.nanmin(m):8.2g} med {np.nanmedian(m):8.2g}  floor med "
                  f"{np.median(r['floor']) if r['floor'] else np.nan:8.1e}"
                  + ("  DECLINED" if r["declined"] else ""))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
