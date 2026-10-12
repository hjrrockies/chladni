"""Phase 5 step 0b: what limits the douse shapes no recipe brings to 1e-10?  (2026-10-12)

`notes/planner_recipes.md`: on 16 of 35 douse benchmark cases no recipe of the family (Fourier-
Bessel at every corner, 1.0-2.0x lappy's sizing, with or without a source layer) meets 1e-10, and
most of them meet it on NO level, at margins near 0.02 -- a hundred times short -- barely moved by
doubling the expansion. Before choosing a ladder, find what binds. On four hard cases and two
controls, at prec 1e-10 on the reference window:

    V0  F2.0                       the best recipe measured (baseline)
    V1  F2.0, collocation x2       6 boundary and 2 interior points per column, not 3 and 1
    V2  F3.0                       does the trend in scale continue?
    V3  F2.0 + lightning           Gopal-Trefethen clustered sources at every corner
                                   (`Fundamental.lightning`, 12 per corner)

and, independently of any solve, the basis's own evaluation accuracy: `Sampling.accuracy(k)` --
each column's relative-error bound, at the window's top wavenumber, over the boundary points.
Also the geometry: corner angles, and k*r at each corner (r = distance to the farthest vertex,
the radius the expansion must reach).

    .venv/bin/python notes/hard_cases.py
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
HARD = ("T2.2", "T2.3-pre-ref0", "T2.5", "T2.6-N5k4-L7-ref0")
CONTROLS = ("T1.4-oct", "T2.9-N5k3convex")
VARIANTS = ("V0 F2.0", "V1 F2.0 colloc x2", "V2 F3.0", "V3 F2.0 + lightning")
OUT = os.path.join(HERE, "hard_cases.json")


def build(dom, k, variant):
    from chladni.helmholtz import Fundamental
    from planner_recipes import basis
    scale = 3.0 if variant.startswith("V2") else 2.0
    b = basis(dom, k, scale, False, 0.0)
    if variant.startswith("V3"):
        for c in dom.corners:
            try:
                b = b + Fundamental.lightning(c, 12)
            except Exception:  # noqa: BLE001 -- a corner with no exterior direction
                pass
    ratio = (6, 2) if variant.startswith("V1") else (3, 1)
    return b, ratio


def run_one(args):
    import make_references as MR
    from chladni.mps import MPSEigensolver
    case, window, variant = args
    lo, hi = window
    dom = MR.polygon(case["vertices"])
    k = float(np.sqrt(hi))
    b, (rb, ri) = build(dom, k, variant)
    n = len(b)
    acc = np.asarray(b.at(dom.bdry_quad(3 * n)).accuracy(k), float)
    t0 = time.perf_counter()
    s = MPSEigensolver(dom, basis=b, n_bdry=rb * n, n_int=ri * n, prec=PREC, rng=0, reg="svd")
    res = s.search_interval(lo, hi)
    lv = res.levels
    return dict(id=case["id"], variant=variant, n_cols=n, wall=time.perf_counter() - t0,
                acc_max=float(acc.max()), acc_p90=float(np.percentile(acc, 90)),
                n_acc_bad=int(np.sum(acc > PREC)),
                declined=bool(res.diagnostics.declined),
                met=[bool(L.met(PREC)) for L in lv],
                margin=[float(L.margin) for L in lv],
                floor=[float(L.sigma) for L in lv], lam=[float(L.lam) for L in lv])


def geometry(case):
    import make_references as MR
    dom = MR.polygon(case["vertices"])
    zs = np.array([c.z for c in dom.corners])
    rows = []
    for c in dom.corners:
        r = float(np.max(np.abs(zs - c.z)))
        rows.append((round(c.omega / np.pi, 3), round(r, 3)))
    return dom, rows


def main(workers=8):
    import make_references as MR
    cases = {c["id"]: c for c in MR.load_cases()}
    refs = {r["id"]: r for r in json.load(open(os.path.join(ROOT, "benchmarks", "douse",
                                                            "references.json")))["cases"]}
    ids = HARD + CONTROLS
    jobs = [(cases[i], refs[i]["window"], v) for i in ids for v in VARIANTS]
    with ProcessPoolExecutor(workers) as ex:
        runs = list(ex.map(run_one, jobs))
    json.dump(dict(runs=runs), open(OUT, "w"), indent=1)
    for i in ids:
        dom, g = geometry(cases[i])
        k = float(np.sqrt(refs[i]["window"][1]))
        tag = "HARD" if i in HARD else "control"
        print(f"\n{i} ({tag}): area {dom.area:.3g}, diameter {dom.diameter:.3g}, k_top {k:.2f}")
        print(f"  corners (omega/pi, r): {g}")
        print(f"  {'variant':22s} {'cols':>5s} {'wall':>6s} {'met':>6s} {'margin med/min':>16s} "
              f"{'floor med':>9s} {'acc max':>8s} {'cols acc>prec':>13s}")
        for r in runs:
            if r["id"] != i:
                continue
            m = np.array(r["margin"]) if r["margin"] else np.array([np.nan])
            print(f"  {r['variant']:22s} {r['n_cols']:5d} {r['wall']:5.1f}s "
                  f"{np.mean(r['met']) if r['met'] else 0:6.0%} "
                  f"{np.nanmedian(m):8.3g}/{np.nanmin(m):<7.2g} "
                  f"{np.median(r['floor']) if r['floor'] else np.nan:9.2e} {r['acc_max']:8.1e} "
                  f"{r['n_acc_bad']:6d}/{r['n_cols']:<6d}"
                  + ("  DECLINED" if r["declined"] else ""))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
