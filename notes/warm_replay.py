"""The warm path as douse drives it, replayed deterministically.  (2026-11-03)

douse's cluster optimizer solves every trial shape warm, with per-value windows
(`Eigenproblem.solve(..., near=prev, windows=h)` -> `MPSEigensolver.track_levels`), and refuses a
shape on any flag but "count audit: unaudited" (`douse/chladni_backend.py`). This replays that
path on the final shapes of the hardest douse ratio runs (`hadamard_hard_shapes_cases.json`):
a cold solve, then STEPS seeded random vertex steps of relative size STEP, each solved warm with
windows `h_i = WINDOW * lam_i` (overlapping windows merge into one searched window, as douse's
clusters do).

Per step it records the deterministic cost (`bench_counters`), whether the warm solve was
accepted, the flags, and whether douse would refuse it. The before/after gate for the warm-path
items of `accuracy_audit_2026-11-03.md`.

usage: python notes/warm_replay.py [WORKERS] [OUT]
"""
import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CASES = json.load(open(os.path.join(HERE, "hadamard_hard_shapes_cases.json")))["cases"]
PREC, STEP, WINDOW = 1e-10, 1e-3, 3e-3
STEPS = int(os.environ.get("WARM_REPLAY_STEPS", 10))


def douse_refuses(flags):
    return any(not f.startswith("count audit: unaudited") for f in flags)


def run(ci):
    warnings.simplefilter("ignore")
    import bench_counters as BC
    from chladni.eigenproblem import Eigenproblem
    from chladni.geometry import Polygon
    BC.install()
    c = CASES[ci]
    v = np.array([complex(*z) for z in c["verts"]])
    n = len(c["lams"])
    rng = np.random.default_rng(1000 + ci)
    c0 = BC.snapshot()
    prev = Eigenproblem(Polygon(v, bc="dir")).solve(n, prec=PREC)
    rows = [dict(case=ci, step=0, kind="cold", counts=BC.since(c0), flags=list(prev.flags),
                 lam=[float(x) for x in prev.lam])]
    scale = np.abs(np.diff(np.append(v, v[0]))).sum() / len(v)
    for s in range(1, STEPS + 1):
        v = v + STEP * scale * (rng.normal(size=len(v)) + 1j * rng.normal(size=len(v)))
        lam = np.asarray(prev.lam, dtype=float)
        mult = np.asarray(prev.mult, dtype=int)
        h = WINDOW * np.repeat(lam, mult)[:n]             # per value, as `windows` wants
        c0 = BC.snapshot()
        try:
            spec = Eigenproblem(Polygon(v, bc="dir")).solve(n, prec=PREC, near=prev, windows=h)
        except Exception as e:                            # noqa: BLE001 -- a crash is a row
            rows.append(dict(case=ci, step=s, error=repr(e)[:300], counts=BC.since(c0)))
            break
        rows.append(dict(case=ci, step=s, kind="warm" if spec.warm else "cold",
                         note=str(spec.warm_note)[:160] if spec.warm_note else None,
                         counts=BC.since(c0), flags=list(spec.flags),
                         refused=douse_refuses(spec.flags), lam=[float(x) for x in spec.lam],
                         mult=[int(m) for m in spec.mult], n_levels_prev=len(mult)))
        prev = spec
    return rows


def report(rows):
    steps = [r for r in rows if r["step"] > 0]
    tot = lambda k: sum(r["counts"][k] for r in rows if "counts" in r)  # noqa: E731
    warm = sum(r.get("kind") == "warm" for r in steps)
    refused = sum(bool(r.get("refused")) for r in steps)
    errors = sum("error" in r for r in steps)
    print(f"steps {len(steps)}: warm {warm}, cold fallbacks {len(steps) - warm - errors}, "
          f"douse would refuse {refused}, errors {errors}")
    print(f"assemblies {tot('assemblies')}, cauchy calls {tot('cauchy_calls')}, "
          f"cauchy nodes {tot('cauchy_nodes')}")
    for r in steps:
        if r.get("refused") or "error" in r:
            print("  ", r["case"], r["step"], r.get("kind"), (r.get("flags") or [r.get("error")])[:2])


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "warm_replay.json")
    with ProcessPoolExecutor(workers) as ex:
        rows = [r for rs in ex.map(run, range(len(CASES))) for r in rs]
    json.dump(dict(prec=PREC, steps=STEPS, step=STEP, window=WINDOW, rows=rows),
              open(out, "w"), indent=1)
    report(rows)
