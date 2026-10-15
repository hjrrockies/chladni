"""How much does the threaded seed grid save on a cold solve?  (2026-10-14)

At douse's N = 6, k = 2 ratio optimum (chladni_vs_lappy_constrained.jsonl, chladni, seed 0):
plan once, then time `Eigenproblem._solve_with` (the cold path) with the seed batch on 1 worker
(the serial loop) and on `DEFAULT_WORKERS`, two repeats each, on a fresh solver each time so
no cache carries over. The evaluation counts must be identical; only the wall time moves.

    cd ../douse && .venv/bin/python ../chladni/notes/seed_workers.py \
        benchmarks/run/chladni_vs_lappy_constrained.jsonl
"""
import json
import sys
import time
import warnings

import numpy as np

warnings.simplefilter("ignore")

from chladni.eigenproblem import Eigenproblem                 # noqa: E402
from chladni.geometry import Polygon                          # noqa: E402
from chladni.mps import objective as O                         # noqa: E402

N_EIGS = 4


def main(path):
    from douse.polygon import VertexPolygon
    rows = [json.loads(line) for line in open(path)]
    x = np.array([r for r in rows if (r["backend"], r["N"], r["k"], r["seed"])
                  == ("chladni", 6, 2, 0)][0]["x"])
    fam = VertexPolygon(6, perim=4.0)
    dom = Polygon(np.asarray(fam.vertices(x), complex), bc="dir")
    ep = Eigenproblem(dom)
    plan = ep.plan(N_EIGS, prec=1e-10)
    print(f"plan rung {plan.rung}, {plan.n_cols} columns, DEFAULT_WORKERS = {O.DEFAULT_WORKERS}")
    default = O.DEFAULT_WORKERS
    for workers in (1, default, 1, default):
        O.DEFAULT_WORKERS = workers
        t0 = time.perf_counter()
        res, _ = ep._solve_with(plan, N_EIGS, 1e-10, "flag")
        dt = time.perf_counter() - t0
        d = res.diagnostics
        print(f"workers {workers}: {dt:6.1f} s, {d.evaluations} evaluations "
              f"(seed {d.search.seed_evals}, cover {d.search.cover_evals}), "
              f"lam {np.round(res.unfolded[:N_EIGS], 8)}")
    O.DEFAULT_WORKERS = default


if __name__ == "__main__":
    main(sys.argv[1])
