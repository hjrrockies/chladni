"""The lappy side of the scoreboard: a cold `lappy` solve on every case, as `douse` makes one.

Runs under DOUSE's venv (which has `lappy`), read-only on both:

    /Users/hringer/projects/python/douse/.venv/bin/python benchmarks/douse/lappy_runner.py

Per case, the solve `douse.evaluate.SpectrumEvaluator` makes when it does not track: a basis plan
(`lappy.basis_plan.plan_basis` at `weyl_est(lam_max_window)`, `target = 1e-14`), realized on the
shape, `MPSEigensolver.from_domain(..., prec = 1e-13)`, `Eigenproblem.solve(n_eigs)`. Twice where
the case stores a frozen plan: once planned at the shape ("fresh"), once at `plan_at` ("frozen").
`n_eigs` is the case's request where known, else 4; `lam_max_window = max(8, 2 n_eigs)`, as the
ratio runs used. NONE of douse's guards (Weyl, spurious, multiplicity re-checks) run: this is the
solver those guards were written around. Writes `lappy_runs.json`.
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
PREC, TARGET = 1e-13, 1e-14


def run(args):
    case, mode = args
    from lappy import Eigenproblem, Polygon
    from lappy import basis_plan as BP
    from lappy.asymp import weyl_est
    from lappy.mps import MPSEigensolver
    warnings.simplefilter("ignore")
    n = case.get("n_eigs_requested") or 4
    win = max(8, 2 * n)
    dom = Polygon(np.asarray(case["vertices"], dtype=float) @ np.array([1, 1j]))
    plan_dom = dom
    if mode == "frozen":
        plan_dom = Polygon(np.asarray(case["plan_at"]["vertices"], dtype=float) @ np.array([1, 1j]))
    t0 = time.perf_counter()
    try:
        plan = BP.plan_basis(plan_dom, weyl_est(win, plan_dom), target=TARGET)
        solver = MPSEigensolver.from_domain(dom, lam_max=weyl_est(win, dom),
                                            basis=BP.realize(plan, dom), prec=PREC)
        evp = Eigenproblem(dom, eval_solver=solver, precision=PREC)
        lams = np.asarray(evp.solve(n), dtype=float)
        sig = [float(np.atleast_1d(solver.sigma(float(x)))[0]) for x in lams]
        return dict(id=case["id"], mode=mode, n_eigs=n, lams=lams.tolist(), sigma=sig,
                    n_basis=int(plan.n_total), wall=time.perf_counter() - t0)
    except Exception as e:  # noqa: BLE001 -- an exception is a scored outcome
        return dict(id=case["id"], mode=mode, n_eigs=n, error=f"{type(e).__name__}: {e}"[:300],
                    wall=time.perf_counter() - t0)


def main():
    cases = json.load(open(os.path.join(HERE, "cases.json")))["cases"]
    jobs = [(c, "fresh") for c in cases] + [(c, "frozen") for c in cases if c.get("plan_at")]
    out = []
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as ex:
        for r in ex.map(run, jobs):
            out.append(r)
            print(f"{r['id']:22s} {r['mode']:6s} {r['wall']:6.1f}s "
                  f"{r.get('error') or np.round(r['lams'], 4).tolist()}", flush=True)
    json.dump(dict(prec=PREC, target=TARGET, runs=out),
              open(os.path.join(HERE, "lappy_runs.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
