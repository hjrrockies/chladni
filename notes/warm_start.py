"""Step 0 of the warm-start plan: what would a warm window save?  (2026-10-13)

At douse's N = 6, k = 2 ratio optimum (chladni_vs_lappy.jsonl, seed 0), plan once, move the
vertices by ~1e-3, and solve the moved shape twice on the same plan:

- COLD: `Eigenproblem._solve_with`, the window [0.5 FK, Weyl(n + HEADROOM) * 1.15], widened until
  n + HEADROOM eigenvalues are found;
- WARM: `search_interval(a, b)` on the warm window, `a = min(0.98 FK, lam_1 - h_1)`,
  `b = lam_n + h_n`, `h = (local mean Weyl spacing) / 3` at the previous values.

Reports tension evaluations by search stage, wall time, and the values found.

    cd ../douse && .venv/bin/python ../chladni/notes/warm_start.py benchmarks/run/chladni_vs_lappy.jsonl
"""
import json
import sys
import time
import warnings

import numpy as np

warnings.simplefilter("ignore")

from chladni import asymp                                     # noqa: E402
from chladni.bounds import faber_krahn                        # noqa: E402
from chladni.eigenproblem import Eigenproblem                 # noqa: E402
from chladni.geometry import Polygon                          # noqa: E402

N_EIGS = 4


def spacing(lam, dom):
    """Local mean Weyl spacing at `lam`, as lappy's tracking uses it (`lappy/evp.py:472`)."""
    n = max(float(asymp.weyl_count_poly(lam, dom)), 0.5)
    return float(asymp.weyl_est_poly(n + 1, dom) - asymp.weyl_est_poly(n, dom))


def stages(res):
    d = res.diagnostics
    keys = ("search_evals", "refine_evals", "multiplicity_evals", "probe_evals", "side_evals")
    return {k.replace("_evals", ""): getattr(d, k, 0) for k in keys}


def main(path):
    from douse.polygon import VertexPolygon
    rows = [json.loads(line) for line in open(path)]
    x = np.array([r for r in rows if (r["backend"], r["N"], r["k"], r["seed"])
                  == ("chladni", 6, 2, 0)][0]["x"])
    fam = VertexPolygon(6, perim=4.0)
    dom0 = Polygon(np.asarray(fam.vertices(x), complex), bc="dir")
    ep0 = Eigenproblem(dom0)
    plan = ep0.plan(N_EIGS, prec=1e-10)
    prev = ep0.solve(N_EIGS, prec=1e-10, plan=plan)
    lp = prev.unfolded[:N_EIGS]
    print(f"plan rung {plan.rung}, {plan.n_cols} columns; previous {np.round(lp, 8)}")

    rng = np.random.default_rng(0)
    x1 = x + 1e-3 * rng.normal(size=x.size)
    dom = Polygon(np.asarray(fam.vertices(x1), complex), bc="dir")
    ep = Eigenproblem(dom)

    t = time.perf_counter()
    res_c, (lo, hi) = ep._solve_with(plan, N_EIGS, 1e-10, "flag")
    tc = time.perf_counter() - t
    uc = res_c.unfolded
    print(f"\nCOLD  window [{lo:.2f}, {hi:.2f}]  {tc:.1f}s  evals {sum(stages(res_c).values())} "
          f"{stages(res_c)}\n      found {len(uc)}: {np.round(uc, 8)}")

    h = np.array([spacing(v, dom) / 3 for v in lp])
    a = min(0.98 * float(faber_krahn(dom)), lp[0] - h[0])
    b = lp[-1] + h[-1]
    solver = plan.solver(dom, prec=1e-10)
    t = time.perf_counter()
    res_w = solver.search_interval(a, b)
    tw = time.perf_counter() - t
    uw = res_w.unfolded
    print(f"WARM  window [{a:.2f}, {b:.2f}]  {tw:.1f}s  evals {sum(stages(res_w).values())} "
          f"{stages(res_w)}\n      found {len(uw)}: {np.round(uw, 8)}  h = {np.round(h, 3)}")
    print(f"      audit {res_w.diagnostics.count_audit.verdict if res_w.diagnostics.count_audit else None}"
          f"; unmet {res_w.diagnostics.unmet}; declined {res_w.declined}")
    k = min(len(uw), N_EIGS)
    print(f"\nwarm vs cold, first {k}: max rel diff "
          f"{np.max(np.abs(uw[:k] - uc[:k]) / uc[:k]):.1e};  speedup {tc / tw:.2f}x")


if __name__ == "__main__":
    main(sys.argv[1])
