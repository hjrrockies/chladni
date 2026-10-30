"""Step 5 first full-pipeline try (2026-10-27): solve(coarse_bracket=True) on the apex lightning
basis (the threshold search declines) and the L-shape rung 0, prefix windows for n = 15, prec 1e-10.
Values against references; counts; met; cost by stage."""
import time
import numpy as np
from chladni import asymp
from chladni.bounds import faber_krahn
from chladni.geometry import L_shape, Polygon
from chladni.plan import HEADROOM, Plan, plan_for
from chladni.reference import L_shape_eigs

apex = Polygon(np.array([0, 1, 0.1 + 0.25j]), bc="dir")
p0 = plan_for(apex, 15, 1e-10, 0)
light = Plan(prec=1e-10, k_max=p0.k_max, corner_orders=p0.corner_orders, lightning_n=10, rung=3)
apex_ref = np.array([288.9643854097, 451.0919790235, 628.4568669251, 821.3823602455, 911.5895344006,
                     1043.9357242892, 1216.3832526905, 1293.9127902104, 1504.8635788596,
                     1579.2988086407, 1788.5259323635, 1842.6310031747, 1922.8118975078,
                     2106.9668363896, 2220.6128167775])
for name, dom, mk, ref in (("apex lightning", apex, lambda: light.solver(apex), apex_ref),
                           ("L-shape rung 0", L_shape(), lambda: plan_for(L_shape(), 15, 1e-10, 0).solver(L_shape()),
                            np.asarray(L_shape_eigs(25)))):
    lo = 0.5 * float(faber_krahn(dom))
    hi = float(asymp.weyl_est_poly(15 + HEADROOM, dom)) * 1.15
    for coarse in (False, True):
        s = mk()
        t = time.time()
        res = s.search_interval(lo, hi, coarse_bracket=coarse)
        d = res.diagnostics
        lv = sorted(res.levels, key=lambda L: L.lam)
        errs = [abs(L.lam - ref[np.argmin(abs(ref - L.lam))]) / L.lam for L in lv if L.lam <= ref[-1] * 1.001]
        print(f"{name:15s} {'coarse' if coarse else 'shipped'}: declined {d.declined} levels {len(lv)} "
              f"mult {sum(L.mult for L in lv)} met {sum(L.met(1e-10) for L in lv)} unmet {len(d.unmet)} "
              f"uncertain {d.n_uncertain} | evals {d.evaluations} (search {d.search_evals}, refine "
              f"{d.refine_evals}, mult {d.multiplicity_evals}, probe {d.probe_evals}, side {d.side_evals}) "
              f"frozen roots {d.n_reranked} | max err vs ref {max(errs) if errs else float('nan'):.1e} "
              f"| worst rel bound {max((L.bound / L.lam for L in lv), default=float('nan')):.1e} "
              f"min margin {min((L.margin for L in lv), default=float('nan')):.3g} | {time.time() - t:.0f}s",
              flush=True)
