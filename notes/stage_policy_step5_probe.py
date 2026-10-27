"""Step 5 feasibility (2026-10-27): bracketing on the coarse penalty, alone.
`bracket_interval` driven through `PolicyView(obj, Policy.penalty())` on the apex triangle's lightning
basis (shipped rung 3) over the planner's prefix window for n = 15, where the threshold search
declines in the seed. Asked for prec_b = 10 tau (the coarse floor binds below that). Reports whether it
settles, the brackets against the 19 reference values, and the cost; then the same on the L-shape
rung 0 for contrast."""
import time
import numpy as np
from chladni import asymp
from chladni.bounds import faber_krahn
from chladni.geometry import L_shape, Polygon
from chladni.mps.bracket import bracket_interval
from chladni.mps.objective import PolicyView
from chladni.mps.policy import Policy, TAU_REL
from chladni.plan import HEADROOM, Plan, plan_for
from chladni.reference import L_shape_eigs

apex = Polygon(np.array([0, 1, 0.1 + 0.25j]), bc="dir")
p0 = plan_for(apex, 15, 1e-10, 0)
light = Plan(prec=1e-10, k_max=p0.k_max, corner_orders=p0.corner_orders, lightning_n=10, rung=3)
apex_ref = [288.9643854097, 451.0919790235, 628.4568669251, 821.3823602455, 911.5895344006,
            1043.9357242892, 1216.3832526905, 1293.9127902104, 1504.8635788596, 1579.2988086407,
            1788.5259323635, 1842.6310031747, 1922.8118975078, 2106.9668363896, 2220.6128167775]
cases = [("apex lightning", apex, light.solver(apex), apex_ref),
         ("L-shape rung 0", L_shape(), plan_for(L_shape(), 15, 1e-10, 0).solver(L_shape()),
          list(L_shape_eigs(25)))]
prec_b = 10 * TAU_REL
for name, dom, s, ref in cases:
    lo = 0.5 * float(faber_krahn(dom))
    hi = float(asymp.weyl_est_poly(15 + HEADROOM, dom)) * 1.15
    obj = s.objective(lo, hi)
    view = PolicyView(obj, Policy.penalty())
    t = time.time()
    res = bracket_interval(view, prec=prec_b)
    d = res.diagnostics
    br = [(obj.in_lam(b.lo), obj.in_lam(b.hi)) for b in res.brackets]
    inside = [sum(1 for r in ref if a <= r <= b) for a, b in br]
    refs_in_window = [r for r in ref if lo <= r <= hi]
    missed = [r for r in refs_in_window if not any(a <= r <= b for a, b in br)]
    print(f"{name}: window [{lo:.1f}, {hi:.1f}] declined={d.declined} {d.decline_reason[:80]}")
    print(f"   {len(br)} brackets, holding {inside}; reference values in window {len(refs_in_window)}, "
          f"missed {missed}; evals {obj.n_evals} ({d.evaluations}); {time.time() - t:.0f}s; "
          f"floor_limited {d.floor_limited}")
    w = [(b - a) / b for a, b in br]
    print(f"   bracket widths rel: median {np.median(w):.1e} max {np.max(w):.1e}")
