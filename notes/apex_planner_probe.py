"""Per-rung probe of the planner on one triangle (2026-10-26): which rung, which levels, how far.

usage: python notes/apex_planner_probe.py ax ay [n] [prec] [rungs...]
"""
import json, sys, time
import numpy as np
from chladni.geometry import Polygon
from chladni.eigenproblem import Eigenproblem, _met_fraction
from chladni.plan import plan_for

ax, ay = float(sys.argv[1]), float(sys.argv[2])
n = int(sys.argv[3]) if len(sys.argv) > 3 else 15
prec = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-10
rungs = [int(r) for r in sys.argv[5:]] or list(range(5))
v = np.array([0, 1, ax + 1j * ay])
dom = Polygon(v, bc="dir")
ep = Eigenproblem(dom)
print("corners:", [(round(np.degrees(c.omega), 2), c.is_reflection_closed) for c in dom.corners])
out = {"apex": [ax, ay], "n": n, "prec": prec, "rungs": {}}
for r in rungs:
    p = plan_for(dom, n, prec, r)
    if p.n_cols == 0:
        continue
    t = time.time()
    res, win = ep._solve_with(p, n, prec, "flag")
    dt = time.time() - t
    lv = sorted(res.levels, key=lambda L: L.lam)
    d = res.diagnostics
    rows = [(float(L.lam), int(L.mult), float(L.bound / L.lam), float(L.margin)) for L in lv]
    out["rungs"][r] = dict(cols=p.n_cols, orders=p.corner_orders, n_src=p.n_sources,
                           k_max=p.k_max, time=dt, met=_met_fraction(res, n, prec),
                           declined=bool(d.declined), reason=d.decline_reason, levels=rows)
    print(f"rung {r}: cols {p.n_cols} orders {p.corner_orders} src {p.n_sources} "
          f"{dt:.0f}s met {_met_fraction(res, n, prec):.2f} declined {d.declined} {d.decline_reason}")
    for row in rows:
        print("   lam %.10f m%d relbound %.2e margin %.3g" % row)
    sys.stdout.flush()
json.dump(out, open(f"notes/apex_probe_{ax}_{ay}.json", "w"), indent=1)
