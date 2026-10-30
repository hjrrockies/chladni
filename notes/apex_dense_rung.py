"""Prefix solve of one candidate rung on a triangle (2026-10-26): FB at the singular corners
at `scale` plus the offset layer at `density` per wavelength. Probe-local: `LADDER` is replaced in
this process only.

usage: python notes/apex_dense_rung.py ax ay scale density [offset]
"""
import json, sys, time
import numpy as np
import chladni.plan as cp
from chladni.geometry import Polygon
from chladni.eigenproblem import Eigenproblem, _met_fraction

ax, ay, scale, dens = (float(x) for x in sys.argv[1:5])
off = float(sys.argv[5]) if len(sys.argv) > 5 else 0.05
cp.LADDER = (cp.Recipe(scale, "singular", dens, off),)
dom = Polygon(np.array([0, 1, ax + 1j * ay]), bc="dir")
p = cp.plan_for(dom, 15, 1e-10, 0)
t = time.time()
res, _ = Eigenproblem(dom)._solve_with(p, 15, 1e-10, "flag")
lv = sorted(res.levels, key=lambda L: L.lam)[:15]
d = res.diagnostics
print(json.dumps(dict(apex=[ax, ay], scale=scale, density=dens, offset=off, cols=p.n_cols,
                      time=time.time() - t, met=_met_fraction(res, 15, 1e-10),
                      declined=bool(d.declined), reason=d.decline_reason,
                      worst_rel=max(L.bound / L.lam for L in lv),
                      min_margin=min(L.margin for L in lv),
                      lam=[L.lam for L in lv])))
