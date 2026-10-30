"""plan_and_solve along a line of triangle apexes, stock vs. a wider short-edge rule (2026-10-26).

usage: python notes/apex_line.py ax ay [short_edge] [coarse]   (short_edge overrides
chladni.plan.SHORT_EDGE in this process only; "-" keeps it; "coarse" plans with
Eigenproblem(coarse_bracket=True), stage-policy step 5)
"""
import json, sys, time
import numpy as np
import chladni.plan as cp
from chladni.geometry import Polygon
from chladni.eigenproblem import Eigenproblem

ax, ay = float(sys.argv[1]), float(sys.argv[2])
if len(sys.argv) > 3 and sys.argv[3] != "-":
    cp.SHORT_EDGE = float(sys.argv[3])
coarse = len(sys.argv) > 4 and sys.argv[4] == "coarse"
dom = Polygon(np.array([0, 1, ax + 1j * ay]), bc="dir")
t = time.time()
plan, rec = Eigenproblem(dom, coarse_bracket=coarse).plan_and_solve(15, prec=1e-10)
dt = time.time() - t
rel = [L.bound / L.lam for L in rec.levels]
row = dict(apex=[ax, ay], short_edge=cp.SHORT_EDGE, coarse=coarse, rung=plan.rung, cols=plan.n_cols,
           patches=plan.edge_patches, time=dt, met=sum(L.met for L in rec.levels),
           n=len(rec.levels), worst_rel=max(rel), median_rel=float(np.median(rel)),
           min_margin=min(L.margin for L in rec.levels), lam=[L.lam for L in rec.levels],
           rel=rel, flags=list(rec.flags),
           angles=[float(np.degrees(c.omega)) for c in dom.corners],
           edges=[float(s.length) for b in dom.boundaries for s in b.segments])
print(json.dumps(row))
