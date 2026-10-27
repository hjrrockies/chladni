"""Where the lightning basis's tension jumps, and why (2026-10-26). Follows
`notes/rank_freeze_check.py`: a rank frozen per cell did not remove the jumps. This takes the
steepest adjacent pair on a 401-point grid of the frozen pencil over the lam_8 cell, bisects
it toward the jump, and at each step reports the tension, the rank-r and rank-(r+1) singular
values of the equilibrated stack, and how much the RAW collocation matrices moved, split by
basis part.
usage: python notes/rank_freeze_jump.py [a b rank]
"""
import sys
import numpy as np
import scipy.linalg as la
from chladni.geometry import Polygon
from chladni.mps import MPSEigensolver
from chladni.plan import Plan, plan_for

prec = 1e-10
a, b, r = (float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 \
    else (1283.296895, 1304.685177, 120)
dom = Polygon(np.array([0, 1, 0.1 + 0.25j]), bc="dir")
p0 = plan_for(dom, 15, prec, 0)
plan = Plan(prec=prec, k_max=p0.k_max, corner_orders=p0.corner_orders, lightning_n=10, rung=3)
basis, n_bdry, n_int = plan.realize(dom)
s = MPSEigensolver(dom, basis=basis, n_bdry=n_bdry, n_int=n_int, prec=prec, rng=0, rank=r)
pen = s.pencil
xs = np.linspace(a, b, 401)
T = s.tension(xs)
d = np.abs(np.diff(T)) / np.diff(xs)
i = int(np.argmax(d))
print(f"median |slope| {np.median(d):.3g}; steepest pair {xs[i]:.10f} .. {xs[i+1]:.10f}: "
      f"{T[i]:.6g} -> {T[i+1]:.6g} (slope {d[i]:.3g})")
# part boundaries in the column space
parts, edges = [], [0]
for f in getattr(basis, "parts", getattr(basis, "families", [])):
    parts.append(type(f).__name__)
    edges.append(edges[-1] + len(f))
lo, hi = xs[i], xs[i + 1]
for step in range(40):
    m = 0.5 * (lo + hi)
    tl, tm, th = (float(s.tension(x)) for x in (lo, m, hi))
    if abs(tm - tl) > abs(th - tm):
        hi = m
    else:
        lo = m
    if step % 8 == 7 or hi - lo < 1e-12 * lo:
        Bl, Il = pen.matrices(lo)
        Bh, Ih = pen.matrices(hi)
        rel = lambda X, Y: np.max(np.abs(X - Y), axis=0) / np.maximum(np.max(np.abs(X), axis=0), 1e-300)
        cb, ci = rel(Bl, Bh), rel(Il, Ih)
        by_part = [f"{parts[j] if parts else j}: {max(cb[e0:e1].max(), ci[e0:e1].max()):.2e}"
                   for j, (e0, e1) in enumerate(zip(edges[:-1], edges[1:]))] if parts else \
                  [f"all: {max(cb.max(), ci.max()):.2e}"]
        sl = la.svdvals(np.vstack(pen.pencil(lo)))
        sh = la.svdvals(np.vstack(pen.pencil(hi)))
        print(f"  width {hi - lo:.2e}: T {float(s.tension(lo)):.6g} | {float(s.tension(hi)):.6g}; "
              f"s_r/s_1 {sl[r-1]/sl[0]:.3e}|{sh[r-1]/sh[0]:.3e}, s_r+1/s_1 {sl[r]/sl[0]:.3e}|"
              f"{sh[r]/sh[0]:.3e}; raw column change {'; '.join(by_part)}", flush=True)
    if hi - lo < 1e-12 * lo:
        break
