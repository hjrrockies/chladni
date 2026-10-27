"""The equilibrated singular values of the lightning basis's pencil around the cut, at lam_1,
lam_8 and lam_15 (2026-10-26): is there a gap the rank could be cut at, and where?"""
import numpy as np
import scipy.linalg as la
from chladni.geometry import Polygon
from chladni.mps import MPSEigensolver
from chladni.plan import Plan, plan_for

prec = 1e-10
dom = Polygon(np.array([0, 1, 0.1 + 0.25j]), bc="dir")
p0 = plan_for(dom, 15, prec, 0)
plan = Plan(prec=prec, k_max=p0.k_max, corner_orders=p0.corner_orders, lightning_n=10, rung=3)
basis, n_bdry, n_int = plan.realize(dom)
s = MPSEigensolver(dom, basis=basis, n_bdry=n_bdry, n_int=n_int, prec=prec, rng=0)
for lam in (288.9643854029, 1293.9127901997, 1293.9127901997 * (1 + 1e-12), 2220.6128169992):
    sv = la.svdvals(np.vstack(s.pencil.pencil(lam)))
    rel = sv / sv[0]
    gaps = rel[:-1] / rel[1:]
    band = np.where((rel[:-1] > 1e-14) & (rel[:-1] < 1e-9))[0]
    top = band[np.argsort(gaps[band])[-3:]][::-1]
    print(f"lam {lam:.6f}: threshold rank {int(s.rank_diagnostics(lam)[0])}; "
          f"largest log-gaps in (1e-14, 1e-9): "
          + ", ".join(f"after {i + 1}: {rel[i]:.2e}/{rel[i + 1]:.2e} = {gaps[i]:.1f}x" for i in top))
    print("   s/s_1 at 85..130:", " ".join(f"{x:.1e}" for x in rel[84:130:3]))
