"""Step 2 cost, measured fairly on a shared machine (2026-10-27): threshold vs coarse-penalty
tension per evaluation, single-threaded BLAS, fresh pencils, the two interleaved over 7 rounds of
8 points each, reporting the MEDIAN round. Assembly alone is timed the same way, so the
factorization's share is visible."""
import time
import numpy as np
from chladni.geometry import L_shape, Polygon
from chladni.mps import MPSPencil
from chladni.mps.policy import Policy
from chladni.plan import Plan, plan_for

apex = Polygon(np.array([0, 1, 0.1 + 0.25j]), bc="dir")
p0 = plan_for(apex, 15, 1e-10, 0)
cases = [("L-shape rung 0", plan_for(L_shape(), 15, 1e-10, 0).solver(L_shape())._pencil, 41.47),
         ("apex lightning", Plan(prec=1e-10, k_max=p0.k_max, corner_orders=p0.corner_orders,
                                 lightning_n=10, rung=3).solver(apex)._pencil, 1293.91),
         ("apex rung 1", plan_for(apex, 15, 1e-10, 1).solver(apex)._pencil, 1293.91)]
for name, p, lam in cases:
    rounds = {"assembly": [], "threshold": [], "coarse": []}
    for r in range(7):
        xs = lam * (1 + 1e-6 * (np.arange(8) + 10 * r + 1))
        for label in (("threshold", "coarse", "assembly") if r % 2 else ("coarse", "threshold", "assembly")):
            q = MPSPencil(p.bdry, p.interior, reg=p.reg, rtol=p.rtol, prec=p.prec)
            t0 = time.perf_counter()
            for x in xs:
                if label == "assembly":
                    q.matrices(x)
                else:
                    q.tensions(x, Policy.penalty() if label == "coarse" else None)
            rounds[label].append((time.perf_counter() - t0) / len(xs))
    med = {k: 1e3 * float(np.median(v)) for k, v in rounds.items()}
    print(f"{name:16s} cols {p.n_cols}: assembly {med['assembly']:.1f} ms, threshold {med['threshold']:.1f}, "
          f"coarse {med['coarse']:.1f} ms/eval ({100 * (med['coarse'] / med['threshold'] - 1):+.0f}%)", flush=True)
