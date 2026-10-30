"""A STABLE rank, frozen per cell, on the lightning basis (2026-10-26). Follows
`notes/rank_freeze_gap.py`: below a knee near 1e-12 the singular values form a slow tail that
moves ~20% under a 1e-12 relative change in lam -- rounding, not basis content -- and the
threshold cut (1.5e-13) lands inside it. The stable rank at lam keeps the leading directions
whose singular values move by less than `TOL` (relative) between lam and lam (1 + ETA), never
more than the threshold rank. Per cell (the lattice of `notes/rank_freeze_check.py`, delta =
2 / r) the cell's rank is the minimum stable rank at its ends and middle, frozen with
`MPSEigensolver(rank=)`. Reported per cell: stable ranks at the nodes, the threshold rank there,
roughness on a 201-point grid, the seed bootstrap, and `search_interval` over the cell.
usage: python notes/rank_stable_cells.py lam_ref [TOL] [ETA]
"""
import sys, time
import numpy as np
import scipy.linalg as la
from chladni.geometry import Polygon
from chladni.mps import MPSEigensolver
from chladni.plan import Plan, plan_for

prec = 1e-10
lam_ref = float(sys.argv[1])
TOL = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-3
ETA = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-12
dom = Polygon(np.array([0, 1, 0.1 + 0.25j]), bc="dir")
p0 = plan_for(dom, 15, prec, 0)
plan = Plan(prec=prec, k_max=p0.k_max, corner_orders=p0.corner_orders, lightning_n=10, rung=3)
basis, n_bdry, n_int = plan.realize(dom)
thr = MPSEigensolver(dom, basis=basis, n_bdry=n_bdry, n_int=n_int, prec=prec, rng=0)


def stable_rank(lam):
    s1 = la.svdvals(np.vstack(thr.pencil.pencil(lam)))
    s2 = la.svdvals(np.vstack(thr.pencil.pencil(lam * (1 + ETA))))
    moved = np.abs(s1 - s2) / s1 > TOL
    r = int(np.argmax(moved)) if moved.any() else len(s1)
    return min(r, int(thr.rank_diagnostics(lam)[0]))


def seed(s, a, b):
    xs = list(np.linspace(a, b, 9))
    vals = dict(zip(xs, s.tension(np.array(xs))))
    hist = []
    for _ in range(6):
        x = np.array(sorted(vals))
        v = np.array([vals[t] for t in x])
        hist.append(float(np.max(np.abs(np.diff(v) / np.diff(x)))))
        if len(hist) > 1 and hist[-1] <= 1.10 * hist[-2]:
            return True
        mids = 0.5 * (x[1:] + x[:-1])
        vals.update(zip(mids, s.tension(mids)))
    return False


r_ref = int(thr.rank_diagnostics(lam_ref)[0])
q = np.log1p(2.0 / r_ref)
for j in range(int(np.floor(np.log(lam_ref * 0.975) / q)), int(np.ceil(np.log(lam_ref * 1.025) / q))):
    a, b = float(np.exp(j * q)), float(np.exp((j + 1) * q))
    nodes = (a, 0.5 * (a + b), b)
    rs = [stable_rank(x) for x in nodes]
    rt = [int(thr.rank_diagnostics(x)[0]) for x in nodes]
    s = MPSEigensolver(dom, basis=basis, n_bdry=n_bdry, n_int=n_int, prec=prec, rng=0,
                       rank=min(rs))
    xs = np.linspace(a, b, 201)
    T = s.tension(xs)
    d = np.abs(np.diff(T)) / np.diff(xs)
    t = time.time()
    res = s.search_interval(a, b, on_shortfall="flag")
    lv = [f"{L.lam:.10f} relb {L.bound / L.lam:.1e} mg {L.margin:.3g}" for L in res.levels]
    print(f"cell [{a:.4f}, {b:.4f}] stable {rs} threshold {rt} -> rank {min(rs)}; max/median "
          f"secant {d.max() / np.median(d):.1f}; seed {'settles' if seed(s, a, b) else 'NO'}; "
          f"search {'DECLINED' if res.diagnostics.declined else lv} ({time.time() - t:.0f}s)",
          flush=True)
