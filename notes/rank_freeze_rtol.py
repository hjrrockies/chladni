"""The lightning basis's tension at a looser truncation cutoff (2026-10-26). Follows
`notes/rank_freeze_jump.py`: the jumps are rounding, at a cut that sits inside a continuum of
singular values near the default `rtol = sqrt(prec eps)` ~ 1.5e-13. For each `rtol` (threshold
rank), over the lam_8 cell: roughness (max over median |secant| on a 401-point grid, and the seed
bootstrap of `notes/rank_freeze_check.py`), the rank range, and `search_interval` over the cell
and over the lam_1 / lam_15 cells (relative bound, margin).
usage: python notes/rank_freeze_rtol.py rtol [rtol ...]
"""
import sys, time
import numpy as np
from chladni.geometry import Polygon
from chladni.mps import MPSEigensolver
from chladni.plan import Plan, plan_for

prec = 1e-10
dom = Polygon(np.array([0, 1, 0.1 + 0.25j]), bc="dir")
p0 = plan_for(dom, 15, prec, 0)
plan = Plan(prec=prec, k_max=p0.k_max, corner_orders=p0.corner_orders, lightning_n=10, rung=3)
basis, n_bdry, n_int = plan.realize(dom)
cells = {"lam_1": (286.219228, 292.060437), "lam_8": (1283.296895, 1304.685177),
         "lam_15": (2210.439310, 2245.525648)}
refs = {"lam_1": 288.9643854029, "lam_8": 1293.9127901997, "lam_15": 2220.6128169992}


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


for rtol in (float(x) for x in sys.argv[1:]):
    s = MPSEigensolver(dom, basis=basis, n_bdry=n_bdry, n_int=n_int, prec=prec, rng=0, rtol=rtol)
    a, b = cells["lam_8"]
    xs = np.linspace(a, b, 401)
    T = s.tension(xs)
    d = np.abs(np.diff(T)) / np.diff(xs)
    ranks = [int(s.rank_diagnostics(x)[0]) for x in xs[::20]]
    out = [f"rtol {rtol:.1e}: lam_8 cell rank {min(ranks)}-{max(ranks)}, max/median secant "
           f"{d.max() / np.median(d):.1f}, seed {'settles' if seed(s, a, b) else 'NO'}"]
    for name, (lo, hi) in cells.items():
        t = time.time()
        res = s.search_interval(lo, hi, on_shortfall="flag")
        L = min(res.levels, key=lambda L: abs(L.lam - refs[name]), default=None)
        out.append(f"{name}: " + ("DECLINED" if res.diagnostics.declined else
                   "none" if L is None else
                   f"err {abs(L.lam - refs[name]) / refs[name]:.1e} relb {L.bound / L.lam:.1e} "
                   f"mg {L.margin:.3g}") + f" ({time.time() - t:.0f}s)")
    print(" | ".join(out), flush=True)
