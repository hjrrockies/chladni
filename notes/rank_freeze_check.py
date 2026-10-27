"""Local rank freezing on the lightning basis of the apex-(0.1, 0.25) triangle (2026-10-26).

The check proposed in `notes/triangle_apex_planner.md` section 8, before anything is built. The
basis is the shipped rung 3: FB 1x at every corner plus 10 lightning poles per corner, 180
columns. With the threshold rank it searches lam_1 fine and DECLINES at lam_8 and lam_15
("the slope estimate never settled") on rank flips 119-122. Cells sit on a geometric lattice in
lam anchored at lam = 1, with ratio `1 + delta`, `delta = 2 / r` (r = the threshold rank at the
reference value). Each cell's rank is the MINIMUM threshold rank at its two ends and its middle,
frozen with `MPSEigensolver(rank=)` on the same basis and the same collocation. For each cell
that overlaps +-2.5% of the reference value this compares, threshold against frozen:
- the rank profile: the threshold rank on a 201-point grid over the cell, and its deficit below
  the cell rank;
- the seed bootstrap: max secant over 6 midpoint doublings from 9 points (`mps/bracket._seed`),
  and whether it settled (growth <= 10%);
- `search_interval` on the cell itself: declined or not, levels found, relative bound, margin.

usage: python notes/rank_freeze_check.py lam_ref [delta_factor]   (delta = delta_factor / r)
"""
import json, sys, time
import numpy as np
from chladni.geometry import Polygon
from chladni.mps import MPSEigensolver
from chladni.plan import Plan, plan_for

prec = 1e-10
lam_ref = float(sys.argv[1])
fac = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
dom = Polygon(np.array([0, 1, 0.1 + 0.25j]), bc="dir")
p0 = plan_for(dom, 15, prec, 0)
plan = Plan(prec=prec, k_max=p0.k_max, corner_orders=p0.corner_orders, lightning_n=10, rung=3)
basis, n_bdry, n_int = plan.realize(dom)
thr = MPSEigensolver(dom, basis=basis, n_bdry=n_bdry, n_int=n_int, prec=prec, rng=0)
r_ref = thr.rank_at(lam_ref) if hasattr(thr, "rank_at") else thr.rank_diagnostics(lam_ref)[0]
r_ref = int(thr.rank_diagnostics(lam_ref)[0])
delta = fac / r_ref
q = np.log1p(delta)
j_lo = int(np.floor(np.log(lam_ref * (1 - 0.025)) / q))
j_hi = int(np.ceil(np.log(lam_ref * (1 + 0.025)) / q))


def seed(s, a, b):
    xs = list(np.linspace(a, b, 9))
    vals = dict(zip(xs, s.tension(np.array(xs))))
    hist = []
    for _ in range(6):
        x = np.array(sorted(vals))
        v = np.array([vals[t] for t in x])
        hist.append(float(np.max(np.abs(np.diff(v) / np.diff(x)))))
        if len(hist) > 1 and hist[-1] <= 1.10 * hist[-2]:
            return hist, True
        mids = 0.5 * (x[1:] + x[:-1])
        vals.update(zip(mids, s.tension(mids)))
    return hist, False


def search(s, a, b):
    t = time.time()
    res = s.search_interval(a, b, on_shortfall="flag")
    d = res.diagnostics
    return dict(declined=bool(d.declined), reason=d.decline_reason[:120], time=time.time() - t,
                levels=[(float(L.lam), int(L.mult), float(L.bound / L.lam), float(L.margin))
                        for L in res.levels])


out = dict(lam_ref=lam_ref, r_ref=r_ref, delta=delta, cells=[])
for j in range(j_lo, j_hi):
    a, b = float(np.exp(j * q)), float(np.exp((j + 1) * q))
    grid = np.linspace(a, b, 201)
    r_grid = np.array([int(thr.rank_diagnostics(x)[0]) for x in grid])
    r_cell = int(min(thr.rank_diagnostics(x)[0] for x in (a, 0.5 * (a + b), b)))
    frz = MPSEigensolver(dom, basis=basis, n_bdry=n_bdry, n_int=n_int, prec=prec, rng=0,
                         rank=r_cell)
    cell = dict(a=a, b=b, r_cell=r_cell, r_min=int(r_grid.min()), r_max=int(r_grid.max()),
                n_flips=int(np.sum(np.diff(r_grid) != 0)),
                deficit_max=int(r_grid.max() - r_cell),
                below_cell=int(np.sum(r_grid < r_cell)))
    for name, s in (("threshold", thr), ("frozen", frz)):
        h, ok = seed(s, a, b)
        cell[name] = dict(seed_hist=h, settled=ok, search=search(s, a, b))
    out["cells"].append(cell)
    f, t = cell["frozen"], cell["threshold"]
    print(f"cell [{a:.6f}, {b:.6f}] r_cell {r_cell} thr {cell['r_min']}-{cell['r_max']} "
          f"flips {cell['n_flips']} below {cell['below_cell']} | seed thr "
          f"{'ok' if t['settled'] else 'NO'} {t['seed_hist'][-1]:.3g} frz "
          f"{'ok' if f['settled'] else 'NO'} {f['seed_hist'][-1]:.3g} | search thr "
          f"{'DECL' if t['search']['declined'] else t['search']['levels']} frz "
          f"{'DECL' if f['search']['declined'] else f['search']['levels']}", flush=True)
json.dump(out, open(f"notes/rank_freeze_check_{lam_ref:.0f}_{fac:g}.json", "w"), indent=1)
