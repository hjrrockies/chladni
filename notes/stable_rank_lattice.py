"""The stable rank over a whole search window, on the lattice of `notes/triangle_apex_planner.md`
section 8 (2026-10-27). Apex (0.1, 0.25), shipped rung 3 (FB 1x + 10 lightning poles per corner,
180 columns). The window is the planner's prefix window for n = 15 (`Eigenproblem._solve_with`:
half Faber-Krahn up to the padded Weyl estimate of level 19). Cells sit on a geometric lattice
anchored at lam = 1 with ratio 1 + delta, delta = 2 / r_top (r_top = the threshold rank at the
window's top), so a cell's rank is a pure function of lam. At each cell's two ends and middle:
the threshold rank, and the stable rank (leading singular values that move < TOL relative under
lam -> lam (1 + ETA), capped at the threshold rank); the cell rank is the minimum stable rank of the
three. Three example cells (those holding lam_1, lam_8, lam_15) are also sampled at 41 points
each, to show how the stable rank moves inside a cell. Writes notes/stable_rank_lattice.json.
"""
import json
import numpy as np
import scipy.linalg as la
from chladni import asymp
from chladni.bounds import faber_krahn
from chladni.geometry import Polygon
from chladni.mps import MPSEigensolver
from chladni.plan import Plan, plan_for, HEADROOM

prec, TOL, ETA = 1e-10, 1e-2, 1e-12
dom = Polygon(np.array([0, 1, 0.1 + 0.25j]), bc="dir")
p0 = plan_for(dom, 15, prec, 0)
plan = Plan(prec=prec, k_max=p0.k_max, corner_orders=p0.corner_orders, lightning_n=10, rung=3)
basis, nb, ni = plan.realize(dom)
s = MPSEigensolver(dom, basis=basis, n_bdry=nb, n_int=ni, prec=prec, rng=0)
lo = 0.5 * float(faber_krahn(dom))
hi = float(asymp.weyl_est_poly(15 + HEADROOM, dom)) * 1.15
refs = [288.9643854029, 451.0919790235, 628.4568669251, 821.3823602455, 911.5895344006,
        1043.9357242892, 1216.3832526905, 1293.9127902104, 1504.8635788596, 1579.2988086407,
        1788.5259323635, 1842.6310031747, 1922.8118975078, 2106.9668363896, 2220.6128167775]


def ranks(lam):
    s1 = la.svdvals(np.vstack(s.pencil.pencil(lam)))
    s2 = la.svdvals(np.vstack(s.pencil.pencil(lam * (1 + ETA))))
    thr = int(s.rank_diagnostics(lam)[0])
    moved = np.abs(s1 - s2) / s1 > TOL
    stable = int(np.argmax(moved)) if moved.any() else len(s1)
    return thr, min(stable, thr)


r_top = ranks(hi)[0]
delta = 2.0 / r_top
q = np.log1p(delta)
j0, j1 = int(np.floor(np.log(lo) / q)), int(np.ceil(np.log(hi) / q))
cells = []
for j in range(j0, j1):
    a, b = float(np.exp(j * q)), float(np.exp((j + 1) * q))
    nodes = [a, float(np.sqrt(a * b)), b]
    rr = [ranks(x) for x in nodes]
    cells.append(dict(a=a, b=b, nodes=nodes, thr=[r[0] for r in rr], stable=[r[1] for r in rr],
                      rank=min(r[1] for r in rr)))
    print(f"[{a:9.3f}, {b:9.3f}] thr {cells[-1]['thr']} stable {cells[-1]['stable']} -> {cells[-1]['rank']}",
          flush=True)
examples = []
for lam in (refs[0], refs[7], refs[14]):
    c = next(c for c in cells if c["a"] <= lam < c["b"])
    xs = np.geomspace(c["a"], c["b"], 41)
    rr = [ranks(x) for x in xs]
    examples.append(dict(lam=lam, a=c["a"], b=c["b"], x=xs.tolist(), thr=[r[0] for r in rr],
                         stable=[r[1] for r in rr], rank=c["rank"], node_stable=c["stable"]))
json.dump(dict(lo=lo, hi=hi, r_top=r_top, delta=delta, TOL=TOL, ETA=ETA, refs=refs, cells=cells,
               examples=examples), open("notes/stable_rank_lattice.json", "w"))
print("done", len(cells), "cells, delta", delta)
