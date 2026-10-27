"""Can the stable rank come from what the search already computes? (2026-10-27)

Apex (0.1, 0.25), shipped rung 3 (180 columns). Three questions:
1. PAIR SPACING: the stability test compares spectra at lam and lam (1 + eta). The search's refinement
   samples neighbouring points ever closer. For which eta does the test still split head from tail?
   Stable rank (TOL 1e-2) at lam_8 for eta = 1e-12 .. 1e-5.
2. KNEE: a rank from ONE spectrum (no second evaluation): the first index where the geometric-mean
   decay of sigma over the next K indices falls below RHO, capped at the threshold rank. Compared
   with the stable rank at 41 points in each of the cells holding lam_1, lam_8 and lam_15.
3. COST of one evaluation's parts: assembly (basis evaluation), the stack's QR + SVD, the GSVD.
"""
import json, time
import numpy as np
import scipy.linalg as la
from chladni.geometry import Polygon
from chladni.mps import MPSEigensolver
from chladni.plan import Plan, plan_for

prec, TOL = 1e-10, 1e-2
dom = Polygon(np.array([0, 1, 0.1 + 0.25j]), bc="dir")
p0 = plan_for(dom, 15, prec, 0)
plan = Plan(prec=prec, k_max=p0.k_max, corner_orders=p0.corner_orders, lightning_n=10, rung=3)
basis, nb, ni = plan.realize(dom)
s = MPSEigensolver(dom, basis=basis, n_bdry=nb, n_int=ni, prec=prec, rng=0)
sv = lambda lam: la.svdvals(np.vstack(s.pencil.pencil(lam)))
thr = lambda lam: int(s.rank_diagnostics(lam)[0])


def stable(lam, eta=1e-12):
    a, b = sv(lam), sv(lam * (1 + eta))
    m = np.abs(a - b) / a > TOL
    return min(int(np.argmax(m)) if m.any() else len(a), thr(lam))


def knee(lam, K=4, RHO=1.3, FLOOR=1e-8, s1=None):
    """Only below FLOOR x sigma_1: the top of the spectrum is flat too, and is not a tail."""
    v = sv(lam) if s1 is None else s1
    x = np.log(v / v[0])
    dec = (x[:-K] - x[K:]) / K                     # mean log-decay over the next K indices
    bad = (dec < np.log(RHO)) & (x[:-K] < np.log(FLOOR))
    return min(int(np.argmax(bad)) if bad.any() else len(x), thr(lam))


out = {}
lam8 = 1293.9127901997
out["eta"] = {f"{e:.0e}": stable(lam8, e) for e in (1e-12, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)}
print("1. stable rank at lam_8 vs pair spacing eta:", out["eta"], flush=True)

cells = {"lam_1": (285.152, 289.904), "lam_8": (1283.297, 1304.685), "lam_15": (2214.218, 2251.122)}
out["cells"] = {}
for name, (a, b) in cells.items():
    xs = np.geomspace(a, b, 41)
    st = [stable(x) for x in xs]
    row = {"stable": st, "thr": [thr(x) for x in xs]}
    for K, RHO in ((4, 1.25), (4, 1.4), (6, 1.25), (8, 1.25)):
        row[f"knee K{K} rho{RHO}"] = [knee(x, K, RHO) for x in xs]
    out["cells"][name] = row
    print(f"2. {name}: stable {min(st)}-{max(st)} | " + " | ".join(
        f"{k} {min(v)}-{max(v)}" for k, v in row.items() if k.startswith("knee")) +
        f" | thr {min(row['thr'])}-{max(row['thr'])}", flush=True)

pen = s.pencil
t0 = time.perf_counter(); [pen.matrices(lam8 * (1 + 1e-9 * i)) for i in range(20)]; ta = (time.perf_counter() - t0) / 20
B, I = pen.pencil(lam8)
t0 = time.perf_counter(); [la.svd(la.qr(np.vstack([B, I]), mode="economic")[1]) for _ in range(20)]; tq = (time.perf_counter() - t0) / 20
t0 = time.perf_counter(); [s.tension(lam8 * (1 + 1e-9 * (i + 100))) for i in range(20)]; tt = (time.perf_counter() - t0) / 20
out["cost_ms"] = dict(assembly=1e3 * ta, qr_svd=1e3 * tq, whole_tension=1e3 * tt)
print("3. cost per evaluation (ms):", {k: round(v, 1) for k, v in out["cost_ms"].items()})
json.dump(out, open("notes/stable_rank_free.json", "w"))
