"""Stage-policy step 2 gate (2026-10-27): the coarse penalty reduced from the "svd" factorization.
1. agreement with "tikhonov_qr" at the same absolute tau (head of the tension vector);
2. cost per evaluation: threshold tension vs coarse-penalty tension (fresh pencils, so no hits);
3. continuity: max/median |secant| of sigma_1 on a 401-point grid in the apex lightning basis's
   lam_8 cell (threshold: ~10, the defect), and the coarse floor vs plateau at lam_1, lam_8, lam_15.
"""
import time
import numpy as np
from chladni.geometry import L_shape, Polygon
from chladni.mps import MPSPencil, _gsvd
from chladni.mps._regularize import reduce_penalty, regularize, svd_factors
from chladni.mps.policy import Policy
from chladni.plan import Plan, plan_for

cases = []
dom = L_shape(); cases.append(("L-shape rung 0", plan_for(dom, 15, 1e-10, 0).solver(dom)._pencil, 41.47))
apex = Polygon(np.array([0, 1, 0.1 + 0.25j]), bc="dir")
p0 = plan_for(apex, 15, 1e-10, 0)
light = Plan(prec=1e-10, k_max=p0.k_max, corner_orders=p0.corner_orders, lightning_n=10, rung=3)
cases.append(("apex lightning", light.solver(apex)._pencil, 1293.9127901997))
cases.append(("apex rung 1", plan_for(apex, 15, 1e-10, 1).solver(apex)._pencil, 1293.9127901997))
for name, p, lam in cases:
    B, I = p.pencil(lam)
    f = svd_factors(B, I)
    tau = np.sqrt(np.finfo(float).eps) * f.s[0]
    a = _gsvd.tensions_of(reduce_penalty(f, tau))[:10]
    b = _gsvd.tensions_of(regularize(B, I, reg="tikhonov_qr", rtol=tau / np.linalg.norm(np.vstack([B, I]))))[:10]
    fresh = lambda: MPSPencil(p.bdry, p.interior, reg=p.reg, rtol=p.rtol, prec=p.prec)
    xs = lam * (1 + 1e-6 * np.arange(1, 13))
    q = fresh(); t0 = time.perf_counter(); [q.tensions(x) for x in xs]; t_thr = (time.perf_counter() - t0) / len(xs)
    q = fresh(); t0 = time.perf_counter(); [q.tensions(x, Policy.penalty()) for x in xs]; t_pen = (time.perf_counter() - t0) / len(xs)
    print(f"{name:16s} cols {p.n_cols}: max rel diff vs tikhonov_qr (10 smallest) {np.max(np.abs(a - b) / b):.1e}; "
          f"ms/eval threshold {1e3 * t_thr:.1f}, coarse {1e3 * t_pen:.1f} (+{100 * (t_pen / t_thr - 1):.0f}%)", flush=True)

p = cases[1][1]
xs = np.linspace(1283.296895, 1304.685177, 401)
for label, pol in (("threshold", None), ("coarse", Policy.penalty())):
    T = np.array([p.tension(x, pol) for x in xs])
    d = np.abs(np.diff(T)) / np.diff(xs)
    print(f"apex lightning lam_8 cell, {label:9s}: max/median |secant| {d.max() / np.median(d):.1f}, "
          f"min sigma_1 {T.min():.2e}, median {np.median(T):.2e}")
for lam in (288.9643854029, 1293.9127901997, 2220.6128169992):
    w = np.linspace(lam * 0.99, lam * 1.01, 81)
    T = np.array([p.tension(x, Policy.penalty()) for x in w])
    print(f"coarse at lam {lam:.2f}: floor {T.min():.2e} at offset {(w[T.argmin()] - lam) / lam:+.1e} rel, "
          f"floor/plateau {T.min() / np.median(T):.1e}, threshold floor {p.tension(lam):.2e}")
