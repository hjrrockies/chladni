"""How faithfully does the fitted surrogate reproduce a real rect(1+delta, 1) tension curve?

The question is whether algorithms can be prototyped on it instead of on real pencils --
the surrogate is microseconds against ~11 ms, so a knob sweep that takes an hour would take
seconds. What matters is not overall fidelity but whether the FEATURES the band failures turn
on survive:

    sigma_1 near the pair   the two dips and the bump between them, which a splitting
                            method has to detect
    the bump height         c*d/2 in the model; if this is wrong the splitting threshold is
                            wrong
    sigma_2                 lappy's recursion gate reads it
    the SWING angle         the surrogate's `matrix(k)` is Hermitian, so its minimising
                            eigenvector exists and can flip at a crossing exactly as the real
                            one does. If that survives, swing methods are prototypable too --
                            which was not obvious, since the surrogate models no geometry.

Exploratory; not tracked.
"""
import sys
import time
import warnings

sys.path.insert(0, "tests")
warnings.simplefilter("ignore")

import numpy as np

import rect_suite as rs
from surrogate import Surrogate, Well


def fit(case, h_frac=0.005):
    """Wells from the truth: location exact, `eps` and `c` measured off the real pencil."""
    p = case.pencil
    lv, mult = np.unique(case.truth, return_counts=True)
    wells = []
    for x, m in zip(lv, mult):
        h = h_frac * abs(x)
        eps = float(p.tensions(x)[m - 1])
        c = 0.5 * (p.tension(x + h) + p.tension(x - h)) / h
        wells.append(Well(float(x), eps=eps, c=float(c), mult=int(m)))
    return Surrogate(wells, n=24, rotate=True)


def modes_of(surr, k):
    M = surr.matrix(k)
    _, V = np.linalg.eigh(M)
    return V[:, 0]


def compare(case, n=400):
    p = case.pencil
    s = fit(case)
    xs = np.linspace(case.a, case.b, n)
    real = np.array([p.tensions(x)[:3] for x in xs])
    surr = np.array([s.tensions(x)[:3] for x in xs])

    def relerr(a, b):
        return np.abs(a - b) / np.maximum(np.abs(a), 1e-300)

    e1, e2 = relerr(real[:, 0], surr[:, 0]), relerr(real[:, 1], surr[:, 1])

    lv = np.sort(case.truth)
    # the pair, where present: the two closest distinct levels
    d = np.diff(np.unique(lv))
    out = dict(all_s1=np.median(e1), max_s1=e1.max(), all_s2=np.median(e2))
    if d.size and d.min() < 1.0:
        i = int(np.argmin(d))
        a_, b_ = np.unique(lv)[i], np.unique(lv)[i + 1]
        dd = b_ - a_
        zr = np.linspace(a_ - 3 * dd, b_ + 3 * dd, 200)
        rz = np.array([p.tension(x) for x in zr])
        sz = np.array([s.tension(x) for x in zr])
        out["near_pair_s1"] = float(np.median(relerr(rz, sz)))
        # the bump between the two dips: what a splitting method must see
        mid = 0.5 * (a_ + b_)
        out["bump_real"] = float(p.tension(mid))
        out["bump_surr"] = float(s.tension(mid))
    return out, s


if __name__ == "__main__":
    print("Fitted surrogate vs real pencil, rect(1+delta, 1), window [10, 88]\n")
    print(f"{'case':>14} {'med |ds1|':>10} {'max |ds1|':>10} {'med |ds2|':>10} "
          f"{'near pair':>10} {'bump real':>11} {'bump surr':>11} {'bump err':>9}")
    for case in rs.family_B():
        o, s = compare(case)
        b_r, b_s = o.get("bump_real", np.nan), o.get("bump_surr", np.nan)
        be = abs(b_s - b_r) / b_r if np.isfinite(b_r) and b_r else np.nan
        print(f"{case.label:>14} {o['all_s1']:10.2e} {o['max_s1']:10.2e} {o['all_s2']:10.2e} "
              f"{o.get('near_pair_s1', np.nan):10.2e} {b_r:11.2e} {b_s:11.2e} {be:9.2e}",
              flush=True)

    print("\n\nSpeed, and does the SWING signal survive?  (case B:d=5.9e-03)")
    case = {c.label: c for c in rs.family_B()}["B:d=5.9e-03"]
    s = fit(case)
    xs = np.linspace(case.a, case.b, 120)
    t0 = time.time(); [case.pencil.tensions(x) for x in xs]; t_real = (time.time()-t0)/120
    t0 = time.time(); [s.tensions(x) for x in xs]; t_surr = (time.time()-t0)/120
    print(f"  tension: real {t_real*1e3:.2f} ms   surrogate {t_surr*1e6:.1f} us   "
          f"speedup {t_real/t_surr:,.0f}x")

    def real_modes(x):
        _, I = case.pencil.pencil(float(x))
        u = I @ case.pencil.coef(float(x), 1)[:, 0]
        return u / np.linalg.norm(u)

    lv = np.unique(case.truth)
    gaps = [(lv[i], lv[i+1]) for i in range(len(lv)-1)]
    print(f"  {len(gaps)} gaps between {len(lv)} distinct levels")
    print(f"  {'per_gap':>8} {'source':>10} {'median ang':>11} {'max ang':>9} "
          f"{'peaks>5med':>11} {'gaps hit':>9}")
    for pg in (6, 24, 96):
        n = int(np.ceil((case.b - case.a) / (26.0 / pg))) + 1
        zz = np.linspace(case.a, case.b, n)
        for tag, fn in (("real", real_modes), ("surrogate", lambda x: modes_of(s, x))):
            U = np.array([fn(x) for x in zz])
            ang = 1.0 - np.abs(np.sum(U[:-1] * U[1:], axis=1))
            mid = 0.5 * (zz[:-1] + zz[1:])
            j = np.nonzero((ang[1:-1] > ang[:-2]) & (ang[1:-1] >= ang[2:]))[0] + 1
            j = j[ang[j] > 5 * np.median(ang)]
            lo_e, hi_e = zz[j], zz[j+1]
            hit = sum(1 for a_, b_ in gaps if np.any((hi_e > a_) & (lo_e < b_)))
            print(f"  {pg if tag=='real' else '':>8} {tag:>10} {np.median(ang):11.2e} "
                  f"{ang.max():9.2e} {len(j):11d} {hit:>4}/{len(gaps)}", flush=True)
