"""Do miscounts sit below a ratio of request to local floor scale?  (2026-10-09)

The question behind "count at a licence the evidence supports": for every TRUE level in the
bound-coverage grid (`notes/bound_coverage.jsonl`: L-shape, 3pi/2 sector, eq. triangle, disk;
starved to generous bases; prec 1e-8..1e-13; pre-(C2) code), compare the count the solver returned
with the true multiplicity, and record

    R = prec * lam / (delta_res / 2)

where `delta_res = DELTA_SAFETY * eps / c` (`mps/refine.py`, DELTA_SAFETY = 2), so `delta_res/2` is
the refiner's own reading of the local floor scale `eps/c`. R >= 2 is exactly "met" (`delta_res <=
prec*lam`). Caveats: `delta_res` is clamped from below (4 ulps, 2*settle) and widened to the bracket
on a stalled refine, so R is a proxy; the emitted bracket's near side, which the 10 pi^2 trace says
matters, was not recorded; a DROPPED level has no level of its own, so it is scored by the nearest
returned level's R and flagged as such.

    .venv/bin/python notes/count_ratio.py
"""
import json
import os
from collections import defaultdict

import numpy as np

from chladni import reference as REF

HERE = os.path.dirname(__file__)
rows = [json.loads(s) for s in open(os.path.join(HERE, "bound_coverage.jsonl"))]
WIN = {"L_shape": (5.0, 100.0), "sector270": (5.0, 100.0), "eq_tri": (40.0, 300.0), "disk": (5.0, 80.0)}


def truth(dom):
    lo, hi = WIN[dom]
    if dom == "L_shape":
        t = np.asarray(REF.L_shape_eigs(25), float)
    elif dom == "sector270":
        t = np.asarray(REF.sector_spectrum(lo, hi, omega=1.5 * np.pi), float)
    elif dom == "eq_tri":
        t = np.asarray(REF.eq_tri_spectrum(lo, hi), float)
    else:
        t = np.asarray(REF.disk_spectrum(lo, hi), float)
    t = np.sort(t[(t > lo) & (t < hi)])
    vals, mult = [], []
    for x in t:                     # distinct values; exact doubles in all four tables
        if vals and abs(x - vals[-1]) <= 1e-9 * x:
            mult[-1] += 1
        else:
            vals.append(float(x)); mult.append(1)
    return np.array(vals), np.array(mult)


solves = defaultdict(list)
summ = {}
for r in rows:
    k = (r["domain"], r["kind"], r["n"], r["prec"])
    if r.get("row") == "level":
        solves[k].append(r)
    elif r.get("row") == "summary":
        summ[k] = r

out = []          # one record per TRUE level per solve
for k, s in summ.items():
    dom, kind, n, prec = k
    vals, mult = truth(dom)
    levels = solves.get(k, [])
    got = np.zeros(len(vals), int)
    R_of = [[] for _ in vals]
    extra_far = 0
    for L in levels:
        j = int(np.argmin(np.abs(vals - L["lam"])))
        # A level is "at" a true value when it is closer to it than to any other: assignment by
        # nearest. A spurious level far from everything still lands somewhere; it is marked.
        gap = np.min(np.abs(np.delete(vals, j) - vals[j])) if len(vals) > 1 else np.inf
        if abs(L["lam"] - vals[j]) > 0.25 * gap:
            extra_far += L["mult"]
            continue
        got[j] += L["mult"]
        R_of[j].append(prec * L["lam"] / (L["delta_res"] / 2.0))
    lam_ret = np.array([L["lam"] for L in levels]) if levels else np.array([])
    for j, (v, m) in enumerate(zip(vals, mult)):
        if R_of[j]:
            R, how = float(min(R_of[j])), "own"
        elif len(lam_ret):
            i = int(np.argmin(np.abs(lam_ret - v)))
            R, how = prec * levels[i]["lam"] / (levels[i]["delta_res"] / 2.0), "nearest"
        else:
            R, how = float("nan"), "none"
        verdict = "ok" if got[j] == m else ("drop" if got[j] == 0 else
                                             ("under" if got[j] < m else "over"))
        out.append(dict(dom=dom, kind=kind, n=n, prec=prec, lam=v, mult=int(m), got=int(got[j]),
                        R=R, how=how, verdict=verdict))
    if extra_far:
        out.append(dict(dom=dom, kind=kind, n=n, prec=prec, lam=float("nan"), mult=0,
                        got=extra_far, R=float("nan"), how="spurious", verdict="spurious"))

json.dump(out, open(os.path.join(HERE, "count_ratio.json"), "w"), indent=0)

ok = [o for o in out if o["verdict"] == "ok" and np.isfinite(o["R"])]
bad = [o for o in out if o["verdict"] in ("under", "over")]
drops = [o for o in out if o["verdict"] == "drop"]
spur = [o for o in out if o["verdict"] == "spurious"]
print(f"{len(out)} true-level records over {len(summ)} solves: {len(ok)} ok, {len(bad)} wrong "
      f"multiplicity, {len(drops)} dropped, {len(spur)} solves with spurious far levels")

q = lambda xs: np.round(np.quantile(xs, [0, 0.05, 0.5, 0.95, 1]), 2) if len(xs) else []
print("\nR = prec*lam/(delta_res/2) quantiles (0, 5, 50, 95, 100 %)")
print("  correct counts            ", q([o["R"] for o in ok]))
print("  correct, true multiplicity >= 2", q([o["R"] for o in ok if o["mult"] >= 2]))
print("  wrong multiplicity        ", q([o["R"] for o in bad]))
print("  dropped (nearest level's R)", q([o["R"] for o in drops if np.isfinite(o["R"])]))

print("\nwrong multiplicities:")
for o in sorted(bad, key=lambda o: -o["R"]):
    print(f"  R={o['R']:9.3g}  {o['dom']:9s} {o['kind']:12s} n={o['n']:<4} prec={o['prec']:.0e} "
          f"lam={o['lam']:.6f} true x{o['mult']} got x{o['got']}")

print("\nthreshold T: wrong (mult + drops-by-own-or-nearest) at R >= T, and correct lost below T")
allbad = bad + [o for o in drops if np.isfinite(o["R"])]
for T in (2, 5, 10, 20, 50, 100, 200, 500, 1000):
    nb = sum(o["R"] >= T for o in allbad)
    lost = sum(o["R"] < T for o in ok)
    lost2 = sum(o["R"] < T for o in ok if o["mult"] >= 2)
    print(f"  T={T:5d}: wrong left above T {nb:3d} of {len(allbad)};  correct below T {lost:3d} of "
          f"{len(ok)}  (of which degenerate {lost2})")

print("\nby (domain, prec): fraction of true levels with R >= 20")
for dom in WIN:
    line = []
    for p in (1e-8, 1e-10, 1e-12, 1e-13):
        xs = [o for o in out if o["dom"] == dom and o["prec"] == p and np.isfinite(o["R"])]
        if xs:
            line.append(f"{p:.0e}: {np.mean([o['R'] >= 20 for o in xs]):.2f}")
    print(f"  {dom:9s} " + "  ".join(line))
