"""Read `notes/bound_coverage.jsonl` (written by `bound_coverage.py`) into the tables of
`notes/bound_coverage.md`.

    .venv/bin/python notes/bound_coverage_read.py
"""
import json
import os
from collections import defaultdict

import numpy as np

rows = [json.loads(s) for s in open(os.path.join(os.path.dirname(__file__),
                                                  "bound_coverage.jsonl"))]
summ = [r for r in rows if r.get("row") == "summary"]
lev = [r for r in rows if r.get("row") == "level"]
errs = [r for r in rows if "error" in r]

print(f"{len(summ)} solves, {len(lev)} levels, {len(errs)} crashes")
for r in errs:
    print("  CRASH", r)

# 1. Per solve: count and coverage.
print("\n== per solve: found/truth, uncovered by bound / by delta_res / by mp ==")
by = defaultdict(list)
for r in lev:
    by[(r["domain"], r["kind"], r["n"], r["prec"])].append(r)
for s in summ:
    k = (s["domain"], s["kind"], s["n"], s["prec"])
    L = by.get(k, [])
    nb = sum(not r["covers"] for r in L)
    nd = sum(not r["dr_covers"] for r in L)
    nm = sum(np.isfinite(r["mp_bound"]) and not r["mp_covers"] for r in L)
    worst = max((r["err"] / r["bound"] for r in L), default=float("nan"))
    print(f"{s['domain']:9s} {s['kind']:12s} n={s['n']:<4} ({s['n_cols']:3d} cols) "
          f"prec={s['prec']:.0e}  {s['n_found']:2d}/{s['n_truth']:2d}  "
          f"uncov bound {nb:2d} dr {nd:2d} mp {nm:2d}  worst err/bound {worst:8.1e}  "
          f"attained {s['attained']:.1e}")

# 2. Does disagreement between the two bars predict delta_res undercover?
fin = [r for r in lev if np.isfinite(r["mp_bound"]) and r["delta_res"] > 0]
ratio = np.array([r["mp_bound"] / r["delta_res"] for r in fin])
dr_ok = np.array([r["dr_covers"] for r in fin])
print(f"\n== mp_bound / delta_res, {len(fin)} levels with both ==")
print(f"delta_res covers: n={dr_ok.sum()}, ratio quantiles (0, 50, 90, 99, 100%) "
      f"{np.quantile(ratio[dr_ok], [0, .5, .9, .99, 1]) if dr_ok.any() else []}")
print(f"delta_res FAILS:  n={(~dr_ok).sum()}, ratio quantiles "
      f"{np.quantile(ratio[~dr_ok], [0, .5, .9, .99, 1]) if (~dr_ok).any() else []}")
for thr in (3, 10, 30, 100, 300, 1000):
    guard = ratio > thr          # distrust delta_res there
    missed = (~dr_ok) & ~guard   # an undercover the guard would not catch
    print(f"  guard at ratio>{thr:5d}: catches {((~dr_ok) & guard).sum():3d} of {(~dr_ok).sum()}"
          f" failures, misses {missed.sum():3d}; demotes {(dr_ok & guard).sum():3d} good levels")

# 3. err / delta_res for the failures, and how it relates to the ratio.
print("\n== delta_res failures: err/delta_res vs mp/delta_res ==")
for r in sorted((r for r in fin if not r["dr_covers"]), key=lambda r: r["mp_bound"] / r["delta_res"]):
    print(f"{r['domain']:9s} {r['kind']:12s} n={r['n']:<4} prec={r['prec']:.0e} lam={r['lam']:.6f} "
          f"err/dr {r['err']/r['delta_res']:8.1e} mp/dr {r['mp_bound']/r['delta_res']:8.1e} "
          f"mp_covers={r['mp_covers']} met={r['met']} sigma={r['sigma']:.1e}")

# 4. Moler-Payne failures, which would be the serious finding.
mpf = [r for r in fin if not r["mp_covers"]]
print(f"\n== Moler-Payne failures: {len(mpf)} ==")
for r in mpf:
    print(f"  {r}")
