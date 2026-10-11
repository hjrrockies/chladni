"""Phase 4 step 5: the shipped count audit on the douse benchmark's real solves.  (2026-10-11)

`asymp.count_audit` exactly as `MPSEigensolver.search_interval` calls it: the window's top as a
COARSE check only (an arbitrary `b` need not sit in a gap), the mean over the cuts between
returned levels. The solves are `notes/admission_bound.json` (truncation; bases D and A; 1e-10 and
1e-8; five windows). Labels:

    FEM      the reference's finite-element count below the window's top: SHORT / COMPLETE / EXTRA
    DROP     a trusted reference level no returned level covers, by Phase 3's resolution groups
             (`notes/phase3_rescore.py`), which also sees a drop balanced by a duplicate

    .venv/bin/python notes/phase4_rescore.py
"""
import json
import os
import sys
import warnings
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, "benchmarks", "douse"), HERE]
warnings.simplefilter("ignore")


def main():
    import make_references as MR
    import phase3_rescore as P3
    from chladni.asymp import count_audit

    cases = {c["id"]: c for c in MR.load_cases()}
    refs = {r["id"]: r for r in json.load(open(os.path.join(ROOT, "benchmarks", "douse",
                                                            "references.json")))["cases"]}
    runs = json.load(open(os.path.join(HERE, "admission_bound.json")))["runs"]
    doms = {}
    table = Counter()
    misses, alarms = [], []
    for r in runs:
        if "levels" not in r or r.get("declined"):
            continue
        ref = refs[r["id"]]
        lo, hi = ref["window"][0] * (1 + r["jit"][0]), ref["window"][1] * (1 + r["jit"][1])
        eigs = np.concatenate([[L["lam"]] * L["mult"] for L in r["levels"]]) if r["levels"] \
            else np.array([])
        dom = doms.setdefault(r["id"], MR.polygon(cases[r["id"]]["vertices"]))
        a = count_audit(eigs, dom, lo=lo, top=hi)
        n_fem = ref["fem_count_below_hi"]
        fem = "SHORT" if eigs.size < n_fem else "EXTRA" if eigs.size > n_fem else "COMPLETE"
        # Phase 3's DROP label needs `met`; reconstruct it from the saved bars
        p = r["prec"]
        for L in r["levels"]:
            L["met"] = L["bound"] <= p * abs(L["lam"])
        p3, n_drops = P3.new_score(dict(r, prec=p), ref)
        truth = "drop" if (fem == "SHORT" or n_drops) else ("extra" if fem == "EXTRA" else "clean")
        table[(truth, a.verdict)] += 1
        if truth == "drop" and a.verdict != "short":
            misses.append((r["id"], r["basis"], p, tuple(r["jit"]), fem, n_drops,
                           round(a.statistic, 3), a.verdict))
        if truth == "clean" and a.verdict not in ("ok", "unaudited"):
            alarms.append((r["id"], r["basis"], p, tuple(r["jit"]), round(a.statistic, 3),
                           a.verdict))
    print("truth x verdict (solves):")
    for k in sorted(table):
        print(f"  {k[0]:6s} -> {k[1]:10s} {table[k]:4d}")
    print(f"\nmissed drops ({len(misses)}):")
    for m in misses:
        print("  ", m)
    print(f"\nalarms on clean solves ({len(alarms)}):")
    for m in alarms:
        print("  ", m)


if __name__ == "__main__":
    main()
