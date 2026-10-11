"""The douse benchmark re-scored with Phase 3's resolution and count_shortfall.  (2026-10-11)

No re-solving: `notes/margin_rule_offline{,_A}.json` hold every level's value, multiplicity,
bound, `met` and count flags (truncation, basis D and A, prec 1e-10 and 1e-8, five windows per
case). Phase 3's quantities follow from those:

    resolution      = max(prec * lam, bound)      (no collapsed groups: `collapse=False`)
    count_shortfall = not met

Two scorings of the same solves:

    old   `benchmarks/douse/scoreboard.score_chladni`: match at 1e-6 relative, count flags only
    new   the contract of `tests/test_level_resolution.py`: per group of levels whose resolution
          intervals overlap, the total matches the trusted reference levels whose own interval
          (value +- reference bound) meets the union, or a member is flagged. A trusted level
          covered by no group is a DROP (Phase 4's).

    .venv/bin/python notes/phase3_rescore.py
"""
import json
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "douse"))


def trusted_and_top(ref):
    trusted = []
    for T in ref["levels"]:
        if T["status"] == "provisional":
            break
        trusted.append(T)
    if not trusted:
        return [], None
    top = trusted[-1]["lam"] * (1 + 1e-4)
    if len(trusted) < len(ref["levels"]):
        top = np.sqrt(trusted[-1]["lam"] * ref["levels"][len(trusted)]["lam"])
    return trusted, top


def new_score(run, ref):
    """`(outcome, n_drops)`: outcome is ok / flagged / SILENT over the groups; drops apart."""
    trusted, top = trusted_and_top(ref)
    if not trusted:
        return "unscored", 0
    if run.get("declined"):
        return "declined", 0
    p = run["prec"]
    lv = sorted((dict(L, res=max(p * abs(L["lam"]), L["bound"] if np.isfinite(L["bound"]) else 0.0),
                      short=not L["met"]) for L in run["levels"] if L["lam"] < top),
                key=lambda L: L["lam"])
    groups = []
    for L in lv:
        if groups and L["lam"] - L["res"] <= groups[-1][-1]["lam"] + groups[-1][-1]["res"]:
            groups[-1].append(L)
        else:
            groups.append([L])
    all_lam = np.array([T["lam"] for T in ref["levels"]])
    # A reference level is an INTERVAL, its value +- its own bound: "certified" certifies the
    # count, not the value to 1e-10 (T1.3's reference bound is 4e-6 relative).
    all_bar = np.array([T["bound"] if np.isfinite(T.get("bound", np.nan)) else 0.0
                        for T in ref["levels"]])
    covered = np.zeros(len(all_lam), bool)
    worst = "ok"
    for g in groups:
        lo = min(L["lam"] - L["res"] for L in g)
        hi = max(L["lam"] + L["res"] for L in g)
        inside = np.flatnonzero((all_lam + all_bar >= lo) & (all_lam - all_bar <= hi))
        covered[inside] = True
        if any(ref["levels"][i]["status"] == "provisional" for i in inside):
            continue
        if sum(ref["levels"][i]["mult"] for i in inside) != sum(L["mult"] for L in g):
            flagged = any(L["short"] or L["count_flag"] for L in g)
            worst = "SILENT" if not flagged else ("flagged" if worst == "ok" else worst)
    drops = sum(1 for i, T in enumerate(ref["levels"][:len(trusted)]) if not covered[i])
    if drops and worst == "ok":
        worst = "DROP"              # the counts that were read are right; a level is missing
    return worst, drops


def main():
    import scoreboard as SB
    refs = {r["id"]: r for r in json.load(open(os.path.join(ROOT, "benchmarks", "douse",
                                                            "references.json")))["cases"]}
    for name in ("margin_rule_offline.json", "margin_rule_offline_A.json"):
        basis = "A" if name.endswith("_A.json") else "D"
        runs = json.load(open(os.path.join(HERE, name)))["runs"]
        for p in (1e-10, 1e-8):
            old, new, drops, silent_new = Counter(), Counter(), 0, []
            for r in runs:
                if r["prec"] != p or "error" in r:
                    continue
                ref = refs[r["id"]]
                sc = SB.score_chladni(r, ref)
                o = sc["outcome"]
                old[o if o != "wrong" else ("SILENT" if sc.get("silent") else "flagged")] += 1
                o2, d = new_score(r, ref)
                new[o2] += 1
                drops += d
                if o2 == "SILENT":
                    silent_new.append((r["id"], tuple(r["jit"])))
            print(f"basis {basis}, prec {p:g}:")
            print(f"  old (1e-6 match, count flags):        {dict(old)}")
            print(f"  new (resolution groups, + shortfall): {dict(new)};  drops {drops}")
            for x in silent_new:
                print(f"    still SILENT: {x}")


if __name__ == "__main__":
    main()
