"""Phase 4 step 1: does a Weyl count audit catch dropped eigenvalues on real solves?  (2026-10-11)

The audit compares the corner-corrected Weyl count (`asymp.weyl_count_poly`) against how many
returned eigenvalues lie below each mid-gap cut. Two statistics:

    worst   the largest deficit over the cuts -- `asymp.weyl_deficit`, douse's guard today
    mean    the mean deficit over the cuts. Calibrated IN-SAMPLE on the 31 complete reference
            spectra (TODO, Phase 4): complete spectra in [-0.07, 0.04]; a simulated whole-level
            drop >= 0.16 when it lies in the lower two-thirds of the spectrum.

Each statistic is scored with and without the window's top as an extra cut. The benchmark
windows end at a mid-gap cut, so that cut is legitimate; a `solve(n)` that returns the whole
level at `n` has one too.

Part A, out of sample: the real solves in `notes/admission_bound.json` (truncation; bases D and
A; prec 1e-10 and 1e-8; five windows per case). Each solve is labelled INDEPENDENTLY of every
MPS basis by the reference's finite-element count below the window's top
(`fem_count_below_hi`): COMPLETE when the returned total equals it, SHORT when below, EXTRA when
above.

Part B, headroom: on the reference spectra, how many levels past the last one a caller consumes
the audit needs before a drop near the top is caught.

    .venv/bin/python notes/count_audit.py
"""
import json
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "douse"))
warnings.simplefilter("ignore")

RTOL = 1e-6          # values this close are one level, as in `weyl_deficit`


def deficits(eigs, dom, top=None):
    """The Weyl deficit at the mid-gap cut between each pair of distinct levels, and at `top`."""
    from chladni.asymp import weyl_count_poly
    e = np.sort(np.asarray(eigs, float))
    levels = [e[0]]
    for x in e[1:]:
        if abs(x - levels[-1]) > RTOL * max(abs(x), 1.0):
            levels.append(x)
    cuts = [0.5 * (a + b) for a, b in zip(levels, levels[1:])]
    if top is not None:
        cuts.append(top)
    return np.array([float(weyl_count_poly(c, dom)) - np.sum(e <= c) for c in cuts])


STATS = {
    "worst": lambda d: d.max(),
    "mean": lambda d: d.mean(),
}


def part_a():
    import make_references as MR
    cases = {c["id"]: c for c in MR.load_cases()}
    refs = {r["id"]: r for r in json.load(open(os.path.join(ROOT, "benchmarks", "douse",
                                                            "references.json")))["cases"]}
    runs = json.load(open(os.path.join(HERE, "admission_bound.json")))["runs"]
    doms = {}
    rows = []
    for r in runs:
        if "levels" not in r or r.get("declined") or not r["levels"]:
            continue
        ref = refs[r["id"]]
        lo, hi = ref["window"][0] * (1 + r["jit"][0]), ref["window"][1] * (1 + r["jit"][1])
        eigs = np.concatenate([[L["lam"]] * L["mult"] for L in r["levels"]])
        if eigs.size < 2:
            continue
        n_fem = ref["fem_count_below_hi"]
        label = ("COMPLETE" if eigs.size == n_fem else "SHORT" if eigs.size < n_fem else "EXTRA")
        dom = doms.setdefault(r["id"], MR.polygon(cases[r["id"]]["vertices"]))
        row = dict(id=r["id"], basis=r["basis"], prec=r["prec"], jit=r["jit"], label=label,
                   missing=n_fem - eigs.size, n=eigs.size)
        for top_name, top in (("", None), ("+top", hi)):
            d = deficits(eigs, dom, top)
            for k, f in STATS.items():
                row[k + top_name] = float(f(d))
        rows.append(row)

    labels = sorted({x["label"] for x in rows})
    print(f"Part A: {len(rows)} real solves; labels "
          + ", ".join(f"{lab} {sum(x['label'] == lab for x in rows)}" for lab in labels))
    for k in ("worst", "worst+top", "mean", "mean+top"):
        comp = np.array([x[k] for x in rows if x["label"] == "COMPLETE"])
        short = np.array([x[k] for x in rows if x["label"] == "SHORT"])
        print(f"\n  {k:10s} COMPLETE quantiles 0/50/99/100: "
              f"{np.round(np.percentile(comp, [0, 50, 99, 100]), 3)}")
        print(f"  {'':10s} SHORT    quantiles 0/10/50:     {np.round(np.percentile(short, [0, 10, 50]), 3)}")
        for th in ((0.5, 0.75, 1.0) if k.startswith("worst") else (0.06, 0.1, 0.15)):
            print(f"    threshold {th:5.2f}: false alarms {np.sum(comp > th):4d}/{len(comp)}   "
                  f"drops caught {np.sum(short > th):3d}/{len(short)}")
    print("\n  SHORT solves, by the mean+top statistic (lowest first):")
    for x in sorted((x for x in rows if x["label"] == "SHORT"), key=lambda x: x["mean+top"])[:12]:
        print(f"    {x['id']:20s} {x['basis']} {x['prec']:.0e} jit {str(x['jit']):22s} missing "
              f"{x['missing']}  mean+top {x['mean+top']:.3f}  worst+top {x['worst+top']:.2f}")
    print("\n  COMPLETE solves, highest mean+top:")
    for x in sorted((x for x in rows if x["label"] == "COMPLETE"), key=lambda x: -x["mean+top"])[:6]:
        print(f"    {x['id']:20s} {x['basis']} {x['prec']:.0e} jit {str(x['jit']):22s} "
              f"mean+top {x['mean+top']:.3f}")
    extra = [x for x in rows if x["label"] == "EXTRA"]
    if extra:
        print("\n  EXTRA solves (more returned than FEM counts):")
        for x in extra:
            print(f"    {x['id']:20s} {x['basis']} {x['prec']:.0e} jit {str(x['jit']):22s} "
                  f"extra {-x['missing']}  mean+top {x['mean+top']:.3f}")
    json.dump(rows, open(os.path.join(HERE, "count_audit.json"), "w"), indent=1)


def part_b(threshold=0.1):
    """Drops near the top: the audit sees levels up to the n-th plus `h` more; a drop among the
    first n is what a caller consuming n levels cares about."""
    import make_references as MR
    cases = {c["id"]: c for c in MR.load_cases()}
    refs = json.load(open(os.path.join(ROOT, "benchmarks", "douse", "references.json")))["cases"]
    print(f"\nPart B: headroom, mean statistic with the top cut, threshold {threshold}")
    for h in (0, 1, 2, 3, 4):
        caught = trials = false = n_clean = 0
        for r in refs:
            tr = []
            for T in r["levels"]:
                if T["status"] == "provisional":
                    break
                tr.append(T)
            if len(tr) < 4 + h:
                continue
            dom = MR.polygon(cases[r["id"]]["vertices"])
            seen = tr[:len(tr)]                 # the audit sees all of these
            consumed = len(tr) - h              # the caller uses the first `consumed` levels
            top = (0.5 * (tr[-1]["lam"] + r["levels"][len(tr)]["lam"])
                   if len(tr) < len(r["levels"]) else r["window"][1])
            full = np.concatenate([[T["lam"]] * T["mult"] for T in seen])
            n_clean += 1
            false += deficits(full, dom, top).mean() > threshold
            for i in range(consumed):
                e = np.concatenate([[X["lam"]] * X["mult"] for j, X in enumerate(seen) if j != i])
                trials += 1
                caught += deficits(e, dom, top).mean() > threshold
        print(f"  headroom {h}: drops among the consumed levels caught {caught}/{trials} "
              f"({caught / max(trials, 1):.0%}); false alarms on complete {false}/{n_clean}")


if __name__ == "__main__":
    part_a()
    part_b()
