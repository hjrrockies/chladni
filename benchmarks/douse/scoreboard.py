"""The douse scoreboard: lappy and chladni against `references.json`, with a stability score.

    .venv/bin/python benchmarks/douse/scoreboard.py          # after make_references, lappy_runner

Reads `references.json` (make_references.py) and `lappy_runs.json` (lappy_runner.py), runs the
chladni stand-in, and writes `scoreboard.json` and the table in `scoreboard.md`.

THE CHLADNI STAND-IN IS NOT A PLANNER. chladni has none yet (`TODO.md` section 3), so its row is
`MPSEigensolver.solve_interval` on run "D": Fourier-Bessel at every corner at 1.0x lappy's per-corner
sizing, smaller than any reference run (A 1.5x, B 1.2x plus sources, C 2.0x), at `prec` 1e-8
and 1e-13, over the reference window. It measures the search and the certificate on a plausible
basis; what a planner will deliver is not measured here.

Scoring, against the reference levels that are `certified` or `fem_resolved` (a case whose
first levels include a `provisional` one is scored only up to it):

    dropped     reference copies with no returned value within `MATCH` relative
    extra       returned values matching no reference copy (spurious, or a duplicated copy)
    mult_wrong  levels returned with the wrong multiplicity (chladni; lappy returns values only)
    silent      any of the above on a result that raised no flag (lappy: always silent)
    max_err     worst relative error over matched values
    uncovered   chladni levels whose reported `bound` is below the true error
    stable      chladni only: the same levels on all `JITTERS` windows (ulp- to 1e-12-relative
                changes of the window ends); `n_wrong` of them scored wrong
"""
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_references as MR  # noqa: E402

MATCH = 1e-6
MR.SCALE["D"] = 1.0
CHLADNI_PRECS = (1e-8, 1e-10)
#: Window jitters for the STABILITY score: (relative change to lo, relative change to hi). The
#: first is the reference window itself. A count that changes across these is decided by ulps of
#: the window, not by the domain (`notes/nondeterminism.md`: one ulp flipped T1.2's count).
JITTERS = ((0.0, 0.0), (4e-16, -4e-16), (-1e-14, 1e-14), (1e-13, 1e-13), (-1e-12, -1e-12))


def ref_copies(ref):
    """Reference eigenvalues, repeated by multiplicity, up to the first untrusted level."""
    out = []
    for L in ref["levels"]:
        if L["status"] == "provisional":
            break
        out += [L["lam"]] * L["mult"]
    return np.array(out)


def match(values, truth):
    """Greedy one-to-one matching within `MATCH`: (dropped, extra, max_err)."""
    used = np.zeros(len(truth), bool)
    extra, errs = 0, []
    for v in sorted(values):
        d = np.abs(truth - v) / max(abs(v), 1e-300)
        d[used] = np.inf
        j = int(np.argmin(d)) if len(d) else -1
        if j >= 0 and d[j] <= MATCH:
            used[j] = True
            errs.append(float(abs(truth[j] - v) / truth[j]))
        else:
            extra += 1
    return int((~used).sum()), extra, (max(errs) if errs else float("nan"))


def chladni_run(args):
    case, window, prec, jit = args
    window = (window[0] * (1 + jit[0]), window[1] * (1 + jit[1]))
    from chladni.helmholtz import PrecisionWarning
    from chladni.mps import MPSEigensolver
    warnings.simplefilter("ignore", PrecisionWarning)
    lo, hi = window
    dom = MR.polygon(case["vertices"])
    b = MR.basis_for(dom, "D", np.sqrt(hi))
    n = len(b)
    t0 = time.perf_counter()
    try:
        s = MPSEigensolver(dom, basis=b, n_bdry=3 * n, n_int=n, prec=prec)
        res = s.search_interval(lo, hi)
    except Exception as e:  # noqa: BLE001
        return dict(id=case["id"], prec=prec, jit=jit, error=repr(e)[:300],
                    wall=time.perf_counter() - t0)
    d = res.diagnostics
    return dict(id=case["id"], prec=prec, jit=jit, n_cols=n, wall=time.perf_counter() - t0,
                declined=bool(d.declined), unmet=len(d.unmet),
                levels=[dict(lam=float(L.lam), mult=int(L.mult), bound=float(L.bound),
                             count_flag=bool(L.count_uncertain or L.grouping_ambiguous),
                             met=bool(L.met(prec)))
                        for L in res.levels])


def score_lappy(run, ref):
    truth = ref_copies(ref)
    if "error" in run:
        return dict(outcome="exception", detail=run["error"][:120], silent=False)
    n = len(run["lams"])
    if len(truth) < n:
        return dict(outcome="unscored", detail=f"only {len(truth)} trusted reference copies")
    dropped, extra, err = match(run["lams"], truth[:n])
    bad = dropped or extra
    return dict(outcome="wrong" if bad else "ok", dropped=dropped, extra=extra, max_err=err,
                silent=bool(bad), max_sigma=max(run["sigma"]), wall=run["wall"])


def score_chladni(run, ref):
    if "error" in run:
        return dict(outcome="exception", detail=run["error"][:120], silent=False)
    if run["declined"]:
        return dict(outcome="declined", silent=False, wall=run["wall"])
    trusted = []
    for L in ref["levels"]:
        if L["status"] == "provisional":
            break
        trusted.append(L)
    if not trusted:
        return dict(outcome="unscored", detail="no trusted reference levels")
    top = trusted[-1]["lam"] * (1 + 1e-4)
    if len(trusted) < len(ref["levels"]):           # stop scoring short of the untrusted level
        nxt = ref["levels"][len(trusted)]["lam"]
        top = np.sqrt(trusted[-1]["lam"] * nxt)
    truth = np.concatenate([[L["lam"]] * L["mult"] for L in trusted])
    levels = [L for L in run["levels"] if L["lam"] < top]
    values = [L["lam"] for L in levels for _ in range(L["mult"])]
    dropped, extra, err = match(values, truth)
    mult_wrong, uncovered = 0, 0
    for L in levels:
        j = int(np.argmin([abs(T["lam"] - L["lam"]) for T in trusted]))
        T = trusted[j]
        if abs(T["lam"] - L["lam"]) <= MATCH * T["lam"]:
            mult_wrong += L["mult"] != T["mult"]
            uncovered += abs(T["lam"] - L["lam"]) > L["bound"]
    bad = dropped or extra or mult_wrong
    # Two strengths of "flagged". A COUNT flag (`count_uncertain`, `grouping_ambiguous`) says the
    # count is in doubt. An unmet level says only that its value missed `prec` -- on this
    # unplanned basis nearly every solve has one, so counting it would excuse almost any wrong
    # count. `silent` uses count flags only; `silent_lenient` lets an unmet level excuse it too.
    count_flag = any(L["count_flag"] for L in levels)
    any_unmet = any(not L["met"] for L in levels)
    # The COUNT structure: which levels, with what multiplicity. Values at 6 significant figures,
    # well above the stand-in basis's ~1e-8 relative resolution -- a stability score keyed on the
    # 5th decimal called a case unstable for value noise on correct counts (T1.3 near lam 800).
    sig = tuple((f"{L['lam']:.6g}", L["mult"]) for L in levels)
    return dict(outcome="wrong" if bad else "ok", sig=sig, dropped=dropped, extra=extra,
                mult_wrong=mult_wrong, uncovered=uncovered, max_err=err,
                silent=bool(bad and not count_flag),
                silent_lenient=bool(bad and not (count_flag or any_unmet)),
                count_flag=count_flag, unmet=run["unmet"], wall=run["wall"])


def main():
    cases = {c["id"]: c for c in MR.load_cases()}
    refs = {r["id"]: r for r in json.load(open(os.path.join(HERE, "references.json")))["cases"]}
    lruns = json.load(open(os.path.join(HERE, "lappy_runs.json")))["runs"]
    if "--rescore" in sys.argv:     # re-score the saved solves; nothing is re-solved
        cruns = json.load(open(os.path.join(HERE, "scoreboard.json")))["chladni_runs"]
        for r in cruns:
            r["jit"] = tuple(r["jit"])
    else:
        jobs = [(cases[i], refs[i]["window"], p, j) for i in refs for p in CHLADNI_PRECS
                for j in JITTERS]
        with ProcessPoolExecutor(8) as ex:
            cruns = list(ex.map(chladni_run, jobs))
    rows = []
    for i in refs:
        row = dict(id=i, ref_levels=len(refs[i]["levels"]),
                   ref_trusted=sum(L["status"] != "provisional" for L in refs[i]["levels"]))
        for r in lruns:
            if r["id"] == i:
                row[f"lappy_{r['mode']}"] = score_lappy(r, refs[i])
        for p in CHLADNI_PRECS:
            runs = [r for r in cruns if r["id"] == i and r["prec"] == p]
            scores = [score_chladni(r, refs[i]) for r in runs]
            base = scores[[tuple(r["jit"]) for r in runs].index(JITTERS[0])]
            sigs = {str(x.get("sig", x["outcome"])) for x in scores}
            base.update(n_jit=len(scores), n_distinct=len(sigs),
                        n_wrong=sum(x["outcome"] == "wrong" for x in scores),
                        stable=len(sigs) == 1, sig=None,
                        wall_median=float(np.median([r["wall"] for r in runs])))
            row[f"chladni_{p:.0e}"] = base
        rows.append(row)
    json.dump(dict(match=MATCH, rows=rows, chladni_runs=cruns),
              open(os.path.join(HERE, "scoreboard.json"), "w"), indent=1)

    cols = ["lappy_fresh", "lappy_frozen"] + [f"chladni_{p:.0e}" for p in CHLADNI_PRECS]

    def cell(s):
        if s is None:
            return ""
        txt = _cell(s)
        if s.get("n_jit") and not s["stable"]:
            txt += f" · UNSTABLE ({s['n_distinct']} outcomes, {s['n_wrong']}/{s['n_jit']} wrong)"
        return txt

    def _cell(s):
        o = s["outcome"]
        if o == "ok":
            extra = []
            if s.get("uncovered"):
                extra.append(f"{s['uncovered']} unc")
            if s.get("unmet"):
                extra.append(f"{s['unmet']} unmet")
            return "ok" + (f" ({', '.join(extra)})" if extra else "")
        if o == "wrong":
            parts = [f"{k[:4]} {s[k]}" for k in ("dropped", "extra", "mult_wrong") if s.get(k)]
            tag = (("**SILENT** " if s.get("silent_lenient", True) else "silent (unmet only) ")
                   if s["silent"] else "count-flagged ")
            return tag + ", ".join(parts)
        return o
    lines = ["| case | ref levels (trusted) | " + " | ".join(cols) + " |",
             "|---|---|" + "---|" * len(cols)]
    for row in rows:
        lines.append(f"| {row['id']} | {row['ref_levels']} ({row['ref_trusted']}) | "
                     + " | ".join(cell(row.get(c)) for c in cols) + " |")
    tally = {c: {} for c in cols}
    for row in rows:
        for c in cols:
            s = row.get(c)
            if s:
                k = s["outcome"] + ("-silent" if s.get("silent") else "")
                tally[c][k] = tally[c].get(k, 0) + 1
    lines.append("")
    for c in cols:
        extra = ""
        if c.startswith("chladni"):
            st = [row[c] for row in rows if row.get(c) and row[c].get("n_jit")
                  and row[c]["outcome"] in ("ok", "wrong")]
            if st:
                extra = (f"; stable under window jitter on {sum(x['stable'] for x in st)} of "
                         f"{len(st)} scored cases; median wall per solve "
                         f"{np.median([x['wall_median'] for x in st]):.1f} s")
        lines.append(f"* `{c}`: {tally[c]}{extra}")
    open(os.path.join(HERE, "scoreboard_table.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
