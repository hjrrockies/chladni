"""Run the bracketing filter: every bracketer, every case, every `prec`.

The deliverable is a CAPABILITY curve, not a score: for each (case, prec, method), the MINIMUM
COST at which the method achieves coverage 1.0 with zero impure brackets, censored at a cap. A
capability claim then reads "M cannot bracket `d = 1e-4` within 4000 evaluations at any tested
setting", which is falsifiable and budget-explicit. Unbounded best-of would be vacuous: section
26 already showed every pair is reachable by raising `per_gap`.

Two columns, and only one of them may be aggregated:

    ORACLE      the best setting for THIS case, chosen with knowledge of the answer.
                An upper bound on capability, and biased upward by the number of knobs --
                `swing` has more than `uniform_grid`, so oracle-vs-oracle is not a fair
                comparison and is reported per case only.
    AUTONOMOUS  one setting fixed across the whole suite, chosen before any per-case result.
                This is what a solver actually ships with, and it is the only column that may
                be summed or averaged.

The gap between them is itself a finding: it measures how much of a method's apparent
capability depends on being told the answer.

Exploratory; not tracked.
"""
from __future__ import annotations

import sys
import time
import warnings

sys.path.insert(0, "tests")
warnings.simplefilter("ignore")

import numpy as np

import bracketers as br
import rect_suite as rs
from bracket_score import score_brackets

#: `prec` values swept. 1e-14 is admissible on every case by the BRACKETING criterion
#: (`prec*lam > 3*eps/c`), measured headroom 2.8x to 7.0x; it is marginal by the MULTIPLICITY
#: criterion, which is a different phase and not under test here.
PRECS = (1e-4, 1e-8, 1e-11, 1e-14)

#: Censoring cap. A method that has not succeeded by here is reported as FAIL(>cap), never as
#: a number, because the cost curve beyond the cap was not measured.
COST_CAP = 1500

#: Knob grids. Small and deliberately similar in size, since best-of-K is biased upward in K.
GRIDS = {
    "uniform_grid":   [dict(n_per_well=n) for n in (8, 32, 128)],
    "lappy":          [dict(ppl=n) for n in (20, 80, 320)],
    "lappy_repaired": [dict(ppl=n, max_depth=d) for n, d in ((20, 4), (20, 6), (40, 7))],
    "lipschitz":      [dict(gamma=g, alpha=al) for g, al in
                       ((2.0, 0.25), (2.0, 0.05), (4.0, 0.05))],
    "swing_cells":    [dict(per_gap=p) for p in (6.0, 32.0, 160.0)],
    "swing_direct":   [dict(per_gap=p) for p in (6.0, 32.0, 160.0)],
}

#: The AUTONOMOUS setting: index into each grid, fixed before any per-case result was seen.
#: The middle of each grid, on the reasoning that a shipped default is neither the cheapest
#: nor the most aggressive setting.
AUTONOMOUS = {k: 1 for k in GRIDS}


class BudgetExceeded(Exception):
    """Raised inside the evaluation wrapper when a run passes `COST_CAP`.

    Enforcing the cap at the point of evaluation rather than checking afterwards is what makes
    the sweep finite: an unbounded knob setting (`lappy_repaired` at depth 8, or the swing at
    `per_gap = 384`) otherwise runs for minutes before producing a result that would be
    censored anyway. A run that raises is recorded as FAIL(>cap), which is exactly the
    censoring the capability curve already specifies.
    """


def memoized(case, cap=None):
    """Serve repeated `lam` from a dict so the knob sweep is cheap in WALL CLOCK, while the
    Counter still records every logical request. Cost is what the algorithm asked for; the
    memo only stops us paying for the same GSVD in five different knob settings."""
    tmemo, mmemo = {}, {}
    raw_t, raw_m = case.tensions, case.modes

    def guard():
        if cap is not None and case.counter.calls > cap:
            raise BudgetExceeded

    def tensions(x):
        k = float(x)
        if k not in tmemo:
            tmemo[k] = raw_t(k)
        else:
            case.counter.calls += 1
        guard()
        return tmemo[k]

    def modes(x):
        k = float(x)
        if k not in mmemo:
            mmemo[k] = raw_m(k)
        else:
            case.counter.calls += 1
        guard()
        return mmemo[k]

    return tensions, modes


def run_one(case, name, kwargs, prec, tensions, modes):
    case.reset()
    f = br.BRACKETERS[name](**kwargs) if name in br.BRACKETERS else br.CONTROLS[name]()
    t0 = time.time()
    try:
        bs = f(tensions, case.a, case.b, prec=prec, modes=modes)
    except BudgetExceeded:
        return None, f">cap({COST_CAP})"
    except Exception as ex:
        return None, f"{type(ex).__name__}: {ex}"
    dt = time.time() - t0
    s = score_brackets(bs, case.truth, prec, label=f"{name}@{prec:.0e}",
                       evaluations=case.evaluations(), seconds=dt)
    return s, None


def sweep_case(case, names=None, precs=PRECS, verbose=True):
    """All methods x all knob settings x all `prec`, on one case."""
    tensions, modes = memoized(case, cap=COST_CAP)
    names = names or list(GRIDS)
    rows = []
    for prec in precs:
        for name in names:
            grid = GRIDS[name]
            for i, kw in enumerate(grid):
                s, err = run_one(case, name, kw, prec, tensions, modes)
                rows.append(dict(case=case.label, prec=prec, method=name, setting=i,
                                 kwargs=kw, score=s, err=err,
                                 autonomous=(i == AUTONOMOUS[name])))
                if verbose and err:
                    print(f"    {name}[{i}] @ {prec:.0e}  ERROR {err[:60]}", flush=True)
    return rows


def capability(rows, method, prec):
    """(oracle_cost, autonomous_ok, autonomous_cost) for one method at one `prec`.

    `oracle_cost` is the cheapest setting that reached coverage 1.0 with zero impure
    brackets, or None if no setting did within the cap.
    """
    sub = [r for r in rows if r["method"] == method and r["prec"] == prec and r["score"]]
    ok = [r for r in sub if r["score"].ok and r["score"].evaluations <= COST_CAP]
    oracle = min((r["score"].evaluations for r in ok), default=None)
    auto = next((r for r in sub if r["autonomous"]), None)
    a_ok = bool(auto and auto["score"].ok and auto["score"].evaluations <= COST_CAP)
    a_cost = auto["score"].evaluations if auto else None
    return oracle, a_ok, a_cost


def table(rows, precs=PRECS, names=None):
    names = names or list(GRIDS)
    out = []
    out.append(f"{'method':>16} | " + " | ".join(f"{f'prec={p:.0e}':>21}" for p in precs))
    out.append(f"{'':>16} | " + " | ".join(f"{'oracle':>10}{'auto':>11}" for _ in precs))
    for name in names:
        cells = []
        for p in precs:
            oc, aok, ac = capability(rows, name, p)
            o = f"{oc:>10d}" if oc is not None else f"{'FAIL':>10}"
            a = f"{ac:>10d}*" if aok and ac is not None else f"{'FAIL':>10} "
            cells.append(o + a)
        out.append(f"{name:>16} | " + " | ".join(cells))
    out.append("  cost = evaluations to reach coverage 1.0 with 0 impure; "
               f"FAIL = no setting did so within {COST_CAP}. * = autonomous setting.")
    return "\n".join(out)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "B"
    fams = rs.all_families()
    cases = fams[which] if which in fams else [c for v in fams.values() for c in v]
    print(f"=== family {which}: {len(cases)} cases, {len(PRECS)} precs, "
          f"{len(GRIDS)} methods x 4 settings\n")
    for case in cases:
        ok, frows = rs.assert_floor(case)
        if not ok:
            print(f"{case.label}: FLOOR PRECONDITION FAILED -- skipped")
            continue
        t0 = time.time()
        rows = sweep_case(case)
        print(f"--- {case.label}   {len(case.truth)} eigenvalues   "
              f"[{case.a:.4g}, {case.b:.4g}]   ({time.time()-t0:.0f}s)")
        print(table(rows))
        print(flush=True)
