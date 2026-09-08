"""The two adversarial controls, on every case, in the same format as the main sweep.

They are not candidates. They exist to validate the metrics, and the report must state them
that way:

    single_bracket   returns [(a, b)]. Scores coverage 1.0 and zero spurious while being
                     useless, so PURITY must be what rejects it. If it passes a case with more
                     than one level, the metric is broken.
    fine_grid        returns n equal slivers, committed where the tension is below tau. Has
                     near-perfect purity and absurd cost, so SPURIOUS and COST must be what
                     reject it.

They have no knobs to sweep, so oracle and autonomous coincide; the output is written in the
main sweep's row format so `make_bracket_tables.py` can merge the two logs.

Exploratory; not tracked.
"""
from __future__ import annotations

import sys
import time
import warnings

sys.path.insert(0, "tests")
sys.path.insert(0, "notes")
warnings.simplefilter("ignore")

import bracketers as br
import rect_suite as rs
from bracket_score import score_brackets
from run_bracket_filter import COST_CAP, PRECS, BudgetExceeded, memoized

NAMES = ["single_bracket", "fine_grid"]


def main():
    fams = rs.all_families()
    cases = [c for v in fams.values() for c in v] + rs.family_stress()
    print(f"=== controls: {len(cases)} cases, {len(PRECS)} precs, {len(NAMES)} controls\n")
    for case in cases:
        tensions, modes = memoized(case, cap=COST_CAP)
        t0 = time.time()
        results = {}
        for name in NAMES:
            cells = []
            for prec in PRECS:
                case.reset()
                f = br.CONTROLS[name]()
                try:
                    bs = f(tensions, case.a, case.b, prec=prec, modes=modes)
                    s = score_brackets(bs, case.truth, prec,
                                       evaluations=case.evaluations())
                    ok = s.ok and s.evaluations <= COST_CAP
                    cells.append((s.evaluations if ok else None, ok, s.evaluations,
                                  s.n_brackets, s.spurious, s.impure, s.coverage))
                except BudgetExceeded:
                    cells.append((None, False, COST_CAP, None, None, None, None))
            results[name] = cells
        print(f"--- {case.label}   {len(case.truth)} eigenvalues   "
              f"[{case.a:.4g}, {case.b:.4g}]   ({time.time()-t0:.0f}s)")
        print(f"{'method':>16} | " + " | ".join(f"{f'prec={p:.0e}':>21}" for p in PRECS))
        print(f"{'':>16} | " + " | ".join(f"{'oracle':>10}{'auto':>11}" for _ in PRECS))
        for name in NAMES:
            cells = []
            for oc, ok, ev, nbr, spur, imp, cov in results[name]:
                o = f"{oc:>10d}" if oc is not None else f"{'FAIL':>10}"
                a = f"{ev:>10d}*" if ok else f"{'FAIL':>10} "
                cells.append(o + a)
            print(f"{name:>16} | " + " | ".join(cells))
        # the diagnostic the controls exist for, at one prec
        for name in NAMES:
            oc, ok, ev, nbr, spur, imp, cov = results[name][1]
            print(f"    {name:>14} @1e-08: brackets={nbr} coverage={cov} "
                  f"impure={imp} spurious={spur} evals={ev}")
        print(flush=True)


if __name__ == "__main__":
    main()
