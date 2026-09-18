"""Can anything cheap DETECT that a basis silently dropped the levels it resolves worst?

`tests/test_solve_interval.py::test_a_starved_basis_does_not_DROP_levels_silently` pins the
defect: on the disk over `[8, 60]` at `prec = 1e-12`, plane waves at `pad = 0.45` return 5 of 6
LEVELS with no decline, nothing in `unmet`, and no warning -- the levels that ARE found meet the
request. `TODO.md` named the Weyl deficit as the obvious candidate alarm. This measures it, and
`findability` beside it.

Two constructions, on the same starvation ladder:

    WEYL      `asymp.weyl_deficit` on the returned spectrum. It cuts at the MIDPOINT between
              adjacent distinct returned levels, and at each cut compares the three-term count
              `N(lam)` against how many returned values lie below it, worst over the cuts.
    FIND      `MPSEigensolver.findability`, which asks whether the visible catchment of a dip
              is at least one scan step wide: `sigma_base >= (kappa/2q)*(A/P)*dk`.

**PASS THE SPECTRUM UNFOLDED.** `weyl_count_poly` counts eigenvalues WITH multiplicity, and the
disk's are mostly double. Auditing one entry per distinct level against it reports a deficit of
+4.70 on the TRUE spectrum, which is pure bookkeeping and not a signal -- the error that
prompted this script to be written carefully. `SolveResult.unfolded` is the right input.

Exploratory; the tables are `notes/landscape_findings.md` section 38.

Run from `tests/`:  PYTHONPATH=. ../.venv/bin/python ../notes/starved_basis_alarm.py
"""
import warnings

import numpy as np

warnings.filterwarnings("ignore")

import hard_roster as hr                                         # noqa: E402
from chladni import asymp                                        # noqa: E402
from chladni.eigensolver import lam_to_k                         # noqa: E402
from chladni.geometry import disk                                # noqa: E402
from chladni.mps.search import solve                             # noqa: E402
from chladni.reference import disk_spectrum                      # noqa: E402

A, B, PREC = 8.0, 60.0, 1e-12
PADS = (0.30, 0.35, 0.40, 0.45, 1.30)


def main():
    dom = disk(1.0)
    truth = np.sort(disk_spectrum(A, B, 1.0))
    print(f"disk [{A}, {B}] at prec {PREC:.0e}: {np.unique(truth).size} distinct levels, "
          f"{truth.size} eigenvalues with multiplicity\n")
    print(f"  weyl_deficit(TRUE spectrum, unfolded)      {asymp.weyl_deficit(truth, dom):+.2f}")
    print(f"  weyl_deficit(TRUE spectrum, levels only)   "
          f"{asymp.weyl_deficit(np.unique(truth), dom):+.2f}   <- the multiplicity trap\n")

    dk = (lam_to_k(B) - lam_to_k(A)) / 200.0
    print(f"{'pad':>5} {'cols':>5} {'levels':>7} {'unfolded':>9} {'MISSING':>8} "
          f"{'WEYL':>7} {'sigma_base':>11} {'FIND ok':>8} {'shortfall':>10}")
    for pad in PADS:
        s = hr.solver_for(dom, B, label=f"starved{pad}", pad=pad, prec=PREC)
        r = solve(s.objective(A, B, prec=PREC), prec=PREC)
        unf = np.sort(r.unfolded)
        wey = asymp.weyl_deficit(unf, dom) if unf.size >= 2 else float("nan")
        cal = s.findability(A, B, dk=dk)
        print(f"{pad:5.2f} {s.n_cols:5d} {len(r.levels):7d} {unf.size:9d} "
              f"{truth.size - unf.size:8d} {wey:7.2f} {cal.sigma_base:11.3e} "
              f"{str(cal.achieved):>8} {cal.shortfall:10.2f}")


if __name__ == "__main__":
    main()
