"""What does the rank policy cost, and on which axis?

The standing dilemma, stated by the two sides of `_regularize`:

* FREEZE the rank and it is wrong as `lam` moves, because the pencil's numerical rank grows
  with `k`. `choose_rank` takes the MINIMUM over nine probe parameters, so a frozen rank is the
  one the BOTTOM of the window needs and the top of the window is starved.
* THRESHOLD the rank per `lam` and it is right everywhere, but it is an integer function of
  `lam` that jumps -- and `sigma_1` jumps with it, so the tension curve is discontinuous and no
  search can bound a slope across it. That is section 34, where it costs 30 of 55 cells.

This script measures the cost of the first, which section 34 did not: its 2% figure was taken
on a 6%-wide window, which is the easy case and generalizes to nothing.

Two measurements:

    the COST     tension at every true eigenvalue of rect(1, 1/phi) over [5, 300], threshold
                 rank against a rank frozen over that whole window
    the AXIS     how far the threshold rank moves as a function of window WIDTH, at three
                 centres, which is what separates the two regimes

Exploratory; the tables it produces are `notes/landscape_findings.md` section 35.

Run from `tests/`:  PYTHONPATH=. ../.venv/bin/python ../notes/rank_policy.py
"""
import warnings

import numpy as np

warnings.filterwarnings("ignore")

import rect_suite as rs                                          # noqa: E402
from chladni.mps import MPSPencil                                # noqa: E402


def cost_of_freezing():
    """Floor inflation from one frozen rank across a 60x window."""
    case = rs.family_A()[0]                 # rect(1, 1/phi) over [5, 300], 11 distinct levels
    p = case.pencil
    truth = np.unique(rs.complete_truth(case))
    frozen = MPSPencil.from_basis(p.basis, p.bdry, p.interior, reg=p.reg, rtol=p.rtol,
                                  lam_ref=(case.a, case.b))

    print(f"window [{case.a:g}, {case.b:g}]  n_cols={p.n_cols}  frozen rank={frozen.rank}")
    rk = [p.rank_at(float(l)) for l in truth]
    print(f"threshold rank at the true eigenvalues: {min(rk)} .. {max(rk)}\n")
    print(f"{'lam':>12} {'rank':>5} {'short by':>9} {'THRESHOLD':>12} {'FROZEN':>12} "
          f"{'inflation':>12} {'ranks/decade':>13}")
    for l, r in zip(truth, rk):
        a, b = float(p.tension(float(l))), float(frozen.tension(float(l)))
        short, decades = r - frozen.rank, np.log10(b / a)
        print(f"{l:12.4f} {r:5d} {short:9d} {a:12.3e} {b:12.3e} {b / a:12.2e} "
              f"{short / decades:13.2f}")


def rank_against_window_width():
    """Where the frozen and threshold regimes part company: it is the WIDTH."""
    p = rs.family_A()[0].pencil
    print(f"\n{'centre lam':>11} {'width':>7} {'rank lo..hi':>13} {'d(rank)':>8}")
    for centre in (50.0, 150.0, 280.0):
        for frac in (0.01, 0.03, 0.10, 0.30, 1.00):
            lo, hi = centre * (1 - frac / 2), centre * (1 + frac / 2)
            if lo < 5.0 or hi > 300.0:
                continue
            r = [p.rank_at(float(l)) for l in np.linspace(lo, hi, 25)]
            print(f"{centre:11.0f} {frac:6.0%} {min(r):6d}..{max(r):<6d} {max(r) - min(r):8d}")


if __name__ == "__main__":
    cost_of_freezing()
    rank_against_window_width()
