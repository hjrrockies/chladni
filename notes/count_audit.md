# Phase 4 step 1: the mean Weyl deficit catches every drop on real solves (2026-10-11)

Script: `notes/count_audit.py`. Per-solve rows: `notes/count_audit.json`.

**The audit.** At each mid-gap cut between returned levels (values within 1e-6 relative count as
one level, as in `asymp.weyl_deficit`), plus the window's top when that is a mid-gap cut, take
`weyl_count_poly(cut) - #(returned <= cut)`. The **mean** over the cuts is the statistic.
Today's `weyl_deficit`, douse's guard, takes the **worst** cut instead.

**Data, out of sample.** The threshold was first calibrated on the 31 complete reference
spectra; this tests it on different data. That is 819 real solves in
`notes/admission_bound.json`: truncation; bases D and A; prec 1e-10 and 1e-8; five windows per
case. Each is labelled by the reference's finite-element count below the window's top, which is
independent of every MPS basis: 717 COMPLETE (total equals FEM), 99 SHORT, 3 EXTRA.

## Result

| statistic | complete: 99th pct / max | short: min | at the best threshold |
|---|---|---|---|
| worst cut (today) | 1.04 / 1.35 | 1.24 | 20 false alarms / 717, 99 / 99 caught |
| **mean, with the top cut** | **0.022 / 0.50*** | **0.47** | **1* false alarm / 717, 99 / 99 caught** |

\* The one alarm is **not false.** T2.2, basis A, 1e-8, reference window: the solve **dropped**
118.3839 (one member of a pair 3.5e-5 apart) and **returned 284.1169 twice**, as two levels 1.3e-5
apart. The total matches FEM, so the count label called it complete; the audit scored it 0.50.
That is the failure douse's history records, a drop plus a spurious value that balances the
count (`douse_failure_cases.md`, T2.1). **Against the true labels: 100 of 100 drops caught, 0
false alarms in 716.**

- **Margins.** Complete solves have a mean deficit in [-0.096, 0.037]. Short solves start at
  0.47 (T2.6-N3k5-L6, one missing). Any threshold from 0.06 to 0.15 behaves identically, and 0.15
  leaves 4x above the clean maximum and 3x below the smallest drop.
- **Extras are visible too.** The three solves returning one eigenvalue too many score -0.42 to
  -0.62, against the clean minimum of -0.096. A two-sided audit (`excess` below -0.25, say) would
  catch duplicates; most are already flagged per level by `count_shortfall` (N5k3convex).
- **Headroom (reference spectra, simulated whole-level drops, threshold 0.1, with the top cut).**
  A drop among the levels a caller consumes is caught 97% of the time with no headroom, 98% with
  one extra level, and **100% with two or more**, with no false alarms at any headroom.
  `weyl_deficit`'s docstring already says to ask for more than you consume.

## Recommendation for step 2

- Statistic: the mean deficit, with the window's top as a cut when it is mid-gap.
- Threshold: `short` above 0.15. Optionally, `excess` below -0.25.
- Headroom: the caller (later `Eigenproblem.solve(n)`) audits at least two levels past the n it
  returns.
- Scope: prefix windows only (decided). `unaudited` when `lo` is not below the Faber-Krahn bound
  on lam_1.

## Not established

- **Spectrum depth.** These are spectra of 6-26 eigenvalues (lam up to ~450 on these shapes).
  Weyl's remainder grows with lam, and the mean statistic's spread at n ≈ 100 is unmeasured.
  douse's targets are low-index, but a deeper check is cheap: FEM references at higher n.
- Polygons only; Dirichlet only (the corner sum is the Dirichlet one).
- A drop of the very TOP level, with no headroom, is the blind spot by construction.
- The threshold is chosen after seeing the data. The margins are wide enough that the exact value
  does not matter here, but a held-out shape family would settle it.
