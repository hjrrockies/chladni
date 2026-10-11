# Phase 4: the Weyl count audit -- a backstop for drops, with measured limits (2026-10-11)

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

## Then the wider calibration changed the threshold (`notes/count_audit_wide.py`)

The first threshold (0.1, then 0.15) came from the douse benchmark: 41 compact polygons, 6-26
eigenvalues each. While step 2 was being built, the L-shape's own table gave a COMPLETE spectrum
scoring 0.13-0.14 when cut off at 16-17 levels. Weyl's remainder runs +0.5 .. +0.9 over several
consecutive cuts (lam 78-97) before a double swings it back. So the calibration was widened to
everything with known truth: 18 shapes (7 tabulated; rectangles at aspect 1, phi, sqrt(8/3), 2 and
5; three triangles; the disk; sectors at 90 and 270 degrees), every complete prefix, and closed
forms to depth 100.

- **Complete spectra reach +0.23** (the 5:1 rectangle, one of T1.7's misfires) **and -0.48** (an
  8-gon cut at 6 levels).
- **Depth dilutes a drop.** A drop with `h` levels above it in an `n`-level spectrum moves the mean
  by about `h/n`. A local statistic does not rescue it: the maximum tail mean over the last cuts of
  complete deep spectra reaches 2.3, because Weyl's remainder wanders about two eigenvalues there.
  **A Weyl audit fundamentally cannot see a single drop high in a deep spectrum.**
- **Shallow spectra (<= 30 levels, douse's regime), 320 complete:**

  | `short` threshold | false alarms | caught, 3 levels headroom | caught, 4 levels headroom |
  |---|---|---|---|
  | 0.15 | 4 (all the 5:1 rectangle) | 97.2% | 98.6% |
  | 0.20 | 2 | 94.6% | 96.8% |
  | **0.25 (shipped)** | **0** | 91.2% | **94.0%** |

## Shipped (decided 2026-10-11)

- `asymp.count_audit`:
  - `short` when the mean deficit is above 0.25; `excess` when it is below -0.6;
  - the window's top joins the cuts only when the caller says it is mid-gap (`top_is_gap`).
    Otherwise it is a coarse check at ±2.5, since a single cut spans [-1.57, 1.84] on complete
    spectra.
- No false alarm on any complete spectrum measured. **A backstop, not a certificate:** about 94%
  of single drops caught on spectra of up to 30 levels with four levels of headroom, all 100 on
  the douse benchmark, and weaker with depth. The planner's basis-sufficiency check remains the
  primary defence against drops.
- Prefix windows only: `lo` below the Faber-Krahn bound. That bound is weak for elongated shapes:
  the 5:1 rectangle has lam_1 = 257 and a bound of 91. So a caller should start its window well
  below the bound, which `solve(n)` can always do.
- Wired as `diagnostics.count_audit` by `MPSEigensolver.search_interval`, including on an empty
  result. There `b` is a coarse top only.

## Not established

- Non-polygonal shapes with corners (cut square, mushroom) were not in the wide set.
- Dirichlet only (the corner sum is the Dirichlet one). Curved boundaries: only the disk and two sectors.
- A drop of the very TOP level, with no headroom, is the blind spot by construction.
- The threshold was chosen after seeing the data, on 18 shapes. A held-out family with known
  spectra, especially elongated or many-cornered ones, would settle it.
