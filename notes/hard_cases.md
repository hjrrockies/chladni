# What limits the douse shapes no recipe brought to 1e-10: basis SIZE (2026-10-12)

Script: `notes/hard_cases.py`; data in `notes/hard_cases.json`. The follow-up runs (scale 4.0;
sharp corners only) are in the table below. Four hard cases (0% of levels met at 1e-10 under
every recipe in `notes/planner_recipes.py`) and two controls, on the reference windows.

| case | 2.0x (baseline) | colloc x2 | 3.0x | 4.0x | 2.0x + sharp corners x4 | 2.0x + lightning |
|---|---|---|---|---|---|---|
| T2.3-pre-ref0 | 0% met, margin 0.15 | 0%, 0.10 | 27%, 0.47 | **73%, 3.7** | 0%, 0.15 | 0%, 0.19 |
| T2.5 | 0%, 0.26 | 0%, 0.16 | 33%, 1.8 | **100%, 25** | 0%, 0.22 | 0%, 0.29 |
| T2.6-N5k4-L7-ref0 | 0%, 0.09 | 0%, 0.12 | 27%, 0.62 | **91%, 9.3** | 0%, 0.09 | DECLINED |
| T2.2 | 0%, 0.28 | 0%, 0.20 | 0%, 0.29 | 0%, 0.17 | 0%, 0.28 | 0%, 0.33 |
| control T2.9-N5k3convex | 70%, 4.5 | 70%, 4.7 | 100%, 30 | | | DECLINED |
| control T1.4-oct | 100%, 89 | 100%, 98 | 100%, 110 | | | DECLINED |

(Each cell: fraction of levels met; median margin. "4.0x" is the corner terms at 4x lappy's
per-corner sizing, 330-480 columns, 34-64 s per solve.)

- **It is the size of the corner expansions.** Margins rise steadily with scale, 0.1 -> 0.5 ->
  3-25 from 2x to 3x to 4x, and the tension floor falls 10-30x per step. lappy's per-corner rule
  `(k r + 2 (k r)^(1/3) + 3)/alpha` is sized for a much looser target. These shapes need about 4x
  at 1e-10.
- **Collocation is not binding.** Doubling the points changes nothing.
- **Fourier-Bessel evaluation accuracy is not binding** in the sense that matters. About 17-20% of
  columns report a relative-error bound above 1e-10 (worst ~1e-7) at the window's top, on the
  controls as well, and the controls meet 1e-10 with margins of 20-110.
- **Sharp corners are not the limit.** Quadrupling only the corners under 45 degrees did nothing.
- **Lightning clusters, as configured (12 per corner, untapered, default sigma), are harmful.**
  They help nothing and make three solves DECLINE, controls included: near-duplicate columns,
  which `Fundamental.lightning`'s docstring warns about.
- **T2.2 resists everything,** up to 476 columns. It has a 15-degree corner, a re-entrant one,
  and a pair of eigenvalues 3.5e-5 apart. Its floor sits near 2e-7 whatever the basis. Not
  diagnosed: for the planner it is a case that is flagged, not solved.

**For the planner:**
- The ladder must reach about 4x, and the sizing should probably carry `prec`. lappy's rule
  has no precision term, while `PlaneWave.n_useful` grows with `ln(1/prec)^(2/3)`.
- Cost rises with it: a 4x solve is 2-3x the wall time of 2x.

## Part 2: the eleven cases no rung passes (`notes/hard_cases2.py`, `notes/hard_cases2.json`)

`notes/planner_recipes.md`: 11 of 35 cases pass no recipe at 1e-10, even at 4x. Their worst
margins against size (2x, 2.5x, 3x, 4x) split them into two groups.

**Group A: just needs more (6 cases).** T2.3 (3 cases), T2.6-N5k4-L7 (2), T2.1-rerun. Margins
rise steadily; extrapolated, they pass at 4.1-4.7x. They share a geometry: near-right angles plus
one slightly re-entrant corner (1.07-1.09 pi). At **5x**, three of them reach 90-91% of levels met
with worst margins of 1.2-1.8, one level short of passing, at 60-65 s per solve. About 5.5-6x
would pass them.

**Group B: stagnant (4 cases), all with very sharp corners.** T1.3 (11 and 16 degrees), T2.2
(15 degrees, plus a re-entrant corner), T2.6-N3k4-L8 (18 and 20 degrees), and T2.6-N3k5-L6 (40 and
55 degrees, slow). Base 3x, then:

| case | base 3x: met / min margin / floor | any one corner at 9x | columns with error bound > 1e-11 removed |
|---|---|---|---|
| T2.2 | 0% / 0.038 / 1.7e-7 | 0%, floor 0.8-2.2e-7 | floor 1e-2 |
| T1.3 | 11% / 0.032 / 3.1e-7 | 0-11%, floor 3-4e-7 | no levels found |
| T2.6-N3k5-L6 | 0% / 0.0015 / 1.7e-7 | 0%, floor 1.4-1.6e-7 | no levels found |
| T2.6-N3k4-L8 | 0% / 0.27 / 4.2e-10 | 0-10%, margin stuck at 0.25-0.27 | no levels found |

- **No single corner is the bottleneck.** Enriching any one corner threefold, sharp ones
  included, does not move the floor.
- **The inaccurate columns are needed.** Removing every Fourier-Bessel column whose own error
  bound exceeds 1e-11 destroys the solve, so their content matters.
- **The floor stalls at 1.5-3e-7** on T2.2, T1.3 and T2.6-N3k5. That is the same order as the
  worst column-evaluation error bounds on these shapes (~1e-7, `Sampling.accuracy`). Hypothesis,
  not established: the Fourier-Bessel EVALUATION of high-order terms on sharp-cornered shapes caps
  the floor near 1e-7, whatever the basis size. Testing it needs a higher-precision evaluation of
  one case's tension (e.g. mpmath Bessel functions at a located level). That is a library question,
  not a planner one.
- **T2.6-N3k4-L8 stalls differently:** its floor is small (4e-10) but its margin is stuck near
  0.27, so the wall slope is what is small. Not diagnosed.

## Part 3: the evaluation hypothesis is refuted (`notes/bessel_precision.py`)

On T2.2 with the 3x basis (359 columns, the solver's exact point sets), every column was
recomputed with J_mu(k r) evaluated by mpmath to 30 digits, normalized in extended precision, and
compared with `chladni`'s float64 columns. The comparison was made at three reference
eigenvalues.

| lam | sigma_1..3, float64 | sigma_1..3, mpmath | worst column disagreement |
|---|---|---|---|
| 37.183126 | 1.007e-7, 0.949, 1.45 | 1.007e-7, 0.949, 1.45 | 9.5e-15 |
| 68.260292 | 1.847e-7, 0.988, 1.02 | 1.847e-7, 0.988, 1.02 | 7.7e-14 |
| 152.295845 | 1.476e-7, 0.665, 0.691 | 1.469e-7, 0.665, 0.705 | 2.1e-12 |

- **Fourier-Bessel evaluation is accurate to ~1e-14 here.** The per-column bounds from
  `Sampling.accuracy` (~1e-7) are very pessimistic, and they are not what caps group B.
- The absolute sigma_1 values in this table are NOT the floor. These lam values are the
  reference rounded to 6 decimals, so sigma_1 includes the slope times the rounding. The
  conclusion rests on the two columns agreeing, and the solver's own located floors (part 2) are
  independent of it.
- **So group B's cap lies in what the basis can represent on these shapes.** Candidates, none
  tested:
  - the corner expansions' convergence regions on shapes with very sharp corners and long thin
    wedges;
  - too few interior points inside the thin wedges, so the interior norm does not control the
    solution there;
  - corner-clustered sources (lightning). These were harmful as configured (untapered, 12 per
    corner), but they are the textbook tool for exactly this geometry.
