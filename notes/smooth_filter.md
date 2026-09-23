# Smooth spectral filtering instead of a rank

2026-09-18. `landscape_findings.md` section 35 ends by proposing a smooth spectral filter as
"a third option that dissolves the dilemma rather than picking a side", and lists what it
would cost with the note "none of it measured". This is the measurement.
`notes/smooth_filter.py` reproduces every table and `notes/smooth_filter_figs.py` every figure
(`fig_filter_tension`, `_jump`, `_shape`, `_trade`).

**"Nothing in `src/` is touched" stood here and is no longer true.** Sections 1-8 were written
that way, against scratch implementations. Section 9 landed `reg="tikhonov_qr"` as an option on
2026-09-21, and on 2026-09-23 it became the INTENDED DEFAULT (`TODO.md` section 4). Sections 1-8
are the measurement that decision rests on and are left as they were written; 9 onward are what
shipped and what is still open.

## 1. The proposal as stated does not work, and it is worth knowing why

Section 35 proposes to "weight direction `j` by `f(s_j/s_0)`". That is a right-diagonal factor
on the stacked pencil, and **the GSVD is invariant under any invertible right factor** -- the
generalized singular values of `(B M, I M)` equal those of `(B, I)` for every nonsingular `M`.
Measured, on a random pencil:

           right factor M          max relative change in sigma
              diagonal, mild                       2.6e-15
     diagonal, spanning 1e-8                       4.6e-15
           random invertible                       3.6e-15

So a weighting changes nothing until a weight reaches **exactly** zero, at which point it is
hard truncation again. Truncation works precisely because the projector is singular. Applying
a smooth weight and watching the tension move is possible -- an aggressive `w = (s/s0)^4`
spanning 1e-65 did move `sigma_1` from 3.4e-16 to 2.0e-4 -- but that is the weighting
destroying the conditioning of the GSVD, not filtering anything.

**The filter has to be a PENALTY, not a weight.** Add a term to the numerator:

    sigma_mu(lam) = min_c  sqrt( ||A_B c||^2 + mu^2 ||c||^2 ) / ||A_I c||

which is the GSVD of `([A_B; mu I], A_I)`. In whitened coordinates `y = diag(s) V^T c` the
pencil is `([U_B; diag(g)], U_I)` with `g_j = mu / s_j`: a penalty of `0` leaves a direction
untouched, a penalty of `inf` forbids it (= truncation), and `mu/s_j` is a one-decade-wide
smooth transition centred at `s_j = mu`. This is Tikhonov, and unlike a weight it really is
continuous in the matrix. It also needs **no SVD at all** -- the direct form is one `vstack`.

## 2. It is a single continuous dial between the two incumbents

`rect(1, 1/phi)` over `[5, 300]`, 126 columns, frozen rank 29, threshold rank 39..61 -- the
same pencil section 35's table is built on. Three axes:

* **floor** -- worst `sigma_1` over the 11 true eigenvalues in the window
* **step** -- relative jump across a threshold-rank change, bisected to 1e-10 in `lam`
* **noise** -- relative residual of a local linear fit, 11 samples over a 1e-8 relative
  interval on the plateau

        policy  worst floor  worst step  worst noise
      mu=1e-15    9.317e-14    1.23e-04     1.54e-04
      mu=1e-14    9.682e-14    1.25e-04     1.14e-04
      mu=1e-13    2.805e-13    1.50e-04     5.18e-05
      mu=1e-12    1.444e-12    5.09e-06     9.15e-07
      mu=1e-11    1.400e-11    4.86e-07     1.54e-07
      mu=1e-10    1.337e-10    7.53e-08     6.36e-08
      mu=1e-09    1.334e-09    4.39e-09     6.21e-09
      mu=1e-08    1.289e-08    2.98e-10     2.81e-09
      mu=1e-07    1.217e-07    1.60e-10     4.80e-11
     THRESHOLD    1.569e-15    3.74e-03     1.00e-03
        FROZEN    2.259e-04    2.13e-08     3.42e-11

Three things this settles.

* **The floor is `mu`, flat across the whole 60x window.** 1.3e-9 at `lam = 35` and 1.3e-9 at
  `lam = 272`. That is the contrast with the frozen rank, whose floor inflates by 2.0e11x over
  the same window because one integer cannot serve both ends. Section 35 predicted "the floor
  rises to about the filter width"; it does, to within 1.7x, and it does not drift.
* **The exchange rate is one decade of floor for one decade of noise**, from `mu = 1e-12`
  upward. Below `mu ~ 1e-12` it saturates: the floor sticks near 1e-13 and the noise near
  1e-4, because a weakly penalised full-column pencil is simply ill-conditioned and its GSVD
  has its own noise floor. **Smooth filtering cannot reach the threshold's 1.6e-15.** Two
  decades short, structurally -- truncation *removes* the noise directions and a penalty only
  leans on them.

  **Corrected 2026-09-21 (section 10): that saturation is the literal form's.** This table was
  measured with `([A_B; mu I], A_I)` handed to the GSVD directly, where LAPACK re-truncates
  internally at about 1e-13. The QR form's floor tracks `mu` down to 2.1e-15. What survives is
  the noise, which grows as `mu` shrinks.
* **On THIS window the frozen rank is very nearly dominated.** `mu = 1e-9` gives a floor five
  decades better (1.3e-9 against 2.3e-4) for 200x more evaluation noise, at a noise level
  (6e-9) no search cares about.

  An earlier draft of this section added "there is no window width at which freezing is the
  better of those two". That is wrong and section 7.2 disproves it. This table is a 60x
  window, which is where one frozen integer cannot serve both ends. Section 35's own width
  table says freezing costs one or two ranks below about 10% width -- and on the 6%-wide
  split-degeneracy windows where the noise actually decides solver outcomes, freezing is
  nearly free and beats every filter measured, 51 right against 33.

## 3. What the noise buys, and a correction to how section 34 states the problem

Drawing the curve corrected something. **The threshold rank does not step; it FLICKERS**, and
`sigma_1` with it. Around a plateau point, counting rank changes over 201 samples:

    half-width     distinct ranks   changes   ptp(sigma_1)/median
       2e-01         [39, 40, 41]        31            1.5e-01
       2e-02         [39, 40, 41]        75            1.8e-02
       2e-04             [39, 40]        88            3.8e-03
       2e-06             [39, 40]        74            4.0e-03
       2e-08         [39, 40, 41]        63            3.6e-03
       2e-10             [39, 40]        85            4.7e-03

The rank is a near-coin-flip between 39 and 40 at **every** scale, and the induced spread in
`sigma_1` stops shrinking at about **4e-3 and stays there forever**. So the right statement is
not "the tension jumps at isolated parameters where the rank steps" -- it is that threshold
truncation carries an **irreducible 0.4% noise floor everywhere on the plateau**. The 22
jumps-per-400-samples an earlier pass reported were an aliasing artefact of the grid.

That matters for how the fix is described. A discontinuity at known places could be routed
around -- restart the search at sub-window boundaries, which is what section 35's planner
paragraph proposes. Scale-free noise cannot be routed around by anything.

The same measurement is the test for differentiability, and the filtered objective passes it:

    half-width      threshold     mu=1e-11      mu=1e-9
       2e-02          1.8e-02      1.6e-02      1.6e-02
       2e-04          3.8e-03      1.6e-04      1.6e-04
       2e-06          4.0e-03      1.7e-06      1.6e-06
       2e-08          3.6e-03      2.7e-07      2.1e-08

Spread proportional to width is what a smooth function does; a flat column is what noise does.
The filtered objective rides the proportional line down until it reaches its own noise floor,
which is `~mu` -- so `mu` buys differentiability over a range of scales, exactly as it buys
floor, and the two are the same purchase.

## 4. Locating power is the floor, as Betcke says

Well half-width at twice the floor, which is the intrinsic resolution of the minimum:

        lam |  THRESH floor/width |  FROZEN floor/width |   mu=1e-9 floor/width
     35.709 |  7.74e-16  7.77e-16 |  4.77e-11  5.01e-11 |  9.50e-10   9.35e-10
    142.834 |  9.15e-16  5.55e-16 |  1.19e-06  6.61e-07 |  1.25e-09   7.00e-10
    261.270 |  1.15e-15  3.33e-16 |  2.26e-04  8.64e-05 |  1.30e-09   4.73e-10

Half-width tracks the floor within a factor of two everywhere, for every policy. So **`mu`
sets the deliverable precision directly**, and the natural rule is `mu ~ prec`: ask for 1e-8
and pay 1e-9 noise; ask for 1e-12 and pay 1e-6 noise, still three decades quieter than
threshold truncation. That is the `prec`-aware, don't-over-work design principle getting
something back for once -- the smoothness is bought with the precision you did not ask for.

Minimiser accuracy agrees: over the 11 levels, worst relative error 9.3e-9 at every `mu` from
1e-13 to 1e-7 (optimiser-limited, and 8.0e-9 for threshold truncation), against 4.7e-6 for the frozen rank.

## 5. Transfer, and the one case where it is a pure cost

Same three measurements on the roster.

* **`disk`** (84 plane waves, `[5, 60]`, threshold rank 37..59) and **`eq_tri`** (90 columns,
  `[50, 300]`, rank 41..55) behave exactly like the rectangle: floor `= mu` flat across the
  window, plateau noise 8.3e-9 and 9.2e-9 at `mu = 1e-9` against 4.0e-5 and 3.0e-5 for threshold, while the frozen
  rank's floor reaches 8.5e-9 and 2.6e-10 respectively. Degenerate pairs are untouched -- the
  two members of each disk double agree to every printed digit at every `mu`.
* **`L_shape`** (80 pure corner FB, `[5, 26]`) is the exception and it is instructive: the
  **threshold rank is 80 of 80 columns**, so truncation is entirely inert there, threshold and
  frozen are the same function, and the floor is the basis's own 2e-15. Adding a penalty can
  only raise it -- `mu = 1e-9` turns a 2.2e-15 floor into 1.3e-9 and buys nothing, because
  plateau noise is already 4.5e-14 for every policy.

So the filter is a dial worth having exactly where truncation is *active*. On a
corner-adapted basis that is fully resolved it is a tax. Any policy should be able to tell
those apart, and the threshold rank against `n_cols` is the tell.

## 6. Cost, and the seam

Per tension evaluation on the 126-column pencil at `lam = 220`, assembly cached:

    threshold truncation (QR + SVD + reduced GSVD)   4.37 ms
    frozen truncation    (QR + SVD + reduced GSVD)   2.40 ms
    Tikhonov             (no factorisation, full GSVD)  2.91 ms

Section 35 expected the penalty to be "several times the cost per evaluation" because it
keeps all `n` columns. It is not: it **skips the QR and SVD entirely**, and the larger GSVD
roughly cancels them. Cheaper than the threshold path, 20% dearer than the frozen one.

The real bill is the one section 35 named and it is unchanged: **this is a seam change, not a
parameter.** `rank_at`, `coef(lam, mult)`, `choose_rank`, `RankDiagnostics` and `n_useful` all
assume an integer rank, and `Objective.sigma` reads the tension vector's LENGTH as the rank.
Under a penalty there is no rank -- the vector is always `n_cols` long. A `reg="tikhonov"`
mode would have to answer what `rank_at` means, and the honest answer is `sum_j f_j` with
`f_j = s_j^2/(s_j^2 + mu^2)`, the effective rank, which is not an integer.

## 6b. The hybrid, measured

2026-09-19. Section 7 below listed the hybrid as "the interesting object ... wants its own
measurement". Measured. `tension_hybrid` in `notes/smooth_filter.py`: hard-cut below
`eps*s_1`, penalise with `g_j = (c*s_1/s_j)**p` across the band, and DROP any column whose
`g_j` exceeds 1e8 rather than carry it -- `f_j < 1e-16` there, so dropping is below the
arithmetic's notice, and it bounds the dynamic range of `diag(g)`, which is what an uncapped
`p >= 4` destroys. Same pencil, same three axes, plus the spread test of section 3.

           c      p   worst floor   worst noise   spread@2e-6
    1.49e-15    1.0     8.776e-15     4.680e-03     7.377e-04
    1.49e-15    2.0     2.176e-15     5.350e-04     7.295e-04
    1.49e-15    4.0     1.508e-15     6.169e-04     7.131e-04
       1e-14    2.0     9.912e-15     2.609e-05     4.740e-06
       1e-13    2.0     1.304e-13     1.174e-06     4.232e-06
       1e-11    2.0     1.634e-11     3.895e-08     1.606e-06
    THRESHOLD           1.569e-15     1.003e-03     3.415e-03
    FROZEN              2.259e-04     3.419e-11     1.562e-06
    tikhonov 1e-13      2.805e-13     5.182e-05     5.426e-05
    tikhonov 1e-11      1.400e-11     1.538e-07     1.674e-06

**The floor model from section 5 holds.** `floor ~ max(eps_basis, c, ~6*c^p)`: at `c = rtol`,
`p = 1` gives `c^p = 1.5e-15` and 8.8e-15 measured; `p >= 2` kills the `c^p` term and the
floor drops to `c` itself, 1.5e-15 -- **the threshold's own floor**, flat across the 60x
window (7.4e-16 to 2.2e-15 at the eleven levels, against the threshold's 7.7e-16 to 1.6e-15).
So the hybrid does what plain Tikhonov cannot: it reaches the bottom.

**It strictly dominates plain Tikhonov.** At every matched floor it is quieter, and it
continues two decades past where Tikhonov saturates:

    at floor ~1e-13   hybrid 1.2e-06 noise   tikhonov 5.2e-05    44x quieter
    at floor ~1.5e-11 hybrid 3.9e-08 noise   tikhonov 1.5e-07    3.9x quieter
    at floor ~1e-14   hybrid 2.6e-05 noise   tikhonov cannot reach this floor at all

**But at the threshold's own floor it buys much less than hoped: 1.6x on noise, 4.7x on the
spread test.** That is the honest headline and it is not what section 35 implies. The reason
is visible in `fig_filter_shape`: to hold a 1.5e-15 floor the ramp must sit ON the noise
floor, where the `s_j` are themselves jittery in `lam` -- and a smooth function of a jittery
argument is still jittery. Smoothing removes the discontinuity in `f`, not the noise in `s`.
The gain only becomes decades once `c` is raised clear of the jitter, and raising `c` is
exactly what costs floor.

**One clean surprise: on the L-shape the hybrid is FREE.** Plain Tikhonov taxed that domain
(2e-15 floor becomes 1.3e-11 at `mu = 1e-9`, buying nothing, section 5). The hybrid at
`c = rtol` does not, because with the ramp at `eps` it only touches directions the fully
resolved corner basis does not have:

      domain        policy        worst floor    plateau noise
      L_shape    THRESHOLD          5.780e-15        4.517e-14
      L_shape    hybrid c=rtol      5.839e-15        2.652e-14
      disk       THRESHOLD          1.649e-15        4.040e-05
      disk       hybrid c=1e-13     1.600e-15        5.718e-07
      eq_tri     THRESHOLD          1.483e-15        3.024e-05
      eq_tri     hybrid c=1e-13     8.162e-14        8.565e-07

That removes the objection section 5 raised, and it is the single best argument for the
hybrid over the penalty: **it degrades to the current behaviour where the current behaviour
is already right.**

**What `c` costs is domain-dependent, and the axis is basis sufficiency.** On the disk,
raising `c` from `rtol` to 1e-11 costs NOTHING -- the floor stays at 7e-16..1.6e-15 at every
one of its seven levels. On the rectangle the same move costs four decades. The spectra look
similar (4 directions in `[1e-15, 1e-13]` on the rect at `lam = 261`, 6 on the disk at
`lam = 57.6`), so it is not how many marginal directions exist -- it is whether the
eigenfunction NEEDS them. The rectangle at `lam = 272` with 126 plane waves is basis-starved
where the disk at `lam = 57` with 84 is comfortable. That is the same quantity `rtol`
calibration already probes (`mps.calibrate`, `MPSPencil.with_rtol`), which is reassuring: `c`
is not a new kind of parameter, it is `rtol` with a ramp on it.

**Cost is the one place the hybrid is worst.** Per evaluation at `lam = 220`, assembly cached:

    frozen truncation    2.29 ms
    tikhonov             2.73 ms
    threshold truncation 4.09 ms
    hybrid               5.92 ms

45% dearer than the current default, because it pays for the SVD *and* a wider GSVD, where
the penalty skipped the factorisation and truncation shrank the GSVD. Nothing here is
optimised -- `tension_hybrid` recomputes a full SVD that `regularize` already performs -- so
treat 5.92 ms as an upper bound, not a measurement of the design.

## 7. The search-level test, which is where the case falls apart

2026-09-19/20. Everything above measures the OBJECTIVE. Sections 2-6b say the filter is a
better-behaved function than what it replaces, and that is not in question. Whether a
better-behaved function makes the SOLVER better is a separate claim, and it is the one that
matters. Three measurements, and the filter loses two of them.

Runners are in the scratchpad rather than `notes/`, because two of the three are null and the
third is an argument against the idea; the numbers are here and the code is reconstructible
from this section.

### 7.1 The bracketing families: no effect whatsoever

`notes/run_bracket_filter.py`'s machinery, driven with `Case.tensions` swapped and everything
else -- pencil, collocation, `modes`, bracketers, knob grids, `prec` ladder, budget cap --
held fixed. Five objectives: threshold, tikhonov `mu=1e-11`, hybrid at `c = rtol`, `1e-13`,
`1e-11`.

    family A (isolated, [5, 300], 11 levels)      oracle 8/24   auto 3/24   median cost 799
    family B (tight pairs to d=5.9e-12, 6 cases)  oracle 69/144 auto 66/144 median cost 161

**Identical for all five objectives, cell for cell, down to the evaluation count.** Verified
not to be a plumbing artefact: the five return different values and different vector lengths
(53, 126, 58, 58, 53 columns at `lam = 150`). The reason is simple in hindsight -- these
bracketers find wells by DEPTH, the wells drop fifteen decades, and 0.4% of plateau noise is
nothing against that.

So the noise the filter removes is not what limits bracketing. Whatever the filter is for, it
is not this.

### 7.2 Section 34's grid is the real test, and the first run of it was misconfigured

Section 34's "30 of 55 cells" is NOT a bracketing measurement -- an earlier draft of this note
cited it as one, wrongly. It counts DECLINES from `bracket_interval`'s slope-estimation seed
loop, carrying "the slope estimate never settled", on the split degeneracy at `145*pi^2`. The
remedy section 34 adopts is freezing the rank. So the question is exactly: **does a filtered
objective recover those declines without frozen rank's floor?**

Re-run with only `Objective.tensions` varying, 5 `prec` x 11 bands:

             objective   right   wrong  declined  of which slope
       threshold (current)     3       0        52              52
               frozen rank    51       0         4               4
         tikhonov mu=1e-11    33      22         0               0
         hybrid c=rtol p=2     2       0        53              53
        hybrid c=1e-13 p=2    33      22         0               0
        hybrid c=1e-11 p=2    33      22         0               0

Three readings, and the middle one is a mistake of the harness rather than of the method.

* **Frozen rank wins outright**: 51 right, 0 wrong, 4 declined. Section 34 reproduces.
* **The fixed-strength filters trade declines for WRONG ANSWERS**, 33/22/0. That is strictly
  worse than declining, and section 34 draws the same distinction about a different remedy
  that "bought its 27 with 9 wrong answers".
* **But those 22 are a misconfiguration.** A single `mu` or `c` was held across a `prec`
  ladder spanning eight decades. Locating uncertainty tracks the floor (section 4) and the
  pair separation on this grid is `band*prec*lam`, so a 1e-13 floor cannot resolve a 1e-14
  pair. Checked directly -- every wrong cell is at `prec` 1e-12 or 1e-14, 22 = two rungs x
  eleven bands:

        prec  band    sep (abs)   floor (c=1e-13)   verdict
       1e-06   0.5    7.155e-04             1e-13     right
       1e-10   6.0    8.587e-07             1e-13     right
       1e-14   0.5    7.276e-12             1e-13     wrong
       1e-14   6.0    8.572e-11             1e-13     wrong

  A filter strength has to be a function of `prec`, exactly as `rtol` already is. Section 4
  says so and this run did not do it.

* **`hybrid c=rtol` declines 53 of 55, no better than threshold's 52.** That one is NOT a
  misconfiguration, and it is the result that matters: at the single setting where the hybrid
  holds the threshold's floor, it does not smooth the curve enough for the slope loop to
  settle. Section 6b measured 1.6x at matched floor; 1.6x is not enough, and here is what
  that buys in solver outcomes.

### 7.3 Prec-aware filtering, which is the fair test

2026-09-20. Same grid, same everything, with the filter strength made a FUNCTION of `prec`.
Two rules: `c = sqrt(prec*eps)`, which is `chladni`'s own `rtol` rule evaluated at the
requested precision, and `c = 0.01*prec`, chosen to sit two decades under the merge licence.

                 objective   right   wrong  declined
       threshold (current)       3       0        52
               frozen rank      51       0         4
   hybrid c=sqrt(prec*eps)      45       1         9
        hybrid c=0.01*prec      34      10        11
     tikhonov mu=0.01*prec      43       0        12

    per prec (right / wrong / declined)
                 objective     p=1e-06     p=1e-08     p=1e-10     p=1e-12     p=1e-14
       threshold (current)    1/ 0/10     0/ 0/11     1/ 0/10     1/ 0/10     0/ 0/11
               frozen rank   10/ 0/1     11/ 0/0     10/ 0/1     10/ 0/1     10/ 0/1
   hybrid c=sqrt(prec*eps)   11/ 0/0     10/ 1/0     11/ 0/0     11/ 0/0      2/ 0/9
        hybrid c=0.01*prec   11/ 0/0     11/ 0/0      1/10/0     11/ 0/0      0/ 0/11
     tikhonov mu=0.01*prec   11/ 0/0     11/ 0/0     11/ 0/0     10/ 0/1      0/ 0/11

Every decline in every row is "the slope estimate never settled".

**The misconfiguration diagnosis was right.** Tying the strength to `prec` turns 33 right / 22
wrong into 45 right / 1 wrong. The 22 wrong answers were the harness, not the method.

**The best rule is the one already in the code.** `c = sqrt(prec*eps)` -- `MPSPencil`'s own
`rtol` default, evaluated at the requested precision -- is the best filter measured, and it
beats the hand-picked `0.01*prec` by eleven cells. So `c` is not a new parameter to calibrate;
it is `rtol` with a ramp on it, which is what section 6b suspected from the disk/rect split.

**Every filter still collapses at the bottom rung, and frozen rank does not.** At
`prec = 1e-14` the three filters score 2, 0 and 0 right of 11, while the frozen rank scores
10. This is structural and was predicted: locating power tracks the floor, so at `prec = 1e-14`
the cut must fall to about `eps`, and there the ramp sits on the noise floor where the `s_j`
are themselves jittery -- section 6b's 1.6x, arriving as nine declines.

**And it is still behind.** 45 against 51, on the one grid where the objective's noise is known
to decide the outcome.

**One failure NOT explained.** `c = 0.01*prec` is non-monotone in `prec`: clean at 1e-12
(11/0/0), broken at 1e-10 (1 right, 10 wrong). The failure is MISSED COVERAGE, not impurity --
at band 0.5 it returns 6 brackets covering 6 of 10 levels, 0 impure, with the widest bracket
1.1e-7 against a licence of 1.5e-7. So the bracketer is not merging levels it should split; it
is not finding them at all. A floor of 1e-12 against a separation of 7e-8 should resolve these
comfortably, so the floor model does not explain it. Checked that the verdict is not a scoring
artefact: this criterion is `bracket_score.BracketScore.ok`'s own (`covered/n_true`, impurity
by span), not a stricter hand-rolled one.

**A wrong answer is the dangerous outcome**, and a rule that produces ten of them at one rung
while being clean at the rungs either side is not a rule anyone should ship on this evidence.
That it is the hand-picked rule rather than the derived one is some comfort, and not much.

### 7.4 The wide window, which is where it finally wins

2026-09-20. Section 8's item 6 named the untested configuration: `bracket_interval` on a WIDE
window. Two runs close it, and the first one shows why the question had to be posed carefully.

**Width alone does not discriminate.** `rect(1, 1/phi)` at `prec = 1e-8`, windows from 6% to
the full 60x, all four objectives: every cell `right`, bracket counts equal, evaluation counts
within 2%. Even the frozen rank at 29 of 63 columns on `[5, 300]` -- the case section 35 calls
hopeless, floor inflated to 2.3e-4 -- brackets all eleven levels. Family A's minimum relative
gap is 2.0e-03, so a well bottoming at 2.3e-4 is still three decades deep against a plateau
near 0.5. **A blown floor only bites when something must be resolved BELOW it**, and isolated
levels never ask that.

So the discriminating case has to be WIDE *and* hold something TIGHT. That configuration had
never been run: section 34's grids are 6% wide, and family A is wide but has no close pair.
Section 34's split rectangle over `[5, 1474]` instead of its own +-3% -- 108 levels, 258
columns, the `band*prec` pair at `145*pi^2` buried among them -- 5 `prec` rungs x 3 bands:

                     objective  right  wrong  decl    evals  spurious br
           threshold (current)     15      0     0    82339          807
                   frozen rank      0      0    15   151695            0
       hybrid c=sqrt(prec*eps)     14      1     0    42330           38
    tikhonov mu=sqrt(prec*eps)     11      4     0    42591           61

    right-count per prec (of 3 bands)
                     objective   1e-06  1e-08  1e-10  1e-12  1e-14
           threshold (current)       3      3      3      3      3
                   frozen rank       0      0      0      0      0
       hybrid c=sqrt(prec*eps)       3      3      2      3      3
    tikhonov mu=sqrt(prec*eps)       3      3      3      2      0

**The frozen rank collapses completely**, declining all fifteen cells on "the slope estimate
never settled" and burning 10113 evaluations each to do it -- 151k evaluations for nothing.
Rank 31 where the window needs up to 108. This is section 35's wide-window prediction
confirmed at the SEARCH level for the first time, and it is the exact inverse of 7.3, where
the frozen rank scored 51 of 55 and every filter lost.

**The threshold is correct everywhere and pays for it in spurious brackets**: 807 across the
sweep against the hybrid's 38, at 1.9x the evaluations. The worst cell is band 6 at
`prec = 1e-14` -- 360 brackets for 108 levels, 252 of them holding nothing, 19510 evaluations
and 410 seconds, against the hybrid's 108 brackets, 0 spurious, 3916 evaluations. That is the
0.4% plateau noise of section 3 manufacturing false wells, and here is what it costs.

**The hybrid is the best policy on this window**: 14 of 15, half the threshold's evaluations,
a twentieth of its spurious brackets, one missed level in one cell.

**And it beats plain Tikhonov for exactly the reason section 6b predicted.** Tikhonov is fine
down to `prec = 1e-10` and then collapses -- 2 of 3 at 1e-12, **0 of 3 at 1e-14**, losing
half the spectrum (60, 75 and 59 levels covered of 108). At `prec = 1e-14` the rule asks for
`mu = 1.5e-15` and section 2 measured that a penalty SATURATES near 9e-14: it cannot deliver
that floor, so it loses the levels. The hybrid can, because hard-cutting the null zone is what
buys the last two decades. Section 6b argued this from the objective; here it decides four
cells. An earlier three-cell read of this sweep had Tikhonov ahead of the hybrid -- that was
the one `prec` rung where the difference does not show.

### 7.5 The L-shape, and the axis this whole note has been mis-stating

2026-09-20. Every measurement above 7.5 is a rectangle with a plane-wave basis, which is a
poor place to settle a question about a solver whose stated purpose is corner singularities.
The L-shape closes that, and the most important thing it says was established BEFORE the run.

**With a pure corner-FB basis the rank does not move.** Threshold rank against `n_cols`, over
`[5, 130.119]` -- the full extent of the 14-digit table, 25 levels, so truth is complete:

    n_fb    n_cols   threshold rank   truncation
      80        80         80..80     INERT
     160       160       160..160     INERT
     240       240       232..234     active, spread 2

Against the rectangle's `29..63` over its 60x window -- spread 34. Truncation is inert at the
two smaller bases whatever the window, and at 240 it trims five to eight columns and moves by
two. **There is no dilemma here to dissolve.**

That reframes the whole note. Sections 2-6b, and section 35 before them, treat this as a
question about DOMAINS and WINDOW WIDTHS. It is not. It is a question about whether the
basis's numerical rank GROWS WITH `lam`, and that is a property of the family:
`PlaneWave.n_useful_at(k)` predicts the growth for plane waves and a corner-adapted FB family
mostly does not have it. The rectangle has the problem because of its basis, not its shape.

**And the measurement agrees -- nothing discriminates.** `bracket_interval`, four `prec`
rungs, both bases:

    n_fb = 160 (INERT control): all four objectives identical at every rung -- 22 brackets,
    same evaluation counts to within three, 0 spurious. At prec = 1e-12 all four are wrong
    the same way (24 of 25 covered), which is the basis floor, not a policy.

    n_fb = 240 (active):
      prec     threshold          frozen             hybrid             tikhonov
      1e-06    right 22br/0sp     right 23br/1sp     right 22br/0sp     right 22br/0sp
      1e-08    right 22br/0sp     right 22br/0sp     right 22br/0sp     right 22br/0sp
      1e-10    right 22br/0sp     right 22br/0sp     right 23br/1sp     right 23br/1sp
      1e-12    right 22br/0sp     WRONG 24/25        WRONG 24/25        right 22br/0sp

Differences of one bracket, scattered in both directions with no pattern -- frozen is the odd
one out at 1e-6, the two filters at 1e-10, frozen and the hybrid at 1e-12. On 25 levels that
is noise. (22 brackets for 25 levels is correct: the table carries an exact double at
128.3049, and the merges are legitimate -- full coverage, 0 impure.)

**What this costs the case for filtering.** The filter can only matter where the rank grows
with `lam`. So far that means plane waves, which means smooth or right-angled boundaries --
and `CLAUDE.md` calls composite bases (FB at the corners PLUS plane waves or clustered
sources) essential for real domains. Whether a composite basis inherits the plane-wave rank
growth is the question this turns into, and it is not measured here.

## 8. What I would take from this

Rewritten twice: once after section 7.3, which made the case look dead, and again after 7.4
and 7.5, which made it regime-dependent. The flat "no" that stood here between those two
revisions was wrong, and it was wrong because the only wide-window test at the time used
isolated levels, which cannot discriminate. Annotated a third time on 2026-09-23, when the
project decided the penalty is the intended default (item 6); the measurements are unchanged and
items 4 and 5 are the ones that decision has to answer.

1. **Section 35's proposal needs restating**: a filter must be a PENALTY, not a weight, by
   right-invariance (section 1). That correction is independent of everything below.
2. **Section 34's problem is noise, not a discontinuity** (section 3): the threshold rank
   flickers at every scale, leaving an irreducible 0.4% spread that no sub-windowing can route
   around. Section 7.4 prices it -- 807 spurious brackets across a sweep where the hybrid has
   38, and one cell where it costs 252 false wells and 19510 evaluations.
3. **The axis is the BASIS, not the domain or the window** (7.5). Everything here turns on
   whether the numerical rank grows with `lam`. Plane waves: 29..63 over a 60x window, and the
   policy decides whether the solver works at all. Corner-adapted FB on the L-shape: 232..234
   over a 26x window, and the policy is invisible -- all four objectives differ by at most one
   bracket, scattered, at every `prec`. Section 35 and sections 2-6b of this note all frame it
   as a domain/window question. That framing is wrong.
4. **No single policy is right, and the regimes are cleanly separated.**

       regime                                    winner            loser
       narrow window, tight pair  (7.3)          frozen 51/55      filters 45, threshold 3
       wide window, tight pair    (7.4)          hybrid 14/15      frozen 0/15 (all declined)
       wide window, isolated      (7.4)          nobody -- all four identical
       corner-FB basis            (7.5)          nobody -- rank does not move

   The two horns are real and they are OPPOSITE. Freezing is nearly free below ~10% width and
   catastrophic above it; filtering is the reverse. A solver that picks one globally is wrong
   half the time, and the quantity that decides is `d(rank)` over the window -- which
   `MPSPencil` can measure before choosing.

   **This is the row the 2026-09-23 decision has to survive**, and it is not yet measured for
   the mode that ships: the narrow-window numbers above are hand-set `tau` and the order-2 form,
   never `"tikhonov_qr"` at its own derived `rtol`. `TODO.md` section 4, gate 0.
5. **Within filtering, the hybrid is the right form and plain Tikhonov is not** (7.4). Tikhonov
   collapses at `prec = 1e-14`, losing half the spectrum, because a penalty saturates near
   9e-14 and cannot deliver a 1.5e-15 floor (section 2). Hard-cutting the null zone buys those
   last two decades, which is what section 6b argued from the objective and what decides four
   cells here. `c = sqrt(prec*eps)` -- `MPSPencil`'s own `rtol` rule -- is the strength, so
   this is not a new parameter.

   **Read that as "at `prec = 1e-14`", not in general.** Plain Tikhonov in its QR form is what
   ships and what is intended as the default; what it cannot do is the bottom rung, where its
   floor lands above the precision asked for. That is gate 2 in `TODO.md` section 4 -- a
   `prec`-dependent default, or the bottom rung waiting for the second-order form.
6. **Should it replace the incumbents? On the evidence below, JOIN them -- but the project has
   since decided otherwise, and deliberately.** DECIDED 2026-09-23: the penalty is the intended
   DEFAULT going forward, work remaining and all (`TODO.md` section 4). The argument that beats
   the table below is not in the table: a threshold rank puts a decision in the hands of rounding,
   that decision has already cost a real answer on Linux, and no tuning removes it -- while every
   cost here is a number that can be measured and paid. What follows is the evidence that decision
   has to live with, which is why it is left standing rather than rewritten.

   On that evidence: keep the frozen rank for narrow
   windows, where it wins outright and the filter cannot help; the filter earns its place only
   where the rank moves a lot across the window, and it pays 45% per evaluation plus a seam
   change through `rank_at`, `coef`, `choose_rank`, `RankDiagnostics`, `n_useful` and
   `Objective.sigma` for the privilege. Section 35's own cheapest-thing-that-follows -- size
   sub-windows so `d(rank)` stays small, and freeze within each -- buys the same outcome with
   no new numerics, and 7.4's result does not displace it. It only shows what the bill is when
   sub-windowing is not done: the wide window is not hypothetical, it is what a first pass over
   an unknown spectrum looks like.
7. **The load-bearing gap is now composite bases.** The filter matters only where the rank
   grows with `lam`, which so far means plane waves, which means smooth or right-angled
   boundaries. `CLAUDE.md` calls composite bases essential for real domains -- FB at the
   corners PLUS plane waves or clustered sources -- and whether a composite inherits the
   plane-wave rank growth decides whether any of this is reachable on the domains the project
   is for. Not measured.
8. **Calibration.** This investigation cost two harness bugs (one silently re-ran family A
   under family B's label; one would have scored every cell wrong by squaring `lam` brackets),
   one misconfiguration corrected in 7.3, and three retracted claims -- the "22 jumps"
   aliasing artefact of section 3, a deleted claim in section 2 that no window width favours
   freezing, and the flat rejection this section carried between 7.3 and 7.4. Every one of
   those was caught by running the next measurement rather than by reasoning, which is the
   argument for item 7 being run rather than predicted.

## 9. Integrated: the plain penalty, as an option

2026-09-21. Landed on branch `regularization/tikhonov` as `reg="tikhonov_qr"` and
`reg="tikhonov"` on `MPSPencil` and `MPSEigensolver`. **Not the default yet -- it is the INTENDED
default** (decided 2026-09-23, gates in `TODO.md` section 4), and deliberately the
SIMPLE form: a plain penalty with `tau = rtol * ||[A_B; A_I]||_F`, which on the equilibrated
stack is exactly `rtol * sqrt(n_cols)`. The hybrid of section 6b is not implemented; it stays
here, and section 7.4 is the evidence for wanting it later.

**Two realizations were specified, and only one removes the rank.** `"tikhonov"` is the GSVD of
`([A_B; tau*I], A_I)`; `"tikhonov_qr"` takes an unpivoted QR of `[A_B; tau*I; A_I]` and hands the
GSVD the two row blocks of `Q`. In exact arithmetic they are the same problem -- `R` is
invertible, and section 1's right-invariance applies -- and at the default `rtol` they agree to
6e-8 .. 4e-7 over the first five tensions. But `tau*I` makes the STACK full rank and does nothing
for the INTERIOR block, and LAPACK's `?ggsvp3` decides that block's rank with its own tolerance.
So the literal form moves the rank decision inside LAPACK rather than removing it:

    finite-tension count                  rect(1,1/phi), 126 cols      L-shape fixture, 50 cols
                                          lam = 35.7 / 150 / 261       24 lam over [9, 60]
    threshold rank (the svd default)      39..61 over the window       42..48
    "tikhonov"                            33 / 43 / 48                 41..50
    "tikhonov_qr"                         63 / 63 / 63  (= n_int)      50 at every lam

`"tikhonov_qr"` hands LAPACK an orthonormal stack and there is nothing for it to decide. It is the
mode to use; `"tikhonov"` ships as the literal formulation and as the oracle the QR form is
checked against.

**What was verified** (`tests/test_regularization.py`, all passing):

* No rank change. `rank_at == n_cols` and the finite count is constant across a scan where the
  threshold takes seven values -- and the literal form's count moves, which the test pins.
* Differentiability where the threshold flickers (section 3's point, `lam = 38.2617`). Spread
  of `sigma_1` at half-widths 2e-4 .. 2e-8: the penalty falls 1.56e-4, 1.56e-5, 1.53e-6,
  1.61e-7 -- a decade per decade -- then flattens at its own floor, 8.5e-8. The threshold stays
  at 2.8e-3 .. 3.8e-3 throughout.
* The root cause, modelled. A 1e-15 relative jitter -- a cross-BLAS difference -- at twelve
  points whose first dropped singular value sits at 0.98x the cutoff: the threshold rank flips
  at five and `sigma_1` moves by up to 4.2e-3; the penalty moves by at most 6.6e-8.
* The 2026-09-21 CI refusal, replayed. At the refused bracket end the injected rank-86/85 ratio
  reproduces the session note's 0.962 / 0.781 / 0.461 / 0.328 / 1.000; the penalty is a
  straight ramp there (quadratic-fit residual 5.6e-10). `refine` on that bracket with the flip
  injected refuses `not_a_minimum` at `sigma = 5.330e-7`, as recorded; under `"tikhonov_qr"` it
  accepts, converged, 4 evaluations, 1e-10 relative from the root. The whole cell solves 4 of 4.

**What it costs.** Per tension evaluation, assembly cached: 4.83 ms against the threshold's
3.18 on 126 columns, 13.7 against 8.85 on 190 -- about 1.5x. (Measured while the full suite
ran alongside, so the absolute numbers are inflated; the ratio was consistent.) The literal
form is the cheapest of the three, 2.55 and 7.51 -- and is the one that leaves a rank decision
in place.

**Rules for using it**, each already paid for once:

* **Use the `prec`-derived `rtol`**, which is `MPSPencil`'s default. At the suite's fixed
  `rtol = 1.5e-15` the two realizations part by 1e-3 and a plain penalty is in its own
  saturated-noise regime (section 2). The regression tests build penalty pencils with `prec=`,
  never with the suite pencil's `rtol`.
* **Mind the noise at tight `prec`.** Written here first as "not below `prec` of about 1e-12,
  because a plain penalty saturates near 9e-14" -- that is WRONG for the QR form, and section 10
  corrects it: the saturation was the literal form's, and the QR form's floor tracks `tau` to
  2e-15. What limits it is plateau noise, which grows as `tau` shrinks.

**Still open**, and recorded in `TODO.md`:

1. **The Linux runner.** The CI failure needed OpenBLAS's rounding; on macOS the threshold
   passes the cell too. The penalty's solve of the cell is the portable acceptance test, and
   only a Linux run of it closes the blocking item.
2. Reaching `prec = 1e-14` -- section 10.
3. Making it the default, which is now the stated intent rather than an open question. What is
   open is the gate list in `TODO.md` section 4 -- above all that the grids have never been run
   under `"tikhonov_qr"` at its own derived `rtol`, only under hand-set `tau`.
4. `calibrate_rtol` under a penalty: it runs, because raising `rtol` raises the floor either way,
   but its model is written for truncation and is untested here.
5. A `PrecisionWarning` for a penalty mode at `prec` below its reach -- wanted, but there is no
   calibrated threshold to warn at yet.
6. The CS-decomposition shortcut: `[Q_1; Q_2]` is orthonormal, so an SVD of `Q_2` alone gives the
   sines. Cheaper, but it computes small tensions to absolute rather than relative accuracy,
   so it is not a free substitution.

## 10. Reaching `prec = 1e-14`: the floor was never the obstacle

2026-09-21, in progress. Every measurement behind "a plain penalty saturates near 9e-14" --
sections 2 and 7.4 -- used `tension_tikhonov`, the LITERAL form, which section 9 found lets
LAPACK re-truncate internally. Measured on the QR form instead, family A, 126 columns, worst
`sigma_1` over the 11 true levels and plateau noise (linear-fit residual, 1e-8 window):

      rtol      tau        literal floor    QR floor    QR noise
    1.49e-12  1.7e-11        2.3e-11        2.3e-11     5.2e-08
    1.49e-14  1.7e-13        4.5e-13        2.4e-13     7.2e-06
    1.49e-15  1.7e-14        1.0e-13        2.4e-14     5.8e-04
    3.0e-16   3.4e-15        9.3e-14        5.2e-15     1.0e-02
    1.0e-16   1.1e-15        9.3e-14        2.1e-15     6.1e-02
    threshold (1.49e-15)                    1.6e-15     1.0e-03

**The QR form's floor is 1.38-1.45 `tau` down to `tau = 1.7e-14`**, rising to 1.9 `tau` at the
bottom where the basis's own resolution enters -- 2.1e-15 there, within 1.3x of the threshold's. The literal form's 9.3e-14 is LAPACK's `?ggsvp3` tolerance, not the penalty.

**The obstacle is noise, and it is multiplicative, not a plateau effect.** Relative noise holds
at 6e-3 (`rtol = 3e-16`) and 3e-2 (`1e-16`) all the way down a flank to `sigma ~ 3e-8`, and a
smaller `tau` gives a LOWER `sigma` there (3.08e-2 threshold, 2.96e-2 and 2.39e-2 penalty, at
1e-2 relative from a root): a weak penalty lets the minimizer lean on near-null directions whose
content is rounding garbage. The reason it cannot be tuned away: an ORDER-1 penalty charges the
eigenfunction itself `tau * ||c||` -- that IS the floor -- so floor and noise are both set by
`tau` against the rounding level `eps * ||A||`, and there is no `tau` where both are good.
Section 7.2's decline grid says what 1e-3 of noise already costs the slope loop.

**The lead: a second-order penalty, built from QRs.** For the stack `A = Q_0 R_0`,
`||R_0^-T c|| = ||Sigma^-1 V^T c||` exactly, so a numerator block `tau^2 * R^-T` is the filter
`g_j = (tau / s_j)^2` -- nearly free on well-conditioned directions, so the floor can sit far
BELOW `tau`, and it beats the rounding garbage in noise directions by the SQUARE of the ratio.
Taking `R` from the `tau_1`-augmented QR that `"tikhonov_qr"` already computes keeps it invertible
and bounded (`||R^-T|| <= 1/tau_1`). No SVD, no rank decision. First measurement, `tau_1 = 1e-16 *
sqrt(n)`:

      penalty                    worst floor   median floor   worst noise
      threshold                    1.6e-15       1.2e-15        1.0e-03
      order 1, tau = 3.4e-15       5.2e-15       4.8e-15        1.0e-02
      order 2, tau_2 = 1e-13       3.1e-14       8.2e-15        8.2e-06
      order 2, tau_2 = 3e-14       7.1e-15       4.3e-15        3.7e-05
      order 2, tau_2 = 1e-14       4.1e-15       2.2e-15        3.2e-04

At a matched 5e-15 floor, order 2 is about 270x quieter than order 1 and 27x quieter than the
threshold -- the first penalty to beat the threshold on noise at a 1e-14-capable floor. This is
section 6b's `p = 2` hybrid WITHOUT its SVD or its hard cut, which is why it is worth pursuing.
**Through the solver: it beats the frozen rank on the frozen rank's own grid.** Section 34's
`145*pi^2` split-degeneracy grid at `prec = 1e-14`, 11 bands, `bracket_interval`, only
`Objective.tensions` varying:

                   objective   right   wrong  declined
         threshold (current)       0       0        11     all "slope estimate never settled"
                 frozen rank      10       0         1
       order-1 QR, rtol 3e-16      0       0        11     all "slope estimate never settled"
          order-2, tau_2 3e-14     11       0         0
          order-2, tau_2 1e-14     10       0         1

This is the grid where the frozen rank beat every filter in section 7.3, at the rung where every
filter scored 0 to 2 of 11. Order 2 at `tau_2 = 3e-14` takes all eleven, no wrong answers, no
declines. The ordering between the two strengths is the noise table's: `tau_2 = 1e-14` is 9x
noisier and loses one cell to the slope loop. Order 1 at the `tau` a 1e-14 floor needs declines
every cell, exactly as its 1e-2 noise predicts. On the one cell probed ahead of the row (band
1.3), the threshold's decline reproduces section 34's secant history digit for digit
(0.7101, 0.8090, 2.201, 3.920), which says the harness measures what section 34 measured.

What this does NOT yet show: one grid, one rung, one margin of a single cell over the frozen rank;
`tau_2` is a hand-picked constant, not a rule in `prec`; the WIDE window of section 7.4, where the
frozen rank declined all fifteen, is unmeasured for order 2; and the cost is about 2x the QR
order-1 form. `notes/tikhonov_derivations.tex` has the derivation.

### 10.1 The `tau_2` rule: what is settled

2026-09-21. **Floor law, five pencils.** Worst `sigma_1` at the true levels over `tau_2` from 1e-6
to 1e-14, absolute on the equilibrated pencil (`tau_1 = 1e-16 sqrt(n)`):

    pencil               n     worst floor / tau_2
    rect(1, 1/phi)      126    0.08 .. 0.41
    near-square N65     190    0.07 .. 0.28
    eq_tri               90    0.10 .. 0.30
    disk                 84    << 1 -- on the basis floor, 1.8e-15, for every tau_2 <= 1e-10
    L-shape, FB 240     240    << 1 -- on the basis floor, 8e-13, for every tau_2 <= 1e-8

So **worst floor <~ max(eps_basis, 0.41 tau_2)**, and the constant carries across n = 84 .. 258
with no `sqrt(n)`: the natural unit for `tau_2` is the RMS column norm (1 on an equilibrated
pencil), not the Frobenius norm order 1 uses. Noise falls monotonically as `tau_2` grows on
every pencil (N65 looked flat under a linear-fit probe; that was curvature between close roots,
and a quadratic-detrended probe shows the same fall as the others).

**`tau_1` does not matter.** Floor and noise are flat over `tau_1` = 1e-18 .. 1e-15 on two
pencils. It only keeps `R_1` invertible; fix it at `eps` in RMS-column units.

**The first candidate, `tau_2 = C * prec`, fails -- and it fails SILENTLY.** End to end on the
cluster-totals grid (full `solve`, even 50*pi^2 triple and uneven 65*pi^2 quadruple), the
wrong-total count at `prec = 1e-14`, 18 exercisable cells:

    C        1    1.5    2    2.5    3    4    5    8
    wrong    2     1     2     1     0   11   10   16

C = 3 scored 34 of 34 across three rungs in the first pass and looked like the answer. It is an
ISOLATED zero: every neighbour has wrong totals, and above it the grid collapses. The wrong cells
are almost all the even triple at 493.48, declared 2 of 3 -- a member LOST, not merged. The
mechanism, corrected after the loose-`prec` cells were pulled apart: a floor near the merge
licence (`C * prec` puts it at up to 0.41 C prec) breaks multiplicity ACCOUNTING in BOTH
directions. At `prec = 1e-6`, C = 2, band 1.3, the triple is declared 5 of 3: the middle member
reported three times, at 493.47955009 and twice at 493.47959664 -- 2e-15 apart, an exact duplicate
that must merge -- because a well whose bottom is as wide as the floor holds several local minima
and nothing merges them. A first draft of this paragraph said only "wells blur", which is half of
it. (For reference the threshold runs at floor ~0.16 prec at 1e-14.)

One wrong cell does NOT fit: C = 0.1 at `prec = 1e-6`, band 0.5, declared 2 of 3 with the middle
member lost, at a floor of ~4e-8, far under the separation. That cell's triple has DIAMETER
exactly equal to the licence (2 x 0.5 prec), the configuration `rect_suite.cluster_diameter_ratio`
calls "an UNDECIDED one ... should be reported, not scored". It is a boundary cell, not evidence
about the floor, but it is scored by the grid as written.

Accuracy was never the problem: in every RIGHT cell, for every C and every rung, the worst
eigenvalue error was at most 0.071 prec. And the decline grid, which scores brackets and not
totals, gives C = 2 and C = 3 both 11 of 11 -- it cannot see this failure, which is why both
grids are needed.

Criterion from here: **the largest C with zero wrong totals across a NEIGHBOURHOOD of C**, with
declines accepted as honest.



### 10.2 The sweep below C = 1: no robust rule at `prec = 1e-14`

2026-09-21. `tau_2 = C * prec`, `tau_1 = eps`, both grids, every rung (right / wrong / declined):

      C    cluster 1e-14   cluster 1e-10   cluster 1e-6   decline 1e-14
     0.1      0/0/18           8/0/0          7/1/0*          1/0/10
     0.2      2/0/16           8/0/0          8/0/0           0/0/11
     0.3      3/0/15           8/0/0          8/0/0           0/1/10
     0.5     10/0/8            8/0/0          8/0/0           3/0/8
     0.6     14/1/3            8/0/0          8/0/0           9/0/2
     0.7     18/0/0            8/0/0          8/0/0          11/0/0
     0.8     18/0/0            8/0/0          8/0/0          11/0/0
     0.9     17/1/0            8/0/0          8/0/0          11/0/0

    * the diameter-equals-licence boundary cell of section 10.1

Wrong totals on the 1e-14 cluster row across the whole sweep, 10.1 included:

    C       0.1  0.2  0.3  0.5  0.6  0.7  0.8  0.9  1.0  1.5  2.0  2.5  3.0  4.0  5.0  8.0
    wrong    0    0    0    0    1    0    0    1    2    1    2    1    0   11   10   16

**No robust rule exists at `prec = 1e-14`.** C = 0.7 and 0.8 are perfect on both grids at every
rung -- but they are an island flanked by wrong totals at 0.6 and 0.9, the same shape as the
isolated zero at C = 3 that section 10.1 declined to trust. From C = 0.6 upward wrong totals appear
sporadically; below it the 1e-14 row drowns in slope-loop declines (honest, useless). There is no
C with zero wrong totals across a neighbourhood and few declines. The wrong cells are
multiplicity ACCOUNTING failures (section 10.1), which is a defect of the search's merge step
reacting to flat-bottomed wells, not something a choice of C can remove.

**At `prec` of 1e-10 and looser the rule is easy:** every C from 0.2 to 0.9 is clean on every cell
measured. So order 2 with `tau_2 = C * prec`, C around 0.5, is a defensible rule down to 1e-10; it
is the 1e-14 rung, and only that rung, where it has no safe setting.

**Where this leaves order 2.** It is the only penalty measured that gets the SLOPE LOOP through at
1e-14 (11 of 11 on the decline grid where the threshold and order 1 decline all eleven), so the
noise mechanism of section 10 is right and the construction works. What stops it is downstream: the
multiplicity accounting of a solver whose merge step reports one root several times (5 of 3, two of
them 2e-15 apart). Two things should come before any further tuning here:

1. **Fix the merge.** An exact duplicate 2e-15 apart must merge under the project's own policy,
   whatever the objective. That is a search defect this sweep exposed, not a regularization one.
2. **Right-size the bases** (`notes/coarse_to_fine_basis.md`). The noise directions exist because
   plane-wave bases are over-complete; a basis sized to `n_useful` may remove enough of them that
   the threshold stops flickering and no penalty is needed.

Then re-ask the 1e-14 question. The order-2 construction and its derivation
(`notes/tikhonov_derivations.tex`) are kept for that.

### 10.3 The accounting failure was two defects, and one is fixed

2026-09-22. Section 10.2's "one root reported three times" at 50*pi^2, `prec = 1e-6`, band 1.3
(order 2, `tau_2 = 2 prec`: 5 of 3) is two separate defects, traced:

1. **One eigenvalue reported TWICE, 1e-12 apart -- fixed.** Stage 3's cache wall slope,
   `search._slope_from_cache`, uses the cone identity `c = (v_lo + v_hi)/(hi - lo)`, which assumes
   `sigma` reaches zero. With a 9.6e-7 floor and the nearest samples a few ulps apart it read
   `2*floor/(hi - lo)` = 8.45e6 against a true 0.02-0.08. The side search excludes `r +- half`
   with `half = 0.5 sigma_j / c`, so the radius collapsed to 1e-13, the side search refined against
   the root itself, and found it again as a new level. Fixed with a GUARD, after a first version
   failed the gate: a floor-aware estimate (samples within twice the root's `sigma` skipped as flat
   bottom, flank values reduced to `sqrt(v^2 - floor^2)`) applied UNCONDITIONALLY flipped four
   calibrated tests, because at a hump top accepted as a root (65*pi^2, prec 1e-8, band 0.3)
   `sigma` is not a floor and the estimate moved 2.3x. The two estimates agree within 4x at every
   genuine or hump root measured and differ by 4.6e8 at the defect, so the floor-aware one is used
   only where the blind one exceeds it by more than 100x; everywhere else the calibrated slope is
   unchanged bit for bit. 5 of 3 became 4 of 3.
2. **A flank point accepted as a root -- not fixed, recorded in `TODO.md` section 4.** The bracket
   lying between two members converged 2 ulps inside its lower end, on the wall of the member
   below. `refine`'s end probe reaches `sigma / c_wall` outward with `c_wall` the steepest secant
   INSIDE the bracket (the hump side here), so it undershoots the gentle descent toward the
   neighbour: probes read 0.54-0.94 of the point against the 0.5 that marks a phantom. That test
   assumes a cone, carries five calibrated cases, and only a penalty with a floor near the licence
   defeats it -- so it waits until such a penalty is actually headed for `src/`.

So the multiplicity failures that block a `prec = 1e-14` rule for order 2 (section 10.2) are
partly the solver's, and one of them is now gone. The other is why order 2 stays parked.

### 10.4 The prec limit of the SHIPPED option, drawn

2026-09-23. `notes/fig_penalty_prec_limit.py` -> `notes/fig_penalty_prec_limit.png`. The worst
floor at the true levels under `MPSPencil(reg="tikhonov_qr", prec=P)` -- the shipped path, `rtol`
derived by its own rule -- against `P`:

    prec      rect(1,1/phi)      disk            eq_tri        floor/prec
    1e-06     2.24e-10         1.00e-10        1.89e-10          ~0.002
    1e-08     2.31e-11         1.00e-11        1.91e-11          ~0.002
    1e-10     2.40e-12         1.00e-12        1.96e-12          ~0.02
    1e-12     2.43e-13         1.00e-13        2.07e-13          0.10-0.24
    1e-14     2.43e-14         1.01e-14        2.11e-14          1.0-2.4
    threshold 1.28e-15         1.65e-15        1.48e-15

**The rule is mismatched to the mechanism, and that is the whole of it.** `rtol = sqrt(prec*eps)`
was derived for a TRUNCATION, whose floor is set by the basis and barely moves with the cutoff --
the threshold row is flat at ~1.5e-15 whatever `prec`. A PENALTY's floor IS its `tau`, so it
scales as `sqrt(prec)` while the requirement scales as `prec`, and `floor/prec` grows like
`1/sqrt(prec)`. On the rect, `1.45*sqrt(n*eps/prec)` predicts 2.43 against 2.43 measured. It must
cross 1, and it crosses between 1e-12 and 1e-14.

What that costs, on the 50*pi^2 triple dialled to one merge licence at `prec = 1e-14` (members
1.0e-14 apart, relative):

    threshold     floor 1.9e-15, barrier/floor 23.6 -- uncertainty BELOW the spacing: resolvable
    tikhonov_qr   floor 2.5e-14, barrier/floor  2.1 -- uncertainty ABOVE the spacing: not

The three wells are still visible under the penalty; what fails is that its locating uncertainty,
which section 4 measured as the floor itself, is wider than the spacing it has to resolve.

Two honest qualifications. This is OBJECTIVE-LEVEL -- gate 0 (`TODO.md` section 4) is still the
measurement nobody has made. And the disk lands at `floor/prec = 1.0` rather than above it: its
basis is comfortable enough that the floor sits below `tau`, so the limit bites hardest exactly
where the basis is working hardest.
