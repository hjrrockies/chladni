# Narrow tension wells, and calibrating the rank truncation

Findings from an investigation of a **`lappy` 0.9.0** failure, carried out in the `douse`
repository (`benchmarks/missing_mode_cache.py`, `benchmarks/sigma_base_ablation.py`,
`benchmarks/tension_slope_universality.py`, `jupyter_notebooks/missing_mode.ipynb`). Dated
2026-09-02; numbers are measurements of `lappy`, not of `chladni`.

Read section 6 before porting any threshold value. **`lappy`'s `rtol` is an absolute cutoff on
singular values; `chladni`'s is relative to the largest.** The conclusions transfer; the numbers
do not.

This note overlaps `notes/landscape_findings.md` and `docs/design/lipschitz-cone-mps.md`, which
already have the cone model and per-solve calibration. Section 4 is the part that is new, and it
argues against a conclusion those files reach.

---

## 1. The failure, concretely

During a shape-optimization run maximizing `lam_3/lam_1` over perimeter-4 triangles, the
optimizer stepped onto a triangle with interior angles 21.9 / 25.0 / 133.1 degrees. A **fresh**
full solve there returned

    lams  = [93.0733, 198.4611, 256.2323, 381.8813]
    sigma = 2.8e-14 at lam_1

with no exception and no warning. A P1 FEM reference (one-sided, so this is a proof rather than a
heuristic) gives the true spectrum as `[93.16, 148.00, 198.86, 256.90, 306.83, 328.55, 383.36]`.
The returned set is **(lam_1, lam_3, lam_4, lam_7)** -- every value accurate, the set not
consecutive. The objective the optimizer recorded was 2.7530; the true `lam_3/lam_1` is 2.1346.

Three properties of this failure are worth carrying into `chladni`'s design:

- **The solver's own quality indicator is useless here.** The tension at `lam_1` was 2.8e-14. It
  measures how well the basis resolves what was *found*, and is structurally blind to what was
  not.
- **Counting detectors are one-sided in the wrong way.** A Weyl deficit scored +2.02 and the FEM
  count 2, against three modes actually absent: no cut can be placed above the top returned
  value, so the topmost gap is invisible. Worse, a *duplicated* root (the same eigenvalue
  returned twice, observed at a neighbouring shape) adds a value exactly where one is missing, so
  every counting test balances. The only hint there was the Weyl deficit going **negative**.
- **It is not reproducible by replaying the run.** Re-running the identical start -- same seed,
  same machine, same thread count -- diverged at a trust-radius shrink and converged cleanly,
  never revisiting the shape. A known-bad shape has to be *stored and re-solved*, not replayed.
  The shape itself reproduces bit-for-bit.

---

## 2. Mechanism: the wells are narrower than the scan grid

`lappy`'s scan lays `ppl * m` points uniformly in `lam` across `[faber_krahn, weyl_est(k+1)]`.
For this solve: **100 points over [45.9, 347.0], a step of 3.011**. The tension wells at that
shape are **0.16 to 0.32 wide**. The grid is 10-19x wider than the target.

Consequences measured directly:

- `solve_interval` returned **one** eigenvalue. The other three in the "answer" came from the
  rescue loop, which re-scans deficient gaps `while len(eigs) < k_search` -- a stopping condition
  on the **count**, not on consecutiveness. It filled the quota with whatever it could reach.
- Of 13 rejected brackets, **none** lay near a missing mode. The search did not find those minima
  and reject them; it never bracketed them.
- No sampled node reached `ttol` at *any* shape tested. The search does not work by sampling below
  the acceptance bar -- it brackets local minima of the sampled curve and polishes. So the
  quantity that matters is whether a sampled local minimum lands near a mode. On the failing
  shape: **1 of 6**. On a nearby shape that solved correctly: **6 of 6**.

The second shape is the uncomfortable one. Its scan also found only **2 of 6** modes; it returned
a correct consecutive spectrum only because the rescue loop's fill-in happened to land right. Both
shapes have a broken scan. Only one got away with it.

---

## 3. The dip is a clipped V

Widths measured at 30/50/70/90% of the plateau come out in the ratio 0.3 : 0.5 : 0.7 : 0.9 --
straight flanks, no curvature:

    sigma(lam) = min(sigma_base, C_lam * |lam - lam_k|)

This is the far-field limit of `landscape_findings.md`'s cone model `sigma ~ sqrt(eps^2 + c^2
delta^2)`; the two agree once `delta >> eps`, and everything below is measured in that regime.

Written in wavenumber `k = sqrt(lam)`, the slope `C_k = 2 sqrt(lam) C_lam` is **constant across
modes** -- 0.23 +- 0.01 over 14 measurements spanning a factor of 4 in `lam` on two shapes. So
**the well has a fixed width in `k`**, and widens in `lam` only as an artifact of the change of
variables. `chladni` already works in `k`; the payoff is that the well width becomes a single
number per solve rather than a function of mode index, which is worth asserting in a test.

The width at any threshold `q * sigma_base` is then

    w_k = 2 q sigma_base / C_k

which reproduced every measured width to about 2%. **Only `sigma_base` and `C_k` matter.** The
floor does not, the mode index does not, `lam` does not.

---

## 4. `C_k` scales with `|Omega| / |dOmega|` -- and this may overturn "calibrate per solve"

Across 11 domains and 61 measurements:

    C_k                    spread 46%
    C_k / area             spread 35%
    C_k / sqrt(area)       spread 19%
    C_k / (area/perimeter) spread 10%     2.384 +- 0.231

The dimensional argument says only the last can be right: `sigma` is a ratio of norms and so
dimensionless, `k` is an inverse length, therefore `C_k = dsigma/dk` **has units of length**. Area
does not. Holding the perimeter fixed makes `|Omega|` and `|Omega|/|dOmega|` proportional, which is
how a good-looking area fit arises from a family that cannot distinguish them.

A scale test separates them cleanly. The same 3:1 rectangle at three sizes:

    perimeter    2        4        8
    area         0.1875   0.7500   3.0000
    C_k          0.225    0.451    0.901
    C_k/(A/P)    2.403    2.403    2.403     <-- identical to 4 significant figures

`C_k/area` over those three would read 1.20, 0.601, 0.300. And an equilateral triangle (area
0.770) sits at `C_k = 0.437` against the 3:1 rectangle's 0.451 (area 0.750) -- nearly equal areas,
completely different aspect and corner set -- so the variable is the length scale, not elongation.

**Why this matters here.** `landscape_findings.md` section 2 reports the cone coefficient `c` as
"stable within a domain, not across domains -- 12x across domains", and concludes that per-solve
calibration is necessary. Those same numbers, divided by `|Omega|/|dOmega|`, appear to collapse:

    domain      c (reported)   A/P (assumed)   c / (A/P)
    eq_tri      0.45 - 0.49    0.144           3.1 - 3.4
    rect 2x1    0.98 - 1.17    0.333           2.9 - 3.5
    L-shape     1.10           0.375           2.9
    disk        1.56 - 1.70    0.500           3.1 - 3.4

**The `A/P` column is assumed, not read from the code** -- unit side length, unit radius, and a
3-square L-shape. If those are wrong the arithmetic is wrong. But if they are close, a reported
12x spread becomes roughly 10-15%, matching the independent measurement above, with a constant
near 3.2 rather than 2.38 (a different normalization of `sigma` changes the constant and should
not change the scaling).

That would mean per-solve calibration of the *slope* is **not** necessary: a geometric prefactor
plus a safety factor would do, and the calibration budget could go to `sigma_base` instead, which
genuinely does vary by two orders of magnitude and has no geometric predictor. This is worth ten
minutes against the real areas and perimeters before accepting `lipschitz-cone-mps.md` 3.3 as
settled.

---

## 5. The findability criterion, and two priced fixes

Requiring the visible catchment to be at least one scan step wide, `w_k >= dk`, with `q = 0.9`:

    sigma_base  >=  (kappa / 2q) * (|Omega| / |dOmega|) * dk   ~   1.32 * (|Omega|/|dOmega|) * dk

Everything on the right is known **before any mode is located**. On the failing triangle this
demands `sigma_base >= 1.6e-2` against a measured 2.5e-3 -- the 6.5x shortfall, derived from
geometry and grid alone. It would have flagged the shape as unfindable before the scan ran.

`sigma_base` itself is one cheap measurement: the median tension over an off-resonance grid
(~40 evaluations, no knowledge of the spectrum needed).

Read the inequality either way, and it prices two independent fixes:

1. **Raise `sigma_base`** by loosening the rank truncation. Free at scan time; costs basis
   conditioning. `sigma_base` is monotone in the truncation threshold, so this is a five-step
   bisection onto a computed target rather than a search.
2. **Shrink `dk`** -- `dk <= sigma_base / (1.32 * |Omega|/|dOmega|)`. On the failing triangle,
   `dk <= 0.0191` against the actual 0.124: **100 -> 650 scan points**. Costs linearly in time and
   touches nothing else.

For a black-box solver aiming at reliability over speed, (2) is the safer default: it is a
guarantee, not a tuning gamble, and until this criterion existed there was no way to know that 650
was the number. (1) is the cheaper one when the budget is tight. They compose.

Both need an upper guard: the well floor must stay well below `ttol`, or a bracket that *is* found
gets rejected. In practice this was never binding -- see section 7.

---

## 6. **The rtol convention differs, and the numbers do not port**

`lappy`, `mps.regularize_pencil`, `reg_type='svd'` (the default path):

```python
Q, R = la.qr(A, mode='economic')        # A = vstack([A_B(lam), A_I(lam)])
Z, s, Yt = _svd_gesdd_then_gesvd(R)
cutoff = (s >= rtol).sum()              # ABSOLUTE threshold on singular values of R
```

with `rtol_default = 1e-12`. The `'qrp'` path is likewise absolute (`abs(diag(R)) >= rtol`). Only
`reg_type='implicit'` is relative, and differently so -- it scales `rtol` by
`max(shape) * norm(A, 1)` per block.

**`chladni` thresholds relative to the largest singular value of `R`.** That is the better
convention -- it is invariant under scaling of the domain and of the basis normalization, where
`lappy`'s absolute cutoff silently changes meaning when either moves. But it means:

    rtol_lappy  ~  rtol_chladni * s_max(R)

and `s_max(R)` depends on the basis normalization (`lappy` calls `basis.to_normalized(...)`
before assembly), on the number of collocation points, and on `k`. It is not a constant, so
**there is no fixed conversion factor.** Section 7's values -- 1e-11 and 1e-10 -- are absolute
cutoffs under `lappy`'s normalization at one `lam_max` on two triangles. Porting them as relative
thresholds would be meaningless.

Three specific hazards when implementing this:

- **Do not port a number; port the procedure.** Calibrate against the *observable* `sigma_base`
  and its computed target (section 5). That sidesteps the convention mismatch entirely, and is
  robust to any later change in normalization. If a relative threshold is bisected onto a
  `sigma_base` target, the resulting value is meaningful in `chladni`'s own units by construction.
- **A relative threshold interacts with the rank growing in `k`.**
  `landscape_findings.md` section 3 already records that the numerical rank grows with `k` at
  fixed threshold. With a relative convention, `s_max(R)` also moves with `k`, so the *absolute*
  level being discarded drifts through a scan even when the ratio is held fixed. Whether the
  calibration should be done once per solve or re-done across the band is an open question, and
  the answer is not inherited from `lappy`, which had the opposite convention.
- **The truncation threshold may also size the basis.** In `lappy` it does:
  `basis_plan._indep_digits` sets `ln(1/rtol)` as the dynamic range at which two columns stop
  counting as independent, so changing `rtol` changes the *plan*. Getting that coupling wrong made
  achieved accuracy non-monotone in the requested target. If `chladni` has an analogous coupling,
  calibrating the threshold changes the basis underneath the calibration, and the loop needs to be
  written to converge rather than oscillate. If it has no such coupling, that is a simplification
  worth keeping.

---

## 7. What the calibration bought, and what did not work

Running the real scan at the two study shapes (6 true modes in each window):

    configuration        failing shape   good shape   spurious brackets
    default                    1/6           2/6         13 / 14
    truncation 1e-11           6/6           6/6         10 / 11
    truncation 1e-10           6/6           6/6          2 /  0
    lam_max_window 12          6/6           6/6          8 /  8

On modes that both configurations locate, the values agree to **5e-15 - 2e-14**. Widening the
wells does not move the answer. The spurious brackets collapsing from 13/14 to 2/0 is the
diagnostic tell: the search becomes **cleaner and more complete at once**, which is what
distinguishes this from a trade-off. Discarding directions below the level at which the matrix
entries are actually known removes noise that was both hiding real minima and manufacturing fake
ones.

Dead ends and traps, all measured:

- **`ttol` is never the binding constraint.** Across 44 configurations the well floor stayed at or
  below 3.8e-4 -- every mode that got bracketed got accepted. Raising `ttol` would achieve
  nothing. The failure is findability, never acceptance.
- **The interior-point draw is a symptom, not a knob.** Changing only the RNG seed moved
  `sigma_base` by 14x at fixed basis, and one seed gave 6/6 on one shape and 1/6 on the other,
  returning 246.156 for a mode every other configuration puts at 246.087. Worse, the *default*
  seed was the lucky one on the shape that solved correctly. A random normalization denominator is
  fragility to remove, not tuning to exploit.
- **"Use a coarser basis" is wrong.** The dependence is non-monotone: `target=1e-12` was worse
  (2/6 and 1/6) than either `1e-10` or `1e-14`. Do not expect a monotone accuracy/findability
  trade.
- **Curvature-based auto-selection of the threshold** (`lappy.mps.adapt_rtol`) is reported by its
  author as unreliable. Estimating a noise floor from the smoothness of `sigma` is a harder
  problem than targeting `sigma_base`, which is a direct measurement of the quantity that actually
  matters.

---

## 8. Interaction with `chladni`'s design principles

- **`prec` pulls against findability.** The most accurate settings gave the narrowest wells:
  `target=1e-14` produced `sigma_base = 2.5e-3` and an unfindable mode, `target=1e-10` produced
  1.2e-1 and a findable one. A single `prec` dial driving both basis richness and search
  resolution will fight itself. The criterion in section 5 is the coupling that keeps them
  consistent: given `prec` (which sets `sigma_base` via the basis) it *computes* the `dk` the scan
  needs, instead of leaving them independently tuned.
- **The black-box path needs this check, not just this fix.** `sigma_base` and the target are both
  cheap. A solver that measures them can *report* "this domain needs 650 grid points at this
  precision" rather than silently returning a short spectrum. Given the design preference for
  working harder for a reliable answer over sorta-hard for a probably-reliable one, the check is
  arguably more valuable than the fix.
- **Shape optimization is the demanding client.** The failing shape and a shape that behaves are
  **4.6e-07 apart** in parameter space, deterministically, with the well-behaved one recoverable by
  moving 5.7e-08. The bad set is not something an optimizer can be steered around. Robustness has
  to come from the solver being able to tell, at any shape, whether what it returned is the
  spectrum.

---

## 9. Caveats

- All measurements are `lappy` 0.9.0, Dirichlet, convex polygons, mostly `target=1e-14`, one
  basis-planning scheme. `kappa = 2.38` is specific to `lappy`'s normalization of `sigma`; expect
  a different constant in `chladni` and verify the `|Omega|/|dOmega|` scaling independently.
- The residual 10% in `C_k/(A/P)` is systematic rather than noise: fat domains sit at 2.44-2.58,
  sharp or elongated ones at 2.24-2.31. There is likely a weak corner-angle dependence that a
  wider family would resolve. Carry a safety factor.
- The clipped-V model is a far-field description. Near the floor it must round over into the cone
  model; `w_k` at `q = 0.9` is measured well away from that region, but a criterion evaluated at
  small `q` would need the cone form.
- **Nothing here explains what sets `sigma_base`.** It is a knob, and its default value varies by
  two orders of magnitude across domains in a way that basis column count does not predict --
  81-162 columns spanning `sigma_base` from 3.0e-1 to 2.5e-3, with the sharp-cornered triangles
  far off the trend the rectangles follow. The open hypothesis is the pencil's effective rank and
  the near-dependence of a dense near-boundary source cloud compensating for corner blocks that
  `_fb_ceiling` has starved (the failing triangle got **3** Fourier-Bessel terms at each sharp
  corner against a continuity requirement of ~13, and 112 free-space sources at offsets of
  0.05-0.09 of the diameter). Untested.

---

## 10. Ported into `chladni`: what held, what did not

Written 2026-09-03, after implementing the check as `src/chladni/mps/calibrate.py` and testing
it in `tests/test_calibrate.py`. Section 6 said to port the procedure and not the numbers; that
was right about the procedure and **too pessimistic about the numbers**.

### `kappa` ports. Section 9's caveat was careful and turned out not to bite

Section 9 says "`kappa = 2.38` is specific to `lappy`'s normalization of `sigma`; expect a
different constant in `chladni` and verify the `|Omega|/|dOmega|` scaling independently."
Re-measured on six domains whose plane-wave basis reaches a floor below 1e-6, taking `C_k` as
`sigma_1(k* +/- dk)/dk` on the straight flank:

    domain      A/P      C_k     C_k/(A/P)
    square    0.2500   0.7033      2.813
    rect 2x1  0.3333   0.8582      2.575
    rect 3x1  0.3750   0.8936      2.383
    rect 5x1  0.4167   0.9243      2.218
    eq_tri    0.1443   0.3951      2.737
    disk      0.5000   1.4107      2.821

    kappa_chladni = 2.591 +- 0.226      against    kappa_lappy = 2.384 +- 0.231

Within one standard deviation, so the two normalizations of `sigma` coincide closely enough.
`chladni` ships `KAPPA = 2.85` -- the conservative end above every measured ratio, since a
larger `kappa` raises the target and errs toward flagging.

The residual spread is systematic in the same direction section 9 reports: fat domains high
(disk 2.82, square 2.81), elongated low (rect 5x1 2.22), against the note's "fat 2.44-2.58,
sharp or elongated 2.24-2.31". Two independent measurements finding the same weak aspect
dependence is worth more than either.

### The `A/P` scaling is confirmed exactly, by section 4's own test

The same 3:1 rectangle at three sizes, which is the test that separates `A/P` from area:

    scale        0.5      1.0      2.0
    A/P       0.1875   0.3750   0.7500
    C_k       0.4613   0.9225   1.8448      exactly proportional to the length scale
    C_k/(A/P) 2.4600   2.4600   2.4598      identical to four significant figures

`C_k/area` over those three reads 2.46, 1.23, 0.61. So section 4's dimensional argument is not
merely tidy -- it is measured twice, in two codebases, with different normalizations.

**Section 4's stronger claim also holds**: per-solve calibration of the SLOPE is not necessary.
A geometric prefactor plus a safety factor does it, and the calibration budget goes to
`sigma_base`, which is what genuinely varies.

### Fix (1) is INERT in `chladni`, and this is the one real divergence

Section 5 prices two fixes and section 7 measures fix (1) -- raise `sigma_base` by loosening
the truncation -- moving `sigma_base` from 2.5e-3 to 1.2e-1, a factor of 48. On a `chladni`
rectangle pencil the same lever does nothing:

    rtol      1e-15    1e-11    1e-07    1e-03
    rank@50      47       39       29       19
    sigma_base 0.3329   0.3345   0.3346   0.3346

**A range of 0.5% while the retained rank falls by a factor of 2.5.** Two candidate reasons,
neither settled:

* `chladni` thresholds RELATIVE to the largest singular value, so raising the cutoff removes
  columns proportionally and the plateau -- a ratio of norms -- stays where it is, whereas an
  absolute cutoff bites at a fixed level.
* `chladni`'s plateau is already healthy across domains, 0.12 to 0.36 over seven, where
  `lappy`'s varied two orders. The pathology fix (1) was correcting may simply not be present.

So the shipped default is **the check, not the fix** -- which is what section 5 argues anyway
("for a black-box solver aiming at reliability over speed, (2) is the safer default: it is a
guarantee, not a tuning gamble") and section 8 argues more strongly ("the check is arguably
more valuable than the fix"). `calibrate_rtol` is implemented and exported, with the inertness
pinned by a test so it is not rediscovered as a promising idea.

### The check discriminates, and on the right domains

Measured over `[20, 120]`, `sigma_base` from 24 probes, reporting the scan points the criterion
demands:

    domain          A/P     sigma_base   required dk   scan points
    eq_tri        0.1443      0.412         2.156            4
    rect 1x1      0.2500      0.292         0.882            8
    L_shape       0.3750      0.231         0.466           14
    disk          0.5000      0.258         0.390           17
    iso_tri h=1   0.2071      0.172         0.628           11
    iso_tri h=4   0.3904      0.054         0.105           62

The sharp, elongated triangle -- the shape class that broke the optimizer in section 1 -- needs
**8x the scan density of a square**, and says so from geometry and one plateau measurement,
before any mode is located. On a deliberately starved configuration the check reports
`UNFINDABLE (5.3x short) ... needs dk <= 9.43e-02`, which is the actionable form section 8 asks
for.

### Two hazards from section 6, resolved

**No `rtol` -> basis coupling exists in `chladni`.** `lappy`'s `basis_plan._indep_digits` set the
dynamic range from `rtol`, so calibrating the threshold moved the basis underneath the
calibration. `_rank_from` and `choose_rank` are the only consumers here and `PlaneWave.n_useful`
takes `kR`, not `rtol`. Section 6 called that "a simplification worth keeping"; it is now pinned
by `test_rtol_does_NOT_size_the_basis`.

**The absolute cut is recorded per step**, so the relative-vs-absolute drift across a band can
be measured rather than argued. It has not been measured yet -- with the lever inert there is
nothing yet to choose between.

### One thing section 6 did not anticipate: the floor is PREDICTABLE from `rtol`

Betcke's relation -- for an absolute cutoff, the tension floor inflates in proportion to `rtol`
-- holds in the relative convention too. Measured on `rect(1, 1/phi)` over `[5, 120]`, worst
tension at a true eigenvalue:

    rtol      1e-16    1e-15    1e-13    1e-11    1e-09    1e-08
    eps      7.5e-16  1.2e-15  2.9e-14  2.3e-11  1.2e-09  1.1e-08
    eps/rtol   7.52     1.17     0.29     2.26     1.21     1.10

A log-log slope of 1.13 with `eps/rtol` settling near 1.15, i.e.

    eps  ~  max(eps_basis, rtol)

with `eps_basis = 7.5e-16` on this pencil. That is a predictive rule rather than a table, and it
does three things section 6 wanted and could not get:

* it says **`rtol` is safe to raise up to `prec`** -- the induced floor is about `rtol`, so at
  `rtol = prec` the discarded directions contribute at the precision asked for;
* it says where the derived default belongs. `sqrt(prec * eps_mach)` is the CROSSOVER of the
  two terms: below it the truncation contributes less than the basis already does and buys
  nothing, above it the floor inflates linearly. The rule landing on the crossover is why it is
  the right default and not merely a convenient one;
* it gives the basis planner a contract it can publish -- the largest usable `rtol` follows from
  the adequacy requirement instead of from a sweep.

`tests/test_floor.py` pins the relation.
