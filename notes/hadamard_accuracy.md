# How accurate are the Hadamard eigenvalue derivatives?  (2026-11-03)

The question: how accurate is the vertex-coordinate gradient `d lam / d(x_k, y_k)` that douse
consumes (`douse.chladni_backend.ChladniSolver._velocity` + `SolveRecord.hadamard`), relative
to the eigenvalue itself?

**Answer: on corner domains the HADAMARD QUADRATURE RULE is the limit, not the basis.** The
`'integer'` rule (`integral_quad(..., family="integer")`) reports a model shortfall of 5e-8 to
9e-6 on every shape here with a ruled corner, at every `prec`, and
`MPSEigensolver.hadamard` never reads it. On an exact reference rule, the SAME eigenfunction
gives gradients at about the eigenvalue's own accuracy, and the identities hold to roundoff.

**Status (end of 2026-11-03): fixed** on branch `boundary-integrals-accuracy`. The graded corner rule
in corner-local frames now serves the Hadamard, Gram and certification integrals; see "The Gram
pipeline" and "Fix built" below.

## Errors measured at douse's operating point (prec = 1e-10, 10 levels)

These are the gradient's relative errors (max over the 2N partials / max |partial|) against an
exact reference rule on the same eigenfunction (`hadamard_exact_rule.py`).

| shape | eigenvalue rel. bound | shipped rule | reference rule (identities) | limit |
|---|---|---|---|---|
| rectangle 1 x 0.7 (closed form, 12 levels) | 1e-12 .. 1e-14 | 1-2x the eigenvalue error | -- | none: no ruled corner |
| irregular hexagon | 1e-12 .. 5e-11 | 5e-13 .. **4.6e-10** | 5e-13 .. 1e-11 | the rule, 10-100x |
| irregular octagon | 6e-13 .. 1e-11 | 1.5e-12 .. **4.5e-11** | 4e-13 .. 5e-12 | the rule, ~10x |
| L-shape (prec 1e-12) | 5e-15 .. 4e-14 | 4e-13 .. **1.2e-7** | 1e-15 .. 2e-14 | the rule, up to 1e7x |
| flat triangle (height 0.15) | 2e-11 .. 1e-9 | 8e-11 .. **8e-8** | 2e-11 .. 3e-10 | the rule, up to 1000x |
| triangle (0, 1, 0.3+0.8i) | 2e-13 .. 7e-12 | 2e-11 .. 9e-10 | 7e-12 .. 4e-10 | the eigenfunction (rule ~2x) |

The worst partial on the L-shape is always the re-entrant vertex's own, where `V.n = O(1)`
meets `(d_n u)^2 ~ r^(-2/3)`. Rotation and dilation about that vertex (`V.n = O(r)` or 0 there)
hold to 1e-15 even on the shipped rule.

## How the cause was isolated

1. **Closed form** (`hadamard_accuracy.py` A): on the rectangle every vertex partial is exact
   (`-/+ pi^2 m^2 / L^3`, `-/+ pi^2 n^2 / H^3`); the gradient error is 1-2x the eigenvalue
   error down to 1e-15. Right angles need no corner rule.
2. **Identities** (B): translation, rotation and dilation residuals on six shapes and three
   `prec` values. On every corner domain they sit at a floor that does not move with `prec`
   (hexagon: identical digits at 1e-10 and 1e-12).
3. **Finite differences** (C, `hadamard_accuracy_C.txt`): Richardson-extrapolated central
   differences on a frozen plan agree with the identities on the L-shape: `|H - FD|` = 9e-11,
   1.4e-9, 1.1e-8 for levels 1-3, with the FD settled to 2e-10 or better.
4. **Rounding, the user's two candidates** (`hadamard_rounding_mp.py`, `hadamard_error_sources.py`):
   - (1) per-term `d_n psi_j` accuracy: the Bessel factors recomputed in mpmath (34 digits)
     and `u_n` summed in extended precision on the same coefficients and nodes. Worst column
     entry 5e-11 relative, `u_n` 2.5e-14, **gradient changed by 1.2e-14 to 1.7e-14**. The
     identity residuals are unchanged (1.5e-8, 2.3e-8).
   - (2) cancellation: `kappa = sum |c_j d_n psi_j| / |u_n|` peaks at 3e6 at isolated nodes,
     but weighted by the integrand's mass it is 1.2-1.4. The implied rounding bound on the
     gradient is 5e-16.
   - Neither shows up on the L-shape. Both were measured there only: the hexagon's 1e-12 plan
     is rung 3 (lightning sources), which the mpmath script does not handle, and
     `hadamard_error_sources.py` stops at the hexagon on a level-matching bug. Neither is
     needed, given item 6.
5. **Richer bases, the approximation candidate** (`hadamard_error_sources.txt`): Fourier-Bessel
   x1.5 and x2, and rungs 1 and 2 (sources), at the same prec. All five bases give the
   L-shape **the same 1.5e-8 and 2.3e-8 to two digits**. Every basis converges to the same
   eigenfunction, so the error is in what they share.
6. **The rule** (`hadamard_exact_rule.py`):
   - L-shape: at `nu = 2/3` every exponent of `(d_n u)^2 V` near the vertex is a multiple of
     1/3, so `tau = s^3` with Gauss-Legendre in `s` is exact. Settled to 1e-15; translation
     drops from 1.5e-8 to 3e-15, on all levels.
   - Convex shapes: an hp-graded rule (geometric panels at ratio 0.15 toward every corner,
     24-point Gauss each) settles to 1e-12 (flat triangle 1e-10). It gives the table above.
   - Building the reference needs care: nodes must be formed FROM THE CORNER END. A first
     version formed `p0 + tau (p1 - p0)` with `tau = 1 - O(1e-12)` and was wrong by 1e-7 on two
     levels (`rules.NODE_FLOOR`'s warning, rediscovered).

## Why the shipped rule stalls, and why the earlier check missed it

`hadamard` forces the INTERPOLATORY corner rule over the curved exponent set
`{gamma + j nu + m}` (V's integer powers on top of the corner family). That least-squares fit
is ill-conditioned and bottoms out long before `prec`. `integral_prec` asks for 1e-11 at
prec 1e-10, and the rule achieves:

| shape (lam_max) | Gram rule (`'even'`) | Hadamard rule (`'integer'`) |
|---|---|---|
| L-shape (80) | 1e-13 | 8.7e-6 |
| irregular hexagon (80) | 6.1e-10 | 1.4e-7 |
| irregular octagon (80) | 1.5e-10 | 4.7e-8 |
| flat triangle (2800) | 1.1e-8 | 6.0e-6 |
| triangle (80) | 4.5e-10 | 1.2e-7 |

`corner_order_for_prec` returns the argmin and its achieved model error, and `integral_quad`
records it in `.shortfalls` / `.sizing_prec`. `MPSEigensolver.hadamard` takes `.quad` and
drops both. The model bounds are pessimistic (the L-shape's 3e-7 model bound against an
observed 1.2e-7 worst partial), but they flag exactly the shapes that are affected.

An earlier step of this study refined the Hadamard `prec` (1e-12 -> 1e-14), saw no digit
move, and concluded "quadrature is never limiting". That was wrong: the rule CANNOT refine,
because its order selection is already at the argmin.

## A second, separate defect: unchecked normalization

On the flat triangle at prec 1e-12, level 8 (`lam` 2736.56; planner "no rung met"):
- `eigenfunctions().gram_residual` = 0.16, so the norm is off by 16%;
- `orthonormal` is still True;
- `hadamard` used it: dilation identity off by 0.43, translation (blind to scale) still 1e-7.

Evaluating the same coefficients on a finer `'even'` rule made it worse, so the expansion
itself is unreliable at the added nodes there (a planner-shortfall shape, `[next: default]`
sharp-triangle item). `hadamard` should refuse or flag when `gram_residual >> prec`.

## Consequences for douse

At prec 1e-10 on generic convex N-gons the gradient is good to ~1e-12 .. 5e-10 relative. That
is far below every optimizer threshold in use (criticality 1e-7 .. 8e-4, Gauss-Newton
steps), so no thesis result depends on this. Shapes that make it matter:
- a re-entrant vertex (1e-7);
- flat or sharp triangles at high levels (1e-8 .. 1e-7);
- any claim that the gradient carries the eigenvalue's digits.

## Fix directions (as first written; superseded -- see "Fix built" at the end)

- Hadamard rule: geometric (hp) grading toward each ruled corner, which needs no exponent fit
  and has positive weights. The reference rule here is that, untuned (~900 nodes per segment;
  a production rule needs far fewer). For rational `nu = p/q` (re-entrant 270 deg, 2/3), the
  substitution `tau = s^q` is exact and cheap.
- Alternatively, split `V = V(corner) + (V - V(corner))` and integrate each part on a rule
  for its own exponent set, avoiding the curved-set fit.
- Surface `iq.shortfalls` / `sizing_prec` and `gram_residual` from `hadamard` (a warning, or a
  per-call attained precision like `Spectrum.bound`).

## Files

`hadamard_accuracy.py` (A/B/C), `hadamard_accuracy_{A,B}.json`, `hadamard_accuracy_C.txt`,
`hadamard_error_sources.py` + `.txt`, `hadamard_rounding_mp.py` + `.json`,
`hadamard_exact_rule.py` + `.json`. Run from the repo root with `PYTHONPATH=src`. The bootcamp
venv (chladni + mpmath) was used.

## Addendum, same day: designing the replacement rule

Proposed rule for each corner panel of the Hadamard rule:
- substitute `tau = s^p` with `p = max(1, 1/(2 nu - 1))`, which makes the leading
  `(d_n u)^2 ~ r^gamma` term constant in `s`;
- then grade geometrically in `s` toward the corner, with plain Gauss-Legendre panels (positive
  weights, no fit);
- then evaluate the substitution's Jacobian at each node's ACTUAL distance from the corner.

Model and solve checks:

- **Rational denominators do not matter** (`hadamard_rule_denominators.py`, model integrands).
  `p` is a continuous function of `nu`, and nothing is made exact by rationality. ν = 7/10,
  71/101 and 0.7 + 1.4e-3 give the same errors at every node count (1e-11 / 1e-13 / 2e-15);
  so do 4/7, 53/93 and an irrational neighbour. On a CONVEX corner (`p = 1`, grading only),
  126-176 nodes per corner end reach 1e-13 .. 1e-16 for every ν tried, 1.01 to 11.8.
- **Re-entrant corners and floating point** (`hadamard_rule_translated.py`, real eigenfunctions).
  The `s^3` reference that reached 1e-15 on the L-shape did so partly because the re-entrant
  vertex sits AT THE ORIGIN. Translated by 0.7+0.4i or 3+2i, the same rule gives 1e-9 .. 3e-7,
  worse as nodes approach the vertex. The cause: a node at distance r is stored to
  eps |z_corner| absolutely, its true distance differs from the planned one by up to
  eps |z_c| / r relatively, and the integrand is r^(-2/3) there. Evaluating the Jacobian
  `p s^(p-1)` at `s = (|z_node - z_corner| / h)^(1/p)`, the rounded node's true distance, which
  is computed EXACTLY because the subtraction is Sterbenz-exact, restores 2e-15 .. 3e-14 for
  both shifts and both node densities.
- **Remaining limit: corners past about 290 deg** (ν below ~0.6, p above ~4.5). Even one Gauss
  panel in `s` puts nodes closer than eps |z_c| to the vertex, where they round onto it, and a
  non-negligible share of the integral lives there (r^(2 nu - 1): 1e-2 of it within 1e-13 at
  ν = 4/7). Full accuracy there needs the corner's own Fourier-Bessel family evaluated in
  corner-local (r, theta), a sampling-API change. douse's polygons never have such corners.
- **The shipped L-shape numbers above are origin-favoured too**: in general position the
  shipped rule may be somewhat worse than tabulated.

## Addendum 2: per-corner local frames (`hadamard_rule_local_frames.py`, `.txt`)

Each node on an edge touching the corner is evaluated by `basis.translated(-z_c).at(d e)`, with
the SAME coefficient vector, so its distance from the vertex is exact. Every other node stays
in absolute coordinates. Results at prec 1e-10, worst identity residual:

| case | absolute coordinates | local frame |
|---|---|---|
| L-shape, at the origin | 2e-14 .. 5e-13 | same |
| L-shape, shifted 3+2i | 4e-10 .. 9e-8 | 2e-15 .. 3e-13 |
| 315 deg notch (nu = 4/7), at the origin | 5e-14 .. 4e-13 | same |
| 315 deg notch, shifted 3+2i | fails: nodes round onto the vertex | 9e-15 .. 6e-13 |

The notch's nodes come within 1e-28 of the vertex. With local frames, the true-distance
Jacobian of addendum 1 is unnecessary, and so is the 290-degree limit.

`translated` moves each column's function exactly for `FourierBessel`, the lightning and
offset `Fundamental` sources (2e-15 column-wise), so the coefficients carry over. `PlaneWave`
is the exception: `translated` returns `self`, the same span but not the same columns. Local
frames on a plan with plane waves (rung 4) need it to carry an origin, i.e. a phase.

Translating the WHOLE domain instead puts only one corner at the origin and perturbs the solve
(plan, collocation) at the ulp level. Per-corner frames change only the integrals.

## The Gram pipeline (`gram_accuracy.py`, `gram_accuracy_B.json`, `gram_accuracy_CA.txt`)

`eigenfunctions()` normalizes with the Rellich Gram on the `'even'` rule, and every Hadamard
entry scales as `1/||u||^2`, so the same corner weakness could hide there. Each check below has
ground truth that uses no chladni boundary rule.

**B. Sector apex norms in closed form**, with the apex at the origin and shifted by 3+2i. Each
value is the worst over three Rellich origins.

| apex angle | interpolatory, origin | interpolatory, shifted | graded + local frames (both) |
|---|---|---|---|
| 1.25 pi | 1e-14 | 1e-14 | 1e-14 |
| 1.5 pi | 2e-14 | 2e-14 | 2e-14 |
| 1.6 pi | 8.7e-12 | 6.4e-12 | 1.9e-14 |
| 1.75 pi | 1.9e-11 | **2.5e-10** | 1.3e-15 |
| 1.9 pi | 3.9e-9 | **1.0e-8** | 4e-15 |
| 1.95 pi | demoted (nu within `NU_MARGIN`): 0.15 | same | same |

- With the Rellich origin AT the apex, the old rule reads about 1e-15 even where it is wrong
  elsewhere. The default origin is the smallest-`mu` corner, so the one-origin
  `gram_residual` misses the error.
- The model bound (3e-8 .. 7e-7) was honest.
- This is the user's intuition confirmed: the local-frame fix applies to the Gram pipeline as
  well.

**A. Interior Gram of returned eigenfunctions** against tensor or fan interior cubature, with
the cubature's own error at 2e-16 .. 1e-15. Results:
- Rectangle (at the origin and shifted), square double, and L-shape at the origin: true
  `|G - I|` is at most 6e-13 at prec 1e-10, and at most 2e-13 at 1e-12.
- `gram_residual` and the four-origin max sit 10-1000x BELOW the true error there. They are
  detectors, as documented, not error bars.
- The shifted L-shape needs `Polygon`; `polyomino(...).translated` is a `Domain`, which the
  planner refuses.
- The triangle had no solver: "no rung met".
- Flat triangle at prec 1e-10: true error 7e-11 .. 5e-10, with `gram_residual` within 2-5x of
  it.

**C. The flat triangle's 0.16 was the fitted rule** (level 8, `lam` 2736.56, prec 1e-12;
`flat2.py`/`flat3.py` in the session scratchpad, reproduced inline here):
- The eigenfunction is sound:
  - its interior norm agrees between two independent cubatures to 7e-12 (5.104e-4);
  - |u| on 20 000 boundary points is at most 1e-8;
  - it satisfies Helmholtz;
  - analytic `d_n u` matches finite differences to 7e-6, the FD limit, so the basis gradients
    are not the cause either.
- Yet the Rellich norm swung from 3.58e-4 to 4.75e-4 with the origin, which is impossible for an
  eigenfunction unless `sum_s n_s int_s (d_n u)^2 != 0`.
- The integral of `(d_n u)^2` along the long base (both acute corners, `nu` 9.8 and 11.8) is
  **13.06 on the fitted rule and 18.62 on a dense graded reference**: 30% wrong at a claimed
  1e-8.
- The graded rule gives 18.62 to 9e-11, and `sum_s n_s I_s` vanishes to 1e-10. Level 0 on the
  same rule was fine.
- So a model bound that is "honest" on its model integrand can still miss badly on high-`nu`,
  high-`k` corners. The earlier reading of this case, "the expansion is unreliable at the added
  nodes", was wrong.

## Fix built (branch `boundary-integrals-accuracy`, 2026-11-03)

- Column-exact `translated`, with `PlaneWave` given a phase `origin`.
- `Segment.offset` and corner-anchored `Panel`s.
- `rules.graded_corner_rule` and its sizing, which reaches 1e-14 for `nu` in 0.53 .. 11.8 and
  `k*L` up to 40, at 24 .. 600 nodes per corner panel.
- `cauchy_data` evaluates anchored nodes as `family.translated(-corner).at(offset)`.
- `integral_quad(corners="graded")` is now the default, used by `hadamard`, `eigenfunctions` and
  certification alike.
- `gram_residual` is now the worst over 3 extra origins.

Identities on the graded rule, worst over 6 levels relative to `lam`. Translation and rotation
should give 0; dilation should give `-2 lam`.

| shape | prec 1e-10 | prec 1e-12 |
|---|---|---|
| L-shape at the origin | 3.4e-13 | 3.7e-14 |
| L-shape shifted 3+2i | 8.5e-12 | 8.7e-14 |
| 315 deg notch shifted 3+2i | 2.5e-11 | 9.2e-14 |
| irregular hexagon | 1.4e-10 | 8.7e-13 |

The fitted rule read 2.8e-9 on the L-shape translation identity at every prec.

Certification was switched only after old-vs-new deterministic benchmarks (two worktrees at the
same commit):
- **`bound_coverage.py`**, 717 levels: identical `lam`, `mp_covers` 717/717 and `met` 448/448
  both ways; `mp_bound` ratio 0.89 .. 1.1.
- **`stage_policy_step4_douse.py 1e-10`**: identical tallies, flags and rungs in every mode; only
  `bound` values differ, by at most 1.8e-5 relative.
- **`count_audit.py`** re-reads stored solves and never certifies, so it cannot change.

## Out of sample: the hardest douse ratio shapes (`hadamard_hard_shapes.py`, `.sbatch`)

Ten final iterates of `thesis_ratio_v1/v2` runs on ARC (chladni e1ebaf8): eight whose trust
radius collapsed to ~6e-15, the noise-limited signature, and two plateau controls. Mostly k = 4..5,
re-entrant corners up to 228 deg, edges down to 0.05. Old against new, identities per level and
vertex gradients against a Richardson FD reference; a 50-task array, run on ARC because the FD
references take hours locally. Local smoke result, `v1/N5_k4/task_0` at prec 1e-10: worst
identity error 8.0e-10 relative (old) against 2.2e-10 (new). Results: pending.

## Does H's error track the eigenfunction's Moler-Payne eps? (2026-11-03)

`HadamardReport.attained` bounded the rule and the normalization only. `u_n` from an MPS
eigenfunction is wrong to first order in its boundary residual, whose relative size is the
Moler-Payne `eps`, so the accuracy audit (1.9) asked whether `attained` should include it.
`notes/hadamard_vs_mp_eps.py` measures H's true error through the exact identities (translation
and rotation: H = 0; dilation: eig H = -2 lam, all relative to lam). That is 40 levels: the three
corner-domain `IDENTITY_CASES` at 1e-10 and 1e-12, plus three hard douse shapes at 1e-10. The
315-degree notch at 1e-12 does not solve, because no source offset keeps the fundamental
sources outside it. That is an existing basis limit, recorded as a row.

| claim | levels under-reported | worst true error / claim |
|---|---|---|
| `attained` (rule, norm, gram) | 23 / 40 | 8.0 |
| `max(attained, mp_eps)` | 14 / 40 | 4.9 |
| `attained + mp_eps` | 6 / 40 | 2.6 |
| `2 * max(attained, mp_eps)` | 3 / 40 | 2.4 |

`mp_eps` tracks the error in part. On the hard shapes it is the dominant term, at 1e-11 to
3.5e-10 against the rule's 1e-11. Even the sum is not a strict bound. **Decision.** `attained`
is now `max(rule, norm, gram) + mp_eps`, documented as an ESTIMATE with these numbers, and
`HadamardReport.mp_eps` reports the term (`eigensolver.Level.mp_eps` carries it from the
certification). On the hard douse shapes, the true error exceeds 1e-10 on 7 of 15 levels. Those
Hadamard matrices are not good to the solve's prec, and the warning now says so.

**Update, 2026-11-04 (ARC, 850 levels on the 51 douse shapes).** This sample was too small and
too easy. On the douse shapes `attained + mp_eps` under-reports on 488 of 850 levels: worst 33.6x,
median 2.0x at 1e-10. At 1e-10, 91 levels have an H error above prec while `attained` says met.
See `accuracy_audit_2026-11-03.md`, "ARC accuracy studies".

## A gradient reference (2026-11-05, `hadamard_hard_shapes.py` FDref, ARC)

The old FD column stepped 4e-3, the size of the shortest edges. `FDref` plans once at the base
shape and solves every perturbed shape on that plan (no rung switch between steps), over steps
1e-3, 3e-4, 1e-4 at +-h and +-2h, Richardson-combined. Its bar is the levels' certified noise
plus the truncation measured on cluster SUMS between consecutive steps. On the 10 hard ratio
shapes x 4 directions (2 random, x and y of the most re-entrant vertex) the bar is 2e-10 .. 1e-6
of lam (the unmet-solve cases are the wide ones). Every SIMPLE level agrees with H at 1e-12 and at
1e-10 within the bar. Every large disagreement is a CLUSTER:

| shape | cluster | spread | H(1e-10) - FD | H(1e-12) - FD |
|---|---|---|---|---|
| v2/N6_k5/task_0 | triple, lam 110.961 | 1.6e-10 | 7.8e-4 | 1.7e-5 |
| v2/N8_k2/task_6 | pair, lam 64.924 | 6.6e-10 | 6.5e-5 | 4.5e-7 |
| v2/N6_k4/task_0 | pair, lam 125.805 | 9.4e-10 | 1.1e-5 | 6.8e-7 |

(relative to lam, the cluster's summed gradient), with the identities at 1e-10 .. 1e-12. These
are ratio optima: what douse optimizes.

## Clusters (2026-11-05)

**What goes wrong.** Levels are grouped by value. A triple spread 1.4e-10 at prec 1e-10 or 1e-12
comes back as 1+1+1 (or 2+1), each level's eigenfunction the near-null vector at its own lam.
Davis-Kahan: an eigenvalue error eps moves an eigenvector by eps / gap, so between "values
resolved" and "eigenfunctions resolved" there is a band where each level's vector carries a
piece of its neighbour's. The per-level vectors then overlap (cross-level Gram 1e-4), and a
Hadamard block stacked from them is wrong at that size: its trace is the trace over a
non-orthonormal basis. Orthonormalizing the members TOGETHER fixes it: on the triple, all of
(a) one joint null space at the mean lam, (b) joint Löwdin of the per-level vectors, (c)
projection into (a) then Löwdin, agree with FD within its 2e-10 bar.

**The Gram across different lam.** Between Dirichlet eigenfunctions at lam != mu, the boundary
Rellich form is `(lam + mu) <u, v> + (lam - mu)/2 int (v r.grad u - u r.grad v)`: a bias
proportional to the gap, FLAT in prec and AFFINE in the Rellich origin; real overlap shrinks with
prec and is origin-free. Translation alone keeps `(lam - mu) int v a.grad u`, which is also why
`identity_residual` is read only inside each level's own block (a block of distinct levels read
0.41 before). Two traps found on the way: evaluating a level's coefficients at a neighbour's lam
(the basis moves with k) reads 1e-4 .. 2e-2 on well-separated pairs, so every column is evaluated
at its own lam; and one joint null space at the mean lam fails once the spread is well above prec
(identity ~ spread, orthonormality 1e-10 on T2.7-lowest).

**ARC overlap study** (`accuracy_studies.py` overlap, 51 shapes, 43 adjacent pairs within 1e-2,
at 1e-8, 1e-10, 1e-12). Entangled pairs (overlap > prec, origin spread 0 .. 30%) and biased ones
(flat in prec, origin spread 39 .. 100%, bias up to 2e-2 x gap) separate. Entangled at 1e-12:
8 of the 10 hard ratio shapes (gaps 1e-10 .. 9e-10, overlap 2e-6 .. 1.4e-5), T1.2 at lam 68.85
(gap 1.3e-11, overlap 1.6e-4), T2.9-N4opt (1.1e-10, 9e-6), T2.6-N3k5 (8e-9), T2.6-N3k2 (2.6e-8),
T2.6-N5k4-L7-ref3 (1.2e-7), T2.7-lowest and -median (1.5e-6), T2.6-N5k4-L7-ref0 (1.2e-5), T2.2
(3.5e-5). Tightening prec does not resolve them: the overlap shrinks, but stays above prec.

**The trigger.** Moler-Payne's eigenfunction bound (Betcke-Trefethen 2005, Thm 8.1, eq. 8.2):
`||u - u_k|| <= (eps/alpha) sqrt(1 + eps^2/alpha^2)`, alpha the relative gap to the nearest other
eigenvalue. As a screen, `eps / alpha > prec` caught EVERY entangled pair at all three precs, and
also 26-28 of 43 that were not (gaps 1e-4 .. 1e-2, where it is loose). So the solver screens by it
and LINKS a screened pair only when the measured cross-Gram exceeds prec and moves by less than
`CLUSTER_ORIGIN_SPREAD = 1/3` over four Rellich origins (`eigenproblem.link_clusters`). The
borderline: T2.6-N3k2 read 30% at 1e-10 (0% at 1e-8 and 1e-12), against the biased pairs' 39%.

**Built** (`eigensolver.Cluster`, `Spectrum.clusters`, `Level.cluster`, `Level.gap`;
`SolveRecord.eigenfunctions`, `.cluster_eigenfunctions`; `MPSEigensolver.joint_eigenfunctions`,
`.cross_gram`; `hadamard(..., groups=)`). A purified member is the raw columns at their own lam
plus a mixing matrix (`ClusterEigenfunctions`); `Lam = mix^T sym(G diag(lams)) mix`.
`SolveRecord.hadamard` refuses a list that holds part of a cluster. Test: the 1 x (1 + 1e-9)
rectangle, overlap 4.3e-7 at 1e-10, whose block's eigenvalues have a closed form
(`tests/test_clusters.py`). What a cluster does NOT promise: which member is which. A quantity
built on one member carries an error of about `Cluster.overlap`.

## Calibration (2026-11-05, `accuracy_studies.py` grad, ARC)

Vertex-hat-field gradients at prec 1e-10 against 1e-12 on all 51 shapes (the 1e-12 reference
used only where its own identities are at most max(1e-11, 0.1 x the test level's); FDref
confirms it on the hard shapes). Rule set in advance: adopt `k * estimate` if some k <= 3 covers
99% of levels with a worst ratio <= 10. On 327 simple levels, error / estimate:

| estimate | median | p99 | worst | silent / 178 short |
|---|---|---|---|---|
| `mp_eps` | 4.7 | 1823 | 2526 | 164 |
| eps / alpha (Moler-Payne) | 0.29 | 182 | 893 | 42 |
| identity residual | 0.88 | 33 | 113 | 7 |
| old `attained` | 1.40 | 12 | 34 | 147 |
| **max(identity, eps / alpha)** | **0.29** | **1.41** | **2.77** | **2** |

The identities catch well-separated levels whose `u_n` is off on the boundary (high-lam
triangles, 5e-8, where eps/alpha reads 3e-10); eps/alpha catches near-pairs (5e-6 .. 8e-4 where
the identities read 3e-10). The combination passes the rule at k = 3 (100% covered).
`HadamardReport.attained` is now `max(rules + mp_eps, 3 max(identity, eps/alpha))`
(`HADAMARD_ESTIMATE_K`), with `eigenfunction_bound` reported. Cost: about 90 of 327 levels warn
without being short. Not covered: exact doubles of symmetric shapes (the regular hexagon's split
error, 4.5e-9 at 1e-10 against a trace error of 4e-14, is inside one level, where alpha is the
gap to OTHER levels).
