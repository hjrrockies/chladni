# How accurate are the Hadamard eigenvalue derivatives?  (2026-11-03)

The question: how accurate is the vertex-coordinate gradient `d lam / d(x_k, y_k)` that douse
consumes (`douse.chladni_backend.ChladniSolver._velocity` + `SolveRecord.hadamard`), relative
to the eigenvalue itself?

**Answer: on corner domains the HADAMARD QUADRATURE RULE is the limit, not the basis.** The
`'integer'` rule (`integral_quad(..., family="integer")`) reports a model shortfall of 5e-8 to
9e-6 on every shape here with a ruled corner, at every `prec`, and
`MPSEigensolver.hadamard` never reads it. On an exact reference rule, the SAME eigenfunction
gives gradients at about the eigenvalue's own accuracy, and the identities hold to roundoff.

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

## Fix directions (not built)

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
