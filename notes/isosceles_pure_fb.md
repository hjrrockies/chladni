# Pure Fourier-Bessel on isosceles triangles

`notes/isosceles_pure_fb.py`: apex `omega` at the origin, unit legs, Dirichlet, a flat 100 FB terms
per corner (planner bypassed), default collocation (3x boundary, 1x interior), `prec = 1e-14`,
`lam_1` only. Measures: the Moler-Payne bound and the tension at the minimum.

## Pre-registered guesses (2026-11-07, written before the sweep was read)

Already known from the smoke test, so not blind: both `omega = 0.05` and `pi - 0.05` fail outright
(the solver declines, Moler-Payne refuses). At 0.05 the tension is flat near 1e-16 over the
whole window (spurious, numerical rank 188/300). At `pi - 0.05` it is flat near 1.5e-4 (no
well). Equilateral and right isosceles: bound about 1e-14.

Order counting uses `plan.corner_order`'s rule: degrees up to `nu_max = kR + t(prec) (kR)^(1/3)`,
with `t(1e-4) = 4.6` and `t(1e-14) = 10.5`.

**`omega -> pi`: the apex runs out of degrees.** Let `eps = (pi - omega)/2`. Then the base angles
are `eps`, the height is about `eps`, and `k` is about `pi/eps`. The apex is close to a straight
angle (`alpha ~ 1`), and it has to reach the base corners at `R = 1`, so it needs about `pi/eps`
degrees. The base corners need only a few. 100 apex degrees run out at:
- `1e-14`: `pi - omega ~ 0.11` (`omega/pi ~ 0.965`)
- `1e-4`: `pi - omega ~ 0.08` (`omega/pi ~ 0.975`)
- outright failure (no well): `pi - omega <~ 0.06` (`omega/pi >~ 0.98`)

Expected shape: a bound at about prec up to about 0.96, then a steep, smooth climb, roughly
exponential in the degree shortfall. The tension at the minimum rises with it. Medium confidence;
`k` runs somewhat above `pi/eps`, which pushes every threshold slightly toward the middle.

**`omega -> 0`: not degree starvation, so a different mechanism.** The apex needs 2-3 degrees
(`alpha = pi/omega`), and the base corners (`alpha ~ 2`) need about `k/2 ~ pi/(2 omega)`. Order
counting therefore says 100 per corner suffices down to `omega ~ 0.02`, and the smoke test at
0.05 already contradicts that. Guess: collocation starvation plus redundancy. The 97 surplus apex
columns are, after scaling, monomials `r^nu sin(nu theta)` living within `1/nu` of the base. The
base is only about `omega` long and carries roughly `450 omega` boundary points, too few to
resolve 100 apex angular modes. Combinations then vanish at the collocation points without
vanishing in between, which gives a flat, spurious near-zero tension.
- onset of outright failure: `omega ~ 0.2-0.3` (`omega/pi ~ 0.07-0.1`)
- `1e-4` limit: about the same, `omega ~ 0.3`, because the failure should be ABRUPT (a spurious
  floor appears) rather than a gradual climb

Low confidence in the numbers. The mechanism is the real bet, and it is testable at `omega = 0.05`:
cut the apex to about 3 terms, or raise the boundary ratio. Either should bring the well back if
the guess is right.

## Results

The first sweep was stopped 2026-11-07 before reading: the small-apex failure it was heading for
was collocation, not basis (`notes/collocation_allocation.md`: on the length split the base got 24
points at apex 0.05 and 83 at 0.2 for 100 apex columns; at 200, both locate lam_1 with a
Moler-Payne bound near 1e-6). The rerun uses the basis-aware split, which `Plan.solver` now applies
by default. The pre-registered guess for `omega -> 0` named this mechanism; its NUMBERS (onset
0.2-0.3) were about the old split and no longer apply. The `omega -> pi` guess is untouched.

### The rerun (2026-11-07, branch `basis-aware-collocation`)

71 angles, log-spaced toward both ends (`notes/isosceles_pure_fb.json`; figures
`isosceles_pure_fb.png`, `_ends.png`, `_scans.png`). Read from the SOLVER column (`search_interval`'s
lowest level). The DIRECT golden-section reading sometimes settles in a higher well: at 0.01155 it
gives 113850 against the solver's 80703, and lam_1 must fall as omega grows.

| omega | omega/pi | Moler-Payne / lam_1 | tension at min |
|---|---|---|---|
| 0.010 - 0.032 | 0.003 - 0.01 | 2.9e-7 - 8.5e-7 | 4e-8 - 2e-7 |
| 0.037 - 0.43 | 0.01 - 0.14 | 1e-6 - 2.5e-6 (plateau) | ~3e-7 |
| 0.49 - 1.36 | 0.16 - 0.43 | 1e-6 -> 2e-9 | falling |
| 1.78 - 2.26 | 0.57 - 0.72 | 6e-12 - 2.4e-14 | 1e-14 - 1e-15 |
| 2.48 -> 2.71 | 0.79 -> 0.86 | 1.9e-9 -> 1.7e-5 -> 2.0e-3 | rising smoothly |
| >= 2.77 | >= 0.88 | refused (eps >= 1) or declined | 1e-6 - 5e-4 |
| >= 3.128 | >= 0.9957 | ValueError (inf/NaN in the basis) | -- |

**omega -> 0: no failure down to 0.01.** Before the fix this end failed outright by 0.05. Now the
bound sits on a ~1e-6 plateau with no spurious floor (the median tension stays at 1e-2 - 1e-1).
The plateau is the basis, not collocation: 100 terms at the apex are mostly surplus, and the base
corners' 100 do not reach 1e-14. The pre-registered mechanism (collocation starvation) was right,
and fixing it removed the failure entirely, so the onset numbers (0.2-0.3) are moot.

**omega -> pi: the right shape, the wrong place.** The guess predicted a bound near prec to
omega/pi ~0.965, 1e-4 lost at ~0.975, and outright failure above ~0.98. Measured: the bound is
already 1e-9 at 0.79, 1e-4 is lost between 0.84 and 0.86, and nothing usable remains past 0.88. The
climb is smooth and the median tension stays high, so this is a basis shortfall and not a spurious
floor. The guess's MECHANISM, apex degree starvation, is NOT supported by the counts: at
omega = 2.71 (k ~ 17, R = 1) the rule asks for nu_max ~ 44, and the apex's 100 terms reach order
~116. Something else limits the basis here. Candidates, none tested: the base corners' columns
(orders 1e3+, which evaluate to only about 1e-11 relative per their own PrecisionWarning); the
apex's near-straight wedge (orders ~1.1 l, so 100 nearly-consecutive integer-like orders, heavily
redundant against each other); the window growing with k. Not established.

**Two declines inside the good region are a solver edge case, not a basis one.** At exactly
omega = pi/2 the search declines deterministically ("the slope estimate never settled": 0.48 ->
2.87 over 6 doublings). With the third vertex written as `1j` instead of `exp(0.5j pi)`
(6e-17 apart), it does not decline and returns 49.3480220054468 (5 pi^2, rel 1e-15). Both tension
curves jitter by ~3% far from the eigenvalue (0.794, 0.766, 0.789 at neighbouring lam), and the two
spellings differ by up to 0.03. So a non-smooth tension, not geometry, trips the slope estimator.
The likely source is a massively redundant basis: integer orders at all three corners, 300 columns
of entire functions, with the regularized rank switching columns in and out as lam moves.
Unverified. The decline is the correct response to a tension it cannot bound. The decline at 2.379
(near, not at, 3pi/4) gives the same reason; exact 3pi/4 does NOT decline (lam = 107.039346378, MP
6.5e-10, 3 of 3 runs). A planner-built basis would not be this redundant, so this is a note, not a
blocker.

**The last three angles (pi - omega <= 0.013) raise** ValueError from inf/NaN in the basis
evaluation, likely the base corners' orders (~lpi / 0.006, into the 1e4-1e5 range). That is a hard
error where a decline belongs. Recorded, not fixed.
