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
