# Basis-aware boundary collocation (2026-11-07)

`notes/collocation_allocation.py`; the code is `chladni.mps.bdry_counts` and
`HelmholtzFamily.collocation_demand`.

## The failure

A Fourier-Bessel column satisfies the boundary condition exactly on its corner's two sides, so
collocation points there tell the solver nothing about its coefficient; only the other sides do.
The old default split `n_bdry` by side length alone. A short side opposite a heavily loaded corner
was then starved, and coefficient combinations vanishing at its few points (but not between them)
gave a flat, spurious tension floor over the whole window.

Isosceles triangle, unit legs, 100 FB terms per corner, prec 1e-14, 900 boundary points
(`starve`):

| apex | split | base points | tension min / median | search |
|---|---|---|---|---|
| 0.05 | length | 24 | 3.5e-16 / 4.9e-16 | declined |
| 0.05 | per-side | 200 | 7.1e-4 / 7.7e-2 | lam_1 = 4968.6069131607, MP 1.3e-6 |
| 0.2 | length | 83 | 3.5e-14 / 8.9e-14 | declined |
| 0.2 | per-side | 200 | 4.5e-3 / 2.6e-1 | lam_1 = 435.06202040916, MP 2.5e-6 |

The remaining 1e-6 is the basis (97 of the 100 apex terms are surplus there), not collocation.

## Why the demand is spread by angle, not length

lappy's `pts_per_seg` gave each corner `2 x terms` points on its far sides, split by length.
Here the split is by the angle each side subtends from the corner. The angular factors
`sin(mu theta)` are orthogonal, so a face-on side separates the columns well. An edge-on (radial)
side varies them only through `J_mu(k r)`, which is monomial-like (`mu > k r`) or a phase shift of
one oscillation (`mu < k r`), both ill-conditioned. One corner, 2N points on segments of equal
length (`conditioning`), ratio of extreme singular values of the normalized columns:

| corner | N = k | edge-on | face-on |
|---|---|---|---|
| pi/2 | 20 | 1.3e-11 | 0.99 |
| pi/2 | 40 | 2.0e-17 | 0.85 |
| 3pi/4 | 20 | 1.2e-6 | 0.99 |
| 3pi/4 | 40 | 2.2e-13 | 0.85 |

For triangles the two splits agree: each corner has one far side. The angle split also needs no
adjacency lookup, since a corner's own sides subtend zero angle.

## The rule

Per segment, the largest of: the length share of `n_bdry`; the basis's demand
(`COLLOCATION_MULT = 2` points per FB column on the corner's far sides, by subtended angle); and,
with a wavenumber, `k L / pi`. The total can exceed `n_bdry`. `plan_for` freezes the counts in
`Plan.bdry_counts`, so nearby douse iterates are collocated identically (before, `allocate` could
move a point between sides from one iterate to the next). Not handled: visibility on non-convex
shapes (a side partly hidden behind a reflex vertex is credited its whole span).

## What it changes (`census`)

`plan_for(dom, 10, 1e-10)` on every rung for 19 shapes: library shapes, GWW1, plus, H, chevron,
triangles with apex 5/13/30/150/170 degrees, douse's stalled hexagon. 93 of 95 plans keep their
counts bit for bit. Two differ, both rung 0 on flat-apex triangles, where the near-straight apex
carries most of the columns and its demand on the base exceeds the base's length share: apex 170
(+37 points on the base, 177 -> 214) and apex 150 (+1). The wavenumber floor never binds.

So on planner-sized bases the length split was almost always enough ("collocation density was
never binding", `plan.py`), and this does NOT explain the thin triangles in `notes/hard_cases.md`
(apex 5 and 13 are unchanged). Where it matters is bespoke and fixed-size bases, like the pure-FB
isosceles sweep (`notes/isosceles_pure_fb.md`), and any future planner rule that loads one corner
heavily.
