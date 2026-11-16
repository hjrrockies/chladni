# Smooth (Fourier-parameterized) domains: first contact, 2026-11-16

Working notes. They record what was measured when a smooth domain was first put through
`chladni`: Osting's (2010) optimal domains for lambda_n/lambda_1, digitized from his Fig. 3 as
cosine series r(t) = sum_k b_k cos(k t). Scripts live outside this repository:
`~/projects/latex/thesis/figures/src/osting2010_shapes.py` (the fit) and
`~/projects/thesis-runs/osting_lam11/eval_osting.py` (the solve).

## What worked

* `ParametricSegment(p, dp, 0, 2*pi)` on z(t) = r(t) e^{it}, in one `Boundary`, in a `Domain`.
  The parametric layer itself was no trouble.
* `MPSEigensolver` with an explicit Fourier-Bessel family at an interior point:
  `CornerModes(nu=1.0, kind="cos"|"sin", offset=0.0, omega=2*pi, beta=pi)` gives integer orders
  J_l(k r) cos(l t) or sin(l t), single-valued (the cut at beta = pi is harmless for integer
  orders). For a domain symmetric about the x-axis, cos and sin are the even and odd classes;
  solving them separately and merging keeps a double eigenvalue's two members apart.
  * Exact unit disk, centred and offset by 0.135: the first 7 eigenvalues to 8 printed digits.
  * Osting n = 11 (rounded triangle, R/r_min about 1.5): degrees 0..65 and 0..87 agree to 1e-7.

## What did not

* **Planner:** `Eigenproblem(...).plan_and_solve` refuses a non-polygon:
  "the planner covers Dirichlet polygons only" (`plan.py`, `_require_dirichlet_polygon`).
* **Silent missing roots:** with too few FB degrees (L = 35 for n = 11, k_max R about 14),
  `solve_interval` on [3, 123] returned 5 of 16 eigenvalues (cos 2, sin 3), with
  `declined=False` and no flags. At L = 60 it finds all 21 in [3, 110], and L = 90 agrees. A search
  that loses two thirds of its roots without a word is the failure `chladni` exists to prevent
  (section 0 of `TODO.md`); on polygons the planner's count audit would catch it.
* **Plane waves:** `PlaneWave.uniform(1.5..2 * n_useful, radius=diameter)` on the digitized
  n = 2 disk (off-centre: b_1 = -0.135) gave lambda_2/lambda_1 = 2.4576 with lambda_3 missing,
  against about 2.538. Yet the same path solved the exact circle, centred, to all printed digits.
  Not diagnosed.
