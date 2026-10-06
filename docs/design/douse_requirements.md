*Status: requirements, agreed 2026-10-06 (interview with the user; `douse` source read where cited).
Not yet built. Supersedes nothing; it is the yardstick `TODO.md` section 0 prioritizes by.*

# What `douse` needs from `chladni`

`chladni` exists to fix the `lappy` failures that hurt `douse` (`notes/session_2026-10-05.md`).
This document records what `douse` needs, where the line between the two packages is drawn, and the
interface across it. `douse` may be changed to fit a better workflow -- it is not held to `lappy`'s
API -- but it keeps everything that depends on HOW a domain is parameterized.

## Requirements

| | |
|---|---|
| **Domains** | Polygons: convex small-`n`, non-convex with several reentrant corners (angles toward `2 pi`), near-regular and exactly symmetric. Iterates get short edges, near-collinear vertices (angles near `pi`), and high aspect ratio. Curved boundaries are not needed yet. |
| **Indices** | `lam_k` up to `k ~ 10-30`. |
| **Precision** | Chosen by `douse` per solve and tightened as the optimizer converges. Not one fixed target. |
| **Outputs** | Eigenvalues with multiplicities; a certified bound per level; Hadamard shape-derivative integrals, as an `m x m` matrix per velocity field for a group of levels. Eigenfunctions are needed only as the input to those integrals. |
| **Time** | 10 s to 1 min per solve is acceptable. Reliability comes first. |
| **Failures to fix** | All of: dropped eigenvalues, lost precision, wrong multiplicities near degeneracies, and solver exceptions that abort a run. |
| **On a shortfall** | Return the best result, an honest bound and flags; `douse` decides. Exceptions only for invalid input. |

## Where the line is drawn

**`chladni` owns a domain**: its spectrum, the eigenfunctions, the boundary quadrature, and the
Hadamard integrals for a normal velocity given on its boundary.

**`douse` owns families of domains**: the parameterization, `d vertices / dp` or any other map to a
boundary velocity, the chain rule to `d lam / dp`, grouping levels into clusters for the optimizer,
the `prec` schedule, and when to re-plan.

`chladni` assumes no parameterization. In particular it does not compute vertex velocities or
derivatives with respect to vertices; `douse.derivative.vertex_velocity` and `velocity_fields`
stay in `douse`.

## Why: the workarounds `douse` carries today

`douse/douse/evaluate.py` (module docstring) and `douse/douse/derivative.py` record the `lappy`
behaviour they work around. Each is a missing solver-side contract:

* Seeded tracking follows a mode through a crossing, so the result must be re-sorted. Omitting the
  sort reported `lam_2/lam_1 = 2.69`, above the Payne-Polya-Weinberger bound of 2.5387.
* Two seeds can converge onto the same mode; `_verify_multiplicity` checks coincident groups
  against the tension spectrum and falls back to a full solve.
* `ttol` gates both "is this an eigenvalue" and "has it converged". `track` returned a value
  2.9e-3 from the true eigenvalue inside `ttol = 1e-3`; tightening `ttol` to 1e-8 made `solve(1)`
  return zero eigenvalues on a triangle. `douse` re-verifies against its own `track_ttol`.
* `EigensolverFailure` and `SourcesInsideDomain` are caught and retried.
* `detect_cluster` groups by relative gap at its own `rtol = 1e-6` rather than trust `lappy`'s
  multiplicities, which were wrong in both directions on `reg_ngon_6`.
* `cluster_matrices` must build a near-degenerate cluster's matrices from each member's own
  eigenfunctions: `eigenfunction_coef(mult=m)` returns an arbitrary basis, and adding the splitting
  `Lambda = diag(lam_i - lam_k)` in a mismatched basis gave trust-region ratios of -0.2 to -1.2.

The interface below is designed so that each of these is removed from `douse` -- once the `chladni`
backend passes the benchmark without them, not before.

## The interface

### Spectrum

    ep   = Eigenproblem(dom)
    spec = ep.solve(n, prec=..., plan=None, warm=None)

* Returns a `Spectrum` of levels, sorted, each with value, multiplicity, certified bound
  (`min(delta_res, mp_bound)`) and flags. The count and the order are `chladni`'s guarantee.
* `warm=` (a previous `Spectrum`) may seed the search for speed, but the result carries the SAME
  contract as a cold solve: sorted, no duplicate modes, no silent drops. Tracking is an
  implementation detail, not a mode the caller has to verify.
* A shortfall -- a frozen plan that no longer resolves the shape, a corner the quadrature cannot
  reach, a count that cannot be certified -- is a flag and an honest bound, never an exception.

### Plans: plan once, realize per iterate

`douse` freezes the basis plan at a starting shape so that `n_basis` is constant across iterates
and the objective is continuous (`evaluate.py`, "Plan once, realize per iterate").

    plan = ep.plan(prec=..., n=...)          # a frozen, shape-relative Plan
    spec = Eigenproblem(dom2).solve(n, prec=..., plan=plan)   # realized on a nearby shape

* `plan.realize(dom)` gives the same basis size on any nearby domain with the same corner
  structure. Source placement is relative to the realized shape, so there is no
  `SourcesInsideDomain`.
* The certified bound is how `douse` learns that a frozen plan has stopped resolving the shape.
* **Changing `prec` is an explicit re-plan** (decided 2026-10-06). A plan is sized for one `prec`;
  a solve with a tighter `prec` than its plan reports the shortfall in its bound and flags rather
  than silently enriching. `douse` re-plans when it tightens `prec` and accepts one discontinuity
  in the objective there.

### Hadamard integrals

    H = spec.hadamard(levels, V)        # (n_fields, m, m), plus the members' eigenvalues

* **`V` is a normal velocity on `chladni`'s boundary, given per boundary curve as a callable**
  `V(curve_index, t) -> (n_fields, len(t))`, with `t` the curve's own parameter (decided
  2026-10-06). `chladni` evaluates it wherever its quadrature needs, so it can grade toward
  corners without `douse` tracking nodes. Because `V` is evaluated per curve, it may be
  discontinuous at a corner -- the normal velocity of a moving vertex is -- and each side is read
  from its own curve. The node-array form (`chladni` hands out nodes, curve indices, `t`,
  positions and normals; `douse` returns values) is offered as a fast path.
* **Many fields at once**: one Cauchy-data evaluation serves all `n_fields` (`douse` passes
  `2 n_vert` today).
* **The boundary-condition formula is `chladni`'s**: Dirichlet `-int d_n u_i d_n u_j V ds`;
  Neumann `int (d_t u_i d_t u_j - lam u_i u_j) V ds`, per curve by its condition. `douse` never
  chooses a sign or branches on boundary conditions. Dirichlet first; Neumann and mixed follow
  `boundary_integrals.md` M6. The raw `weighted_integral` kernels stay public underneath.
* **`levels` is a group `douse` chooses** (one level, or several near-degenerate ones). The
  matrices are built in each member's own orthonormal eigenfunctions, returned with the members'
  eigenvalues, so the splitting `Lambda` and `H` share a basis. For a level of multiplicity `m`
  the basis of its eigenspace is arbitrary and only the spectrum of `sum_i d_i H_i` is meaningful;
  that is stated in the docstring, as `douse` states it now.
* **A bound on the integrals** -- from the eigenfunction residual and the quadrature error -- is
  WANTED but not yet derived or measured. M3 bounds eigenvalues only. Do not promise it until it is
  measured.

## Migration and the benchmark

* `douse` gets a small backend seam -- `solve`, `plan`, `hadamard` -- with a `lappy` and a `chladni`
  implementation. The `lappy` calls are concentrated in `evaluate.py` and `derivative.py`.
* The seam is the acceptance benchmark: the same `douse` runs through either backend, scored on
  dropped eigenvalues, certified precision and wall time, with a 1e-14 tier on representative
  domains. Cases: `douse/benchmarks`, `douse/tests`, and the `lappy` failures scattered in
  `douse/jupyter_notebooks` and `douse/archive` (to be collected), including `reg_ngon_6`.
* The workarounds listed above are removed from `douse` one at a time, each after the benchmark
  shows the `chladni` backend no longer needs it.

## Not yet established

* Only `evaluate.py`'s docstring and `derivative.py` were read. `certify.py`, `cluster_opt.py`,
  `spectral_guard.py` (which uses `lappy.asymp.weyl_count_poly`), `quasi_newton.py` and `opt.py`
  (`PolygonEVP`) may need more -- e.g. Weyl counts, or second-order information.
* Whether `douse` wants interior eigenfunction evaluation for anything (answered "no" for the
  inner loop).
* Which `lappy` failure cases hurt most -- to be collected, not assumed.
