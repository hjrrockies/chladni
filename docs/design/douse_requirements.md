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
| **Domains** | Polygons up to N = 16: convex, non-convex with several reentrant corners (angles toward `2 pi`), near-regular and exactly symmetric. Iterates get short edges, near-collinear vertices (angles near `pi`), and high aspect ratio. Curved boundaries are not needed yet. |
| **Indices** | `lam_k` up to `k ~ 10-30`. |
| **Precision** | Chosen by `douse` per solve and tightened as the optimizer converges. Not one fixed target. `douse`'s default today is 1e-13; **the first integration targets 1e-10** (agreed 2026-10-09), with 1e-13 a later tightening. |
| **Outputs** | Eigenvalues with multiplicities; a certified bound per level; Hadamard shape-derivative integrals, as an `m x m` matrix per velocity field for a group of levels. Eigenfunctions are needed only as the input to those integrals. |
| **Time** | 10 s to 1 min per solve is acceptable -- 1-7 hours per optimization start at a few hundred solves (confirmed 2026-10-06). Reliability comes first. |
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

`SpectrumEvaluator` (`evaluate.py`) adds three more, each from a lost run:

* **Spurious values.** A stalled run carried `58.1948` as its `lam_3` at a tension of 0.62 -- not an
  eigenvalue under any basis -- while reporting `worst_sigma = 1.3e-09`, because `worst_sigma` was
  measured at `lams[0]` only. `spurious_tol = 1e-4` now checks the tension at EVERY returned value
  and escalates to a full solve.
* **Missed modes.** A run optimized `lam_3/lam_1` for 92 iterations believing it was `lam_2/lam_1`
  (ratio 2.62, above PPW): `lam_2` was absent, and every returned value was a genuine eigenvalue.
  `douse.spectral_guard` audits the count against Weyl's three-term law with corner terms
  (`lappy.asymp.weyl_count_poly`) at mid-gap cuts, and escalates, re-plans, then refuses. A dropped
  mode plus a spurious one preserves the count, which is why both guards exist.
* **Re-planning on drift.** A frozen plan that stops resolving the shape shows up as rising
  tension; `replan_tol = 1e-9` re-plans at the current shape. `cluster_opt._degeneracy_move` also
  re-plans at a trial shape. Re-planning is routine in a run, not only a `prec`-change event.

The interface below is designed so that each of these is removed from `douse` -- once the `chladni`
backend passes the benchmark without them, not before.

## The interface

### Spectrum

    ep   = Eigenproblem(dom)
    spec = ep.solve(n, prec=..., plan=None, warm=None)

* Returns a `Spectrum` of levels, sorted, each with value, multiplicity, certified bound (the
  Moler-Payne bound, or the tighter `delta_res` where that met `prec` -- `c1c3f53`) and flags. The
  count and the order are `chladni`'s guarantee.
* `warm=` (a previous `Spectrum`) may seed the search for speed, but the result carries the SAME
  contract as a cold solve: sorted, no duplicate modes, no silent drops. Tracking is an
  implementation detail, not a mode the caller has to verify.
* A shortfall -- a frozen plan that no longer resolves the shape, a corner the quadrature cannot
  reach, a count that cannot be certified -- is a flag and an honest bound, never an exception.
* **Every returned level is certified, not just the first.** The bound is per level; a value that
  is not an eigenvalue cannot come back with a clean flag (the `spurious_tol` failure).
* **The count is part of the contract.** `solve(n)` returns `lam_1 .. lam_n` with nothing skipped,
  or flags that it cannot say so. Weyl's corner-corrected count is a `Domain` quantity, so
  `chladni` provides it (`weyl_count(dom, lam)`, as `lappy.asymp.weyl_count_poly` does); `douse`'s
  guard keeps running on it as an independent audit during migration.
* **A level straddling `n` is returned whole.** `solve(5)` on the square returns
  `[2, 5, 5, 8, 10, 10] pi^2` and says the last level has multiplicity 2, rather than truncating a
  degenerate cluster. `douse` drives shapes toward degeneracy, so this is the common case
  (`spectral_guard.py`: truncation made a clean set and a set missing a mode indistinguishable).

### Precision per level: loosen, report, optionally abort (agreed 2026-10-09)

*The agreed direction; not designed in detail and not built. The first integration works at
`prec = 1e-10` (TODO.md section 0).* `prec` is a request, and the
basis decides per level whether it can be met. The floor varies by orders of magnitude across one
window, so a shortfall is a property of a LEVEL, not of a solve.

    spec = ep.solve(n, prec=1e-10, plan=plan, on_shortfall="flag")     # or "raise"
    L = spec.levels[i]
    L.lam, L.mult, L.bound, L.resolution, L.met, L.flags
    spec.count_audit                                                    # solve-level: drops

* **Two numbers per level, because they come apart.** `bound` says how well the VALUE is known
  (it exists today). `resolution` says over what neighbourhood the COUNT was taken: a level of
  multiplicity `m` at resolution `r` stands for the eigenvalues within `r` of `lam`, whatever their
  internal splitting. A value can be known to 1e-14 while its count is only resolved at 1e-9 -- the
  L-shape's `10 pi^2` is exactly that. `met` requires both within `prec * lam`. `Level.width` (today,
  for collapsed groups only) is the special case this generalizes.
* **The solver loosens `resolution` per level instead of reading noise.** Where the request is too
  close to the local floor scale `eps/c` for the multiplicity evidence to be unambiguous, the level is
  counted at a coarser licence and reports it. First evidence for where that line sits:
  `notes/count_ratio.py` -- on the bound-coverage grid (4 domains, pre-(C2) code) every wrong
  multiplicity had `prec*lam / (eps/c) <= 31`, against a median of 117 for correctly counted
  degenerate levels, and counting below a threshold of 32-50 at `T * eps/c` would have merged a
  distinct neighbour once in 363 levels. Twelve failures, one dataset, and the rule not yet
  prototyped. `eps/c` must be read where the cone has left the floor (the wide bracket), not from the
  cache slope that read 14000x steep on the spike triangle (`notes/spike_x8.py`).
* **This is clause (2) of the search's contract, extended from values to counts.** Today a value is
  within `prec` or the solver reports what it attained (`docs/design/search.md` section 2.4);
  `resolution` makes the same promise for a count. It is also the "best result plus an honest bound
  and flags" this document already asks for on a shortfall.
* **`on_shortfall="flag"` is the default; `"raise"` is an opt-in early abort.** douse spends 10 s - 1
  min per solve and wants to stop a solve that cannot meet `prec` as soon as that is known. The
  exception (`PrecisionShortfall`, name open) carries the first deficient level, how far short it was
  (the request against the local floor scale), and the partial spectrum -- the input to a re-plan,
  not just the fact of failure. Flags, not warnings, are what douse branches on; a warning in
  addition serves interactive use, under the rule that only substantive losses warn.
* **The retry is a re-plan, not a tighter `prec`.** A shortfall means the basis's floor at that level
  sits above what `prec` needs; the same plan fails the same way at any tighter request. douse
  re-plans at the same `prec` (an enriched basis), consistent with "changing `prec` is an explicit
  re-plan" below. The exception's payload says where to enrich.
* **"As soon as possible" is after the first deficient level is refined.** A level's floor is known
  only once it is located; it cannot be predicted before the solve starts. Strict mode saves the
  rest of the window, not the whole solve.
* **Drops cannot be expressed per level.** A missing eigenvalue has no level to carry a flag, and the
  sector with too few apex terms dropped modes whose tension has no well at all
  (`notes/count_ratio.py`): no local reading can see them. `count_audit` -- Weyl's corner-corrected
  count against what was returned, per window -- is the solve-level check, and the first thing
  strict mode should abort on, since silent drops are douse's worst failure class.
* **Locating can fail where counting is sound, and the design must cover it.** T1.2's near-double
  (two eigenvalues 9e-10 apart, ~45 floor scales, so resolvable) comes back as 1, 2 or 3 depending
  on ulps of the window's ends (`notes/nondeterminism.md`): the pair is missed or one member found
  twice, unflagged. Loosening the counting licence does not fix a member stage 1/2 never located.
  Either the search is fixed so the pair is found every time, or a level whose neighbourhood shows
  unexplained structure is loosened to cover it. Not yet traced.

#### Phase 3, decided 2026-10-11: report, don't re-count

Measured before designing (`notes/margin_rule_offline.md`, with its 2026-10-11 correction): on the
douse benchmark no level that meets `prec` had a wrong count, on either basis. Every count defect
was one of four kinds:
- (a) a whole level dropped, with no solved level near it;
- (b) a value off by 1e-6 to 1e-4 on an unmet level, whose bound covered the error;
- (c) the partner of a close pair missed, beside an unmet level;
- (d) a simple level read as a double inside its own floor, on an unmet level.

So the per-level rule is a REPORTING change, with no new search machinery:

* **`Level.margin = 2 * prec * lam / delta_res`**, the R of the plan, with `delta_res = DELTA_SAFETY *
  eps/c`. `met` holds exactly when `margin >= 2` (all 8062 benchmark levels).
* **`Level.resolution = max(prec * lam, bound, width)`**, the neighbourhood the count stands for. On
  a met level it is the merge licence. On an unmet level it widens to the value's own bar, since a
  level whose value is known to 1e-4 cannot claim its count at 1e-10. `width` (collapsed groups)
  is the special case.
* **`Level.count_shortfall`** is True on an unmet level: the count was read where the floor blurs
  it, which catches (c) and (d). It is a FLAG and never raises. It does not feed
  `certified_counts`, which `solve_interval` raises on: about 40% of levels at 1e-10 are unmet
  even on the references' basis. Diagnostics count it.
* **Not here, by design:**
  - (a) is the solve-level `count_audit` (Phase 4).
  - A double read as simple on a MET level happens only on starved bases (3 cases on the
    `count_ratio` grid, R up to 18). Catching it would need T ≈ 20, which flags about half the
    healthy levels on douse shapes. It belongs to the planner, which must not stop at a basis that
    fails `basis_sufficiency`'s multiplicity precondition (Phase 5).
* **Dropped from the plan:** the neighbourhood chosen at a spectral gap, the coarser re-count, and
  a T threshold sweep. Nothing measured needs them.

### Plans: plan once, realize per iterate

`douse` freezes the basis plan at a starting shape so that `n_basis` is constant across iterates
and the objective is continuous (`evaluate.py`, "Plan once, realize per iterate").

    plan = ep.plan(prec=..., n=...)          # a frozen, shape-relative Plan
    spec = Eigenproblem(dom2).solve(n, prec=..., plan=plan)   # realized on a nearby shape

* `plan.realize(dom)` gives the same basis size on any nearby domain with the same corner
  structure. Source placement is relative to the realized shape, so there is no
  `SourcesInsideDomain`.
* The certified bound is how `douse` learns that a frozen plan has stopped resolving the shape,
  replacing `replan_tol` on the tension. Re-planning happens many times in a run -- on drift, on a
  degeneracy move, on a Weyl shortfall -- so `ep.plan` must be cheap next to a solve.
* A `Plan` is recordable: `douse` logs refused shapes with the shape their plan was built at, so a
  refusal can be re-solved off-line. Pickling the `Plan` (or rebuilding it from its domain and
  `prec`) must reproduce it exactly.
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
  measured. It matters: `certify.first_order_report` tests `tr M = 0` and `grad lam_1 = 0` at
  `tol = 1e-8` relative, and those facts were measured to 1e-13, so the integrals need roughly
  `prec` relative accuracy to be useful there.
* **No second derivatives.** `quasi_newton` builds BFGS curvature from gradients, and
  `certify.second_order_probe` differences the objective along null directions. Neither needs a
  Hessian from `chladni`, but the probe needs eigenvalues accurate well below its differences
  (~3e-4 at `t = 0.02`).

## Migration and the benchmark

* `douse` gets a small backend seam -- `solve`, `plan`, `hadamard` -- with a `lappy` and a `chladni`
  implementation. The `lappy` calls are concentrated in `evaluate.py` and `derivative.py`.
* The seam is the acceptance benchmark: the same `douse` runs through either backend, scored on
  dropped eigenvalues, certified precision and wall time, with a 1e-14 tier on representative
  domains. Cases: `douse/benchmarks`, `douse/tests`, and the `lappy` failures scattered in
  `douse/jupyter_notebooks` and `douse/archive` (to be collected), including `reg_ngon_6`.
* The workarounds listed above are removed from `douse` one at a time, each after the benchmark
  shows the `chladni` backend no longer needs it.

## What the `douse` code says about scale and precision

Read 2026-10-06: `evaluate.py`, `derivative.py`, `certify.py`, `cluster_opt.py`,
`spectral_guard.py`, `quasi_newton.py`, `opt.py`.

* **Precision in use today is 1e-13.** `SpectrumEvaluator` defaults to `prec = 1e-13` and a basis
  `target = 1e-14`, because finite-difference checks are limited by the tension floor. A 1e-13
  tier on ordinary (non-clustered) levels is therefore `douse`'s working default, not a stretch
  goal.
* **Clusters are wide.** `cluster_opt` treats levels split by ~1e-3 as one eps-active cluster, with
  the width set by the trust radius. `hadamard(levels, V)` must accept well-separated members; it
  is not limited to levels `chladni` merged.
* **Polygons up to N = 16** (`n_p = 28`, `quasi_newton.py`).
* **A solve is ~300 ms today** (`cluster_opt.py`, `quasi_newton.py`), and a run is hundreds of
  iterations (92, >250, 400-iterate starts). The 10 s - 1 min budget agreed above is per solve;
  at a few hundred solves per run that is 1-7 hours per start. **Confirmed acceptable by the user
  (2026-10-06)**: reliability is bought with time, up to that envelope.
* `opt.py` is dead: it imports `.param` and `.evp`, which no longer exist. `PolygonEVP` is not a
  requirement.

## Not yet established

* Whether `douse` wants interior eigenfunction evaluation for anything (answered "no" for the
  inner loop).
* Which `lappy` failure cases hurt most. They are collected in `douse_failure_cases.md`
  (2026-10-06); the ranking is not yet made.
* `objective.py`, `optimize.py`, `constraints.py`, `sampling.py` and `polygon.py` were not read for
  solver needs; `polygon.py` uses `lappy.Polygon` and `lappy.utils` geometry, which `chladni.geometry`
  would have to cover.
