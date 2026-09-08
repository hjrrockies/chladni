# Symmetry Handling in an MPS/MFS Laplacian Eigensolver

**Status:** design note / handoff document
**Audience:** an agent or developer implementing this in an existing MPS/MFS package
**Scope:** automatic symmetry detection from a piecewise-smooth boundary curve, and what to
actually do with the result in a spectral shape-optimization loop

---

## 0. Context and the short version

The package solves Dirichlet Laplacian eigenvalue problems on planar domains by the method of
particular solutions / method of fundamental solutions, with an eigensolver-agnostic `geometry`
module. The primary use case is shape optimization and shape inference with spectral objectives.

A symmetry-aware solve path already exists in `lappy`. The open question was whether to detect symmetry
automatically from the geometry and switch to the reduced path when it is found, up to a
user-set tolerance.

**Recommendations, in priority order:**

1. **Build the detector.** It is cheap, robust, and useful as a diagnostic. (§1)
2. **Do not auto-swap to the reduced solve on detected symmetry.** A tolerance threshold makes
   the objective discontinuous in shape, and it fires precisely in the endgame of the
   optimization where that hurts most. (§2)
3. **Build a symmetry-adapted basis instead.** Same conditioning benefit, acquired continuously
   as the domain approaches symmetry, with zero modeling error and no switch. (§3.1)
4. **Expose exact symmetry reduction only behind an explicit user declaration**, never behind
   detection. (§3.3)
5. **Separately, implement cluster-aware shape derivatives.** This is where the optimization is
   actually losing near optimizers, and no amount of eigensolver accuracy fixes it. (§4)

Motivating observation from the user: symmetric domains show up in the wild for spectral shape
optimization essentially only *at* the optimizers. That is exactly why the auto-swap is
dangerous and why §4 matters.

---

## 1. Automatic symmetry detection

### 1.1 What we are searching over

The symmetry group of a bounded planar domain is a finite subgroup of `O(2)`, hence either
`C_m` (cyclic, rotations only) or `D_m` (dihedral, rotations plus `m` mirrors).

The fixed point is **forced to be the area centroid**. The centroid map is equivariant under
isometries, so any symmetry of the domain must fix it. This eliminates the translation search
entirely. Translate to the centroid once and the remaining unknowns are the order `m` and, for
dihedral groups, one mirror axis angle `θ`. Very small search space.

### 1.2 Proposal step: complex area moments

Define

```
c_pq = ∫∫_Ω (x + iy)^p (x - iy)^q dA
```

Compute these as boundary integrals via Green's theorem over the existing arc parameterization.
Cost is `O(N)` boundary evaluations — negligible next to a single eigensolve.

Transformation rules:

- rotation by `α`:  `c_pq  ↦  e^{i(p-q)α} c_pq`
- reflection:       `c_pq  ↦  conj(c_qp)` (up to a phase set by the axis)

Consequences used for detection:

- **Rotation order.** `C_m` symmetry forces `c_pq = 0` unless `m | (p - q)`. Scan `|c_pq|`,
  normalized by `|c_00|` and the appropriate power of the radius of gyration, over small values
  of `p - q`. Read off the largest `m` consistent with the observed near-zeros.
- **Mirror axis.** A mirror at angle `θ` forces `arg c_pq ≡ (p-q)·θ  (mod π)`. All the phases
  align, so a weighted least-squares fit of `arg c_pq` against `(p-q)` over several `(p,q)`
  pairs recovers `θ` directly. Weight by `|c_pq|` so that near-vanishing moments (whose phases
  are noise) do not pollute the fit.

Use moments up to total degree ~8. Higher moments are increasingly sensitive to boundary detail
and increasingly ill-scaled.

### 1.3 Verification step: geometric residual

Moments give a candidate, not a certificate. Two different domains can share low moments. For
each candidate isometry `g`, compute

```
ε(g) = sup_{x ∈ ∂Ω} dist(g·x, ∂Ω) / diam(Ω)
```

using closest-point queries against the arcs. Evaluate the sup over a fine sample of boundary
points (denser than the collocation set). Then polish: a few Levenberg–Marquardt steps over the
axis angle `θ` (and the centroid, if you want to allow for quadrature error in it) will sharpen
`ε` by orders of magnitude when a true symmetry is present, and will fail to improve when it is
not. That failure-to-improve is itself a useful signal.

Report `ε` per group element and take the group to be the set of elements below the caller's
threshold — but see §1.5 on why `ε` alone is the wrong quantity to gate on.

### 1.4 Short-circuit for structured parameterizations

If the shape parameterization is a radial Fourier series, a periodic spline with uniform knots,
or similar, symmetry detection reduces to reading coefficients:

- `C_m`: only harmonics with index divisible by `m` are nonzero.
- mirror: all coefficients become real after a single global phase rotation.

If this is the design space in use, skip the moment machinery entirely and inspect coefficients.
Much cheaper and exact.

### 1.5 What the detector should return

```python
@dataclass
class SymmetryReport:
    group: str            # "C1", "C2", "D1", "D4", ...
    order: int
    axes: list[float]     # mirror axis angles, radians, from centroid
    center: complex       # area centroid
    eps: dict             # per-group-element geometric residual, normalized
    delta_lambda_rel: dict | None   # predicted relative eigenvalue shift, see §2
```

Note `delta_lambda_rel`. The geometric residual `ε` is *not* the quantity a caller cares about.
What matters is how much the spectrum would move under symmetrization, and that requires the
Hadamard formula in §2. If normal derivatives from a recent solve are available, fill this in;
otherwise leave it `None` and say so.

---

## 2. Why a tolerance-gated auto-swap is the wrong design

Symmetry-reducing an *approximately* symmetric domain does not merely run faster. It solves a
different problem: the reduced solve returns the spectrum of a nearby symmetrized domain `Ω̃`,
not of `Ω`.

By Hadamard's formula, for a simple eigenvalue with deviation field `V` carrying `∂Ω` to `∂Ω̃`,

```
δλ_j = - ∫_{∂Ω} |∂_n u_j|² (V · n) ds
```

This is a systematic modeling error, not a discretization artifact. Three consequences:

1. **The objective becomes discontinuous in shape.** Crossing the tolerance mid-run makes `λ`
   jump. Line searches and quasi-Newton updates (BFGS in particular) degrade badly, and the
   failure is invisible in the code — it presents as a single bad function evaluation, or as an
   inexplicable loss of superlinear convergence.

2. **The discontinuity lands in the endgame.** Since symmetric domains appear near optimizers,
   the threshold fires exactly when the iterates are converging and the optimizer is most
   sensitive to objective noise.

3. **The natural tolerance is not geometric.** One would want to gate on the predicted
   `|δλ|/λ` from the formula above (which is computable — `∂_n u_j` is already on hand for shape
   derivatives). But that only replaces a geometric threshold with a spectral one. It is still a
   smooth quantity crossing a hard threshold, and the discontinuity remains.

There is no threshold that fixes this. The problem is the switch itself.

---

## 3. What to do with detection instead

### 3.1 Symmetry-adapted basis (recommended, build this)

Keep the full domain. Symmetrize the *basis*, not the *problem*.

Given a candidate group `G` with irreducible characters `χ`, replace each charge/source
contribution by an irrep-projected combination:

```
Φ_χ(x, q) = Σ_{g ∈ G} χ(g) · Φ(x, g·q)
```

for each charge point `q` in a fundamental domain of the exterior charge curve, and each irrep
`χ`. For `D_m` with two-dimensional irreps, use the full projection operator rather than the
character alone, or equivalently take both basis vectors of the irrep.

Properties:

- **Completeness is preserved.** The union over all irreps spans the same space as the original
  basis. Nothing is lost, no error is introduced, and the result is exact for any `Ω`,
  symmetric or not.
- **Continuity is preserved.** No switch, no threshold, no discontinuity in the objective.
- **Conditioning improves continuously.** As `Ω → Ω̃`, the least-squares (or GSVD / subspace-angle)
  matrix becomes block-diagonal by irrep. The near-degenerate eigenvalues that wreck MPS
  conditioning separate into different blocks. The benefit arrives smoothly as `ε → 0`.

The trade-off: this does not give the `|G|²`-type flop savings of a genuine reduced solve, since
the full boundary is still discretized. But **conditioning, not flops, is the MPS bottleneck** at
the problem sizes in play, so this is the win that matters.

Implementation notes:

- Charge points must be placed `G`-equivariantly for the projection to be meaningful. If the
  detector reports a group, generate the exterior charge curve on a fundamental domain and
  replicate under `G`.
- The same applies to collocation points on `∂Ω`. Approximate symmetry in the geometry plus
  exact symmetry in the point sets is fine and is what produces the near-block structure.
- Use the detected group even when `ε` is loose (say `10⁻²`). A wrong guess costs only that the
  blocks fail to decouple; it never costs accuracy.

### 3.2 Irrep labels for mode tracking

Eigenvalue branches cross during optimization. Tracking `λ_k` by index across iterations is a
well-known source of garbage gradients.

The overlap characters `⟨u_j, g·u_j⟩` are computable for approximately symmetric domains too
(evaluate `u_j` at a point set and its image under `g`). They give a robust label for following
a branch through a crossing, and they cost one extra eigenfunction evaluation pass.

This is cheap and worth wiring in regardless of the rest.

### 3.3 Exact reduction only when symmetry is *declared*

If the user asserts symmetry, or the parameterization enforces it by construction, the reduced
path is exact and should be taken. Design rule:

- Symmetry is a **property on the geometry object**, set by the user or by a constructor
  (`Domain.from_symmetric_fourier(...)`, `domain.declare_symmetry("D2", axes=[0, pi/2])`).
- The detector **reports** and never **sets**. Its output is a diagnostic and a hint for basis
  construction (§3.1), not a control-flow switch.

This separation keeps the fast path honest and the objective smooth.

For the reduced solve itself on a domain with `D_2` symmetry (e.g. the stadium), prefer the
**image-charge** formulation over explicit domain cutting: reflect the charges across both
symmetry axes and use

```
G_χ(x, q) = Σ_{j} ± G(x, σ_j q)
```

with signs chosen per parity class. The symmetry conditions are then satisfied identically, only
the physical quarter of the boundary is discretized, and the four parity classes are four sign
patterns on the same geometry. This also avoids introducing mixed Dirichlet/Neumann conditions
on the symmetry cuts, which is a real complication if the package ever grows a BIE backend.

---

## 4. The thing actually biting the optimizer: eigenvalue clusters

This is independent of symmetry detection and is probably the higher-value work item.

Optimizers of `λ_k` generically have **multiple eigenvalues** — that is *why* symmetry appears
at optimizers in the first place. And multiple eigenvalues are not differentiable in shape. They
are only directionally differentiable. For a cluster `{u_i}` of multiplicity `r` and deviation
field `V`, the directional derivative is the largest eigenvalue of the `r × r` matrix

```
M_ij(V) = - ∫_{∂Ω} (∂_n u_i)(∂_n u_j) (V · n) ds
```

Consequences for the optimizer:

- Smooth single-eigenvalue gradients **will stall or oscillate** in a cluster, no matter how
  accurate the eigensolve is. Chasing eigensolver digits will not fix it.
- The correct move is **cluster-aware descent**: detect the near-degenerate group by a relative
  gap test, assemble `M(V)` for the candidate directions, and obtain a descent direction from a
  small SDP or LP over the cluster. The subproblem is `r × r` with `r` typically 2–4, so it is
  cheap.
- Symmetry detection is useful here mainly as a **predictor**: the irrep structure tells you
  which clusters to expect and how large they will be, which makes the gap test far more robust
  than a bare numerical threshold.

### 4.1 If you want the symmetry speedup in the optimization proper

The clean route is to **restrict the design space** to `G`-symmetric shapes, rather than
symmetrizing the solve on a general shape.

Palais's principle of symmetric criticality applies: for a `G`-invariant objective on a
`G`-invariant design manifold, the shape gradient at a symmetric design is automatically
`G`-symmetric, so the symmetric subspace is invariant under the flow, and any critical point
found within it is genuinely critical for the full problem. Stability is then checked separately
by testing antisymmetric second-variation directions.

This gives the full `|G|²` speedup, **exactly**, with a smooth objective throughout. The price is
committing to the symmetry up front rather than discovering it. For many spectral shape problems
where the symmetry class of the optimizer is conjectured or known, this is the right trade.

---

## 5. Suggested implementation order

| # | Item | Effort | Notes |
|---|------|--------|-------|
| 1 | Moment-based detector + geometric verification (§1) | 1–2 days | Pure `geometry` module; no eigensolver coupling |
| 2 | `SymmetryReport` plumbed through as diagnostic output | hours | Report only, never sets solver state |
| 3 | Irrep mode labels via overlap characters (§3.2) | 1 day | Immediate payoff for branch tracking |
| 4 | Symmetry-adapted (irrep-projected) MPS basis (§3.1) | 3–5 days | The main conditioning win |
| 5 | Declared-symmetry exact reduction via image charges (§3.3) | 2–3 days | Behind explicit user flag only |
| 6 | Cluster-aware shape derivatives (§4) | 1–2 weeks | Highest value for the optimization itself |

Items 1–3 are self-contained and low risk. Item 4 is the recommended replacement for the
auto-swap. Item 6 is separable and can proceed in parallel.

---

## 6. Validation

**Detector:**

- Exactly symmetric test cases: disk (`O(2)`, should report high `m` and saturate), square (`D_4`),
  equilateral triangle (`D_3`), stadium (`D_2`), regular `m`-gons for `m` up to 8.
- Chiral test case: a `C_3`-symmetric non-mirror shape (e.g. a radial Fourier shape with
  harmonics 3 and 6 at complex phases that do not align). The detector must report `C_3`, not `D_3`.
- Near-symmetric sweep: take a `D_2` shape and perturb by amplitude `δ` over
  `δ ∈ [10⁻¹⁰, 10⁻¹]`. Check that reported `ε` tracks `δ` linearly and that the axis angle
  estimate remains stable. This sweep is also the evidence for §2: plot the reduced-solve `λ`
  against the full-solve `λ` over the same sweep and confirm the systematic offset scales as
  Hadamard predicts.
- Rotated/translated copies of every case above: results must be equivariant, with axis angles
  shifting exactly by the applied rotation.

**Symmetry-adapted basis:**

- On an exactly symmetric domain, confirm the least-squares matrix is block-diagonal to
  round-off and that the per-irrep spectra match the reduced solve.
- On a perturbed domain, confirm the full spectrum is recovered to the same accuracy as the
  unprojected basis (this is the completeness check — it must not degrade).
- Confirm the condition number improves monotonically as `δ → 0` with no discontinuity.

**Cluster derivatives:**

- Finite-difference the directional derivative along several `V` at a known degenerate
  configuration and compare against the largest eigenvalue of `M(V)`. The naive single-eigenvalue
  formula should visibly fail the same test — keep that as a regression check so the
  distinction does not get optimized away later.

---

## 7. References

- Betcke & Trefethen, *Reviving the method of particular solutions*, SIAM Review 47 (2005) —
  subspace-angle / GSVD formulation, interior normalization.
- Barnett & Betcke, *Stability and convergence of the method of fundamental solutions for
  Helmholtz problems on analytic domains*, J. Comput. Phys. 227 (2008) — convergence governed by
  analytic continuation, not Sobolev regularity.
- Henrot, *Extremum Problems for Eigenvalues of Elliptic Operators*, Birkhäuser (2006) —
  Hadamard formulas, multiplicity at optimizers, directional differentiability.
- Palais, *The principle of symmetric criticality*, Comm. Math. Phys. 69 (1979).
- Cox & Ross, and subsequent work on multiple-eigenvalue optimization — for the SDP/LP
  formulation of the cluster descent subproblem.
- Standard references on moment invariants for planar shape symmetry detection (complex moment
  phase alignment) — the construction in §1.2 is elementary enough to re-derive, but the
  literature has useful notes on conditioning and moment-order selection.

---

## 8. One-line summary for the implementing agent

Detect symmetry, report it, and use it to build a better-conditioned *complete* basis and to
label modes — but never let detection silently change which problem is being solved.
