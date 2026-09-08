# Candidate approaches: a breadth-first enumeration

Working note. The goal is a SHORTLIST, and the filter is `docs/design/search.md` section 2 --
the operational definition. An approach earns a place only if it can plausibly deliver all
three clauses: the multiset `S(prec)`, per-eigenvalue accuracy, and honest self-report.

---

# 0. The reframing that generates the list

This is **not global optimization**. Global optimization finds `argmin f`. We need

    { x in [a, b] : x is a local minimum of sigma_1 with prominence >= P }

with a multiplicity at each, and a certificate that none was missed. That is a **level-set /
sublevel-set problem** -- find and certify `{f <= tau}` -- with a local-structure problem
inside each component. Two consequences:

* the hard half is the CERTIFICATE (what is provably absent), not the location;
* methods designed to converge fast to ONE minimum are mostly irrelevant, and methods designed
  to COUNT or ENCLOSE are directly relevant even when they locate nothing.

The list below is organized by what supplies the certificate.

---

# 1. Deterministic sampling, local extraction

Certificate: "the samples were fine enough." Weakest kind, and everything currently built is
here.

**A1. Uniform grid.** The control. `harness.uniform_grid_search`.
*Falsifiable:* a grid at `s` samples per Weyl gap reaches recall 1.0 for `s >= s*`.
*Measured:* `s* ~ 18` on the roster, and it is not a bound -- section 20.2.

**A2. Weyl-STRATIFIED grid.** Sample equispaced in `N(lam)`, not in `lam`. Where the local
gap is tight, `N` rises fast, so points concentrate there automatically.
*Falsifiable:* at equal point count, stratified recall >= uniform recall, with the gain
largest where `min gap / mean gap` is smallest (`mushroom` 0.127, `H_shape` 0.210).
*Test:* resample the cached fine scans. Nearly free. **RUN BELOW.**

**A3. Recursive bisection into every dip.** `lappy_repaired_search`. 10/12 at 5573 evals.
*Falsifiable:* the recursion floor is not load-bearing once it is `max(prec*lam, delta_res)`.
*Note:* the open recursion-floor bug lives here.

**A4. Audit-guided local refinement.** Section 21.
*Measured:* works where the miss is a resolution problem; structurally cannot reach a miss
that sits AT a level; no basis guard.

**A5. Coarse-to-fine cascade sharing evaluations.** One sorted cache, repeated passes at
halved stride, extraction from the cache each time.
*Falsifiable:* total evaluations to recall 1.0 are within 2x of a single well-chosen grid,
so adaptivity costs nothing when it is not needed.

---

# 2. Bounding and certification

Certificate: a proven lower bound on `f` over an interval. The only family that can certify
absence rather than assert it.

**A6. Piyavskii-Shubert with a calibrated `L`.** The originating proposal. Built, broken,
7/12.
*Falsifiable:* the certificate is only as good as `L`, and `L` from pilot secants
underestimates the true slope on some real domain.
*Test:* compare `gamma * max|ds/dlam|` from a pilot against the true max over the fine scans.
**RUN BELOW.**

**A7. DIRECT (DIviding RECTangles).** Lipschitz-free: no constant to calibrate, considers all
Lipschitz constants at once by keeping the potentially-optimal set.
*Falsifiable:* DIRECT's "potentially optimal" rule is for a single global minimum; extended
to keep every interval potentially containing a point below `tau`, it reaches recall 1.0 at
fewer evaluations than A1.
*Test:* simulate on cached curves. Cheap. **RUN BELOW.**

**A8. Cone-model envelope.** Bound from the MEASURED structure
`sigma_1 ~ sqrt(eps**2 + c**2 delta**2)` rather than from secants. Between two samples the
cone model says how deep a dip can hide.
*Falsifiable:* the cone bound is tighter than the Lipschitz bound at equal validity, i.e. it
prunes more intervals without ever pruning one containing a root.

**A9. Interval arithmetic / rigorous enclosure.** The only route to a true proof.
*Assessment:* needs an interval extension of a generalized singular value through a GSVD.
Not available and not plausible to build. **REJECT, with the reason recorded.**

**A10. Derivative-bounded envelope.** If `dsigma/dlam` is available AT each sample, a
two-point Hermite bound is far tighter than a secant bound.
*Gated on A13.*

---

# 3. Analytic root counting

Certificate: a contour integral counts the roots inside, exactly and with multiplicity. This
is the only family that answers the completeness question DIRECTLY rather than by sampling.

**A11. Argument principle on `det A_B(lam)`.** With a square collocation
(`n_bdry = n_cols`), `det A_B(lam)` is ANALYTIC in `lam`, and
`(1/2*pi*i) * oint d(log det)` counts its zeros.
*Known objection:* the determinant method is numerically bad -- that is exactly why
Betcke-Trefethen introduced the subspace angle, and why this project uses the GSVD.
*But the falsifiable claim is narrower and is not the same claim:* **counting needs only
`arg det`, not `|det|`.** The magnitude spans hundreds of orders and is hopeless; the phase
may survive. If it does, we get an exact count in `[a, b]` -- which is the certificate every
other family lacks.
*Test:* track `arg det A_B` along the real axis on the L-shape and compare the phase winding
against the known eigenvalue count. **RUN BELOW.**

**A12. Beyn / Sakurai-Sugiura contour integration.** The standard tool for "all eigenvalues
inside a contour, with multiplicity", for nonlinear eigenproblems `T(lam)v = 0`.
*Obstacle:* MPS is rectangular and least-squares; `T(lam)` is not square, so the moment
matrices are not defined. A square reformulation exists only if the collocation is square,
which is A11's setting and gives up least-squares stability.
*Falsifiable:* on a square MPS collocation, Beyn's first two moments recover the eigenvalues
in a disk to `prec` and their multiplicities exactly.

**A13. Inertia / Sylvester count.** `N(lam)` exactly, by counting negative eigenvalues of a
shifted Hermitian pencil.
*Assessment:* FEM and BIM have this; MPS does not, because there is no Hermitian matrix whose
inertia counts Laplacian eigenvalues below `lam`. **REJECT** -- but record it, because it is
the single feature whose absence makes this problem hard, and a hybrid that borrows a coarse
FEM count purely as a COUNTING oracle is a real option (A24).

---

# 4. Derivative-based

**A14. Exact `dsigma/dlam` from existing primitives.** Every particular solution here is
`psi(x) = Psi(k*x)` -- plane waves `exp(i*k*d.x)` and Fourier-Bessel `J_nu(k*r)e^{i*nu*t}`
both. So

    d(psi)/dk = (x . grad psi) / k

exactly, and `x . grad psi` is the RELLICH MULTIPLIER, which the boundary-integral layer is
already planned around. Then `dsigma_1/dlam` follows by first-order perturbation of the GSVD.
*Falsifiable:* the analytic derivative matches a central finite difference to the accuracy the
finite difference deserves. **RUN BELOW.**
*Why it matters:* it makes A10, A15 and A16 available, and it costs one gradient evaluation
rather than two extra tension evaluations.

**A15. Sign-of-derivative bracketing.** With `dsigma/dlam` at every sample, a bracket is a
sign change -- no extra evaluations, and no reliance on three-point local-minimum tests that
fail in the flat bottom.
*Falsifiable:* sign-change bracketing finds every dip the three-point rule finds, at strictly
coarser stride.

**A16. Curvature for `delta_res`.** The second derivative at the minimum gives `delta_res`
directly rather than through a parabola fit over three points.

---

# 5. Structure of the branch family

**A17. Track all `sigma_j` as smooth curves.** `sigma_1` is `min_j` of branches and has kinks
at crossings; the individual branches are smooth. Continuation of the branch family turns a
non-smooth minimization into several smooth ones.
*Falsifiable:* branch `j` is smooth (no crossings within a branch) over an interval where
`sigma_1` has several kinks, so the number of smooth pieces is far smaller than the number of
samples.

**A18. `sigma_2` as a between-dip certificate.** Between two eigenvalues `sigma_1` is the
meeting of two cones; `sigma_2` is bounded below there. A lower bound on `sigma_2` over an
interval may certify "no third eigenvalue hiding here."
*Falsifiable:* `min sigma_2` over an inter-dip interval exceeds `tau` by a margin that grows
with the local gap.

**A19. Eigenvector continuation.** Track `c(lam)` rather than `sigma(lam)`. The coefficient
vector swings as `lam` crosses an eigenvalue, and the swing is a large O(1) signal where the
tension change may be small.
*Falsifiable:* the angle between `c(lam)` and `c(lam + h)` peaks at eigenvalues and is a
more coarsely-samplable indicator than `sigma_1`'s dip. **RUN BELOW** -- cheap, and it reuses
section 15's machinery.

---

# 6. Model-based and statistical

**A20. Gaussian process with a level-set acquisition.** Evaluations cost 14-100 ms, which is
expensive enough that a model-based placement can pay.
*Falsifiable:* GP-driven placement reaches recall 1.0 in fewer evaluations than A2 on the
roster.
*Concern:* a GP prior is stationary and `sigma_1` is emphatically not -- deep narrow dips on a
flat plateau. Would need the cone model as a mean function, which is A21.

**A21. Parametric cone-mixture fit.** Fit `sigma_1 ~ min_i sqrt(eps_i**2 + c**2 (lam-lam_i)**2)`
over a window, then place the next sample where adding one more cone would be least
detectable.
*Falsifiable:* the fitted mixture predicts held-out samples better than a GP, and its
"where could one more cone hide" criterion is computable in closed form.

**A22. Multistart local minimization with clustering.** Classic global optimization.
*Assessment:* no certificate at all, and the starts have to come from somewhere -- which is
A1. **REJECT as a primary**, keep as a refinement stage.

**A23. Stochastic global search (annealing, evolutionary, CMA-ES).** **REJECT.** No
completeness certificate, and clause (1) is a multiset match. These answer a different question.

---

# 7. Exploiting problem structure

**A24. A coarse FEM counting oracle.** Borrow `N(lam)` from a cheap low-order FEM inertia
count, use it as the completeness certificate, and keep MPS for accuracy.
*Falsifiable:* a FEM coarse enough to be cheap relative to one MPS scan still counts
eigenvalues in `[a, b]` exactly.
*Attraction:* it supplies precisely the missing A13, and the Weyl audit is a weak surrogate
for it -- Weyl is blind at the top of the range and needs five distinct levels; inertia is
exact and local.

**A25. Symmetry decomposition into irreps.** For a DECLARED symmetry, solve each irrep
separately: each has a simple spectrum, and multiplicity becomes bookkeeping across irreps
rather than a numerical test.
*Falsifiable:* on `plus_shape` (D4) the two measured doubles split into distinct irreps and
each irrep's spectrum is simple, removing the (C1)/(C2) test entirely for those levels.
*Constraint:* declared only, never detected -- `mps_symmetry.md` section 2.

**A26. Two-level rank.** Low rank for the scan (smooth, wide dips), high for refinement
(accurate floor), one factorization.
*Falsifiable:* the low-rank scan has strictly fewer spurious minima at equal stride, and the
high-rank refinement locates strictly better, with no second assembly.
*Measured already:* rank 111 locates `lam_1` to 3.2e-12 against 1.4e-10 at rank 61 -- section 6.

**A27. Basis-adaptive escalation.** When the audit says the basis is binding
(`gap < 20*eps/c`), raise the basis rather than the sampling.
*Falsifiable:* `rect81`'s deficit falls below 0.5 when the basis is enriched, and does not
fall at any stride. **Half-measured already:** 4094 evaluations did not fix it -- section 21.

---

# 8. Triage

    tier   approaches                          why
    ----   ---------------------------------   ------------------------------------------
    TEST   A2, A6, A7, A11, A14, A19           cheap, and each falsifies something load-bearing
    KEEP   A3, A4, A5, A8, A15, A24, A25,      plausible, needs the tested ones first
           A26, A27
    LATER  A12, A17, A18, A20, A21             real but expensive to evaluate
    REJECT A9, A13, A22, A23                   reasons recorded above

The six in TEST are chosen because each one, if it fails, removes a whole family:

* **A2** -- if stratification does not beat uniform, no sampling schedule will, and the answer
  has to come from the certificate side.
* **A6** -- if pilot secants underestimate the true slope, every Lipschitz certificate in the
  design is unsound as built.
* **A7** -- if Lipschitz-free subdivision matches it, the calibration problem disappears.
* **A11** -- if the phase of `det A_B` survives, we get an EXACT count, which nothing else
  offers.
* **A14** -- if the derivative identity holds, four other approaches become available at the
  cost of one gradient.
* **A19** -- if the eigenvector swing is coarser-samplable than the dip, the stride problem of
  section 20.2 is solved outright.


---

# 9. Results, and the shortlist

All six of the TEST tier were run. Full numbers in `landscape_findings.md` section 22.

    A2   Weyl-stratified sampling      FALSIFIED   worse or equal in 22 of 28 comparisons
    A6   Lipschitz calibration         UNSOUND     GWW1 at n_pilot=20 reads 0.672: the
                                                   certificate is FALSE, not merely loose
    A7   DIRECT, Lipschitz-free        STRONG      mean recall 0.95 vs uniform's 0.75,
                                                   and no constant to calibrate
    A11  argument principle on det     FALSIFIED   455 spurious sign flips against 9 roots
    A14  exact dk identity             CONFIRMED   4.65e-09 once the k-dependent column
                                                   scale is corrected for
    A19  eigenvector swing             REINTERPRETED

## What each result removes or opens

**A2 kills the a-priori-schedule family.** `N(lam)` is nearly linear in `lam`, so stratifying
by it is nearly uniform sampling. Local clustering is a FLUCTUATION -- level repulsion,
accidental near-degeneracy -- and Weyl's law describes only the DENSITY. No schedule computed
from Weyl can put samples where they are needed. The answer has to come from the certificate
side, not the schedule side.

**A6 wounds every Lipschitz certificate in the current design.** It is not that `L` is loose;
it is that a pilot can UNDERESTIMATE the true slope, so `search.md` section 6's conditional
claim ("no `x` with `sigma_1(x) <= tau_scan` outside the returned set, GIVEN `L`") is
conditional on something that can be false. Usable only if something else sizes the pilot.

**A11 kills the only exact-count candidate, and more strongly than expected.** The standard
objection to the determinant method is about magnitude. Measured, the conditioning destroys the
determinant's TOPOLOGY too -- the sign is noise. A12 on a square MPS collocation goes with it.

**A14 opens four approaches for one gradient evaluation.** A10 (Hermite envelope), A15 (sign
bracketing, no extra evaluations), A16 (curvature for `delta_res`), and a `c` calibrated
exactly rather than by secants -- which is the direct repair for A6.

**A19 changes meaning.** As proposed -- the swing peaks at eigenvalues -- it is false, recall
0.000 at fine stride. What it actually detects is BRANCH CROSSINGS, which lie between
eigenvalues, and that is a CELL DECOMPOSITION: at four samples per Weyl gap, six of seven
domains get exactly one peak per gap, at 5-7% more cost per sample than a tension evaluation.
Against the 18 per gap that dip detection needs, a 3-4.5x reduction.

## The shortlist

**1. A19b + A7 composed.** The swing at ~4 per gap cuts `[a, b]` into cells holding one
eigenvalue each; DIRECT runs inside each cell against `tau` with no constant to calibrate;
refinement runs inside a bracket known to hold exactly one minimum. This is the first
candidate with a completeness argument that neither A6 (measured unsound) nor the Weyl audit
(blind at the top, needs five distinct levels) can supply.

**2. A14 as infrastructure under both.** Exact `c` per solve repairs A6; sign bracketing
repairs the three-point local-minimum test that fails in the flat bottom.

**3. A3/A5 as the fallback.** The repaired bisection is still 10/12 and still the thing to
beat on evaluation count.

**4. A26, A27 as orthogonal multipliers.** Two-level rank and basis escalation are independent
of the scan and both are already half-measured.

## What the shortlist does NOT yet address

Every result above is about SEPARATING levels that are already distinguishable. The
pair-resolution question -- two eigenvalues closer than the cell decomposition can split -- is
untouched by all six tests, and it is what sections 9.4 and 19 say is the real failure mode.
A19b's denominator is gaps between DETECTED dips, so it cannot report on pairs it never saw.
That is the next thing to test, and the instrument for it is `rect(1+delta, 1)`, which is
already rigged.
