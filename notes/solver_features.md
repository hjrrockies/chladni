# What the eigenvalue solver has to have

Working note. Derived from `docs/design/search.md` section 2 (the operational definition) and
from the measurements in `landscape_findings.md`, especially sections 15-20. Every entry names
the measurement that forces it; anything without one is marked as a judgement call.

The frame: a user sets `prec` and gets eigenvalues to that relative accuracy, or an honest
statement that MPS could not reach it. This document is the list of capabilities that requires.

---

# 1. Functional

**F1. Recall first, and recall is conditional.** Missing an eigenvalue cannot be repaired
downstream; a located root a digit short can. Every claim of completeness is conditional on the
scan's own assumptions and must say so.

**F2. Adaptive subdivision, not a uniform grid.** A pair at separation `d` is resolved once
sampling near it is finer than about `d/3`. A uniform grid pays that everywhere at `n ~ 3(b-a)/d`;
subdivision pays `O(log(1/d))`. Measured 2048 vs 47 evaluations at `d = 3e-03`, and 6e+10 vs 97
at `d = 1e-10`. `d` is not known in advance, so the uniform cost is paid against a pair that may
not exist. -- section 3 of `search.md`.

**F3. A stride that is corrected in the loop.** Six samples per gap loses eigenvalues on four
of seven real domains; full recall needs about 18 per median gap. And the governing quantity is
the MINIMUM LOCAL gap, which runs 0.13 to 0.79 of Weyl's mean -- so no a-priori stride is safe
and the correction has to happen during the solve. -- section 20.2.

**F4. Per-solve calibration of `c`.** 0.45 to 1.73 across domains, stable within one. A fixed
Lipschitz constant is either wrong or wasteful. -- section 2.

**F5. Threshold rank, re-derived at each `lam`.** The numerical rank GROWS with `lam`, so a rank
frozen at one point is unsafe at the low end of a range: 39 columns chosen at `k = 3.95` gives
85 spurious minima at `k = 2.35`, where the correct 31 gives exactly one. Threshold cannot pick
an unsafe rank by construction, at 1.2-1.6x the cost. -- sections 3, 5, 8.

**F6. Two-level rank from ONE factorization.** The scan wants a low rank (smooth, wide dips);
the refinement wants a high one (accurate floor). The L-shape locates `lam_1` to 3.2e-12 at
rank 111 against 1.4e-10 at rank 61, because what limits it is the floor rather than the
roughness. One SVD serves both -- no second basis, no second assembly. -- section 6. NOT BUILT.

**F7. Refine in `lam`.** Fitting `sigma_1**2` over identical physical points, the residual is
smaller in `lam` in 17 of 17 cases across four domains, median 5.1x, and the located vertex is
5.1x more accurate. The SCAN variable is a separate choice. -- section 3b.

**F8. Squared objective.** Minimise `sigma_1**2`, never `sigma_1`: squaring removes the kink
and makes refinement machine-exact. -- `search.md` section 3.

**F9. Multiplicity by (C1) AND (C2).** The ratio gap alone double-counts once `d <~ eps/c`,
reporting multiplicity 2 at a SIMPLE eigenvalue; (C2) rejects it every time because `sigma_2`
there is a monotone wall. Confirmed on a domain with four reentrant corners: on `plus_shape`
the two D4 doubles are found and the Weyl deficit falls 2.79 -> 0.79. -- sections 9.4, 20.3.

**F10. Collapse-versus-merge by the interior Gram angle.** Two roots at the same separation can
be one eigenvalue found twice or two genuine ones; no length threshold decides it. Distinct
pairs read `|cos| <= 0.10`, duplicates `>= 0.948`. Within one `lam` the test is VACUOUS -- the
GSVD makes the interior images orthogonal by construction, measured 4.6e-16 -- so it is only
informative across roots. -- section 15.

**F11. Work scaled to `prec`.** Refinement stops at `max(prec*lam, delta_res)`; the first term
binds whenever the basis is better than the request. Completeness is NOT a precision question
and the stride must not tighten with `prec`. -- `search.md` section 2.6.

**F12. Deterministic output.** The same domain returns the same eigenvalue twice. An unseeded
interior fill gives a 110x run-to-run spread in the located eigenvalue and puts that into the
shape objective as noise. -- section 17.

---

# 2. Diagnostic -- required output, not instrumentation

`S` is unknown on every domain that matters, so these are the only signals that F1 has failed.

**D1. The Weyl deficit, on the UNFOLDED list.** The only completeness check that does not
depend on the search being right. It found the recursion-floor bug that prefix truth did not.
Two hard limits, both measured firing on real domains: run BEFORE multiplicity it reads as a
false alarm on every symmetric domain (2.79 on `plus`, which is really 0.79), and below about
five distinct levels it says nothing at all while a third of the spectrum is missing
(`chevron`, 0.12 at recall 0.667). -- section 20.3.

**D2. `L` and the largest observed secant slope, with their ratio.** A small margin means the
constant was being pushed and the run deserves a rerun. The only signal that the first pass was
too coarse.

**D3. Per root: `delta_res`, and which termination bound was active.** This is what makes
clause (2) enforceable while its constant is uncalibrated.

**D4. The basis-adequacy predicate, per interval.** `gap > 20*eps/c`, from a direct
measurement -- (C1) correct at `d/(eps/c) = 23`, wrong at 0.45 and 0.15. -- section 9.4.

**D5. A flag when `tol_merge` approaches the local resolution limit.** At `prec ~ 4e-09` the
discretization's own splitting of a degenerate pair exceeds `tol_merge` and a true double is
reported as two simple eigenvalues. -- section 19.

**D6. Evaluation count and attic revivals.**

---

# 3. Anti-features -- measured not to work

**A1. No absolute tension threshold, anywhere.** The floor spreads four to five orders WITHIN
one scan on three of seven domains. Thresholds are fractions of the plateau, which is 0.12 to
0.36 across the whole roster. -- section 20.1.

**A2. No `sigma_2`-VALUE gate for cluster detection.** `sigma_2`'s minimum is `c*d/2` only at
the crossing point, which is never a sample, so a grid of spacing `h` sees `c*h/2`. The gate
tests the GRID, not the pair. -- section 11.

**A3. No offset detector expecting early warning.** The `sigma_2` offset relation is real
(1% of `d/2`, confirmed over six decades) but it fires at exactly the same coarseness at which
`sigma_1` first shows two minima. It measures `d` once resolved; it does not detect earlier.
-- sections 9.2, 16.

**A4. No distance-only collapse/merge.** See F10.

**A5. No structured interior fill.** `method="grid"` is 30x-1e4x more accurate as a discrete
inner product AND annihilates the mode `side+1` exactly, to 1e-56, because its points sit on
that mode's nodal lines. The interior block wants NON-DEGENERACY, not accuracy. -- section 18.

**A6. No interior-cubature project.** Accurate interior inner products for eigenfunctions are
hard and the path is dysfunctional. The interior block only has to bound `||u||` away from
zero. Anything that must be ACCURATE belongs on the Rellich boundary integrals.

**A7. No fixed tolerance that expresses resolution or accuracy.** `search.md` section 2.1.

**A8. No auto-swap to a symmetry-reduced solve on detected symmetry.** A tolerance threshold
makes the objective discontinuous in shape, and it fires in the endgame of the optimization.
-- `mps_symmetry.md` section 2.

**A9. Do not design against a narrow dip.** No domain produces one; measured `c` is 0.445 to
1.725 over 19 eigenvalues. The real failure is a CLUSTER. -- section 14.

---

# 4. The interface has to widen

`search(tensions, a, b) -> [(lam, mult)]` is the current harness contract, and `search.md`
defends it: the search sees only the tension function, which is what makes candidates
comparable and lets a later FEM or boundary-integral method reuse it.

**F10 breaks that.** The Gram angle needs the candidate eigenFUNCTIONS, which
`tensions(x)` cannot supply. Two ways out:

1. **Widen the interface to `modes(x) -> (sigma[:], U[:, :])`**, where `U`'s columns are the
   candidate functions sampled on a FIXED common point set. FEM and BIM produce eigenvectors
   too, so this stays method-agnostic and keeps candidates comparable. The fixed point set is
   already what `MPSPencil` has -- `interior` is set at construction and only `k` changes.
2. Move Phase D above the search, into `MPSEigensolver`, as post-processing.

**(1) is better** and it is a small change: the harness's `Case` grows one callable, and
`uniform_grid_search` and the lappy baselines ignore it. (2) makes the collapse decision
un-benchmarkable, which is precisely the decision that most needs benchmarking.

---

# 5. Ranked build order

1. **The stride rule and the in-loop Weyl correction (F3, D1).** The measured defect is here --
   six per gap loses eigenvalues on four of seven domains -- and recall is the headline.
2. **The interface widening (section 4), then F10.** Everything about multiplicity is
   un-benchmarkable until the harness can see eigenfunctions.
3. **`prec`-derived termination (F11)** and the `delta_res` report (D3). This is what makes
   the precision contract real rather than aspirational; nothing currently reads `prec` at all.
4. **Two-level rank (F6).** A pure win already measured, and cheap: one factorization.
5. Basis planning, which bounds all of it and is a separate project.

---

# 6. Open, and honestly unmeasured

* The 18-per-gap stride is a FITTED number on seven domains, not a bound. The governing
  quantity -- the minimum local gap -- is not predictable a priori, so this can only ever be a
  starting point for F3's in-loop correction.
* `C` in the accuracy clause remains uncalibrated. Clause (3) is the workaround.
* The wide bracket for (C2) was taken here as `0.02*lam`, an ad-hoc fraction. It should be
  derived from the refinement's own parabola, and is not.
* Everything above is at low `lam` -- the roster tops out at 90 and mostly at 30-40.
* No domain with a hole, and no curved boundary except `cut_square`, which was dropped for a
  basis-planning reason rather than a curve one.
