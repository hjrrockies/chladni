# Coarse-to-fine bases for the eigensolve: a design question, not yet measured

2026-09-21. Raised while reviewing the penalty-regularization cost (`notes/smooth_filter.md`
sections 9-10). Recorded because it reframes part of that work.

## Two stopgaps considered and not recommended

**A golden-section fallback in `refine`** for rank-flip refusals would probably rescue the
2026-09-21 CI cell: only the flipped end `HI` was rank 86, and golden section compares interior
points only. But it cannot be triggered specifically -- nothing reading the refiner's own values
tells a flipped end from an empty bracket (`notes/session_2026-09-19.md`), so it fires on every
`not_a_minimum`, and 33 of 35 rejections on the macOS grids are benign. Worse, it trusts the
single lowest value, so a flip at an INTERIOR point of an empty bracket becomes a false root: it
trades an honest refusal for a silent acceptance. It also misses the silent location error
(2.6e-5 from the root without the hint) and the stage-1 slope-loop declines of section 34. The
session note already argues against refiner heuristics that distrust ends.

**Wall time.** No test measures it. Per evaluation, assembly cached, measured under load, so
trust the ratios over the milliseconds:

    columns   threshold   tikhonov_qr (order 1)   order 2
      126      3.23 ms          1.54x              1.87x
      190      9.14 ms          1.55x              1.92x
      258     16.89 ms          1.96x              2.43x

An earlier draft estimated order 2 at "about 2x" order 1; measured, it is about 1.2x order 1,
and 1.9-2.4x the threshold, the gap widening with columns. Per SOLVE the penalty may break even or win where the
threshold's noise makes false wells (section 7.4: 42k vs 82k evaluations, 38 vs 807 spurious
brackets), and loses where it does not. An end-to-end timed `solve` under both modes, on both
platforms, has not been run.

## The question: most evaluations do not need the high-accuracy basis

* Stage 1 dominates the count -- the N65 solve was 73 search evaluations to 16 refine -- and it
  finds wells by DEPTH (sections 7.1, 7.4: even a frozen rank with a 2.3e-4 floor bracketed all
  11 isolated levels on the 60x window).
* Cost grows steeply with columns (about 3 ms at 126, 9 ms at 190).
* **An over-complete basis is what makes the rank flicker in the first place.** On the L-shape at
  FB80 the threshold rank equals `n_cols`, truncation is inert, and every regularization agrees
  (section 7.5); `SUITE_RTOL`'s comment records 264 spurious minima from a window over-enriched
  at the top. Part of the regularization problem is a symptom of over-provisioned bases.

## Constraints any coarse-first design must meet

1. A level the coarse pass never brackets is lost silently (`TODO.md` section 4; landscape
   section 38). The Weyl audit is the only search-independent completeness check -- make it
   mandatory on the coarse pass.
2. A coarse floor near the merge licence breaks multiplicity ACCOUNTING, in both directions
   (`notes/smooth_filter.md` section 10.1). Under the order-2 rule `tau_2 = C * prec` the floor
   is up to 0.41 C prec, and on the 50*pi^2 triple at `prec = 1e-6`: members LOST (declared 2 of
   3) and one well reported SEVERAL times (5 of 3 -- the middle member three times, two of them
   2e-15 apart, an exact duplicate the merge policy says must merge). It is erratic in C, not
   monotone, so it cannot be tuned around by picking C. The coarse pass should therefore emit
   CLUSTERS under a loose licence and never COUNT multiplicity -- counting belongs to the fine
   pass, whose floor must sit well below its own licence.
3. Brackets must be widened when the basis changes, by the coarse root's displacement --
   `delta_res = DELTA_SAFETY * eps / c` gives it in closed form.
4. Thin per family, never by dropping a family: plane waves alone floor at 3.8e-1 on the
   L-shape against 4.3e-14 with Fourier-Bessel.
5. No hill-climbing on column count (`TODO.md` section 3): FB regresses 10x from 40 to 80 terms
   per corner on the L-shape, and disk enrichment past `n_useful` does nothing.

## Adaptive along lam, not along iterations

`PlaneWave.n_useful_at(k)` gives `n_useful ~ kR + 11 (kR)^(1/3)`, so a basis sized for the top of
a window is over-complete at the bottom, which is where the noise directions come from. Sizing
per sub-window (coarse) and per bracket (fine) keeps each pencil near its resolvable dimension:
section 35's sub-window idea with the BASIS re-sized, not only the rank frozen. Prerequisite:
`Basis` has no composite `n_useful_at` (`TODO.md` line 434).

## Proposed design, to be tested

* Coarse pass: per sub-window, basis sized to `n_useful` at a loose floor (about 1e-6); bracket,
  cluster under a loose licence, Weyl-audit.
* Fine pass: per bracket, basis sized for the target `prec` at that lam; refine and count
  multiplicity only there.

Measurements that would decide it:
* the stage-1 / stage-2 wall-time split on real solves;
* whether coarse bases bracket every roster level, including the L-shape's exact double at 128.3;
* how often clusters need splitting in the fine pass;
* **whether, with bases sized to `n_useful`, the threshold still flickers enough to need the
  penalty at all**. Worth knowing before investing further in order 2.

  **Re-framed 2026-09-23**, when the penalty became the intended default (`TODO.md` section 4):
  the question is no longer whether to adopt one, but how much right-sized bases reduce what the
  default has to survive. They bear on the same gates either way -- fewer noise directions means
  a smaller `tau` is safe, which is exactly what the tight-`prec` rungs need.