# The suite's ulp sensitivity is mostly its pencil's truncation (2026-10-11)

Question, from `notes/regularization_1e10.md` follow-up 2: the rectangle suite builds every pencil
at `SUITE_RTOL = sqrt(1e-14 * eps)` ≈ 1.5e-15 (`tests/rect_suite.py`, `FINEST_PREC`), a threshold
sized for 1e-14. Tests then run it at 1e-8 and 1e-10. How much of the `ulp_sensitive` population
comes from that over-fine truncation?

Method:
- `notes/jitter_sweep.py` over the 16 `ulp_sensitive` tests, seeds 0..16 at 4 ulps (seed 0 is
  unjittered), under three values of `SUITE_RTOL`.
- `SUITE_RTOL` is overridden by `notes/suite_rtol_plugin.py` (`PYTEST_ADDOPTS="-p
  suite_rtol_plugin"`, `CHLADNI_SUITE_RTOL`). Nothing in `tests/` changed.
- 15 of the 16 come from modules that import `rect_suite`. `test_floor_multiplicity[A]` does
  not, and passed every seed in every configuration.

| `SUITE_RTOL` | sized for | test × seed failures (of 256) | tests that flip | control (seed 0) |
|---|---|---|---|---|
| 1.5e-15 (shipped) | 1e-14 | **15** | 10 | 16/16 pass |
| 1.5e-12 | 1e-8 | 2 | 2 | 16/16 pass |
| 1.5e-13 | 1e-10 | **0** | 0 | 16/16 pass |

Raw output: `notes/suite_rtol_sweep_{suite,r8,r10}.json`.

**At a threshold sized for the precision the tests actually run at, the population that flips under
ulp jitter is empty.** At 1.5e-13 every test passes every seed, with its contract unchanged. The near-null directions a 1.5e-15 cut keeps are what 4-ulp noise
shakes. That matches the refine gap-root trace ("a pencil truncated at `rtol` 1.5e-15") and the
170π² stage-1 declines.

## What breaks if the suite pencil is resized

The CI tier (`-m "not gate and not robust"`, 1991 run) was run at `SUITE_RTOL = 1.5e-13`:
1985 passed, **3 failed**. All three are the floor PRECONDITION tests, which assert that the suite
pencil's floor reaches the 1e-14 target:
- `test_floor.py::test_the_floor_holds_across_a_WIDE_window_at_every_level`;
- `test_floor.py::test_the_floor_precondition_holds_on_the_crossover_cases[cluster]` (worst 2.3e-14);
- `test_rect_suite.py::test_the_floor_precondition_holds_on_a_representative_case` (worst 3.0e-13).

These are expected: they test the 1e-14 premise, which a 1e-10 pencil does not serve.

## What it suggests (not done)

- Size the suite pencil to the rung under test. Either set `FINEST_PREC = 1e-10` (since the prune,
  no default-tier test runs below it) and restate the three floor preconditions at that target, or
  key the pencil memo on a per-rung `rtol`.
- The evidence modules in `notes/evidence_tests/` that still run `PREC_LADDER_FULL` to 1e-14 would
  keep a 1e-14 pencil of their own.
- The `ulp_sensitive` quarantine would then be re-measured, aiming at the 0 of N target in TODO.
- **This is a test-fixture change, and it moves every suite pencil.** It is for review, not made here.

## Caveats

- One macOS machine. The jitter harness stands in for Linux.
- 16 seeds. 0 of 256 is a strong result, but not a guarantee across CPUs.
- The 1e-8 tests at 1.5e-13 are not exactly "rung-sized" (that would be 1.5e-12, which showed 2
  flips). Both are far better than the shipped 1.5e-15.
