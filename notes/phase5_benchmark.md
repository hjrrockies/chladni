# Phase 5: the black-box path on the douse benchmark (2026-10-12)

Script: `notes/phase5_benchmark.py`; data in `notes/phase5_benchmark.json`.
`Eigenproblem(dom).solve(n, prec)` on all 35 scored douse benchmark cases, with NOTHING about the
basis supplied. `n` is each case's trusted reference count.

Each solve is scored against the references: by resolution groups (`notes/phase3_rescore.py`),
and by whether any MET level sits further than `prec*lam` from a TIGHT reference value.

| prec | CLEAN (right, no flags) | FLAGGED-OK (right, flagged) | FLAGGED-WRONG (honest) | SILENT | rung 0 / 1 / 2 | wall: median / 90th / max |
|---|---|---|---|---|---|---|
| 1e-10 | **30** | 3 | 1 (T2.2) | 1* (T2.8-N5s2) | 18 / 14 / 3 | 79 s / 212 s / 992 s |
| 1e-8 | **32** | 3 | 0 | **0** | 25 / 8 / 2 | 17 s / 131 s / 1717 s |

(Wall time covers planning and solving, 8 processes on one machine. The ladder: rung 0 is
Fourier-Bessel at every corner; rungs 1-2 are Fourier-Bessel at the singular corners plus
boundary-offset sources; see `chladni.plan`.)

- **At 1e-10, 33 of 35 shapes come back right with no basis supplied.** Before Phase 5 the best
  single recipe passed 24 of 35.
  - The hybrid rungs carry it: 17 of the 35 needed rung 1 or 2.
  - The previously hopeless group A pentagons and the thin triangle T2.6-N3k4 now pass.
- **T2.2 is FLAGGED, not silent.** Every level is reported unmet (margin 0.001), and the planner
  says no rung met the request.
- **\* T2.8-N5s2: suspect REFERENCE, not an established solver error.** One met level, at
  179.30225112, is 7.2e-10 relative from the reference value 179.302250990678. But:
  - That reference has status `fem_resolved`: ONE unflagged run (recipe B) plus a finite-element
    count. Runs A and C disagreed with it at 179.30225112.
  - Recipe B re-run on current code does NOT reproduce it: it gives 179.3022510764, with an
    uncertified Moler-Payne bound of 2.5e-5.
  - Every current basis agrees at 179.3022511196 to ~1e-12: corner-only recipes A and C, and the
    planner's hybrid with sources (520 columns), whose Moler-Payne bound is 4.7e-11.
  - So the earlier "biased well" reading of T2.8 (`notes/admission_bound.md`) is in doubt. The
    reference value came from older code and does not reproduce.
  - Not settled here. Rebuilding T2.8's reference on current code, e.g. two hybrid runs at 1e-12,
    would settle it.
- **At 1e-8: 35 of 35 right, none silent,** at a median of 17 s.
- **Cost.** Planning pays for every rung tried. At 1e-10 the median is 79 s, and the worst case
  (T2.2, which climbs all three rungs and still fails) is 992 s. douse plans once and re-uses the
  plan (`Plan.realize`), so the steady-state cost per iterate is one solve at the chosen rung.

## Not established

- macOS only.
- Rung 2 was not measured on its own before shipping. It was reached in 3 cases at 1e-10.
- The starved-basis guard (`confirm=True`) was off throughout.
- T2.8's reference, as above.
