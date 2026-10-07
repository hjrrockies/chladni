# The douse acceptance benchmark: first scoreboard (2026-10-07)

*Built overnight 2026-10-06/07 on branch `overnight/2026-10-07`. The benchmark's case list, its
reference spectra, and a first scoring of `lappy` against a chladni stand-in. Not yet the douse
backend seam that `TODO.md` section 0 asks for; this is the scoring core that seam will call.*

## Pipeline

| step | script | output | runs under |
|---|---|---|---|
| 1. cases | `extract_cases.py` | `cases.json`: 41 Tier 1/2 shapes from `docs/design/douse_failure_cases.md` as exact vertices, with frozen-plan shapes, lappy's recorded output and any claimed truth | douse's venv (read-only on douse) |
| 2. references | `make_references.py` (+ `fem.py`) | `references.json`/`.jsonl`: per level, value, multiplicity, bound, status | chladni `.venv` with the `bench` extra |
| 3. lappy | `lappy_runner.py` | `lappy_runs.json`: douse's cold solve, fresh and frozen plan | douse's venv |
| 4. score | `scoreboard.py` | `scoreboard.json`, `scoreboard_table.md` | chladni `.venv` |

Wall time: references ~25 min on 8 processes, lappy ~15 min on 3, scoreboard ~10 min.

## References: how far to trust them

Each shape is solved by three chladni runs on different bases (Fourier-Bessel at every corner at
1.5x and 2.0x lappy's per-corner sizing; 1.2x plus an offset source layer) at `prec = 1e-12`, and
by P2 FEM at two refinements for COUNTS. A level is `certified` when two runs agree on its
multiplicity without a count flag and FEM agrees on the count around it; `fem_resolved` when FEM
and one run agree; else `provisional`. 417 levels: **322 certified, 44 fem_resolved, 51
provisional**.

* **Against independent truth, the trusted levels agree to <= 2.4e-13, multiplicities included**
  -- closed forms (two rectangles, the kink pentagon's rectangle limit) and lappy's certified
  tables (regular 5-, 6-, 8-, 12-, 16-gons), compared index by index.
* **Their certified bounds are much looser than that**: Moler-Payne reports 1e-6 .. 3e-5 relative
  on these polygons where the runs agree to ~1e-8 and the closed forms to 1e-13. The references
  are good for counts, multiplicities and ~1e-8 values; they cannot yet score a 1e-13 tier.
* **The catalogue's T1.3 "consensus" reference is missing an eigenvalue**: 916.0816 lies between
  its 887.47 and 1020.40, found by all three runs, FEM 917.39 at refine 6 (an upper bound).
* Weak cases: the regular 12- and 16-gons (chladni's runs find 4 and 2 of FEM's 8 levels in the
  window; 1 trusted each), T2.6-N3k2 (0 trusted), T2.8-N6s1 (3 of 9).

## Scoreboard

`scoreboard_table.md` is the full table. Matching is one-to-one within 1e-6 relative against
the trusted reference copies; a case is `unscored` when its first levels are not trusted.

| solver | scored | ok | wrong, silent | wrong, count-flagged | exception |
|---|---|---|---|---|---|
| lappy, fresh plan, `prec = 1e-13` | 28 | 19 | **9** | -- | 0 |
| lappy, frozen plan (where stored) | 18 | 5 | **12** | -- | 1 (T2.8-kink, `IndexError` in `realize`) |
| chladni stand-in, `prec = 1e-8` | 33 | 23 | **5** | 5 | 0 |
| chladni stand-in, `prec = 1e-13` | 33 | 22 | **7** | 4 | 0 |

"Silent" for chladni means no COUNT flag (`count_uncertain`, `grouping_ambiguous`) on the
result. Most of chladni's silent rows do carry unmet levels (`scoreboard_table.md` marks them
"silent (unmet only)"); only three (`1e-8`: T1.4-oct, ngon5, ngon8) have no flag of any kind.

**What this does and does not say.**

* lappy's catalogued failures REPRODUCE from the stored shapes: T1.1 exactly (the values in the
  catalogue to 4 decimals), T2.3's duplicated `lam_1` on the frozen plan, T2.2's spurious triple,
  T2.5's doubled `lam_1`, the T2.8 `IndexError`. lappy's frozen-plan path is wrong on 12 of 18.
* **The chladni stand-in is NOT a planner and is not what chladni would ship.** It is one fixed,
  unplanned basis per shape (1.0x lappy's per-corner sizing, FB only), smaller than any reference
  run. On it chladni is right where lappy failed on most Tier 2 shapes (T1.1, T2.2 partly, T2.3,
  T2.5, T2.6, T2.7), and honest about precision (most "ok" rows report every level unmet: the
  values are right, the certified bound says it cannot vouch for `prec`).
* **chladni's silent failures are mostly in DEGENERATE spectra**: 8 of the 12 silent rows are
  regular polygons and rectangles (T1.4, T1.5, ngon5/8) -- drops and multiplicity errors at exact
  doubles, the class of `notes/double_10pi2.py` -- and T1.2 is a near-double (gap 8.9e-10). The
  other three are single drops on T2.2 (both `prec`) and T2.6-N5k4-ref3 (1e-13), not traced. douse
  drives shapes into degeneracy, so this is the class that matters most.
* Not measured: wall time is recorded per run (`scoreboard.json`) but not compared, because the
  stand-in basis is not what a planner would use; tracking (`warm=`), which douse uses most of the
  time; the douse-side guards (Weyl, spurious, multiplicity re-check), which would catch some
  lappy rows.
