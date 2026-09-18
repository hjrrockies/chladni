*Status: an audit, not a plan. Written after a night spent rediscovering things this document's
subject already knew. Every row is read from the two sources, not measured, except where a
measurement is named.*

> **RECOVERED AND ANNOTATED 2026-09-18.** This file lived only on
> `wip/v1-promotion-and-multiplicity`, which was never merged -- its work was re-implemented on
> `main` and re-measured -- while `src/chladni/mps/multiplicity.py` cited section 3 as "the
> argument". The citation was dangling; the file is restored for it.
>
> **Sections 1, 3 and 4 describe defects that are now resolved**, and are kept because the
> ARGUMENTS in them are why the resolutions took the shape they did. Each carries an AS-BUILT
> note. Sections 2 and 5 are unaffected and remain the reason to read this: the diameter rule,
> and the process rule the audit records against itself.

# Where `chladni`'s stages 3 and 4 diverge from `bracket_search`

`bracket_search` is the prior repo, one directory over, and it is the reason `CHLADNI_V1.md`
exists. Its `SUCCESS.md` is the specification the whole apparatus here was built from, and
`level_score.py` and `bracket_score.py` cite it directly. **The scorers ported its rules
faithfully; stages 3 and 4 did not.** That gap is what this file records.

The stages that transferred cleanly are not listed. Stage 1's exclusion argument, the descent
certificate, the isolation gate and the refiner are all recognisably the same mechanisms, and
`CHLADNI_V1.md` records where they were re-measured.

---

## 1. Merging: permitted there, mandatory here

    bracket_search/search/certify.py
        "the search never merges deliberately -- the merges it performs are the ones it
         could not resolve"

    chladni/mps/search.py  (stage 4)
        "Two located roots within `prec*lam` of each other ARE one level of `S(prec)`"

`SUCCESS.md` states the rule three times, once as a named consequence -- *"Merging is permitted,
never required... Resolving more than was asked for is more information, not an error."*
`bracket_score.py` and `level_score.py` both implement it as a LATTICE, accepting any answer
between one level per eigenvalue and one per level of `S(prec)`.

Stage 4 folds instead, and reads multiplicity once at the fold. That is the origin of the
under-count: once two RESOLVED roots become one representative, the partner's branch is minimal
at the partner rather than at the representative, and (C2) correctly refuses it.

**Not yet fixed here, and relaxing it alone is measured to be a regression** -- see section 3.

> **AS BUILT.** Fixed, and not by relaxing it. `merge="attained"` folds only roots that agree to
> within the accuracy the refinement REACHED, so a fold is the same point found twice rather than
> a policy; and ATTRIBUTION gives each branch the read counts inside exactly one owner, a located
> level or the root. The under-count this section diagnoses -- the partner's branch being minimal
> at the partner rather than at the representative -- is what attribution addresses directly.
> `notes/landscape_findings.md` and `search.md`'s multiplicity contract record the result: 216
> right, 0 wrong over five cluster grids on macOS/Accelerate -- one cell short of that on
> Linux/OpenBLAS, see the contract's platform note.

## 2. Grouping is by DIAMETER, and this was got wrong again

    SUCCESS.md
        "Grouping is by DIAMETER, not by chained gaps... Compare each candidate against the
         group's first member, not its predecessor."

Listed there as a thing "got wrong at least once". It was got wrong again while auditing this,
in a rewrite of the stage-4 prefilter, and the resulting numbers were reported as findings about
the solver before the rewrite had been checked against the baseline. The rewrite never reached a
commit. The rule stands; the lesson is in section 5.

## 3. Multiplicity: a coincidence test there, a ratio gap here

The sharpest divergence, and the one the open defect sits on.

| | `bracket_search` | `chladni` |
|---|---|---|
| what is counted | branches AT THE FLOOR with `sigma_1` | branches below the largest ratio gap |
| threshold | `floor_ratio = 1.0` -- numerically coincident | `GAP_RATIO = 10.0` |
| filter | none needed | `PROMINENCE = 2.0` over a bracket `WIDE_FACTOR = 2` licences wide |
| where it is read | at the COLLAPSED point, "where the branches have separated by then" | at the located root, over a wide bracket |

`isolate.py` chose `floor_ratio = 1.0` deliberately, and says so: it *"effectively disables the
multiplicity reading, and it is deliberate"*. Loosening it to 2.0 saved 40% on exactly degenerate
cases and lost eigenvalues elsewhere, with a failure count that is **not monotone** in the
threshold -- 0, 1, 0, 1, 2 at 1.0, 1.05, 1.2, 1.5, 2.0 -- and the conclusion drawn is that *"a
parameter whose safety alternates as it is tightened has no safe setting, only settings that pass
the cases tried."*

**`chladni` ported that lesson into the isolation gate and not into the multiplicity stage.**
`bracket.py`'s `isolation` docstring reproduces the non-monotonicity argument almost verbatim and
declines to read multiplicity there. Stage 3 then reads it anyway, with a threshold ten times
looser than the one that argument rejected, and compensates with a prominence filter over a wide
bracket -- and it is that bracket's reach which lands on neighbouring levels.

> **AS BUILT.** The ratio gap is no longer the default. `rule="reach"` counts `mu*(B)` -- the
> eigenvalues INSIDE the emitted bracket -- by comparing a distance `sigma_j/c` against the
> bracket's reach, which is the comparison the question actually calls for: a ratio on `sigma`
> measures in units of the floor while the bracket is in units of `prec*lam`, and no threshold on
> the first can answer about the second. "gap" is kept reachable as the ablation. This section is
> why.

## 4. The composed stage: a cross-check that has no counterpart

`_collapse_ambiguous` exists only here. It chains located levels within `WIDE_FACTOR` licences,
compares the sum of their counts against one count taken over the whole neighbourhood, and on
disagreement replaces the fine grouping with the spanning count.

**Measured by intervention**: the band of separations where the declared total is wrong tracks
`WIDE_FACTOR` proportionally. At `prec = 1e-8` on the `65*pi^2` cluster, the failing bands are

    WIDE_FACTOR    1.0    2.0    4.0    8.0
    failing at     0.3-1.0  0.8-2.0  2.0-4.0  4.0-6.0   licences

so the radius does not merely coincide with the failure -- it sets it. Moving the radius moves
the band by the same factor.

It cannot simply be removed. Over the 78 corpus cases with complete truth:

    merge      collapse | correct  wrong  over  under  flagged  false+
    licence    True     |    62      14     0     14      13      11     <- shipped
    licence    False    |    59      17    12      5       3       1
    attained   True     |    63      13     0     13      12      23
    attained   False    |    46      30    29      1       0       0

Retiring it trades 14 under-counts for 12 over-counts, because it is compensating for section 3.
**Two defects, one masking the other**, and neither is fixed by moving a constant.

> **AS BUILT.** Exactly right, and the resolution was to fix section 3's defect first. With
> `rule="reach"` and attained merge in place, `_collapse_ambiguous` is net-NEGATIVE and is off by
> default (`collapse=False`); `tests/test_cluster_totals.py` keeps the evidence under the
> configuration this table describes. The "shipped" row above -- `licence`/`True` -- is no longer
> shipped. Note the corpus here is 78 cases scored the way section 5 criticises; the committed
> grids that replaced it score 251 cells with the precondition checked before scoring.

## 5. The process rule this file exists to record

`SUCCESS.md` says how to report:

> **"State capability, not score."** ... "The natural summary is a capability curve -- the minimum
> cost at which a method passes, as a function of separation and `prec`, censored at a stated cap."

and

> **"Separate tuned from shipped."** "If you tune per case using knowledge of the answer, that is
> an upper bound on capability... A method that would ship must also be scored at one fixed
> setting chosen before seeing per-case results."

The audit that produced this file did neither. It optimised a scalar -- wrong cells out of 78 --
over a family of ADVERSARIAL DIALS, weighting a `prec = 1e-14, band = 0.5` construction equally
with the disk, and it moved defaults between runs. The one result in it that is worth anything is
the `WIDE_FACTOR` table in section 4, and that is worth something precisely because it is an
INTERVENTION: a mechanism was moved and the predicted consequence was observed, rather than a
number being compared against another number.

A hypothesis was also killed cheaply and is worth recording as killed. `SUCCESS.md` describes a
band `prec < d/x < limit` -- "must split, and cannot" -- that widens as `prec` tightens, and the
failures here were guessed to be it. They are not: measured in licence units the failing band is
FIXED across `prec` from 1e-6 to 1e-14, so the separations that fail are proportional to `prec`
and the cause is algorithmic rather than a resolution limit. That is what pointed at
`WIDE_FACTOR`.
