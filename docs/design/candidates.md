# Candidate algorithms: the API they present, and how they are compared

A ledger for the eigenvalue-search *instrument*, not for any one search. `search.md` records
what one design intended and `CHLADNI_V1.md` records what a second one measured; this file
records the seams both of them plug into, so that a third can be scored against them without a
new instrument being built for it.

The reason this exists as its own document: reducing GSVD calls without losing robustness is a
standing goal, so candidates will keep arriving. `notes/landscape_findings.md` section 29
measured five of them and the conclusion turned twice on the *instrument* rather than on the
algorithms -- once on a flat evaluation cap (section 29.2) and once on a scorer that rewarded
the wrong behaviour (section 27.2). An instrument that changes between candidates cannot
support a comparison at all.

---

# 1. The four stages

Many searches decompose as **bracket -> refine -> multiplicity**. Each stage is scored on its
own as well as through the composed pipeline, because a stage score and a pipeline score fail
differently and one number for both hides which happened: section 27.2's defect was exactly
that recall-with-multiplicity is one number measuring two things, so a method that never split
a pair but claimed multiplicity 2 scored perfectly.

    # stage 1 -- BRACKET
    brackets(tensions, a, b, *, prec, modes=None) -> [(lo, hi), ...] | Declined

    # stage 2 -- REFINE
    refine(tensions, lo, hi, *, prec) -> RefineResult

    # stage 3 -- MULTIPLICITY
    multiplicity(tensions, x, *, wide_bracket, prec) -> int

    # stage 4 -- PIPELINE
    solve(tensions, a, b, *, prec, modes=None) -> [(x, mult), ...]

A decomposed candidate registers stages 1-3 and gets stage 4 by composition. A candidate that
does not decompose registers stage 4 alone. **Stage 4 is the only mandatory seam.**

`tensions(x)` returns the ascending vector `sigma_1 <= sigma_2 <= ...`, and reading `sigma_2` is
free -- one evaluation returns the whole vector, which is why the isolation gate can exist at
all. `modes(x)` returns the normalized candidate eigenfunction on a fixed shared point set,
supplied only by cases that have one; a candidate that does not want it ignores it.

## Stage 1 must COMMIT

The returned intervals are the ones the method *believes* hold an eigenvalue, not a subdivision
of `[a, b]`. Measured on `B:d=5.9e-03` over `[10, 88]`: a partitioning method tiles the interval
(total width 78.0 of 78.0) and scores coverage 4/4 for free, while a committing one covers 5.9
of 78.0 and has to earn it. Without this rule the two are not comparable, and a partition
returned *without evaluating anything* satisfies coverage, purity and a search-only budget --
which is why the cost model of section 4 is not optional.

## Stage 3 spends no evaluations

Multiplicity is read from the evaluation cache. It also needs a **wide** bracket -- the pair of
evaluated points either side of the root from before the bracket collapsed -- because the final
bracket is far too tight for the higher branches to have moved, and reading them there reads
noise.

---

# 2. The variable, and why the seam is `k` here

**`tensions`' argument is named `x` and its meaning is the case's business**; truth is supplied
in the same variable. The harness never assumes one.

The canonical setup is `x = k = sqrt(lam)`, and `CHLADNI_V1.md` section 2.1 is why: the wall
slope satisfies `c(lam) = C/sqrt(lam)`, so `dsigma/du` is constant under `u = lam**p` for
`p = 1/2` and no other power. Measured slope spread over 15 levels is **5% in `k` against 57%
in `lam`**, and that change of variable was worth 1.8x of a 2.6x total improvement -- more than
every algorithmic change combined.

This looks like it contradicts `docs/conventions.md`, which says "`mps` and above is `lam`" and
that above that seam "`k` buys nothing". It does not, for two reasons, and both are worth
stating because the apparent conflict will come up again:

* The conventions rule is a **naming discipline plus a placement rule for the conversion**: any
  function may work in whichever variable suits it provided its argument is named for that
  variable, and the `lam -> k` conversion falls at the `mps`/`helmholtz` boundary. `Objective`
  converts in `from_pencil`, at that boundary, and calls the pencil with `lam`. Nothing above
  `mps` is handed a `k` it did not ask for.
* The conventions entry's argument against `k` is specifically about eigenvalue **spacing**
  being asymptotically uniform in `lam`, "which is the variable a minimizer's step control
  should see". That is right, and is a separate claim from the slope: `search.md` measured a
  quadratic fit to `sigma_1**2` to be better in `lam` in **17 of 17 cases**, median 5.1x. So the
  two stages want two variables, and they get them -- **cover in `k`, refine in `lam`** -- which
  is not a compromise but what both measurements say.

Brackets are returned in `lam`, widened by ~4 ulps: the round trip is not exact, and a bracket
correctly fencing `k*` can fail to fence `lam* = k*^2` after squaring because an endpoint rounds
the wrong way.

---

# 3. Comparing methods that do not decompose

Three mechanisms, none of which asks a method to restructure itself.

**Uniform cost accounting.** Every candidate is driven through one `Objective`, so cost is
GSVD-level evaluations -- cache *misses* -- in every case, and nothing else is counted. The
budget is **per eigenvalue, never flat**: section 29.2 measured a flat 1500-evaluation cap
turning a 38-evaluation miss into the recorded claim "nobody handles family E", because family B
holds 4 eigenvalues (375 each) and family E holds 29 (52 each).

**Induced brackets.** For each root a monolithic method reports, take the tightest pair of
*sampled* cache points straddling it. That yields a bracket set for a method that never produced
one, so coverage and purity read on it directly. It is computed by the harness from
`Objective.samples`, identically for every method, and it is a derived reading rather than a
claim the method made -- so it is reported in its own column and never mixed with a stage-1
score.

**The oracle/autonomous two-column protocol.** Every candidate is scored twice: hand-tuned per
case, and self-configuring. This is what exposed `lappy_repaired` at 6/12 oracle against
**0/12** autonomous -- almost all of its capability was per-case tuning, and the method that
looked strongest tuned would have shipped as the weakest.

A stage score is never compared against a pipeline score, and no candidate may read truth
outside its oracle column.

---

# 4. Cost is measured, not charged flat

A flat per-bracket charge for the refinement a bracket hands on is wrong in both directions --
3 evaluations for a bracket of width 1e-12, 12 for width 20, and 2-7 to *reject* an empty one.
Measured against a flat 10 the model overcharged tight brackets threefold, undercharged spurious
ones by the same factor, and **reversed the answer to "should the search refine or hand off?"
twice** (`CHLADNI_V1.md` section 3.4).

So a bracket set is charged by running the real refiner on the brackets it returned, against the
same `Objective`, where cache hits are free. Only the surrogate tier, which has no pencil, uses
a fitted `predicted_refine_cost`.

---

# 5. Declining is a third outcome

A method may assume the pencil is well behaved at the scales its precision makes relevant,
detect that it is not, and stop. That is neither a pass nor a failure, and it is only honest if
reported: **the decline rate is quoted beside the pass rate over the same case set, always**. A
method that declines whenever it is about to be wrong has a perfect record over what it accepted
and has demonstrated nothing.

`Declined` is therefore a value a stage returns, never an exception, and it carries the brackets
found so far so they can still be inspected.

---

# 6. What a candidate must not do

* Read truth outside its oracle column.
* Evaluate the objective except through the `Objective` it was handed -- an uncounted evaluation
  is an unmeasured cost.
* Use `expected_count` for anything but **sizing**. It comes from Weyl's law, which says nothing
  about *where* anything is and must not be used as if it did.
* Assume the tension floor is any particular number. It varies **five orders within a single
  scan** on a corner domain, structurally, so no absolute threshold on `sigma` can serve as a
  test; thresholds are fractions of the measured plateau.
