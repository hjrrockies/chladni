# Reference spectra, bounds, and asymptotics

Three facilities that answer questions about a spectrum without solving for it, plus the
closed-form eigenfunctions that let a solver be checked against something it did not produce.

    chladni.reference   eigenvalues and eigenfunctions that are known
    chladni.bounds      rigorous inequalities
    chladni.asymp       Weyl asymptotics

All three are public API, not test scaffolding. That is the point of the exercise: `chladni`
had private closed forms for the rectangle, disk, sector and equilateral triangle sitting in
`tests/conftest.py` where no user could reach them, and `eigensolver.weyl_window` had a
hard-coded first zero of `J_0` with a Faber-Krahn bound and a Weyl inversion inlined beside
it. Both are now one call into these modules.

Ported from `lappy`'s `reference.py`, `bounds.py` and `asymp.py`, with the changes recorded
below.

---

# 1. Accuracy is data, not prose

A reference value is only usable if you know how far to trust it, and `lappy`'s tables vary
from 14 digits to under 3. Its docstrings say so, carefully and at length -- but a docstring
cannot be read by the test that needs to pick a tolerance, so in practice the tolerance gets
guessed and the caveat gets lost.

So every reference function carries its provenance as attributes:

    fn.digits    the honest MINIMUM accurate decimal digits over the whole table
    fn.source    where the numbers came from
    fn.notes     per-parameter, per-index caveats: {params: {index: message}}

`notes` uses `None` as the index key for a caveat covering a whole parameter set. A caller
that wants a tolerance writes `10.0 ** -fn.digits`, and one that wants to know whether a
particular entry is trustworthy looks in `fn.notes`.

`catalog()` returns every reference function with its digits and source, so
`chladni.reference.catalog()` answers "what is available and how good is it" in one call.

This follows the rule `conventions.md` already sets one layer down: **a reported precision is
never a claim contradicted elsewhere in the same object.** `Domain.achievable_prec` and
`Sampling.accuracy` are the same idea.

## Per-parameter accuracy

`digits` alone is the worst case over a whole table, which punishes the good parameter sets
for the company they keep: `reg_ngon` is 10.2 digits at N=5,6,7 and 2.8 at N=8, and reporting
2.8 for all four makes three usable tables look unusable. So `digits_by` gives the accuracy
per parameter set and `digits_for(key)` reads it, while `digits` stays the worst case for a
caller who ignores the refinement. An unknown key falls back to the worst case, which is the
safe direction.

## Three values that were wrong, or needlessly inexact

**The disputed ellipse value is CORRECTED here.** lappy ships
`ellipse_eigs(k, a=2)[0] = 3.566726599853406` and documents it as disputed:

    lappy's table                  3.566726599853406
    lappy's own solves, 3 bases    3.566726602928859 / 862 / 861

The three agree with each other to 7.5e-16 and differ from the table by 8.62e-10 relative,
while `lam_2..4` agree to 2.7e-13 - 4.4e-12, so it is that one entry and not the pipeline
behind it. Corroborated by the MPS tension at the tabulated value sitting at 8.46e-10 for
every basis tried -- the signature of a displaced reference rather than a basis limit.

lappy left the value unchanged deliberately. `chladni` takes the corroborated one, because
shipping a number known to be wrong is worse than shipping the better of two that are already
public. The substitution is recorded in `notes`, since silently disagreeing with a published
reference is its own kind of wrong.

**`H_shape` index 6 is exactly `2*pi**2`.** lappy flagged it as "suspiciously close to
2*pi**2, likely a mode that vanishes on the connecting web". It is: the tabulated value
agrees with `2*pi**2` to 2.5e-15, which is float rounding. The guess is confirmed and the
entry costs nothing to make exact.

**`iso_tri(h=1)` is not a table at all.** Base 2 and height 1 puts the vertices at `(-1,0)`,
`(0,1)`, `(1,0)`, which is the right isosceles triangle with legs `sqrt(2)` -- so its
spectrum is `pi**2 (m**2+n**2)/2` for `m > n >= 1`, in closed form. lappy's ten tabulated
values agree to 3e-15 and are superseded. That parameter set goes from 13.0 digits to 15.6.

## What remains inexact, and is not wrong

Nothing else is incorrect; the rest is honestly-labelled low precision, and fixing it would
need re-solves:

    reg_ngon N=8, index 3    2.8 digits, an under-resolved mode
    chevron (2,3), (2,4)     3-4 digits, corners near 11 degrees
    chevron (1,1.5)          5-7 digits
    cut_square r=0.25        6.4 digits
    ellipse a=3, a=4         6.7-7 digits, not re-verified
    iso_tri h=16             7.8 digits, not re-verified

All are approximately right -- usable to answer "is this the same eigenvalue" and not much
more -- and all report their own accuracy through `digits_for`.

## The recipe, not just the digit count

A digit count cannot be acted on. It says a value is worth 14 digits without saying what it
would take to get 15, or to check it independently. So every table also carries `settings` --
the basis and collocation it was solved with -- readable as `recipe(fn)`. These come from
lappy's `benchmarks/reference/*.py`, which escalate `n_basis` until every tension is at or
below 1e-12.

It is immediately useful. `L_shape_eigs` was solved by lappy at `n_basis=240`, but the recipe
records what the domain actually needs: a PURE Fourier-Bessel basis at the single reentrant
corner, nothing else. Reproduced here at 1400 boundary and 500 interior points, the tension at
the first five tabulated eigenvalues runs

    40 terms    1e-08
    60 terms    4e-13 to 2e-12      about 12.5 digits
    80 terms    2e-15 to 1e-14      clears the table's 14

which corroborates the table from a second implementation and says exactly what to do to
better it.

`lam_3 = 2*pi**2` reaches 3e-15 by 40 terms on its own, and the reason is worth stating
precisely: it is not that the eigenfunction vanishes at the reentrant corner -- every
Dirichlet eigenfunction does, that is the boundary condition. It is that this one has NO
SINGULAR PART there. With `omega = 3*pi/2` the corner expansion runs in `r**(l*nu)` with
`nu = 2/3`, so the terms go `r**(2l/3)` and are non-smooth unless `l` is a multiple of three;
this mode has zero amplitude on all the others.

Which has a clean explanation, and the simplest statement of it is that `2*pi**2` is just the
UNIT SQUARE'S FIRST EIGENVALUE, `pi**2 (1 + 1)`. `L_shape` is three unit cells on the integer
grid and `H_shape` is seven, so `sin(pi*x) sin(pi*y)` -- which vanishes on the whole grid --
is an exact eigenfunction of each, and it is analytic everywhere. The square's fundamental
mode simply tiles.

So `rect_eigs(1, 1, 1)` and `polyomino_eig(1, 1)` both predict two tabulated entries from a
closed form: a cross-check inside this module between its exact half and its tabulated half,
and one the tables cannot perform for themselves.

It is also the mechanism behind the five-order spread in `eps` measured on the L-shape: on a
domain with a singular corner, the modes that happen to carry no singular amplitude are
resolved far better than their neighbours.

## Bessel zeros

The disk and sector spectra ARE the Bessel zeros, so they have to be more accurate than
anything checked against them. One function, `bessel_zero(nu, n, derivative=False)`, holds
the policy:

* **Integer order goes to scipy**, measured at or below one ulp against mpmath at 50 dps over
  32 cases spanning order 0 to 400 and index 1 to 100 -- worst relative error 1.73e-16 for
  `J` and 1.76e-16 for `J'`, against a machine epsilon of 2.22e-16. There is nothing to gain
  by replacing it.
* **Fractional order goes to mpmath**, where scipy offers nothing and the obvious substitute
  is worse than it looks: a `brentq` scan loses two to three digits at fractional order,
  which put lappy's sector references out by up to 1.4e-13 relative and once made a solver
  appear to violate its own error bound when the reference was the thing that was wrong.

An eigenvalue is the zero SQUARED, so the relative error doubles: about 3.5e-16, or 15.5
digits. At `m = 0` with `derivative=True` the two libraries index differently -- `J_0' = -J_1`
and `J_1(0) = 0`, so mpmath counts `x = 0` and scipy does not. Every integer order goes to
scipy, so the convention is scipy's throughout.

---

# 2. Provenance and the ground-truth rule

`conventions.md` says ground truth is never regenerated from `chladni` output. Nine of these
tables are not closed forms; they are another solver's output. That is a case the rule as
written does not cover, and it is admissible: values from an INDEPENDENT implementation, with
provenance and accuracy recorded as data, are exactly what a reference should be. A ledger
entry in `conventions.md` says so.

What is NOT done here is a cross-check test that re-solves each table with `chladni`. It was
considered and rejected: such a test fails for basis-planning reasons far more often than for
reference reasons, and basis planning is deferred. Agreement between the two implementations
has been spot-checked by hand -- the L-shape's first five reference values are reproduced to
between 0.0 and 1.9e-10 relative -- and that is recorded here rather than automated.

---

# 3. What is new relative to lappy

**Complete spectra in an interval.** `lappy` offers "the first k". Recall testing needs "every
eigenvalue in `[lam_lo, lam_hi]`, guaranteed complete", which is a different question and
cannot be got from the first by truncation. `*_spectrum(lam_lo, lam_hi, ...)` provides it for
the families where enumeration is exact -- rectangle, disk, sector, equilateral triangle,
isosceles right triangle, polyomino.

**Both query shapes return the same thing: a flat ascending array with multiplicity expressed
by REPETITION.** A spectrum is a multiset and that is the data structure for one. An earlier
version returned `[(lam, mult), ...]` from `*_spectrum` while `*_eigs` returned a flat array,
which was two encodings of one object inside a single module.

Unfolding loses nothing PROVIDED a true degeneracy comes out bit-identical, so that two equal
floats mean one doubly-degenerate eigenvalue and two unequal floats mean two eigenvalues. That
holds wherever the arithmetic is symmetric, and it does not hold everywhere:

    eq_tri, iso_right_tri, polyomino, disk      0.0       worst spread within a
    rect with a rational aspect ratio           0.0       mathematically degenerate group
    rect(sqrt(2), 1)                            4.4e-16
    rect(sqrt(3), 1)                            3.6e-16

`L**2` for `L = sqrt(2)` is `2.0000000000000004`, so `m**2/L**2` rounds differently for
different `m` even when `m**2/2 + n**2` is mathematically equal. An earlier version of this
document claimed bit-identity generally, on a measurement taken over rectangles with rational
aspect ratios only.

So `snap(lam, SNAP_RTOL)` runs inside every `*_spectrum`: values within 1e-13 are the same
real number reached by different arithmetic and are replaced by their mean, repeated. That is
float-noise correction in the reference, NOT a policy about what counts as one eigenvalue --
1e-13 sits about 230x above the worst measured noise and 120x below the tightest
near-degeneracy worth resolving, 1.2e-11 on `rect(1 + 1e-11, 1)`.

`collapse(lam, rtol)` is the separate, deliberate question, and it belongs at comparison time
rather than in the truth. The harness's `levels_agree(found, truth, rtol)` collapses both
sides at the same tolerance and asks "up to rtol, did the search recover the values AND the
multiplicities" -- one coarse yes/no beside the fine-grained diagnosis. It also covers what
`snap` cannot: a tabulated spectrum whose degenerate members were solved separately and agree
only to the table's own precision.

It also makes the comparison uniform. A search concludes "an eigenvalue at x with
multiplicity m", so it returns pairs; the harness unfolds both sides and matches multisets,
which lets it separate "you missed an eigenvalue" from "you found the level and undercounted
it" -- different bugs with different fixes, and previously indistinguishable.

**Corner angles from `corners`, not from an `isinstance` gate.** `lappy`'s `weyl_count_poly`
tests `isinstance(domain, Polygon)` and refuses anything else. `chladni.Domain.corners`
carries `omega` for *any* domain, so the three-term count applies to `cut_square` and
`mushroom` too -- domains that have corners without being polygons.

**`collapse(lam, rtol)`** turns a with-multiplicity list into distinct levels and counts, once,
where `lappy` open-codes it per family. **It defaults to EXACT equality**, and that is the
load-bearing choice.

A tolerance here does not describe the spectrum, it imposes a policy on it -- "these two
count as one eigenvalue" -- and a reference that has already made that decision cannot be
used to measure a solver that makes it differently. Ground truth says how many eigenvalues
there are; whether a search may merge two of them is the search's business to declare and the
harness's to score.

Exact equality is enough because a true degeneracy is BIT-identical, not merely close. The
eigenvalue formula is symmetric in the indices, so measured over the unit square and the 2x1
rectangle, every mathematically degenerate group has a relative spread of exactly 0.0.
Meanwhile a near-degenerate pair on `rect(1 + delta, 1)` is 1.2e-11 apart at `delta = 1e-11`,
eleven orders above float noise. There is no overlap between the two cases to arbitrate.

An earlier version of this module defaulted to `rtol = 1e-9`, which merged genuinely distinct
pairs across three orders of magnitude -- exactly the regime the cluster tests exist to probe.
A solver that correctly resolved such a pair would have been scored as reporting a spurious
root. `rtol` remains a parameter for a caller who deliberately wants a coarser view.

---

# 4. What is deliberately not ported

`rect_eig_mult` and `rect_eig_bound_idx` use a hard-coded ABSOLUTE tolerance (`1e-12`) to
decide whether two eigenvalues coincide, which misbehaves for large `lam` or large `L`. Their
job is done here by `collapse(lam, rtol)` with a relative tolerance.

`lappy`'s `brentq` import in `reference.py` is dead. Its `_bessel_zero` docstring is worth
reading and its lesson is kept: fractional-order Bessel zeros come from `mpmath` at extended
precision, not from a `brentq` scan, because the scan lost 2-3 digits at fractional order and
once made a solver look like it had violated its own error bound when the reference was wrong.
Integer-order zeros still come from `scipy`, where it is accurate.

---

# 5. Bounds

    faber_krahn            lam_1 >= pi*j01^2/A          lower, Dirichlet
    szego_weinberger       mu_1  <= pi*j11^2/A          upper, Neumann
    payne_polya_weinberger lam_2/lam_1 <= (j11/j01)^2   upper on the ratio, Dirichlet
    inradius_upper         lam_1 <= j01^2/R^2           upper, Dirichlet
    polya                  lam_k >= 4*pi*k/A            lower, Dirichlet

Each takes either a `Domain` or keyword scalars. `polya` keeps `lappy`'s honesty in its
docstring: proven for tiling domains, conjectured in general. It is the only unproven
statement in the module and it is labelled.

---

# 6. Asymptotics

    weyl_count(lam)        N(lam) ~ (A*lam -/+ P*sqrt(lam)) / (4*pi)      forward
    weyl_est(k)            closed-form inverse of the above               inverted
    weyl_count_poly(lam)   the same plus the corner sum                   forward
    weyl_est_poly(k)       numerically inverted three-term count          inverted
    weyl_deficit(lams)     how many modes the count says are missing      diagnostic
    weyl_count_check(lams) pointwise (i+1) - N(lam_i)                     diagnostic

The corner correction is `sum (pi**2 - omega**2) / (24*pi*omega)` and is independent of `lam`.

**Two sign flips, and they are the easy thing to get wrong.** In the forward count the
perimeter term SUBTRACTS for Dirichlet and adds for Neumann. On inversion it moves to the
other side of the quadratic and the sign flips again, so the Dirichlet inverse carries `+P`.
Both flips carry their comment at the code.

`bc_type` of `'rob'`, or a mixed boundary, raises `NotImplementedError`. There is no curvature
term, so a smooth domain gets only the two-term law; that is a weaker detector rather than a
wrong one.

---

# 7. Eigenfunctions

`*_eigfun(...)` returns `(u, norm2)` where `u(z)` is vectorized over complex `z = x + iy` and
`norm2` is the EXACT squared L2 norm. Un-normalized, because an independently-normalized
eigenfunction is ground truth that a quadrature can be checked against rather than something
a quadrature defines. `normalized_eigfun` divides by `sqrt(norm2)` when that is what is wanted.

`*_eigfun_grad(...)` returns `du_dx + 1j*du_dy` as a single complex array, matching
`conventions.md`'s gradient convention exactly.

**Placement is load-bearing.** Every eigenfunction assumes the placement of the factory it
names, and the norms are wrong if they disagree. The shipped factories match `lappy`'s
placement exactly, which is why the formulas transfer unchanged:

    rect(L, H)            corner at the origin, [0,L] x [0,H]
    iso_right_tri(l)      right angle at the origin, legs on +x and +y
    disk(r)               centred at the origin
    disk_sector(r, th)    apex at the origin, angular extent [0, th]
    polyomino(cells)      on the integer grid, which is what makes it exact
    eq_tri(l)             corner at the origin, base on +x

The polyomino family carries a caveat worth repeating: it is the only multi-reentrant-corner
domain with closed-form eigenfunctions, and a closed-form eigenfunction on a nonconvex domain
is necessarily SMOOTH at the reentrant corners -- which is precisely why the L-shape has none.
It exercises geometry, panel splitting and assembly on several reentrant corners at once with
exact truth, and carries ZERO singular amplitude. Use it as a control, not as a corner test.

---

# 8. The shape library

Ten factories are added, matching `lappy`'s placement so the eigenfunctions transfer:
`iso_tri`, `right_trapezoid`, `parallelogram`, `chevron`, `H_shape`, `ellipse`,
`disk_sector`, `cut_square`, `mushroom`, `stadium`.

Circular arcs use `ArcSegment`, which is closed form, where `lappy` used a general
`ParametricSegment` and paid the adaptive machinery for something with an exact answer. Only
`ellipse` genuinely needs the parametric form.

**One bug is fixed in the port.** `lappy`'s `stadium` builds its third segment without passing
`bc`, so the top straight edge silently ignores the requested boundary condition. Here it does
not, and `tests/test_library.py` pins it.

---

# 9. Tests

| file | what it defends |
|---|---|
| `tests/test_reference.py` | closed forms against independent computation; `*_spectrum` complete and correctly multiplied; `catalog()` covers every function; every table's `digits` and `notes` reachable |
| `tests/test_eigenfunctions.py` | each eigenfunction solves the Helmholtz equation, vanishes on a Dirichlet boundary, and its `norm2` matches quadrature; gradients match finite differences away from a singular apex |
| `tests/test_bounds.py` | every bound holds on the shipped shapes; equality for the disk; BC rejection |
| `tests/test_asymp.py` | round-trip both BCs; the corner correction is `lam`-independent and exactly 0.25 for the unit square; mixed BC raises |
| `tests/test_library.py` | the new factories construct, validate, and have the expected area and corner count |

`tests/conftest.py` loses its private closed forms and imports from `chladni.reference`.
