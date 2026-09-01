"""Provenance metadata, and the shared spectrum helpers.

A reference value is only usable if you know how far to trust it, and the tables here run
from 14 digits to under 3. lappy says so in its docstrings, carefully and at length -- but a
docstring cannot be read by the test that has to pick a tolerance, so in practice the
tolerance gets guessed and the caveat gets lost.

So accuracy is DATA. Every reference function carries `digits`, `source` and `notes`, and
`catalog()` lists them. This is the rule `conventions.md` already sets one layer down --
a reported precision is never a claim contradicted elsewhere in the same object -- applied to
values rather than to computations.
"""
from __future__ import annotations

import numpy as np

__all__ = ["exact", "tabulated", "catalog", "recipe", "collapse", "snap",
           "SNAP_RTOL", "REGISTRY"]

#: Decimal digits float64 carries. `exact` claims this rather than `inf`: a closed form is
#: exact as mathematics and still evaluates in floating point.
FLOAT64_DIGITS = 15.6

#: Every decorated reference function, in definition order.
REGISTRY = []


def _register(fn, digits, source, notes, digits_by=None, settings=None):
    fn.digits = float(digits)
    fn.source = source
    fn.settings = settings
    fn.notes = dict(notes or {})
    fn.digits_by = dict(digits_by or {})
    fn.digits_for = lambda key=None, _f=fn: _f.digits_by.get(key, _f.digits)
    REGISTRY.append(fn)
    return fn


def exact(source):
    """Mark a closed-form reference. `digits` is float64's, not infinity."""
    def deco(fn):
        return _register(fn, FLOAT64_DIGITS, source, {})
    return deco


def tabulated(digits, source, notes=None, digits_by=None, settings=None):
    """Mark a tabulated reference.

    `digits` is the honest MINIMUM over the whole table -- not the best case, and not the
    figure quoted for the easiest parameter set. `notes` maps a parameter key to
    `{index: message}`, with the index `None` for a caveat covering a whole parameter set.
    A caller wanting a tolerance writes `10.0 ** -fn.digits`; one wanting to know whether a
    particular entry is trustworthy looks in `fn.notes`.

    `digits_by` gives the accuracy PER PARAMETER SET, because a single worst-case scalar
    punishes the good sets for the company they keep: `reg_ngon` is 10.2 digits at N=5,6,7
    and 2.8 at N=8, and reporting 2.8 for all of them makes the good tables unusable at their
    real precision. `fn.digits` stays the worst case, so a caller who ignores the refinement
    is still safe; `fn.digits_for(key)` gives the sharp value.

    `settings` records the BASIS AND COLLOCATION the values were solved with, so the
    provenance says how to REPRODUCE a number rather than only how good it is. A digit count
    alone cannot be acted on: it says a value is worth 14 digits without saying what it would
    take to get 15, or to check it independently. These come from lappy's
    `benchmarks/reference/*.py`, which escalate `n_basis` until every tension is at or below
    1e-12. Read it with `recipe(fn)`.
    """
    def deco(fn):
        return _register(fn, digits, source, notes, digits_by, settings)
    return deco


def catalog():
    """Every reference function, with its accuracy and provenance.

    Returns a list of `(name, digits, n_caveats, source)`, so
    `chladni.reference.catalog()` answers "what is available and how good is it".
    """
    return [(fn.__name__, fn.digits, sum(len(v) for v in fn.notes.values()), fn.source)
            for fn in REGISTRY]


def recipe(fn):
    """How a tabulated reference was produced: basis, collocation, and the digits reached.

    `None` for a closed form, which needs no recipe.
    """
    return getattr(fn, "settings", None)


def collapse(lam, rtol=0.0):
    """`[(lam, multiplicity), ...]` from an ascending list with multiplicity repeated.

    **The default is EXACT equality, and that is deliberate.** A tolerance here does not
    describe the spectrum, it imposes a policy on it -- "these two count as one eigenvalue" --
    and a reference that has already made that decision cannot be used to measure a solver
    that makes it differently. Ground truth should say how many eigenvalues there are; whether
    a search is allowed to merge two of them is the search's business to declare and the
    harness's to score, not the reference's to pre-empt.

    Exact equality is enough because a true degeneracy is BIT-identical, not merely close.
    Measured over the unit square and the 2x1 rectangle, every mathematically degenerate
    group has a relative spread of exactly 0.0: the arithmetic is symmetric in the indices,
    so the same real number comes out as the same float. Meanwhile a near-degenerate pair on
    `rect(1 + delta, 1)` is 1.2e-11 apart at `delta = 1e-11` -- eleven orders above float
    noise. There is no overlap between the two cases to arbitrate.

    An earlier version defaulted to `rtol = 1e-9`, which merged genuinely distinct pairs
    across three orders of magnitude, exactly the regime the cluster tests exist to probe. A
    solver that correctly resolved such a pair would have been scored as reporting a spurious
    root.

    `rtol` remains, for a caller who deliberately wants a coarser view. It is relative, unlike
    lappy's absolute 1e-12, which stops working for large `lam`.
    """
    lam = np.sort(np.asarray(lam, dtype=float))
    if lam.size == 0:
        return []
    out = [[float(lam[0]), 1]]
    for x in lam[1:]:
        if abs(x - out[-1][0]) <= rtol * max(abs(x), 1.0):
            out[-1][1] += 1
        else:
            out.append([float(x), 1])
    return [(a, b) for a, b in out]


#: Tolerance at which two computed eigenvalues are the SAME real number, differently rounded.
#:
#: This is float-noise correction inside the reference, not a policy about what counts as one
#: eigenvalue -- see `collapse` for that distinction. It is needed because a true degeneracy
#: is only bit-identical when the arithmetic happens to be symmetric. Measured worst spread
#: within a mathematically degenerate group:
#:
#:     eq_tri, iso_right_tri, polyomino, disk        0.0     (integer indices, or one zero
#:     rect with a rational aspect ratio             0.0      emitted twice)
#:     rect(sqrt(2), 1)                              4.4e-16  L**2 is 2.0000000000000004, so
#:     rect(sqrt(3), 1)                              3.6e-16  m**2/L**2 rounds per m
#:
#: against the tightest near-degeneracy worth resolving, 1.2e-11 on `rect(1 + 1e-11, 1)`.
#: 1e-13 sits about 230x above the noise and 120x below the signal.
SNAP_RTOL = 1e-13


def snap(lam, rtol=SNAP_RTOL):
    """Round a computed spectrum so that a true degeneracy is bit-identical.

    Values within `rtol` are the same real number arrived at by different arithmetic; they are
    replaced by their mean, repeated. The result is a flat ascending array in which a repeated
    float means a degenerate eigenvalue and two distinct floats mean two eigenvalues -- which
    is what lets ground truth be a plain multiset with no `(value, count)` bookkeeping.
    """
    lam = np.sort(np.asarray(lam, dtype=float))
    if lam.size == 0:
        return lam
    out = np.empty_like(lam)
    i = 0
    while i < lam.size:
        j = i + 1
        while j < lam.size and abs(lam[j] - lam[i]) <= rtol * max(abs(lam[j]), 1.0):
            j += 1
        out[i:j] = lam[i:j].mean()
        i = j
    return out


def _take_k(lam, k):
    lam = np.sort(np.asarray(lam, dtype=float))
    if lam.size < k:
        raise ValueError(
            f"index grid produced only {lam.size} eigenvalues; {k} were asked for. "
            f"Widen the grid."
        )
    return lam[:k]


def _check_k(k):
    k = int(k)
    if k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    return k


def _check_bc(bc_type):
    if bc_type not in ("dir", "neu"):
        raise ValueError(f"bc_type must be 'dir' or 'neu', got {bc_type!r}")
    return bc_type
