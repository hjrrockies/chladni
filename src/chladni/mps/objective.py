"""The shared seam every candidate search is driven through.

Promoted from `tests/objective.py`, now a re-export shim until the scored candidates are
repointed here. `docs/design/candidates.md` is the contract; this is the implementation of its
section 1 seam and section 3's cost accounting.

It is the counted, memoized seam a search is driven through, and the reason it belongs in the
package rather than in `tests/` is that `MPSEigensolver.solve_interval` -- still to be built --
needs exactly this object to hand `chladni.mps.search` a pencil.

An `Objective` is a tension callable plus an interval. It holds no domain, no basis and no
pencil, which is what makes two candidate searches comparable and what will let a later FEM or
boundary-integral objective be scored on the same instrument.

# The cache is REQUIRED, not an optimization

`MPSPencil` memoizes the equilibrated and regularized blocks (`pencil.py`, FIFO 64) but **not
the GSVD**: `tensions(lam)` calls `_gsvd.tensions_of` every time. So a search that re-reads a
bracket endpoint -- which the locator does constantly, since every emitted endpoint must be a
sampled point -- pays a full evaluation for a number it already had. Two candidates whose only
difference is how often they re-read would then be ranked by their bookkeeping rather than by
their sampling.

Cost is therefore counted as cache **misses**, and every candidate gets the same cache. That is
also the number `CHLADNI_V1.md` section 1 counts cost in, and the number the harness charges.

# `n_evals` versus `n_calls`

`n_evals` is misses: real work, the cost currency. `n_calls` is every ask. The ratio is a
diagnostic about the candidate's bookkeeping and is reported, never scored -- a method that
re-asks a lot is not thereby worse, it is just paying nothing for it here, and on a caller
without a cache it would pay.

# The variable

`x` is whatever the case set up, and truth is given in the same variable; see
`candidates.md` section 2 for why the canonical setup is `x = k` and why that does not
contradict `docs/conventions.md`. `from_pencil` is the one place that knows about `lam`, and it
converts at the `mps` boundary, which is where the conventions doc puts the seam.
"""
from __future__ import annotations

import numpy as np

from .._core import PREC

__all__ = ["Objective", "Declined"]


class Declined:
    """A stage's third outcome: it detected that its premises do not hold and stopped.

    Neither a pass nor a failure (`candidates.md` section 5). Carries whatever was found
    before the decline, so it can still be inspected, and the reason, which the harness
    reports -- a decline rate quoted beside a pass rate is the only thing that makes the
    mechanism honest.

    Truthiness is FALSE, so `if result:` treats a decline as "nothing usable" without having
    to remember to check. Iterating gives the partial brackets, so a caller that wants to
    score them anyway can, deliberately.
    """

    __slots__ = ("reason", "partial", "diagnostics")

    def __init__(self, reason, partial=(), diagnostics=None):
        self.reason = str(reason)
        self.partial = tuple(tuple(map(float, b)) for b in partial)
        self.diagnostics = diagnostics

    def __bool__(self):
        return False

    def __iter__(self):
        return iter(self.partial)

    def __len__(self):
        return len(self.partial)

    def __repr__(self):
        return f"Declined({self.reason!r}, {len(self.partial)} partial)"


class Objective:
    """A counted, memoized tension function on `[lo, hi]`.

    Not `Frozen`: the cache and the counters mutate by design, which is the whole point of
    the object. Deliberately NOT thread-safe -- the search that drives it is sequential, and a
    lock here would make the miss count depend on scheduling. Parallelism in this package is
    across whole solves (`douse`) and inside LAPACK, not inside one search.
    """

    def __init__(self, tensions, lo, hi, *, prec=PREC, variable="k",
                 expected_count=None, max_cache=None, modes=None):
        if not callable(tensions):
            raise TypeError("tensions must be callable")
        if modes is not None and not callable(modes):
            raise TypeError("modes must be callable or None")
        lo, hi = float(lo), float(hi)
        if not np.isfinite(lo) or not np.isfinite(hi) or not lo < hi:
            raise ValueError(f"need a finite interval with lo < hi, got ({lo}, {hi})")
        if variable not in ("k", "lam"):
            raise ValueError(f"variable must be 'k' or 'lam', got {variable!r}")
        self._f = tensions
        self._m = modes
        self.lo = lo
        self.hi = hi
        self.prec = float(prec)
        self.variable = variable
        self._expected = expected_count
        self._max_cache = None if max_cache is None else int(max_cache)
        self._cache = {}
        self._mode_cache = {}
        self._n_calls = 0
        self._n_mode_calls = 0

    # --- evaluation ------------------------------------------------------------------------

    def tensions(self, x):
        """The ascending vector at `x`. One evaluation returns the WHOLE vector.

        Reading `sigma_2` is free and carries the information that decides clustering, so the
        cache stores the vector rather than `sigma_1`: a gate that later wants `sigma_2` at a
        point the cover phase already visited must not pay for it again.
        """
        key = float(x)
        self._n_calls += 1
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        v = np.atleast_1d(np.asarray(self._f(key), dtype=float))
        # A miss is the unit of cost. Evict only under an explicit cap: a window is ~10**3
        # evaluations and dropping one silently converts a hit into a charge.
        if self._max_cache is not None and len(self._cache) >= self._max_cache:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = v
        return v

    @property
    def has_modes(self):
        """Can this objective supply candidate eigenfunctions? Ask before calling `modes`."""
        return self._m is not None

    def modes(self, x):
        """The candidate eigenfunction at `x` on the FIXED shared point set, or `None`.

        `None` when no callable was supplied, so a rule that wants the interior Gram angle can
        ask `has_modes` and fall back rather than crash on an objective built from a bare
        tension function. `tests/harness.py` states the seam: a search that does not want this
        ignores it, which is why it rides on the objective instead of widening `solve`.

        **Informative only ACROSS `x`** -- see `MPSPencil.interior_image`, which is what
        `from_pencil` wires in here.
        """
        if self._m is None:
            return None
        key = float(x)
        self._n_mode_calls += 1
        hit = self._mode_cache.get(key)
        if hit is not None:
            return hit
        u = np.atleast_1d(np.asarray(self._m(key), dtype=float))
        self._mode_cache[key] = u
        return u

    def tension(self, x):
        """`sigma_1` at `x`. The objective a locator minimizes -- as its SQUARE."""
        return float(self.tensions(x)[0])

    def sigma(self, x, j):
        """`sigma_{j+1}` at `x`, zero-indexed, `inf` past the retained rank.

        `inf` rather than an IndexError because the rank is re-derived at every parameter and
        so the vector length moves: a gate asking for `sigma_2` where only one direction
        survived should read "no neighbour anywhere near", which is what `inf` says.
        """
        v = self.tensions(x)
        return float(v[j]) if j < v.size else float("inf")

    # --- cost and history ------------------------------------------------------------------

    @property
    def n_evals(self):
        """Cache misses: the only cost currency (`candidates.md` section 3).

        **One evaluation is one `x` at which the pencil was FACTORIZED**, by either reader. That
        keeps the definition `candidates.md` section 3 gives and that every committed grid is
        denominated in: a mode at a FRESH `x` buys the factorization and is charged in full, or a
        candidate could scan the whole window through `modes` for nothing.

        **What it does not charge is the second LAPACK call**, and the size of that is worth
        stating rather than waving at, because two different comparisons get confused here.
        Measured on a `family_B` pencil, `prec = 1e-8`:

            fresh tension (assemble, equilibrate, regularize, gsvdvals)     1.81 ms
            mode at a COLD lam (the same, then a separate-mode gsvd)        1.83 ms
            mode at an lam already tensioned (the separate gsvd alone)      1.16 ms

        So a mode CALL costs about what a tension call costs -- which is the 5-7% figure
        `tests/rect_suite.py`'s `Counter` records, and why one counter serves both there -- but
        the MARGINAL cost of a mode at a point already evaluated is about 0.6 of an evaluation,
        not a twentieth of one. Charging it as a full second miss would overstate by ~1.6x;
        charging nothing understates by ~0.6. Neither is free of error and the currency cannot
        express a fraction.

        It stays at zero, and the omission is REPORTED instead: `n_mode_evals` below counts the
        points where a mode was computed, so `search.md` section 2.6's budget -- one
        `separate`-mode GSVD per REPORTED ROOT, at the end -- is a number a run prints rather
        than an assumption. That budget is what keeps the omission small: measured 4 mode calls
        against 89 evaluations on the same case, so ~3% of the total even at 0.6 each. Inflating
        the currency to cover it would invalidate every grid measured in it, for 3%.

        The UNION of the two caches, which is exactly `len(self._cache)` when no modes are in
        play -- eviction included, so an objective under `max_cache` charges the same as before.
        """
        return len(self._cache.keys() | self._mode_cache.keys())

    @property
    def n_mode_evals(self):
        """`x` values at which a candidate eigenfunction was computed. Reported, not scored."""
        return len(self._mode_cache)

    @property
    def n_mode_calls(self):
        """Every mode ask, hits included. Diagnostic only, never scored."""
        return self._n_mode_calls

    @property
    def n_calls(self):
        """Every ask, hits included. Diagnostic only, never scored."""
        return self._n_calls

    @property
    def samples(self):
        """`(xs, sigma_1)` over everything evaluated, ascending in `x`.

        Every emitted bracket endpoint must be a SAMPLED point carrying a three-point descent
        certificate, never an extrapolated root -- two-point secant roots from opposite walls
        of a close pair were measured to CROSS, giving an inverted bracket or a sliver holding
        nothing. This is where a candidate reads them from, so it needs no bookkeeping of its
        own, and it is also what the harness uses to induce brackets for a method that never
        produced any.

        Reads the TENSION cache only, deliberately. An `x` where a mode was computed but no
        tension was has no `sigma_1`, so it is charged but is not a sample -- putting it here
        would hand `search._samples`, `wide_bracket_for` and the slope readers a point with no
        value, which is worse than not having it.
        """
        if not self._cache:
            return np.empty(0), np.empty(0)
        xs = np.array(sorted(self._cache))
        return xs, np.array([self._cache[x][0] for x in xs])

    def reset(self):
        """Drop the caches and the counters. A fresh cost measurement on the same callable.

        The mode cache goes too, or a candidate scored after another would inherit a warm one --
        `bracket_score.run_bracketer` calls this precisely so a candidate is charged for its own
        evaluations and nobody else's.
        """
        self._cache = {}
        self._mode_cache = {}
        self._n_calls = 0
        self._n_mode_calls = 0

    # --- Weyl, for SIZING only -------------------------------------------------------------

    def expected_count(self, lo=None, hi=None):
        """How many eigenvalues Weyl's law expects in the sub-interval; `None` if unavailable.

        SIZING ONLY. It sets a starting sample density and nothing else. It says nothing about
        WHERE anything is and must not be used as if it did -- and it is an estimate of the
        MEAN gap, while recall is bounded by the MINIMUM local gap, which was measured 1.3x to
        8x tighter than the Weyl mean across seven domains.
        """
        if self._expected is None:
            return None
        return float(self._expected(self.lo if lo is None else float(lo),
                                   self.hi if hi is None else float(hi)))

    # --- adapters --------------------------------------------------------------------------

    @classmethod
    def from_pencil(cls, pencil, lo, hi, *, variable="k", **kw):
        """Wrap an `MPSPencil`. The ONE place in this layer that knows about `lam`.

        The conversion falls here, at the `mps` boundary, which is where `docs/conventions.md`
        puts it: the pencil is always called with `lam`, and a caller working in `k` never
        hands `lam` to something expecting `k` or the reverse.
        """
        if variable == "k":
            def tensions(k):
                return pencil.tensions(float(k) ** 2)

            def modes(k):
                return pencil.interior_image(float(k) ** 2)
        else:
            def tensions(lam):
                return pencil.tensions(float(lam))

            def modes(lam):
                return pencil.interior_image(float(lam))
        kw.setdefault("prec", getattr(pencil, "prec", PREC))
        kw.setdefault("modes", modes)
        return cls(tensions, lo, hi, variable=variable, **kw)

    def in_lam(self, x):
        """`x` as a `lam`, whichever variable this objective is parameterized by."""
        return float(x) ** 2 if self.variable == "k" else float(x)

    def in_lam_bracket(self, lo, hi):
        """A bracket in this objective's variable as a `lam` bracket that still fences it.

        Squaring is not exact: `[a*a, b*b]` in float64 can miss `lam* = k*^2` by an ulp even
        when `[a, b]` correctly fenced `k*`, and a refiner handed that bracket clips onto the
        endpoint and rejects a real eigenvalue. So each end moves outward by 4 ulps -- the
        widening `bracket.py`'s module docstring promises -- here, where every caller that
        converts a bracket passes, rather than at each call site.
        """
        a, b = self.in_lam(lo), self.in_lam(hi)
        return a - 4.0 * np.spacing(abs(a)), b + 4.0 * np.spacing(abs(b))

    def from_lam(self, lam):
        """A `lam` as an `x` in this objective's variable."""
        return float(lam) ** 0.5 if self.variable == "k" else float(lam)

    def __repr__(self):
        return (f"Objective({self.variable}=[{self.lo:.6g}, {self.hi:.6g}], "
                f"prec={self.prec:.1e}, {self.n_evals} evals)")
