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

import concurrent.futures as cf
import os
from typing import NamedTuple, Optional

import numpy as np

from .._core import PREC
from .policy import Policy

__all__ = ["Objective", "Declined", "Evaluation", "PolicyView"]


class Evaluation(NamedTuple):
    """What one evaluation can hand the objective besides the tension vector: the stack's
    singular values at the same point (`MPSPencil.spectrum`), a VECTOR. A tension callable may
    return one of these instead of a bare vector; a bare vector still works, with no spectrum."""

    tensions: np.ndarray
    spectrum: Optional[np.ndarray] = None


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


#: Threads for a batch of tension evaluations (`Objective.from_pencil`). Bounded below the
#: machine's core count because each evaluation is itself a LAPACK call on a BLAS pool, and
#: douse runs several solves in separate processes besides.
DEFAULT_WORKERS = max(1, min(8, (os.cpu_count() or 2) // 2))

#: How far (in ulps of `lam`) a bracket converted to `lam` is widened at each end, so that a
#: minimum whose exact square rounds just outside the `k` bracket is not lost; and, in
#: `refine`, the margin within which two points are the SAME point. One constant, because the
#: two interact: a 4-ulp margin against a 4-ulp widening is exactly what let a search's deepest
#: sample -- on a neighbour's wall -- through as a root, which is why `refine` judges a point at
#: a widened end by the curve beyond it (`refine._end_verdict`) rather than by the margin.
BRACKET_ULPS = 4.0

class Objective:
    """A counted, memoized tension function on `[lo, hi]`.

    Not `Frozen`: the cache and the counters mutate by design, which is the whole point of
    the object. Deliberately NOT thread-safe -- the search that drives it is sequential, and a
    lock here would make the miss count depend on scheduling. Parallelism in this package is
    across whole solves (`douse`) and inside LAPACK, not inside one search -- with ONE batch
    seam: `tensions_many` hands a list of cache MISSES to `batch` (a thread pool over the pencil
    in `from_pencil`) and inserts the results on the calling thread, in input order, so the
    cache and the miss count come out exactly as a serial loop would leave them. The seed grid
    goes through it (`bracket._seed`); the cover pass is sequential by nature (2026-10-14).
    """

    def __init__(self, tensions, lo, hi, *, prec=PREC, variable="k",
                 expected_count=None, max_cache=None, modes=None, batch=None,
                 policy_tensions=None):
        if not callable(tensions):
            raise TypeError("tensions must be callable")
        if modes is not None and not callable(modes):
            raise TypeError("modes must be callable or None")
        if batch is not None and not callable(batch):
            raise TypeError("batch must be callable or None")
        if policy_tensions is not None and not callable(policy_tensions):
            raise TypeError("policy_tensions must be callable or None")
        self._batch = batch
        self._pf = policy_tensions
        #: The `MPSPencil` this objective evaluates, when built by `from_pencil`; None for a bare
        #: callable. `MPSEigensolver.search_interval` refuses an objective whose pencil is not its
        #: own, because certification reads that solver's pencil.
        self.pencil = None
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
        self._spec = {}
        self._n_calls = 0
        self._n_mode_calls = 0

    # --- evaluation ------------------------------------------------------------------------

    def tensions(self, x, policy=None):
        """The ascending vector at `x`. One evaluation returns the WHOLE vector.

        Reading `sigma_2` is free and carries the information that decides clustering, so the
        cache stores the vector rather than `sigma_1`: a gate that later wants `sigma_2` at a
        point the cover phase already visited must not pay for it again.

        `policy` (`chladni.mps.policy.Policy`) names the rank to read at. None (or the pencil's
        own) is keyed by `float(x)` exactly as before; any other is keyed `(x, policy.key)`, so
        two policies' values at one `x` are never mixed, and each is charged as an evaluation.
        """
        policy = Policy.normalize(policy)
        key = float(x)
        self._n_calls += 1
        ckey = key if policy is None else (key, policy.key)
        hit = self._cache.get(ckey)
        if hit is not None:
            return hit
        if policy is None:
            out = self._f(key)
        elif self._pf is None:
            raise ValueError("this objective was built from a bare tension callable and cannot "
                             "evaluate another policy; build it with Objective.from_pencil")
        else:
            out = self._pf(key, policy)
        v = self._unpack(key, out)
        # A miss is the unit of cost. Evict only under an explicit cap: a window is ~10**3
        # evaluations and dropping one silently converts a hit into a charge.
        if self._max_cache is not None and len(self._cache) >= self._max_cache:
            self._cache.pop(next(iter(self._cache)))
        self._cache[ckey] = v
        return v

    def _unpack(self, key, out):
        """The tension vector from what a callable returned, keeping a spectrum if it came."""
        if isinstance(out, Evaluation):
            if out.spectrum is not None and key not in self._spec:
                self._spec[key] = out.spectrum
            out = out.tensions
        return np.atleast_1d(np.asarray(out, dtype=float))

    def spectrum(self, x):
        """The stack's singular values recorded at `x`, or None where none was: `x` not yet
        evaluated, or an objective built from a bare tension callable. A VECTOR; the stable rank
        is read from two of these, never from a matrix."""
        return self._spec.get(float(x))

    def spectra_near(self, x, radius):
        """`[(x_i, spectrum_i), ...]` recorded within `radius` of `x` (in this objective's
        variable), nearest first."""
        x = float(x)
        near = [(abs(k - x), k) for k in self._spec if abs(k - x) <= radius]
        return [(k, self._spec[k]) for _, k in sorted(near)]

    def tensions_many(self, xs):
        """The vectors at each of `xs`, as a list. Misses are evaluated together -- through
        `batch` when the objective has one, else one by one -- and stored in input order, so
        `n_evals`, `samples` and every later hit are identical to calling `tensions` in a loop.
        """
        keys = [float(x) for x in xs]
        self._n_calls += len(keys)
        have = {k: self._cache[k] for k in keys if k in self._cache}
        misses = list(dict.fromkeys(k for k in keys if k not in have))
        if misses:
            vals = self._batch(misses) if self._batch is not None else [self._f(k) for k in misses]
            for k, v in zip(misses, vals):
                v = self._unpack(k, v)
                if self._max_cache is not None and len(self._cache) >= self._max_cache:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[k] = v
                have[k] = v                     # returned even if evicted later in this batch
        return [have[k] for k in keys]

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

    def tension(self, x, policy=None):
        """`sigma_1` at `x`. The objective a locator minimizes -- as its SQUARE."""
        return float(self.tensions(x, policy)[0])

    def sigma(self, x, j, policy=None):
        """`sigma_{j+1}` at `x`, zero-indexed, `inf` past the retained rank.

        `inf` rather than an IndexError because the rank is re-derived at every parameter and
        so the vector length moves: a gate asking for `sigma_2` where only one direction
        survived should read "no neighbour anywhere near", which is what `inf` says.
        """
        v = self.tensions(x, policy)
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
        return self.samples_for(None)

    def samples_for(self, policy):
        """`(xs, sigma_1)` over everything evaluated under `policy`, ascending in `x`.
        `samples` is the pencil's own policy's."""
        policy = Policy.normalize(policy)
        if policy is None:
            keys = [k for k in self._cache if not isinstance(k, tuple)]
            xs = np.array(sorted(keys))
            return (xs, np.array([self._cache[x][0] for x in xs])) if keys else (np.empty(0),
                                                                                 np.empty(0))
        pk = policy.key
        keys = sorted(k[0] for k in self._cache if isinstance(k, tuple) and k[1] == pk)
        if not keys:
            return np.empty(0), np.empty(0)
        return np.array(keys), np.array([self._cache[(x, pk)][0] for x in keys])

    def reset(self):
        """Drop the caches and the counters. A fresh cost measurement on the same callable.

        The mode cache goes too, or a candidate scored after another would inherit a warm one --
        `bracket_score.run_bracketer` calls this precisely so a candidate is charged for its own
        evaluations and nobody else's.
        """
        self._cache = {}
        self._mode_cache = {}
        self._spec = {}
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
    def from_pencil(cls, pencil, lo, hi, *, variable="k", max_workers=None, **kw):
        """Wrap an `MPSPencil`. The ONE place in this layer that knows about `lam`.

        The conversion falls here, at the `mps` boundary, which is where `docs/conventions.md`
        puts it: the pencil is always called with `lam`, and a caller working in `k` never
        hands `lam` to something expecting `k` or the reverse.

        `max_workers` sizes the thread pool `tensions_many` evaluates a batch of misses on
        (`DEFAULT_WORKERS` when None; 1 keeps every evaluation on the calling thread). The work
        is LAPACK, which releases the GIL; the pencil's cache tolerates concurrent writers.
        """
        def evaluate(lam, policy=None):
            # The spectrum is read right after the tension, on the same thread: under `"svd"` it
            # was stored by the same factorization, so this is a cache hit, not a second SVD.
            t = pencil.tensions(lam, policy)
            spectrum = getattr(pencil, "spectrum", None)
            return Evaluation(t, spectrum(lam) if spectrum is not None and
                              getattr(pencil, "reg", None) == "svd" else None)

        if variable == "k":
            def tensions(k):
                return evaluate(float(k) ** 2)

            def modes(k):
                return pencil.interior_image(float(k) ** 2)

            def to_lam(x):
                return float(x) ** 2
        else:
            def tensions(lam):
                return evaluate(float(lam))

            def modes(lam):
                return pencil.interior_image(float(lam))

            def to_lam(x):
                return float(x)
        workers = DEFAULT_WORKERS if max_workers is None else max(1, int(max_workers))

        def batch(xs):
            lams = [to_lam(x) for x in xs]
            if workers == 1 or len(lams) == 1:
                return [evaluate(l) for l in lams]
            with cf.ThreadPoolExecutor(max_workers=min(workers, len(lams))) as ex:
                return list(ex.map(evaluate, lams))

        def policy_tensions(x, policy):
            return evaluate(to_lam(x), policy)
        kw.setdefault("prec", getattr(pencil, "prec", PREC))
        kw.setdefault("modes", modes)
        kw.setdefault("batch", batch)
        kw.setdefault("policy_tensions", policy_tensions)
        obj = cls(tensions, lo, hi, variable=variable, **kw)
        obj.pencil = pencil
        return obj

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
        return a - BRACKET_ULPS * np.spacing(abs(a)), b + BRACKET_ULPS * np.spacing(abs(b))

    def from_lam(self, lam):
        """A `lam` as an `x` in this objective's variable."""
        return float(lam) ** 0.5 if self.variable == "k" else float(lam)

    def __repr__(self):
        return (f"Objective({self.variable}=[{self.lo:.6g}, {self.hi:.6g}], "
                f"prec={self.prec:.1e}, {self.n_evals} evals)")


class PolicyView:
    """An `Objective` read under one `Policy`: the same window, variable, cache and counters,
    every tension asked at that policy. What a stage that froze a rank around one root hands
    `refine` and the multiplicity count, so neither needs to know policies exist and neither can
    mix a value read under another (`samples` is this policy's only)."""

    def __init__(self, objective, policy, cell=None):
        self._o = objective
        self.policy = Policy.normalize(policy)
        #: `(lo, hi)` in `lam`, or None. `samples` is restricted to it: a rank frozen around one
        #: root is a statement about that root's cell, and another root's cell read at the same
        #: rank is a different pencil's neighbourhood -- its samples sit on ANOTHER eigenvalue,
        #: where the tension is near zero, and read as this root's flanks they cut a real branch
        #: (the L-shape's 49.348 double read 1, 2026-10-27).
        self.cell = None if cell is None else (float(cell[0]), float(cell[1]))
        self.lo, self.hi, self.prec = objective.lo, objective.hi, objective.prec
        self.variable = objective.variable
        self.pencil = getattr(objective, "pencil", None)

    def tensions(self, x):
        return self._o.tensions(x, self.policy)

    def tensions_many(self, xs):
        return [self.tensions(x) for x in xs]

    def tension(self, x):
        return float(self.tensions(x)[0])

    def sigma(self, x, j):
        v = self.tensions(x)
        return float(v[j]) if j < v.size else float("inf")

    @property
    def samples(self):
        xs, vs = self._o.samples_for(self.policy)
        if self.cell is None or xs.size == 0:
            return xs, vs
        lam = np.array([self._o.in_lam(x) for x in xs])
        keep = (lam >= self.cell[0]) & (lam <= self.cell[1])
        return xs[keep], vs[keep]

    @property
    def n_evals(self):
        return self._o.n_evals

    @property
    def n_calls(self):
        return self._o.n_calls

    def spectrum(self, x):
        return self._o.spectrum(x)

    def expected_count(self, lo=None, hi=None):
        return self._o.expected_count(lo, hi)

    def in_lam(self, x):
        return self._o.in_lam(x)

    def from_lam(self, lam):
        return self._o.from_lam(lam)

    def in_lam_bracket(self, lo, hi):
        return self._o.in_lam_bracket(lo, hi)

    def __repr__(self):
        return f"PolicyView({self._o!r}, {self.policy})"
