"""Baselines: honest but dumb bracketers, kept separate from the controls.

TEST SUPPORT ONLY. `CHLADNI_V1.md` section 5.3.

The distinction from `bracket_controls.py` is not cosmetic. A control may read truth and exists
to break the metric; a **baseline may not read truth** and exists to be beaten. Mixing them
would let a candidate be compared against something that had the answer.

# What they are for

They fail in the two ways the metric is built to separate:

    uniform_grid       holds COVERAGE, fails PURITY
    grid_then_bisect   holds PURITY, loses a third of the eigenvalues (coverage 1.0 -> 0.577)

That opposition is the check that the scoring is not merely agreeing with itself. If both failed
the same way, the two metrics would be one metric.

`lipschitz-cone-mps.md` section 3.2 concedes that the honest comparison for its branch and bound
is "a well-chosen uniform grid", not a strawman fine one. So `uniform_grid` is tuned the way
someone competent would tune it, and a proposed search that cannot beat it has not earned its
complexity. Measured for calibration (`CHLADNI_V1.md` section 4): a working method costs ~30
evaluations per eigenvalue, a uniform grid dense enough to detect everything ~43 while failing
purity, and grid-then-bisect 74-95 while losing a third of the spectrum.

The five previous-generation candidates in `tests/searches.py` and `tests/bracketers.py` are
also baselines in this sense -- they are the measured thing-to-beat behind sections 29 and 30 --
and are kept for that reason. The specific number to beat is where their splitting fails:
`d/lam` between 1.2e-04 and 3.6e-05, four to five orders coarser than the pencil supports.
"""
from __future__ import annotations

import numpy as np

__all__ = ["uniform_grid", "grid_then_bisect", "random_probe", "BASELINES"]


def _plateau(tensions, a, b, n=24):
    """The far-field level, as a median. `tau_scan` is a FRACTION of this and never an
    absolute tension: the floor varies FIVE ORDERS within a single scan on a corner domain,
    structurally, while the plateau spans only 0.12 to 0.36 across seven domains."""
    xs = np.linspace(a, b, n)
    return float(np.median([tensions(x)[0] for x in xs]))


def _local_minima(v):
    """Interior indices strictly below both neighbours, ties broken LEFT.

    A well sampled symmetrically has two exactly equal flanking values, and a strict
    comparison on both sides then finds nothing at all.
    """
    v = np.asarray(v, dtype=float)
    if v.size < 3:
        return np.empty(0, dtype=int)
    left, mid, right = v[:-2], v[1:-1], v[2:]
    return np.nonzero((mid < left) & (mid <= right))[0] + 1


def uniform_grid(per_gap=8.0, tau_frac=0.5, n_max=20_000, n_pilot=32):
    """Sample once at a Weyl-sized stride, take discrete minima, bracket with the neighbours.

    Deliberately NOT `prec`-aware beyond the cap: a uniform grid cannot reach a spacing of
    `prec * lam` at any usable cost, so its brackets stay grid-spacing wide whatever `prec`
    asks for. That is the honest behaviour of the method and the score should show it -- which
    is exactly how it holds coverage and fails purity.

    `per_gap = 8` is a competent tuning, not a strawman, but note what it cannot fix: recall is
    bounded by the MINIMUM LOCAL gap, measured 1.3x to 8x tighter than the Weyl mean across
    seven domains, and full recall on that roster needed about 18 samples per median gap.
    """
    def brackets(tensions, a, b, *, prec, modes=None):
        a, b = float(a), float(b)
        # A pilot pass estimates the mean spacing from the dips themselves rather than from
        # Weyl, so the baseline needs no domain and stays method-agnostic.
        xs = np.linspace(a, b, n_pilot)
        v = np.array([tensions(x)[0] for x in xs])
        idx = _local_minima(v)
        spacing = (b - a) / max(idx.size, 1)
        n = int(np.clip(per_gap * (b - a) / spacing, 16, n_max))
        xs = np.linspace(a, b, n)
        v = np.array([tensions(x)[0] for x in xs])
        tau = tau_frac * float(np.median(v))
        idx = _local_minima(v)
        idx = idx[v[idx] < tau]
        return [(float(xs[i - 1]), float(xs[i + 1])) for i in idx]

    return brackets


def grid_then_bisect(per_gap=6.0, tau_frac=0.5, max_depth=40):
    """Grid to find dips, then bisect each toward its minimiser and emit a tight bracket.

    Holds purity -- the emitted brackets are narrow -- and loses eigenvalues, because a
    locator that converges to a single minimiser will happily converge to one member of a
    close pair and leave the other outside the bracket. That is precisely how this falls from
    coverage 1.0 to **0.577** (`CHLADNI_V1.md` section 2.5), and it is the failure the
    `sigma_2` isolation gate exists to prevent.

    The bisection is on the SIGN OF THE SLOPE rather than on a value, which is the strongest
    honest version: it descends the sampled cell without ever needing a threshold.
    """
    def brackets(tensions, a, b, *, prec, modes=None):
        a, b = float(a), float(b)
        pilot = np.linspace(a, b, 32)
        pv = np.array([tensions(x)[0] for x in pilot])
        spacing = (b - a) / max(_local_minima(pv).size, 1)
        n = int(np.clip(per_gap * (b - a) / spacing, 16, 20_000))
        xs = np.linspace(a, b, n)
        v = np.array([tensions(x)[0] for x in xs])
        tau = tau_frac * float(np.median(v))
        out = []
        for i in _local_minima(v):
            if v[i] >= tau:
                continue
            lo, hi = float(xs[i - 1]), float(xs[i + 1])
            target = max(prec * max(abs(0.5 * (lo + hi)), 1.0), 4 * np.spacing(hi))
            depth = 0
            while hi - lo > target and depth < max_depth:
                m = 0.5 * (lo + hi)
                # Follow the descending side: whichever half of the cell holds the lower of
                # the two quarter-point values.
                q1, q2 = 0.5 * (lo + m), 0.5 * (m + hi)
                if tensions(q1)[0] <= tensions(q2)[0]:
                    hi = m
                else:
                    lo = m
                depth += 1
            out.append((lo, hi))
        return out

    return brackets


def random_probe(per_eig=30.0, tau_frac=0.5, seed=0, n_pilot=32):
    """Probe at random points, bracket the discrete minima of what came back.

    The null model: the same evaluation budget a working method spends, spent with no
    structure at all. It is here because "the method beat a uniform grid" is a weaker claim
    than it sounds if the grid was badly sized, and a random draw at matched budget cannot be
    accused of a bad stride -- it has none.

    Seeded, because an unseeded baseline makes the comparison irreproducible: changing only
    the RNG seed moved a related quantity by 14x elsewhere in this project.
    """
    def brackets(tensions, a, b, *, prec, modes=None):
        a, b = float(a), float(b)
        rng = np.random.default_rng(seed)
        pilot = np.linspace(a, b, n_pilot)
        pv = np.array([tensions(x)[0] for x in pilot])
        n_eig = max(_local_minima(pv).size, 1)
        xs = np.sort(rng.uniform(a, b, int(per_eig * n_eig)))
        v = np.array([tensions(x)[0] for x in xs])
        tau = tau_frac * float(np.median(v))
        idx = _local_minima(v)
        idx = idx[v[idx] < tau]
        return [(float(xs[i - 1]), float(xs[i + 1])) for i in idx]

    return brackets


BASELINES = {
    "uniform_grid":     uniform_grid,
    "grid_then_bisect": grid_then_bisect,
    "random_probe":     random_probe,
}
