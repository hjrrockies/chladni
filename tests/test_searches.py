"""The baselines in `tests/searches.py`, and the one open defect among them.

`lappy_repaired_search` is lappy's search with its two measured defects repaired -- it exists to
answer how much of the branch-and-bound's advantage is available without the branch and bound --
and it carried a third defect that nothing explained. `search.md`, As built: it "misses a
DIFFERENT L-shape eigenvalue depending on `floor_frac`", which was "load-bearing in a way nothing
yet explains". It is the only known correctness failure in a search that otherwise works, and it
was found by the Weyl audit rather than by prefix truth.

# It is explained, and `floor_frac` is not what does it

Over `k` in `[3, 5]` on the L-shape, whose three levels are at `k = 3.104790, 3.898365,
4.442883`:

    floor_frac   located, error in k          error/h              kept   dropped by
    1e-03        6.4e-09, 5.9e-09, 3.5e-09    2.06, 1.51, 0.78     0/3    the interiority probe
    1e-06        4.4e-15, 1.2e-08, 3.6e-15    0.00, 2.97, 0.00     2/3    the same, middle level
    1e-09        3.1e-10, 1.4e-10, 1.3e-10    0.10, 0.04, 0.03     3/3    --
    1e-12        5.7e-14, 2.8e-13, 9.0e-14    0.00, 0.00, 0.00     3/3    --

**Every level is LOCATED at every setting**, to at worst 1.2e-08 in `k`, and at the default
`floor_frac = 1e-3` all three are then thrown away. So this was never a search failure: it is an
acceptance failure downstream of a correct search, which is why no amount of looking at the
recursion explained it.

What throws them away is the interiority probe:

    h = max(1e-9 * max(abs(r.x), 1.0), 1e-12)
    if not (T(r.x - h)[0] > f and T(r.x + h)[0] > f):
        continue                      # a wall point next to a deeper root

`h` is a FIXED relative step. A probe at `r.x +/- h` only demonstrates interiority when the
located root is nearer the true minimum than `h` is -- otherwise one of the two steps moves
TOWARD the minimum, reads a smaller tension, and the genuine root is refused. The controlling
quantity is therefore `error / h`, and the two populations separate by a factor of eight:

    error / h    0.000  0.000  0.000  0.000  0.029  0.037  0.098 | 0.782  1.508  2.061  2.974
    verdict      ------------------ kept ------------------------|--------- dropped ---------

Not a sharp threshold at 1, and it should not be one: the probe compares tension VALUES on two
flanks whose slopes differ, so near `error ~ h` the outcome depends on the local asymmetry
rather than on the distance alone. `k = 4.442883` at `floor_frac = 1e-3` is dropped at
`error/h = 0.78`. What the measurement supports is the mechanism and the separation, and not a
constant -- which is the distinction this project keeps having to redraw.

`floor_frac` enters only because it sets how tight a bracket `minimize_scalar` is handed, and
therefore how accurate the located root is. That the MIDDLE level is the one lost at 1e-6 is the
L-shape's five-order floor spread showing through: Brent's attained accuracy on a fixed bracket
width depends on the well's own `eps`, and `CHLADNI_V1.md` section 3.1 measured that variation as
5.4e-10 to 5.6e-5 across brackets.

# V1's acceptance rule is immune, and that is not a coincidence

`v1_refine._accept` asks the same question a different way: strictly interior, and below
the tension at BOTH SAMPLED ENDPOINTS of the bracket. It keeps 3 of 3 at every `floor_frac`,
including the default where the fixed-step probe keeps none.

Its docstring already argued this, about a different wrong version of the same probe: "a probe
whose reach is set by the bracket width cannot know what it is stepping across; the endpoints
can, because they are where the bracket actually ends." A probe whose reach is set by a CONSTANT
is worse still -- it cannot know what it is stepping across, and it cannot know how accurate the
point it is stepping from is either. This file is that argument arriving independently, from a
baseline rather than from the candidate.
"""
import functools

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

import hard_roster as hr
import searches as S
from chladni.reference import L_shape_eigs

pytestmark = pytest.mark.slow

#: `k` in [3, 5]; the L-shape has exactly three levels there.
K_LO, K_HI = 3.0, 5.0
LAM_LO, LAM_HI = K_LO**2, K_HI**2
TTOL = 1e-3          # `lappy_repaired_search`'s own acceptance threshold on the tension


@functools.lru_cache(maxsize=1)
def _probe():
    """A cached tension in `k`, the truth in `k`, and the search's own grid, per `floor_frac`."""
    case = hr.l_shape_case(a=LAM_LO, b=LAM_HI)
    truth = np.sqrt(np.array([x for x in L_shape_eigs(12) if LAM_LO <= x <= LAM_HI]))
    assert truth.size == 3, f"the window is supposed to hold three levels, got {truth.size}"

    def run(floor_frac):
        cache = {}

        def T(x):
            k = float(x)
            if k not in cache:
                cache[k] = case.pencil.tensions(k * k)
            return cache[k]

        n = max(int(20 * (K_HI - K_LO) / S._spacing(T, K_LO, K_HI)), 32)
        S._descend(T, K_LO, K_HI, n, 9, 60, floor_frac * (K_HI - K_LO))
        xs = np.array(sorted(cache))
        v = np.array([cache[x][0] for x in xs])
        out = []
        for i in S._local_minima(v):
            lo, hi = xs[i - 1], xs[i + 1]
            r = minimize_scalar(lambda x: T(x)[0] ** 2, bounds=(lo, hi), method="bounded",
                                options={"xatol": 1e-10})
            f = float(np.sqrt(max(r.fun, 0.0)))
            if f > TTOL:
                continue
            out.append(dict(x=float(r.x), sigma=f, lo=float(lo), hi=float(hi), T=T,
                            error=float(np.min(np.abs(truth - r.x)))))
        return out

    return run, truth


FLOORS = [1e-3, 1e-6, 1e-9, 1e-12]


@pytest.mark.parametrize("floor_frac", FLOORS)
def test_every_level_is_LOCATED_at_every_floor_frac(floor_frac):
    """The finding that moves the defect downstream of the search.

    Whatever `floor_frac` does, it does not lose an eigenvalue: all three are found at every
    setting, at worst 1.2e-08 out in `k`. So the recursion is not where to look, which is why
    looking at the recursion never explained it.
    """
    run, truth = _probe()
    cands = run(floor_frac)
    for t in truth:
        assert np.any([abs(c["x"] - t) < 1e-6 for c in cands]), (
            f"floor_frac={floor_frac:.0e}: no candidate within 1e-6 of k={t:.6f}")


@pytest.mark.parametrize("floor_frac", FLOORS)
def test_the_FIXED_STEP_probe_is_what_drops_them_and_it_tracks_error_over_h(floor_frac):
    """`error / h` separates kept from dropped by a factor of eight, over all twelve rows.

    The probe reads `T(x-h)` and `T(x+h)` and demands both exceed `T(x)`. That demonstrates
    interiority only when the located root is nearer the true minimum than `h` is; otherwise one
    step moves TOWARD the minimum and reads lower. `h` is a fixed `1e-9` relative, so it knows
    neither the bracket nor the accuracy of the point it steps from.

    Asserted as a SEPARATION rather than as a threshold at 1. The probe compares tension values
    on flanks of unequal slope, so the boundary is not sharp, and the one row near it -- dropped
    at `error/h = 0.78` -- is why this is stated as a population and not a rule.
    """
    run, _ = _probe()
    for c in run(floor_frac):
        h = max(1e-9 * max(abs(c["x"]), 1.0), 1e-12)
        T, f = c["T"], c["sigma"]
        accepted = bool(T(c["x"] - h)[0] > f and T(c["x"] + h)[0] > f)
        r = c["error"] / h
        if accepted:
            assert r < 0.2, (f"floor_frac={floor_frac:.0e}, k={c['x']:.8f}: kept at "
                             f"error/h = {r:.3f}, which is inside the dropped population")
        else:
            assert r > 0.5, (f"floor_frac={floor_frac:.0e}, k={c['x']:.8f}: dropped at "
                             f"error/h = {r:.3f}, which the mechanism does not explain")


def test_the_default_floor_frac_drops_EVERY_level():
    """The sharpest form of it: at the default `floor_frac = 1e-3` the search locates all three
    levels to 6.4e-09 and returns none of them."""
    run, truth = _probe()
    cands = run(1e-3)
    assert len(cands) == len(truth), "all three are located"
    kept = 0
    for c in cands:
        h = max(1e-9 * max(abs(c["x"]), 1.0), 1e-12)
        if c["T"](c["x"] - h)[0] > c["sigma"] and c["T"](c["x"] + h)[0] > c["sigma"]:
            kept += 1
    assert kept == 0, "and none survives the probe"


@pytest.mark.parametrize("floor_frac", FLOORS)
def test_V1s_ENDPOINT_rule_keeps_all_three_at_every_floor_frac(floor_frac):
    """The same question asked against SAMPLED points instead of a constant step.

    `v1_refine._accept`: strictly interior, and below the tension at both endpoints of
    the bracket -- which are points the search actually evaluated, so the test cannot be fooled
    by how accurate the located root happens to be. It keeps 3 of 3 everywhere, including where
    the fixed-step probe keeps 0.
    """
    run, truth = _probe()
    kept = [c for c in run(floor_frac)
            if c["lo"] < c["x"] < c["hi"]
            and c["sigma"] < min(c["T"](c["lo"])[0], c["T"](c["hi"])[0])]
    assert len(kept) == len(truth), (
        f"floor_frac={floor_frac:.0e}: endpoint rule kept {len(kept)} of {len(truth)}")
