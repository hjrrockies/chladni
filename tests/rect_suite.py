"""Rectangle benchmark cases for FILTERING eigenvalue-search candidates.

TEST SUPPORT ONLY.

# Why rectangles, and why filtering rather than ranking

The search layer is basis-agnostic by construction -- `docs/design/search.md` says it "sees
only the tension function and the interval: no domain, no basis, no pencil". So a method can
only respond to the SHAPE of the tension curve, and that shape has four parameters: the floor
`eps`, the wall slope `c`, the plateau, and the eigenvalue spacing. Rectangles control all
four, and they add three things no real domain offers:

* **closed-form truth**, so accuracy can be checked to float64 rather than to a table's own
  precision (`H_shape`'s table is 7.8 digits, `chevron`'s nominally 3.0);
* **exact control of clustering and multiplicity** -- the square reaches multiplicity 2, 3, 4,
  6 and 8, and perturbing a multiplicity-`m` level splits it into `m` eigenvalues whose
  spacing scales with the perturbation;
* **cost**, 8-37 ms per evaluation against GWW1's 111, which is what buys error bars.

The purpose is to FILTER candidates, not rank them. Each candidate gets its best setting
WITHIN A FIXED EVALUATION BUDGET; if it still fails, the failure is a capability limit, and
overfitting can only flatter a method, so a rejection survives it. Results are therefore
capability claims -- "M cannot separate d = 1e-4 within 500 evaluations at any tested
setting" -- never performance claims.

The budget is what makes that sound. Best-of tuning at UNBOUNDED budget is vacuous: any method
with a resolution knob passes every case eventually.

# What rectangles cannot do

On a corner domain `eps` jumps orders between ADJACENT modes -- structurally, because whichever
modes vanish at the reentrant corner are resolved far better than their neighbours
(`landscape_findings` section 20.1: 3.1e-13 next to 5.5e-08). On a rectangle `eps` varies
smoothly with `lam`. A method that implicitly assumes locally-smooth `eps` would pass this
suite and fail on an L-shape, so the corner roster stays as a transfer check.
"""
from __future__ import annotations

import numpy as np

from chladni.geometry import rect
from chladni.helmholtz import PlaneWave
from chladni.mps import MPSPencil
from chladni.reference import rect_eig, rect_spectrum

from harness import Case

__all__ = ["Counter", "pencil_for", "make_case", "diagnostics", "assert_floor",
           "assert_smooth",
           "SUITE_RTOL", "FINEST_PREC",
           "modes_in_window", "splitting_rates", "tightest_pair", "delta_for",
           "crossover_cases", "split_degeneracy_cases", "gate_cases",
           "cluster_diameter_ratio", "multiplicity_undecided",
           "CROSSOVER_WINDOWS", "WIDE_WINDOW",
           "FLOOR_TARGET", "PHI",
           "family_A", "family_B", "family_C", "family_D", "family_stress",
           "all_families"]

PI2 = np.pi**2
PHI = (1.0 + 5.0**0.5) / 2.0

#: `snap` in `chladni.reference` merges within this, so truth below it is not trustworthy.
SNAP_FLOOR_REL = 1e-13

#: The finest precision the sweep exercises. The pencil must serve it, because it is fixed
#: across the whole `prec` ladder and the ladder's bottom rung is the hardest requirement.
FINEST_PREC = 1e-14

#: Rank-truncation threshold for every suite pencil: `sqrt(FINEST_PREC * eps)`, which is
#: `chladni`'s OWN derived default rule evaluated at the tightest precision this suite has to
#: serve. Fixed, so the objective is identical across the `prec` sweep and a difference in
#: bracketing behaviour is attributable to the algorithm rather than to the pencil.
#:
#: Measured on `rect(1, 1/phi)` over `[5, 120]`, 900 samples, 4 true eigenvalues -- floor is
#: the worst `sigma_mu` and "spurious" counts sampled local minima not near a true level:
#:
#:     rtol       rank(lo, hi)   floor      floor ok   spurious minima
#:     1e-16        42, 92       1.1e-15      yes            264
#:     1e-15        28, 51       2.1e-15      yes              3
#:     1.49e-15     28, 51       2.1e-15      yes              0
#:     1.5e-12      23, 45       4.2e-13      NO               0
#:     1e-10        --           1.4e-10      NO               0
#:
#: Three things this settles, two of which correct what was expected going in:
#:
#: * **`1e-16` is the inert regime, and it is inert from the TOP of the window down.** The rank
#:   at `lam = 120` comes back at 92 of 92 columns -- every noise-floor direction retained --
#:   while at the low end it truncates normally. That is `CHLADNI_V1.md` section 6's finding
#:   and its separate warning that "the floor fails from the TOP of the window down, where
#:   basis demand is highest", arriving together. The cost is not the floor, which stays at
#:   1.1e-15; it is 264 spurious local minima in a window holding four eigenvalues.
#: * **V1's `1e-15` does not fully transfer**, because `chladni` thresholds RELATIVELY and
#:   `lappy` did not. It leaves 3 spurious minima here where V1 measured 0.
#: * **The choice is set by the FLOOR, not by the spurious count.** `1.5e-12` -- the derived
#:   default at `prec = 1e-8` -- also reaches 0 spurious, but costs two orders of floor
#:   (4.2e-13) and fails the precondition outright, with headroom 0.02 against the 1.0 needed.
#:   The bottom of the `prec` ladder would become unexercisable. Three residual spurious minima
#:   would have been cheap -- a refiner rejects an empty bracket in 2 to 7 evaluations -- but at
#:   `sqrt(FINEST_PREC * eps)` there are none to pay for.
#:
#: And it is a check on the `sqrt(prec * eps)` rule itself: the rule delivers, at each `prec`,
#: about the floor that `prec` needs -- 4e-13 at 1e-8 and 2e-15 at 1e-14 -- rather than a
#: constant. That is the behaviour it was derived to have.
#:
#: **Why this value and not another, structurally.** Betcke's relation -- the floor inflates in
#: proportion to the truncation threshold -- holds here in the relative convention too:
#: measured over `rtol` from 1e-13 to 1e-8 the log-log slope is 1.13 with `eps/rtol` near 1.15,
#: so `eps ~ max(eps_basis, rtol)` with `eps_basis = 7.5e-16` on this pencil. `sqrt(FINEST_PREC
#: * eps_mach) = 1.49e-15` is therefore the CROSSOVER of those two terms: below it the
#: truncation contributes less than the basis already does and buys nothing, above it the floor
#: inflates linearly. The derived rule landing on the crossover is why it is the right default
#: rather than a convenient one, and `tests/test_floor.py` pins the relation.
SUITE_RTOL = float(np.sqrt(FINEST_PREC * np.finfo(float).eps))

_PENCILS: dict = {}


class Counter:
    """Counts every pencil call a search makes, tensions and modes alike.

    `modes` costs 5-7% more than `tensions` because both need the same regularization and
    factorization, so one counter for both is close enough and keeps the comparison honest --
    a method that leans on `modes` is not given them for free.
    """

    def __init__(self):
        self.calls = 0


def pencil_for(L, H, lam_lo, lam_hi, seed=0, pad=1.6, nb=None, ni=None,
               rtol=None):
    """A plane-wave pencil for the `L x H` rectangle over `[lam_lo, lam_hi]`, memoized.

    Every rectangle corner is a right angle, so `pi/omega = 2` is an integer and the
    eigenfunction is analytic there -- `Corner.is_reflection_closed` is True at all four. No
    corner-adapted family is needed and plane waves alone are the right basis.

    Memoized because five methods are scored on the SAME pencil: assembly is paid once, not
    five times, and every method then sees identical numbers.

    **PLANE WAVES, not corner Fourier-Bessel.** A right angle has `pi/omega = 2`, so the
    eigenfunction is ANALYTIC there (`Corner.is_reflection_closed`) and no fractional expansion
    is called for. Measured on the unit square at `rtol = 1e-16`, `sigma_mu` counting
    multiplicity: FB at one corner gives 1.7e-15 / 2.6e-14 / 1.4e-11 / 3.2e-12 at the levels of
    multiplicity 1 / 2 / 3 / 4, while sized plane waves give 1.2e-15 / 8.7e-16 / 1.5e-15 /
    2.3e-15. Four orders, and the FB floor does not move with the order (80 -> 300 columns),
    with collocation (400 -> 6400), or with `rtol` (1.5e-12 -> 0). Pure FB winning on the
    L-shape does not transfer to a domain with no singular corner.

    **`rtol` is FIXED at `SUITE_RTOL`, not derived from the algorithm's `prec`.** The pencil
    must be identical across the whole `prec` sweep, or a bracketing result confounds the
    algorithm with the objective; `prec` enters only as an input to the bracketing algorithm.
    An earlier version pinned `1e-16`, which under a RELATIVE threshold is below machine
    epsilon and leaves the truncation inert at the top of the window -- 264 spurious local
    minima in a window holding four eigenvalues. See `SUITE_RTOL` for the measurement.

    **THRESHOLD RANK, never frozen.** A frozen `lam_ref=(lo, hi)` takes the MINIMUM rank over
    the range (section 5), which the LOW end sets -- so over a wide window the high end is
    starved. Measured on these very cases: `rect(1, 1/phi)` over `[5, 300]` freezes at rank 23
    at every `lam` and reaches a tension of 3.5e-02 at the worst true eigenvalue, and adding
    columns does not help at all (102 -> 314 columns leaves the rank at 23 and the tension
    slightly WORSE). Threshold rank re-derives per `lam` -- 35, 49, 53 across the same window --
    and the worst tension falls to 2.1e-12. Ten orders, at equal or lower cost.
    """
    rtol = SUITE_RTOL if rtol is None else float(rtol)
    key = (float(L), float(H), float(lam_lo), float(lam_hi), int(seed), float(pad),
           nb, ni, float(rtol))
    hit = _PENCILS.get(key)
    if hit is not None:
        return hit
    dom = rect(L, H)
    nw = int(pad * PlaneWave.n_useful(np.sqrt(lam_hi) * dom.diameter))
    # SIZE THE COLLOCATION TO THE PENCIL, not to a fixed floor: 3x boundary and 1x interior
    # per column. Measured on `rect(1, 1/phi)` over `[5, 120]`, 92 columns, at `SUITE_RTOL`,
    # taking the worst tension at a true eigenvalue and the cost of an evaluation:
    #
    #     nb/n  ni/n   worst eps   ms/eval
    #      8.0   3.0    1.4e-15      3.2      <- what this used to do
    #      3.0   1.0    1.2e-15      2.1
    #      1.2   1.0    6.1e-16      2.0
    #
    # The floor does not move at all across that range -- every row passes the 1e-14 target --
    # so the extra points were buying nothing and costing 1.5x. The old `max(8*nw, 1200)` and
    # `max(3*nw, 600)` were worse than the ratios suggest, because the FLOORS bind on small
    # bases: a 40-column basis got 1200 boundary points, thirty per column.
    #
    # 3x boundary rather than the 1.2x the measurement would allow: the margin is what keeps a
    # test that has to RUN from failing on a domain whose symmetry makes the boundary sampling
    # degenerate, and it costs 5%.
    nb = nb if nb is not None else 3 * nw
    ni = ni if ni is not None else nw
    p = MPSPencil.from_basis(
        PlaneWave.uniform(nw, radius=dom.diameter), dom.bdry_quad(nb),
        dom.interior_pts(ni, rng=np.random.default_rng(seed)), rtol=rtol)
    _PENCILS[key] = p
    return p


def _truth_in_window(L, H, a, b, modes=None):
    """Complete truth in `[a, b]`, and a flag for whether `snap` could have corrupted it.

    `rect_spectrum` is complete by construction -- its index bound is derived from `lam_hi`
    rather than guessed -- but it returns `snap(...)`, which merges values within
    `1e-13 * lam`. For a family whose whole point is a controlled separation that is the one
    thing we cannot afford, so when the modes are known the values come from `rect_eig`
    directly and `rect_spectrum` is kept only as a cross-check on the COUNT.
    """
    spec = np.sort(np.asarray(rect_spectrum(a, b, L, H), dtype=float))
    if modes is None:
        return spec, spec
    direct = np.sort(np.array([rect_eig(m, n, L, H) for m, n in modes], dtype=float))
    return direct, spec


def make_case(label, L, H, a, b, *, seed=0, modes=None, variable="lam", pad=1.6):
    """One scored case: a shared pencil, counted access, and complete closed-form truth."""
    p = pencil_for(L, H, a, b, seed=seed, pad=pad)
    truth, spec = _truth_in_window(L, H, a, b, modes)
    c = Counter()

    def tensions(x):
        c.calls += 1
        return p.tensions(float(x))

    def modes_fn(x):
        c.calls += 1
        _, I = p.pencil(float(x))
        u = I @ p.coef(float(x), 1)[:, 0]
        return u / np.linalg.norm(u)

    case = Case(label=label, tensions=tensions, a=a, b=b, truth=truth, counter=c,
                domain=rect(L, H), complete=True, variable=variable, modes=modes_fn)
    case.pencil = p                      # for diagnostics; the search never sees it
    case.spectrum = spec
    return case


FLOOR_TARGET = 1e-14        #: sigma_mu must sit at or below this at every true eigenvalue


def assert_floor(case, target=FLOOR_TARGET, finest_prec=1e-14, margin=3.0, h_frac=0.005):
    """HARD PRECONDITION, not a diagnostic. Run before a case is scored, ever.

    Two checks, because this session produced three separate basis-construction errors -- a
    frozen rank over a wide window (section 27.1), too few Fourier-Bessel terms on a corner
    domain (section 28), and corner-FB on a rectangle, which has no singular corner at all.
    Every one silently degraded the objective by orders while looking fine.

    1. `sigma_mu` at or below `target` at every true eigenvalue, COUNTING MULTIPLICITY -- at a
       multiplicity-m level that is `sigma_m`, not `sigma_1`.
    2. `finest_prec * lam >= margin * eps/c`. **This is the BRACKETING criterion, not the
       multiplicity one.** A bracket of width `prec*lam` is usable when it is wider than the
       well's flat bottom, whose scale is `eps/c` -- below that the curve is flat and carries
       no signal. Section 9.4's `20 * eps/c` is a different question (can (C1)+(C2) separate a
       PAIR) and importing it here would apply a bound from a phase that is not under test.
       Measured, the two differ by exactly that factor: the finest admissible `prec` is
       5e-16 to 1.2e-15 by the bracketing criterion and 9.6e-15 to 2.4e-14 by the multiplicity
       one, so `prec = 1e-14` is comfortable for bracketing and marginal for multiplicity.
       `mult_prec` is reported for when the multiplicity phase is tested.

    `h_frac` is 0.005: at 0.02 the wall-slope estimate starts sampling the saturated part of
    the curve on the small-`lam` cases. Measured stable over 0.005 down to 2e-4.

    Returns `(ok, rows)`; the caller raises or drops the case.
    """
    p = case.pencil
    lv, mult = np.unique(case.truth, return_counts=True)
    rows, ok = [], True
    for x, m in zip(lv, mult):
        s_ = p.tensions(float(x))
        smu = float(s_[m - 1])
        after = float(s_[m]) if len(s_) > m else np.inf
        h = h_frac * abs(x)
        c = 0.5 * (p.tension(x + h) + p.tension(x - h)) / h
        e_c = smu / c if c else np.inf
        headroom = (finest_prec * x) / (margin * e_c) if np.isfinite(e_c) else 0.0
        good = smu <= target and headroom >= 1.0
        ok &= good
        rows.append(dict(lam=float(x), mult=int(m), sigma_mu=smu, gap_after=after,
                         c=float(c), eps_over_c=float(e_c), headroom=float(headroom),
                         brack_prec=float(e_c / x), mult_prec=float(20 * e_c / x),
                         ok=bool(good)))
    return ok, rows


def assert_smooth(case, prec, *, n_seed=16, doublings=4, settle=0.10):
    """PRECONDITION: does this objective admit a Lipschitz bound at the scales the search needs?

    The companion to `assert_floor`, and it checks the OTHER premise. `assert_floor` asks
    whether the basis resolves the eigenfunctions; this asks whether the tension curve between
    them is smooth enough for an exclusion argument to mean anything. A search whose whole
    method rests on a slope bound cannot be scored on a curve that has none -- it will decline,
    correctly, and the decline is a property of the case.

    The test is the searcher's own bootstrap: seed uniformly, double, and see whether the
    largest observed secant STOPS GROWING. That is not a proxy for the premise, it is the
    premise.

    **Why this is needed.** The gate found the multiplicity-4 split cases at `65*pi^2` reporting
    `slope never settled` with the largest secant running 0.64 -> 61 -> 261 across doublings,
    against a measured wall slope of 0.64 on the same curve. The cause is narrow spurious
    structure in the PLATEAU, 0.24 away from any eigenvalue: at `k = 25.0555441` the tension
    reads 0.074 against 0.173 one part in 1e8 away, a 2.3x dip in a window of width 1e-8. The
    floor there is healthy (3.4e-15) and `assert_floor` passes, so the floor check cannot see
    this at all.

    The rank was the obvious suspect and is not the whole story: `s_r/s_0 = 2.27e-15` sits right
    at `SUITE_RTOL = 1.49e-15` with `s_{r+1}/s_0 = 1.3e-15` just below it, so the threshold rank
    flips 85 <-> 86 on a one-ulp change in `lam` and each flip moves `sigma_1` by 2.3x -- but
    FREEZING the rank over the window does not fix it (max secant 700 against 261). So this is
    basis or collocation adequacy at this window, not rank policy, and it is left as a measured
    limitation rather than papered over.

    Returns `(ok, history)`.
    """
    p = case.pencil
    lo, hi = float(case.a), float(case.b)
    k_lo, k_hi = np.sqrt(lo), np.sqrt(hi)
    ks = list(np.linspace(k_lo, k_hi, n_seed))
    vals = {k: p.tension(k * k) for k in ks}
    hist, previous = [], 0.0
    for _ in range(doublings):
        arr = np.array(sorted(vals))
        v = np.array([vals[k] for k in arr])
        dk = np.diff(arr)
        keep = dk > 1e-9 * (k_hi - k_lo)
        raw = float(np.max(np.abs(np.diff(v)[keep] / dk[keep]), initial=0.0))
        hist.append(raw)
        if previous > 0.0 and raw <= (1.0 + settle) * previous:
            return True, tuple(hist)
        previous = raw
        for a, b in zip(ks[:-1], ks[1:]):
            m = 0.5 * (a + b)
            vals.setdefault(m, p.tension(m * m))
        ks = sorted(vals)
    return False, tuple(hist)


def diagnostics(case, h_frac=0.02):
    """Per-case facts a filter needs but a search must not see.

    `eps` and `c` at every true eigenvalue, hence `eps/c` -- the resolution limit the PENCIL
    imposes. A method must not be scored as failing below it: section 9.4 measured (C1)
    correct at `d/(eps/c) = 23` and wrong at 0.45 and 0.15, so `20 * eps/c` is the contract,
    and a case tighter than that is a basis limit rather than a search failure.
    """
    p = case.pencil
    lv = np.unique(case.truth)
    out = []
    for x in lv:
        h = h_frac * abs(x)
        eps = p.tension(x)
        c = 0.5 * (p.tension(x + h) + p.tension(x - h)) / h
        out.append((float(x), float(eps), float(c), float(eps / c) if c else np.inf))
    gaps = np.diff(lv) if lv.size > 1 else np.array([np.inf])
    worst = min((g / out[i][3] for i, g in enumerate(gaps)), default=np.inf)
    return dict(levels=out, min_gap=float(gaps.min()), d_over_epsc=float(worst),
                within_contract=bool(worst >= 20.0),
                snap_safe=bool(gaps.min() > SNAP_FLOOR_REL * lv.max()))


# --- the families -------------------------------------------------------------------------
#
# A  isolated       an irrational aspect ratio, so nothing is degenerate or near-degenerate
# B  pair ladder    the headline axis: one separation, swept over decades
# C  cluster 3/4    a fixed cluster SHAPE whose scale sweeps -- gaps in ratio 1 : 2.2 : 1
# D  degeneracy     multiplicity 2, 3, 4, 6, exactly
# E  mixed          isolated levels, pairs and a triple in ONE window
# F  floor spread   rect(8,1), measured 9.2e+04 spread in `eps` within one scan

def family_A(seed=0):
    """`rect(1, 1/phi)`: 11 distinct eigenvalues below 300, minimum relative gap 2.0e-03."""
    return [make_case("A:isolated", 1.0, 1.0 / PHI, 5.0, 300.0, seed=seed)]


#: (2,1) and (1,2) on rect(1+delta, 1), degenerate at 5*pi^2 and splitting like 6*pi^2*delta.
_B_MODES = ((2, 1), (1, 2))
#: Chosen so the ladder STRADDLES each `prec`, not merely spans decades. A case only
#: exercises `prec` if the pair must be separated at that `prec` and merged at the looser one:
#:
#:     delta      d/lam      1e-4    1e-8    1e-11   1e-14
#:     1e-4     1.20e-04    SPLIT   SPLIT   SPLIT   SPLIT
#:     1e-6     1.20e-06    merge   SPLIT   SPLIT   SPLIT   <- distinguishes 1e-4 from 1e-8
#:     1e-8     1.20e-08    merge   SPLIT   SPLIT   SPLIT
#:     3e-10    3.60e-10    merge   merge   SPLIT   SPLIT   <- distinguishes 1e-8 from 1e-11
#:     1e-11    1.20e-11    merge   merge   SPLIT   SPLIT
#:     1e-13    1.20e-13    merge   merge   merge   SPLIT   <- NOT USABLE, see below
#:
#:     1e-13    1.20e-13    merge   merge   merge   SPLIT   <- distinguishes 1e-11 from 1e-14
#:     3e-14    3.60e-14    merge   merge   merge   SPLIT
#:
#: `1e-13` and `3e-14` ARE admissible, and an earlier version of this comment wrongly said
#: they were not. The reasoning was that `snap` merges within `1e-13 * lam = 4.9e-12` and
#: `d = 5.9e-12` clears it by only 1.2x -- but **family B does not use snapped truth**. It
#: takes the pair values from `rect_eig` directly, precisely so a controlled separation is not
#: at the mercy of a merge policy. Measured: `rect_eig` puts the two 833 ulps apart at
#: `delta = 1e-13` and 250 apart at `3e-14`, and the pencil has `d/(eps/c) = 138` of headroom
#: at the first. So `prec = 1e-14` CAN be exercised as a requirement here.
B_DELTAS = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 1e-8, 3e-10,
            1e-11, 1e-13, 3e-14)


def family_B(deltas=B_DELTAS, seed=0):
    out = []
    for dl in deltas:
        L = 1.0 + dl
        la, lb = (rect_eig(m, n, L, 1.0) for m, n in _B_MODES)
        d = abs(lb - la)
        out.append(make_case(f"B:d={d:.1e}", L, 1.0, 10.0, 88.0, seed=seed,
                             modes=((1, 1), (2, 1), (1, 2), (2, 2))))
    return out


#: square levels with multiplicity 3 and 4, and the modes that make them
_C_LEVELS = {50: ((1, 7), (5, 5), (7, 1)),
             65: ((1, 8), (4, 7), (7, 4), (8, 1))}
C_DELTAS = (1e-2, 1e-3, 1e-4, 1e-5)


def family_C(deltas=C_DELTAS, seed=0):
    out = []
    for N, modes in _C_LEVELS.items():
        lam = N * PI2
        for dl in deltas:
            L = 1.0 + dl
            lv = sorted(rect_eig(m, n, L, 1.0) for m, n in modes)
            spread = lv[-1] - lv[0]
            out.append(make_case(f"C:m={len(modes)},s={spread:.1e}", L, 1.0,
                                 lam * 0.97, lam * 1.03, seed=seed, modes=modes))
    return out


#: 325 (multiplicity 6) is NOT here -- 356 columns and far the most expensive
#: case. It lives in `family_stress`, run once rather than swept.
_D_LEVELS = {25: 2, 50: 3, 65: 4}


def family_D(seed=0):
    out = []
    for N, mult in _D_LEVELS.items():
        lam = N * PI2
        out.append(make_case(f"D:mult={mult}", 1.0, 1.0, lam * 0.97, lam * 1.03, seed=seed))
    return out


# Family E (rect(1.001, 1) on [150, 340], 29 eigenvalues in 8 groups) is DROPPED from the
# swept set. Two reasons, and the second is the one that matters:
#
#   * it is by far the most expensive case -- 29 eigenvalues, 178 columns, and a sweep that
#     took 1049 s against 137-190 s for a family B case;
#   * every method failed it at the flat 1500-evaluation cap, and section 29.2 showed that was
#     the CAP, not the methods: raising it to 4000 has `lipschitz` succeed at 1538 evaluations,
#     38 over the line. So as configured it measured the budget policy rather than the
#     algorithms.
#
# What it was good for is recorded in section 29.2: a dense window is what exposes a flat
# evaluation cap as the wrong instrument, because the cap has to scale with the number of
# eigenvalues in the window. Reinstate it only with a per-eigenvalue budget.


# Family F (rect(8,1), "floor variation") is DROPPED. Its premise was a 9.2e+04 spread in the
# tension floor, which section 27.1 showed to be a frozen-rank artifact: with threshold rank the
# same case reaches 1.1e-12 at every level and there is no spread left to test.


def family_stress(seed=0):
    """Multiplicity 6 at `N = 325`. One stress case, NOT part of the swept families: 352
    columns and 1.45 s per evaluation, against 6-82 ms for everything else."""
    lam = 325 * PI2
    return [make_case("S:mult=6", 1.0, 1.0, lam * 0.99, lam * 1.01, seed=seed)]


def all_families(seed=0):
    return {"A": family_A(seed), "B": family_B(seed=seed), "C": family_C(seed=seed),
            "D": family_D(seed)}


# --- the crossover sweep: a CROSS PRODUCT against the prec ladder ---------------------------
#
# `docs/design/candidates.md` and `bracket_score.PREC_LADDER`. Every candidate is scored at six
# precisions, and a separation is only informative relative to the `prec` it is judged against,
# so the separation ladder is dialled PER PRECISION rather than once globally. Section 29
# recorded results that were IDENTICAL at 1e-8, 1e-11 and 1e-14 because one global ladder left
# the requirement unchanged at all three -- the sweep measured cost while claiming capability.

#: Widest gate window. Holds `50*pi^2` (multiplicity 3: (1,7), (5,5), (7,1)), which is the case
#: the gate was widened for and which `test_gate` asserts it still covers.
WIDE_WINDOW = (10.0, 520.0)

#: The three windows the sweep runs on, coarsest structure first.
CROSSOVER_WINDOWS = {
    "pair":    (10.0, 88.0),                    # 4 eigenvalues, one degenerate pair
    "cluster": (50 * PI2 * 0.97, 50 * PI2 * 1.03),   # the multiplicity-3 level alone
    "wide":    WIDE_WINDOW,                     # 30+ eigenvalues, 15 degenerate pairs
}


def modes_in_window(a, b, max_mode=64):
    """Every `(m, n)` whose square eigenvalue lies in `[a, b]`. Complete by construction.

    Used so truth can come from `rect_eig` DIRECTLY rather than through `rect_spectrum`, which
    returns `snap(...)` and merges within `1e-13 * lam`. At the bottom of the `prec` ladder the
    dialled separation is `5e-15` relative -- two orders BELOW the snap floor -- so snapped
    truth would silently merge exactly the pair the case exists to split, handing back an exact
    degeneracy wearing the split-pair label.
    """
    out = []
    for m in range(1, max_mode + 1):
        if PI2 * m * m > b:
            break
        for n in range(1, max_mode + 1):
            lam = PI2 * (m * m + n * n)
            if lam > b:
                break
            if lam >= a:
                out.append((m, n))
    return tuple(out)


def splitting_rates(a, b, max_mode=64):
    """Square levels in `[a, b]` that SPLIT under `delta`, with their splitting RATE.

    On `rect(1+delta, 1)`, `1/L**2 ~ 1 - 2*delta`, so

        lam(m, n) = pi^2*(m^2/L^2 + n^2) ~ pi^2*(m^2 + n^2) - 2*pi^2*delta*m^2

    -- every mode of a level shifts DOWN by an amount proportional to its own `m^2`. So a
    level splits according to the distinct `m^2` values it contains, and its minimum adjacent
    relative gap grows at

        r = 2 * min(diff(sorted(m^2))) / (m^2 + n^2)        giving   d/lam = r*delta.

    **This is not the same as the pair formula, and the difference is not small.** An earlier
    version computed `r` from the outermost pair, `2*|n^2 - m^2|/(m^2 + n^2)` -- which is
    `CHLADNI_V1.md` section 5.4's expression and is correct for a level holding exactly two
    modes. At `50*pi^2` the modes are (1,7), (5,5) and (7,1), i.e. `m^2` in {1, 25, 49}: the
    outermost pair spans 48 but (5,5) sits exactly halfway, so the minimum adjacent gap grows
    at 24, **half** the outer rate. Every cluster case dialled with the pair formula landed at
    exactly 0.5x its intended band, which is how this was found.

    Returned ascending in `r`, so `[0]` splits SLOWEST and is the tightest at any `delta`.
    """
    levels = {}
    for m in range(1, max_mode + 1):
        for n in range(1, max_mode + 1):
            N = m * m + n * n
            lam = PI2 * N
            if a <= lam <= b:
                levels.setdefault(N, set()).add(m * m)
    out = []
    for N, m2 in levels.items():
        if len(m2) < 2:
            continue                      # a level with one distinct m^2 does not split
        step = int(np.min(np.diff(sorted(m2))))
        out.append((2.0 * step / N, int(N), float(PI2 * N), tuple(sorted(m2))))
    return sorted(out)


def tightest_pair(a, b):
    """The level that governs the window's tightest RELATIVE separation. Dial on THIS one.

    Two ways the obvious choice is wrong, both measured while building this sweep:

    * **It is not the pair you would think to dial.** On `[10, 520]` the obvious handle is
      (2,1)/(1,2) at `r = 1.2`, but (4,5)/(5,4) sits at `r = 0.439` -- it splits at 0.366
      times the rate, so at any `delta` its separation is **2.7x tighter than intended**.
    * **Tightest means RELATIVELY tightest.** (4,5)/(5,4) at `lam = 404.7` is relatively 2.7x
      tighter than (2,1)/(1,2) at `lam = 49.3` while being absolutely 3x WIDER. Purity is
      keyed to `span > prec*lam`, so the relative one is the binding pair, and an absolute
      comparison names the wrong level.
    """
    rates = splitting_rates(a, b)
    return rates[0] if rates else None


def delta_for(band, prec, a, b):
    """The `delta` placing the window's tightest separation at `band * prec`, relatively.

    First order is enough -- the perturbation is at most `1e-2` and usually far smaller -- and
    exactness is not wanted anyway: every case is classified by the separation it ACHIEVED, so
    a dial that is 1% off is recorded at what it actually produced rather than corrected.
    """
    tp = tightest_pair(a, b)
    if tp is None:
        raise ValueError(f"no degenerate pair in [{a}, {b}] to dial")
    return band * prec / tp[0]


def crossover_cases(prec, window="pair", bands=(0.5, 0.8, 1.3, 3.0), seed=0):
    """One case per band at one `prec`: two that must MERGE, two that must SPLIT.

    Straddling is the whole point. A method that always merges and one that always splits both
    look correct on a sweep that stays on one side of the licence, and with a single global
    separation ladder the sweep silently stays on one side at most of the six precisions.

    Truth comes from `rect_eig` through `modes_in_window`, never from `rect_spectrum`.
    """
    a, b = CROSSOVER_WINDOWS[window] if isinstance(window, str) else window
    out = []
    for band in bands:
        dl = delta_for(band, prec, a, b)
        L = 1.0 + dl
        if L == 1.0:
            # The dial rounded to nothing: `1 + delta` is not representable at this band and
            # precision, so the case would be an EXACT degeneracy wearing the split-pair
            # label. Skip it and let the sweep report the cell as empty.
            continue
        out.append(make_case(f"X:{window}/b={band:g}/p={prec:.0e}", L, 1.0, a, b,
                             seed=seed, modes=modes_in_window(a, b)))
    return out


def split_degeneracy_cases(prec, bands=(0.5, 1.3), seed=0, levels=_D_LEVELS):
    """Split counterparts of the exact-degeneracy levels, dialled at one `prec`.

    The merge/split decision has to be exercised on MULTIPLETS and not only on pairs: a
    cluster of three with unequal gaps is measurably harder than a pair (3 of 8 against 6 of
    12 for one method in section 29), and it is the regime a shape optimization crosses
    whenever eigenvalues meet.

    For a multiplet "the separation" is the MINIMUM ADJACENT GAP within the cluster, which is
    what the multiple of `prec` refers to, and it is what `classify_separation` reads back.
    """
    out = []
    for N, mult in levels.items():
        lam = N * PI2
        a, b = lam * 0.97, lam * 1.03
        tp = tightest_pair(a, b)
        if tp is None:
            continue
        for band in bands:
            dl = band * prec / tp[0]
            L = 1.0 + dl
            if L == 1.0:
                continue
            out.append(make_case(f"X:split{mult}/b={band:g}/p={prec:.0e}", L, 1.0, a, b,
                                 seed=seed, modes=modes_in_window(a, b)))
    return out


def cluster_diameter_ratio(case, prec):
    """A cluster's DIAMETER over the merge licence. The quantity `S(prec)` actually keys on.

    `crossover_cases` and `split_degeneracy_cases` dial the MINIMUM ADJACENT GAP to a multiple
    of `prec`, because that is what `classify_separation` reads and what decides whether two
    neighbours must be told apart. But `S(prec)` groups by DIAMETER, and for an `m`-member
    cluster the diameter is about `(m-1)` times the dialled gap -- so dialling the gap does not
    control the licence, and the two coincide only for a pair.

    Measured on the multiplicity-3 cluster at `50*pi^2`:

        band   min gap      diameter / licence
        0.5    0.5*prec           1.0000        <- exactly on the boundary
        0.8    0.8*prec           1.6

    A ratio of 1.0 is not a hard case, it is an UNDECIDED one: whether the cluster is one level
    or two is settled by a comparison at the 1e-4 relative level, and a scorer and a solver can
    round it differently while both behave correctly. Cases landing there should be reported,
    not scored -- the same rule the three tiers already apply to floors and slope bounds.

    This is `CHLADNI_V1.md` section 5.4's trap in a form the document does not name: it warns
    that the tightest PAIR is not the one you are dialling, and for a cluster the DIAMETER is
    not the thing you are dialling either.
    """
    t = np.unique(np.asarray(case.truth, dtype=float))
    if t.size < 2:
        return 0.0
    return float((t[-1] - t[0]) / (prec * max(abs(t[-1]), 1.0)))


def multiplicity_undecided(case, prec, *, reach=2.0, tol=0.02):
    """Is this cluster's MULTIPLICITY undetermined at this precision, whatever the method?

    Yes when its diameter lies between the merge licence and the wide bracket's reach --
    `1 < diameter/(prec*lam) < reach`. Such a cluster is too wide to be one level of `S(prec)`,
    and its members sit closer together than the flanks a multiplicity read needs, so there is
    no window that is outside the level and inside the gap to the next one. Measured on the
    multiplicity-3 cluster at `50*pi^2`, with `chladni.mps.multiplicity.WIDE_FACTOR = 2`:

        diameter/licence   0.40   1.00   1.60   2.60   4.00   6.00
        verdict            pass   FAIL   FAIL   pass   pass   pass

    The band is bounded by the reach on one side and the licence on the other, exactly as the
    argument predicts, so this is a property of the configuration rather than a tuning failure.
    Below the licence the cluster is one level and the reach spans it; above the reach its
    members are further apart than the flanks and each is read alone.

    **How these cases come to sit there is worth recording.** `crossover_cases` dials the
    MINIMUM ADJACENT GAP to a multiple of `prec`, because that is what decides whether two
    neighbours must be told apart -- but `S(prec)` groups by DIAMETER, and for an `m`-member
    cluster the diameter is about `(m-1)` times the dialled gap. So dialling the gap does not
    control the licence, and at `m = 3, band = 0.5` the diameter lands on it exactly, to four
    figures. That is `CHLADNI_V1.md` section 5.4's trap in a form the document does not name:
    it warns that the tightest PAIR is not the one you are dialling, and for a cluster the
    DIAMETER is not the thing you are dialling either.
    """
    # `tol` on the lower edge because a case dialled ONTO the licence lands a hair either
    # side of it: at `m = 3, band = 0.5` the ratio comes out at 0.99999999999, and a strict
    # `>= 1` would score the most ambiguous case in the set as if it were decided.
    r = cluster_diameter_ratio(case, prec)
    return (1.0 - tol) <= r <= float(reach)


def gate_cases(prec, seed=0):
    """Everything the standing gate runs at one `prec`: the sweep, the splits, the degeneracies.

    Exact degeneracy is swept as a CLASS rather than as one case -- the crossover sweep varies
    a pair's separation and never reaches zero, while at `delta = 0` every level may be
    genuinely multiple.
    """
    out = []
    for window in CROSSOVER_WINDOWS:
        out += crossover_cases(prec, window=window, seed=seed)
    out += split_degeneracy_cases(prec, seed=seed)
    out += family_D(seed=seed)
    return out
