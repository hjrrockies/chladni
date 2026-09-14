"""The multiplicity problem ALONE: a bracket, and how many eigenvalues are in it.

TEST SUPPORT ONLY. Ported from `wip/v1-promotion-and-multiplicity` (commits `eeb5287`,
`b54f6f5`) onto the promoted `chladni.mps` modules, plan Phase 3a. The floor and reach rules and
the outside-slope estimator come with Phase 3b, which adds the `rule=` seam they call.

This is the instrument for one question and nothing else. It runs no search,
scores no coverage, prices no evaluations. It hands an estimator a bracket and asks for a count.

# The definition being scored

    mu*(B) = |{ lam in S : lam in B }|,  counted with multiplicity

An estimator is CORRECT on `B` when it returns `mu*(B)`. That is the whole specification, and it
is a set-membership count over a GIVEN interval -- no reference to identity, to policy, or to
precision.

**It is well-posed only because of what stage 1 already guarantees**, which is why the problem is
sited on a bracket rather than on a point:

* PURITY says the true eigenvalues inside an emitted bracket span at most `prec*lam`. So a
  bracket's contents are already a legitimate level of `S(prec)`, and the estimator never decides
  whether merging is permitted -- the bracket embodies that decision. `SUCCESS.md`: "Merging is
  permitted, never required."
* COVERAGE says every eigenvalue lies in some bracket. So `sum_B mu*(B) = |S|`: a correct
  estimator gets the total right on every window BY CONSTRUCTION, and no index can move. That is
  the invariant the whole solver exists to protect, obtained structurally rather than by a guard.

# Three questions that are not the same, and have been answered interchangeably

    how many eigenvalues are exactly COINCIDENT here   a floor test, sigma_j ~ sigma_1
    how many lie within prec*lam of lam*               the merge licence
    how many lie INSIDE B                              <- this, and only this, is mu*(B)

`chladni`'s (C1)+(C2) answers a blur of the first two. `bracket_search`'s `floor_ratio` answers
the first, deliberately and correctly, for a different purpose.

# The three regimes, which are physically different

    coincident    m eigenvalues at one lam. sigma_1..sigma_m all at the basis floor eps.
    unresolved    m eigenvalues inside B, separated by more than eps/c but less than width(B).
                  sigma_j ~ c*d_j: distinct, and well above the floor.
    simple        one eigenvalue in B, its neighbours outside.

**Regime `unresolved` is where every existing rule fails**: a floor test says 1 because they are
not coincident, a ratio-gap test says "however many are small", and neither is asking about `B`.
A ladder that does not dial it measures nothing.

# Declining is a third outcome

An estimator may return `None` for "cannot certify". Per `SUCCESS.md` a decline is neither a pass
nor a failure, and **a capability claim must quote the decline rate beside the pass rate over the
same case set**. A silent wrong count is the only unacceptable outcome.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import rect_suite as rs
from chladni.mps.objective import Objective
from chladni.mps.refine import refine

__all__ = ["MultCase", "MultScore", "ladder_cases", "degenerate_cases",
           "as_built_cases", "asymmetric_cases", "score_estimator",
           "regimes", "REGIMES", "PREC_LADDER", "GAP_LADDER", "gap_rule"]

#: The precision ladder, matching the bracketing tests so the two are comparable.
PREC_LADDER = (1e-6, 1e-8, 1e-10, 1e-12, 1e-14)

#: Adjacent-gap ladder, in units of the merge licence `prec*lam`, which is also the scale that
#: sets the bracket width. Spans both sides of the bracket edge: below 1 the neighbours are
#: inside the bracket and must be COUNTED, above 1 they are outside and must be REJECTED. That
#: crossing is the whole discrimination, and a ladder that stays on one side of it cannot see a
#: method that always merges or always splits.
GAP_LADDER = (0.05, 0.1, 0.3, 0.5, 0.8, 1.3, 2.0, 3.0, 6.0)

REGIMES = ("coincident", "unresolved", "simple")


@dataclass
class MultCase:
    """One bracket, and how many eigenvalues are in it."""

    label: str
    tensions: object            # lam -> ascending sigma vector
    lo: float
    hi: float
    lam_star: float             # the located minimizer inside B, from the real refiner
    mu_true: int                # |{lam in S : lam in B}|, with multiplicity
    regime: str
    prec: float
    truth_in: tuple = ()        # the true eigenvalues inside B, for diagnostics
    modes: object = None
    pencil: object = None

    @property
    def width(self):
        return self.hi - self.lo

    @property
    def span(self):
        """Diameter of the true eigenvalues inside `B`, in units of the merge licence.

        `<= 1` is what stage 1's purity guarantees. A case above 1 is a bracket stage 1 would
        never have emitted, and scoring one measures the harness rather than the estimator.
        """
        if len(self.truth_in) < 2:
            return 0.0
        lic = self.prec * max(abs(self.lam_star), 1.0)
        return float((self.truth_in[-1] - self.truth_in[0]) / lic)


def _regime_of(inside, eps_over_c, lam, prec):
    """Which of the three physical situations this bracket is in.

    Coincidence is judged against the BASIS RESOLUTION `eps/c` and not against a fixed tolerance:
    two eigenvalues closer than that are indistinguishable to this pencil however exactly they
    differ, which is what "at the floor together" means operationally.
    """
    if len(inside) < 2:
        return "simple"
    gaps = np.diff(np.asarray(inside, dtype=float))
    return "coincident" if float(np.max(gaps)) <= eps_over_c else "unresolved"


def _eps_over_c(pencil, lam, h_frac=0.005):
    """The pencil's own resolution at `lam`: the width of the well's flat bottom.

    `h_frac = 0.005` is `rect_suite.assert_floor`'s value, measured stable from 0.005 down to
    2e-4; at 0.02 the slope estimate starts sampling the saturated part of the curve.
    """
    h = h_frac * abs(lam)
    eps = float(pencil.tension(float(lam)))
    c = 0.5 * (pencil.tension(lam + h) + pencil.tension(lam - h)) / h
    return (eps / c) if c > 0 else float("inf")


def _case_from_bracket(label, case, lo, hi, prec, *, locate=True):
    """Wrap one bracket of a `rect_suite` case as a `MultCase`, with `mu*(B)` from truth.

    `lam_star` comes from the REAL refiner rather than from truth, so the estimator receives the
    point it would actually receive. A bracket the refiner rejects is not a bracket the estimator
    would ever see, so it returns `None` rather than being scored.
    """
    truth = rs.complete_truth(case) if hasattr(case, "dims") else np.sort(
        np.asarray(case.truth, dtype=float))
    inside = truth[(truth >= lo) & (truth <= hi)]
    if inside.size == 0:
        return None

    obj = Objective(case.tensions, min(lo, case.a), max(hi, case.b), prec=prec, variable="lam")
    if locate:
        r = refine(obj, lo, hi, prec=prec)
        if not r.accepted:
            return None
        lam_star = float(r.lam)
    else:
        lam_star = float(np.median(inside))

    eoc = _eps_over_c(case.pencil, float(inside[0]))
    return MultCase(
        label=label, tensions=case.tensions, lo=float(lo), hi=float(hi), lam_star=lam_star,
        mu_true=int(inside.size), regime=_regime_of(inside, eoc, lam_star, prec),
        prec=prec, truth_in=tuple(float(v) for v in inside),
        modes=getattr(case, "modes", None), pencil=case.pencil)


def ladder_cases(prec_ladder=PREC_LADDER, gap_ladder=GAP_LADDER, levels=(25, 50, 65)):
    """The ladder: gap x precision x multiplicity, brackets built around known clusters.

    The pencils and the dial come from `rect_suite`, which already knows how to place a cluster
    of chosen multiplicity at a chosen separation and carries closed-form truth for it. What is
    added here is the BRACKET: `[lam* - w, lam* + w]` with `w` the merge licence, which is the
    widest interval stage 1 may emit without failing purity.

    Sweeping `gap` across 1.0 is the point. Below it the cluster is inside one bracket and every
    member must be counted; above it the members fall outside and must not be. An estimator that
    always merges and one that always splits are indistinguishable on a ladder that stays on one
    side.
    """
    out = []
    for prec in prec_ladder:
        for N in levels:
            lam = N * rs.PI2
            a, b = lam * 0.97, lam * 1.03
            tp = rs.tightest_pair(a, b)
            if tp is None:
                continue
            for gap in gap_ladder:
                L = 1.0 + gap * prec / tp[0]
                if L == 1.0:                      # not representable at this precision
                    continue
                case = rs.make_case(f"M:N={N}/g={gap:g}/p={prec:.0e}", L, 1.0, a, b,
                                    modes=rs.modes_in_window(a, b))
                truth = np.sort(np.asarray(case.truth, dtype=float))
                near = truth[np.abs(truth - lam) < 0.02 * lam]
                # Centre on a MEMBER, not on the cluster's median. An even-membered cluster has
                # its median between two members, and a bracket one licence wide around that
                # point can contain nothing at all -- which silently drops the cell rather than
                # scoring it. Measured: centring on the median lost 7 of 9 cells at `65*pi^2`,
                # where the multiplicity is 4.
                centre = float(near[int(np.argmin(np.abs(near - np.median(near))))])
                w = 0.5 * prec * abs(centre)
                mc = _case_from_bracket(case.label, case, centre - w, centre + w, prec)
                if mc is not None:
                    out.append(mc)
    return out


def asymmetric_cases(prec_ladder=(1e-8, 1e-12), levels=(50, 65), offsets=(0.25, 0.4)):
    """Clustered brackets with the located root OFF-CENTRE, which is the normal case.

    `ladder_cases` builds `[lam* - w, lam* + w]`, symmetric about the root. A bracket the search
    emits is not: the refiner puts `lam*` at the minimum, and the bracket's ends are wherever
    exclusion left them. The two agree on every quantity except the one that matters here --
    how far the bracket reaches on each side.

    That difference hid a real defect. The reach rule compared distances against the SHORTER
    side, which rejects an eigenvalue lying inside the bracket on the longer one; measured on an
    as-built bracket holding two eigenvalues with the root near the top, the partner read
    `d/reach` 4.48 against the short side and 1.14 against the long one. Symmetric brackets
    cannot see it, so the ladder scored 50/50 while the pipeline under-counted.
    """
    out = []
    for prec in prec_ladder:
        for N in levels:
            lam = N * rs.PI2
            a, b = lam * 0.97, lam * 1.03
            tp = rs.tightest_pair(a, b)
            if tp is None:
                continue
            L = 1.0 + 0.3 * prec / tp[0]
            if L == 1.0:
                continue
            case = rs.make_case(f"A:N={N}/p={prec:.0e}", L, 1.0, a, b,
                                modes=rs.modes_in_window(a, b))
            truth = np.sort(np.asarray(case.truth, dtype=float))
            near = truth[np.abs(truth - lam) < 0.02 * lam]
            centre = float(near[int(np.argmin(np.abs(near - np.median(near))))])
            w = prec * abs(centre)
            for off in offsets:
                # Push the bracket so the root sits off-centre, the way an emitted one does.
                mc = _case_from_bracket(f"A:N={N}/off={off:g}/p={prec:.0e}", case,
                                        centre - w * (1 - off), centre + w * off, prec)
                if mc is not None:
                    out.append(mc)
    return out


def degenerate_cases(prec_ladder=PREC_LADDER):
    """The `coincident` regime, which no dial produces: EXACT degeneracies.

    `rect_suite.family_D` is the unperturbed square, where `25/50/65*pi^2` carry multiplicity 2,
    3 and 4 exactly; the disk and equilateral triangle carry geometric and arithmetic
    degeneracies. These are the cases where `sigma_1..sigma_m` genuinely sit at the basis floor
    together, and they are the only ones a floor test gets right -- so a ladder without them
    cannot show that a new rule keeps what the old one had.
    """
    out = []
    for prec in prec_ladder:
        for case in rs.family_D():
            truth = np.sort(np.asarray(case.truth, dtype=float))
            centre = float(np.median(truth))
            w = 0.5 * prec * abs(centre)
            mc = _case_from_bracket(f"D:{case.label}/p={prec:.0e}", case,
                                    centre - w, centre + w, prec)
            if mc is not None:
                out.append(mc)
    return out


def as_built_cases(prec_ladder=(1e-8, 1e-12), windows=("pair", "cluster")):
    """The other supply: brackets the SEARCH actually emits, paired with `mu*(B)` from truth.

    Scored beside `ladder_cases` because a synthetic bracket is dialled and an emitted one is
    whatever stage 1 chose. Divergence between the two is a finding about the harness, not about
    the estimator, and it is better to see it than to assume it away.
    """
    import bracket_score as bs
    from chladni.mps.bracket import bracket_interval

    out = []
    for prec in prec_ladder:
        for w in windows:
            for case in rs.crossover_cases(prec, window=w):
                obj = bs.objective_for(case, prec)
                res = bracket_interval(obj, prec=prec)
                if res.declined:
                    continue
                for k, br in enumerate(res.brackets):
                    mc = _case_from_bracket(f"{case.label}#{k}", case, br.lo, br.hi, prec)
                    if mc is not None:
                        out.append(mc)
    return out


def regimes(cases):
    """How many cases of each regime a set holds. A ladder missing one measures nothing."""
    return {r: sum(1 for c in cases if c.regime == r) for r in REGIMES}


@dataclass
class MultScore:
    """Capability, not a score: what an estimator can do, and what it declined."""

    label: str = ""
    n: int = 0
    correct: int = 0
    over: int = 0
    under: int = 0
    declined: int = 0
    by_regime: dict = field(default_factory=dict)
    worst: tuple = ()          # (label, mu_hat, mu_true) for the largest miss

    @property
    def wrong(self):
        return self.over + self.under

    @property
    def scored(self):
        return self.n - self.declined

    @property
    def pass_rate(self):
        return self.correct / self.scored if self.scored else 0.0

    @property
    def decline_rate(self):
        return self.declined / self.n if self.n else 0.0

    @property
    def silent_wrong(self):
        """The only unacceptable outcome: a count returned, and wrong."""
        return self.wrong

    def __str__(self):
        r = "  ".join(f"{k}:{v['correct']}/{v['scored']}" for k, v in sorted(self.by_regime.items()))
        return (f"{self.label:<28} pass {self.correct:>3}/{self.scored:<3} "
                f"({self.pass_rate:>5.1%})  declined {self.declined:>3}/{self.n:<3} "
                f"({self.decline_rate:>5.1%})  over {self.over:>3} under {self.under:>3}  | {r}")


def score_estimator(fn, cases, label=""):
    """Run `fn(case) -> int | None` over a case set and report capability by regime."""
    sc = MultScore(label=label, n=len(cases))
    worst_gap = -1
    for c in cases:
        mu = fn(c)
        reg = sc.by_regime.setdefault(c.regime, {"scored": 0, "correct": 0, "over": 0,
                                                 "under": 0, "declined": 0})
        if mu is None:
            sc.declined += 1
            reg["declined"] += 1
            continue
        reg["scored"] += 1
        if mu == c.mu_true:
            sc.correct += 1
            reg["correct"] += 1
        else:
            key = "over" if mu > c.mu_true else "under"
            setattr(sc, key, getattr(sc, key) + 1)
            reg[key] += 1
            if abs(mu - c.mu_true) > worst_gap:
                worst_gap = abs(mu - c.mu_true)
                sc.worst = (c.label, mu, c.mu_true)
    return sc


# --- estimators and controls ------------------------------------------------------------------
#
# An estimator is `fn(case) -> int | None`. `None` is a decline. The controls are not decoration:
# `SUCCESS.md` requires them, and `level_score.py` records that the adversary
# `always_report_mult_2` was "the cheapest possible check that the fix worked, and it should have
# existed from the start". If an adversary scores well the metric is broken and no other result
# from it means anything.


def always(n):
    """Adversary: ignore the evidence and answer `n`. MUST fail the ladder."""
    def fn(case):
        return n
    fn.__name__ = f"always_{n}"
    return fn


def count_below_plateau(frac=0.5):
    """Adversary: count branches below a fixed fraction of the plateau. MUST fail.

    The shape of every rule this project has tried and retired, and it fails for the reason
    `multiplicity.py` already gives: the floor varies five orders within a single scan on a
    corner domain, so no absolute level on `sigma` can serve.
    """
    def fn(case):
        t = np.atleast_1d(np.asarray(case.tensions(case.lam_star), dtype=float))
        return max(int(np.count_nonzero(t <= frac * float(np.median(t)))), 1)
    fn.__name__ = f"below_plateau_{frac:g}"
    return fn


def oracle(case):
    """Positive control: reads truth. MUST pass everything."""
    return case.mu_true


def decliner(case):
    """Positive control: declines everything. MUST come back declined, never as a pass."""
    return None


def gap_rule(case):
    """The shipped (C1)+(C2) rule, on this instrument."""
    from chladni.mps.multiplicity import multiplicity, wide_bracket_for

    xs = np.array(sorted({case.lo, case.lam_star, case.hi,
                          case.lam_star - case.width, case.lam_star + case.width}))
    c = _slope(case)
    wb = wide_bracket_for(xs, case.lam_star, float(case.tensions(case.lam_star)[0]), c,
                          prec=case.prec)
    if wb is None:
        return 1
    return multiplicity(case.tensions, case.lam_star, wide_bracket=wb, prec=case.prec).mu


def _slope(case, frac=0.25):
    """The wall slope at `lam_star`, cancellation-free, from points inside the bracket.

    `c = (v_lo + v_hi)/(hi - lo)`: on a cone the flanks straddle the minimizer, so there is no
    subtraction of nearly-equal quantities and no dependence on where the minimum is. This is
    `bracket.isolation`'s estimator and not the global `L`, which is a bound inflated by a safety
    factor rather than an estimate.
    """
    h = frac * case.width
    lo, hi = case.lam_star - h, case.lam_star + h
    v_lo = float(np.atleast_1d(case.tensions(lo))[0])
    v_hi = float(np.atleast_1d(case.tensions(hi))[0])
    return (v_lo + v_hi) / (hi - lo) if hi > lo else 0.0
