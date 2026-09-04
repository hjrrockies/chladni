"""Scoring a LEVEL SET -- `[(lam, mult), ...]` -- against complete closed-form truth.

TEST SUPPORT ONLY. The companion to `bracket_score.py`, which deliberately carries no
multiplicity at all, and it exists because that omission cannot survive stage 3.

# The metric this replaces was blind AND INVERTED

`harness.score` matches greedily at a fixed `rtol = 1e-6` RELATIVE, each truth entry claimable
once, with `unfold` expanding `(x, mult)` into `mult` copies of `x`. So two copies of ONE
location can match two DISTINCT truth entries whenever they lie within `rtol`. Measured at
`lam = 49.35` against a true pair separated by `d`
(`notes/landscape_findings.md` section 27.2):

         d      d/lam   |  reports ONE root, claims mult 2  |  reports ONE root, mult 1
    5.9e-04   1.2e-05   |  recall 0.000  spur 2             |  recall 0.000  spur 1
    5.9e-05   1.2e-06   |  recall 1.000  spur 0  PERFECT    |  recall 0.500  spur 0
    5.9e-06   1.2e-07   |  recall 1.000  spur 0  PERFECT    |  recall 0.500  multX 1
    5.9e-07   1.2e-08   |  recall 1.000  spur 0  PERFECT    |  recall 0.500  multX 1

**A method that never splits the pair and inflates the multiplicity scores perfectly, and the
honest answer is penalised.** It does not merely go blind; it rewards the wrong behaviour, and
it does so below `rtol * lam = 4.9e-05`, which is essentially the adequacy contract boundary
`20 * eps/c = 4e-05` at that pencil. The metric stops working at exactly the line between "the
basis can resolve this, so the solver must" and "it cannot".

Section 27.2 names three fixes, and all three are here.

# 1. The matching tolerance is tied to the CASE

`rtol_case = max(prec, MATCH_FRACTION * d_true / lam)`, so a match must land within a fraction
of the true separation. A fixed tolerance is what let one location claim two truth entries.

# 2. SEPARATION is a first-class metric, distinct from recall

Recall-with-multiplicity provably cannot tell "split the pair" from "guessed 2", because it is
one number measuring two things. So `levels_found / levels_true` is reported separately and is
the number to read when the question is whether a method resolved anything. Recall is kept
because a lost eigenvalue is still the worst failure, but it is never read alone.

# 3. There is an adversarial control that must score badly

`controls.always_report_mult_2` in this module. If it scores well the metric is broken rather
than the control, and section 27.2 calls it "the cheapest possible check that the fix worked,
and it should have existed from the start".

# 4. Acceptance is a LATTICE, not a fixed grouping

        fine end    one reported level per true eigenvalue
        coarse end  one reported level per level of `S(prec)`

and anything between is correct. Merging within `prec` is permitted and never required, and
`bracket_search`'s `certify.py` is explicit that splitting beyond the licence is not an error
either -- so a scorer comparing against ONE grouping penalises whichever end it did not pick.
Measured both ways: against the fine grouping, a legal merge of a pair at half the licence read
as a MISSED level; against the coarse grouping, a correct split of the same pair read as
`mult-under`.

So each TRUE eigenvalue is assigned to the nearest reported level within `rtol_case`, and a
reported level is judged on what it was assigned -- which accepts both ends without accepting a
guess, since `always_report_mult_2` declares 2 where one eigenvalue is assigned.

`S(prec)` is itself `prec`-indexed, deliberately: `search.md` section 2.3 notes that two levels
4e-09 apart are ONE level at `prec = 1e-08` and TWO at `prec = 1e-12`. Truth stays unfolded and
is grouped only at comparison time -- `chladni.reference`'s `*_spectrum` merges nothing, and an
earlier harness that collapsed at a fixed `rtol = 1e-9` would have scored a search which
correctly resolved a 4e-09 pair as reporting a spurious root.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["LevelScore", "score_levels", "match_tolerance", "MATCH_FRACTION"]

#: Fraction of a case's own tightest true separation that a report must land within. A tenth:
#: tight enough that one location cannot claim both members of the pair it failed to split,
#: loose enough that a correct answer at the resolution limit is not rejected for being a few
#: `delta_res` off.
MATCH_FRACTION = 0.1


def _levels_at(truth, prec):
    """Levels of `S(prec)`: greedy grouping with each group's DIAMETER <= prec*lam.

    Anchored to the group's first member, which on a sorted array IS the diameter rule and
    matches `search.md` Phase D and `bracket_score._levels_at`.
    """
    t = np.sort(np.asarray(truth, dtype=float))
    if t.size == 0:
        return np.empty(0), np.empty(0, dtype=int)
    groups, cur = [], [t[0]]
    for x in t[1:]:
        if x - cur[0] <= prec * max(abs(x), 1.0):
            cur.append(x)
        else:
            groups.append(cur)
            cur = [x]
    groups.append(cur)
    return (np.array([np.mean(g) for g in groups]),
            np.array([len(g) for g in groups], dtype=int))


def match_tolerance(truth, prec, *, fraction=MATCH_FRACTION):
    """The relative tolerance a match must fall inside, for THIS case.

    `max(prec, fraction * d/lam)` over the tightest pair of DISTINCT levels of `S(prec)`. Keyed
    to the case because a fixed tolerance is exactly what let a single reported location match
    two separate truth entries and score perfectly for not resolving them.
    """
    t, _ = _levels_at(np.asarray(truth, dtype=float), prec)
    if t.size < 2:
        return float(prec)
    scales = np.maximum(np.abs(t[1:]), 1.0)
    return max(float(prec), fraction * float(np.min(np.diff(t) / scales)))


@dataclass
class LevelScore:
    label: str = ""
    prec: float = 0.0
    rtol_case: float = 0.0
    n_true: int = 0             # eigenvalues, WITH multiplicity
    n_levels_true: int = 0      # levels of S(prec): the COARSE end of the lattice
    n_levels_fine: int = 0      # distinct true values: the FINE end
    n_levels_found: int = 0
    n_matched: int = 0          # reported levels assigned at least one eigenvalue
    n_assigned: int = 0         # true eigenvalues accounted for
    n_mult_right: int = 0       # declared mult == the count assigned to it
    n_mult_over: int = 0
    n_mult_under: int = 0
    n_impure: int = 0           # a level spanning more than prec*lam, or than it declared
    n_declared_wide: int = 0    # levels that DECLARED a span wider than the licence
    n_spurious: int = 0         # reported levels assigned nothing
    n_missed: int = 0           # true eigenvalues assigned to nothing
    found_total: int = 0        # sum of reported multiplicities
    evaluations: int = 0
    worst_rel_err: float = 0.0

    @property
    def coverage(self):
        """Every true eigenvalue accounted for. HARD -- a lost one cannot be recovered."""
        return self.n_assigned / self.n_true if self.n_true else 1.0

    @property
    def separation(self):
        """Reported levels over the COARSE end of the lattice. 1.0 is at least fully resolved;
        more means the answer sits nearer the fine end, which is also correct.

        The metric recall-with-multiplicity provably cannot express: it is what distinguishes
        "split the pair" from "guessed 2".
        """
        return self.n_matched / self.n_levels_true if self.n_levels_true else 1.0

    @property
    def recall(self):
        """Eigenvalues accounted for, with multiplicity. NEVER read alone: all three
        adversaries score 1.00 here."""
        return min(self.found_total, self.n_true) / self.n_true if self.n_true else 1.0

    @property
    def multiplicity_exact(self):
        return self.n_mult_right / self.n_matched if self.n_matched else 0.0

    @property
    def in_lattice(self):
        """Reported count between the two admissible ends. Outside it is implied by another
        failure; this only localizes which."""
        return self.n_levels_true <= self.n_levels_found <= self.n_levels_fine

    @property
    def total_right(self):
        """Declared eigenvalues equal to eigenvalues present. **The one that decides an
        index**: a wrong total shifts `lam_3` onto a different eigenvalue, which is how a
        solver comes to report an accurate but non-consecutive spectrum."""
        return self.found_total == self.n_true

    @property
    def ok(self):
        return (self.coverage == 1.0 and self.n_impure == 0 and self.n_spurious == 0
                and self.n_mult_over == 0 and self.n_mult_under == 0)

    @property
    def verdict(self):
        if self.n_missed:
            return "missed"
        if self.n_impure:
            return "impure"
        if self.n_spurious:
            return "spurious"
        if self.n_mult_over:
            return "mult-over"
        if self.n_mult_under:
            return "mult-under"
        return "pass"


def _unpack(item):
    """`(lam, mult, declared_width)` from a plain tuple or from anything with those attributes.

    A level may DECLARE that it stands for a span wider than `prec*lam` -- `v1_solve` does this
    when a neighbourhood's internal grouping is arbitrary, collapsing it and reporting the
    total, which is the quantity that was never in doubt. A scorer that ignored the declaration
    would mark the level impure for a width it announced, and would be penalising the honest
    answer over a wrong one: the alternative was two narrow levels whose counts summed to 4
    where 3 exist, which is the error that corrupts an index.

    So a declared width is held to: the level may span what it says it spans and no more.
    Plain tuples declare nothing and are held to `prec*lam`, so the adversaries are unaffected.
    """
    lam = getattr(item, "lam", None)
    if lam is None:
        lam, mult = item
        return float(lam), int(mult), 0.0
    return float(lam), int(item.mult), float(getattr(item, "width", 0.0) or 0.0)


def score_levels(found, truth, prec, *, label="", evaluations=0, fraction=MATCH_FRACTION):
    """Score `[(lam, mult), ...]` against an UNFOLDED truth multiset, on the lattice.

    Items may also be objects carrying `lam`, `mult` and a declared `width`; see `_unpack`.
    """
    t_all = np.sort(np.asarray(truth, dtype=float))
    coarse, _ = _levels_at(t_all, prec)
    fine = np.unique(t_all)
    rtol = match_tolerance(t_all, prec, fraction=fraction)

    rep = sorted((_unpack(item) for item in found), key=lambda p: p[0])
    # Per level: the relative radius it may claim, which is the case tolerance or its own
    # declared width, whichever is larger.
    radius = [max(rtol, (w / max(abs(r), 1.0)) if w > 0 else 0.0) for r, _, w in rep]
    assigned = [[] for _ in rep]
    missed = []
    for x in t_all:
        j, best = -1, np.inf
        for i, (r, _, _) in enumerate(rep):
            err = abs(x - r) / max(abs(r), 1.0)
            if err < best:
                j, best = i, err
        if j >= 0 and best <= radius[j]:
            assigned[j].append(float(x))
        else:
            missed.append(float(x))

    matched = [i for i, g in enumerate(assigned) if g]
    right = sum(1 for i in matched if rep[i][1] == len(assigned[i]))
    over = sum(1 for i in matched if rep[i][1] > len(assigned[i]))
    under = sum(1 for i in matched if rep[i][1] < len(assigned[i]))
    # A level may span what it DECLARED, and no more.
    impure = sum(1 for i in matched
                 if max(assigned[i]) - min(assigned[i])
                 > max(prec * max(abs(max(assigned[i])), 1.0), rep[i][2]))
    declared_wide = sum(1 for i in matched if rep[i][2] > 0)
    worst = max((abs(rep[i][0] - float(np.mean(assigned[i]))) / max(abs(rep[i][0]), 1.0)
                 for i in matched), default=0.0)

    return LevelScore(
        label=label, prec=prec, rtol_case=rtol,
        n_true=int(t_all.size), n_levels_true=int(coarse.size), n_levels_fine=int(fine.size),
        n_levels_found=len(rep), n_matched=len(matched),
        n_assigned=int(t_all.size) - len(missed),
        n_mult_right=right, n_mult_over=over, n_mult_under=under, n_impure=impure,
        n_declared_wide=declared_wide,
        n_spurious=sum(1 for g in assigned if not g), n_missed=len(missed),
        found_total=sum(m for _, m, _w in rep), evaluations=evaluations, worst_rel_err=float(worst))
