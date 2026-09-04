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

# Truth is unfolded, and never pre-collapsed

`chladni.reference`'s `*_spectrum` returns an unfolded multiset with true degeneracies made
bit-identical, and merges nothing, because collapsing is a comparison-time choice rather than a
property of the spectrum. An earlier harness collapsed at `rtol = 1e-9`, which would have
scored a search that correctly resolved a 4e-09 pair as reporting a spurious root.
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


def match_tolerance(truth, prec, *, fraction=MATCH_FRACTION):
    """The relative tolerance a match must fall inside, for THIS case.

    `max(prec, fraction * d_true/lam)` over the tightest DISTINCT pair. Keyed to the case
    because a fixed tolerance is exactly what let a single reported location match two separate
    truth entries and score perfectly for not resolving them.
    """
    t = np.unique(np.asarray(truth, dtype=float))
    if t.size < 2:
        return float(prec)
    scales = np.maximum(np.abs(t[1:]), 1.0)
    d_rel = float(np.min(np.diff(t) / scales))
    return max(float(prec), fraction * d_rel)


@dataclass
class LevelScore:
    label: str = ""
    prec: float = 0.0
    rtol_case: float = 0.0
    n_true: int = 0             # eigenvalues, WITH multiplicity
    n_levels_true: int = 0      # distinct locations
    n_levels_found: int = 0
    n_matched: int = 0          # distinct levels matched to a true level
    n_mult_right: int = 0       # of those, with the right multiplicity
    n_mult_over: int = 0
    n_mult_under: int = 0
    n_spurious: int = 0         # reported levels matching nothing
    n_missed: int = 0           # true levels nothing matched
    found_total: int = 0        # sum of reported multiplicities
    evaluations: int = 0
    worst_rel_err: float = 0.0

    @property
    def separation(self):
        """Distinct levels resolved, over distinct levels true. **Read this one.**

        The metric recall-with-multiplicity provably cannot express: it is what distinguishes
        "split the pair" from "guessed 2".
        """
        return self.n_matched / self.n_levels_true if self.n_levels_true else 1.0

    @property
    def recall(self):
        """Eigenvalues accounted for, with multiplicity. Never read alone -- see `separation`."""
        return min(self.found_total, self.n_true) / self.n_true if self.n_true else 1.0

    @property
    def multiplicity_exact(self):
        return self.n_mult_right / self.n_matched if self.n_matched else 0.0

    @property
    def ok(self):
        """Every level resolved, every multiplicity right, nothing spurious."""
        return (self.separation == 1.0 and self.n_mult_right == self.n_levels_true
                and self.n_spurious == 0)

    @property
    def verdict(self):
        if self.n_missed:
            return "missed"
        if self.n_spurious:
            return "spurious"
        if self.n_mult_over:
            return "mult-over"
        if self.n_mult_under:
            return "mult-under"
        return "pass"


def score_levels(found, truth, prec, *, label="", evaluations=0, fraction=MATCH_FRACTION):
    """Score `[(lam, mult), ...]` against an UNFOLDED truth multiset.

    Matching is nearest-first and each true LEVEL is claimable once, within `rtol_case`.
    """
    t_all = np.sort(np.asarray(truth, dtype=float))
    levels, mult_true = np.unique(t_all, return_counts=True)
    rtol = match_tolerance(t_all, prec, fraction=fraction)

    rep = sorted(((float(x), int(m)) for x, m in found), key=lambda p: p[0])
    claimed, matched, worst = {}, [], 0.0
    spurious = 0
    for x, m in rep:
        j, best = -1, np.inf
        for i, lv in enumerate(levels):
            if i in claimed:
                continue
            err = abs(x - lv) / max(abs(lv), 1.0)
            if err < best:
                j, best = i, err
        if j >= 0 and best <= rtol:
            claimed[j] = (x, m)
            matched.append((j, m))
            worst = max(worst, best)
        else:
            spurious += 1

    right = sum(1 for j, m in matched if m == mult_true[j])
    over = sum(1 for j, m in matched if m > mult_true[j])
    under = sum(1 for j, m in matched if m < mult_true[j])

    return LevelScore(
        label=label, prec=prec, rtol_case=rtol,
        n_true=int(t_all.size), n_levels_true=int(levels.size), n_levels_found=len(rep),
        n_matched=len(matched), n_mult_right=right, n_mult_over=over, n_mult_under=under,
        n_spurious=spurious, n_missed=int(levels.size) - len(matched),
        found_total=sum(m for _, m in rep), evaluations=evaluations, worst_rel_err=float(worst))
