"""Controls for the LEVEL scorer. `notes/landscape_findings.md` section 27.2's third fix.

TEST SUPPORT ONLY. Each must fail, and the first one is the whole reason this file exists: a
metric that rewards a method for inflating multiplicities instead of resolving levels is worse
than no metric, because it reads green.
"""
from __future__ import annotations

import numpy as np

__all__ = ["always_report_mult_2", "never_split", "collapse_everything", "inflate_by_one",
           "ADVERSARIES"]


def always_report_mult_2(truth, prec):
    """Report each RESOLVED-AT-`prec` location with multiplicity 2, whatever is there.

    The control section 27.2 says "should have existed from the start". Under the old metric
    this scored a PERFECT recall of 1.000 with zero spurious on a pair it never split, while
    the honest answer -- one root, multiplicity 1 -- was penalised at recall 0.500. If it
    scores well here the metric is broken rather than the control.
    """
    t = np.sort(np.asarray(truth, dtype=float))
    keep, cur = [], t[0]
    for x in t[1:]:
        if x - cur > prec * max(abs(x), 1.0):
            keep.append(cur)
            cur = x
    keep.append(cur)
    return [(float(x), 2) for x in keep]


def never_split(truth, prec):
    """Report one location per `prec`-level with the multiplicity the merge implies.

    Honest about what it did and still wrong wherever the pencil could have resolved more. It
    is here as the near-miss: it differs from a correct answer only in `separation`, which is
    exactly the metric recall cannot express.
    """
    t = np.sort(np.asarray(truth, dtype=float))
    groups, cur = [], [t[0]]
    for x in t[1:]:
        if x - cur[0] <= 1e-6 * max(abs(x), 1.0):
            cur.append(x)
        else:
            groups.append(cur)
            cur = [x]
    groups.append(cur)
    return [(float(np.mean(g)), len(g)) for g in groups]


def collapse_everything(truth, prec):
    """One level holding the whole window. Must fail on separation and on spurious placement."""
    t = np.asarray(truth, dtype=float)
    return [(float(np.mean(t)), int(t.size))]


def inflate_by_one(truth, prec):
    """The correct answer with every multiplicity one too high. Must fail on `mult-over`."""
    lv, mult = np.unique(np.asarray(truth, dtype=float), return_counts=True)
    return [(float(x), int(m) + 1) for x, m in zip(lv, mult)]


#: Each must fail, by the criterion named.
ADVERSARIES = {
    "always_report_mult_2": (always_report_mult_2, "mult-over"),
    "collapse_everything":  (collapse_everything, "missed"),
    "inflate_by_one":       (inflate_by_one, "mult-over"),
}
