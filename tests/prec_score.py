"""One verdict for one (case, prec) cell of a pipeline benchmark.

TEST SUPPORT ONLY. `tests/test_prec_score.py` pins every verdict on a hand-built result.

# Why not `level_score.score_levels`

`level_score` assigns each true eigenvalue to its NEAREST reported level. That is right for the
question it was built for and wrong for this one: an exact degeneracy reported as two simple
eigenvalues at the same `lam` -- legal, by decision -- hands both truth values to one level and
calls the other spurious. So this scorer matches by CAPACITY: a level declaring multiplicity `m`
may claim `m` eigenvalues, claims are made closest-first, and each eigenvalue is claimed once.

# The acceptance rule

    width <= prec*lam      MAY merge, never MUST      merged and split both pass
    gap   >  prec*lam      MUST split                 a merge fails unless the basis
                                                      cannot resolve multiplicity
    total                  scored everywhere          no unscored band; a count error the
                                                      solve FLAGS is `flagged`, acceptable
    accuracy               within prec, OR reported   `b-excused` only where the basis
                           with a bar that holds      cannot serve `prec`

The excuses come from `tests/basis_sufficiency.py` and nowhere else.

# Association

Each true value `x` can be claimed by a level within `R(x) = max(prec, half the gap to the
nearest OTHER distinct true value)`, relatively. Half the gap rather than `prec` alone, so that
a level which is simply inaccurate is still graded on accuracy -- as `shortfall-silent` or
`b-fail` -- instead of dissolving into a spurious level plus a missed eigenvalue.

A level may also reach as far as the width it DECLARES. Only a collapsed level declares one --
`grouping_ambiguous`, its wide bracket -- and its members lie inside it by construction, so a
cluster it under-counts is a `wrong-total` (the neighbourhood was found, the count was short)
rather than `missed` (which points at the bracketer). Declaring a width buys REACH, never
licence: a declared-wide level whose members span more than `prec*lam` is still a merge above
the licence.

A true value no level claimed is then offered to its NEAREST level if that level has capacity
left. That is how a merge across the licence shows up -- the level counted the eigenvalue, it
just put it in the wrong place -- as distinct from `missed`, where no level is near at all.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Sufficiency", "CellScore", "score_cell", "ACCEPTABLE"]

#: The verdicts a benchmark may accept. Everything else is a failure.
#:
#: `flagged` (2026-10-09): a COUNT error -- spurious, missed, merged across the licence, wrong
#: total -- in a solve that says its counts are uncertain. Right, declined or flagged are the
#: honest outcomes; a silent wrong count is the failure. Before this, an honestly flagged miscount
#: failed every ladder built on this scorer (notes/test_inventory_2026-10-09.md). Flags never
#: excuse ACCURACY: a level claiming to have met `prec` and wrong is `shortfall-silent` regardless.
ACCEPTABLE = frozenset({"pass", "declined", "flagged", "b-excused", "merge-excused",
                        "wrong-total-excused"})

#: The count-error verdicts a count flag turns into `flagged`.
_COUNT_ERRORS = frozenset({"spurious", "missed", "merge-above-licence", "wrong-total"})


@dataclass(frozen=True)
class Sufficiency:
    """Whether a case's basis can serve a `prec`: the only source of an excuse.

    `frozen` is False when the pencil's `rtol` was derived from `prec`. That is reported and
    never turned into an excuse -- it is a configuration error in the benchmark, not a limit of
    the basis.
    """
    accuracy: bool = True
    multiplicity: bool = True
    frozen: bool = True
    reasons: tuple = ()


@dataclass(frozen=True)
class CellScore:
    verdict: str
    prec: float
    n_true: int = 0
    n_levels: int = 0
    found_total: int = 0
    worst_rel_err: float = float("nan")
    detail: str = ""

    @property
    def acceptable(self):
        return self.verdict in ACCEPTABLE


def score_cell(result, truth, prec, sufficiency):
    """Score a `SolveResult` against COMPLETE, UNFOLDED truth in the window."""
    t = np.sort(np.asarray(truth, dtype=float))
    d = result.diagnostics
    if d.declined:
        reason = str(d.decline_reason or "")
        return CellScore("declined" if reason.strip() else "decline-unreasoned", prec,
                         n_true=int(t.size), detail=reason)

    levels = sorted(result.levels, key=lambda L: L.lam)
    lam = np.array([float(L.lam) for L in levels])
    left = [int(L.mult) for L in levels]
    reach = [float(getattr(L, "width", 0.0) or 0.0) / abs(float(L.lam)) if L.lam else 0.0
             for L in levels]
    found_total = int(sum(left))

    distinct = np.unique(t)
    radius = np.empty(t.size)
    for j, x in enumerate(t):
        others = distinct[distinct != x]
        gap = float(np.min(np.abs(others - x))) if others.size else np.inf
        radius[j] = max(prec, 0.5 * gap / abs(x))

    def rel(i, j):
        return abs(lam[i] - t[j]) / abs(t[j])

    owner = [-1] * t.size
    members = [[] for _ in levels]
    for _, i, j in sorted((rel(i, j), i, j) for i in range(len(levels))
                          for j in range(t.size) if rel(i, j) <= max(radius[j], reach[i])):
        if owner[j] < 0 and left[i] > 0:
            owner[j], left[i] = i, left[i] - 1
            members[i].append(j)
    if levels:
        for j in range(t.size):
            if owner[j] < 0:
                i = int(np.argmin(np.abs(lam - t[j])))
                if left[i] > 0:
                    owner[j], left[i] = i, left[i] - 1
                    members[i].append(j)

    errs = [max(rel(i, j) for j in members[i]) for i in range(len(levels)) if members[i]]
    base = dict(n_true=int(t.size), n_levels=len(levels), found_total=found_total,
                worst_rel_err=float(max(errs, default=np.nan)))

    flagged = (not d.certified_counts or getattr(d, "n_ambiguous", 0) > 0
               or any(L.count_uncertain or L.grouping_ambiguous for L in levels))

    def verdict(v, detail=""):
        if flagged and v in _COUNT_ERRORS:
            v, detail = "flagged", f"{v}, flagged by the solve: {detail}"
        return CellScore(v, prec, detail=detail, **base)

    spurious = [float(lam[i]) for i in range(len(levels)) if not members[i]]
    if spurious:
        return verdict("spurious", f"levels at {spurious} claim no eigenvalue")

    unclaimed = [j for j in range(t.size) if owner[j] < 0]
    far = [j for j in unclaimed
           if not any(rel(i, j) <= max(radius[j], reach[i]) for i in range(len(levels)))]
    if far:
        return verdict("missed", f"no level near {[float(t[j]) for j in far]}")

    for i in range(len(levels)):
        if len(members[i]) > 1:
            vals = t[members[i]]
            if vals.max() - vals.min() > prec * abs(vals.max()):
                v = "merge-above-licence" if sufficiency.multiplicity else "merge-excused"
                return verdict(v, f"level at {lam[i]:.15g} spans "
                                  f"{(vals.max() - vals.min()) / abs(vals.max()):.2e} relative")

    if found_total != t.size or unclaimed:
        v = "wrong-total" if sufficiency.multiplicity else "wrong-total-excused"
        return verdict(v, f"declared {found_total} eigenvalues against {t.size}")

    worst = "pass"
    rank = {"pass": 0, "b-excused": 1, "b-fail": 2, "bar-violated": 3, "shortfall-silent": 4}
    for i, L in enumerate(levels):
        err = max(rel(i, j) for j in members[i])
        if err <= prec:
            continue
        dist = max(abs(lam[i] - t[j]) for j in members[i])
        if L.met(prec):
            v = "shortfall-silent"
        elif not (np.isfinite(L.delta_res) and dist <= L.delta_res):
            v = "bar-violated"
        else:
            v = "b-excused" if not sufficiency.accuracy else "b-fail"
        if rank[v] > rank[worst]:
            worst = v
    return verdict(worst)
