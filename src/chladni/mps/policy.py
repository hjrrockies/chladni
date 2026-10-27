"""`Policy`: which rank or regularization a search stage reads the pencil under.

The MPS search has stages that want different things from the same pencil
(`notes/triangle_apex_planner.md` sections 9-12, 2026-10-27). Bracketing wants a CONTINUOUS
`sigma_1(lam)` and no deep digits. Refinement and the multiplicity count want the deep digits near
one root, with no rank change across that root's neighbourhood. Certification wants a coefficient
vector, and any rank gives it a valid bound. The shipped threshold rank serves none of them well
on a basis with a noise tail: its cut lands inside the tail, rounding decides which noise
directions survive, and on a lightning basis `sigma_1` jumps ~10% between neighbouring samples
until bracketing's seed declines.

So a stage names the policy it reads under, and the pencil and the objective key their caches by
it. Without the key, two policies' values at one `lam` would be mixed silently -- every cache was
keyed by `lam` alone.

Three policies: the pencil's OWN (whatever it was built with; `None` everywhere, so every
existing call is unchanged); a FROZEN rank (`reg="svd"`'s truncation at a given count); and the
coarse PENALTY bracketing reads, `([A_B; tau I], A_I)` with `tau = tau_rel x` the stack's spectral
norm, which chooses no rank and so cannot flip one. `tau_rel` defaults to `sqrt(eps)`, the geometric
mean of 1 and the arithmetic: ~3 decades above the worst noise tail measured (2e-11, five domains,
`notes/knee_generalize_results.txt`) and ~7 below a tension plateau (~0.1-0.3). It does not depend
on `prec`: bracketing does not need the deep digits.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

#: The coarse penalty's default weight relative to the stack's spectral norm.
TAU_REL = float(np.sqrt(np.finfo(float).eps))

#: A singular value of the stack that moves by more than this RELATIVE amount between two nearby
#: `lam` is rounding, not basis content. The noise tail moves ~10-40% under a 1e-12 change; the
#: directions above it move < 1e-6 (`notes/triangle_apex_planner.md` sections 9, 11). The stable
#: rank came out the same from 1e-2 to 5e-2; 1e-3 cut real directions.
STABLE_TOL = 1e-2

#: How far apart (relative) the two spectra may be. The head of the spectrum is still unmoved at
#: 3e-4 and starts moving at 1e-3 (`notes/stable_rank_free.py`, lam_8 of the apex lightning rung),
#: so a root's CELL is `lam (1 +/- CELL_RHO)`: the rank frozen there is read from two of the
#: search's own samples inside it.
CELL_RHO = 3e-4

__all__ = ["Policy", "TAU_REL", "STABLE_TOL", "CELL_RHO", "stable_rank"]


def stable_rank(s_a, s_b, cap=None, tol=STABLE_TOL):
    """The leading singular values that agree between two nearby spectra `s_a`, `s_b` (each the
    stack's, descending): the count before the first index that moves by more than `tol`
    relative, at most `cap`. Vectors in, an integer out; no matrix is needed."""
    a, b = np.asarray(s_a, dtype=float), np.asarray(s_b, dtype=float)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    with np.errstate(divide="ignore", invalid="ignore"):
        moved = ~(np.abs(a - b) <= tol * np.abs(a))
    r = int(np.argmax(moved)) if moved.any() else n
    return r if cap is None else min(r, int(cap))


@dataclass(frozen=True)
class Policy:
    """How a stage reads the pencil. Hashable; `key` is what caches are keyed by.

    Build with `Policy.pencil()` or `Policy.frozen(rank)`. `normalize` maps the pencil's own
    policy to None, so a cache keyed by `normalize(policy)` keeps every default key exactly as it
    was before policies existed."""

    kind: str = "pencil"
    rank: Optional[int] = None
    tau_rel: Optional[float] = None

    def __post_init__(self):
        if self.kind not in ("pencil", "rank", "penalty"):
            raise ValueError(f"unknown policy kind {self.kind!r}; expected 'pencil', 'rank' or "
                             f"'penalty'")
        if self.kind == "rank":
            if self.rank is None or int(self.rank) != self.rank or int(self.rank) < 1:
                raise ValueError(f"a frozen rank must be a positive integer, got {self.rank!r}")
        elif self.rank is not None:
            raise ValueError(f"the {self.kind!r} policy takes no rank")
        if self.kind == "penalty":
            if self.tau_rel is None or not np.isfinite(self.tau_rel) or not self.tau_rel > 0:
                raise ValueError(f"a penalty needs tau_rel > 0, got {self.tau_rel!r}")
        elif self.tau_rel is not None:
            raise ValueError(f"the {self.kind!r} policy takes no tau_rel")

    @classmethod
    def pencil(cls):
        """The pencil's own policy: what it was built with (the threshold rank by default)."""
        return cls("pencil")

    @classmethod
    def frozen(cls, rank):
        """Truncate at exactly `rank` directions, at every `lam` it is read at."""
        return cls("rank", int(rank))

    @classmethod
    def penalty(cls, tau_rel=None):
        """The coarse penalty at `tau = tau_rel x` the stack's spectral norm (`TAU_REL`)."""
        return cls("penalty", None, TAU_REL if tau_rel is None else float(tau_rel))

    @property
    def key(self):
        """None for the pencil's own policy, else a tuple: the cache key."""
        if self.kind == "pencil":
            return None
        return (self.kind, self.rank) if self.kind == "rank" else (self.kind, self.tau_rel)

    @staticmethod
    def normalize(policy):
        """None for the pencil's own policy (given as None or `Policy.pencil()`), else the
        policy. Raises on anything that is not a `Policy`."""
        if policy is None:
            return None
        if not isinstance(policy, Policy):
            raise TypeError(f"expected a Policy or None, got {type(policy).__name__}")
        return None if policy.key is None else policy
