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

Two policies today: the pencil's OWN (whatever it was built with; `None` everywhere, so every
existing call is unchanged) and a FROZEN rank (`reg="svd"`'s truncation at a given count). The
coarse penalty for bracketing is the next step's.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

__all__ = ["Policy"]


@dataclass(frozen=True)
class Policy:
    """How a stage reads the pencil. Hashable; `key` is what caches are keyed by.

    Build with `Policy.pencil()` or `Policy.frozen(rank)`. `normalize` maps the pencil's own
    policy to None, so a cache keyed by `normalize(policy)` keeps every default key exactly as it
    was before policies existed."""

    kind: str = "pencil"
    rank: Optional[int] = None

    def __post_init__(self):
        if self.kind not in ("pencil", "rank"):
            raise ValueError(f"unknown policy kind {self.kind!r}; expected 'pencil' or 'rank'")
        if self.kind == "rank":
            if self.rank is None or int(self.rank) != self.rank or int(self.rank) < 1:
                raise ValueError(f"a frozen rank must be a positive integer, got {self.rank!r}")
        elif self.rank is not None:
            raise ValueError("the pencil's own policy takes no rank")

    @classmethod
    def pencil(cls):
        """The pencil's own policy: what it was built with (the threshold rank by default)."""
        return cls("pencil")

    @classmethod
    def frozen(cls, rank):
        """Truncate at exactly `rank` directions, at every `lam` it is read at."""
        return cls("rank", int(rank))

    @property
    def key(self):
        """None for the pencil's own policy, else a tuple: the cache key."""
        return None if self.kind == "pencil" else (self.kind, self.rank)

    @staticmethod
    def normalize(policy):
        """None for the pencil's own policy (given as None or `Policy.pencil()`), else the
        policy. Raises on anything that is not a `Policy`."""
        if policy is None:
            return None
        if not isinstance(policy, Policy):
            raise TypeError(f"expected a Policy or None, got {type(policy).__name__}")
        return None if policy.key is None else policy
