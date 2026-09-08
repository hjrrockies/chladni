"""Stage 4 on `candidates.md`'s scored seam -- `chladni.mps.search` plus the factory.

TEST SUPPORT ONLY, and TEMPORARY. The pipeline moved into the package; what stays is the
`v1_solve` factory, which wraps a bare tension callable in an `Objective` so the candidate is
measured on the same instrument as the baselines, and which is the only user of `Declined`
here -- which is why `search.py` imports nothing from `objective`.

The whole module namespace is carried across rather than an explicit list, on the same
reasoning as `v1_bracket.py`: a hand-written list is the grep-visible subset, and this file's
privates (`_collapse_ambiguous`, `_spanning_bracket`, `_slope_from_cache`) are exactly the
kind a later ablation reaches for.
"""
import chladni.mps.search as _promoted
from chladni.mps.search import *  # noqa: F401,F403

from objective import Declined

# Every module-level name, privates included; dunders skipped so this file keeps its own
# `__doc__` and the `__all__` below.
globals().update({k: v for k, v in vars(_promoted).items() if not k.startswith("__")})

__all__ = ["Level", "SolveResult", "SolveDiagnostics", "solve", "v1_solve"]


def v1_solve(**kw):
    """The candidate on the stage-4 seam: `solve(tensions, a, b, *, prec, modes=None)`.

    `modes` is accepted and ignored -- Phase D is what would use it. Declining propagates as a
    `Declined` value rather than an empty list, because an empty list looks like success.
    """
    def fn(tensions, a, b, *, prec, modes=None, objective=None, expected_count=None):
        # `v1_solve(collapse=False)` reaches the ablation through **kw.
        from objective import Objective
        obj = objective if objective is not None else Objective(
            tensions, a, b, prec=prec, variable="lam", expected_count=expected_count)
        res = solve(obj, prec=prec, **kw)
        fn.last = res
        if res.declined:
            return Declined(res.diagnostics.decline_reason,
                            partial=[(x.lam, x.mult) for x in res.levels],
                            diagnostics=res.diagnostics)
        return [(x.lam, x.mult) for x in res.levels]

    fn.last = None
    return fn
