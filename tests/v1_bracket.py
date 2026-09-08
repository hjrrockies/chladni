"""Stage 1 on `candidates.md`'s scored seam -- `chladni.mps.bracket` plus the factory.

TEST SUPPORT ONLY, and TEMPORARY. The algorithm moved into the package; what stays here is the
`v1_brackets` factory, which is apparatus rather than package surface: it wraps a bare tension
callable in an `Objective` so the candidate is measured on the same instrument as the
baselines. It is also the only user of `Declined` in this file, which is why `bracket.py`
imports nothing from `objective`.

Everything else is re-exported so that the promotion changed no test file. The PRIVATES are
read from outside more than usual here -- `test_search.py` binds this module as `V` and reads
`_depth`, `_envelope_min`, `_secants` and `_seed`, and `docs/figures/make_bracket_figures.py`,
which pytest does not collect, reaches `_locator_step` as well -- so the whole module namespace
is carried across rather than an explicit list. That is deliberate: an explicit list is exactly
what a grep-visible subset gives you, and writing one here already produced an ImportError on
a name that does not exist.
"""
import chladni.mps.bracket as _promoted
from chladni.mps.bracket import *  # noqa: F401,F403

from objective import Declined

# Every module-level name, constants and privates included. Dunders are skipped, so this
# module keeps its own `__doc__` and the `__all__` set below.
globals().update({k: v for k, v in vars(_promoted).items() if not k.startswith("__")})

__all__ = ["Bracket", "SearchDiagnostics", "SearchResult", "bracket_interval", "v1_brackets",
           "live_regions", "wall_ladder_ratio", "width_is_pure", "descent_certificate",
           "isolation"]


def v1_brackets(**kw):
    """The candidate, on the stage-1 seam: `brackets(tensions, a, b, *, prec, modes=None)`.

    Wraps the callable in an `Objective` so V1 is scored on the identical instrument as the
    baselines rather than through one built for it.
    """
    def brackets(tensions, a, b, *, prec, modes=None, objective=None, expected_count=None):
        from objective import Objective
        obj = objective if objective is not None else Objective(
            tensions, a, b, prec=prec, variable="k", expected_count=expected_count)
        res = bracket_interval(obj, prec=prec, **kw)
        brackets.last = res
        # Already in the objective's variable -- `bracket_interval` converts on the way out.
        out = [(x.lo, x.hi) for x in res.brackets]
        if res.declined:
            return Declined(res.diagnostics.decline_reason, partial=out,
                            diagnostics=res.diagnostics)
        return out

    brackets.last = None
    return brackets
