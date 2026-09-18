"""V1 on `candidates.md`'s scored seams -- the adapters, and nothing else.

TEST SUPPORT ONLY. The algorithm lives in `chladni.mps.{bracket,refine,multiplicity,search}`;
what is here is the apparatus that puts it on the seam a candidate is SCORED on, which is not
package surface and should not become it.

Both factories do the same one job: wrap a bare `tensions` callable in an `Objective` so V1 is
measured on the identical instrument as the baselines rather than through one built for it
(`candidates.md` section 3, "uniform cost accounting"). They are also the only users of
`Declined` on this side of the seam, which is why none of the promoted modules imports it.

This file replaces the `tests/v1_*.py` shims that carried the promotion. Those existed so that
moving the code changed no test; now that the tests point at the package directly, the shims
have no job and the factories needed a home that is not a shim.
"""
from chladni.mps.bracket import bracket_interval
from chladni.mps.objective import Declined, Objective
from chladni.mps.search import solve

__all__ = ["v1_brackets", "v1_solve"]


def v1_brackets(**kw):
    """Stage 1 on the seam: `brackets(tensions, a, b, *, prec, modes=None)`."""
    def brackets(tensions, a, b, *, prec, modes=None, objective=None, expected_count=None):
        obj = objective if objective is not None else Objective(
            tensions, a, b, prec=prec, variable="k", expected_count=expected_count,
            modes=modes)
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


def v1_solve(**kw):
    """Stage 4 on the seam: `solve(tensions, a, b, *, prec, modes=None)`.

    `modes` is CARRIED on the `Objective` and read by nothing yet -- Phase D's interior Gram
    angle is what will use it. It rides on the objective rather than on `solve`'s signature
    because `solve` already receives the objective, and because a second callable beside the
    first would mean a second cache and a charged evaluation count that depends on which was
    consulted (`bracket_score.run_bracketer` makes the same argument for `objective=`).

    Declining propagates as a `Declined` value rather than an empty list, because an empty list
    looks like success.
    """
    def fn(tensions, a, b, *, prec, modes=None, objective=None, expected_count=None):
        # `v1_solve(collapse=False)` reaches the ablation through **kw.
        obj = objective if objective is not None else Objective(
            tensions, a, b, prec=prec, variable="lam", expected_count=expected_count,
            modes=modes)
        res = solve(obj, prec=prec, **kw)
        fn.last = res
        if res.declined:
            return Declined(res.diagnostics.decline_reason,
                            partial=[(x.lam, x.mult) for x in res.levels],
                            diagnostics=res.diagnostics)
        return [(x.lam, x.mult) for x in res.levels]

    fn.last = None
    return fn
