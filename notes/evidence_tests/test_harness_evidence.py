"""Design evidence moved out of `tests/test_harness.py` by the 2026-10-09 prune.

`notes/test_inventory_2026-10-09.md`, decision 6: the surrogate's exactly symmetric wells blind a
strict-`<` local-minimum rule on 85 of 184 grid sizes, while a real disk pencil blinds 0 of 144.
It records the surrogate being HARDER than reality, which is a finding about the test double, not
a contract. Moved verbatim. Run on demand from the repository root:

    .venv/bin/python -m pytest -m '' notes/evidence_tests/test_harness_evidence.py
"""
import numpy as np

from harness import Case, run, uniform_grid_search
from surrogate import Surrogate, Well


# --- a place where the surrogate is HARDER than reality -----------------------------------


def test_a_symmetrically_sampled_well_is_still_found():
    """The surrogate's curves are exact parabolas in the square, so a well sampled
    symmetrically has two flanking values that are EXACTLY equal, and a local-minimum rule
    using a strict `<` on both sides finds nothing at all. Measured: 85 of the 184 grid
    sizes from 16 to 199 make the well invisible.

    This does not transfer -- on a real disk pencil, 0 of 144 grid sizes go blind, because
    a real curve is not symmetric about its well (smallest neighbour gap 6e-4). It is
    recorded here rather than engineered away because it is a clean example of the surrogate
    being harder than the thing it stands in for, which is the direction a test double should
    err in, and because a detector a tie can blind is wrong on its own terms.
    """
    s = Surrogate([Well(5.0, eps=1e-10)], n=16)
    blind = 0
    for n in range(16, 200):
        xs = np.linspace(4.0, 6.0, n)
        v = np.array([s.tension(x) for x in xs])
        if len(np.nonzero((v[1:-1] < v[:-2]) & (v[1:-1] < v[2:]))[0]) == 0:
            blind += 1
    assert blind > 50, "the surrogate no longer exercises the tie case"

    case = Case("symmetric", s.tensions, 4.0, 6.0, s.truth, counter=s)
    assert run(uniform_grid_search(), [case])[0].recall == 1.0
