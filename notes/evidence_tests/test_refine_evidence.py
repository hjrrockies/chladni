"""The `delta_res` ablation from `tests/test_refine.py`, kept as runnable evidence.

Moved out of the suite verbatim on 2026-10-09 by the test-suite prune
(`notes/test_inventory_2026-10-09.md`, decision 6), with its helpers. It records why the
refiner's bar is `eps/c` and not the machine-epsilon formula it replaced: put the old formula back
and it contains the error at none of the disk's levels above rounding. Design evidence, not a
contract. Run on demand from the repository root:

    .venv/bin/python -m pytest -m '' notes/evidence_tests/test_refine_evidence.py
"""
from __future__ import annotations

import numpy as np
import pytest

from chladni.mps.objective import Objective


#: Errors at or below this many ulps are rounding: any bar contains them, so they cannot show a
#: bar to be loose or an old formula to be short. Measured: 0 and 1 ulp at two disk levels.
_ROUNDING = 4.0


def _disk_levels(prec=1e-10):
    """The disk over `[8, 60]` through the REAL pipeline, against exact truth.

    Through the pipeline on purpose. Handed an artificially tight bracket the refiner reaches
    the true floor and lands exactly on the eigenvalue, so every bound contains a zero error
    and nothing is tested. The search hands it a bracket 0.43 wide, the refiner stops where the
    curve goes flat, and THAT is the regime a bound has to describe.
    """
    from chladni.geometry import disk
    from chladni.mps.search import solve
    from chladni.reference import disk_spectrum
    from hard_roster import solver_for

    p = solver_for(disk(1.0), 60.0, label="disk", prec=prec).pencil
    obj = Objective.from_pencil(p, 8.0, 60.0, variable="lam", prec=prec)
    truth = np.unique(disk_spectrum(8.0, 60.0, 1.0))
    for L in solve(obj, prec=prec).levels:
        yield truth[int(np.argmin(np.abs(truth - L.lam)))], L


@pytest.mark.slow
def test_ABLATION_the_machine_epsilon_formula_contains_NOTHING():
    """The correction shown to be load bearing, by putting the old formula back.

    It read `sqrt(eps * eps_mach / curvature)`, assuming the tension carries ABSOLUTE noise of
    order machine epsilon. It does not -- the error is `eps/c` -- so the two differ by
    `sqrt(eps_mach / eps)`, and the shortfall GROWS as the floor rises. `search.md` section 3
    had already recorded the assumption as wrong; the consequence had never been applied here.
    """
    from chladni.mps.refine import DELTA_SAFETY

    missed = 0
    total = 0
    for lam0, L in _disk_levels():
        if abs(L.lam - lam0) <= _ROUNDING * np.spacing(lam0):
            continue                     # located to rounding: every bar contains it
        total += 1
        old = L.delta_res * np.sqrt(np.spacing(1.0) / L.sigma) / DELTA_SAFETY
        missed += abs(L.lam - lam0) > old
    assert total >= 3, f"only {total} levels above rounding: the ablation measures nothing"
    assert missed == total, (
        f"the old formula contained the error at {total - missed} of {total} levels above "
        f"rounding; it was supposed to fail at every one")
