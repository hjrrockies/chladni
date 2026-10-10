"""Controls for stage 2: refiners that must score as passing, and ones that must fail by name.

`CHLADNI_V1.md` section 5.2's rule, applied to the refiner through the WHOLE pipeline. The
defect this campaign opened with was a refiner that stopped early and reported it honestly --
and the ladder benchmark accepted every such cell, because nothing had ever checked that its
scorer could tell an early stop from a basis limit. These controls are that check.

Each control stands in for `chladni.mps.search.refine`. The search, the multiplicity stage and
`tests/prec_score.py` run unchanged around it, on an exact cone whose floor is 1e-15, so the
basis can serve every request and no excuse is available.

**Oracles must pass**, or the scorer rejects correct answers. **Adversaries must fail by the
verdict named for them**, or the scorer is rejecting them for a reason that will let the next
defect of the same kind through.
"""
from __future__ import annotations

import numpy as np
import pytest

import chladni.mps.search as search
import refine_controls as rc
from chladni.mps.objective import Objective
from prec_score import Sufficiency, score_cell

TRUTH = np.array([2.0, 5.0, 8.0]) * np.pi**2
EPS, C_K, PLATEAU = 1e-15, 0.27, 0.2


def cone(prec):
    """Exact cones in `k` with the floor of a reference-grade pencil, one branch per level."""
    ks = np.sqrt(TRUTH)

    def tensions(lam):
        d = np.abs(ks - np.sqrt(float(lam)))
        return np.sort(np.minimum(np.sqrt(EPS**2 + (C_K * d) ** 2), PLATEAU))

    return Objective(tensions, 15.0, 110.0, prec=prec, variable="lam")


def run(control, prec, monkeypatch):
    monkeypatch.setattr(search, "refine", control)
    res = search.solve(cone(prec), prec=prec)
    return score_cell(res, TRUTH, prec, Sufficiency(accuracy=True, multiplicity=True))


#: Douse's rungs. Until the 2026-10-09 prune the tight end was 1e-14; the cone's 1e-15 floor serves
#: either, and 1e-10 is the precision douse works at.
TIGHT = 1e-10


@pytest.mark.parametrize("prec", [1e-8, TIGHT])
@pytest.mark.parametrize("name", sorted(rc.ORACLES))
def test_every_refine_oracle_passes(name, prec, monkeypatch):
    s = run(rc.ORACLES[name](TRUTH), prec, monkeypatch)
    assert s.verdict == "pass", f"{name} at {prec:.0e}: {s.verdict} -- {s.detail}"


@pytest.mark.parametrize("name", sorted(rc.ADVERSARIES))
def test_every_refine_adversary_fails_by_its_named_verdict(name, monkeypatch):
    """Each adversary is caught by the criterion named for it. Caught means one of two things.
    The scorer rejects it under that verdict; or the error is a COUNT error that the solve ITSELF
    flags (`certified_counts` False), which `prec_score` scores `flagged` with the underlying
    verdict in its detail. The second is the honest outcome by the 2026-10-09 contract -- right,
    declined or flagged -- and it is what `rejects_everything` now gets: three bracketed
    eigenvalues refused, no level returned, and the solve saying its count is uncertain.
    An accuracy error is never excusable by a flag, so the other three must still fail outright.
    """
    factory, expected = rc.ADVERSARIES[name]
    s = run(factory(TRUTH), TIGHT, monkeypatch)
    if s.verdict == "flagged":
        assert s.detail.startswith(f"{expected}, flagged by the solve"), (
            f"{name}: flagged, but not for {expected}: {s.detail}")
        return
    assert not s.acceptable, f"{name} must fail"
    assert s.verdict == expected, f"{name} must be rejected as {expected}, got {s.verdict}"


def test_the_real_refiner_passes_on_the_same_cone(monkeypatch):
    """The instrument's premise: on this objective the shipped refiner is an oracle."""
    s = run(search.refine, TIGHT, monkeypatch)
    assert s.verdict == "pass", f"{s.verdict} -- {s.detail}"
