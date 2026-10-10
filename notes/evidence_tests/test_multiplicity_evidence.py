"""Design evidence moved out of `tests/test_multiplicity.py` by the 2026-10-09 prune.

`notes/test_inventory_2026-10-09.md`, decision 6: section 9.4's starved-basis measurements. (C2)
rejects the double count (C1) alone makes near an unresolved neighbour, and the dangerous regime
is entered by having a MEDIOCRE basis -- it is `eps` that sets `eps/c`. Both read the retired
`rule="gap"`; the second is the finding that basis quality and multiplicity correctness are the
same problem. Moved verbatim, with the helpers they need. Run on demand from the repository root:

    .venv/bin/python -m pytest -m '' notes/evidence_tests/test_multiplicity_evidence.py
"""
from __future__ import annotations

import numpy as np
import pytest

import chladni.mps.multiplicity as M
from chladni.geometry import rect
from chladni.helmholtz import PlaneWave
from chladni.mps import MPSPencil
from chladni.reference import rect_eig


# --- (C2) is load-bearing: the starved-basis regime ----------------------------------------------

def _starved(ndir, target_ratio):
    """A pencil whose basis is deliberately too small, with the pair placed at a chosen
    multiple of `eps/c`. Section 9.4's construction: it is `eps` that sets `eps/c`, so the
    dangerous regime is entered by having a mediocre basis."""
    probe = _starved_pencil(ndir, 1e-9)
    epsc = probe[2] / probe[3]
    delta = target_ratio * epsc / (6 * np.pi**2)
    return _starved_pencil(ndir, delta)


def _starved_pencil(ndir, delta):
    L = 1.0 + delta
    dom = rect(L, 1.0)
    la, lb = rect_eig(2, 1, L, 1.0), rect_eig(1, 2, L, 1.0)
    lam1 = min(la, lb)
    basis = PlaneWave.uniform(ndir, radius=dom.diameter)
    # The threshold, explicitly: `eps` here is the BASIS's floor, which a penalty would replace
    # with its own `tau` (the default at prec 1e-8 since 2026-09-24).
    p = MPSPencil.from_basis(basis, dom.bdry_quad(3 * ndir),
                             dom.interior_pts(ndir, rng=np.random.default_rng(0)), reg="svd")
    eps = p.tension(lam1)
    h = 0.02 * lam1
    c = 0.5 * (p.tension(lam1 + h) + p.tension(lam1 - h)) / h
    return p, lam1, eps, c, abs(lb - la)


@pytest.mark.slow
@pytest.mark.parametrize("ratio", [0.42, 0.86])
def test_C2_REJECTS_the_double_count_that_C1_alone_makes(ratio):
    """Section 9.4's measurement, reproduced. Near a simple eigenvalue whose neighbour is
    unresolved, the second branch is the OTHER cone's descending wall -- small, so it passes any
    gap test, but monotone rather than minimal. (C1) reads 2 at a SIMPLE eigenvalue and would do
    so again at its neighbour: the double count. (C2) rejects it."""
    p, lam1, eps, c, d = _starved(10, ratio)
    epsc = eps / c
    assert d / epsc == pytest.approx(ratio, rel=0.5), "the case landed in the ambiguous band"

    reach = M.WIDE_FACTOR * epsc
    r = M.multiplicity(lambda z: p.tensions(float(z)), lam1,
                       wide_bracket=(lam1 - reach, lam1 + reach))
    assert r.c1_read == 2, "the gap test alone says 2"
    assert r.mu == 1, "and (C2) cuts it to 1"
    assert r.c2_rejected


@pytest.mark.slow
def test_the_dangerous_regime_is_entered_by_having_a_MEDIOCRE_BASIS():
    """The consequence larger than the multiplicity rule: it is `eps` that sets `eps/c`, so a
    black-box solver whose planner produces a merely adequate basis is in the double-counting
    regime by default. Basis quality and multiplicity correctness are the same problem."""
    starved = _starved_pencil(10, 3.28e-5)
    good = _starved_pencil(40, 3.28e-5)
    ratio_bad = starved[4] / (starved[2] / starved[3])
    ratio_good = good[4] / (good[2] / good[3])
    assert ratio_bad < 1.0, f"10 directions: d/(eps/c) = {ratio_bad:.2f}, unresolvable"
    assert ratio_good > 20.0, f"40 directions: {ratio_good:.3g}, inside the contract"
    assert good[2] < starved[2] / 1e6, "and the only thing that changed is the floor"
