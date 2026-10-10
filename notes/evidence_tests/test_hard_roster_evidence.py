"""Design evidence moved out of `tests/test_hard_roster.py` by the 2026-10-09 prune.

`notes/test_inventory_2026-10-09.md`, decision 6: `FLOOR_SPREAD_IS_A_BASIS_ARTIFACT` records
that the L-shape's five-order tension-floor spread (`notes/landscape_findings.md` section 20.1)
was the basis, not the domain. It is a finding, not a contract. The inventory flags that it
contradicts TODO section 3's "eps varies five orders" line; which one is stale is for TODO to
settle. Moved verbatim. Run on demand from the repository root:

    .venv/bin/python -m pytest -m '' -s notes/evidence_tests/test_hard_roster_evidence.py
"""
from __future__ import annotations

import numpy as np
import pytest

import hard_roster as hr

pytestmark = [pytest.mark.slow]


# --- the floor spread the rectangle cannot show -------------------------------------------------

def test_the_l_shapes_FLOOR_SPREAD_IS_A_BASIS_ARTIFACT():
    """The five-order floor spread attributed to this domain does not survive a good basis.

    `notes/landscape_findings.md` section 20.1 measured the L-shape's tension floor at 3.1e-13
    at `k = 4.443` and 5.5e-08 at `k = 5.649` -- five orders WITHIN one scan -- and explained it
    structurally: whichever modes VANISH at the reentrant corner are analytic there and resolved
    almost exactly, while their neighbours carry the `r**(2/3)` singularity. That claim is the
    stated justification for the per-eigenvalue accuracy clause.

    With the recipe-grade basis -- 80 pure Fourier-Bessel terms at the reentrant corner -- the
    same levels over `[5, 40]` read:

        lam      9.640   15.197   19.739   29.521   31.913
        k        3.105    3.898    4.443    5.433    5.649
        eps    1.9e-15  5.4e-15  2.8e-15  8.2e-15  5.3e-15      spread 4.2x

    **5.3e-15 at `k = 5.649`, against the 5.5e-08 recorded there: seven orders.** So the spread
    is a property of the BASIS that was used, not of the domain, and it collapses from 1e+05 to
    4.2 once the basis is adequate. That is the same correction section 27.1 made about
    `rect81`, where a 9.2e+04 floor spread turned out to be a frozen rank over a wide window.

    What this does and does not touch. It does NOT retire the per-eigenvalue accuracy clause:
    the floor still varies, just by 4x rather than 1e+05, and a domain whose basis cannot be
    made adequate everywhere will still show the large spread -- which is exactly why
    `finest_prec` is measured per case rather than assumed. What it does retire is "the L-shape
    is the domain where the floor varies five orders", which was a statement about a basis.
    """
    case = hr.l_shape_case(a=5.0, b=40.0)
    p = case.pencil
    lv = np.unique(case.truth)
    floors = np.array([p.tension(float(x)) for x in lv])
    spread = floors.max() / floors.min()
    print(f"\n  L_shape [5, 40], {p.n_cols} pure-FB columns: "
          f"floors {['%.1e' % f for f in floors]}, spread {spread:.1f}x, "
          f"finest_prec {hr.finest_prec(case):.2e}")

    assert floors.max() < 1e-13, (
        "with a recipe-grade basis every level is resolved to better than 1e-13; the recorded "
        f"5.5e-08 at k = 5.649 was the basis, not the domain (got {floors.max():.2e})")
    assert spread < 100.0, f"and the spread is small, not five orders: {spread:.1f}x"
    # The level the notes single out is present and is well resolved.
    i = int(np.argmin(np.abs(lv - 31.9126)))
    assert abs(lv[i] - 31.9126) < 1e-3
    assert floors[i] < 1e-13, f"k = 5.649 reads {floors[i]:.2e}, not 5.5e-08"
