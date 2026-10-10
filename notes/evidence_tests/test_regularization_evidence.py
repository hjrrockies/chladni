"""Evidence moved out of `tests/test_regularization.py` by the 2026-10-09 prune.

`notes/test_inventory_2026-10-09.md`, decision 6: design evidence, not a contract. It records the
honest verdict on the frozen rank against the threshold -- on the enriched L-shape across `lam_1`
the two rules give bit-identical curves -- which is a measurement about one problem, not a
property douse relies on. Moved verbatim with its imports; run on demand from the repository root:

    .venv/bin/python -m pytest -m '' notes/evidence_tests/test_regularization_evidence.py
"""
import importlib.util
import os

import numpy as np

from chladni.helmholtz import FourierBessel
from chladni.mps import MPSPencil, choose_rank
from chladni.mps import tensions as tensions_fn

# `tests/conftest.py`'s constants, loaded under their own name: in this folder `conftest` is
# the evidence folder's own, which only puts `tests/` on the path. The one other change from the
# original is that the test's own `from conftest import L_SHAPE_LAM1` is dropped for this.
_spec = importlib.util.spec_from_file_location(
    "_tests_conftest", os.path.join(os.path.dirname(__file__), "..", "..", "tests", "conftest.py"))
_tc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tc)
L_SHAPE, L_SHAPE_LAM1 = _tc.L_SHAPE, _tc.L_SHAPE_LAM1


def test_the_two_rules_agree_on_a_real_scan_through_an_eigenvalue():
    """The honest verdict, pinned so it is not quietly forgotten.

    Scanning the enriched L-shape across `lam_1` -- the configuration where the tension gets
    smallest, 1.7e-12, and where a cutoff-crossing would do the most damage -- the threshold
    rank never moves and the two rules produce BIT-IDENTICAL curves. There is no accuracy
    argument for the frozen rank on this problem.

    What survives is narrower and is what the design now rests on: the rank is a property of
    the solver rather than of `lam`, so a scan is reproducible and a re-solve at a visited
    parameter returns the same number; and it skips one `svdvals` of the stack per
    evaluation, measured at 11.3 ms against 13.4 ms per point, about 19%.
    """
    from chladni.geometry import Polygon as _P

    dom = _P(L_SHAPE, bc="dir")
    b = sum((FourierBessel.at_corner(c, n=22, domain=dom) for c in dom.corners[1:]),
            FourierBessel.at_corner(dom.corners[0], n=22, domain=dom))
    p = MPSPencil.from_basis(b, dom.bdry_quad(800),
                             dom.interior_pts(300, rng=np.random.default_rng(0)),
                             lam_ref=L_SHAPE_LAM1)

    lams = L_SHAPE_LAM1 * (1 + np.linspace(-4e-3, 4e-3, 41))
    for lam in lams:
        A, I = p.pencil(lam)
        r = choose_rank(A, I, p.rtol)
        assert r == p.rank, f"the threshold rank moved to {r} at lam={lam}"
        assert (tensions_fn(A, I, reg="svd", rank=r)[0]
                == tensions_fn(A, I, reg="svd", rank=p.rank)[0])
