"""The posterior-penalty cells of `tests/test_rule_seam.py`, kept as runnable evidence.

Moved out of the suite verbatim on 2026-10-09 by the test-suite prune
(`notes/test_inventory_2026-10-09.md`, decision 4). The posterior penalty is PARKED
(`notes/posterior_penalty.md`), and each of these pins an exact count at prec 1e-12 (one at 1e-10),
below the rungs douse uses. They record the search defects the posterior tensions exposed: the
wide-bar dedup, the inner-end retry's standoff, the wide-bar merge, and the hump-start
sub-bracket. Run on demand from the repository root:

    .venv/bin/python -m pytest -m '' notes/evidence_tests/test_rule_seam_evidence.py
"""
from __future__ import annotations

import numpy as np
import pytest

from chladni.mps.search import solve


@pytest.mark.slow
def test_a_wide_refiner_bar_does_not_swallow_a_found_neighbour():
    """50*pi^2, band 0.3, prec 1e-10, searched on POSTERIOR tensions -- the unpenalized
    Rayleigh-Ritz tensions on the 8 lowest vectors of a penalty at tau = 2e-14
    (`notes/posterior_penalty.md`). The side search found all three members; the dedup then
    discarded the third because it lay within the second find's refiner bar (0.24 licences,
    against a 0.22-licence gap) -- the WIDER of the two bars. 2 of 3, silent. Merging only within
    the narrower bar keeps it; a true duplicate is still caught by the find's own reach."""
    import scipy.linalg as la

    import bracket_score as bs
    import rect_suite as rs
    from chladni.mps import MPSPencil
    from chladni.mps.objective import Objective

    prec = 1e-10
    (case,) = rs.split_degeneracy_cases(prec, bands=(0.3,), levels={50: 3})
    p = case.pencil
    q = MPSPencil(p.bdry, p.interior, reg="tikhonov_qr", prec=prec,
                  rtol=2e-14 / np.sqrt(p.n_cols))

    def posterior(lam, k=8):
        pen = q.tensions(float(lam))
        B, I = q.pencil(float(lam))
        X = q.coef(float(lam), k)
        _, R = la.qr(I @ X, mode="economic")
        post = np.sort(la.svdvals(la.solve_triangular(R, (B @ X).T, trans="T").T))
        return np.concatenate([post, pen[k:]])

    base = bs.objective_for(case, prec)
    obj = Objective(posterior, base.lo, base.hi, prec=prec, variable=case.variable,
                    expected_count=base.expected_count)
    res = solve(obj, prec=prec)
    assert not res.declined
    assert sum(L.mult for L in res.levels) == 3


def _posterior_solve(N, m, band, prec, pad, C, k=8):
    """A cluster cell rebuilt at basis padding `pad`, searched on POSTERIOR tensions -- the
    unpenalized Rayleigh-Ritz tensions on the `k` lowest vectors of a penalty at
    `tau = C eps sqrt(n)` (`notes/posterior_tau_sweep.py`). Returns the solve."""
    import scipy.linalg as la

    import bracket_score as bs
    import rect_suite as rs
    from chladni.mps import MPSPencil
    from chladni.mps.objective import Objective

    lam = N * rs.PI2
    a, b = lam * 0.97, lam * 1.03
    L = 1.0 + band * prec / rs.tightest_pair(a, b)[0]
    case = rs.make_case(f"post/{N}/{band}/{pad}", L, 1.0, a, b, modes=rs.modes_in_window(a, b),
                        pad=pad)
    p = case.pencil
    tau = C * np.finfo(float).eps * np.sqrt(p.n_cols)
    q = MPSPencil(p.bdry, p.interior, reg="tikhonov_qr", prec=prec, rtol=tau / np.sqrt(p.n_cols))

    def posterior(x):
        pen = q.tensions(float(x))
        B, I = q.pencil(float(x))
        X = q.coef(float(x), k)
        _, R = la.qr(I @ X, mode="economic")
        post = np.sort(la.svdvals(la.solve_triangular(R, (B @ X).T, trans="T").T))
        return np.concatenate([post, pen[k:]])

    base = bs.objective_for(case, prec)
    obj = Objective(posterior, base.lo, base.hi, prec=prec, variable=case.variable,
                    expected_count=base.expected_count)
    return solve(obj, prec=prec)


@pytest.mark.slow
def test_the_inner_end_retry_stands_off_a_CONVERGED_root():
    """50*pi^2, band 0.3, 1e-12, n = 172, C = 8, posterior. The inner-end retry took its
    exclusion from the root's own reach -- the FLOOR at a converged root, ~1e-12 -- so it re-searched
    onto the root's own wall, refused there and flagged the level, 2 of 3. The standoff rule, as
    for finds (`_SIDE_STANDOFF`)."""
    res = _posterior_solve(50, 3, 0.3, 1e-12, pad=1.6, C=8.0)
    assert sum(L.mult for L in res.levels) == 3 and res.diagnostics.n_uncertain == 0


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::chladni._core.PrecisionWarning")   # C = 3 is the noise jaw
@pytest.mark.parametrize("pad, C", [(1.6, 30.0), (2.4, 3.0)])
def test_a_wide_bar_does_not_MERGE_a_two_member_bracket_into_its_neighbour(pad, C):
    """65*pi^2, band 0.3, 1e-12, posterior. The middle stage-1 bracket held members 1 and 2; its
    root landed between them with a refiner bar of ~0.5 licences, and `_same_point` merged it into
    the neighbouring root 0.33 licences away -- within the WIDER bar -- whose bracket held neither
    member. 2 of 4, silent."""
    res = _posterior_solve(65, 4, 0.3, 1e-12, pad=pad, C=C)
    assert sum(L.mult for L in res.levels) == 4


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::chladni._core.PrecisionWarning")   # C = 30 is past the reach
def test_a_post_find_sub_bracket_that_starts_on_a_HUMP_is_sampled_before_it_is_given_up():
    """50*pi^2, band 0.3, 1e-12, n = 258, C = 30, posterior. After finding member 1 the side search
    resumed on the hump between members 1 and 2, `refine` refused at that inner end with member 2
    0.15 licences inside and unsampled, and the end read exactly member 1's wall -- nothing to flag.
    2 of 3, silent. The refused sub-bracket is now sampled once and refined again."""
    res = _posterior_solve(50, 3, 0.3, 1e-12, pad=2.4, C=30.0)
    assert sum(L.mult for L in res.levels) == 3
