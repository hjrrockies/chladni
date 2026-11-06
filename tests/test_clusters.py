"""Clusters: distinct levels whose eigenfunctions are not separately determined.  (2026-11-05)

Levels are grouped by VALUE. Two levels can be certified distinct while their eigenfunctions are
entangled: an eigenvalue error eps moves an eigenvector by about eps / gap (Davis-Kahan; Moler-Payne,
Betcke-Trefethen 2005 Thm 8.1), so between "values resolved" and "eigenfunctions resolved" there is
a band where each level's own near-null vector carries a piece of its neighbour's. Per-level
eigenfunctions there overlap, and a Hadamard block built on them is wrong at the size of the
overlap -- 7.8e-4 of lam on a hard douse shape, where the identities read 2e-10
(`notes/hadamard_accuracy.md`, "Clusters").

The solver links such levels into a `Cluster` (`Spectrum.clusters`, `Level.cluster`): screened by
the Moler-Payne eigenfunction bound eps/alpha > prec, linked when the measured cross-level Gram
exceeds prec and does not move with the Rellich origin (the gap's own bias does). Members are
orthonormalized together.

Ground truth: the 1 x b rectangle, b = 1 + 1e-9, whose (1,2) and (2,1) modes split the square's
double by 1.2e-9 relative. With u = (2/sqrt b) sin(m pi x) sin(n pi y / b), moving the x = 1 side
with normal velocity V = y gives, in the true basis,

    H = [[-pi^2 b, -64 b / 9], [-64 b / 9, -4 pi^2 b]]

((1,2) first). Its eigenvalues do not depend on which orthonormal basis of the pair is used, which
is all a cluster can promise. Measured before the fix: per-level overlap 4.3e-7 at prec 1e-10.
"""
import warnings

import numpy as np
import pytest

from chladni._core import PrecisionWarning
from chladni.eigenproblem import Eigenproblem
from chladni.geometry import rect
from chladni.integrals import cauchy, identities

PREC = 1e-10
B = 1.0 + 1e-9


def _y_on_the_right_side(dom):
    segs = [s for b in dom.boundaries for s in b.segments]

    def V(curve, t):
        s = segs[curve]
        if abs(s.p0.real - 1.0) < 1e-12 and abs(s.p1.real - 1.0) < 1e-12:
            return np.atleast_2d(s.p(np.asarray(t)).imag)
        return np.zeros((1, len(t)))
    return V


TRUE_H = np.array([[-np.pi ** 2 * B, -64 * B / 9], [-64 * B / 9, -4 * np.pi ** 2 * B]])


@pytest.fixture(scope="module")
def split():
    return Eigenproblem(rect(1.0, B)).solve(4, prec=PREC)


def _cross(rec, efs, levels):
    """Max |off-diagonal block| of the joint Gram of two levels' functions on one rule, each
    level's raw columns at its own lam (a purified member's through its mixing matrix)."""
    q = efs[1].rule.quad
    basis = rec.solver._pencil.basis
    cds = [ef.cauchy_data(basis, q) if hasattr(ef, "mix")
           else cauchy.cauchy_data(basis, ef.coef, float(L.lam), q) for ef, L in zip(efs, levels)]
    cd = cauchy.CauchyData(q, float(levels[1].lam), np.hstack([c.u for c in cds]),
                           np.hstack([c.u_n for c in cds]), np.hstack([c.u_t for c in cds]))
    lams = np.concatenate([[float(L.lam)] * int(L.mult) for L in levels])
    G = identities.gram(cd, efs[1].rule.x0, lams=lams)
    m = int(levels[0].mult)
    return float(np.abs(G[:m, m:]).max())


@pytest.mark.slow
def test_an_entangled_split_double_is_one_cluster(split):
    levels = split.levels
    assert [int(L.mult) for L in levels[:4]] == [1, 1, 1, 1]     # resolved by value
    (c,) = split.spectrum.clusters
    assert c.levels == (1, 2) and c.mult == 2
    assert c.spread == pytest.approx((levels[2].lam - levels[1].lam) / levels[2].lam)
    assert c.overlap > PREC
    assert [L.cluster for L in levels[:4]] == [None, 0, 0, None]


@pytest.mark.slow
def test_resolved_pairs_are_not_clusters():
    """1 x 1.001 (gap 1.2e-3) and 1 x 0.7: values and eigenfunctions both resolved."""
    for b in (1.001, 0.7):
        rec = Eigenproblem(rect(1.0, b)).solve(4, prec=PREC)
        assert rec.spectrum.clusters == (), b
        assert all(L.cluster is None for L in rec.levels)


@pytest.mark.slow
def test_a_levels_gap_is_the_moler_payne_alpha(split):
    """`Level.gap`: alpha, the relative distance to the nearest eigenvalue outside its cluster,
    less that eigenvalue's own bar. A cluster's members share the cluster's."""
    lam, bnd = split.lam, split.bound
    a0 = (lam[1] - lam[0] - bnd[1]) / lam[1]
    assert split.levels[0].gap == pytest.approx(a0, rel=1e-9)
    out = min((lam[1] - lam[0] - bnd[0]) / lam[0], (lam[3] - lam[2] - bnd[3]) / lam[3])
    assert split.levels[1].gap == split.levels[2].gap == pytest.approx(out, rel=1e-9)


@pytest.mark.slow
def test_cluster_members_are_orthonormal_to_each_other(split):
    s, (L1, L2) = split.solver, split.levels[1:3]
    raw = _cross(split, [s.eigenfunctions(L1), s.eigenfunctions(L2)], [L1, L2])
    assert raw > 100 * PREC                        # the per-level vectors really overlap
    pure = _cross(split, [split.eigenfunctions(1), split.eigenfunctions(2)], [L1, L2])
    assert pure < PREC, (raw, pure)
    for i in (1, 2):
        ef = split.eigenfunctions(i)
        assert ef.orthonormal and ef.gram_residual < PREC


@pytest.mark.slow
def test_cluster_eigenfunctions_carries_the_basis_and_lams(split):
    ce = split.cluster_eigenfunctions(0)
    assert ce.coef.shape[1] == 2 and ce.mix.shape == (2, 2) and ce.levels == (1, 2)
    np.testing.assert_array_equal(ce.lams, split.lam[1:3])
    # diag(lams) to within overlap * spread * lam
    np.testing.assert_allclose(ce.Lam, np.diag(split.lam[1:3]), rtol=0, atol=1e-12 * split.lam[2])
    assert ce.orthonormal


@pytest.mark.slow
def test_the_clusters_hadamard_block_has_the_true_spectrum(split):
    V = _y_on_the_right_side(split.domain)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PrecisionWarning)
        H, rep = split.hadamard([1, 2], V, report=True)
    want = np.linalg.eigvalsh(TRUE_H)
    np.testing.assert_allclose(np.linalg.eigvalsh(H[0]), want, rtol=10 * PREC, atol=0)
    assert np.trace(H[0]) == pytest.approx(np.trace(TRUE_H), rel=10 * PREC)


@pytest.mark.slow
def test_the_per_level_block_was_wrong(split):
    """The bug this fixes, kept as a measurement: the solver's own per-level eigenfunctions,
    stacked, give a trace off by about the overlap."""
    V = _y_on_the_right_side(split.domain)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PrecisionWarning)
        H = split.solver.hadamard(list(split.levels[1:3]), V)
    assert abs(np.trace(H[0]) - np.trace(TRUE_H)) > 100 * PREC * abs(np.trace(TRUE_H))


@pytest.mark.slow
def test_a_list_that_cuts_through_a_cluster_is_refused(split):
    V = _y_on_the_right_side(split.domain)
    for idx in (1, 2, [0, 1], [2, 3]):
        with pytest.raises(ValueError, match="cluster"):
            split.hadamard(idx, V)


@pytest.mark.slow
def test_attained_carries_the_eigenfunction_bound(split):
    """`HadamardReport.eigenfunction_bound` is Moler-Payne's eps/alpha (Thm 8.1), worst over the
    block, and `attained` is at least `HADAMARD_ESTIMATE_K` times the worse of it and the
    identity residual: the estimate calibrated on 327 levels of the douse shapes
    (`notes/hadamard_accuracy.md`, "Calibration")."""
    from chladni.mps.solver import HADAMARD_ESTIMATE_K
    V = _y_on_the_right_side(split.domain)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PrecisionWarning)
        for idx in (0, [1, 2], 3):
            _, rep = split.hadamard(idx, V, report=True)
            levels = [split.levels[j] for j in np.atleast_1d(idx)]
            eps = max(L.mp_eps for L in levels)
            a = levels[0].gap
            assert rep.eigenfunction_bound == pytest.approx(
                eps / a * np.sqrt(1 + (eps / a) ** 2), rel=1e-12)
            assert rep.attained >= HADAMARD_ESTIMATE_K * max(rep.identity_residual,
                                                             rep.eigenfunction_bound)


# --- the link rule, on a stub solver (no solve) ---------------------------------------------------

class _SearchLevel:
    """What `link_clusters` reads off the search's levels."""
    def __init__(self, lam, eps, bound=1e-12):
        self.lam, self.mult, self.bound, self.mp_eps = lam, 1, bound, eps
        self.margin = self.resolution = float("nan")
        self.count_shortfall = self.count_uncertain = self.grouping_ambiguous = False

    def met(self, prec):
        return True


class _StubSolver:
    def __init__(self, cross):
        self.cross = cross                  # {(lam_a, lam_b): overlaps at the four origins}

    def cross_gram(self, A, B, prec=None):
        return np.asarray(self.cross[(A.lam, B.lam)], float)


def _link(lams, eps, cross, prec=1e-10):
    from chladni.eigenproblem import link_clusters
    found = [_SearchLevel(l, e) for l, e in zip(lams, eps)]
    return link_clusters(_StubSolver(cross), found, len(found), prec)


def test_an_origin_free_overlap_under_the_moler_payne_ceiling_is_linked():
    # gap 1e-9 relative, eps 1e-10: eps/alpha ~ 0.1, ceiling 0.2; overlap 1e-5, origin-free
    lams = [10.0, 20.0, 20.0 + 2e-8, 30.0]
    levels, clusters = _link(lams, [1e-10] * 4, {(10.0, 20.0): [1e-12] * 4,
                                                 (20.0, 20.0 + 2e-8): [1e-5] * 4,
                                                 (20.0 + 2e-8, 30.0): [1e-12] * 4})
    assert [c.levels for c in clusters] == [(1, 2)]
    assert [L.cluster for L in levels] == [None, 0, 0, None]


def test_an_overlap_that_moves_with_the_origin_is_the_gaps_bias():
    lams = [20.0, 20.0 + 2e-8]
    _, clusters = _link(lams, [1e-10] * 2, {(lams[0], lams[1]): [1e-5, 2e-5, 4e-5, 1e-6]})
    assert clusters == ()


def test_an_overlap_above_the_moler_payne_ceiling_is_not_real():
    """A 0.11 gap at prec 1e-10: eps/alpha ~ 1e-9, so a real overlap is at most ~2e-9. A
    cross-Gram of 3e-3 there is the gap's bias, even where the four origins happen to agree
    (seen on a perturbed rectangle in douse's inverse smoke test, 2026-11-05)."""
    lams = [68.59, 77.26]
    _, clusters = _link(lams, [1e-10] * 2, {(lams[0], lams[1]): [3.0e-3, 2.9e-3, 3.1e-3, 3.0e-3]})
    assert clusters == ()


def test_a_pair_inside_the_screen_is_not_measured():
    """eps / alpha <= prec on both sides: resolved by Moler-Payne alone, never measured."""
    lams = [20.0, 30.0]
    _, clusters = _link(lams, [1e-14] * 2, {})          # a KeyError if it measured
    assert clusters == ()
