"""Orthonormal eigenfunctions of a cluster: `lowdin` and `MPSEigensolver.eigenfunctions` (M4).

The check is never the Rellich identity against itself. After orthonormalization, the cluster's
Gram matrix is recomputed by an INTERIOR rule exact for the functions -- tensor Gauss on the
square, a polar rule on the disk -- and must be the identity.
"""
import numpy as np
import pytest

from chladni.eigensolver import lam_to_k
from chladni.geometry import disk, rect
from chladni.geometry._adaptive import gauss_legendre
from chladni.helmholtz import PlaneWave
from chladni.integrals import identities as I
from chladni.mps import MPSEigensolver

pytestmark = pytest.mark.slow


def _square_oracle(n=60):
    x, w = gauss_legendre(n)
    return (x[:, None] + 1j * x[None, :]).ravel(), (w[:, None] * w[None, :]).ravel()


def _disk_oracle(nr=60, nt=120):
    r, wr = gauss_legendre(nr)
    t = 2 * np.pi * np.arange(nt) / nt
    pts = (r[:, None] * np.exp(1j * t[None, :])).ravel()
    return pts, (wr[:, None] * r[:, None] * (2 * np.pi / nt) * np.ones(nt)[None, :]).ravel()


def _interior_gram(s, ef, oracle):
    pts, wts = oracle
    U = s.basis.values(lam_to_k(ef.lam), pts) @ ef.coef
    return (U * wts[:, None]).T @ U


def _plane_wave_solver(dom, n, prec=1e-10):
    """A plane-wave solver. On the square, 24 directions and not 16: at 16 (32 columns) the
    search returns NO level in [45, 55] and does not decline -- the starved-basis loss `TODO.md`
    section 4 records, not a defect of this layer."""
    b = PlaneWave.uniform(n, radius=dom.diameter)
    return MPSEigensolver(dom, basis=b, n_bdry=3 * len(b), n_int=len(b), prec=prec,
                          rng=np.random.default_rng(0))


# --- lowdin -------------------------------------------------------------------------------------

def test_lowdin_orthonormalizes_symmetrically():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((5, 3))
    G = A.T @ A
    D = I.lowdin(G)
    assert np.allclose(D.T @ G @ D, np.eye(3), atol=1e-13)
    assert np.allclose(D, D.T), "symmetric: no column is privileged by its position"


def test_lowdin_refuses_a_deficient_cluster():
    """Two columns that are nearly the same function: orthonormalizing them would manufacture a
    second direction from rounding. The guard is lappy's 1e-3 on the eigenvalue ratio."""
    G = np.array([[1.0, 1 - 1e-6], [1 - 1e-6, 1.0]])
    with pytest.raises(ValueError, match="deficient"):
        I.lowdin(G)


# --- clusters through the solver ------------------------------------------------------------------

def test_the_square_double_comes_back_orthonormal():
    """The unit square's exact double at `5 pi^2`, modes (1, 2) and (2, 1): one level of
    multiplicity two, orthonormal against a tensor Gauss rule to `prec`'s order."""
    s = _plane_wave_solver(rect(1.0, 1.0), 24)
    lvl = next(L for L in s.search_interval(45.0, 55.0).levels if L.mult == 2)
    assert lvl.lam == pytest.approx(5 * np.pi**2, rel=1e-10)
    ef = s.eigenfunctions(lvl)
    assert ef.orthonormal and ef.coef.shape[1] == 2
    G = _interior_gram(s, ef, _square_oracle())
    assert np.allclose(G, np.eye(2), atol=1e-10), G


def test_a_disk_double_comes_back_orthonormal():
    """The disk's `m = 1` pair, the sin/cos partners of `J_1(j_11 r)`: a geometric degeneracy,
    exact, on a curved boundary with no corners."""
    s = _plane_wave_solver(disk(1.0), 14)
    from chladni.reference import bessel_zero

    lam = float(bessel_zero(1, 1) ** 2)
    lvl = next(L for L in s.search_interval(lam - 1, lam + 1).levels
               if abs(L.lam - lam) < 1e-6 * lam)
    assert lvl.mult == 2
    ef = s.eigenfunctions(lvl)
    assert ef.orthonormal
    G = _interior_gram(s, ef, _disk_oracle())
    assert np.allclose(G, np.eye(2), atol=1e-9), G


def test_a_simple_level_is_normalized():
    s = _plane_wave_solver(rect(1.0, 1.0), 24)
    lvl = next(L for L in s.search_interval(15.0, 25.0).levels)        # 2 pi^2, simple
    ef = s.eigenfunctions(lvl)
    assert ef.coef.shape[1] == 1
    assert _interior_gram(s, ef, _square_oracle())[0, 0] == pytest.approx(1.0, abs=1e-10)


def test_the_rellich_residual_is_reported_beside_the_coefficients():
    """`gram_residual` is `max |G - I|` from the Rellich identity at a SECOND origin, after
    orthonormalizing at the first: what a caller can check without an oracle."""
    s = _plane_wave_solver(rect(1.0, 1.0), 24)
    lvl = next(L for L in s.search_interval(45.0, 55.0).levels if L.mult == 2)
    ef = s.eigenfunctions(lvl)
    assert ef.gram_residual < 1e-10


def test_a_deficient_cluster_comes_back_normalized_and_FLAGGED(monkeypatch):
    """When `lowdin` refuses, the columns are normalized one by one and the result says it is not
    orthonormal, and why -- never silently returned as if it were."""
    import chladni.integrals.identities as ids

    def refuse(G, guard=ids.LOWDIN_GUARD):
        raise ValueError("the Gram matrix is deficient (forced)")

    s = _plane_wave_solver(rect(1.0, 1.0), 24)
    lvl = next(L for L in s.search_interval(45.0, 55.0).levels if L.mult == 2)
    monkeypatch.setattr(ids, "lowdin", refuse)
    ef = s.eigenfunctions(lvl)
    assert not ef.orthonormal and "deficient" in ef.reason
    G = _interior_gram(s, ef, _square_oracle())
    assert np.allclose(np.diag(G), 1.0, atol=1e-10), "each column still normalized"
