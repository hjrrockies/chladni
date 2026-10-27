"""Stage-wise rank policy, step 1: the plumbing.  (2026-10-27)

`notes/triangle_apex_planner.md` sections 9-12 and the plan they led to: each search stage may read the
pencil under its own rank/regularization POLICY, so a policy has to be a named, hashable thing that
the pencil and the objective key their caches by -- or values computed under two policies at the
same `lam` would be mixed silently, the cache being keyed by `lam` alone until now. Step 1 changes no
behaviour: the default policy is the pencil's own, every default key is what it was, and the
golden replay (`tests/test_search_golden.py`) is the gate. What it adds:

- `Policy`: the pencil's own (`Policy.pencil()`, normalized to None) or a FROZEN rank.
- `MPSPencil.tensions(lam, policy)`, keyed by policy; the default call is unchanged.
- `MPSPencil.spectrum(lam)`: the stack's singular values, a VECTOR the threshold path already
  computes and used to throw away. The stable rank is read from two of these, never from a matrix.
- `MPSPencil.coef(lam, mult, rank=)`: certification at a stage's rank.
- `Objective.tensions(x, policy)`, `samples_for(policy)`, `spectrum(x)`.
- `Level.rank`: the rank a level was refined at, None for the pencil's own.
"""
import numpy as np
import pytest

from chladni.geometry import L_shape
from chladni.mps import MPSPencil, Objective
from chladni.mps.policy import Policy
from chladni.mps.search import Level
from chladni.plan import plan_for

pytestmark = pytest.mark.filterwarnings(
    "ignore:FourierBessel can deliver:chladni._core.PrecisionWarning")

LAM = 37.21


@pytest.fixture(scope="module")
def p():
    dom = L_shape()
    return plan_for(dom, 6, 1e-10).solver(dom)._pencil


def _frozen(p, rank):
    return MPSPencil(p.bdry, p.interior, reg=p.reg, rank=rank, rtol=p.rtol, prec=p.prec)


# --- Policy ------------------------------------------------------------------------------------

def test_the_pencils_own_policy_is_the_default():
    assert Policy.pencil().key is None
    assert Policy.normalize(None) is None and Policy.normalize(Policy.pencil()) is None


def test_a_frozen_policy_is_a_hashable_key():
    a, b = Policy.frozen(40), Policy.frozen(40)
    assert a == b and hash(a) == hash(b) and a.key == ("rank", 40)
    assert Policy.frozen(41) != a


def test_a_frozen_rank_must_be_positive():
    with pytest.raises(ValueError):
        Policy.frozen(0)


# --- MPSPencil ---------------------------------------------------------------------------------

def test_the_default_tension_is_unchanged(p):
    fresh = MPSPencil(p.bdry, p.interior, reg=p.reg, rtol=p.rtol, prec=p.prec)
    assert np.array_equal(p.tensions(LAM), fresh.tensions(LAM))
    assert np.array_equal(p.tensions(LAM, None), p.tensions(LAM))


def test_a_frozen_policy_matches_a_pencil_frozen_at_that_rank(p):
    r = p.rank_at(LAM) - 3
    assert np.array_equal(p.tensions(LAM, Policy.frozen(r)), _frozen(p, r).tensions(LAM))
    assert p.tensions(LAM, Policy.frozen(r)).size == r


def test_two_policies_at_one_lam_are_kept_apart(p):
    r = p.rank_at(LAM) - 5
    own, low = p.tensions(LAM), p.tensions(LAM, Policy.frozen(r))
    assert own.size != low.size
    assert np.array_equal(p.tensions(LAM), own)          # the frozen read did not overwrite it


def test_the_spectrum_is_the_stacks_singular_values(p):
    s = p.spectrum(LAM)
    B, I = p.pencil(LAM)
    ref = np.linalg.svd(np.vstack([B, I]), compute_uv=False)
    assert s.shape == (p.n_cols,) and np.all(np.diff(s) <= 0)
    assert np.allclose(s, ref, rtol=1e-10, atol=1e-14 * ref[0])
    assert p.rank_at(LAM) == int(np.sum(s >= p.rtol * s[0]))


def test_the_spectrum_is_a_vector_kept_by_the_threshold_evaluation(p, monkeypatch):
    from chladni.mps import pencil as _pencil
    q = MPSPencil(p.bdry, p.interior, reg=p.reg, rtol=p.rtol, prec=p.prec)
    q.tensions(LAM)
    calls = {"n": 0}
    real = _pencil.assemble

    def counted(*a, **k):
        calls["n"] += 1
        return real(*a, **k)
    monkeypatch.setattr(_pencil, "assemble", counted)
    q.spectrum(LAM)
    assert calls["n"] == 0                                # no second assembly for the spectrum


def test_coef_at_a_rank_matches_a_frozen_pencil(p):
    r = p.rank_at(LAM) - 2
    assert np.array_equal(p.coef(LAM, 1, rank=r), _frozen(p, r).coef(LAM, 1))
    assert np.array_equal(p.coef(LAM, 1), p.coef(LAM, 1, rank=None))


# --- Objective ---------------------------------------------------------------------------------

def test_the_objective_keys_its_cache_by_policy(p):
    obj = Objective.from_pencil(p, 30.0, 45.0, variable="lam", max_workers=1)
    r = p.rank_at(LAM) - 4
    own = obj.tensions(LAM)
    low = obj.tensions(LAM, Policy.frozen(r))
    assert own.size == p.rank_at(LAM) and low.size == r
    assert np.array_equal(obj.tensions(LAM), own)
    assert obj.n_evals == 2                              # two policies, two factorizations
    xs, _ = obj.samples
    assert list(xs) == [LAM]                             # `samples` is the default policy's
    xr, vr = obj.samples_for(Policy.frozen(r))
    assert list(xr) == [LAM] and vr[0] == low[0]


def test_the_objective_keeps_each_points_spectrum(p):
    obj = Objective.from_pencil(p, 30.0, 45.0, variable="lam", max_workers=1)
    assert obj.spectrum(LAM) is None                     # nothing evaluated there yet
    obj.tensions(LAM)
    assert np.array_equal(obj.spectrum(LAM), p.spectrum(LAM))
    obj.tensions_many([33.0, 41.0])
    assert obj.spectrum(33.0) is not None and obj.spectrum(41.0) is not None


def test_a_bare_callable_objective_is_unchanged():
    obj = Objective(lambda x: np.array([abs(x - 2.0), 1.0]), 0.0, 4.0, variable="lam")
    assert obj.tension(1.0) == 1.0 and obj.spectrum(1.0) is None
    with pytest.raises(ValueError):
        obj.tensions(1.0, Policy.frozen(1))              # nothing to evaluate a policy with


# --- Level -------------------------------------------------------------------------------------

def test_a_level_records_its_rank():
    assert Level(lam=1.0, mult=1).rank is None
    assert Level(lam=1.0, mult=1, rank=40).rank == 40


# --- step 2: one factorization serves the truncation, the spectrum and the coarse penalty -------
#
# Bracketing will read a PENALTY (no rank decision, so continuous in lam) at tau = sqrt(eps) x the
# stack's spectral norm, from the same QR and SVD the threshold path already makes: a small QR of
# [R; tau I] instead of a second QR of the whole tall stack.

def test_a_penalty_policy_is_a_hashable_key():
    a = Policy.penalty()
    assert a == Policy.penalty() and a.key == ("penalty", a.tau_rel)
    assert a.tau_rel == pytest.approx(np.sqrt(np.finfo(float).eps))
    with pytest.raises(ValueError):
        Policy.penalty(0.0)


def test_the_svd_factors_reproduce_regularize_bit_for_bit(p):
    from chladni.mps._regularize import reduce_svd, regularize, svd_factors
    B, I = p.pencil(LAM)
    a = regularize(B, I, reg="svd", rtol=p.rtol)
    b = reduce_svd(svd_factors(B, I), None, p.rtol)
    assert a.rank == b.rank
    assert np.array_equal(a.B, b.B) and np.array_equal(a.I, b.I) and np.array_equal(a.s, b.s)


def test_the_penalty_from_the_factors_matches_tikhonov_qr(p):
    from chladni.mps import _gsvd
    from chladni.mps._regularize import reduce_penalty, regularize, svd_factors
    B, I = p.pencil(LAM)
    f = svd_factors(B, I)
    tau = np.sqrt(np.finfo(float).eps) * f.s[0]
    mine = _gsvd.tensions_of(reduce_penalty(f, tau))
    ref = _gsvd.tensions_of(regularize(B, I, reg="tikhonov_qr",
                                       rtol=tau / np.linalg.norm(np.vstack([B, I]))))
    assert mine.size == ref.size == p.n_cols
    head = slice(0, 20)
    assert np.allclose(mine[head], ref[head], rtol=1e-9, atol=1e-14)


def test_the_penalty_coef_names_the_tension(p):
    from chladni.mps import _gsvd
    from chladni.mps._regularize import reduce_penalty, svd_factors
    B, I = p.pencil(LAM)
    f = svd_factors(B, I)
    tau = np.sqrt(np.finfo(float).eps) * f.s[0]
    reg = reduce_penalty(f, tau)
    c = _gsvd.coef_of(reg, 1)[:, 0] if _gsvd.coef_of(reg, 1).ndim == 2 else _gsvd.coef_of(reg, 1)
    num = np.sqrt(np.linalg.norm(B @ c) ** 2 + (tau * np.linalg.norm(c)) ** 2)
    assert num / np.linalg.norm(I @ c) == pytest.approx(_gsvd.tensions_of(reg)[0], rel=1e-8)


def test_a_penalty_read_keeps_the_spectrum_and_costs_one_assembly(p, monkeypatch):
    from chladni.mps import pencil as _pencil
    q = MPSPencil(p.bdry, p.interior, reg=p.reg, rtol=p.rtol, prec=p.prec)
    calls = {"n": 0}
    real = _pencil.assemble

    def counted(*a, **k):
        calls["n"] += 1
        return real(*a, **k)
    monkeypatch.setattr(_pencil, "assemble", counted)
    t = q.tensions(LAM, Policy.penalty())
    s = q.spectrum(LAM)
    assert calls["n"] == 1
    fin = t[np.isfinite(t)]
    assert t.size == q.n_cols and np.all(np.diff(fin) >= 0) and np.all(np.isinf(t[fin.size:]))
    assert np.allclose(s, p.spectrum(LAM), rtol=1e-12)


def test_the_penalty_floor_sits_near_tau(p):
    # At a true eigenvalue the coarse tension bottoms out near tau (relative), far above the
    # threshold's floor and far below the plateau: what bracketing needs and no more.
    from chladni.reference import L_shape_eigs
    lam1 = float(L_shape_eigs(1)[0])
    t = p.tension(lam1, Policy.penalty())
    assert 1e-10 < t < 1e-6
    assert p.tension(lam1) < t


def test_the_cs_route_matches_the_gsvd_of_the_penalty(p):
    from chladni.mps import _gsvd
    from chladni.mps._regularize import penalty_tensions, reduce_penalty, svd_factors
    for lam in (LAM, 9.6397238440219):                   # a plateau point and the first eigenvalue
        B, I = p.pencil(lam)
        f = svd_factors(B, I)
        tau = np.sqrt(np.finfo(float).eps) * f.s[0]
        ref = _gsvd.tensions_of(reduce_penalty(f, tau))
        mine = penalty_tensions(f, tau)
        assert np.allclose(mine[:20], ref[:20], rtol=1e-9, atol=0)
        assert np.array_equal(p.tensions(lam, Policy.penalty()), mine)
