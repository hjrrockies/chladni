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
