"""The pencil caches VECTORS, never matrices.  (2026-10-13)

The per-`lam` pencil cache held up to 64 entries of equilibrated blocks and regularized pencils,
~10-20 MB each on a 562-column hexagon basis. The douse backend's evaluator holds its last
solver, so ~0.5 GB stayed resident between optimizer steps, and N = 6 runs reached 3-4.4 GB
(`benchmarks/chladni_vs_lappy.py` in douse; one run hit its 4 GB cap).

What a solve actually reuses, measured on the L-shape and a perturbed hexagon at 1e-10:
- the search wants TENSIONS only, and asks for a `lam` it has seen ~78% of the time -- absorbed by
  the objective's tension cache, ~200 kB;
- once the levels are found, certification (and later `eigenfunctions` / `hadamard`) wants each
  level's COEFFICIENTS, `family_coef(lam, coef(lam, m))`. The only matrix-cache hits (~5% of
  lookups) were `family_coef` re-reading the equilibration scales `coef` had just computed.

So the rule: matrices are rebuilt when needed; tensions, scales, ranks and coefficient vectors are
kept. One assembly per `lam` still serves `coef` then `family_coef`.
"""
import numpy as np
import pytest

from chladni.geometry import L_shape
from chladni.mps import pencil as _pencil
from chladni.plan import plan_for

pytestmark = pytest.mark.filterwarnings(
    "ignore:FourierBessel can deliver:chladni._core.PrecisionWarning")


@pytest.fixture
def p():
    dom = L_shape()
    return plan_for(dom, 6, 1e-10).solver(dom)._pencil


@pytest.fixture
def assemblies(monkeypatch):
    """Count matrix assemblies: the cost a cache exists to save."""
    n = {"calls": 0}
    real = _pencil.assemble

    def counting(*a, **k):
        n["calls"] += 1
        return real(*a, **k)
    monkeypatch.setattr(_pencil, "assemble", counting)
    return n


def _is_vector_sized(v, n_cols):
    """An ndarray of at most a handful of n_cols-length columns, or a scalar."""
    if np.isscalar(v):
        return True
    return isinstance(v, np.ndarray) and v.ndim <= 2 and v.size <= 8 * n_cols


def test_a_scan_leaves_only_vectors_behind(p):
    for lam in np.linspace(9.0, 40.0, 20):
        p.tensions(float(lam))
        p.coef(float(lam), 2)
    assert p._cache, "tensions and coefficients should be kept"
    assert all(_is_vector_sized(v, p.n_cols) for v in p._cache.values())


def test_certifications_coef_then_family_coef_assembles_once(p, assemblies):
    lam = 19.7392088
    x = p.coef(lam, 2)
    p.family_coef(lam, x)
    p.log_scale(lam)
    assert assemblies["calls"] == 1


def test_repeats_cost_nothing(p, assemblies):
    lam = 15.2
    t, c = p.tensions(lam), p.coef(lam, 1)
    n = assemblies["calls"]
    np.testing.assert_array_equal(p.tensions(lam), t)
    np.testing.assert_array_equal(p.coef(lam, 1), c)
    assert assemblies["calls"] == n


def test_values_do_not_depend_on_the_cache(p):
    """Fresh from a cold pencil, or after a long scan: the same numbers, bit for bit."""
    dom = L_shape()
    cold = plan_for(dom, 6, 1e-10).solver(dom)._pencil
    lam = 15.2
    first_t, first_c = p.tensions(lam), p.coef(lam, 2)
    for z in np.linspace(20.0, 40.0, 8):
        p.tensions(float(z))
    np.testing.assert_array_equal(cold.tensions(lam), first_t)
    np.testing.assert_array_equal(cold.coef(lam, 2), first_c)
    np.testing.assert_array_equal(cold.family_coef(lam, first_c), p.family_coef(lam, first_c))
