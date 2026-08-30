"""Composing families into one basis.

The composite is where lappy's `MultiBasis` has its sharpest wart, and it is worth naming
because it is the kind that never raises. `_dispatch_cols` documents "sorted ascending" as an
unenforced precondition, filters each sub-basis by range and then `hstack`s the pieces in
SUB-BASIS order -- so an unsorted or interleaved `cols` silently returns columns in a
different order than the one `__call__`'s docstring promises. A caller doing symmetry
projection gets a wrong answer with no signal at all.

So this file passes deliberately shuffled indices, and repeated ones, everywhere.
"""
import numpy as np
import pytest

from conftest import EXACT, L_SHAPE
from chladni.geometry import Polygon
from chladni.helmholtz import Basis, FourierBessel, Fundamental, PlaneWave

K = 4.1


@pytest.fixture
def mixed():
    poly = Polygon(L_SHAPE)
    c = [c for c in poly.corners if c.is_reentrant][0]
    fb = FourierBessel.at_corner(c, n=5)
    fs = Fundamental(np.array([-1 - 1j, 3 + 3j, 3 - 1j]), order=2, radius=1.0)
    pw = PlaneWave.uniform(4)
    return fb + fs + pw


PTS = np.array([0.5 + 0.5j, 1.5 + 0.3j, 0.2 + 1.4j, 0.9 + 0.9j, 1.7 + 0.8j])


# --- composition -------------------------------------------------------------------------

def test_addition_builds_a_basis_and_flattens(mixed):
    assert isinstance(mixed, Basis)
    assert len(mixed.families) == 3
    assert not any(isinstance(f, Basis) for f in mixed.families)


def test_a_basis_plus_a_family_does_not_nest(mixed):
    bigger = mixed + PlaneWave.uniform(2)
    assert len(bigger.families) == 4
    assert len(bigger) == len(mixed) + 4


def test_two_bases_add_to_one_flat_basis(mixed):
    both = mixed + mixed
    assert len(both.families) == 6
    assert len(both) == 2 * len(mixed)


def test_the_column_count_is_the_sum(mixed):
    assert len(mixed) == sum(len(f) for f in mixed.families)
    assert len(mixed) == 5 + 9 + 8


def test_the_columns_are_the_concatenation_in_order(mixed):
    got = mixed.values(K, PTS)
    j = 0
    for f in mixed.families:
        assert np.array_equal(got[:, j:j + len(f)], f.values(K, PTS))
        j += len(f)


def test_column_family_labels_every_column(mixed):
    assert mixed.column_family.tolist() == [0] * 5 + [1] * 9 + [2] * 8


def test_prec_is_the_minimum_over_the_parts():
    """Same rule as a composite Boundary or Domain one layer down: a composite is only as
    precise as its least precise part, and says so rather than claiming the best."""
    a = PlaneWave.uniform(2, prec=1e-6)
    b = PlaneWave.uniform(2, prec=1e-12)
    assert (a + b).prec == 1e-12 or (a + b).prec == 1e-6
    assert (a + b).prec == min(a.prec, b.prec)


# --- the cols contract -----------------------------------------------------------------------

def test_shuffled_cols_come_back_in_the_order_they_were_asked_for(mixed):
    """lappy's MultiBasis returns these in sub-basis order instead, silently."""
    full = mixed.values(K, PTS)
    want = np.array([20, 3, 14, 0, 21, 6, 5, 1])
    assert np.array_equal(mixed.values(K, PTS, cols=want), full[:, want])


def test_reversed_cols_are_honoured(mixed):
    full = mixed.values(K, PTS)
    want = np.arange(len(mixed))[::-1]
    assert np.array_equal(mixed.values(K, PTS, cols=want), full[:, want])


def test_repeated_cols_are_honoured(mixed):
    full = mixed.values(K, PTS)
    want = np.array([7, 7, 2, 7, 2])
    assert np.array_equal(mixed.values(K, PTS, cols=want), full[:, want])


def test_cols_touching_one_family_never_evaluates_the_others(mixed):
    """A corner block asking for its own columns must not pay for a 200-source layer's
    per-k Bessel pass. The saving is the reason `cols` exists at all.

    Note what is NOT skipped: `at(pts)` builds every family's Sampling, because that is the
    k-INDEPENDENT precompute and it happens once. `cols` restricts the per-k transcendental
    work, which is where the cost actually is.
    """
    s = mixed.at(PTS)
    called = []
    for i, part in enumerate(s._parts):
        orig = type(part)._eval
        object.__setattr__(part, "_eval",
                           lambda k, c, g, _o=orig, _p=part, _i=i: (
                               called.append(_i), _o(_p, k, c, g))[1])
    s.values(K, cols=np.array([2, 0]))
    assert called == [0]


def test_a_boolean_mask_selects_columns(mixed):
    full = mixed.values(K, PTS)
    mask = mixed.column_family == 1
    assert np.array_equal(mixed.values(K, PTS, cols=mask), full[:, mask])


def test_gradients_follow_the_same_cols_contract(mixed):
    full = mixed.grad(K, PTS)
    want = np.array([13, 1, 21])
    assert np.array_equal(mixed.grad(K, PTS, cols=want), full[:, want])


# --- the reported quantities concatenate ------------------------------------------------------

def test_log_scale_concatenates_and_follows_cols(mixed):
    full = np.concatenate([f.log_scale(K) for f in mixed.families])
    assert np.array_equal(mixed.log_scale(K), full)
    want = np.array([9, 0, 17])
    assert np.array_equal(mixed.log_scale(K, cols=want), full[want])


def test_singular_exponents_concatenate(mixed):
    s = mixed.singular_exponents()
    assert len(s.origin) == len(mixed) and len(s.exponent) == len(mixed)
    fb = mixed.families[0]
    assert np.array_equal(s.exponent[:5], fb.orders)
    assert np.all(s.origin[:5] == fb.origin)
    assert np.all(np.isnan(s.origin[5:]))


def test_accuracy_concatenates_and_follows_cols(mixed):
    s = mixed.at(PTS)
    full = s.accuracy(K)
    assert len(full) == len(mixed)
    want = np.array([11, 2, 4])
    assert np.array_equal(s.accuracy(K, cols=want), full[want])


# --- the composite is itself a family ------------------------------------------------------------

def test_a_basis_can_be_indexed_for_its_families(mixed):
    assert mixed[0] is mixed.families[0]
    assert len(mixed[1]) == 9


def test_a_basis_is_immutable(mixed):
    with pytest.raises(AttributeError):
        mixed.families = ()


def test_a_basis_sampling_is_reusable_across_k(mixed):
    s = mixed.at(PTS)
    first = s.values(2.0).copy()
    s.values(9.0)
    assert np.array_equal(s.values(2.0), first)


def test_an_empty_basis_is_refused():
    with pytest.raises(ValueError, match="at least one"):
        Basis(())
