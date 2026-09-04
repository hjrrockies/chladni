"""TIER 1's fixed case set: the bands land, and the tier's own limit is pinned.

The limit matters as much as the cases. The surrogate does not model `sigma_3` and above, so a
test here that reads the higher spectrum would be measuring the stand-in rather than the method
-- and the multiplicity phase reads exactly that branch.
"""
from __future__ import annotations

import numpy as np
import pytest

import bracket_score as bs
import surrogate_cases as sc


def test_the_fixed_set_covers_the_regimes_it_claims_to():
    labels = [c.label for c in sc.all_cases()]
    assert any("isolated" in x for x in labels)
    assert sum("pair" in x for x in labels) >= 4, "the crossover ladder, straddling"
    assert any("cluster3" in x for x in labels) and any("cluster4" in x for x in labels)
    assert sum("degenerate" in x for x in labels) >= 2, "degeneracy as a CLASS"
    assert any("rough" in x for x in labels)


@pytest.mark.parametrize("band", (0.5, 0.8, 1.3, 3.0, 30.0))
@pytest.mark.parametrize("prec", bs.PREC_LADDER)
def test_a_surrogate_pair_lands_on_its_band(prec, band):
    """`prec` is relative to `lam`, so a pair at relative separation `s` in `lam` sits at
    `s/2` in `k`. Getting that factor wrong dials every case to twice its band."""
    cases = sc.pair_ladder(bands=(band,), prec=prec)
    if not cases:
        pytest.skip("not representable at this band and precision")
    sep = bs.classify_separation(bs.truth_in_lam(cases[0].truth, "k"), prec)
    assert sep.band == pytest.approx(band), f"achieved {sep.ratio:.4g}"


@pytest.mark.parametrize("mult", (2, 3))
def test_an_exact_degeneracy_is_coincident_to_the_bit(mult):
    """No interval can separate a level from itself, so this must be bit-identical rather
    than merely close -- otherwise the case is a tight cluster wearing a degeneracy's name."""
    case = sc.degenerate(mult)
    t = np.sort(case.truth)
    lv, counts = np.unique(t, return_counts=True)
    assert counts.max() == mult
    assert bs.classify_separation(bs.truth_in_lam(t, "k"), 1e-14).degenerate


def test_a_cluster_has_UNEQUAL_gaps():
    """Evenly spaced would be the easier case wearing the harder name: a method can suit a
    pair and suit a tight group with unequal gaps much less."""
    (case,) = sc.cluster(3, band=30.0, prec=1e-8)
    inner = np.sort(case.truth)[1:4]
    gaps = np.diff(inner)
    assert gaps.max() / gaps.min() == pytest.approx(2.2, rel=0.05)


def test_the_isolated_case_is_genuinely_isolated():
    sep = bs.classify_separation(bs.truth_in_lam(sc.isolated().truth, "k"), 1e-8)
    assert sep.ratio > 1e5, "nothing here is near any crossover"


def test_the_cone_shape_matches_the_measured_local_model():
    """`sigma_1 ~ sqrt(eps^2 + c^2*(k-k*)^2)`: the floor at the well, the slope away from it."""
    case = sc.isolated()
    k0 = float(np.sort(case.truth)[2])
    assert case.tensions(k0)[0] == pytest.approx(sc.EPS, rel=0.5)
    far = case.tensions(k0 + 1e-3)[0]
    assert far == pytest.approx(sc.C_K * 1e-3, rel=0.05), "the wall has the slope it was given"


def test_roughness_is_ten_orders_below_a_true_well():
    """Prominence separates a well from roughness by ten orders: a true well reaches
    `plateau/eps` (worst case 72 measured), roughness minima `1 + 2.4e-10` or less."""
    case = sc.rough(level=1e-10)
    k0 = float(np.sort(case.truth)[1])
    depth = case.surrogate.plateau / case.tensions(k0)[0]
    assert depth > 1e10, "the well is unmistakable against 1 + 2.4e-10 roughness"


def test_the_tier_states_that_it_cannot_judge_multiplicity():
    """Pinning the known limit so nobody quietly relies on `sigma_3` here. Every non-well
    curve sits at the cap, so the higher branches carry no structure at all."""
    case = sc.degenerate(2)
    v = case.tensions(float(np.sort(case.truth)[1]))
    assert v[0] == pytest.approx(v[1], rel=1e-6), "the degenerate pair IS modelled"
    assert v[2] == pytest.approx(case.surrogate.plateau, rel=1e-6), "sigma_3 is just the cap"
