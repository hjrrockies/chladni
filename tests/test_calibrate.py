"""Findability: can a scan of a given step SEE every dip, and what to do when it cannot.

`chladni.mps.calibrate`. The check answers a question the tension at a located eigenvalue
structurally cannot -- the failure it guards against returned a non-consecutive spectrum with an
excellent tension at every value it did return.

One thing here is a measurement rather than an assertion about code, pinned so that a later
change has to confront it: loosening the rank truncation has **no lever** on `chladni` pencils,
even though it was the study's primary remedy. The other, the constant `KAPPA` re-measured in
`chladni` rather than inherited, moved to `notes/evidence_tests/test_calibrate_evidence.py` in the
2026-10-09 prune, with the wall-slope scaling test it rests on.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from chladni._core import PrecisionWarning
from chladni.geometry import iso_tri, rect
from chladni.helmholtz import PlaneWave
from chladni.mps import MPSEigensolver, MPSPencil
from chladni.mps import calibrate as C



def pencil(d, lam_hi, pad=1.3, seed=0):
    """A THRESHOLD pencil, explicitly. `calibrate_rtol` moves a truncation cut, and every number
    here was measured on one; since 2026-09-24 the default at `prec = 1e-8` is the penalty, where
    loosening `rtol` raises `tau` past its reach and warns (`TODO.md` section 4, item 2)."""
    nw = int(pad * PlaneWave.n_useful(np.sqrt(lam_hi) * d.diameter))
    return MPSPencil.from_basis(
        PlaneWave.uniform(nw, radius=d.diameter), d.bdry_quad(max(8 * nw, 900)),
        d.interior_pts(max(3 * nw, 450), rng=np.random.default_rng(seed)), reg="svd")


# --- the probe grid is deterministic ---------------------------------------------------------

def test_the_probe_points_take_no_seed_and_do_not_repeat():
    """`sigma_base` moved 14x on an RNG seed change in the study, with the DEFAULT seed being
    the lucky one on the shape that happened to solve. A quantity that swings 14x on a seed is
    not a measurement, so the sampling here is deterministic by construction."""
    a = C._probe_lams(10.0, 88.0, 40)
    b = C._probe_lams(10.0, 88.0, 40)
    assert np.array_equal(a, b)
    assert np.unique(np.round(a, 9)).size == 40, "no two probes coincide"
    assert a.min() >= 10.0 and a.max() <= 88.0


def test_the_probe_grid_is_not_uniform():
    """An evenly spaced grid can land on a resonance systematically. The offset is irrational
    for that reason, and a random one would reintroduce the seed sensitivity above."""
    d = np.diff(np.sort(C._probe_lams(10.0, 88.0, 40)))
    assert d.std() / d.mean() > 0.1, "spacings vary; this is not a uniform grid"


@pytest.mark.slow
def test_sigma_base_is_a_median_and_survives_probes_landing_in_dips():
    """Which is why it is a median. A probe in a dip pulls one sample orders down; a mean would
    follow it, and the plateau is what the criterion needs."""
    d = rect(1.0, 1.0)
    p = pencil(d, 120.0)
    sb = C.sigma_base(p, 20.0, 120.0)
    assert 0.05 < sb < 1.0, f"a plateau, not a dip: {sb:.3e}"
    # measured across seven domains, the plateau spans 0.12 to 0.36
    assert C.sigma_base(p, 20.0, 120.0, n=24) == pytest.approx(sb, rel=0.5)


# --- the criterion's target and step ----------------------------------------------------------
#
# The two measurements behind `KAPPA` -- the wall slope scales with LENGTH, not area, and kappa
# re-measured in `chladni` (2.591) agrees with the study -- moved to
# `notes/evidence_tests/test_calibrate_evidence.py` in the 2026-10-09 prune. They are design
# evidence; `KAPPA` itself is the contract, and the tests below use it.

def test_the_target_and_the_required_step_are_inverses():
    d = rect(1.0, 1.0)
    dk = 0.25
    target = C.findability_target(d, dk)
    assert C.required_dk(d, target) == pytest.approx(dk)


def test_the_target_scales_linearly_in_the_scan_step_and_in_A_over_P():
    d = rect(1.0, 1.0)
    assert C.findability_target(d, 0.2) == pytest.approx(2 * C.findability_target(d, 0.1))
    big = rect(2.0, 2.0)
    assert (C.findability_target(big, 0.1) / C.findability_target(d, 0.1)
            == pytest.approx((big.area / big.perimeter) / (d.area / d.perimeter)))


# --- the check discriminates, and on the right domains ----------------------------------------

@pytest.mark.slow
def test_a_sharp_elongated_triangle_needs_far_more_scan_points_than_a_square():
    """The shape class that broke the study's optimizer: a sharp, elongated triangle. The
    criterion has to separate it from a benign domain using only geometry and the plateau,
    BEFORE any mode is located."""
    sq, sharp = rect(1.0, 1.0), iso_tri(4.0)
    n = []
    for d in (sq, sharp):
        p = pencil(d, 120.0)
        sb = C.sigma_base(p, 20.0, 120.0, n=24)
        span = np.sqrt(120.0) - np.sqrt(20.0)
        n.append(span / C.required_dk(d, sb))
    assert n[1] > 4 * n[0], f"square needs {n[0]:.0f} points, sharp triangle {n[1]:.0f}"


@pytest.mark.slow
def test_an_unfindable_configuration_warns_and_names_the_remedy():
    """"The solve may be incomplete" is not actionable; "use dk <= 9.4e-02, about 62 scan
    points" is, and it is exactly what the failing triangle was never told."""
    d = iso_tri(4.0)
    s = MPSEigensolver(d, basis=PlaneWave.uniform(60, radius=d.diameter),
                       n_bdry=600, n_int=250)
    with pytest.warns(PrecisionWarning, match="narrower than the scan can see") as rec:
        cal = s.findability(20.0, 120.0, dk=0.5, n=16)
    assert not cal.achieved
    assert cal.shortfall > 1.0
    assert cal.required_dk < cal.dk, "the remedy is a finer step"
    # The remedy is NAMED in the warning itself: the step, and the scan-point count it implies.
    # (This line once read `... in str(warnings.formatwarning) or True`, which asserted nothing.)
    msg = next(str(w.message) for w in rec if "scan can see" in str(w.message))
    assert f"dk <= {cal.required_dk:.3e}" in msg, msg
    assert "scan points" in msg, msg
    assert "UNFINDABLE" in str(cal)


@pytest.mark.slow
def test_a_findable_configuration_is_silent_and_cached():
    d = rect(1.0, 1.0)
    s = MPSEigensolver(d, basis=PlaneWave.uniform(48, radius=d.diameter),
                       n_bdry=600, n_int=250)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cal = s.findability(20.0, 120.0, dk=0.05, n=16)
        again = s.findability(20.0, 120.0, dk=0.05, n=16)
    # Only findability warnings are at issue; the basis emits its own oversizing warning here
    # and that is a different complaint about a different thing.
    assert not [w for w in caught if "scan can see" in str(w.message)]
    assert cal.achieved and not cal.calibrated
    assert again is cal, "cached per (window, step)"


# --- the rank-truncation lever, and the finding that it is inert ------------------------------

@pytest.mark.slow
def test_rtol_does_NOT_size_the_basis():
    """The hazard the note warned about: in `lappy`, `basis_plan._indep_digits` set the
    dynamic range from `rtol`, so changing the threshold changed the PLAN -- and calibrating
    the threshold moved the basis underneath the calibration. `chladni` has no such coupling
    and this pins it, because the simplification is worth keeping."""
    d = rect(1.0, 1.0)
    p = pencil(d, 120.0)
    q = p.with_rtol(1e-8)
    assert q.n_cols == p.n_cols and q.n_bdry == p.n_bdry and q.n_int == p.n_int
    assert q.basis is p.basis, "the same basis object, not merely an equal one"
    assert q.rtol == 1e-8 and p.rtol != 1e-8, "and the pencil is immutable"


@pytest.mark.slow
def test_loosening_the_truncation_has_NO_LEVER_on_sigma_base_here():
    """**A divergence from the study, and the reason the default remedy is the scan step.**

    The study's fix (1) was to raise `sigma_base` by loosening the rank truncation, having
    measured it move from 2.5e-3 to 1.2e-1 -- a factor of 48 -- across truncation settings. On
    a `chladni` rectangle pencil it does not move at all: 0.3329 to 0.3346, a range of 0.5%,
    while the retained rank goes 47 -> 19.

    Two candidate reasons, neither yet settled: `chladni` thresholds RELATIVE to the largest
    singular value, so raising the cutoff removes columns proportionally and the plateau -- a
    ratio of norms -- stays put, where an absolute cutoff bites at a fixed level; and
    `chladni`'s plateau is already healthy across domains (0.12 to 0.36 over seven) where the
    study's varied two orders. Either way the lever is absent, so `calibrate_rtol` is kept and
    documented rather than made the default remedy.
    """
    d = rect(1.0, 1.0)
    p = pencil(d, 120.0)
    out = [(q.rank_at(50.0), C.sigma_base(q, 20.0, 120.0, n=24))
           for q in (p.with_rtol(r) for r in (1e-15, 1e-11, 1e-7, 1e-3))]
    ranks = [r for r, _ in out]
    bases = [s for _, s in out]
    assert ranks[0] > 2 * ranks[-1], f"the rank moves a lot: {ranks}"
    assert max(bases) / min(bases) < 1.05, f"and sigma_base does not: {bases}"


@pytest.mark.slow
def test_the_calibration_early_exits_when_the_default_already_passes():
    """The common case, and it must be cheap: one measurement and nothing more."""
    d = rect(1.0, 1.0)
    p = pencil(d, 120.0)
    cal = C.calibrate_rtol(p, d, 20.0, 120.0, dk=0.05, n=16)
    assert cal.achieved and not cal.calibrated
    assert cal.evaluations == 16, "the check only; no bisection was run"
    assert cal.rtol == p.rtol, "and the threshold is untouched"


def test_a_calibration_reports_its_shortfall_and_its_remedy():
    cal = C.RtolCalibration(rtol=1e-12, sigma_base=2.5e-3, target=1.6e-2, dk=0.124,
                            achieved=False, calibrated=False, required_dk=0.0191)
    assert cal.shortfall == pytest.approx(6.4, rel=0.05), "the study's failing triangle"
    assert "UNFINDABLE" in str(cal) and "9.43e-02" not in str(cal)
