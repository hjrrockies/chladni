"""`prec` is a knob, not decoration.

Two claims, and they are different claims.

**It buys accuracy.** A tighter `prec` must produce a more accurate answer, or the parameter
is a lie. In `geometry` this is real: the adaptive drivers refine until the change falls under
it. In `helmholtz` it is not, and deliberately so -- the cost there is `n_points x n_columns`,
chosen by the caller, and the kernel already delivers the best float64 can do, so there is no
looser-but-cheaper path to take. What `prec` does in that layer is govern what the layer will
CLAIM: it warns when the entries cannot be delivered to it, and caps the basis sizes whose
extra columns would sit below it.

**It costs work.** A looser `prec` must do less. This was entirely untested before this file,
and the effect is large: the polyline of an eccentric ellipse goes from 17 points at 1e-2 to
114389 at 1e-10.

The accuracy half needs a demanding integrand to say anything. On a mild one the base rule is
already exact and every `prec` ties at machine level -- which is what the ellipse in
`test_segments.py` does, making its `tight <= loose` assertion pass vacuously. The curves here
are chosen to make the adaptive machinery work for it.
"""
import warnings

import numpy as np
import pytest

from conftest import PREC
from chladni.geometry import ParametricSegment, Polygon
from chladni.helmholtz import (
    CornerModes, FourierBessel, Fundamental, PlaneWave, PrecisionWarning,
)

#: b/a = 0.02. Eccentric enough that the arclength reparameterization has to adapt; a mild
#: ellipse is resolved to machine level by the coarsest rule and shows nothing.
A, B = 1.0, 0.02


def eccentric(prec):
    return ParametricSegment(lambda t: A * np.cos(t) + 1j * B * np.sin(t),
                             lambda t: -A * np.sin(t) + 1j * B * np.cos(t),
                             0.0, 2 * np.pi, prec=prec)


def exact_perimeter():
    import mpmath as mp

    with mp.workdps(40):
        return float(4 * mp.quad(
            lambda t: mp.sqrt((A * mp.sin(t)) ** 2 + (B * mp.cos(t)) ** 2), [0, mp.pi / 2]))


# --- geometry: prec buys accuracy ---------------------------------------------------------

def test_tighter_prec_is_more_accurate_on_a_demanding_curve():
    """Measured on this curve: 7.0e-12 relative at 1e-6 and below, 2.2e-16 at 1e-8 and
    tighter. The improvement is real but arrives in a STEP, because `refine` doubles."""
    exact = exact_perimeter()
    errs = [abs(eccentric(p).length - exact) / exact for p in (1e-6, 1e-8)]
    assert errs[1] < errs[0] / 100
    assert errs[1] < 1e-14


def test_accuracy_is_monotone_in_prec():
    exact = exact_perimeter()
    errs = [abs(eccentric(p).length - exact) / exact for p in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10)]
    assert all(b <= a * 1.001 for a, b in zip(errs, errs[1:])), errs


def test_a_loose_prec_still_delivers_what_it_promised():
    """The floor of the guarantee: whatever `prec` is asked for, the answer has to be at
    least that good. Not merely monotone -- actually inside the requested tolerance."""
    exact = exact_perimeter()
    for p in (1e-2, 1e-4, 1e-6, 1e-8):
        assert abs(eccentric(p).length - exact) / exact <= p


# --- geometry: prec costs work ------------------------------------------------------------

def test_looser_prec_does_less_work():
    """The half that was never tested. If this ever inverts, an adaptive driver has stopped
    consulting `prec` and the knob has quietly become decoration."""
    counts = [len(eccentric(p).polyline_pts) for p in (1e-2, 1e-4, 1e-6, 1e-8)]
    assert counts == sorted(counts), counts
    assert counts[0] < counts[-1] / 100


def test_the_work_saved_is_large_enough_to_matter():
    """Measured 17 points at 1e-2 against 12997 at 1e-8: a factor of 760. A shape optimizer
    running thousands of iterates at 1e-4 is buying something real, not shaving a constant."""
    loose = len(eccentric(1e-4).polyline_pts)
    default = len(eccentric(PREC).polyline_pts)
    assert default / loose > 50


def test_work_grows_with_prec_on_area_too():
    """Not just the polyline: the adaptive integrators are driven by `prec` as well, so a
    domain built at a loose prec is cheaper end to end."""
    from chladni.geometry import Boundary, Domain

    def disk(prec):
        from chladni.geometry import ArcSegment

        return Domain(Boundary([ArcSegment(0j, 1.0, 0.0, np.pi, prec=prec),
                                ArcSegment(0j, 1.0, np.pi, 2 * np.pi, prec=prec)]))

    assert len(disk(1e-3).boundaries[0].polyline_pts) < len(disk(1e-10).boundaries[0].polyline_pts)


# --- helmholtz: prec governs the claim, not the cost ----------------------------------------

def test_the_kernel_warns_when_it_cannot_meet_prec():
    """`Sampling.accuracy(k)` bounds the relative error of the entries; if that bound exceeds
    the family's own `prec`, saying nothing would be a promise contradicted elsewhere in the
    same object."""
    m = CornerModes.from_wedge(np.pi / 8, 0.0, 0.0)
    pts = np.linspace(0.01, 0.99, 60) * np.exp(0.3j)
    fam = FourierBessel(0j, 1 + 0j, m, m.degrees(20), radius=1.0, prec=1e-14)
    with pytest.warns(PrecisionWarning, match="short of the requested"):
        fam.values(30.0, pts)


def test_the_kernel_is_silent_when_it_can_meet_prec():
    m = CornerModes.from_wedge(np.pi / 8, 0.0, 0.0)
    pts = np.linspace(0.01, 0.99, 60) * np.exp(0.3j)
    fam = FourierBessel(0j, 1 + 0j, m, m.degrees(20), radius=1.0, prec=PREC)
    with warnings.catch_warnings():
        warnings.simplefilter("error", PrecisionWarning)
        fam.values(30.0, pts)


def test_the_warning_is_the_bound_accuracy_reports():
    """The warning and the diagnostic must be the same number, or one of them is lying."""
    m = CornerModes.from_wedge(np.pi / 8, 0.0, 0.0)
    pts = np.linspace(0.01, 0.99, 60) * np.exp(0.3j)
    fam = FourierBessel(0j, 1 + 0j, m, m.degrees(20), radius=1.0, prec=1e-14)
    worst = fam.at(pts).accuracy(30.0).max()
    assert worst > fam.prec
    assert worst < 1e-8


def test_prec_does_not_change_the_numbers_in_this_layer():
    """Stated as a test because it is a design position and not an oversight: the families
    have no adaptive loop and no cheaper-but-looser path, so `prec` governs what they claim
    and never what they compute. If that ever changes, this test is the thing to delete."""
    m = CornerModes.from_wedge(1.4 * np.pi, 0.0, 0.0)
    pts = np.array([0.3 + 0.2j, 0.7 - 0.1j, -0.2 + 0.5j])
    a = FourierBessel(0j, 1 + 0j, m, m.degrees(6), radius=1.0, prec=1e-3)
    b = a.with_prec(1e-13)
    assert np.array_equal(a.values(4.0, pts), b.values(4.0, pts))
    assert np.array_equal(a.log_scale(4.0), b.log_scale(4.0))


# --- helmholtz: prec caps the basis sizes that would be wasted ------------------------------

def test_the_multipole_cap_tightens_as_prec_loosens():
    """`m <= ln(1/prec)/ln(1 + extent/standoff)`: at a looser precision fewer multipole
    degrees are distinguishable, so fewer are worth building."""
    caps = [Fundamental.max_order(1.0, 3.0, p) for p in (1e-13, 1e-8, 1e-3)]
    assert caps == sorted(caps, reverse=True), caps
    assert caps[0] > 2 * caps[2]


def test_the_multipole_cap_tightens_as_the_source_gets_closer():
    """A source hard against the boundary sees a huge dynamic range across Omega, so it
    supports far fewer degrees than a distant one."""
    near = Fundamental.max_order(0.05, 3.0, PREC)
    far = Fundamental.max_order(2.0, 3.0, PREC)
    assert near < far


def test_an_over_high_multipole_order_warns():
    with pytest.warns(PrecisionWarning, match="distinguishable at prec"):
        Fundamental(np.array([5 + 0j]), order=30, radius=1.0, extent=3.0, prec=PREC)


def test_an_order_within_the_cap_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error", PrecisionWarning)
        Fundamental(np.array([5 + 0j]), order=6, radius=1.0, extent=3.0, prec=PREC)


def test_without_an_extent_the_cap_cannot_and_does_not_fire():
    """The cap needs the far radius, and a bare source list does not have one. Warning on a
    guess would be worse than not warning -- the same rule `PlaneWave.n_useful` follows."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", PrecisionWarning)
        Fundamental(np.array([5 + 0j]), order=30, radius=1.0, prec=PREC)


def test_offset_supplies_the_extent_from_the_domain():
    from conftest import L_SHAPE

    poly = Polygon(L_SHAPE)
    fam = Fundamental.offset(poly.bdry_quad(120), 0.12, domain=poly)
    assert fam.extent == pytest.approx(poly.diameter + 0.12, abs=1e-12)


def test_a_looser_prec_admits_a_smaller_plane_wave_basis():
    """The one genuine workload saving in this layer: at 1e-4 a k*R = 20 disk needs about a
    third fewer directions than at 1e-13, and the ones dropped were below the noise floor."""
    assert PlaneWave.n_useful(20.0, 1e-4) < 0.75 * PlaneWave.n_useful(20.0, 1e-13)
