"""Reference eigenvalues: the closed forms are right, and every value states its accuracy.

Two things are defended here.

The closed forms are checked against something computed independently -- mpmath at extended
precision for the Bessel zeros, direct enumeration for the index families -- not against
another chladni function. A reference that is only self-consistent is not a reference.

And every reference function carries its accuracy as DATA. That is the whole point of the
module: lappy's tables run from 14 digits to under 3, its docstrings say so carefully, and a
docstring cannot be read by the test that has to pick a tolerance. Here `fn.digits` can be,
and `fn.notes` names the individual entries that are worse than the table.
"""
import numpy as np
import pytest

from chladni import reference as ref
from conftest import EXACT, MP_DPS


# --- closed forms against independent computation ----------------------------------------


def test_rect_matches_the_formula_directly():
    L, H = 2.0, 1.0
    want = np.sort([np.pi**2 * (m**2 / L**2 + n**2 / H**2)
                    for m in range(1, 12) for n in range(1, 12)])[:10]
    assert np.allclose(ref.rect_eigs(10, L, H), want, rtol=0, atol=EXACT)


def test_rect_neumann_starts_at_zero():
    lam = ref.rect_eigs(5, 1.0, 1.0, "neu")
    assert lam[0] == 0.0
    assert lam[1] == pytest.approx(np.pi**2, rel=1e-14)


def test_disk_zeros_come_from_mpmath_not_from_us():
    """The disk's spectrum IS the Bessel zeros, so check them against extended precision."""
    import mpmath as mp

    with mp.workdps(MP_DPS):
        want = sorted(float(mp.besseljzero(m, n)) ** 2
                      for m in range(0, 4) for n in range(1, 4))
    got = sorted(ref.disk_eig(m, n) for m in range(0, 4) for n in range(1, 4))
    assert np.allclose(got, want, rtol=1e-13, atol=0)


def test_sector_uses_extended_precision_at_fractional_order():
    """The lesson `bessel_zero` records: a brentq scan loses 2-3 digits at fractional order,
    which put lappy's sector references out by up to 1.4e-13 relative and once made a solver
    look like it had violated its own error bound."""
    import mpmath as mp

    omega = 2 * np.pi / 3          # nu = 3/2, genuinely fractional
    nu = np.pi / omega
    with mp.workdps(MP_DPS):
        want = float(mp.besseljzero(mp.mpf(nu), 1)) ** 2
    assert ref.sector_eig(1, 1, 1.0, omega) == pytest.approx(want, rel=1e-14)


def test_a_half_disk_sector_is_the_disk_without_its_radial_modes():
    """`omega = pi` gives `nu = 1`, so the sector's spectrum is the disk's `m >= 1` modes --
    an independent cross-check between two closed forms that share no code path."""
    sec = ref.sector_eigs(6, 1.0, np.pi)
    disk = [ref.disk_eig(m, n) for m in range(1, 5) for n in range(1, 5)]
    assert np.allclose(sec, np.sort(disk)[:6], rtol=1e-12, atol=0)


def test_eq_tri_first_eigenvalue_is_the_lame_value():
    assert ref.eq_tri_eigs(1, 1.0)[0] == pytest.approx(16 * np.pi**2 / 3, rel=1e-14)


def test_iso_right_tri_rejects_the_zero_mode():
    """`m == n` is the zero function on the folded square, not an eigenfunction."""
    with pytest.raises(ValueError, match="m > n"):
        ref.iso_right_tri_eig(2, 2)


def test_scaling_a_domain_scales_its_spectrum():
    """`lam -> lam/s**2` under `Domain.scaled(s)`, for every closed form that takes a size."""
    s = 3.0
    assert np.allclose(ref.rect_eigs(6, 2 * s, 1 * s), ref.rect_eigs(6, 2.0, 1.0) / s**2)
    assert np.allclose(ref.disk_eigs(6, s), ref.disk_eigs(6, 1.0) / s**2)
    assert np.allclose(ref.eq_tri_eigs(6, s), ref.eq_tri_eigs(6, 1.0) / s**2)


# --- complete spectra, which is what recall testing needs --------------------------------


def test_a_spectrum_is_complete_where_eigs_is_merely_first_k():
    """"The first ten" and "all of them below 40" are different questions, and only the
    second can say whether a search missed one. Both return the same shape -- a flat
    ascending array with multiplicity repeated -- so a truncated list and an interval query
    over the same range are literally equal."""
    lam = ref.rect_eigs(20, 2.0, 1.0)
    hi = float(lam[-1])
    spec = ref.rect_spectrum(0.0, hi, 2.0, 1.0)
    assert spec.size == 20
    assert np.allclose(spec, lam, rtol=0, atol=EXACT)


def test_a_spectrum_reports_multiplicity():
    """The square's (1,2)/(2,1) pair is exactly degenerate; the disk doubles every m >= 1."""
    sq = ref.rect_spectrum(0.0, 60.0, 1.0, 1.0)   # 5*pi**2 = 49.3 is the pair
    assert np.sum(np.abs(sq - 5 * np.pi**2) < 1e-9) == 2, f"the pair should repeat: {sq}"
    assert ref.collapse(sq) == [(pytest.approx(2 * np.pi**2), 1),
                                (pytest.approx(5 * np.pi**2), 2)]

    d = ref.disk_spectrum(0.0, 30.0, 1.0)
    assert [m for _, m in ref.collapse(d)] == [1, 2, 2]


def test_eq_tri_multiplicity_is_arithmetic():
    """`m**2 + m*n + n**2` collides for distinct index pairs, on top of the (m,n)/(n,m)
    pairing -- a different mechanism from the disk's geometric degeneracy."""
    assert any(m >= 2 for _, m in ref.collapse(ref.eq_tri_spectrum(0.0, 250.0, 1.0)))


def test_collapse_is_relative_not_absolute():
    """lappy compares against an absolute 1e-12, which stops working for large `lam`."""
    big = 1e8
    assert ref.collapse([big, big * (1 + 1e-13)], rtol=1e-9) == [(big, 2)]
    assert len(ref.collapse([big, big * (1 + 1e-6)], rtol=1e-9)) == 2


# --- accuracy is data --------------------------------------------------------------------


def test_every_reference_function_declares_its_accuracy():
    assert len(ref.catalog()) > 20
    for name, digits, n_caveats, source in ref.catalog():
        assert 0 < digits <= 16, name
        assert source, name


def test_closed_forms_claim_float64_and_no_more():
    """A closed form is exact as mathematics and still evaluates in floating point, so it
    claims float64's digits rather than infinity."""
    assert ref.rect_eigs.digits == pytest.approx(15.6)
    assert ref.disk_eigfun.digits == pytest.approx(15.6)


def test_a_table_reports_the_WORST_case_not_the_best():
    """`ellipse_eigs` is 13.3 digits for a=2 and about 6.7 for a=3. It must report 6.7:
    a caller deriving a tolerance from `digits` has to be safe for every parameter set."""
    assert ref.ellipse_eigs.digits == pytest.approx(6.7)
    assert ref.chevron_eigs.digits == pytest.approx(3.0)
    assert ref.reg_ngon_eigs.digits == pytest.approx(2.8)


def test_the_disputed_ellipse_value_is_corrected_and_says_so():
    """lappy ships 3.566726599853406 and documents it as disputed. Three of its own
    independent solves agree on 3.56672660292886x to 7.5e-16, so chladni takes that -- and
    records the substitution, because silently disagreeing with a published reference is its
    own kind of wrong."""
    assert ref.ellipse_eigs(1, a=2.0)[0] == pytest.approx(3.566726602928861, rel=1e-15)
    note = ref.ellipse_eigs.notes[(2.0, 1.0)][0]
    assert "CORRECTED" in note and "8.62e-10" in note


def test_accuracy_is_reported_per_parameter_set_as_well_as_worst_case():
    """A single worst-case scalar punishes the good tables for the company they keep:
    `reg_ngon` is 10.2 digits at N=5,6,7 and 2.8 at N=8. `digits` stays the safe worst case
    for a caller who ignores the refinement; `digits_for` gives the sharp value."""
    assert ref.reg_ngon_eigs.digits == pytest.approx(2.8)
    assert ref.reg_ngon_eigs.digits_for(5) == pytest.approx(10.2)
    assert ref.reg_ngon_eigs.digits_for(8) == pytest.approx(2.8)
    assert ref.ellipse_eigs.digits_for((2.0, 1.0)) == pytest.approx(13.3)
    assert ref.chevron_eigs.digits_for((1.0, 2.0)) == pytest.approx(12.0)
    # an unknown key falls back to the worst case, which is the safe direction
    assert ref.reg_ngon_eigs.digits_for(99) == ref.reg_ngon_eigs.digits


def test_two_tabulated_entries_turned_out_to_be_closed_forms():
    """Cheap exactness, taken where it was available.

    `H_shape` index 6 was flagged by lappy as 'suspiciously close to 2*pi**2, likely a mode
    that vanishes on the connecting web'. It is: the tabulated value agrees to 2.5e-15. And
    `iso_tri(h=1)` -- base 2, height 1, vertices (-1,0), (0,1), (1,0) -- is the right
    isosceles triangle with legs sqrt(2), so its whole spectrum is closed form.
    """
    assert ref.H_shape_eigs(7)[6] == 2 * np.pi**2

    assert np.allclose(ref.iso_tri_eigs(10, 1.0),
                       ref.iso_right_tri_eigs(10, np.sqrt(2.0)), rtol=0, atol=EXACT)
    assert ref.iso_tri_eigs.digits_for(1.0) == pytest.approx(15.6)
    assert "CLOSED FORM" in ref.iso_tri_eigs.notes[1.0][None]


def test_bessel_zeros_are_full_double_precision():
    """The reference spectra for the disk and the sector ARE the Bessel zeros, so they have
    to be more accurate than anything checked against them.

    Integer order goes to scipy, measured at or below one ulp over 32 cases spanning order
    0 to 400. Fractional order goes to mpmath, where scipy offers nothing and a brentq scan
    loses two to three digits.
    """
    import mpmath as mp

    with mp.workdps(MP_DPS):
        for m, n in ((0, 1), (1, 1), (5, 3), (40, 10)):
            assert ref.bessel_zero(m, n) == pytest.approx(
                float(mp.besseljzero(m, n)), rel=2.3e-16)
        # Derivative zeros from m >= 1 only: at m = 0 the two libraries index differently.
        # `J_0' = -J_1` and `J_1(0) = 0`, so mpmath counts x = 0 as the first zero while
        # scipy skips it. chladni sends every integer order to scipy, so the convention is
        # scipy's throughout and the Neumann disk never sees the trivial zero -- its constant
        # mode is prepended explicitly instead.
        for m, n in ((1, 1), (5, 3), (40, 10)):
            assert ref.bessel_zero(m, n, derivative=True) == pytest.approx(
                float(mp.besseljzero(m, n, derivative=1)), rel=2.3e-16)
        assert ref.bessel_zero(0, 1, derivative=True) == pytest.approx(
            float(mp.besseljzero(0, 2, derivative=1)), rel=2.3e-16)
        for nu, n in ((1.5, 1), (2.0 / 3.0, 2), (4.0 / 3.0, 3)):
            assert ref.bessel_zero(nu, n) == pytest.approx(
                float(mp.besseljzero(mp.mpf(nu), n)), rel=1e-15)


def test_per_entry_caveats_are_reachable_by_index():
    """`reg_ngon` N=8 has one bad entry among nine good ones. A table-wide number cannot say
    that; the note is keyed by the index it applies to."""
    assert 3 in ref.reg_ngon_eigs.notes[8]
    assert "2.8 digits" in ref.reg_ngon_eigs.notes[8][3]
    assert 6 in ref.H_shape_eigs.notes[()]


def test_tables_refuse_parameters_they_do_not_have():
    for fn, kwargs in ((ref.ellipse_eigs, dict(a=5.0)), (ref.reg_ngon_eigs, dict(N=9)),
                       (ref.cut_square_eigs, dict(r=0.3)), (ref.iso_tri_eigs, dict(h=3.0)),
                       (ref.chevron_eigs, dict(h1=1.0, h2=9.0))):
        with pytest.raises(ValueError, match="no table"):
            fn(1, **kwargs)


def test_tables_refuse_more_than_they_hold():
    with pytest.raises(ValueError, match="only the first"):
        ref.L_shape_eigs(26)
    with pytest.raises(ValueError, match="only the first"):
        ref.mushroom_eigs(11)


def test_the_gww_pair_shares_one_table():
    """The isospectral pair's whole point: one spectrum, two drums."""
    assert ref.gww_eigs(25).size == 25
    assert np.all(np.diff(ref.gww_eigs(25)) > 0)


# --- the truth must not pre-judge what counts as one eigenvalue --------------------------


def test_a_true_degeneracy_is_bit_identical_not_merely_close():
    """Why exact equality suffices in `collapse`, and why nothing coarser is needed.

    The eigenvalue formula is symmetric in the indices, so a mathematically degenerate pair
    comes out as the SAME float, not two nearby ones. Measured over the unit square and the
    2x1 rectangle, every degenerate group has a relative spread of exactly zero.
    """
    for L, H in ((1.0, 1.0), (2.0, 1.0)):
        groups = {}
        for m in range(1, 25):
            for n in range(1, 25):
                groups.setdefault(round(m**2 / L**2 + n**2 / H**2, 12), []).append(
                    float(ref.rect_eig(m, n, L, H)))
        for g in groups.values():
            if len(g) > 1:
                assert max(g) == min(g), f"a true degeneracy is not bit-identical: {g}"


@pytest.mark.parametrize("delta", [1e-6, 1e-8, 1e-9, 1e-11])
def test_a_near_degenerate_pair_stays_two_eigenvalues(delta):
    """Ground truth reports how many eigenvalues there ARE. Whether a search may merge two of
    them is the search's business to declare and the harness's to score -- not the
    reference's to decide in advance.

    `rect(1+delta, 1)` splits the (1,2)/(2,1) pair by a relative 1.2*delta, so at delta=1e-11
    the two are 1.2e-11 apart: eleven orders above float noise and unambiguously distinct. An
    earlier default of `rtol=1e-9` merged these, which would have scored a solver that
    correctly resolved them as reporting a spurious root.
    """
    L = 1.0 + delta
    spec = ref.rect_spectrum(48.0, 51.0, L, 1.0)
    assert spec.size == 2
    assert spec[0] != spec[1], "two distinct eigenvalues must not be the same float"
    assert [m for _, m in ref.collapse(spec)] == [1, 1]


def test_collapse_defaults_to_exact_and_a_tolerance_is_opt_in():
    a = 49.0
    b = a * (1 + 1e-12)
    assert ref.collapse([a, b]) == [(a, 1), (b, 1)]          # default: distinct
    assert ref.collapse([a, b], rtol=1e-9) == [(a, 2)]       # opt in, deliberately
    assert ref.collapse([a, a]) == [(a, 2)]                  # exact equality still merges


def test_snap_makes_a_true_degeneracy_bit_identical_where_arithmetic_does_not():
    """`rect(sqrt(2), 1)` has mathematically degenerate pairs whose computed values differ by
    4.4e-16, because `L**2` is 2.0000000000000004 and `m**2/L**2` rounds per `m`. Ground truth
    has to say "one eigenvalue, twice", not "two eigenvalues 4.4e-16 apart"."""
    L = np.sqrt(2.0)
    raw = np.sort(np.array([float(ref.rect_eig(m, n, L, 1.0))
                            for m in range(1, 30) for n in range(1, 30)]))
    raw = raw[raw <= 400.0]   # the first such pair is at 133.24
    near = [(raw[i], raw[i + 1]) for i in range(raw.size - 1)
            if 0 < abs(raw[i] - raw[i + 1]) / raw[i + 1] < 1e-13]
    assert near, "this aspect ratio no longer exercises the rounding case"

    sp = ref.rect_spectrum(0.0, 400.0, L, 1.0)
    still = [(sp[i], sp[i + 1]) for i in range(sp.size - 1)
             if 0 < abs(sp[i] - sp[i + 1]) / sp[i + 1] < 1e-13]
    assert not still, f"snap left merely-close values: {still}"


def test_snap_does_not_reach_a_genuine_near_degeneracy():
    """1e-13 is 230x above the worst rounding noise and 120x below the tightest near-pair."""
    sp = ref.rect_spectrum(48.0, 51.0, 1 + 1e-11, 1.0)
    assert sp.size == 2 and sp[0] != sp[1]
