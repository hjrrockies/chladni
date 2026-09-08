"""The method-agnostic eigensolver contract.

`Eigensolver` lives at `chladni/eigensolver.py`, not in `chladni/mps/`, because a later
`chladni/fem/` must not import its base class out of the MPS package. That placement is the
first thing this file defends.

What is on the base class is deliberately thin: a domain, a precision, and
`solve_interval(a, b)`. No tension, no basis, no collocation -- none of it survives contact
with FEM, which assembles a sparse symmetric pencil and solves it ONCE for many eigenvalues,
with no scan and no scalar function of `lam` to minimize.

`solve(n)` is concrete over `solve_interval` plus a Weyl window, and it is testable now even
though no search exists: a stub subclass whose `solve_interval` returns closed-form rectangle
eigenvalues in `[a, b]` exercises the windowing completely. That keeps the one concrete method
inside the project's spec-tests-implementation rule instead of shipping untested, and it is
the test lappy did not have when its disk window started ABOVE lam_1 and silently returned
modes 2..k+1 -- "every value correct to ~14 digits, every certificate valid, and nothing in
the pipeline able to detect the omission."
"""
import numpy as np
import pytest

from chladni.eigensolver import Eigensolver, Spectrum, k_to_lam, lam_to_k, weyl_window
from chladni.geometry import disk, eq_tri, rect

from conftest import EXACT, disk_eigs, eq_tri_eigs, rect_eigs


class ClosedForm(Eigensolver):
    """An eigensolver that cheats: it already knows the answer.

    Everything the base class does -- the window, the widening, the truncation to n -- is
    exercised through this, with no MPS anywhere in the call.

    `decline` makes `solve_interval` return a declining `Spectrum` instead of an answer, which
    is how the base class's response to a DECLARED failure is tested. Truth arrives unfolded,
    as `chladni.reference` supplies it, and is grouped into levels here -- so the double
    exercises the round trip rather than sidestepping it.
    """

    def __init__(self, domain, truth, **kw):
        super().__init__(domain, **kw)
        self.calls = []          # a plain list; ClosedForm is a test double, not Frozen
        self._truth = truth
        self.decline = None

    def solve_interval(self, a, b):
        self.calls.append((a, b))
        if self.decline is not None:
            return Spectrum.declining(self.decline, prec=self.prec)
        t = self._truth
        return Spectrum.from_unfolded(t[(t >= a) & (t <= b)], prec=self.prec)


# --- Spectrum: what an eigensolver returns ----------------------------------------------


def test_unfolded_repeats_each_level_by_its_multiplicity():
    """The round trip that the old array-only contract could not make. A COMPUTED
    multiplicity of 2 and two levels a whisker apart are different claims, and an unfolded
    array cannot tell them apart -- so the multiplicity has to survive as an integer."""
    s = Spectrum.from_unfolded(disk_eigs(12, 1.0), prec=1e-8)
    assert np.array_equal(s.mult, [1, 2, 2, 1, 2, 2, 2])
    assert s.count == 12
    assert np.allclose(s.unfolded, disk_eigs(12, 1.0), rtol=0, atol=EXACT)
    assert np.all(np.diff(s.lam) > 0), "levels are strictly ascending"


def test_a_declining_spectrum_carries_its_reason_and_is_falsy():
    """Declining is a VALUE, not an exception (`candidates.md` section 5). Falsy so that
    `if spectrum:` treats it as nothing usable without anyone having to remember to check."""
    s = Spectrum.declining("the slope never settled", prec=1e-8)
    assert s.declined and not s
    assert "slope" in s.reason
    assert s.count == 0 and s.unfolded.size == 0


def test_unmet_is_derived_from_the_attained_bound_against_prec():
    """Clause (3), honesty: the bar is reported, and whether it met the request is a
    comparison anyone can redo rather than a flag the solver sets."""
    lam = np.array([10.0, 20.0])
    tight = Spectrum(lam, np.array([1, 1]), np.array([1e-9, 1e-9]), prec=1e-8)
    loose = Spectrum(lam, np.array([1, 1]), np.array([1e-9, 1e-3]), prec=1e-8)
    assert not tight.unmet.any() and tight.achieved
    assert list(loose.unmet) == [False, True] and not loose.achieved


def test_a_spectrum_with_no_bound_claims_nothing():
    """A solver that does not measure its own error says so with nan, and `achieved` is then
    False -- an unmeasured bound is not a met one."""
    s = Spectrum.from_unfolded(np.array([1.0, 2.0]), prec=1e-8)
    assert np.all(np.isnan(s.bound))
    assert not s.achieved


# --- the spectral parameter -------------------------------------------------------------


def test_lam_and_k_are_inverse():
    lam = np.array([1.0, 9.6397238440219410527, 1e4])
    assert np.allclose(k_to_lam(lam_to_k(lam)), lam, rtol=0, atol=EXACT)


def test_lam_to_k_rejects_a_nonpositive_iterate():
    """The practical argument for carrying `k` below this layer, from `check_k`'s docstring:
    a line search hands `lam` a negative iterate and `sqrt` of that is a complex number
    arriving three functions later as a confusing error about array dtypes. Catch it here."""
    for bad in (0.0, -1.0, np.nan):
        with pytest.raises(ValueError, match="lam must"):
            lam_to_k(bad)


# --- the Weyl window --------------------------------------------------------------------


@pytest.mark.parametrize("domain,truth", [
    (rect(2.0, 1.0), rect_eigs(12, 2.0, 1.0)),
    (rect(1.0, 1.0), rect_eigs(12, 1.0, 1.0)),
    (disk(1.0), disk_eigs(12, 1.0)),
    (eq_tri(1.0), eq_tri_eigs(12, 1.0)),
])
def test_the_window_brackets_the_first_n_eigenvalues(domain, truth):
    a, b = weyl_window(domain, 10)
    assert a < truth[0], "window must start strictly below lam_1"
    assert b >= truth[9], "window must reach the 10th eigenvalue"


def test_the_window_starts_below_lam_1_even_when_faber_krahn_is_sharp():
    """Faber-Krahn is an EQUALITY for the disk, so an unpadded lower endpoint lands exactly
    on lam_1 -- and a grid search that ignores endpoints then misses it. This is lappy's
    `_WINDOW_PAD` bug, pinned so the padding cannot be removed as redundant."""
    a, _ = weyl_window(disk(1.0), 5)
    assert a < disk_eigs(1, 1.0)[0]


def test_the_window_is_monotone_in_n():
    d = rect(2.0, 1.0)
    ends = [weyl_window(d, n)[1] for n in (1, 5, 10, 40)]
    assert ends == sorted(ends)


# --- solve(n) over solve_interval -------------------------------------------------------


@pytest.mark.parametrize("domain,truth", [
    (rect(2.0, 1.0), rect_eigs(20, 2.0, 1.0)),
    (disk(1.0), disk_eigs(20, 1.0)),
    (eq_tri(1.0), eq_tri_eigs(20, 1.0)),
])
def test_solve_returns_the_first_n(domain, truth):
    s = ClosedForm(domain, truth)
    assert np.allclose(s.solve(8), truth[:8], rtol=0, atol=EXACT)


@pytest.mark.parametrize("n", range(1, 13))
def test_solve_widens_the_window_when_it_came_up_short(n):
    """Weyl's law is asymptotic, so the window under-reaches at small n. Sweeping n is how
    that gets exercised: wherever the first window falls short the base class must widen,
    rather than silently returning fewer than asked for."""
    truth = rect_eigs(40, 2.0, 1.0)
    s = ClosedForm(rect(2.0, 1.0), truth)
    got = s.solve(n)
    assert len(got) == n
    assert np.allclose(got, truth[:n], rtol=0, atol=EXACT)
    assert all(a < b for a, b in s.calls)


def test_solve_reports_rather_than_under_delivering():
    """If widening cannot reach n, say so. Returning 6 eigenvalues from `solve(8)` with no
    signal is the failure mode this rules out."""
    s = ClosedForm(rect(2.0, 1.0), rect_eigs(3, 2.0, 1.0))   # only 3 exist to be found
    with pytest.raises(RuntimeError, match="8"):
        s.solve(8)


def test_solve_stops_at_a_DECLARED_decline_instead_of_widening():
    """The defect the rich return type exists to remove.

    Under the old array contract a short return was ambiguous between "there really are only
    k here", "the search missed some" and "the search declined", and the base class resolved
    it the only way it could: widen and ask again, eight times, then blame the spectrum. A
    declared decline is not a short answer, and widening a window whose search has already
    given up costs eight more searches and ends in a message naming the wrong cause.
    """
    s = ClosedForm(rect(2.0, 1.0), rect_eigs(40, 2.0, 1.0))
    s.decline = "the slope never settled"
    with pytest.raises(RuntimeError, match="declined"):
        s.solve(8)
    assert len(s.calls) == 1, f"widened {len(s.calls) - 1} times past a declared decline"


def test_the_decline_message_names_the_reason_the_solver_gave():
    """Clause (3) again: the caller is told what went wrong, not merely that something did."""
    s = ClosedForm(rect(2.0, 1.0), rect_eigs(40, 2.0, 1.0))
    s.decline = "the slope never settled"
    with pytest.raises(RuntimeError, match="slope never settled"):
        s.solve(8)


def test_solve_interval_is_abstract():
    with pytest.raises(TypeError):
        Eigensolver(rect(1.0, 1.0))


# --- the domain contract ----------------------------------------------------------------


def test_the_base_class_carries_domain_prec_and_bc_type():
    d = rect(2.0, 1.0, bc="neu")
    s = ClosedForm(d, rect_eigs(5, 2.0, 1.0))
    assert s.domain is d
    assert s.prec == d.prec
    assert s.bc_type == "neu"


def test_prec_defaults_to_the_domains_own():
    d = rect(2.0, 1.0, prec=1e-12)
    assert ClosedForm(d, rect_eigs(5, 2.0, 1.0)).prec == 1e-12


def test_the_base_class_knows_nothing_about_tension_or_bases():
    """The FEM argument, as an assertion. If any of these appear on `Eigensolver`, the
    abstraction has been drawn around MPS and the next method will not fit it."""
    for name in ("tension", "tensions", "basis", "coef", "pencil", "collocation"):
        assert not hasattr(Eigensolver, name), f"Eigensolver.{name} is MPS-specific"
