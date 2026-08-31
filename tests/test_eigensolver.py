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

from chladni.eigensolver import Eigensolver, k_to_lam, lam_to_k, weyl_window
from chladni.geometry import disk, eq_tri, rect

from conftest import EXACT, disk_eigs, eq_tri_eigs, rect_eigs


class ClosedForm(Eigensolver):
    """An eigensolver that cheats: it already knows the answer.

    Everything the base class does -- the window, the widening, the truncation to n -- is
    exercised through this, with no MPS anywhere in the call.
    """

    def __init__(self, domain, truth, **kw):
        super().__init__(domain, **kw)
        self.calls = []          # a plain list; ClosedForm is a test double, not Frozen
        self._truth = truth

    def solve_interval(self, a, b):
        self.calls.append((a, b))
        t = self._truth
        return t[(t >= a) & (t <= b)]


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
    (rect(2.0, 1.0), rect_eigs(2.0, 1.0, 12)),
    (rect(1.0, 1.0), rect_eigs(1.0, 1.0, 12)),
    (disk(1.0), disk_eigs(1.0, 12)),
    (eq_tri(1.0), eq_tri_eigs(1.0, 12)),
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
    assert a < disk_eigs(1.0, 1)[0]


def test_the_window_is_monotone_in_n():
    d = rect(2.0, 1.0)
    ends = [weyl_window(d, n)[1] for n in (1, 5, 10, 40)]
    assert ends == sorted(ends)


# --- solve(n) over solve_interval -------------------------------------------------------


@pytest.mark.parametrize("domain,truth", [
    (rect(2.0, 1.0), rect_eigs(2.0, 1.0, 20)),
    (disk(1.0), disk_eigs(1.0, 20)),
    (eq_tri(1.0), eq_tri_eigs(1.0, 20)),
])
def test_solve_returns_the_first_n(domain, truth):
    s = ClosedForm(domain, truth)
    assert np.allclose(s.solve(8), truth[:8], rtol=0, atol=EXACT)


@pytest.mark.parametrize("n", range(1, 13))
def test_solve_widens_the_window_when_it_came_up_short(n):
    """Weyl's law is asymptotic, so the window under-reaches at small n. Sweeping n is how
    that gets exercised: wherever the first window falls short the base class must widen,
    rather than silently returning fewer than asked for."""
    truth = rect_eigs(2.0, 1.0, 40)
    s = ClosedForm(rect(2.0, 1.0), truth)
    got = s.solve(n)
    assert len(got) == n
    assert np.allclose(got, truth[:n], rtol=0, atol=EXACT)
    assert all(a < b for a, b in s.calls)


def test_solve_reports_rather_than_under_delivering():
    """If widening cannot reach n, say so. Returning 6 eigenvalues from `solve(8)` with no
    signal is the failure mode this rules out."""
    s = ClosedForm(rect(2.0, 1.0), rect_eigs(2.0, 1.0, 3))   # only 3 exist to be found
    with pytest.raises(RuntimeError, match="8"):
        s.solve(8)


def test_solve_interval_is_abstract():
    with pytest.raises(TypeError):
        Eigensolver(rect(1.0, 1.0))


# --- the domain contract ----------------------------------------------------------------


def test_the_base_class_carries_domain_prec_and_bc_type():
    d = rect(2.0, 1.0, bc="neu")
    s = ClosedForm(d, rect_eigs(2.0, 1.0, 5))
    assert s.domain is d
    assert s.prec == d.prec
    assert s.bc_type == "neu"


def test_prec_defaults_to_the_domains_own():
    d = rect(2.0, 1.0, prec=1e-12)
    assert ClosedForm(d, rect_eigs(2.0, 1.0, 5)).prec == 1e-12


def test_the_base_class_knows_nothing_about_tension_or_bases():
    """The FEM argument, as an assertion. If any of these appear on `Eigensolver`, the
    abstraction has been drawn around MPS and the next method will not fit it."""
    for name in ("tension", "tensions", "basis", "coef", "pencil", "collocation"):
        assert not hasattr(Eigensolver, name), f"Eigensolver.{name} is MPS-specific"
