"""Rigorous eigenvalue inequalities: they hold, and they are tight where theory says.

A bound that is merely never violated on the shapes you happened to try is weak evidence.
What makes these testable is that every one of them is an EQUALITY for the disk -- Faber-Krahn,
Szego-Weinberger, the inradius bound and the PPW ratio all attain their extremum there -- so
each can be checked against a closed form and not just against an inequality.

`polya` is the exception in two ways: it has no equality case among the shipped shapes, and it
is a conjecture rather than a theorem. Both are asserted, the second by reading the docstring.
"""
import numpy as np
import pytest

from chladni import bounds
from chladni import reference as ref
from chladni.geometry import GWW1, L_shape, disk, eq_tri, rect, reg_ngon

DIRICHLET = [("rect 2x1", rect(2.0, 1.0)), ("square", rect(1.0, 1.0)),
             ("L_shape", L_shape()), ("eq_tri", eq_tri(1.0)),
             ("reg_ngon 6", reg_ngon(6)), ("GWW1", GWW1()), ("disk", disk(1.0))]


def lam1_of(dom, name):
    """The true first eigenvalue, from closed form where there is one and the table otherwise."""
    return {
        "rect 2x1": lambda: ref.rect_eigs(1, 2.0, 1.0)[0],
        "square": lambda: ref.rect_eigs(1, 1.0, 1.0)[0],
        "L_shape": lambda: ref.L_shape_eigs(1)[0],
        "eq_tri": lambda: ref.eq_tri_eigs(1, 1.0)[0],
        "reg_ngon 6": lambda: ref.reg_ngon_eigs(1, 6)[0],
        "GWW1": lambda: ref.gww_eigs(1)[0],
        "disk": lambda: ref.disk_eigs(1, 1.0)[0],
    }[name]()


# --- the bounds hold ----------------------------------------------------------------------


@pytest.mark.parametrize("name,dom", DIRICHLET, ids=[n for n, _ in DIRICHLET])
def test_faber_krahn_is_below_the_true_first_eigenvalue(name, dom):
    assert bounds.faber_krahn(dom) <= lam1_of(dom, name) * (1 + 1e-12)


@pytest.mark.parametrize("name,dom", DIRICHLET, ids=[n for n, _ in DIRICHLET])
def test_the_inradius_bound_is_above_it(name, dom):
    assert bounds.inradius_upper(dom) >= lam1_of(dom, name) * (1 - 1e-8)


@pytest.mark.parametrize("name,dom", DIRICHLET, ids=[n for n, _ in DIRICHLET])
def test_the_two_bounds_bracket_the_answer(name, dom):
    """Together they are a search window, which is exactly what `weyl_window` uses them for."""
    lo, hi = bounds.faber_krahn(dom), bounds.inradius_upper(dom)
    assert lo <= lam1_of(dom, name) * (1 + 1e-12) <= hi * (1 + 1e-8)


@pytest.mark.parametrize("name,dom", DIRICHLET, ids=[n for n, _ in DIRICHLET])
def test_polya_is_below_every_eigenvalue(name, dom):
    """Vectorized in k, and a lower bound for every k -- not only the first."""
    truth = {"rect 2x1": lambda: ref.rect_eigs(8, 2.0, 1.0),
             "square": lambda: ref.rect_eigs(8, 1.0, 1.0),
             "L_shape": lambda: ref.L_shape_eigs(8),
             "eq_tri": lambda: ref.eq_tri_eigs(8, 1.0),
             "reg_ngon 6": lambda: ref.reg_ngon_eigs(8, 6),
             "GWW1": lambda: ref.gww_eigs(8),
             "disk": lambda: ref.disk_eigs(8, 1.0)}[name]()
    assert np.all(bounds.polya(np.arange(1, 9), dom) <= truth * (1 + 1e-10))


def test_the_ppw_ratio_bound_holds():
    for name, dom in DIRICHLET:
        t = {"rect 2x1": lambda: ref.rect_eigs(2, 2.0, 1.0),
             "square": lambda: ref.rect_eigs(2, 1.0, 1.0),
             "L_shape": lambda: ref.L_shape_eigs(2),
             "eq_tri": lambda: ref.eq_tri_eigs(2, 1.0),
             "reg_ngon 6": lambda: ref.reg_ngon_eigs(2, 6),
             "GWW1": lambda: ref.gww_eigs(2),
             "disk": lambda: ref.disk_eigs(2, 1.0)}[name]()
        assert t[1] / t[0] <= bounds.payne_polya_weinberger(dom) * (1 + 1e-10), name


# --- and they are EQUALITIES for the disk -------------------------------------------------


def test_faber_krahn_is_exact_for_the_disk():
    """Equality iff the domain is a disk. This is why `weyl_window` pads below it: an
    unpadded window starts exactly ON lam_1, and a grid search that ignores its endpoints
    then misses the first mode -- which lappy did, silently, returning modes 2..k+1."""
    assert bounds.faber_krahn(disk(1.0)) == pytest.approx(ref.disk_eigs(1, 1.0)[0], rel=1e-14)


def test_the_inradius_bound_is_exact_for_the_disk():
    assert bounds.inradius_upper(disk(1.0)) == pytest.approx(ref.disk_eigs(1, 1.0)[0],
                                                             rel=1e-8)


def test_the_ppw_bound_is_attained_by_the_disk():
    lam = ref.disk_eigs(2, 1.0)
    assert lam[1] / lam[0] == pytest.approx(bounds.payne_polya_weinberger(), rel=1e-12)


def test_szego_weinberger_is_exact_for_the_disk():
    """The Neumann twin: mu_1 is the first NONZERO eigenvalue, which is `disk_eigs(2,...)[1]`
    once the constant mode has been prepended."""
    mu1 = ref.disk_eigs(2, 1.0, "neu")[1]
    assert bounds.szego_weinberger(disk(1.0, bc="neu")) == pytest.approx(mu1, rel=1e-14)


# --- the two calling paths, and the guard rails -------------------------------------------


def test_the_scalar_and_domain_paths_agree():
    d = rect(2.0, 1.0)
    assert bounds.faber_krahn(area=d.area) == pytest.approx(bounds.faber_krahn(d))
    assert bounds.polya(3, area=d.area) == pytest.approx(bounds.polya(3, d))
    assert bounds.inradius_upper(inradius=d.inradius) == pytest.approx(
        bounds.inradius_upper(d))


def test_a_domain_path_rejects_the_wrong_boundary_condition():
    """The scalar path cannot check this, which is worth knowing: a caller passing raw area
    can silently apply a Dirichlet bound to a Neumann problem."""
    neu = rect(1.0, 1.0, bc="neu")
    for fn in (bounds.faber_krahn, bounds.payne_polya_weinberger, bounds.inradius_upper):
        with pytest.raises(ValueError, match="bc_type"):
            fn(neu)
    with pytest.raises(ValueError, match="bc_type"):
        bounds.polya(1, neu)
    with pytest.raises(ValueError, match="bc_type"):
        bounds.szego_weinberger(rect(1.0, 1.0))


def test_missing_arguments_say_what_is_missing():
    with pytest.raises(ValueError, match="area"):
        bounds.faber_krahn()
    with pytest.raises(ValueError, match="inradius"):
        bounds.inradius_upper()


def test_polya_says_it_is_a_conjecture():
    """The only unproven statement in the module. It stays labelled: shipping a conjecture
    beside four theorems without saying which is which is the failure mode here."""
    assert "conjectured" in bounds.polya.__doc__.lower()
