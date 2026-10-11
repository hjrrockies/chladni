"""Phase 4 step 3: `on_shortfall="flag" | "raise"`.  (2026-10-11)

`docs/design/douse_requirements.md`, "Precision per level". `"flag"`, the default, changes
nothing: shortfalls are reported on the levels and the diagnostics. `"raise"` turns them into a
`PrecisionShortfall` carrying what a re-plan needs. It raises EARLY, at the first refined level
that misses `prec` -- step 4 measured that saving a median 57% of a failing solve's wall time
(`notes/early_shortfall.md`) -- with that level, its `margin` and the values located so far.
Shortfalls known only at the end (a count audit of `short` or `excess`, an uncertified count, a
decline) raise after the solve, with the whole result and the audit.

A shortfall is any of: a level that misses `prec` (`count_shortfall`), a count audit that says
`short` or `excess`, a count that could not be certified, or a decline.
"""
import numpy as np
import pytest

from chladni.geometry import L_shape
from chladni.mps import PrecisionShortfall, PrecisionWarning

LO, HI = 5.0, 40.0


def _solver(basis="recipe", n=16):
    from chladni.helmholtz import FourierBessel
    from chladni.mps import MPSEigensolver

    dom = L_shape()
    if basis == "recipe":
        from hard_roster import recipe_basis
        b = recipe_basis(dom, 80)
    else:                                  # starved: FB at the re-entrant corner only
        corner = max(dom.corners, key=lambda c: c.omega)
        b = FourierBessel.at_corner(corner, n=n, domain=dom)
    m = len(b)
    return MPSEigensolver(dom, basis=b, n_bdry=3 * m, n_int=m, prec=1e-10,
                          rng=np.random.default_rng(0))


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="on_shortfall"):
        _solver().search_interval(LO, 33.0, on_shortfall="abort")


@pytest.mark.slow
def test_flag_is_the_default_and_changes_nothing():
    """The starved L-shape (n = 16): all five levels unmet. Reported, returned, not raised."""
    res = _solver("starved").search_interval(LO, HI)
    assert len(res.levels) == 5 and all(L.count_shortfall for L in res.levels)


@pytest.mark.slow
def test_raise_on_a_clean_solve_returns_normally():
    """The recipe basis meets 1e-10 at every level in [5, 33] and the audit is ok."""
    res = _solver().search_interval(LO, 33.0, on_shortfall="raise")
    assert len(res.levels) == 5


@pytest.mark.slow
def test_raise_stops_at_the_first_unmet_level_with_what_a_replan_needs():
    with pytest.raises(PrecisionShortfall) as info:
        _solver("starved").search_interval(LO, HI, on_shortfall="raise")
    e = info.value
    assert e.result is None and e.audit is None          # aborted: nothing assembled yet
    assert e.level is not None and e.level.count_shortfall
    assert e.margin == e.level.margin and e.margin < 2
    assert e.level.lam in e.located
    assert "unmet" in str(e)


@pytest.mark.slow
def test_the_early_abort_spends_less():
    """The point of aborting early: the refines after the first unmet level, stage 3 and the
    certification are not paid for."""
    s_flag, s_raise = _solver("starved"), _solver("starved")
    obj_flag, obj_raise = s_flag.objective(LO, HI), s_raise.objective(LO, HI)
    s_flag.search_interval(LO, HI, objective=obj_flag)
    with pytest.raises(PrecisionShortfall):
        s_raise.search_interval(LO, HI, objective=obj_raise, on_shortfall="raise")
    assert obj_raise.n_evals < obj_flag.n_evals, (obj_raise.n_evals, obj_flag.n_evals)


@pytest.mark.slow
def test_raise_on_a_dropped_window_names_the_audit():
    """n = 8 finds nothing in [5, 40]: no level to point at, so the audit is the reason."""
    with pytest.raises(PrecisionShortfall) as info:
        _solver("starved", n=8).search_interval(LO, HI, on_shortfall="raise")
    e = info.value
    assert e.level is None and np.isnan(e.margin)
    assert e.audit.verdict == "short"
    assert "count audit" in str(e)


@pytest.mark.slow
def test_solve_interval_raises_instead_of_warning():
    s = _solver("starved")
    with pytest.warns(PrecisionWarning):
        s.solve_interval(LO, HI)                        # the default still warns and returns
    with pytest.raises(PrecisionShortfall):
        s.solve_interval(LO, HI, on_shortfall="raise")
