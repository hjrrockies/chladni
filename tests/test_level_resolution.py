"""Phase 3: what a level's COUNT stands for, and when it is a shortfall.  (2026-10-11)

`docs/design/douse_requirements.md`, "Phase 3, decided 2026-10-11". Three per-level quantities,
all reporting -- no search behaviour changes:

    margin          = DELTA_SAFETY * prec * lam / delta_res      (R, the request over eps/c)
    resolution      = max(prec * lam, bound, width)              (the neighbourhood counted)
    count_shortfall = the level is unmet                         (a FLAG; it never raises)

Measured before the design (`notes/margin_rule_offline.md`, with its 2026-10-11 correction): on
the douse benchmark no level that met `prec` had a wrong count. Every count defect was a value
off by 1e-6..1e-4 that its bound covered, a missed partner beside an unmet level, a simple level
read double inside its floor, or a whole level dropped -- the last is the count audit's
(Phase 4), since no returned level exists to carry a flag.
"""
import os
import sys

import numpy as np
import pytest

from chladni.mps.refine import DELTA_SAFETY
from chladni.mps.search import Level

PREC = 1e-10


# --- the three quantities, on a Level ------------------------------------------------------

def test_margin_is_the_request_over_the_floor_scale():
    """`delta_res = DELTA_SAFETY * eps/c`, so `prec*lam / (eps/c)` is this."""
    L = Level(lam=100.0, mult=1, delta_res=1e-9, prec=PREC)
    assert L.margin == pytest.approx(DELTA_SAFETY * PREC * 100.0 / 1e-9)


def test_a_met_level_counts_at_the_merge_licence_and_is_not_short():
    L = Level(lam=100.0, mult=2, delta_res=1e-9, prec=PREC)
    assert L.met(PREC)
    assert L.resolution == pytest.approx(PREC * 100.0)
    assert not L.count_shortfall


def test_an_unmet_level_widens_to_its_own_bar_and_flags_its_count():
    """A value known to 5e-6 cannot claim its count at 1e-8 absolute: the eigenvalues it stands
    for are the ones within its bar, and the count there was read where the floor blurs it."""
    L = Level(lam=100.0, mult=1, delta_res=1e-6, mp_bound=5e-6, delta_res_admitted=False,
              prec=PREC)
    assert not L.met(PREC)
    assert L.resolution == pytest.approx(5e-6)
    assert L.count_shortfall
    assert L.margin < 2


def test_a_collapsed_level_stands_for_its_wide_bracket():
    """`width` is the special case `resolution` generalizes (`douse_requirements.md`)."""
    L = Level(lam=100.0, mult=3, delta_res=1e-9, grouping_ambiguous=True,
              wide_bracket=(99.9, 100.1), prec=PREC)
    assert L.resolution == pytest.approx(0.2)


def test_a_level_built_without_a_request_claims_nothing():
    """`prec` is stamped by `solve`. A Level made elsewhere has no request to fall short of."""
    L = Level(lam=100.0, mult=1, delta_res=1e-6)
    assert np.isnan(L.margin) and np.isnan(L.resolution)
    assert not L.count_shortfall


# --- on real solves -------------------------------------------------------------------------

def _starved_lshape(n=16):
    """Fourier-Bessel at the re-entrant corner only, n = 16, at 1e-10: all five levels in
    [5, 40] come back unmet, values ~1e-5 off, counts certified, nothing declined (probed
    2026-10-11). n = 8 returns NO level -- a whole-window drop, which is the count audit's."""
    from chladni.geometry import Polygon
    from chladni.helmholtz import FourierBessel
    from chladni.mps import MPSEigensolver

    dom = Polygon(np.array([0, 2, 2 + 1j, 1 + 1j, 1 + 2j, 2j]))
    corner = max(dom.corners, key=lambda c: c.omega)
    fb = FourierBessel.at_corner(corner, n=n, domain=dom)
    return MPSEigensolver(dom, basis=fb, n_bdry=3 * n, n_int=n, prec=PREC, rng=0)


@pytest.mark.slow
def test_solve_stamps_the_request_and_the_flag_is_the_unmet_list():
    """After `certify_levels` attaches Moler-Payne bounds, too: the properties read the level's
    current bar, so they cannot go stale."""
    from chladni.reference.tables import L_shape_eigs

    res = _starved_lshape().search_interval(5.0, 40.0)
    assert res.levels, "the probe case returns levels"
    assert all(L.prec == PREC for L in res.levels)
    assert {L.lam for L in res.levels if L.count_shortfall} == set(res.diagnostics.unmet)
    assert res.diagnostics.n_count_shortfall == len(res.diagnostics.unmet)
    # Class (b): a value off by ~1e-5 still stands for its true eigenvalue, within its resolution.
    truth = L_shape_eigs(5)
    for L in res.levels:
        assert np.min(np.abs(truth - L.lam)) <= L.resolution, (
            f"{L.lam}: the truth is outside the resolution {L.resolution:.2e} it claims")


@pytest.mark.slow
def test_a_count_shortfall_never_raises():
    """`solve_interval` raises on an UNCERTAIN count and a decline. A count read on an unmet
    level is neither: about 40% of levels at 1e-10 are unmet even on the douse references'
    basis, so it warns (the precision shortfall it already reports) and returns."""
    from chladni.mps import PrecisionWarning

    with pytest.warns(PrecisionWarning):
        spec = _starved_lshape().solve_interval(5.0, 40.0)
    assert len(spec.lam) == 5


# --- the contract on douse shapes -----------------------------------------------------------

_BENCH = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "douse")


def _bench_solve(case_id, jit=(0.0, 0.0)):
    sys.path.insert(0, os.path.abspath(_BENCH))
    import json

    import make_references as MR
    import scoreboard  # noqa: F401 -- registers basis "D"
    from chladni.mps import MPSEigensolver

    cases = {c["id"]: c for c in MR.load_cases()}
    ref = {r["id"]: r for r in json.load(open(os.path.join(_BENCH, "references.json")))["cases"]}
    ref = ref[case_id]
    lo, hi = ref["window"][0] * (1 + jit[0]), ref["window"][1] * (1 + jit[1])
    dom = MR.polygon(cases[case_id]["vertices"])
    b = MR.basis_for(dom, "D", np.sqrt(hi))
    n = len(b)
    s = MPSEigensolver(dom, basis=b, n_bdry=3 * n, n_int=n, prec=PREC, rng=0, reg="svd")
    return s.search_interval(lo, hi), ref


@pytest.mark.slow
# Basis D is the scoreboard's stand-in, sized at 1.0x lappy's per-corner rule: its Bessel
# columns warn that they cannot reach 1e-8 on some points. That shortfall is the subject here.
@pytest.mark.filterwarnings("ignore:FourierBessel can deliver:chladni._core.PrecisionWarning")
@pytest.mark.parametrize("case_id, jit", [
    ("T1.5-r2", (0.0, 0.0)),            # values ~1e-5 off: wrong at the licence, right at the bar
    ("T2.2", (0.0, 0.0)),               # every level unmet; the same, ten times over
    ("T2.9-N5k3convex", (0.0, 0.0)),    # clean here: a control
    ("T2.9-N5k3convex", (-1e-12, -1e-12)),  # a simple level read double inside its floor (2026-10-11)
    ("T1.4-oct", (0.0, 0.0)),           # every level met: a control
])
def test_no_count_is_silently_wrong_within_the_resolution_it_claims(case_id, jit):
    """The Phase 3 contract on the stand-in basis D at 1e-10, on cases that carried count
    defects in `notes/margin_rule_offline.md`: where returned levels disagree with the reference
    inside the resolution they claim, one of them is flagged -- `count_shortfall`, or an existing
    count flag. Right, or flagged; silently wrong is the failure.

    Judged per GROUP of levels whose resolution intervals overlap, not per level: two simple
    levels 2e-8 apart for one true eigenvalue (N5k3convex on the jittered window) each hold
    one truth inside their own resolution, and only their total is wrong.

    Judged only where every reference level inside a group's union is trusted (`certified` or
    `fem_resolved`) and below the first untrusted one, as the scoreboard scores. A reference
    level no returned level covers is a DROP, which Phase 3 cannot flag (no level carries it);
    it is the count audit's, Phase 4, and is not asserted here."""
    res, ref = _bench_solve(case_id, jit)
    trusted = []
    for T in ref["levels"]:
        if T["status"] == "provisional":
            break
        trusted.append(T)
    top = trusted[-1]["lam"] * (1 + 1e-4)
    if len(trusted) < len(ref["levels"]):
        top = np.sqrt(trusted[-1]["lam"] * ref["levels"][len(trusted)]["lam"])
    all_lam = np.array([T["lam"] for T in ref["levels"]])
    # A reference level is an interval, its value +- its own bound: the references certify
    # COUNTS, and some values are known only to 1e-6 relative (T1.3).
    all_bar = np.array([T["bound"] if np.isfinite(T.get("bound", np.nan)) else 0.0
                        for T in ref["levels"]])

    levels = sorted((L for L in res.levels if L.lam < top), key=lambda L: L.lam)
    groups = []
    for L in levels:
        if groups and L.lam - L.resolution <= groups[-1][-1].lam + groups[-1][-1].resolution:
            groups[-1].append(L)
        else:
            groups.append([L])
    silent = []
    for g in groups:
        lo = min(L.lam - L.resolution for L in g)
        hi = max(L.lam + L.resolution for L in g)
        inside = np.flatnonzero((all_lam + all_bar >= lo) & (all_lam - all_bar <= hi))
        if any(ref["levels"][i]["status"] == "provisional" for i in inside):
            continue
        n_true = sum(ref["levels"][i]["mult"] for i in inside)
        n_got = sum(L.mult for L in g)
        flagged = any(L.count_shortfall or L.count_uncertain or L.grouping_ambiguous for L in g)
        if n_true != n_got and not flagged:
            silent.append(([round(L.lam, 9) for L in g], n_got, n_true))
    assert not silent, f"{case_id}: counts silently wrong within their resolution: {silent}"
