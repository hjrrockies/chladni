"""The full pipeline on every real case, at EVERY precision in the ladder.

`bracket_score.PREC_LADDER` is `(1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14)`, and until this file
existed nothing drove the composed pipeline across all of it: `test_gate.py` sweeps the ladder
on the BRACKETING stage, and `test_solve.py` drives the pipeline at one or two precisions. The
gap between those two is where a defect lived unseen -- at `prec = 1e-04`, wide enough for the
merge licence to swallow genuinely separated eigenvalues, a collapsed cluster DROPS members and
reports a total lower than the truth.

Each cell is scored to one of six verdicts, and which ones are allowed is the contract:

    .   exact: complete, no spurious, multiplicities right, every level within `prec`
    b   short of `prec` but REPORTED -- clause (2)'s OR branch, and legitimate
    D   declined, with a reason -- neither pass nor failure (`candidates.md` section 5)
    !   short of `prec` SILENTLY -- must never happen, this is clause (3)
    S   spurious level -- must never happen
    M   a level in the window was not found

`b` dominates at `prec <= 1e-10` because these pencils floor near 1e-10 relative, so the
tighter requests are honestly refused rather than silently missed. That is the contract working
and not a weakness of the run.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from chladni import asymp
from chladni.mps.objective import Objective
from chladni.mps.search import solve
from chladni.reference import rect_spectrum

import hard_roster as hr
import rect_suite as rs
from bracket_score import PREC_LADDER

pytestmark = pytest.mark.slow

#: The cluster cases whose members fall inside `tol_merge` at the loosest precision. Named
#: because they are the ones the open Phase D defect drops members from.
COLLAPSE_VICTIMS = {"B:d=5.9e-03", "B:d=1.8e-03", "C:m=3,s=9.5e-02", "C:m=4,s=1.2e-01"}


def _cases():
    for fn in (rs.family_A, rs.family_B, rs.family_C, rs.family_D, rs.family_stress):
        for c in fn():
            yield "rect", c
    for _, f in hr.ROSTER.items():
        yield "domain", f()


def _truth(group, c):
    """The COMPLETE spectrum in the window. `c.truth` is restricted to a family's own modes --
    `family_C` lists only the cluster -- so scoring against it calls a genuine neighbouring
    eigenvalue spurious."""
    if group == "rect":
        return np.sort(np.asarray(rect_spectrum(c.a, c.b, c.domain.area, 1.0), float))
    return np.sort(np.asarray(c.truth, float))


def _levels_at(truth, prec):
    """`S(prec)`: values closer than `prec` relative are ONE level (section 2.3)."""
    lv, mu = [], []
    for v in truth:
        if lv and abs(v - lv[-1]) <= prec * abs(v):
            mu[-1] += 1
        else:
            lv.append(v); mu.append(1)
    return np.array(lv), np.array(mu)


def _score(group, c, prec):
    truth = _truth(group, c)
    hi = truth[-1] * (1 + 1e-9) if not c.complete else c.b
    truth = truth[truth <= hi]
    obj = Objective.from_pencil(
        c.pencil, c.a, c.b, variable="lam", prec=prec,
        expected_count=lambda lo, up: asymp.expected_count(lo, up, c.domain))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = solve(obj, prec=prec)
    d = res.diagnostics
    n = max(len(truth), 1)
    if d.declined:
        return dict(v="D", per_eig=d.evaluations / n, reason=d.decline_reason, total=0,
                    n_truth=len(truth))

    keep = [L for L in res.levels if L.lam <= hi]
    got = np.sort(np.concatenate([[L.lam] * L.mult for L in keep])) if keep else np.array([])
    err, matched = [], np.zeros(len(truth), bool)
    for g in got:
        j = int(np.argmin(np.where(matched, np.inf, np.abs(truth - g))))
        if not matched[j]:
            matched[j] = True
            err.append(abs(g - truth[j]) / truth[j])
    _, mu = _levels_at(truth, prec)
    worst = max(err) if err else float("nan")
    if (~matched).any():                          v = "M"
    elif len(got) > len(truth):                   v = "S"
    elif not np.array_equal([L.mult for L in keep], mu): v = "m"
    elif worst <= prec:                           v = "."
    elif len(d.unmet):                            v = "b"
    else:                                         v = "!"
    return dict(v=v, per_eig=d.evaluations / n, reason="", total=len(got),
                n_truth=len(truth), worst=worst)


@pytest.fixture(scope="module")
def grid():
    """The whole ladder, computed once. Pencils are memoized, so the cost is the solves."""
    out = {}
    for group, c in _cases():
        try:
            fp = hr.finest_prec(c)
        except Exception:
            fp = float("nan")
        for prec in PREC_LADDER:
            out[(c.label, prec)] = {**_score(group, c, prec), "finest": fp}
    return out


def test_no_case_EVER_misses_prec_silently(grid):
    """Clause (3), the one that makes every other number trustworthy. Falling short of `prec`
    is allowed -- the basis, not the request, decides what is reachable -- but doing it without
    saying so is the failure this whole apparatus exists to prevent."""
    silent = [k for k, r in grid.items() if r["v"] == "!"]
    assert not silent, f"silent precision shortfall at {silent}"


def test_no_case_EVER_reports_a_spurious_level(grid):
    """A value that is not an eigenvalue is worse than a missing one: it cannot be detected
    downstream, where a short count at least reads as short."""
    spurious = [k for k, r in grid.items() if r["v"] == "S"]
    assert not spurious, f"spurious levels at {spurious}"


def test_every_decline_carries_a_reason(grid):
    """`candidates.md` section 5: declining is only honest if it is reported. A decline with an
    empty reason is indistinguishable from a crash that was caught."""
    for k, r in grid.items():
        if r["v"] == "D":
            assert r["reason"].strip(), f"{k} declined with no reason"


@pytest.mark.parametrize("prec", [1e-6, 1e-8])
def test_the_workable_precisions_are_exact_everywhere(prec, grid):
    """The two precisions every case's basis can serve. Every non-declining case must be
    complete, unspurious, correctly counted and within `prec` -- no `b` allowed here, because
    a shortfall at 1e-6 would mean the pencil is worse than its own floor measurement says."""
    bad = {k: r["v"] for k, r in grid.items() if k[1] == prec and r["v"] not in ".D"}
    assert not bad, f"prec={prec:g}: {bad}"


def test_completeness_holds_wherever_the_basis_can_serve_the_request(grid):
    """Levels may only go missing where the request is beyond the basis, or in the cluster
    cases the open Phase D defect drops members from. Anywhere else a miss is a coverage bug,
    which is the failure mode nothing downstream can detect."""
    missed = {k for k, r in grid.items()
              if r["v"] == "M" and k[0] not in COLLAPSE_VICTIMS
              and not (np.isfinite(r["finest"]) and k[1] < r["finest"])}
    assert not missed, f"levels missed within the basis's reach: {missed}"


def test_the_cost_rises_with_the_request_and_then_saturates(grid):
    """`search.md` section 2.6, not more work than asked. The median cost must be monotone in
    `prec` and must saturate: past the point where the pencil floors, refinement stops on
    `delta_res` rather than on `prec*lam`, so asking for more buys nothing and costs nothing."""
    med = [float(np.median([r["per_eig"] for k, r in grid.items() if k[1] == p]))
           for p in PREC_LADDER]
    assert med == sorted(med), f"cost not monotone in prec: {np.round(med, 1)}"
    assert med[-1] < 2.0 * med[0], f"cost more than doubled over six orders: {np.round(med, 1)}"
    assert med[-1] == pytest.approx(med[-2], rel=0.05), "cost should have saturated by 1e-12"


@pytest.mark.xfail(strict=True, reason=(
    "Phase D is not built: a cluster collapsed inside `tol_merge` reads its multiplicity at ONE "
    "representative, where the partner's branch is descending rather than minimal, so (C2) "
    "refuses it and members are DROPPED. See TODO section 1 and search.md 2.4a stage 3. When "
    "the Gram angle lands this XPASSes and the mark comes off."))
def test_a_collapsed_cluster_KEEPS_its_members(grid):
    """The total over a collapsed group must equal the truth in the window. It does not."""
    for label in sorted(COLLAPSE_VICTIMS):
        r = grid[(label, 1e-4)]
        assert r["total"] == r["n_truth"], (
            f"{label} at prec=1e-04: reported {r['total']} of {r['n_truth']} eigenvalues")


def test_the_ladder_report(grid, capsys):
    """Not an assertion -- the table itself, printed with `-s`. A standing report that nothing
    regenerated is how the 1e-04 defect stayed invisible."""
    with capsys.disabled():
        labels = sorted({k[0] for k in grid})
        print(f"\n\n{'case':24}" + "".join(f"{p:>9.0e}" for p in PREC_LADDER))
        print("-" * (24 + 9 * len(PREC_LADDER)))
        for lab in labels:
            print(f"{lab:24}" + "".join(f"{grid[(lab, p)]['v']:>9}" for p in PREC_LADDER))
        print(f"\n{'median evals/eig':24}" + "".join(
            f"{np.median([r['per_eig'] for k, r in grid.items() if k[1] == p]):>9.1f}"
            for p in PREC_LADDER))
