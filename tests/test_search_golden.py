"""A golden trace of `mps.search.solve` on the rectangle suite.  (2026-10-14)

`solve` is being split into named steps. Equivalence is proved here rather than argued: every
level (value, multiplicity, tension, attained bar, root count, certification flags, bracket) and
every diagnostic count of a solve on each rectangle case is recorded once
(`python -m tests.test_search_golden --record`, into `tests/data/search_golden.json`) and every
later version must reproduce them. The cases are `rect_suite.all_families` -- an isolated level,
split pairs down to 1e-13, clusters of three at four separations, exact multiplicities 2-4 -- at
two precisions. The fast tier replays a six-case subset (~10 s); the slow tier replays all 36.

ON THE RECORDING PLATFORM (`_meta.recorded_on`) the replay is exact: floats at 1e-10 relative,
counts, flags, reasons and strings exactly. ELSEWHERE it is structural. A different BLAS moves
the last ulps, and those move bracket edges, tensions and residuals by far more than 1e-10, and
can add or drop a bracket (Linux CI, 2026-10-30: `delta_res` 1.03e-08 against 1.01e-08,
`D:mult=3` 2 brackets against 1). So off the recording platform a solve must find the same
levels -- value to 10 x prec relative, multiplicity, root count -- and the same decline and
shortfall, with diagnostic counts and residual-scale floats not compared. Re-record on the
recording machine, never on CI.
"""
import json
import pathlib
import platform
import sys

import numpy as np
import pytest

from chladni.mps.objective import Objective
from chladni.mps.search import solve

import rect_suite as rs

GOLDEN = pathlib.Path(__file__).parent / "data" / "search_golden.json"
PRECS = (1e-8, 1e-10)
FAST = ("A:isolated", "B:d=5.9e-05", "C:m=3,s=9.5e-02", "D:mult=2", "D:mult=3")
PLATFORM = f"{sys.platform}-{platform.machine()}"


def cases():
    out = []
    for fam in rs.all_families(0).values():
        out.extend(fam)
    return out


def run_case(case, prec):
    obj = Objective(case.tensions, case.a, case.b, prec=prec, variable="lam")
    res = solve(obj, prec=prec)
    d = res.diagnostics
    return dict(
        levels=[dict(lam=float(L.lam), mult=int(L.mult), sigma=float(L.sigma),
                     delta_res=_f(L.delta_res), n_roots=int(L.n_roots),
                     count_uncertain=bool(L.count_uncertain),
                     count_decided_by=str(L.count_decided_by),
                     bracket=[float(t) for t in L.bracket]) for L in res.levels],
        declined=bool(d.declined), decline_reason=str(d.decline_reason),
        n_brackets=int(d.n_brackets), n_rejected=int(d.n_rejected), n_merged=int(d.n_merged),
        n_uncertain=int(d.n_uncertain), search_evals=int(d.search_evals),
        refine_evals=int(d.refine_evals), multiplicity_evals=int(d.multiplicity_evals),
        probe_evals=int(d.probe_evals), side_evals=int(d.side_evals),
        evaluations=int(d.evaluations), n_evals=int(obj.n_evals),
        unmet=len(d.unmet), attained_prec=_f(d.attained_prec))


def _f(x):
    x = float(x)
    return None if not np.isfinite(x) else x


def _compare(got, want, path=""):
    if isinstance(want, float) and isinstance(got, (float, int)):
        assert abs(got - want) <= 1e-10 * max(1.0, abs(want)), (path, got, want)
    elif isinstance(want, list):
        assert isinstance(got, list) and len(got) == len(want), (path, len(got), len(want))
        for i, (g, w) in enumerate(zip(got, want)):
            _compare(g, w, f"{path}[{i}]")
    elif isinstance(want, dict):
        assert set(want) <= set(got), (path, set(want) - set(got))
        for key in want:
            _compare(got[key], want[key], f"{path}.{key}")
    else:
        assert got == want, (path, got, want)


def _golden():
    return json.loads(GOLDEN.read_text())


def _check(got, want, label, prec, recorded_on):
    if PLATFORM == recorded_on:
        _compare(got, want, label)
        return
    for key in ("declined", "decline_reason", "unmet"):
        assert got[key] == want[key], (f"{label}.{key}", got[key], want[key])
    gl, wl = got["levels"], want["levels"]
    assert len(gl) == len(wl), (f"{label}.levels", len(gl), len(wl))
    for i, (g, w) in enumerate(zip(gl, wl)):
        path = f"{label}.levels[{i}]"
        assert abs(g["lam"] - w["lam"]) <= 10 * prec * abs(w["lam"]), (path, g["lam"], w["lam"])
        assert (g["mult"], g["n_roots"]) == (w["mult"], w["n_roots"]), (path, g, w)


@pytest.mark.parametrize("label", FAST)
@pytest.mark.parametrize("prec", PRECS)
def test_solve_reproduces_its_golden_trace(label, prec):
    case = next(c for c in cases() if c.label == label)
    golden = _golden()
    _check(run_case(case, prec), golden[f"{label}@{prec:g}"], label, prec,
           golden["_meta"]["recorded_on"])


@pytest.mark.slow
def test_solve_reproduces_its_golden_trace_on_every_case():
    golden = _golden()
    for case in cases():
        for prec in PRECS:
            _check(run_case(case, prec), golden[f"{case.label}@{prec:g}"], case.label, prec,
                   golden["_meta"]["recorded_on"])


if __name__ == "__main__" and "--record" in sys.argv:
    out = {"_meta": {"recorded_on": PLATFORM}}
    for case in cases():
        for prec in PRECS:
            out[f"{case.label}@{prec:g}"] = run_case(case, prec)
    GOLDEN.write_text(json.dumps(out, indent=1))
    solves = [v for k, v in out.items() if k != "_meta"]
    n_lev = sum(len(v["levels"]) for v in solves)
    print(f"recorded {len(solves)} solves, {n_lev} levels, "
          f"{sum(v['evaluations'] for v in solves)} evaluations on {PLATFORM}")
