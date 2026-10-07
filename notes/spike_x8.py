"""A simple eigenvalue returned with multiplicity 8, unflagged, on douse's spike triangle. (2026-10-07)

Found generating references (`benchmarks/douse/make_references.py`): T1.3, the 11.2 / 16.3 /
152.6 degree triangle at perimeter 4, run "B" (Fourier-Bessel at every corner plus an offset
source layer, 127 columns), window [124.33, 966.95], `prec = 1e-12`. FEM and two other runs
say every level there is simple.

Mechanism (traced 2026-10-07): the reach rule (`multiplicity(rule="reach")`) estimates branch
`j`'s distance as `sigma_j / c` and counts it when that falls inside the emitted bracket. It was
handed `c = 47.7` from the evaluation cache, against a TRUE wall slope of 3.40e-3 measured
directly (`sigma_1(x + d) / d` for d = 1e-3 .. 3), 14000x too steep -- a secant read inside the
floor (`sigma_1` floors at 1.1e-8 here, so readings 1e-9 apart give slopes ~13-47). With it,
`sigma_2 .. sigma_8` = 0.22 .. 0.83, which are O(1) and nowhere near a well, read as distances
0.005 .. 0.017 inside a locator bracket 4.3 wide, and all eight branches are "counted"
(`MAX_BRANCHES = 8`). Nothing flags it, and the Moler-Payne bound for 8 vectors is refused
(`inf`), so `bound` falls back to `delta_res` = 6e-10.

The sibling of the 2026-09-22 `_slope_from_cache` floor defect (`TODO.md` section 4, the
"phantom" item), which was fixed with a guard. The guard runs here and is defeated (traced
2026-10-07): the eight cached samples within 5e-9 of the root read 1.10e-8 .. 2.37e-8,
non-monotone -- the floor is NOISY by ~2x, not flat -- so the two at 5e-10 clear the guard's
flat-bottom margin (`_SLOPE_FLOOR_MARGIN * floor` = 2.2e-8), the floor-aware estimate reads 36.3,
and the guard fires only when the blind 47.7 exceeds it by 100x. Not run
past this one case. Basis A on the same shape gives x8 at three levels (807, 887, 916), so it is
not one basis's accident.

    .venv/bin/python notes/spike_x8.py        # ~15 s

Measured 2026-10-07 at c1c3f53: `807.0076084482 x8  bound 6.1e-10  mp inf  uncertain False`.
"""
import json
import os
import sys
import warnings

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
import numpy as np  # noqa: E402

sys.path.insert(0, "benchmarks/douse")
import make_references as M  # noqa: E402

warnings.simplefilter("ignore")
case = {c["id"]: c for c in M.load_cases()}["T1.3"]
out = M.chladni_solve(case["vertices"], "B", 124.32878268649651, 966.949607249548)
for L in out["levels"]:
    print(f"{L['lam']:.10f} x{L['mult']}  bound {L['bound']:.1e}  mp {L['mp_bound']:.1e}  "
          f"uncertain {L['uncertain']}")
print("levels with mult > 1:", [(round(L["lam"], 6), L["mult"]) for L in out["levels"]
                                if L["mult"] > 1])
