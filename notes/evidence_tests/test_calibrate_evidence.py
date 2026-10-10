"""Design evidence moved out of `tests/test_calibrate.py` by the 2026-10-09 prune.

`notes/test_inventory_2026-10-09.md`, decision 6: the measurements behind
`chladni.mps.calibrate.KAPPA` -- the wall slope `C_k = dsigma/dk` scales with the LENGTH scale
`A/P`, not with area, and kappa re-measured in `chladni` (2.591) agrees with the study's 2.38
within one standard deviation. TODO keeps the number; this file keeps the measurement. Moved
verbatim, with the helpers they need. Run on demand from the repository root:

    .venv/bin/python -m pytest -m '' notes/evidence_tests/test_calibrate_evidence.py
"""
from __future__ import annotations

import numpy as np
import pytest

from chladni.geometry import disk, eq_tri, rect
from chladni.helmholtz import PlaneWave
from chladni.mps import MPSPencil
from chladni.mps import calibrate as C
from chladni.reference import disk_eig, eq_tri_eig, rect_eig


def pencil(d, lam_hi, pad=1.3, seed=0):
    """A THRESHOLD pencil, explicitly. Copied from `tests/test_calibrate.py`, which says why."""
    nw = int(pad * PlaneWave.n_useful(np.sqrt(lam_hi) * d.diameter))
    return MPSPencil.from_basis(
        PlaneWave.uniform(nw, radius=d.diameter), d.bdry_quad(max(8 * nw, 900)),
        d.interior_pts(max(3 * nw, 450), rng=np.random.default_rng(seed)), reg="svd")


def wall_slope(p, lam_star):
    """`C_k = dsigma/dk` on the straight flank, both sides."""
    k0 = np.sqrt(lam_star)
    return float(np.mean([p.tension((k0 + s * dk) ** 2) / dk
                          for dk in (3e-3, 1e-2, 3e-2) for s in (1, -1)]))


# --- the criterion scales with A/P, not with area ---------------------------------------------

@pytest.mark.slow
def test_the_wall_slope_scales_with_the_LENGTH_scale_not_the_area():
    """The test that separates `A/P` from area: holding perimeter fixed makes the two
    proportional, so only SCALING distinguishes them. `sigma` is dimensionless and `k` is an
    inverse length, so `C_k = dsigma/dk` is a length -- and area is not."""
    rows = []
    for s in (0.5, 1.0, 2.0):
        d = rect(3 * s, 1 * s)
        lam = rect_eig(1, 1, 3 * s, 1 * s)
        rows.append((d.area / d.perimeter, abs(wall_slope(pencil(d, lam * 3), lam))))

    ratios = [c / ap for ap, c in rows]
    assert ratios[0] == pytest.approx(ratios[1], rel=1e-3)
    assert ratios[1] == pytest.approx(ratios[2], rel=1e-3)
    slopes = [c for _, c in rows]
    assert slopes[1] / slopes[0] == pytest.approx(2.0, rel=1e-3), "proportional to scale"
    assert slopes[2] / slopes[1] == pytest.approx(2.0, rel=1e-3)
    # C_k/area over the same three would read 2.46, 1.23, 0.61 -- a factor of 4.
    areas = [rect(3 * s, s).area for s in (0.5, 1.0, 2.0)]
    per_area = [c / a for (_, c), a in zip(rows, areas)]
    assert per_area[0] / per_area[2] == pytest.approx(4.0, rel=0.01), "area does NOT work"


@pytest.mark.slow
def test_kappa_re_measured_in_chladni_agrees_with_the_study():
    """The originating note expected this NOT to port -- "kappa = 2.38 is specific to lappy's
    normalization of sigma; expect a different constant in chladni". The two agree within one
    standard deviation. Recorded because the caution was reasonable, and because the number
    would have been silently wrong by a factor had it not been checked.

    Only domains whose plane-wave basis is ADEQUATE are included. On the L-shape plane waves
    leave the tension at 2.8e-01, so the "wall slope" there is the plateau divided by the step
    and reads 44 rather than ~1 -- a basis failure, not a scaling failure, and it is excluded by
    a floor check rather than allowed to corrupt the fit.
    """
    cases = [(rect(1, 1), rect_eig(1, 1, 1, 1)), (rect(2, 1), rect_eig(1, 1, 2, 1)),
             (rect(3, 1), rect_eig(1, 1, 3, 1)), (rect(5, 1), rect_eig(1, 1, 5, 1)),
             (eq_tri(1.0), eq_tri_eig(1, 1, 1.0)), (disk(1.0), disk_eig(0, 1, 1.0))]
    ratios = []
    for d, lam in cases:
        p = pencil(d, lam * 3)
        assert p.tension(lam) < 1e-6, f"basis inadequate at {lam}: {p.tension(lam):.1e}"
        ratios.append(abs(wall_slope(p, lam)) / (d.area / d.perimeter))

    mean = float(np.mean(ratios))
    assert mean == pytest.approx(2.591, rel=0.05), f"kappa_chladni = {mean:.3f}"
    assert abs(mean - C.KAPPA_LAPPY) < np.std(ratios) + 0.231, "within one sd of the study"
    assert C.KAPPA >= max(ratios), "KAPPA is the conservative end, so the check errs to flagging"
