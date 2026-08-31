"""The surrogate must behave like the thing it stands in for.

A synthetic model is worth exactly what its validation is worth. Earlier in this project a
synthetic pencil appeared to show that threshold rank truncation deletes eigenfunctions. It
did -- but only because the construction placed the minimizer where the real problem never
places it, and the conclusion was false for MPS. The lesson was expensive enough to build
into this file: every structural claim the surrogate makes is checked here against a number
measured on a real pencil, and anything a search discovers on the surrogate is provisional
until it is confirmed on a real domain.

The measured references, all from `notes/` runs on real domains:

    cone model residual over the far field      1-3%
    cone slope c, within one domain             0.98 - 1.17 (rect), 1.56 - 1.70 (disk)
    two dips resolved down to a separation of   1.26e-06 in k (rect 1+d by 1)
    sigma_2's minimum sits at                   the midpoint, to ~1% of d/2
    exact degeneracy gives                      one dip, sigma_1 == sigma_2
    roughness, safe rank / unsafe rank          1e-10 / 1e-2 relative
"""
import numpy as np
import pytest

from surrogate import Surrogate, Well


def dips(v):
    return np.nonzero((v[1:-1] < v[:-2]) & (v[1:-1] < v[2:]))[0] + 1


# --- the local model ---------------------------------------------------------------------


@pytest.mark.parametrize("c", [0.45, 1.0, 1.7])
def test_the_far_field_is_a_cone_of_the_requested_slope(c):
    """`sigma ~ c*|delta|` away from the floor, which is what makes a Lipschitz bound
    meaningful. Slopes span the range measured across real domains."""
    s = Surrogate([Well(4.0, eps=1e-12, c=c)], n=8)
    d = np.array([1e-3, 3e-3, 1e-2])
    got = np.array([s.tension(4.0 + x) for x in d])
    assert np.allclose(got / d, c, rtol=1e-6)


def test_the_floor_is_reached_exactly():
    """The surrogate must not be blunter than the solver: real pencils reach 5.8e-16."""
    for eps in (1e-6, 1e-12, 1e-15):
        s = Surrogate([Well(4.0, eps=eps)], n=8)
        assert s.tension(4.0) == pytest.approx(eps, rel=1e-12)


def test_the_cone_model_fits_the_curve_it_generates():
    """Closing the loop: fitting `sqrt(eps^2 + c^2 d^2)` to the surrogate must recover the
    parameters, at a residual far below the 1-3% the same fit leaves on a real pencil."""
    s = Surrogate([Well(4.0, eps=1e-9, c=1.2)], n=8)
    d = np.linspace(1e-4, 1e-2, 50)
    got = np.array([s.tension(4.0 + x) for x in d])
    pred = np.sqrt(1e-9**2 + (1.2 * d) ** 2)
    # machine-limited: the absolute residual is ~3e-16 against a `got` as small as 1.2e-4
    assert np.max(np.abs(got - pred) / got) < 1e-11


# --- clusters, which is what the surrogate exists for ------------------------------------


@pytest.mark.parametrize("d", [4.2e-2, 4.2e-4, 4.2e-6])
def test_two_wells_give_two_dips_down_to_the_measured_resolution(d):
    """The rect(1+delta, 1) experiment resolved two dips down to a separation of 1.26e-06.
    The surrogate must not be easier than that, or a search tuned on it will be optimistic."""
    mid = 7.0
    s = Surrogate([Well(mid - d / 2, eps=1e-12), Well(mid + d / 2, eps=1e-12)], n=12)
    ks = np.linspace(mid - 4 * d, mid + 4 * d, 801)
    v = np.array([s.tensions(k)[0] for k in ks])
    assert len(dips(v)) == 2


def test_sigma_2_has_one_minimum_and_it_sits_at_the_midpoint():
    """The cluster detector of `lipschitz-cone-mps.md` section 3.4, and the one structural
    claim in that document this project has confirmed on real pencils: sigma_2's minimum is
    at the crossing point, so its offset from the nearest sigma_1 minimum measures d/2 --
    a separation that can be READ before the pair is resolved.

    Measured on rect(1+delta, 1): offset from the midpoint 2.9e-04 against d/2 = 2.1e-02,
    i.e. about 1% of it.
    """
    mid, d = 7.0, 4.2e-3
    s = Surrogate([Well(mid - d / 2, eps=1e-12), Well(mid + d / 2, eps=1e-12)], n=12)
    ks = np.linspace(mid - 4 * d, mid + 4 * d, 1601)
    S = np.array([s.tensions(k)[:2] for k in ks])

    assert len(dips(S[:, 0])) == 2
    assert len(dips(S[:, 1])) == 1
    k2 = ks[np.argmin(S[:, 1])]
    assert abs(k2 - mid) < 0.02 * (d / 2)

    # and the offset from sigma_2's minimum to the nearest sigma_1 minimum IS d/2
    k1 = ks[dips(S[:, 0])]
    assert np.min(np.abs(k1 - k2)) == pytest.approx(d / 2, rel=0.02)


def test_an_exact_degeneracy_gives_one_dip_and_coincident_branches():
    """What the square does at delta = 0: sigma_1 and sigma_2 are equal, not merely close,
    and there is one minimum rather than two. That is the case a multiplicity rule must
    distinguish from a tight pair, and the surrogate can produce it exactly."""
    s = Surrogate([Well(7.0, eps=1e-12, mult=2)], n=12)
    ks = np.linspace(6.99, 7.01, 401)
    S = np.array([s.tensions(k)[:2] for k in ks])
    assert len(dips(S[:, 0])) == 1
    assert np.array_equal(S[:, 0], S[:, 1])


def test_multiplicity_three_drives_three_branches_down():
    s = Surrogate([Well(7.0, eps=1e-12, mult=3)], n=12)
    t = s.tensions(7.0)
    assert t[0] == t[1] == t[2] == pytest.approx(1e-12, rel=1e-12)
    assert t[3] > 1e-3


# --- avoided crossings -------------------------------------------------------------------


def test_a_coupling_opens_a_gap_of_the_size_it_advertises():
    """`coupling` is in sigma-squared units, so the observable gap is about `g/eps`. Pinned
    because the first implementation squared it again and produced a gap of 1e-18 where
    1e-06 was intended -- and drove sigma_1 to exactly zero."""
    eps = 1e-6
    for g in (1e-16, 1e-14, 1e-13):
        s = Surrogate([Well(3.0, eps=eps, coupling=g), Well(3.0, eps=eps)], n=8)
        t = s.tensions(3.0)
        assert (t[1] - t[0]) == pytest.approx(g / eps, rel=0.05)
        assert t[0] > 0.0


def test_the_coupling_ceiling_is_enforced():
    """Above `eps**2` the 2x2 block goes indefinite. An avoided crossing cannot open a gap
    wider than the floor it sits on -- which is also what the physics says."""
    with pytest.raises(ValueError, match="eps\\*\\*2"):
        Well(3.0, eps=1e-6, coupling=1e-10)


# --- the family is what it claims to be --------------------------------------------------


def test_the_analytic_curves_are_the_eigenvalues_of_the_stated_matrix():
    """`tensions` takes a closed form for accuracy, but it must agree with `matrix(k)`, or
    the docstring's claim to be a Hermitian PSD family is decoration."""
    s = Surrogate([Well(3.0, eps=1e-4, coupling=1e-9), Well(3.2, eps=1e-5)], n=10)
    for k in (2.8, 3.0, 3.1, 3.2):
        analytic = s.tensions(k)
        numeric = np.sort(np.sqrt(np.maximum(np.linalg.eigvalsh(s.matrix(k)), 0.0)))
        assert np.allclose(analytic, numeric, atol=1e-12)


def test_the_family_is_positive_semidefinite():
    s = Surrogate([Well(3.0, eps=1e-5, coupling=1e-11), Well(4.0, eps=1e-8, mult=2)], n=12)
    for k in np.linspace(2.5, 4.5, 21):
        assert np.linalg.eigvalsh(s.matrix(k)).min() > -1e-15


def test_evaluations_are_counted_and_deterministic():
    """The count is how a search gets scored; determinism is what the real solver has, and
    a surrogate that jittered would flatter any method that re-evaluates."""
    s = Surrogate([Well(3.0)], n=8, roughness=1e-6)
    a = s.tensions(3.5)
    n = s.calls
    assert np.array_equal(a, s.tensions(3.5))
    assert s.calls == n + 1


def test_roughness_is_a_deterministic_function_of_k():
    """Real roughness is not stochastic: the same k returns a bit-identical value. Measured
    at 1e-10 relative on a disk pencil at a safe rank, 1e-2 at one cut at machine epsilon."""
    s = Surrogate([Well(3.0, eps=1e-12)], n=8, roughness=1e-3)
    smooth = Surrogate([Well(3.0, eps=1e-12)], n=8, roughness=0.0)
    ks = np.linspace(3.2, 3.3, 200)
    a = np.array([s.tension(k) for k in ks])
    b = np.array([smooth.tension(k) for k in ks])
    assert np.array_equal(a, np.array([s.tension(k) for k in ks]))
    assert 1e-4 < np.max(np.abs(a - b) / b) < 1e-2
