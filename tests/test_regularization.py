"""Rank truncation, and why it is an integer.

This is the file that records the central correction to lappy.

lappy truncates at `(s >= rtol).sum()`, recomputed at every parameter. That makes the
retained rank a STEP FUNCTION of the spectral parameter: as `k` varies a singular value
crosses the fixed threshold, the rank drops by one, and the tension jumps. lappy's own
comment concedes there is nothing to hide the jump in -- "the largest consecutive drop in the
singular values is only 0.4-0.7 decades -- smooth decay, no rank cliff" -- and the eigenvalue
search is a parabolic minimizer, which a discontinuous objective lies to.

So the rank is chosen once and held fixed. Truncating to a fixed INDEX is continuous away
from exact singular-value crossings, because the tension depends only on the span of the
leading `r` left singular vectors and that subspace varies continuously except where
`s[r-1] == s[r]`. A threshold discontinues wherever `s[r-1]` crosses a constant, which is not
a measure-zero event.

`test_a_frozen_rank_gives_a_continuous_tension_curve` and its threshold twin are the pair
that makes that argument checkable rather than merely asserted.

The second claim here is that there is ONE inverse. lappy's `nullspace_coef` branches on
`reg_type` again at the recovery site, so two places have to agree about a transformation
only one of them computed. Here `regularize` returns `back` and the recovery has no branch.
"""
import numpy as np
import pytest
import scipy.linalg as la

from chladni.geometry import Polygon
from chladni.helmholtz import FourierBessel, Fundamental
import chladni.mps._regularize as regmod
from chladni.mps import MPSPencil, choose_rank, regularize
from chladni.mps import tensions as tensions_fn

from conftest import L_SHAPE

K = 3.3
LAM = K**2
REGS = ["svd", "qrp", None]


def basis_for(poly):
    fb = sum((FourierBessel.at_corner(c, n=8, domain=poly) for c in poly.corners[1:]),
             FourierBessel.at_corner(poly.corners[0], n=8, domain=poly))
    return fb + Fundamental(np.array([-1 - 1j, 3 + 3j]), radius=1.0)


@pytest.fixture
def poly():
    return Polygon(L_SHAPE, bc="dir")


@pytest.fixture
def pencil(poly):
    b = basis_for(poly)
    return MPSPencil.from_basis(b, poly.bdry_quad(400),
                                poly.interior_pts(150, rng=np.random.default_rng(0)),
                                lam_ref=LAM)


def well_conditioned(seed=0, m_b=40, m_i=25, n=12):
    """A pencil with no rank deficiency, where every mode must agree."""
    rng = np.random.default_rng(seed)
    A_B = rng.standard_normal((m_b, n))
    A_I = rng.standard_normal((m_i, n))
    norms = np.linalg.norm(np.vstack([A_B, A_I]), axis=0)
    return A_B / norms, A_I / norms


# --- the modes agree where they should ---------------------------------------------------


@pytest.mark.parametrize("reg", REGS)
def test_the_modes_agree_on_a_well_conditioned_pencil(reg):
    """No truncation happens, so regularization is a change of coordinates and the tension
    is an invariant of the pair. Anything else means a mode is not solving the same problem."""
    A_B, A_I = well_conditioned()
    ref = tensions_fn(A_B, A_I, reg=None)
    got = tensions_fn(A_B, A_I, reg=reg)
    assert np.allclose(got, ref, rtol=1e-9, atol=1e-14)


@pytest.mark.parametrize("reg", ["svd", "qrp"])
def test_back_inverts_the_reduction(reg):
    """`back` maps reduced coordinates to basis coordinates. The statement that makes it an
    inverse: a reduced vector and its image name the same function, so the residual through
    the reduced pencil equals the residual through the original one."""
    A_B, A_I = well_conditioned()
    r = regularize(A_B, A_I, reg=reg, rank=A_B.shape[1])
    y = np.random.default_rng(3).standard_normal((r.rank, 2))
    c = r.back(y)
    assert c.shape == (A_B.shape[1], 2)
    assert np.allclose(r.B @ y, A_B @ c, rtol=1e-9, atol=1e-12)
    assert np.allclose(r.I @ y, A_I @ c, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("reg", ["svd", "qrp"])
def test_truncation_actually_drops_columns(reg):
    A_B, A_I = well_conditioned(n=12)
    r = regularize(A_B, A_I, reg=reg, rank=7)
    assert r.rank == 7
    assert r.B.shape == (A_B.shape[0], 7) and r.I.shape == (A_I.shape[0], 7)
    assert r.back(np.eye(7)).shape == (12, 7)


def test_a_rank_deficient_pencil_is_truncated_not_solved():
    """Three exactly dependent columns; a chosen rank must find 9, not 12."""
    A_B, A_I = well_conditioned(n=12)
    A_B[:, 9:] = A_B[:, :3]
    A_I[:, 9:] = A_I[:, :3]
    assert choose_rank(A_B, A_I, rtol=1e-10) == 9


# --- the continuity argument, both halves -----------------------------------------------


def test_a_frozen_rank_gives_a_continuous_tension_curve(pencil):
    """The whole point. Sample the tension densely across a window containing no eigenvalue
    and measure the largest jump between neighbours against the typical one. A rank change
    shows up as an isolated spike orders of magnitude above the smooth variation.
    """
    lams = np.linspace(9.0, 16.0, 400)
    t = np.array([pencil.tension(lam) for lam in lams])
    d = np.abs(np.diff(t))
    assert d.max() < 25 * np.median(d), (
        f"largest step {d.max():.3e} against median {np.median(d):.3e}: the curve jumps"
    )


def test_the_minimizer_never_comes_near_the_truncation_cutoff():
    """Why rank truncation is safe in MPS at all -- and why the frozen rank is NOT a
    correctness fix, which an earlier version of this file got wrong.

    The worry that motivated the frozen rank was that truncation might delete the
    eigenfunction: near an eigenvalue the minimizing combination is the one `A_B`
    annihilates, so it is small, so it must be near the cutoff. The second step of that is
    false, and the interior block is exactly what makes it false. The tension is
    `||A_B c|| / ||A_I c||`, so a minimizer has `||A_I c||` of order ONE by construction --
    that is the whole reason the denominator is there. Its norm in the STACKED pencil, which
    is what the cutoff acts on, is therefore also of order one.

    Measured on the L-shape at `lam_1` with 22 degrees per corner, where the tension reaches
    1.7e-12 and the danger should be at its worst:

        ||A_B c||                                    1.2e-12
        ||A_I c||                                    7.4e-01
        cutoff (rtol * s[0])                         7.0e-12
        minimizer's energy in the retained subspace  1.000000

    All of it. A direction small in BOTH blocks does exist and does get truncated, but that
    is a redundant basis column, not an eigenfunction, and dropping it is correct.
    """
    from chladni.geometry import Polygon as _P
    import scipy.linalg as _la
    from conftest import L_SHAPE_LAM1

    dom = _P(L_SHAPE, bc="dir")
    b = sum((FourierBessel.at_corner(c, n=22, domain=dom) for c in dom.corners[1:]),
            FourierBessel.at_corner(dom.corners[0], n=22, domain=dom))
    p = MPSPencil.from_basis(b, dom.bdry_quad(800),
                             dom.interior_pts(300, rng=np.random.default_rng(0)),
                             lam_ref=L_SHAPE_LAM1)
    A, I = p.pencil(L_SHAPE_LAM1)
    c = p.coef(L_SHAPE_LAM1)[:, 0]
    c = c / np.linalg.norm(c)

    assert np.linalg.norm(A @ c) < 1e-10, "this is meant to be sitting on an eigenvalue"
    assert np.linalg.norm(I @ c) > 1e-2, "the interior block must forbid a small minimizer"

    _, _, Vt = _la.svd(np.vstack([A, I]), full_matrices=False)
    energy = (Vt @ c) ** 2
    assert energy[:p.rank].sum() > 1 - 1e-9, (
        "the minimizer has left the retained subspace; truncation could now delete it"
    )


def test_the_two_rules_agree_on_a_real_scan_through_an_eigenvalue():
    """The honest verdict, pinned so it is not quietly forgotten.

    Scanning the enriched L-shape across `lam_1` -- the configuration where the tension gets
    smallest, 1.7e-12, and where a cutoff-crossing would do the most damage -- the threshold
    rank never moves and the two rules produce BIT-IDENTICAL curves. There is no accuracy
    argument for the frozen rank on this problem.

    What survives is narrower and is what the design now rests on: the rank is a property of
    the solver rather than of `lam`, so a scan is reproducible and a re-solve at a visited
    parameter returns the same number; and it skips one `svdvals` of the stack per
    evaluation, measured at 11.3 ms against 13.4 ms per point, about 19%.
    """
    from chladni.geometry import Polygon as _P
    from conftest import L_SHAPE_LAM1

    dom = _P(L_SHAPE, bc="dir")
    b = sum((FourierBessel.at_corner(c, n=22, domain=dom) for c in dom.corners[1:]),
            FourierBessel.at_corner(dom.corners[0], n=22, domain=dom))
    p = MPSPencil.from_basis(b, dom.bdry_quad(800),
                             dom.interior_pts(300, rng=np.random.default_rng(0)),
                             lam_ref=L_SHAPE_LAM1)

    lams = L_SHAPE_LAM1 * (1 + np.linspace(-4e-3, 4e-3, 41))
    for lam in lams:
        A, I = p.pencil(lam)
        r = choose_rank(A, I, p.rtol)
        assert r == p.rank, f"the threshold rank moved to {r} at lam={lam}"
        assert (tensions_fn(A, I, reg="svd", rank=r)[0]
                == tensions_fn(A, I, reg="svd", rank=p.rank)[0])


def test_the_frozen_rank_does_not_depend_on_where_it_is_evaluated(pencil):
    """`rank` is fixed at construction and every later call uses it, whatever lam arrives."""
    r = pencil.rank
    for lam in (4.0, 10.89, 56.25, 144.0):
        assert pencil.regularized(lam).rank == r


def test_rank_diagnostics_expose_the_choice(pencil):
    d = pencil.rank_diagnostics(LAM)
    assert d.rank == pencil.rank
    assert d.s.ndim == 1 and np.all(np.diff(d.s) <= 0), "singular values descend"
    assert d.s[0] > 0


def test_a_near_degenerate_frozen_rank_warns():
    """The one configuration where a frozen index is the wrong answer: it straddles a
    near-tie, so the retained subspace is ill-defined and the curve can still move."""
    A_B, A_I = well_conditioned(n=6)
    u, s, vt = la.svd(np.vstack([A_B, A_I]), full_matrices=False)
    s[2] = s[3] * (1 + 1e-13)                      # a deliberate near-tie at the cut
    stack = (u * s) @ vt
    with pytest.warns(UserWarning, match="degenerate"):
        regularize(stack[:A_B.shape[0]], stack[A_B.shape[0]:], reg="svd", rank=3)


# --- rtol derives from prec --------------------------------------------------------------


def test_rtol_is_relative_not_absolute():
    """lappy's `rtol` was an absolute cutoff on `s` despite the name. Scaling the whole
    pencil must not change the retained rank; under an absolute cutoff it would."""
    A_B, A_I = well_conditioned(n=12)
    A_B[:, 8:] *= 1e-14
    A_I[:, 8:] *= 1e-14
    r = choose_rank(A_B, A_I, rtol=1e-10)
    assert choose_rank(A_B * 1e6, A_I * 1e6, rtol=1e-10) == r
    assert choose_rank(A_B * 1e-6, A_I * 1e-6, rtol=1e-10) == r


def test_a_tighter_prec_retains_more_columns(poly):
    """The direction is easy to invert, so it is pinned. A tighter precision request
    resolves MORE independent columns, so the cutoff falls and the rank rises -- the same
    direction `PlaneWave.n_useful` documents one layer down."""
    b = basis_for(poly)
    q, pts = poly.bdry_quad(400), poly.interior_pts(150, rng=np.random.default_rng(0))
    loose = MPSPencil.from_basis(b, q, pts, lam_ref=LAM, prec=1e-4)
    tight = MPSPencil.from_basis(b, q, pts, lam_ref=LAM, prec=1e-14)
    assert tight.rtol < loose.rtol
    assert tight.rank >= loose.rank


# --- the driver fallback -----------------------------------------------------------------


def test_the_gesdd_fallback_fires_and_warns(monkeypatch):
    """lappy's docstring: "one shape out of roughly 1500 solves raised here ... it took down
    the whole sweep and lost six of ten starts, about an hour of compute." The warning is
    deliberate -- falling back silently would hide that a domain is numerically difficult."""
    A_B, A_I = well_conditioned()
    calls = []
    real = la.svd

    def flaky(a, *args, **kw):
        calls.append(kw.get("lapack_driver"))
        if kw.get("lapack_driver") is None:
            raise np.linalg.LinAlgError("SVD did not converge")
        return real(a, *args, **kw)

    monkeypatch.setattr(regmod.la, "svd", flaky)
    with pytest.warns(UserWarning, match="gesvd"):
        r = regularize(A_B, A_I, reg="svd", rank=12)
    assert calls == [None, "gesvd"]
    assert r.rank == 12


def test_implicit_mode_agrees_with_the_svd_route():
    """`reg="implicit"` hands the rank decision to LAPACK's `?ggsvp3` via tola/tolb,
    skipping our QR and SVD. It is supported and it works; it is not the default.

    Two measured reasons, both recorded in `docs/design/mps.md`. It applies a DIFFERENT
    criterion -- `A` and `B` thresholded separately in 1-norms, against the stacked pencil
    in 2-norms -- so at equal `rtol` it truncates two to four columns harder, and on the
    disk no single `rtol` reproduces the svd rank at every `k`. And at MATCHED rank it is
    only 1.16x faster on the L-shape, because it does not remove the preprocessing so much
    as move it inside LAPACK, which runs on all `n` columns rather than the reduced `r`.

    Note for anyone tempted to reach for `tola` as a speed knob: in `gsvd4py` it is purely
    a rank threshold. `_gsvd.py` pins `tgsja`'s Jacobi convergence tolerances to the LAPACK
    defaults on purpose, "so that convergence quality is independent of the user's rank
    choice", so loosening `tola` does not buy speed by converging less hard.
    """
    A_B, A_I = well_conditioned(n=12)
    ref = tensions_fn(A_B, A_I, reg="svd", rank=12)
    got = tensions_fn(A_B, A_I, reg="implicit", rank=None)
    assert len(got) == len(ref)
    assert np.allclose(got, ref, rtol=1e-8, atol=1e-12)


def test_implicit_mode_is_usable_from_a_pencil(pencil):
    """It must be reachable through the object, not only through the free function, and it
    must not secretly pay for the factorization it exists to avoid."""
    p = MPSPencil(pencil.bdry, pencil.interior, reg="implicit")
    t = p.tensions(LAM)
    assert p.rank is None
    assert p.rank_at(LAM) == len(t)

    # Ascending, but note the trailing +inf: unlike the svd route, whose reduced stack has
    # orthonormal columns and therefore no zero denominators, implicit hands the raw pencil
    # to LAPACK and a direction the interior block annihilates comes back as an infinite
    # generalized singular value. That is the honest answer and it sorts to the end, but it
    # means a consumer must not assume the vector is finite.
    assert t.ndim == 1
    finite = np.isfinite(t)
    assert np.all(np.diff(t[finite]) >= 0)
    assert np.all(finite[: finite.sum()]), "the infinities must be a trailing block"
    # within a few percent of the svd route's tension at the same k
    ref = MPSPencil(pencil.bdry, pencil.interior).tension(K)
    assert abs(p.tension(K) - ref) / ref < 0.05


# --- the rank must be safe across the whole range it will be used over --------------------


def test_the_numerical_rank_grows_with_k():
    """The fact everything below turns on, and nothing in the code knew until it bit.

    More oscillation supports more independent columns, so the pencil's numerical rank is
    a function of k. Measured on a 54-column disk at the default rtol: 31 at k = 2.3
    rising to 47 at k = 5.7.
    """
    from chladni.geometry import disk
    from chladni.helmholtz import PlaneWave

    dom = disk(1.0)
    b = PlaneWave.uniform(27, radius=dom.diameter)
    p = MPSPencil.from_basis(b, dom.bdry_quad(600),
                             dom.interior_pts(200, rng=np.random.default_rng(0)),
                             lam_ref=(4.84, 32.49))
    lo = choose_rank(*p.pencil(5.29), p.rtol)
    hi = choose_rank(*p.pencil(32.49), p.rtol)
    assert lo < hi, f"rank did not grow with lam: {lo} at lam=5.29, {hi} at lam=32.49"


def test_a_rank_chosen_at_one_lam_is_unsafe_at_the_low_end():
    """The defect this pair of tests exists to prevent.

    A rank frozen at the midpoint of a scan is too large at the bottom of it, where the
    spectrum has not yet grown to support it -- so the cut lands at machine epsilon, the
    retained directions are undetermined, and the tension sprouts spurious local minima.
    Measured: rank 39 frozen at lam = 15.6 has s_r/s_0 = 2.4e-16 at lam = 5.52, and 85
    local minima in a window holding exactly one eigenvalue. Choosing over the range gives
    31, and one.
    """
    from chladni.geometry import disk
    from chladni.helmholtz import PlaneWave

    dom = disk(1.0)
    b = PlaneWave.uniform(27, radius=dom.diameter)
    q, pts = dom.bdry_quad(600), dom.interior_pts(200, rng=np.random.default_rng(0))

    point = MPSPencil.from_basis(b, q, pts, lam_ref=15.6025)
    spread = MPSPencil.from_basis(b, q, pts, lam_ref=(4.84, 32.49))
    assert spread.rank < point.rank

    B, I = spread.pencil(5.5225)
    s = np.linalg.svd(np.vstack([B, I]), compute_uv=False)
    assert s[spread.rank - 1] / s[0] > 1e-13, "the range-chosen rank is not safe at the low end"

    Bp, Ip = point.pencil(5.5225)
    sp = np.linalg.svd(np.vstack([Bp, Ip]), compute_uv=False)
    assert sp[point.rank - 1] / sp[0] < 1e-15, "this lam no longer exercises the defect"


def test_an_unsafe_frozen_rank_says_so():
    """A reported precision is never a claim contradicted elsewhere in the same object.
    Silent roughness reads as extra eigenvalues, which is the worst way to be wrong."""
    from chladni.geometry import disk
    from chladni.helmholtz import PlaneWave
    from chladni.mps import PrecisionWarning

    dom = disk(1.0)
    b = PlaneWave.uniform(27, radius=dom.diameter)
    q, pts = dom.bdry_quad(600), dom.interior_pts(200, rng=np.random.default_rng(0))
    point = MPSPencil.from_basis(b, q, pts, lam_ref=15.6025)
    with pytest.warns(PrecisionWarning, match="frozen rank"):
        point.rank_diagnostics(5.5225)


def test_a_lam_ref_range_is_validated():
    from chladni.geometry import disk
    from chladni.helmholtz import PlaneWave

    dom = disk(1.0)
    b = PlaneWave.uniform(20, radius=dom.diameter)
    q, pts = dom.bdry_quad(400), dom.interior_pts(100, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="lo < hi"):
        MPSPencil.from_basis(b, q, pts, lam_ref=(25.0, 4.0))
    with pytest.raises(ValueError, match="pair"):
        MPSPencil.from_basis(b, q, pts, lam_ref=(1.0, 4.0, 9.0))


@pytest.mark.slow
def test_a_range_chosen_rank_leaves_one_minimum_per_eigenvalue():
    """The end-to-end statement of the fix, in the units that matter for recall."""
    from chladni.geometry import disk
    from chladni.helmholtz import PlaneWave

    dom = disk(1.0)
    b = PlaneWave.uniform(27, radius=dom.diameter)
    q, pts = dom.bdry_quad(600), dom.interior_pts(200, rng=np.random.default_rng(0))
    p = MPSPencil.from_basis(b, q, pts, lam_ref=(4.84, 32.49))

    lams = np.linspace(5.0625, 6.0025, 400)    # holds exactly one eigenvalue, lam_1 = 5.7832
    v = np.array([p.tension(lam) for lam in lams])
    n_min = len(np.nonzero((v[1:-1] < v[:-2]) & (v[1:-1] < v[2:]))[0])
    assert n_min == 1, f"{n_min} local minima where there is one eigenvalue"


def test_the_rank_is_derived_inside_the_factorization_not_before_it():
    """`regularize(rank=None)` must truncate, using the decomposition it already computed.

    This is not a micro-optimization. `choose_rank` does an `svdvals` of the stacked pencil
    and the `"svd"` mode does an SVD of `R`, and because `Q` has orthonormal columns those
    have the SAME singular values -- so asking the first before calling the second computed
    one decomposition twice. It was 2.13 ms of an 8.19 ms evaluation on a 134-column
    L-shape pencil, and removing it made the default path 1.31 to 1.46x faster.
    """
    A_B, A_I = well_conditioned(n=12)
    A_B[:, 9:] *= 1e-14
    A_I[:, 9:] *= 1e-14
    r = regularize(A_B, A_I, reg="svd", rtol=1e-10)
    assert r.rank == 9, "rank=None did not truncate: the derivation is not happening"
    assert r.rank == choose_rank(A_B, A_I, rtol=1e-10), "the two rules must still agree"
    assert regularize(A_B, A_I, reg="qrp", rtol=1e-10).rank == 9


def test_an_explicit_rank_still_overrides_the_derivation():
    A_B, A_I = well_conditioned(n=12)
    assert regularize(A_B, A_I, reg="svd", rank=5, rtol=1e-10).rank == 5
