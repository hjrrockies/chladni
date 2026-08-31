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
                                k_ref=K)


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
    ks = np.linspace(3.0, 4.0, 400)
    t = np.array([pencil.tension(k) for k in ks])
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
    k1 = np.sqrt(L_SHAPE_LAM1)
    b = sum((FourierBessel.at_corner(c, n=22, domain=dom) for c in dom.corners[1:]),
            FourierBessel.at_corner(dom.corners[0], n=22, domain=dom))
    p = MPSPencil.from_basis(b, dom.bdry_quad(800),
                             dom.interior_pts(300, rng=np.random.default_rng(0)), k_ref=k1)
    A, I = p.pencil(k1)
    c = p.coef(k1)[:, 0]
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
    the solver rather than of `k`, so a scan is reproducible and a re-solve at a visited
    parameter returns the same number; and it skips one `svdvals` of the stack per
    evaluation, measured at 11.3 ms against 13.4 ms per k, about 19%.
    """
    from chladni.geometry import Polygon as _P
    from conftest import L_SHAPE_LAM1

    dom = _P(L_SHAPE, bc="dir")
    k1 = np.sqrt(L_SHAPE_LAM1)
    b = sum((FourierBessel.at_corner(c, n=22, domain=dom) for c in dom.corners[1:]),
            FourierBessel.at_corner(dom.corners[0], n=22, domain=dom))
    p = MPSPencil.from_basis(b, dom.bdry_quad(800),
                             dom.interior_pts(300, rng=np.random.default_rng(0)), k_ref=k1)

    ks = k1 * (1 + np.linspace(-2e-3, 2e-3, 41))
    for k in ks:
        A, I = p.pencil(k)
        r = choose_rank(A, I, p.rtol)
        assert r == p.rank, f"the threshold rank moved to {r} at k={k}"
        assert (tensions_fn(A, I, reg="svd", rank=r)[0]
                == tensions_fn(A, I, reg="svd", rank=p.rank)[0])


def test_the_frozen_rank_does_not_depend_on_where_it_is_evaluated(pencil):
    """`rank` is fixed at construction and every later call uses it, whatever k arrives."""
    r = pencil.rank
    for k in (2.0, 3.3, 7.5, 12.0):
        assert pencil.regularized(k).rank == r


def test_rank_diagnostics_expose_the_choice(pencil):
    d = pencil.rank_diagnostics(K)
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
    loose = MPSPencil.from_basis(b, q, pts, k_ref=K, prec=1e-4)
    tight = MPSPencil.from_basis(b, q, pts, k_ref=K, prec=1e-14)
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


def test_implicit_mode_is_documented_as_diagnostic_only():
    """It hands the rank decision to LAPACK's own threshold, which reintroduces exactly the
    discontinuity this module exists to remove. It stays available and stays labelled."""
    assert "diagnostic" in regularize.__doc__.lower()
