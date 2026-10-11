"""Regularization: rank truncation, the penalty that replaces it, and why either is chosen.

This is the file that records the central correction to lappy.

lappy truncates at `(s >= rtol).sum()`, recomputed at every parameter. That makes the
retained rank a STEP FUNCTION of the spectral parameter: as `k` varies a singular value
crosses the fixed threshold, the rank drops by one, and the tension jumps. lappy's own
comment concedes there is nothing to hide the jump in -- "the largest consecutive drop in the
singular values is only 0.4-0.7 decades -- smooth decay, no rank cliff" -- and the eigenvalue
search is a parabolic minimizer, which a discontinuous objective lies to.

Truncating to a fixed INDEX (a frozen rank) is continuous away from exact singular-value
crossings, because the tension depends only on the span of the leading `r` left singular
vectors and that subspace varies continuously except where `s[r-1] == s[r]`. A threshold
discontinues wherever `s[r-1]` crosses a constant, which is not a measure-zero event -- and
the default IS a threshold, because a rank frozen over a wide window is wrong at one end of it.
`test_a_frozen_rank_gives_a_continuous_tension_curve` and its threshold twin are the pair
that makes that argument checkable rather than merely asserted.

The penalty mode, `"tikhonov_qr"`, is the third answer: no rank at all. Its sections below
check the algebra, check that no rank change survives -- including the one LAPACK makes
internally -- and check the cell of the 2026-09-21 CI refusal (`notes/session_2026-09-19.md`)
that a threshold flip caused. The literal `"tikhonov"` mode is still in `src/` but retired from
this suite (`notes/test_inventory_2026-10-09.md`, decision 4): it relocates the rank decision
into LAPACK's `?ggsvp3` rather than removing it.

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
PENALTY = ["tikhonov_qr"]   # the literal "tikhonov" is retired from the suite

#: `(reg, frozen)` pairs for the contracts that hold in every mode. A penalty has no rank to
#: freeze -- asking is refused -- so it joins in threshold form only.
MODE_RANKS = [(r, f) for r in REGS for f in (None, 30)] + [(r, None) for r in PENALTY]


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


@pytest.mark.parametrize("reg", REGS + PENALTY)
def test_the_modes_agree_on_a_well_conditioned_pencil(reg):
    """No truncation happens, so regularization is a change of coordinates and the tension
    is an invariant of the pair. Anything else means a mode is not solving the same problem.

    The penalty mode joins at the free function's default `rtol = eps`: `tau` is then about
    1e-15 against tensions of order one, too small to move anything here."""
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


# Both of these drive the `pencil` fixture far from its lam_ref of 10.89 -- to 16, and to 144
# -- where FourierBessel reports a digit less than the requested 1e-8. That is the point of
# the tests: the rank is frozen at construction and must hold wherever it is later evaluated.
# Accuracy at those lam is not the claim, and is tested in test_precision.py.
@pytest.mark.filterwarnings(
    "ignore:FourierBessel can deliver:chladni._core.PrecisionWarning")
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


@pytest.mark.filterwarnings(
    "ignore:FourierBessel can deliver:chladni._core.PrecisionWarning")
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


# --- the Accelerate dgeqrf workaround ------------------------------------------------------


# The first shape is tall, non-square and wider than `dgeqrt`'s 32-column block, with a ragged
# last block (32 + 32 + 6), so the multi-block path is what is compared. It used to be 548 x 274,
# which IS Accelerate's standalone `dgeqrf` heap-corruption reproducer (TODO section 4): the
# comparison against `la.qr` then called the faulty route to check the safe one, and failed about
# one default-tier run in ten. 120 x 70 also stays under reference LAPACK's 128-column crossover
# to blocked `dgeqrf`.
@pytest.mark.parametrize("shape", [(120, 70), (40, 12), (12, 40), (7, 1)])
def test_the_geqrt_route_is_an_economic_qr(monkeypatch, shape):
    """The route taken on Accelerate must be a QR of the same shape as `la.qr`'s, whatever
    LAPACK this test runs on -- so it is forced on here rather than detected."""
    monkeypatch.setattr(regmod, "_AVOID_GEQRF", True)
    a = np.random.default_rng(1).standard_normal(shape)
    Q, R = regmod._qr_economic(a)
    Q0, R0 = la.qr(a, mode="economic")
    assert Q.shape == Q0.shape and R.shape == R0.shape
    assert np.allclose(np.triu(R), R)
    assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1e-13)
    assert np.allclose(Q @ R, a, atol=1e-13)
    assert np.allclose(np.abs(np.diag(R)), np.abs(np.diag(R0)), rtol=1e-12)


def test_regularize_never_calls_geqrf_when_avoiding_it(monkeypatch):
    """The svd mode's QR goes through the helper, and agrees with the stock route."""
    A_B, A_I = well_conditioned()
    stock = tensions_fn(A_B, A_I, reg="svd", rank=12)
    monkeypatch.setattr(regmod, "_AVOID_GEQRF", True)

    def geqrf(*args, **kw):
        raise AssertionError("la.qr reached while avoiding Accelerate's dgeqrf")

    monkeypatch.setattr(regmod.la, "qr", geqrf)
    assert np.allclose(tensions_fn(A_B, A_I, reg="svd", rank=12), stock, rtol=1e-10, atol=1e-14)


@pytest.mark.slow
def test_the_170_pi2_rectangle_cluster_solves_without_killing_the_interpreter():
    """REGRESSION, in a subprocess because the failure is the interpreter dying.

    The rectangle suite's multiplicity-4 cluster at `170*pi^2`, first cell of its grid
    (`prec = 1e-6`, bands 0.3 and 0.5), exited 139/133 on every measured run on macOS 26.5
    with Accelerate: heap corruption from `dgeqrf` in `regularize`, surfacing wherever the heap
    was next touched -- sometimes only at shutdown. See `_regularize._qr_economic`. On other
    LAPACKs this is an ordinary solve and passes trivially.
    """
    import subprocess
    import sys
    from pathlib import Path

    code = (
        "import sys, warnings\n"
        f"sys.path.insert(0, {str(Path(__file__).parent)!r})\n"
        "warnings.simplefilter('ignore')\n"
        "import rect_suite as rs, bracket_score as bs\n"
        "from chladni.mps.search import solve\n"
        "for band in (0.3, 0.5):\n"
        "    (case,) = rs.split_degeneracy_cases(1e-6, bands=(band,), levels={170: 4})\n"
        "    res = solve(bs.objective_for(case, 1e-6), prec=1e-6)\n"
        "    print(band, [(L.lam, L.mult) for L in res.levels], flush=True)\n"
    )
    proc = subprocess.run([sys.executable, "-X", "faulthandler", "-c", code],
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, (
        f"exit {proc.returncode}\n--- stdout\n{proc.stdout}\n--- stderr\n{proc.stderr[-3000:]}")


def test_the_unregularized_route_is_usable_from_a_pencil(pencil):
    """`reg=None` must be reachable through the object, not only the free function, and it
    must agree with the svd route on the quantity that matters.

    This is what is left of the old `test_implicit_mode_*` pair. `reg="implicit"` handed the
    rank decision to LAPACK's `?ggsvp3` through `tola`/`tolb`; `gsvd4py` 0.3.0 removed those
    parameters, so the mode is gone. `None` is now the only route that puts the raw,
    un-pre-truncated pencil in front of the GSVD, which makes it the one that inherits the
    behaviour the implicit tests were pinning.
    """
    p = MPSPencil(pencil.bdry, pencil.interior, reg=None)
    t = p.tensions(LAM)
    assert p.rank is None
    assert p.rank_at(LAM) == len(t) < p.n_cols, "44 of 50 here: LAPACK's k + l, not n_cols"

    finite = np.isfinite(t)
    assert t.ndim == 1 and np.all(np.diff(t[finite]) >= 0)
    assert np.all(finite[: finite.sum()]), "the infinities must be a trailing block"

    ref = MPSPencil(pencil.bdry, pencil.interior).tension(LAM)
    assert abs(p.tension(LAM) - ref) / ref < 0.05


@pytest.mark.parametrize("reg, frozen", MODE_RANKS)
def test_rank_at_is_what_the_pencil_actually_yields(pencil, reg, frozen):
    """One contract for every mode: `len(tensions(lam)) == rank_at(lam)`, and asking `coef`
    for that many vectors is always a valid call.

    A pre-truncating mode gets this for free -- it hands the GSVD a reduced stack of full
    column rank, so LAPACK's `k + l` is the truncation index it already chose. `reg=None`
    does not, and used to report `n_cols` instead: 50 against a real 44 on this fixture, so
    `p.coef(lam, mult=p.rank_at(lam))` raised for `None` and worked for the others. That is
    the trap this pins shut.

    It has to hold for a FROZEN rank too. `None` truncates to the first `rank` columns with
    no regard to conditioning, so LAPACK can still find that stack deficient.
    """
    p = MPSPencil(pencil.bdry, pencil.interior, reg=reg, rank=frozen)
    r = p.rank_at(LAM)
    assert r == len(p.tensions(LAM))
    assert p.coef(LAM, mult=r).shape[1] == r


@pytest.mark.parametrize("reg, frozen", MODE_RANKS)
def test_coef_returns_basis_coordinates_except_where_it_says_otherwise(pencil, reg, frozen):
    """`coef` claims `(n_cols, mult)`, and that claim was untested and false in one cell of
    the mode/rank cross product. Five of six return `n_cols` rows; `reg=None` with a frozen
    rank returns `rank` rows, because it keeps the first `rank` columns and maps back with
    the identity instead of scattering into the full basis.

    Documented rather than changed, so this is the guard on the documentation.
    """
    p = MPSPencil(pencil.bdry, pencil.interior, reg=reg, rank=frozen)
    rows = p.coef(LAM, mult=2).shape[0]
    want = frozen if (reg is None and frozen is not None) else p.n_cols
    assert rows == want


def test_the_short_coef_vector_names_the_same_function_when_padded(pencil):
    """The exception above is a shape, not a different answer: zero-padding to `n_cols` gives
    a vector over the full basis with the same tension, so nothing is lost by the narrower
    return. It is emphatically NOT the prefix of the untruncated solve, which minimises over
    a larger space -- measured cosine 0.14 against it here.
    """
    p = MPSPencil(pencil.bdry, pencil.interior, reg=None, rank=30)
    c = p.coef(LAM)[:, 0]
    assert c.shape == (30,)
    B, I = p.pencil(LAM)
    padded = np.zeros(p.n_cols)
    padded[:30] = c
    got = np.linalg.norm(B @ padded) / np.linalg.norm(I @ padded)
    assert np.isclose(got, p.tension(LAM), rtol=1e-10)


# lam = 60 is far above the fixture's lam_ref of 10.89, where FourierBessel reports a digit
# less than the requested 1e-8. The claim here is about counting directions, not about
# accuracy at that lam.
@pytest.mark.filterwarnings(
    "ignore:FourierBessel can deliver:chladni._core.PrecisionWarning")
def test_rank_at_tracks_lam_for_a_non_truncating_mode(pencil):
    """`k + l` is LAPACK's own number and moves with `lam`, so `rank_at` must not be fixed
    at construction. Measured on the L-shape fixture in THRESHOLD mode, where `reg=None`
    keeps all 50 columns: 44, 48, 50 at lam = 10.89, 25, 60, against `n_cols` of 50
    throughout and `choose_rank` of 42, 45, 48 at this rtol (and 50, 50, 50 at eps). It
    agrees with neither, which is why nothing short of the GSVD can predict it.

    The exact values are LAPACK's own tolerance at work, so they are not pinned: what is
    asserted is the contract -- non-decreasing in `lam`, never above `n_cols` -- and that the
    count does move across the scan, which a six-column margin at the ends leaves no
    rounding-level perturbation able to undo.

    Threshold mode on purpose. Freezing a rank truncates to the first `rank` columns before
    LAPACK sees the stack, which caps `k + l` at that rank and hides the growth.
    """
    p = MPSPencil(pencil.bdry, pencil.interior, reg=None)
    assert p.rank is None and p.n_cols == 50
    got = [p.rank_at(lam) for lam in (10.89, 25.0, 60.0)]
    assert all(g <= p.n_cols for g in got), got
    assert got == sorted(got), f"the numerical rank must not fall as lam grows: {got}"
    assert got[0] < got[-1], f"rank_at did not move with lam: {got}"


def test_an_annihilated_direction_comes_back_as_an_infinite_tension():
    """The `s == 0` branch in `tensions_of`, which the implicit tests used to cover.

    A pre-truncating mode reduces to a stack with orthonormal columns, so the denominator
    never vanishes and this cannot arise. `reg=None` hands LAPACK the raw pencil, so an
    interior block with a null space yields directions of infinite generalized singular
    value. `+inf` is the honest answer -- the boundary residual is nonzero where the
    interior norm is zero -- and it sorts to the far end, but it means a consumer must not
    assume the vector is finite.
    """
    rng = np.random.default_rng(0)
    # five rows, twelve columns: a seven-dimensional null space in the interior block
    A_B, A_I = rng.standard_normal((20, 12)), rng.standard_normal((5, 12))
    t = tensions_fn(A_B, A_I, reg=None)
    finite = np.isfinite(t)
    assert (~finite).sum() == 7, f"expected seven infinite tensions, got {(~finite).sum()}"
    assert np.all(np.diff(t[finite]) >= 0)
    assert np.all(finite[: finite.sum()]), "the infinities must be a trailing block"


# --- the rank must be safe across the whole range it will be used over --------------------


# The tests below deliberately oversubscribe the basis: 27 plane-wave directions on a disk
# where about 16 are independent at the low end of the lam range. That is the POINT -- rank
# saturation is the phenomenon under study.
#
# These used to carry `filterwarnings` marks, because `PlaneWave.log_scale` warned on the
# COUNT and so fired on a setup that is correct under a threshold rank. It no longer does,
# and the marks are gone: a `PrecisionWarning` reaching this file now means the RANK is
# unsafe, which is the thing worth hearing. `test_an_unsafe_frozen_rank_says_so` keeps the
# positive `pytest.warns` and its silent twin keeps the negative.
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


def test_the_SAME_over_enriched_basis_is_safe_and_SILENT_under_a_threshold_rank():
    """The complement of the two tests above, and the reason the BASIS must not warn.

    27 directions on a disk is over-enriched by construction -- 54 columns where `n_useful`
    allows about 33 at the low end. Frozen at one lam that is the defect the pair above
    exists to catch. Left to a threshold rank it is the RECOMMENDED configuration, and the
    numbers say so: the cut tracks k (31 -> 47), lands BELOW the useful bound at every lam,
    and stays three decades clear of the noise floor.

    So the same basis is a defect or a good setup depending entirely on the rank policy --
    which the basis cannot see. `PlaneWave.log_scale` used to render that verdict anyway, on
    every evaluation, and it was wrong here every time.
    """
    import warnings

    from chladni.geometry import disk
    from chladni.helmholtz import PlaneWave
    from chladni.mps import PrecisionWarning

    dom = disk(1.0)
    b = PlaneWave.uniform(27, radius=dom.diameter)
    q, pts = dom.bdry_quad(600), dom.interior_pts(200, rng=np.random.default_rng(0))
    lams = (5.5225, 15.6025, 32.49)

    # The silence check runs FIRST, on a pencil whose cache is cold. `MPSPencil` memoizes the
    # equilibrated matrices, so a warmed pencil never re-enters the basis and would pass this
    # for the wrong reason -- measured: it passed while the guard was still installed.
    with warnings.catch_warnings():
        warnings.simplefilter("error", PrecisionWarning)
        cold = MPSPencil.from_basis(b, q, pts, reg="svd")
        for lam in lams:
            cold.tension(lam)
            cold.rank_diagnostics(lam)

    p = MPSPencil.from_basis(b, q, pts, reg="svd")   # no rank, no lam_ref: threshold rank
    assert p.rank is None
    assert p.rank_at(5.5225) < p.rank_at(32.49), "the cut must track k"

    for lam in lams:
        r = p.rank_at(lam)
        useful = b.n_useful_at(np.sqrt(lam))
        assert r < b.n_cols, f"nothing was truncated at lam={lam}"
        assert r <= useful, f"rank {r} exceeds the useful bound {useful:.0f} at lam={lam}"
        B, I = p.pencil(lam)
        s = np.linalg.svd(np.vstack([B, I]), compute_uv=False)
        assert s[r - 1] / s[0] > 1e-13, f"the cut reached the noise floor at lam={lam}"


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


# --- penalty modes: the algebra -----------------------------------------------------------
#
# `"tikhonov_qr"` hands the GSVD the row blocks of `Q` from an unpivoted QR of
# `[A_B; tau*I; A_I]`. `tau = rtol * ||[A_B; A_I]||_F`. The literal `"tikhonov"`, which hands it
# `([A_B; tau*I], A_I)`, is still in `src/` but retired from this suite; see the `_regularize`
# module docstring for why there were two.


@pytest.mark.parametrize("reg", PENALTY)
def test_back_names_the_same_function_under_a_penalty(reg):
    """The penalty analogue of `test_back_inverts_the_reduction`. `B` is the AUGMENTED
    numerator, so the identity is against `[A_B; tau*I]`, not `A_B`."""
    A_B, A_I = well_conditioned()
    r = regularize(A_B, A_I, reg=reg, rtol=1e-6)
    y = np.random.default_rng(3).standard_normal((r.rank, 2))
    c = r.back(y)
    numer = np.vstack([A_B, r.tau * np.eye(A_B.shape[1])])
    assert c.shape == (A_B.shape[1], 2)
    assert np.allclose(r.B @ y, numer @ c, rtol=1e-9, atol=1e-12)
    assert np.allclose(r.I @ y, A_I @ c, rtol=1e-9, atol=1e-12)
    assert np.allclose(r.back(y[:, 0]), c[:, 0]), "a flat vector must map the same way"


@pytest.mark.parametrize("reg", PENALTY)
def test_the_penalty_minimizer_attains_the_reported_tension(pencil, reg):
    """`coef` is the minimizer of `sqrt(||Bc||^2 + tau^2 ||c||^2) / ||Ic||`, so evaluating
    that quotient at it must give back the tension. Measured 5e-8 and 7e-8 relative here,
    which is the minimizer's own accuracy on a plateau point, not a defect of `back`."""
    p = MPSPencil(pencil.bdry, pencil.interior, reg=reg)
    c = p.coef(LAM)[:, 0]
    B, I = p.pencil(LAM)
    tau = p.regularized(LAM).tau
    got = np.hypot(np.linalg.norm(B @ c), tau * np.linalg.norm(c)) / np.linalg.norm(I @ c)
    assert got == pytest.approx(p.tension(LAM), rel=1e-6)


@pytest.mark.parametrize("reg", PENALTY)
def test_tau_is_relative_not_absolute(reg):
    """The penalty twin of `test_rtol_is_relative_not_absolute`: scaling the whole pencil
    scales `tau` with it and leaves every tension alone."""
    A_B, A_I = well_conditioned(n=12)
    A_B[:, 8:] *= 1e-10
    A_I[:, 8:] *= 1e-10
    r = regularize(A_B, A_I, reg=reg, rtol=1e-8)
    ref = tensions_fn(A_B, A_I, reg=reg, rtol=1e-8)
    for a in (1e6, 1e-6):
        s = regularize(A_B * a, A_I * a, reg=reg, rtol=1e-8)
        assert s.tau == pytest.approx(a * r.tau, rel=1e-12)
        assert np.allclose(tensions_fn(A_B * a, A_I * a, reg=reg, rtol=1e-8), ref,
                           rtol=1e-8, atol=0)


def test_tau_is_rtol_sqrt_n_on_an_equilibrated_pencil(pencil):
    """Equilibration gives every column of the stack unit norm, so `||stack||_F = sqrt(n_cols)`
    and `tau = rtol * sqrt(n_cols)` -- a constant for the pencil's life, adding no `lam`
    dependence of its own. (A column that underflowed to exact zeros would keep norm 0.)"""
    p = MPSPencil(pencil.bdry, pencil.interior, reg="tikhonov_qr")
    for lam in (LAM, 16.0, 30.0):
        assert p.regularized(lam).tau == pytest.approx(p.rtol * np.sqrt(p.n_cols), rel=1e-12)


@pytest.mark.parametrize("reg", PENALTY)
def test_a_penalty_mode_refuses_a_rank(pencil, reg):
    """A penalty chooses no rank, so freezing one -- directly or through `lam_ref` -- has
    nothing to act on. Refused rather than silently ignored."""
    with pytest.raises(ValueError, match="penalty"):
        MPSPencil(pencil.bdry, pencil.interior, reg=reg, rank=30)
    with pytest.raises(ValueError, match="penalty"):
        MPSPencil(pencil.bdry, pencil.interior, reg=reg, lam_ref=LAM)
    with pytest.raises(ValueError, match="rtol > 0"):
        MPSPencil(pencil.bdry, pencil.interior, reg=reg, rtol=0.0)
    A_B, A_I = well_conditioned()
    with pytest.raises(ValueError, match="penalty"):
        regularize(A_B, A_I, reg=reg, rank=5)
    with pytest.raises(ValueError, match="all-zero"):
        regularize(0 * A_B, 0 * A_I, reg=reg, rtol=1e-8)


def test_an_unknown_mode_is_refused_at_construction(pencil):
    """It used to be refused on first USE, inside `regularize`, which could be a long way
    from the line that asked for it."""
    with pytest.raises(ValueError, match="unknown regularization"):
        MPSPencil(pencil.bdry, pencil.interior, reg="bogus")


# --- the default: a penalty where it is measured clean, the threshold below ---------------
#
# `notes/smooth_filter.md` sections 10.4-10.7 and `notes/penalty_prec_limit_explained.md`. A
# penalty's floor IS its `tau`, so `tau` must scale with `prec`; rounding sets a floor under
# `tau` that does not. The two meet between 1e-12 and 1e-14. `"auto"` stops at 1e-8, the
# tightest `prec` at which the penalty measured no worse than the threshold it replaces.


def test_the_default_is_auto_on_both_the_pencil_and_the_solver():
    import inspect

    from chladni.mps import MPSEigensolver
    assert inspect.signature(MPSPencil.__init__).parameters["reg"].default == "auto"
    assert inspect.signature(MPSEigensolver.__init__).parameters["reg"].default == "auto"


# Rungs below 1e-10 are parked (`notes/test_inventory_2026-10-09.md`, decision 3).
@pytest.mark.parametrize("prec", [1e-4, 1e-6, 1e-8, 1e-10])
def test_auto_is_the_threshold_at_every_rung(pencil, prec):
    """What `reg` reports is the RESOLVED mode, so a caller can see which one it got.

    `"auto"` was the penalty at `prec >= 1e-8` from 2026-09-24, on rectangle evidence. On the
    douse benchmark the threshold beat it at 1e-6, 1e-8 and 1e-10 -- fewer silent losses, as
    stable or more, a tighter bound, less wall time (`notes/regularization_1e10.md`) -- so
    `"auto"` is the threshold at every rung and a penalty is asked for explicitly."""
    want = "svd"
    p = MPSPencil(pencil.bdry, pencil.interior, prec=prec)
    assert p.reg == want


@pytest.mark.parametrize("knob", [{"rtol": 1e-12}, {"rank": 30}, {"lam_ref": LAM}])
def test_auto_with_a_truncation_knob_is_the_threshold(pencil, knob):
    """`rtol=`, `rank=` and `lam_ref=` are the truncation's settings, so passing one asks for a
    truncation. It is also what keeps every fixture built at `rect_suite.SUITE_RTOL`, and every
    constant calibrated on one, exactly as it was measured."""
    assert MPSPencil(pencil.bdry, pencil.interior, prec=1e-8, **knob).reg == "svd"


@pytest.mark.parametrize("reg", PENALTY)          # "auto" is never a penalty since 2026-10-10
@pytest.mark.parametrize("prec", [1e-6, 1e-8, 1e-10])
def test_a_penalty_s_default_tau_is_proportional_to_prec(pencil, reg, prec):
    """`tau = C_PEN * prec`, absolute on the equilibrated pencil. `C_PEN` is 0.02 as shipped,
    the middle of the clean band [0.007, 0.05] prec measured at 1e-12 (section 10.6), but the
    contract is the proportionality: the constant is a calibration and is not pinned here. The
    shipped rule it replaces, `sqrt(prec*eps)`, was a truncation's, and put the floor at 2.4
    prec by 1e-14 (10.4)."""
    p = MPSPencil(pencil.bdry, pencil.interior, reg=reg, prec=prec)
    assert p.regularized(LAM).tau == pytest.approx(regmod.C_PEN * prec, rel=1e-12)


def test_the_threshold_s_rtol_rule_is_unchanged(pencil):
    p = MPSPencil(pencil.bdry, pencil.interior, reg="svd", prec=1e-8)
    assert p.rtol == pytest.approx(np.sqrt(1e-8 * np.finfo(float).eps), rel=1e-15)


@pytest.mark.parametrize("reg", PENALTY)
def test_a_penalty_below_its_reach_WARNS_and_still_runs(pencil, reg):
    """At 1e-14 no `tau` clears the cluster grid (section 10.6): `C_PEN * prec` is 2e-16, deep
    in the rounding regime. Asked for explicitly it is the caller's call -- warned, not refused."""
    from chladni.mps import PrecisionWarning
    with pytest.warns(PrecisionWarning, match="rounding"):
        p = MPSPencil(pencil.bdry, pencil.interior, reg=reg, prec=1e-14)
    assert np.all(np.isfinite(p.tensions(LAM)[:3]))


@pytest.mark.parametrize("prec", [1e-6, 1e-8, 1e-10])
def test_a_penalty_inside_its_reach_is_silent(pencil, prec):
    import warnings

    from chladni.mps import PrecisionWarning
    with warnings.catch_warnings():
        warnings.simplefilter("error", PrecisionWarning)
        MPSPencil(pencil.bdry, pencil.interior, reg="tikhonov_qr", prec=prec)


# --- penalty modes: no rank change, and what that buys ------------------------------------


# The scan runs to lam = 60, far above the fixture's lam_ref, where FourierBessel reports a
# digit less than 1e-8. The claim is about counting directions, not accuracy.
@pytest.mark.filterwarnings(
    "ignore:FourierBessel can deliver:chladni._core.PrecisionWarning")
def test_the_qr_penalty_never_changes_rank():
    """The stated goal, measured. Over a scan where the threshold rank takes seven values
    (42 to 48 of 50), `"tikhonov_qr"` yields 50 finite tensions at every `lam`: nothing is
    decided, so nothing can flip.

    The literal `"tikhonov"` half is retired with that mode. It reported `rank_at == 50` too,
    but its count of FINITE tensions moved, measured 41 to 50 over this scan: LAPACK's
    `?ggsvp3` deciding the interior block's rank with its own tolerance, which `tau*I` does
    nothing to prevent -- the reason the QR realization is the one shipped.
    """
    poly = Polygon(L_SHAPE, bc="dir")
    b = basis_for(poly)
    base = MPSPencil.from_basis(b, poly.bdry_quad(400),
                                poly.interior_pts(150, rng=np.random.default_rng(0)), reg="svd")
    lams = np.linspace(9.0, 60.0, 24)
    assert len({base.rank_at(lam) for lam in lams}) >= 2, "the scan must exercise flips"

    qr = MPSPencil(base.bdry, base.interior, reg="tikhonov_qr")
    assert {qr.rank_at(lam) for lam in lams} == {qr.n_cols}
    assert {int(np.isfinite(qr.tensions(lam)).sum()) for lam in lams} == {qr.n_cols}


def _flicker_point():
    """A plateau point on `rect(1, 1/phi)` where the threshold rank flips between 39 and 40 at
    every scale -- `notes/smooth_filter.md` section 3: 63-88 changes per 201 samples whether
    the interval is 2e-2 or 2e-10 wide.

    Found on the suite pencil cut at `sqrt(1e-14 eps)`, which is pinned here: the suite's own
    threshold is cut for 1e-10 since 2026-10-11, and the rank does not flicker at this point
    there (one reason the cut moved, `notes/suite_rtol_sweep.md`)."""
    import rect_suite as rs

    p = rs.family_A()[0].pencil
    rtol_1e14 = float(np.sqrt(1e-14 * np.finfo(float).eps))
    return MPSPencil(p.bdry, p.interior, reg="svd", rtol=rtol_1e14), 38.261708924


def _spread(f, centre, w, n=41):
    v = np.array([f(x) for x in centre + np.linspace(-w, w, n)])
    return float(np.ptp(v) / np.median(v))


@pytest.mark.slow
def test_the_qr_penalty_is_differentiable_where_the_threshold_flickers():
    """The test for differentiability: a smooth function's spread over an interval shrinks in
    proportion to the interval; noise does not.

    Measured at half-widths 2e-4 .. 2e-8: the penalty falls 1.56e-4, 1.56e-5, 1.53e-6,
    1.61e-7 -- exactly a decade per decade -- before flattening at its own floor, 8.5e-8. The
    threshold stays at 2.8e-3 to 3.8e-3 throughout, the 0.4% noise a flickering rank leaves.

    The penalty uses `MPSPencil`'s `prec`-derived `rtol`, not the suite's fixed 1.5e-15: at
    that `rtol` a plain penalty sits in its own saturated-noise regime (section 2).
    """
    p, x0 = _flicker_point()
    q = MPSPencil(p.bdry, p.interior, reg="tikhonov_qr", prec=1e-8)
    wide, narrow = _spread(q.tension, x0, 2e-4), _spread(q.tension, x0, 2e-6)
    assert wide / narrow > 50, f"spread {wide:.2e} -> {narrow:.2e}: not shrinking with width"
    assert _spread(q.tension, x0, 2e-8) < 1e-6
    # The CONTRAST needs the threshold to flicker here, and where it does is the BLAS's choice:
    # this point was found on Accelerate, and on OpenBLAS the rank holds at 39 across it (Linux
    # CI, 2026-10-04). The penalty's half above is the claim and has already run.
    if len({p.rank_at(x) for x in x0 + np.linspace(-2e-6, 2e-6, 41)}) < 2:
        pytest.skip("the threshold does not flicker at this point on this LAPACK; "
                    "the penalty's smoothness was asserted, the contrast has no subject")
    assert _spread(p.tension, x0, 2e-8) > 1e-3, "the threshold twin should plateau"


@pytest.mark.slow
def test_a_blas_sized_perturbation_flips_the_threshold_and_not_the_penalty():
    """The 2026-09-21 root cause, modelled. Linux and macOS disagree in the last bits, and
    where a singular value sits near the cutoff that decides the rank. Jitter the pencil by
    1e-15 relative -- a cross-BLAS difference -- at points whose first dropped singular value
    sits at 0.98x the cutoff.

    Measured over twelve such points: the threshold rank flips at five and `sigma_1` moves by
    up to 4.2e-3. The penalty moves by at most 6.6e-8, which is its plateau noise at this
    `tau` and five orders below.

    The penalty's half is the claim and is asserted first. Whether the threshold flips at
    these points is the platform's choice -- they were found on Accelerate, and a few ulps
    elsewhere move the singular values off the cutoff -- so, as in the flicker test above,
    the contrast is skipped when it has no subject rather than failed.
    """
    p, x0 = _flicker_point()
    rng = np.random.default_rng(0)
    rtol_pen = float(np.sqrt(1e-8 * np.finfo(float).eps))
    flips, worst_thr, worst_pen = 0, 0.0, 0.0
    for x in x0 + np.linspace(-2e-3, -1.45e-3, 12):
        B, I = p.pencil(x)
        Bp = B * (1 + 1e-15 * rng.standard_normal(B.shape))
        Ip = I * (1 + 1e-15 * rng.standard_normal(I.shape))
        a = tensions_fn(B, I, reg="svd", rtol=p.rtol)
        b = tensions_fn(Bp, Ip, reg="svd", rtol=p.rtol)
        flips += len(a) != len(b)
        worst_thr = max(worst_thr, abs(b[0] - a[0]) / a[0])
        c = tensions_fn(B, I, reg="tikhonov_qr", rtol=rtol_pen)[0]
        d = tensions_fn(Bp, Ip, reg="tikhonov_qr", rtol=rtol_pen)[0]
        worst_pen = max(worst_pen, abs(d - c) / c)
    assert worst_pen < 1e-6, f"the penalty moved {worst_pen:.2e} under a 1e-15 jitter"
    if flips == 0:
        pytest.skip("the threshold rank does not flip at these points on this LAPACK; "
                    "the penalty's stability was asserted, the contrast has no subject")
    assert worst_thr > 1e-4, f"a flip moved sigma_1 only {worst_thr:.2e}"


# --- the 2026-09-21 CI refusal ------------------------------------------------------------
#
# `notes/session_2026-09-19.md`, the 09-21 trace. On Linux/OpenBLAS the threshold rank
# flipped 85 -> 86 at ONE end of the bracket around the lowest member of the N = 65 cluster
# at `prec = 1e-6`, band 2.6. That lowered `sigma_1` there by 67%, the refiner walked toward
# the end, and a true root was refused `not_a_minimum` -- the silent wrong total in
# `test_the_cluster_total_GRID[N65-1e-06-b2.6]`. These check the penalty on that cell.
#
# They used to also replay the refusal by INJECTING the flip (a frozen rank 86 at the bracket's
# upper end) and pin refine's reason string. That half pinned one platform's trajectory: under
# a 4-ulp jitter (`tests/ulp_jitter.py`) the rank-85/86 jag moves to other offsets or vanishes,
# and the replay failed 8 of 8 seeds (`notes/jitter_sweep_slow_tier.json`). Its record is
# `notes/session_2026-09-19.md`; the contract kept is the penalty's.

_CI = dict(lo=641.5168481866608, hi=641.517275351795, hint=641.517204846656,
           root=641.5171694882)


def _ci_cell():
    import rect_suite as rs

    (case,) = rs.split_degeneracy_cases(1e-6, bands=(2.6,), levels={65: 4})
    return case


@pytest.mark.slow
def test_the_refused_bracket_end_is_smooth_under_the_penalty():
    """At the note's offsets around the refused end, the forced rank-86 / rank-85 ratio ran
    0.962, 0.781, 0.461, 0.328, 1.000 on macOS -- a 67% drop and back inside 8e-9 relative. The
    penalty is a straight ramp there: quadratic-fit residual 6.2e-10.

    Smooth means that residual. Under a 4-ulp jitter it measured 4.9e-10 to 6.4e-10 over seeds
    1-5 (the tensions agree to all printed digits), so the 1e-7 bar has two decades of room for
    rounding, and still sits six below the 0.3-0.7 relative jag a rank flip leaves. The forced
    ratio itself is not asserted: under the same jitter it lands at other offsets (seed 1:
    0.688 at -2e-6, flat elsewhere) or barely appears (seed 2: min 0.991), which is a fact
    about the threshold, not the penalty."""
    case = _ci_cell()
    p = case.pencil
    offs = np.array([-4e-6, -2e-6, -1e-6, 0.0, 1e-6])

    q = MPSPencil(p.bdry, p.interior, reg="tikhonov_qr", prec=1e-6)
    t = np.array([q.tension(_CI["hi"] + o) for o in offs])
    fit = np.polyval(np.polyfit(offs, t, 2), offs)
    resid = np.max(np.abs(t - fit) / t)
    assert resid < 1e-7, f"the penalty is not a smooth ramp at the refused end: {resid:.2e}"


@pytest.mark.slow
def test_the_ci_refusal_does_not_happen_under_the_penalty():
    """The note's one-second reproduction, penalty half. Under `"tikhonov_qr"` there is no rank
    to flip, and the refiner converges in four evaluations to 6.5e-8 from the root. (With the
    flip injected at the bracket's upper end it refused, `not_a_minimum`, `sigma = 5.330e-7`,
    exactly as recorded -- on macOS only; see the section comment.)"""
    from chladni.mps.objective import Objective
    from chladni.mps.refine import refine

    case = _ci_cell()
    p = case.pencil
    lo, hi, hint = _CI["lo"], _CI["hi"], _CI["hint"]

    def run(tensions):
        obj = Objective(tensions, case.a, case.b, prec=1e-6, variable="lam")
        return refine(obj, lo, hi, prec=1e-6, lam_hint=hint, interior=(lo, hi))

    q = MPSPencil(p.bdry, p.interior, reg="tikhonov_qr", prec=1e-6)
    res = run(q.tensions)
    assert res.accepted, f"refused under the penalty: {res.reason}"
    assert abs(res.lam - _CI["root"]) < 1e-6 * _CI["root"]


@pytest.mark.slow
def test_the_ci_cell_counts_four_under_the_penalty():
    """The cell itself, solved end to end under `"tikhonov_qr"`: 4 of 4. This is the portable
    acceptance test the session note asks a replacement regularization to pass. On macOS the
    threshold passes it too -- the failure needed OpenBLAS's rounding -- so here it guards
    against the penalty losing the cluster, and the Linux runner is where it discriminates."""
    import bracket_score as bs
    from chladni.mps.objective import Objective
    from chladni.mps.search import solve

    case = _ci_cell()
    p = case.pencil
    q = MPSPencil(p.bdry, p.interior, reg="tikhonov_qr", prec=1e-6)
    base = bs.objective_for(case, 1e-6)
    obj = Objective(q.tensions, base.lo, base.hi, prec=1e-6, variable="lam",
                    expected_count=getattr(base, "expected_count", None))
    res = solve(obj, prec=1e-6)
    assert not res.declined, res.diagnostics.decline_reason
    assert sum(x.mult for x in res.levels) == 4
