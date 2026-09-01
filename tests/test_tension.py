"""Ground truth: does the tension actually find eigenvalues.

Every reference here is closed form, per `conventions.md`. lappy's tabulated tables are not
ported -- its own `docs/todo.md` records `ellipse_eigs(a=2)[0]` being wrong in the 9th digit,
and a table nobody can rederive is not ground truth.

The sector is the sharp one. On a circular sector of opening `omega`, the Dirichlet
eigenfunctions are exactly `J_{l*nu}(k r) sin(l*nu*theta)` with `nu = pi/omega` -- which is
exactly ONE Fourier-Bessel column of the corner family at the apex. So a one-column basis
represents the eigenfunction with no error at all, and the tension at the true eigenvalue is
limited only by the Bessel kernel and the arithmetic. It also needs no basis planner, which
is what makes it available in this phase at all.

`docs/design/particular_solutions.md` lists these as "the mps layer's first tests". This is
that file.
"""
import numpy as np
import pytest

from chladni.geometry import ArcSegment, Boundary, Domain, LineSegment, Polygon, disk, eq_tri, rect
from chladni.helmholtz import FourierBessel, PlaneWave
from chladni.mps import MPSPencil

from conftest import (
    L_SHAPE,
    L_SHAPE_LAM1,
    disk_eigs,
    eq_tri_eigs,
    rect_eigs,
    sector_eigs,
)

SEED = 0


def _rng():
    return np.random.default_rng(SEED)


# --- the sector: one exact column --------------------------------------------------------


# `Sampling.accuracy` bounds what a general combination of Fourier-Bessel columns can be
# trusted to, and on these sectors it reports 1e1-1e2 -- no accuracy at all. That bound is
# not wrong, it is simply not the situation: with a SINGLE column the common kernel scale
# divides out of the tension entirely, which is the whole reason the sector is the sharp
# reference. The `< 1e-10` assertions below are the evidence, so suppressing here costs
# nothing that the test does not immediately re-check.
@pytest.mark.filterwarnings("ignore::chladni._core.PrecisionWarning")
@pytest.mark.parametrize("omega", [np.pi / 3, np.pi / 2, 2 * np.pi / 3, 3 * np.pi / 2])
def test_the_sector_eigenfunction_is_one_basis_column(omega):
    """`nu = pi/omega`, and the l-th Dirichlet mode is `J_{l*nu}(k r) sin(l*nu*theta)`. With
    a single column the pencil is 1-dimensional, so the tension is the exact relative
    boundary residual of the exact eigenfunction -- machine zero, and nothing else.

    `omega = pi` is absent on purpose. That sector is a half-disk, whose apex sits in the
    middle of a straight diameter, and `Boundary.corners` correctly does not report a corner
    where the tangent does not turn and the boundary condition does not change. There is
    nothing for `at_corner` to attach to, which is the geometry layer being right rather
    than a gap here -- the half-disk's spectrum is checked through `sector_eigs(k, omega=pi)` in
    conftest instead.
    """
    dom = _sector(omega)
    apex = next(c for c in dom.corners if abs(c.z) < 1e-12)
    fam = FourierBessel.at_corner(apex, n=1, domain=dom)

    lam = sector_eigs(1, omega=omega)[0]
    p = MPSPencil.from_basis(fam, dom.bdry_quad(400),
                             dom.interior_pts(60, rng=_rng()), lam_ref=lam)
    assert p.n_cols == 1
    assert p.tension(lam) < 1e-10


@pytest.mark.filterwarnings("ignore::chladni._core.PrecisionWarning")
@pytest.mark.parametrize("omega", [2 * np.pi / 3, 3 * np.pi / 2])
def test_the_sector_tension_has_a_minimum_at_each_eigenvalue(omega):
    """Away from an eigenvalue the same single column cannot satisfy the arc condition, so
    the tension is O(1). That gap is what the deferred minimizer will hunt."""
    dom = _sector(omega)
    apex = next(c for c in dom.corners if abs(c.z) < 1e-12)
    fam = FourierBessel.at_corner(apex, n=3, domain=dom)
    lams = sector_eigs(3, omega=omega)
    p = MPSPencil.from_basis(fam, dom.bdry_quad(500),
                             dom.interior_pts(80, rng=_rng()), lam_ref=lams[0])
    for lam in lams:
        at = p.tension(lam)
        off = min(p.tension(lam * f) for f in (0.93, 1.07))
        assert at < 1e-8, f"tension {at:.2e} at the exact eigenvalue {lam}"
        assert off > 1e3 * at, f"no dip at {lam}: {at:.2e} against {off:.2e} nearby"


@pytest.mark.filterwarnings("ignore::chladni._core.PrecisionWarning")
def test_the_sector_coefficient_reproduces_the_exact_eigenfunction():
    """The coefficient half of the same reference. With one column the eigenfunction is that
    column, so `coef` must return a scalar multiple of it -- checked on interior points the
    pencil never saw, which is the property `log_scale` and the scaled columns exist for."""
    omega = 2 * np.pi / 3
    dom = _sector(omega)
    apex = next(c for c in dom.corners if abs(c.z) < 1e-12)
    fam = FourierBessel.at_corner(apex, n=1, domain=dom)
    lam = sector_eigs(1, omega=omega)[0]
    k = np.sqrt(lam)

    p = MPSPencil.from_basis(fam, dom.bdry_quad(400),
                             dom.interior_pts(60, rng=_rng()), lam_ref=lam)
    c = p.coef(lam)
    assert c.shape == (1, 1)

    # `k` survives here only because `fam.values` is a call into `helmholtz`, which is the
    # far side of the seam; everything asked of the pencil is in `lam`.
    off = dom.interior_pts(50, rng=np.random.default_rng(77))
    got = fam.values(k, off) @ (c[:, 0] * np.exp(-p.log_scale(lam)))
    want = fam.values(k, off)[:, 0]
    ratio = got / want
    assert np.allclose(ratio, ratio[0], rtol=1e-10, atol=0), "not a multiple of the column"


# --- the closed-form domains -------------------------------------------------------------


def _waves(dom, lam):
    """As many plane-wave directions as are numerically independent at `prec`, and no more.

    `PlaneWave.n_useful` is not advice: past that count the extra columns arrive at the
    noise floor and, in its own words, "lower the smallest singular value and corrupt the
    tension" -- which is this file's measurement, not a distant one. Oversizing the basis
    makes the tension look BETTER while meaning less.
    """
    R = dom.diameter
    return PlaneWave.uniform(int(PlaneWave.n_useful(np.sqrt(lam) * R)), radius=R)


def test_the_rectangle():
    """Plane waves alone: a rectangle has no singular corner, `omega = pi/2` gives
    `nu = 2`, and the eigenfunctions are analytic across every vertex."""
    dom = rect(2.0, 1.0)
    lam = rect_eigs(1, 2.0, 1.0)[0]
    b = _waves(dom, lam)
    p = MPSPencil.from_basis(b, dom.bdry_quad(400),
                             dom.interior_pts(120, rng=_rng()), lam_ref=lam)
    assert p.tension(lam) < 1e-6


def test_the_disk():
    dom = disk(1.0)
    lam = disk_eigs(1, 1.0)[0]
    b = _waves(dom, lam)
    p = MPSPencil.from_basis(b, dom.bdry_quad(400),
                             dom.interior_pts(120, rng=_rng()), lam_ref=lam)
    assert p.tension(lam) < 1e-6


def test_the_equilateral_triangle():
    dom = eq_tri(1.0)
    lam = eq_tri_eigs(1, 1.0)[0]
    b = _waves(dom, lam)
    p = MPSPencil.from_basis(b, dom.bdry_quad(400),
                             dom.interior_pts(120, rng=_rng()), lam_ref=lam)
    assert p.tension(lam) < 1e-6


# --- the corner-singularity benchmark ----------------------------------------------------


@pytest.mark.slow
def test_the_l_shape_reaches_twelve_digits_with_corner_enrichment():
    """The corner-singularity benchmark, and the reason `FourierBessel` exists at all.

    The L-shape's reentrant corner has `omega = 3*pi/2`, so `nu = 2/3` and the eigenfunction
    behaves like `r**(2/3)` there. No analytic basis resolves that, and the measurement is
    brutal -- plane waves alone, 40 of them, reach a tension of 3.6e-01 at the exact
    eigenvalue. Adding the corner family to every vertex, degrees converging:

        degrees per corner    tension at lam_1
              14                  3.9e-09
              18                  1.6e-10
              22                  1.7e-12

    Both halves are asserted. The absolute number says the pipeline is accurate; the GAP
    against the unenriched basis says the corner family is actually wired in, and it is the
    half that catches a regression where enrichment silently stops contributing.
    """
    dom = Polygon(L_SHAPE, bc="dir")
    q, pts = dom.bdry_quad(800), dom.interior_pts(300, rng=_rng())

    plain = _waves(dom, L_SHAPE_LAM1)
    enriched = sum((FourierBessel.at_corner(c, n=22, domain=dom) for c in dom.corners[1:]),
                   FourierBessel.at_corner(dom.corners[0], n=22, domain=dom))

    t_plain = MPSPencil.from_basis(plain, q, pts, lam_ref=L_SHAPE_LAM1).tension(L_SHAPE_LAM1)
    t_rich = MPSPencil.from_basis(enriched, q, pts,
                                  lam_ref=L_SHAPE_LAM1).tension(L_SHAPE_LAM1)

    assert t_rich < 1e-11, f"expected ~1.7e-12 at the exact eigenvalue, got {t_rich:.3e}"
    assert t_rich < 1e-8 * t_plain, f"enrichment bought nothing: {t_rich:.2e} vs {t_plain:.2e}"


@pytest.mark.slow
def test_the_l_shape_tension_converges_in_the_corner_degrees():
    """The convergence itself, which is the statement that this is a method and not a fit.
    Each step up in degrees must buy at least an order of magnitude."""
    dom = Polygon(L_SHAPE, bc="dir")
    q, pts = dom.bdry_quad(800), dom.interior_pts(300, rng=_rng())

    got = []
    for n in (14, 18, 22):
        b = sum((FourierBessel.at_corner(c, n=n, domain=dom) for c in dom.corners[1:]),
                FourierBessel.at_corner(dom.corners[0], n=n, domain=dom))
        got.append(MPSPencil.from_basis(b, q, pts,
                                       lam_ref=L_SHAPE_LAM1).tension(L_SHAPE_LAM1))
    assert got[1] < 0.1 * got[0], f"14 -> 18 degrees bought nothing: {got}"
    assert got[2] < 0.1 * got[1], f"18 -> 22 degrees bought nothing: {got}"


# --- helpers ------------------------------------------------------------------------------


def _sector(omega, r=1.0):
    """Circular sector, apex at the origin, first edge along the positive real axis.

    Not in `geometry.library` -- it is built here because it is a test reference rather than
    a shape the library owes anyone.
    """
    return Domain(Boundary([
        LineSegment(0, r),
        ArcSegment(0j, r, 0.0, omega),
        LineSegment(r * np.exp(1j * omega), 0),
    ]))


# --- coefficients attain the tension -----------------------------------------------------


@pytest.mark.parametrize("reg", ["svd", "qrp"])
def test_the_coefficients_attain_the_reported_tension(reg):
    """The defining property of `coef`, and the one that catches a wrong `back`.

    The tension IS `min ||A_B c|| / ||A_I c||`, so the vector `coef` returns must realize
    the number `tension` returns. Checked through the equilibrated pencil, since that is the
    space `coef` reports in, and across both regularization modes -- the recovery map is the
    only thing that differs between them, and a mode whose `back` is wrong produces a
    perfectly plausible vector attaining the wrong ratio.
    """
    dom = Polygon(L_SHAPE, bc="dir")
    b = sum((FourierBessel.at_corner(c, n=14, domain=dom) for c in dom.corners[1:]),
            FourierBessel.at_corner(dom.corners[0], n=14, domain=dom))
    p = MPSPencil.from_basis(b, dom.bdry_quad(800),
                             dom.interior_pts(300, rng=_rng()), lam_ref=L_SHAPE_LAM1,
                             reg=reg)

    B, I = p.pencil(L_SHAPE_LAM1)
    c = p.coef(L_SHAPE_LAM1)[:, 0]
    ratio = np.linalg.norm(B @ c) / np.linalg.norm(I @ c)
    assert ratio == pytest.approx(p.tension(L_SHAPE_LAM1), rel=1e-6)


@pytest.mark.parametrize("reg", ["svd", "qrp"])
def test_mult_returns_the_smallest_few_in_order(reg):
    """What the deferred multiplicity estimator will lean on: an eigenvalue of multiplicity
    m drives m tensions to zero together, and `coef(lam, mult=m)` must hand back that whole
    subspace, in the same order `tensions` reports it."""
    dom = Polygon(L_SHAPE, bc="dir")
    b = sum((FourierBessel.at_corner(c, n=14, domain=dom) for c in dom.corners[1:]),
            FourierBessel.at_corner(dom.corners[0], n=14, domain=dom))
    p = MPSPencil.from_basis(b, dom.bdry_quad(800),
                             dom.interior_pts(300, rng=_rng()), lam_ref=L_SHAPE_LAM1,
                             reg=reg)

    B, I = p.pencil(L_SHAPE_LAM1)
    C = p.coef(L_SHAPE_LAM1, mult=3)
    assert C.shape == (p.n_cols, 3)
    got = [np.linalg.norm(B @ C[:, j]) / np.linalg.norm(I @ C[:, j]) for j in range(3)]
    assert np.allclose(got, p.tensions(L_SHAPE_LAM1)[:3], rtol=1e-6)
