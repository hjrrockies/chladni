"""The Rellich identity on an integral rule: `chladni.integrals.{plan, cauchy, identities}` (M2).

Three kinds of ground truth, none of them another chladni integral:

* EXACT INTERIOR RULES for functions that are Helmholtz-exact but NOT eigenfunctions -- the case
  the full identity exists for, since an MPS candidate is nonzero on the boundary. A tensor
  Gauss rule on a rectangle; a tensor rule graded toward the re-entrant corner on the L-shape.
  These are test oracles only: production never integrates over the interior
  (`notes/solver_features.md` A6).
* CLOSED-FORM NORMS of reference eigenfunctions (`chladni.reference`), the sector's singular
  apex among them.
* ORIGIN INDEPENDENCE, which the identity guarantees and a quadrature error breaks.
"""
import numpy as np
import pytest

from chladni import reference as ref
from chladni.geometry import disk, disk_sector, polyomino, rect
from chladni.geometry._adaptive import gauss_legendre
from chladni.helmholtz import FourierBessel, PlaneWave
from chladni.integrals import cauchy as C
from chladni.integrals import identities as I
from chladni.integrals import plan as P

RNG = np.random.default_rng(11)


def _tensor(x_rule, y_rule, x0=0.0, y0=0.0):
    (xs, wx), (ys, wy) = x_rule, y_rule
    pts = (x0 + xs[:, None]) + 1j * (y0 + ys[None, :])
    return pts.ravel(), (wx[:, None] * wy[None, :]).ravel()


def _graded(toward_end, layers=14, ratio=0.25, order=20):
    """A composite Gauss rule on [0, 1], panels shrinking geometrically toward `toward_end`."""
    edges = np.concatenate([[0.0], ratio ** np.arange(layers, 0, -1), [1.0]])
    xs, ws = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        x, w = gauss_legendre(order)
        xs.append(a + (b - a) * x)
        ws.append((b - a) * w)
    x, w = np.concatenate(xs), np.concatenate(ws)
    return (x, w) if toward_end == 0 else (1.0 - x[::-1], w[::-1])


# --- the rule ----------------------------------------------------------------------------------

def test_a_rectangle_needs_no_corner_rule():
    """A right-angled Dirichlet corner has `mu = 2`, `gamma = 2`: a polynomial. Smooth panels
    only, with the arclength rule's own invariants intact."""
    dom = rect(1.3, 0.7)
    iq = P.integral_quad(dom, 80.0, prec=1e-10)
    assert {s.kind for s in iq.corners} == {"smooth"}
    assert iq.quad.wts.sum() == pytest.approx(dom.perimeter, rel=1e-14)
    assert iq.sizing_prec <= P.integral_prec(1e-10) and iq.shortfalls == ()


def test_the_L_shape_rules_its_reentrant_corner_and_only_that_one():
    dom = polyomino([(0, 0), (1, 0), (0, 1)])
    iq = P.integral_quad(dom, 40.0, prec=1e-10)
    ruled = [s for s in iq.corners if s.needs_rule]
    assert len(ruled) == 1 and ruled[0].corner.z == pytest.approx(1 + 1j)
    assert ruled[0].kind == "substitution"            # straight, 2/nu = 3
    assert iq.x0 == pytest.approx(1 + 1j)             # r . N vanishes on both its edges


def test_a_135_degree_corner_is_ruled_though_it_is_convex():
    """`mu = 4/3` puts `r^(2/3)` in the integrand: bounded, convex, and beyond plain Gauss."""
    dom = disk_sector(1.0, 0.75 * np.pi)
    iq = P.integral_quad(dom, 30.0, prec=1e-12)
    apex = next(s for s in iq.corners if abs(s.corner.z) < 1e-12)
    assert apex.needs_rule and apex.corner.omega < np.pi


# --- the identity, on functions that are NOT eigenfunctions ------------------------------------

def test_the_identity_on_plane_waves_matches_an_exact_interior_rule():
    """Plane waves at `lam = 50`, away from every eigenvalue, nonzero on the boundary: the full
    identity -- tangential and boundary-value terms included -- against a tensor Gauss rule that
    is exact to rounding for them. Two functions, so the off-diagonal is checked too."""
    L, H, lam = 1.3, 0.7, 50.0
    dom = rect(L, H)
    fam = PlaneWave.uniform(8, radius=dom.diameter)
    c = np.zeros((len(fam), 2))
    c[RNG.choice(len(fam), 5, replace=False), 0] = RNG.standard_normal(5)
    c[RNG.choice(len(fam), 5, replace=False), 1] = RNG.standard_normal(5)

    pts, wts = _tensor(*[(lambda r: (s * r[0], s * r[1]))(gauss_legendre(60)) for s in (L, H)])
    U = fam.values(np.sqrt(lam), pts) @ c
    want = (U * wts[:, None]).T @ U

    iq = P.integral_quad(dom, lam, prec=1e-12)
    cd = C.cauchy_data(fam, c, lam, iq.quad)
    for x0 in (iq.x0, 0.0, 0.9 + 0.5j):
        assert np.allclose(I.gram(cd, x0), want, rtol=1e-13, atol=1e-13 * np.abs(want).max())


def test_the_identity_at_a_SINGULAR_corner_matches_a_graded_interior_rule():
    """A random combination of the re-entrant corner's own Fourier-Bessel columns on the L-shape,
    at a `lam` that is no eigenvalue: zero on the corner's two edges, nonzero on the rest, and
    `r^(2/3)` at the corner. The interior oracle is a tensor rule graded toward that corner."""
    lam = 23.0
    dom = polyomino([(0, 0), (1, 0), (0, 1)])
    corner = next(c for c in dom.corners if c.is_reentrant)
    fam = FourierBessel.at_corner(corner, n=6, domain=dom)
    c = RNG.standard_normal(len(fam))

    pts, wts = [], []
    for (i, j), (ex, ey) in {(0, 0): (1, 1), (1, 0): (0, 1), (0, 1): (1, 0)}.items():
        p, w = _tensor(_graded(ex), _graded(ey), i, j)
        pts.append(p)
        wts.append(w)
    pts, wts = np.concatenate(pts), np.concatenate(wts)
    want = float(np.sum(wts * (fam.values(np.sqrt(lam), pts) @ c) ** 2))

    iq = P.integral_quad(dom, lam, prec=1e-12)
    cd = C.cauchy_data(fam, c, lam, iq.quad)
    for x0 in (iq.x0, 0.3 + 0.2j, 1.7 + 0.4j):
        assert I.gram(cd, x0)[0, 0] == pytest.approx(want, rel=1e-12), x0


# --- exact norms of reference eigenfunctions ------------------------------------------------------

#: Bars per apex angle (as a multiple of pi), lappy's leg 1: the capability falls toward the slit.
SECTOR_BARS = [(1.25, 1e-13), (1.5, 1e-13), (1.6, 1e-10), (1.75, 1e-10), (1.9, 1e-8)]


@pytest.mark.parametrize("alpha, bar", SECTOR_BARS)
def test_the_exact_sector_norm_at_a_reentrant_apex(alpha, bar):
    """The sharpest reference: `J_nu(k r) sin(nu theta)` with `nu = pi/omega < 1`, singular at the
    apex, norm in closed form. The arc's corners are right angles and need no rule."""
    omega = alpha * np.pi
    lam = ref.sector_eigs(1, omega=omega)[0]
    u, norm2 = ref.sector_eigfun(1, 1, omega=omega)
    g = ref.sector_eigfun_grad(1, 1, omega=omega)
    iq = P.integral_quad(disk_sector(1.0, omega), lam, prec=1e-13)
    cd = C.CauchyData.from_functions(iq.quad, lam, u, g)
    for x0 in (iq.x0, 0.2 + 0.1j):
        assert I.gram(cd, x0)[0, 0] == pytest.approx(norm2, rel=bar), x0


@pytest.mark.parametrize("name, dom, fn, lam", [
    ("rect", rect(1.3, 0.7), (ref.rect_eigfun(2, 3, L=1.3, H=0.7),
                              ref.rect_eigfun_grad(2, 3, L=1.3, H=0.7)),
     float(np.pi**2 * ((2 / 1.3) ** 2 + (3 / 0.7) ** 2))),
    ("L polyomino", polyomino([(0, 0), (1, 0), (0, 1)]),
     (ref.polyomino_eigfun(2, 1, 3), ref.polyomino_eigfun_grad(2, 1)), 5 * np.pi**2),
    ("disk", disk(1.0), (ref.disk_eigfun(3, 2), ref.disk_eigfun_grad(3, 2)),
     float(ref.bessel_zero(3, 2) ** 2)),
])
def test_exact_norms_of_smooth_reference_eigenfunctions(name, dom, fn, lam):
    (u, norm2), g = fn
    iq = P.integral_quad(dom, lam, prec=1e-13)
    cd = C.CauchyData.from_functions(iq.quad, lam, u, g)
    assert I.gram(cd, iq.x0)[0, 0] == pytest.approx(norm2, rel=1e-13), name


# --- the clearance cap ---------------------------------------------------------------------------

def test_the_clearance_cap_is_what_keeps_a_long_edge_accurate():
    """A 1 x 16 strip with a cell below one end: the re-entrant corner's edge is 15 long against a
    clearance of 1. With the cap, machine precision; without it, the far end of a 15-long corner
    panel is starved (lappy: 7.7e-2 at N = 16)."""
    N = 16
    dom = polyomino([(i, 0) for i in range(N)] + [(0, -1)])
    u, norm2 = ref.polyomino_eigfun(2, 3, N + 1)
    g = ref.polyomino_eigfun_grad(2, 3)
    lam = 13 * np.pi**2
    errs = {}
    for cf in (P.CLEARANCE_FRAC, None):
        iq = P.integral_quad(dom, lam, prec=1e-13, clearance_frac=cf)
        cd = C.CauchyData.from_functions(iq.quad, lam, u, g)
        errs[cf] = abs(I.gram(cd, 0.3 - 0.4j)[0, 0] / norm2 - 1)
    assert errs[P.CLEARANCE_FRAC] < 1e-13 and errs[None] > 1e-6, errs


# --- cancellation ------------------------------------------------------------------------------

def test_the_cancellation_ratio_is_read_from_the_same_data():
    """Two plane waves 1e-6 radians apart are nearly the same function, so their difference is
    ~1e6 times smaller than either term -- the shape of a minimizing vector off any eigenvalue,
    which measured 1.8e13 (`notes/moler_payne_vs_tension.md`). One wave alone is benign."""
    dom = rect(1.0, 1.0)
    fam = PlaneWave(np.exp(1j * np.array([0.3, 0.3 + 1e-6])), radius=dom.diameter)
    iq = P.integral_quad(dom, 30.0, prec=1e-8)
    V = fam.at(iq.quad).values(np.sqrt(30.0))
    a, b = max(((i, j) for i in range(len(fam)) for j in range(i + 1, len(fam))),
               key=lambda ij: abs(np.dot(V[:, ij[0]], V[:, ij[1]]))
               / (np.linalg.norm(V[:, ij[0]]) * np.linalg.norm(V[:, ij[1]])))
    one, diff = np.zeros(len(fam)), np.zeros(len(fam))
    one[a] = 1.0
    diff[a], diff[b] = 1.0, -1.0
    cds = [C.cauchy_data(fam, c, 30.0, iq.quad) for c in (one, diff)]
    benign, cancelled = (cd.cancellation(I.norms(cd, iq.x0), dom.area)[0] for cd in cds)
    assert benign < 3.0 and cancelled > 1e5, (benign, cancelled)


# --- an MPS candidate, not a reference function ---------------------------------------------------

@pytest.mark.slow
def test_the_norm_of_a_real_L_shape_candidate_matches_the_interior_oracle():
    """The first L-shape eigenfunction as the solver represents it -- 80 corner Fourier-Bessel
    columns, singular at the re-entrant corner -- against a tensor rule graded toward that
    corner. This is the case that set `SMOOTH_SAFETY` to 2: at 1 the smooth panels missed by
    1.8e-11, the corner panel being exact. Its cancellation is benign (about 2), which the
    boundary-rms definition this replaced read as 1e15."""
    import sys
    sys.path.insert(0, "tests")
    from hard_roster import solver_for
    from chladni.geometry import L_shape

    dom = L_shape()                                 # re-entrant corner at the origin
    lam = float(ref.L_shape_eigs(1)[0])
    p = solver_for(dom, lam * 1.3, label="L", prec=1e-10).pencil
    fc = p.family_coef(lam, p.coef(lam)[:, 0])

    pts, wts = [], []
    for (x0, y0), (ex, ey) in {(-1, 0): (1, 0), (-1, -1): (1, 1), (0, -1): (0, 1)}.items():
        q, w = _tensor(_graded(ex, layers=16, order=24), _graded(ey, layers=16, order=24), x0, y0)
        pts.append(q)
        wts.append(w)
    pts, wts = np.concatenate(pts), np.concatenate(wts)
    want = float(np.sum(wts * (p.basis.values(np.sqrt(lam), pts) @ fc) ** 2))

    iq = P.integral_quad(dom, lam, prec=1e-12)
    cd = C.cauchy_data(p.basis, fc, lam, iq.quad)
    got = I.gram(cd, iq.x0)[0, 0]
    assert got == pytest.approx(want, rel=1e-13)
    assert cd.cancellation(np.sqrt([got]), dom.area)[0] < 10

