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
import warnings

import numpy as np
import pytest

from chladni import reference as ref
from chladni._core import PrecisionWarning
from chladni.geometry import Polygon, disk, disk_sector, polyomino, rect
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



# --- the graded corner rule in corner-local frames (2026-11-03) --------------------------------
#
# `notes/hadamard_accuracy.md`, addenda; `notes/gram_accuracy.md` section. Every sharp test above
# puts the singular corner AT THE ORIGIN, where a node's absolute position is its distance from
# the corner. Shifted to 3+2i, that distance is rounded at eps*|3+2i|; the graded rule's anchored
# panels evaluate in the corner's own frame instead.

SHIFT = 3 + 2j


def test_graded_corner_panels_are_anchored_at_their_corner():
    dom = polyomino([(0, 0), (1, 0), (0, 1)]).translated(SHIFT)
    iq = P.integral_quad(dom, 40.0, prec=1e-12, corners="graded")
    corner = next(c for c in dom.corners if c.is_reentrant)
    q = iq.quad
    assert q.anchored.any()
    assert set(np.unique(q.anchor_pt[q.anchored])) == {corner.z}
    assert np.all(q.wts > 0)
    np.testing.assert_allclose(q.anchor_pt[q.anchored] + q.offset[q.anchored],
                               q.pts[q.anchored], atol=4e-16 * abs(SHIFT))
    assert iq.sizing_prec <= P.integral_prec(1e-12) and iq.shortfalls == ()
    # The graded rule is the default; the fitted one, kept for measurement, is unanchored.
    assert P.integral_quad(dom, 40.0, prec=1e-12).quad.anchored.any()
    assert not P.integral_quad(dom, 40.0, prec=1e-12,
                               corners="interpolatory").quad.anchored.any()


@pytest.mark.parametrize("shift", [0j, SHIFT], ids=["origin", "shifted"])
def test_the_singular_corner_identity_survives_translation_on_the_graded_rule(shift):
    """`test_the_identity_at_a_SINGULAR_corner_matches_a_graded_interior_rule`, moved off the
    origin. The interior oracle is evaluated in the unshifted frame (translation is
    column-exact), so it does not degrade with the shift."""
    lam = 23.0
    dom0 = polyomino([(0, 0), (1, 0), (0, 1)])
    corner0 = next(c for c in dom0.corners if c.is_reentrant)
    fam0 = FourierBessel.at_corner(corner0, n=6, domain=dom0)
    c = np.random.default_rng(5).standard_normal(len(fam0))
    pts, wts = [], []
    for (i, j), (ex, ey) in {(0, 0): (1, 1), (1, 0): (0, 1), (0, 1): (1, 0)}.items():
        p, w = _tensor(_graded(ex), _graded(ey), i, j)
        pts.append(p)
        wts.append(w)
    pts, wts = np.concatenate(pts), np.concatenate(wts)
    want = float(np.sum(wts * (fam0.values(np.sqrt(lam), pts) @ c) ** 2))

    dom, fam = dom0.translated(shift), fam0.translated(shift)
    iq = P.integral_quad(dom, lam, prec=1e-12, corners="graded")
    cd = C.cauchy_data(fam, c, lam, iq.quad)
    for x0 in (iq.x0, 0.3 + 0.2j + shift, 1.7 + 0.4j + shift):
        assert I.gram(cd, x0)[0, 0] == pytest.approx(want, rel=1e-13), x0


#: Sector apexes for the graded rule, at the origin AND shifted. Measured 2026-11-03, worst over
#: three origins: 1.3e-15 .. 2.0e-14 (238 .. 578 nodes) against the interpolatory rule's
#: 1e-14 .. 1.0e-8, which grows toward the slit and with the shift. 1.95 pi (nu = 0.513) is
#: inside `NU_MARGIN` and demoted by both rules.
GRADED_SECTOR_ALPHAS = [1.25, 1.5, 1.6, 1.75, 1.9]


@pytest.mark.parametrize("shift", [0j, SHIFT], ids=["origin", "shifted"])
@pytest.mark.parametrize("alpha", GRADED_SECTOR_ALPHAS)
def test_the_exact_sector_norm_on_the_graded_rule_anywhere(alpha, shift):
    """The sector apex's own Fourier-Bessel column IS `J_nu(k r) sin(nu theta)` up to a constant,
    read off at one interior point; through `cauchy_data` it gets the corner-local frame."""
    omega = alpha * np.pi
    lam = ref.sector_eigs(1, omega=omega)[0]
    u, norm2 = ref.sector_eigfun(1, 1, omega=omega)
    dom0 = disk_sector(1.0, omega)
    apex = next(c for c in dom0.corners if abs(c.z) < 1e-14)
    fam0 = FourierBessel.at_corner(apex, degrees=[1], domain=dom0)
    probe = 0.4 * np.exp(0.5j * omega)
    scale = fam0.values(np.sqrt(lam), np.array([probe]))[0, 0] / u(np.array([probe]))[0]
    dom = dom0.translated(shift)
    iq = P.integral_quad(dom, lam, prec=1e-13, corners="graded")
    cd = C.cauchy_data(fam0.translated(shift), [1.0], lam, iq.quad)
    for x0 in (iq.x0, 0.2 + 0.1j + shift, -0.3 + 0.4j + shift):
        assert I.gram(cd, x0)[0, 0] == pytest.approx(norm2 * scale**2, rel=1e-13), x0


# --- a corner near another edge (2026-11-08) --------------------------------------------------
# An eigenfunction on a polygon is singular at the vertices and nowhere else, so along an edge its
# continuation in tau is singular at each OTHER vertex's `t* +- i y`, `y` the vertex's distance
# from the edge's line. A smooth panel sized by wavenumber alone cannot see that: douse's N = 6
# stall (a 250-degree corner 0.156 from the long edge) integrated that edge to 1e-6 at a claimed
# 1e-10, with every identity pointing at it and the eigenfunctions good to 1e-9.

_STALL = np.array([1.199468, 1.402903 + 0.281831j, 1.286492 + 0.768503j, 0.586380 + 0.837491j,
                   0.544682 + 0.156420j, 0.0])


def _edge(iq, seg):
    m = np.asarray(iq.quad.seg_idx) == seg
    return np.asarray(iq.quad.pts)[m], np.asarray(iq.quad.wts)[m]


@pytest.mark.parametrize("prec", [1e-9, 1e-12])
def test_an_edge_passing_near_another_corner_resolves_that_corners_singularity(prec):
    """The corner's own leading power, seen from the edge it does not touch: `|z - v|^gamma`."""
    dom = Polygon(_STALL)
    v, gamma = _STALL[4], 2 * np.pi / np.deg2rad(250.474) - 2
    iq = P.integral_quad(dom, 31.33, prec=prec, family="integer", corners="graded")
    pts, wts = _edge(iq, 5)
    x, w = gauss_legendre(40)
    edges = np.linspace(0.0, _STALL[0].real, 201)
    nodes = (edges[:-1, None] + np.diff(edges)[:, None] * x).ravel()
    exact = np.sum((np.diff(edges)[:, None] * w).ravel() * np.abs(nodes - v) ** gamma)
    assert abs(np.sum(wts * np.abs(pts - v) ** gamma) - exact) <= 10 * P.integral_prec(prec)


def test_a_panel_is_split_toward_a_near_corner_and_left_alone_otherwise():
    """Sizing against wavenumber alone already covers a far singularity: then nothing changes."""
    def by_wave(h):
        return 16, 0.0
    # one singularity just off the middle of the panel: the panels shrink toward it
    pieces = P._smooth_panels(0.0, 1.0, np.array([0.45 + 0.13j]), by_wave, 1e-10)
    widths = {round((a + b) / 2, 3): b - a for a, b, _, _ in pieces}
    assert len(pieces) > 1 and pieces[0][0] == 0.0 and pieces[-1][1] == 1.0
    assert all(p[1] == q[0] for p, q in zip(pieces[:-1], pieces[1:]))
    assert widths[min(widths, key=lambda c: abs(c - 0.45))] == min(widths.values())
    # a far one: exactly the panel wavenumber sizing gave
    assert P._smooth_panels(0.0, 1.0, np.array([0.5 + 2j]), by_wave, 1e-10) == [(0.0, 1.0, 16, 0.0)]
    assert P._smooth_panels(0.0, 1.0, np.array([], dtype=complex), by_wave, 1e-10) == [(0.0, 1.0, 16, 0.0)]


@pytest.mark.parametrize("name, verts", [
    ("regular hexagon", np.exp(2j * np.pi * np.arange(6) / 6)),
    ("square", np.array([0, 1, 1 + 1j, 1j])),
    ("triangle", np.array([0, 1, 0.3 + 0.8j]))])
def test_shapes_with_no_near_corner_keep_their_rule_node_for_node(name, verts, monkeypatch):
    dom = Polygon(verts)
    q = P.integral_quad(dom, 40.0, prec=1e-10, family="integer", corners="graded").quad
    monkeypatch.setattr(P, "_near_singularities", lambda *a: np.array([], dtype=complex))
    q0 = P.integral_quad(dom, 40.0, prec=1e-10, family="integer", corners="graded").quad
    assert np.array_equal(q.pts, q0.pts) and np.array_equal(q.wts, q0.wts)


@pytest.mark.slow
def test_the_hadamard_identities_close_on_the_stalled_shape():
    """Level 1 of the stall read 9.9e-7 here before; a converged reference on that one edge gave
    1.3e-10."""
    from chladni.eigenproblem import Eigenproblem
    rec = Eigenproblem(Polygon(_STALL)).solve(1, prec=1e-9)
    with warnings.catch_warnings():     # short of 1e-9 on the eigenfunction term, honestly
        warnings.simplefilter("ignore", PrecisionWarning)
        _, rep = rec.hadamard(0, lambda c, t: np.ones((1, len(t))), report=True)
    assert rep.identity_residual <= 1e-8, rep.identity_fields
