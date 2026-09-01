"""What the layers above `mps` depend on.

The deferred eigenvalue search will be written against this file, the way `mps` was written
against `test_helmholtz_seam.py` and `helmholtz` against `test_solver_seam.py`.

Three things are being pinned.

The tension function is what the search consumes, so it must be an honest function: an
ascending vector, callable on an array, giving bit-identical answers at a repeated parameter,
and safe to call from several threads on one object. lappy's `_tensions_scalar` reached those
properties through a 370-line instance cache with a lock and a global byte ledger; here the
objects are immutable and the properties follow.

Every stage is reachable as a free function on raw arrays. The pipeline is ill-conditioned by
nature, so being able to inspect one stage without standing up a solver is not a convenience
-- it is how the layer gets debugged. The free function holds the algorithm and the method
holds the defaults, and this file checks the two cannot drift apart.

Everything in `mps` speaks `lam`, and the `lam = k**2` seam sits at the `mps`/`helmholtz`
boundary. `MPSPencil` is domain-free and `MPSEigensolver` owns the domain, but they no longer
differ in their spectral parameter -- and since the solver HOLDS a pencil and exposes it as a
property, a user descending from one to the other must not have to change variables to do it.
The one function here that still takes `k` is `assemble`, because it calls into `helmholtz`,
where `k` is what a basis function is made of rather than a parametrisation choice.
"""
import concurrent.futures as cf

import numpy as np
import pytest

from chladni.eigensolver import Eigensolver
from chladni.geometry import Polygon
from chladni.helmholtz import FourierBessel, Fundamental
from chladni.mps import MPSEigensolver, MPSPencil, assemble, coef, regularize, tensions

from conftest import EXACT, L_SHAPE

K = 3.3
LAM = K**2


@pytest.fixture
def poly():
    return Polygon(L_SHAPE, bc="dir")


@pytest.fixture
def basis(poly):
    fb = sum((FourierBessel.at_corner(c, n=3, domain=poly) for c in poly.corners[1:]),
             FourierBessel.at_corner(poly.corners[0], n=3, domain=poly))
    return fb + Fundamental(np.array([-1 - 1j, 3 + 3j]), radius=1.0)


@pytest.fixture
def pencil(poly, basis):
    return MPSPencil.from_basis(basis, poly.bdry_quad(300),
                                poly.interior_pts(120, rng=np.random.default_rng(0)),
                                lam_ref=LAM)


@pytest.fixture
def solver(poly, basis):
    return MPSEigensolver(poly, basis=basis, n_bdry=300, n_int=120,
                          rng=np.random.default_rng(0), lam_ref=LAM)


# --- the tension function is a function --------------------------------------------------


def test_tensions_ascend(pencil):
    t = pencil.tensions(LAM)
    assert t.ndim == 1 and len(t) == pencil.rank
    assert np.all(np.diff(t) >= 0)
    assert t[0] == pencil.tension(LAM)


def test_tensions_are_deterministic(pencil):
    """Not merely close: identical. A minimizer comparing a value against one it computed
    three iterations ago must not see the difference between two runs of the same LAPACK
    call."""
    assert np.array_equal(pencil.tensions(LAM), pencil.tensions(LAM))


def test_tensions_take_an_array(solver):
    lams = np.linspace(8.0, 30.0, 17)
    got = solver.tensions(lams)
    assert got.shape == (len(lams), solver.rank)
    assert np.allclose(got[3], solver.tensions(lams[3]), rtol=0, atol=EXACT)


def test_a_solver_is_safe_to_share_across_threads(solver):
    """`douse` runs solves in a thread pool over generated domains, so the object has to be
    holdable. It is immutable, which is the whole argument -- this checks the argument."""
    lams = np.linspace(8.0, 30.0, 24)
    serial = [solver.tension(l) for l in lams]
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        parallel = list(ex.map(solver.tension, lams))
    assert np.array_equal(np.array(serial), np.array(parallel))


# --- every stage is a free function ------------------------------------------------------


def test_the_free_functions_and_the_methods_agree(pencil):
    """The rule: the free function holds the algorithm, the method holds the defaults. If
    these ever disagree, logic has leaked into a class where it cannot be inspected."""
    A_B, A_I = assemble(pencil.bdry, pencil.interior, np.sqrt(LAM))
    assert np.array_equal((A_B, A_I)[0], pencil.matrices(LAM)[0])

    B, I = pencil.pencil(LAM)
    assert np.array_equal(tensions(B, I, reg=pencil.reg, rank=pencil.rank),
                          pencil.tensions(LAM))
    assert np.allclose(coef(B, I, mult=1, reg=pencil.reg, rank=pencil.rank),
                       pencil.coef(LAM), rtol=0, atol=EXACT)


def test_the_stages_compose_by_hand(pencil):
    """Assemble, equilibrate, regularize, GSVD -- driven one call at a time on raw arrays,
    with no solver involved past the first line. This is the debugging path."""
    from chladni.mps import equilibrate

    A_B, A_I = pencil.matrices(LAM)
    B, I, _ = equilibrate(A_B, A_I)
    r = regularize(B, I, reg="svd", rank=pencil.rank)
    assert r.B.shape[1] == pencil.rank
    assert np.allclose(tensions(B, I, reg="svd", rank=pencil.rank),
                       pencil.tensions(LAM), rtol=0, atol=EXACT)


def test_a_hand_built_pair_of_arrays_needs_no_geometry():
    """The bespoke path at its most extreme: two matrices from nowhere. Nothing in the
    pipeline below the classes may require a Domain, a Quadrature or a basis."""
    rng = np.random.default_rng(1)
    A_B, A_I = rng.standard_normal((30, 8)), rng.standard_normal((20, 8))
    t = tensions(A_B, A_I, reg="svd", rank=8)
    assert t.ndim == 1 and np.all(np.diff(t) >= 0)


# --- the two classes, and their parameters -----------------------------------------------


def test_everything_in_mps_speaks_lam(pencil, solver):
    """`conventions.md`: a parameter's NAME states its variable. The seam is the
    `mps`/`helmholtz` boundary and nothing finer, so both classes take `lam` and agree at
    the SAME argument -- not at corresponding ones.

    The earlier arrangement had the pencil take `k` and the solver `lam`, which put the
    conversion inside one layer that a user walks through: `solver.pencil` is a public
    property, and the variable changed across the hop for no reason visible in the numbers.
    """
    import inspect

    for meth in (MPSPencil.tension, MPSPencil.tensions, MPSPencil.coef, MPSPencil.rank_at,
                 MPSEigensolver.tension, MPSEigensolver.tensions, MPSEigensolver.coef):
        params = inspect.signature(meth).parameters
        assert "lam" in params and "k" not in params, meth.__qualname__

    assert solver.tension(LAM) == pytest.approx(solver.pencil.tension(LAM), rel=1e-12)


def test_assemble_is_where_lam_becomes_k(pencil):
    """The seam itself, as an assertion. `assemble` is the call into `helmholtz`, so it is
    the last function that takes `k`; `matrices` is its `lam`-facing wrapper and the two must
    name the same matrices."""
    import inspect

    assert "k" in inspect.signature(assemble).parameters
    assert np.array_equal(assemble(pencil.bdry, pencil.interior, np.sqrt(LAM))[0],
                          pencil.matrices(LAM)[0])


def test_the_pencil_never_sees_a_domain(pencil):
    """The answer to `mps.md`'s second design question, as an assertion. lappy's severed
    solver still had to be handed a domain for boundary conditions and orthonormalization;
    this one genuinely does not have one."""
    from chladni.geometry import Domain

    for name in vars(pencil):
        assert not isinstance(getattr(pencil, name), Domain), f"MPSPencil.{name} is a Domain"


def test_the_solver_is_an_eigensolver(solver):
    assert isinstance(solver, Eigensolver)
    assert solver.bc_type == "dir"


def test_the_search_is_absent_and_says_so(solver):
    """One honest stub, against lappy's three `pass`-bodied `default_*` methods. The message
    has to name what is missing, or the next person reads it as a bug."""
    with pytest.raises(NotImplementedError, match="search"):
        solver.solve_interval(8.0, 30.0)
    with pytest.raises(NotImplementedError, match="search"):
        solver.solve(4)


# --- the solver's own construction -------------------------------------------------------


def test_collocation_may_be_fixed_or_budgeted(poly, basis):
    """"With the option to fix the basis and/or the collocation points" -- both spellings
    must reach the same pencil."""
    q = poly.bdry_quad(300)
    pts = poly.interior_pts(120, rng=np.random.default_rng(0))
    fixed = MPSEigensolver(poly, basis=basis, bdry_quad=q, int_pts=pts, lam_ref=LAM)
    budgeted = MPSEigensolver(poly, basis=basis, n_bdry=300, n_int=120,
                              rng=np.random.default_rng(0), lam_ref=LAM)
    assert np.array_equal(fixed.tensions(LAM), budgeted.tensions(LAM))


def test_interior_points_are_seeded(poly, basis):
    """lappy's unseeded default made `iso_right_tri` span 2.5 to 5.8 certified digits across
    draws. Two solvers built the same way must be the same solver."""
    a = MPSEigensolver(poly, basis=basis, n_bdry=200, n_int=80,
                       rng=np.random.default_rng(3), lam_ref=LAM)
    b = MPSEigensolver(poly, basis=basis, n_bdry=200, n_int=80,
                       rng=np.random.default_rng(3), lam_ref=LAM)
    assert np.array_equal(a.tensions(LAM), b.tensions(LAM))


def test_the_basis_is_required_while_planning_is_deferred(poly):
    """Basis planning is deferred in `particular_solutions.md`. A placeholder default would
    become load-bearing the way lappy's did, so there is none, and the error says why."""
    with pytest.raises(TypeError, match="basis"):
        MPSEigensolver(poly, n_bdry=200, n_int=80)
