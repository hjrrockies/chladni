"""Rigorous inequalities on Laplacian eigenvalues.

Every function takes either a `Domain` or the raw scalar it needs, so a bound is available
before a domain object exists. When a `Domain` is given its boundary condition is checked;
a scalar caller gets no such protection and can silently apply a Dirichlet bound to a Neumann
problem, which is worth knowing.

All of these are theorems except `polya`, which is a conjecture in general and says so.
"""
from __future__ import annotations

import numpy as np
from scipy.special import jn_zeros, jnp_zeros

__all__ = ["faber_krahn", "szego_weinberger", "payne_polya_weinberger", "inradius_upper",
           "polya"]

#: First zero of `J_0`: the disk's first DIRICHLET mode. Faber-Krahn and the inradius bound.
_J01 = float(jn_zeros(0, 1)[0])
#: First zero of `J_1`, also Dirichlet: the disk's second mode, so `(J11/J01)**2` is its
#: `lam_2/lam_1` and the PPW constant.
_J11 = float(jn_zeros(1, 1)[0])
#: First zero of `J_1'`, the disk's first NONZERO NEUMANN mode. Szego-Weinberger needs this
#: and not `_J11`.
#:
#: lappy shares one constant between PPW and Szego-Weinberger. That is right for PPW and
#: wrong here: it makes the "bound" 14.682 against a true `mu_1` of 3.390 on the unit disk,
#: a factor of 4.3, so it is not an upper bound on the very domain it claims equality for.
#: Caught by testing the equality case rather than the inequality -- an inequality that is
#: too weak by 4x passes every test that only asks whether it is violated.
_J11P = float(jnp_zeros(1, 1)[0])


def _area(domain, area):
    if domain is not None:
        return float(domain.area)
    if area is not None:
        return float(area)
    raise ValueError("pass either a Domain or area=")


def _require_bc(domain, expected):
    if domain is not None and domain.bc_type != expected:
        raise ValueError(
            f"this bound is for bc_type={expected!r}; the domain has "
            f"{domain.bc_type!r}"
        )


def faber_krahn(domain=None, *, area=None):
    """`lam_1 >= pi * j01**2 / |Omega|`, Dirichlet. A LOWER bound.

    Equality if and only if Omega is a disk -- which is why `eigensolver.weyl_window` pads
    below it. An unpadded search window starting exactly at this value has its first
    eigenvalue on the endpoint, and a grid search that ignores endpoints then misses it.
    """
    _require_bc(domain, "dir")
    return np.pi * _J01**2 / _area(domain, area)


def szego_weinberger(domain=None, *, area=None):
    """`mu_1 <= pi * j'11**2 / |Omega|`, Neumann, `mu_1` the first NONZERO eigenvalue.

    An UPPER bound. Equality if and only if Omega is a disk.

    The constant is the first zero of `J_1'` (1.8412), NOT of `J_1` (3.8317) -- see `_J11P`.
    """
    _require_bc(domain, "neu")
    return np.pi * _J11P**2 / _area(domain, area)


def payne_polya_weinberger(domain=None):
    """`lam_2 / lam_1 <= (j11/j01)**2`, about 2.539, Dirichlet. An UPPER bound on the ratio.

    A pure constant -- conjectured by Payne, Polya and Weinberger, proved by Ashbaugh and
    Benguria. `domain` is accepted only so the boundary condition can be checked.
    """
    _require_bc(domain, "dir")
    return (_J11 / _J01) ** 2


def inradius_upper(domain=None, *, inradius=None):
    """`lam_1 <= j01**2 / R**2` with `R` the inradius, Dirichlet. An UPPER bound.

    Domain monotonicity: Omega contains a disk of radius `R`, so its first eigenvalue cannot
    exceed that disk's. Equality if and only if Omega is a disk.
    """
    _require_bc(domain, "dir")
    if domain is not None:
        R = float(domain.inradius)
    elif inradius is not None:
        R = float(inradius)
    else:
        raise ValueError("pass either a Domain or inradius=")
    return _J01**2 / R**2


def polya(k, domain=None, *, area=None):
    """`lam_k >= 4*pi*k / |Omega|`, Dirichlet. A LOWER bound, vectorized in `k`.

    **Proven for tiling domains; conjectured in general.** The only unproven statement in this
    module, and it is labelled rather than quietly presented alongside the theorems.
    """
    _require_bc(domain, "dir")
    return 4 * np.pi * np.asarray(k, dtype=float) / _area(domain, area)
