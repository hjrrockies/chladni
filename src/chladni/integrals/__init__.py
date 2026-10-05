"""Boundary integrals: Rellich norms and Gram matrices, Moler-Payne bounds, and (later) Hadamard
shape derivatives, all from Cauchy data on corner-adapted boundary quadrature.

Design and milestones: `docs/design/boundary_integrals.md`. Depends on `geometry` and
`helmholtz`; `mps` depends on this, never the reverse. Built so far: `rules` (M1); `plan`,
`cauchy`, `identities` (M2).
"""
from . import rules
from .cauchy import CauchyData, cauchy_data
from .identities import gram, norms, weighted_integral
from .plan import CornerSpec, IntegralQuad, integral_prec, integral_quad

__all__ = ["rules", "CauchyData", "cauchy_data", "gram", "norms", "weighted_integral",
           "CornerSpec", "IntegralQuad", "integral_prec", "integral_quad"]
