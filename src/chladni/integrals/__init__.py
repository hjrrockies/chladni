"""Boundary integrals: Rellich norms and Gram matrices, Moler-Payne bounds, and (later) Hadamard
shape derivatives, all from Cauchy data on corner-adapted boundary quadrature.

Design and milestones: `docs/design/boundary_integrals.md`. Depends on `geometry` and
`helmholtz`; `mps` depends on this, never the reverse. Built so far: `rules` (M1).
"""
from . import rules

__all__ = ["rules"]
