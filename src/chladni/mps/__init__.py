"""The GSVD formulation of the Method of Particular Solutions.

Two objects, split by what they know. `MPSPencil` is a basis and two collocation point sets,
domain-free, parameterized by `k`. `MPSEigensolver` adds the domain and speaks `lam`.

Every stage is also a free function on raw arrays -- `assemble`, `equilibrate`, `choose_rank`,
`regularize`, `tensions`, `coef` -- because this pipeline is ill-conditioned by nature and
inspecting one stage without standing up a solver is how it gets debugged. The rule, stated
once: the free function holds the algorithm, the method holds the defaults.

The eigenvalue search and multiplicity estimation are deferred; so is Rellich orthonormaliza-
tion. See `docs/design/mps.md`.
"""
from .._core import PREC, PrecisionWarning
from ..eigensolver import Eigensolver, k_to_lam, lam_to_k, weyl_window
from ._gsvd import coef, coef_of, tensions, tensions_of
from ._regularize import RegularizedPencil, choose_rank, regularize
from .calibrate import (RtolCalibration, calibrate_rtol, findability_target,
                        required_dk, sigma_base)
from .pencil import MPSPencil, RankDiagnostics, assemble, equilibrate
# A MODULE NAMED AFTER ITS VERB DOES NOT RE-EXPORT THE VERB. Importing `refine` or
# `multiplicity` here would rebind the package attribute from the submodule to the function, so
# a later `from . import refine as R` inside the package would bind the function and fail on
# first attribute access. The earlier promotion attempt shipped exactly that bug for
# `multiplicity` and hit it the moment `solve_interval` called it. Result types and the
# helpers that are not the verb; reach a verb through its module.
from .bracket import Bracket, SearchDiagnostics, SearchResult, bracket_interval
from .multiplicity import Multiplicity, wide_bracket_for
from .objective import Declined, Objective
from .refine import RefineResult
# `solve` is safe to export where `refine` and `multiplicity` are not: the module is `search`,
# so the function shadows no submodule.
from .search import Level, SolveDiagnostics, SolveResult, solve
from .solver import MPSEigensolver

__all__ = [
    "Eigensolver", "MPSEigensolver", "MPSPencil",
    "assemble", "equilibrate", "choose_rank", "regularize", "RegularizedPencil",
    "tensions", "tensions_of", "coef", "coef_of", "RankDiagnostics",
    "RefineResult", "Multiplicity", "wide_bracket_for",
    "Bracket", "SearchResult", "SearchDiagnostics", "bracket_interval",
    "Level", "SolveResult", "SolveDiagnostics", "solve",
    "Objective", "Declined",
    "lam_to_k", "k_to_lam", "weyl_window",
    "sigma_base", "findability_target", "required_dk", "calibrate_rtol", "RtolCalibration",
    "PREC", "PrecisionWarning",
]
