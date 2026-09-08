"""Stage 3 on `candidates.md`'s scored seam -- a re-export of `chladni.mps.multiplicity`.

TEST SUPPORT ONLY, and TEMPORARY. The implementation moved into the package; this shim exists
so that the promotion changed no test file, which is what makes an unchanged suite the proof
that the move was a move. It goes when the candidates are repointed at `chladni.mps`.

`test_solve.py` and `test_multiplicity.py` bind this module as `M` and read `M.GAP_RATIO`,
`M.PROMINENCE` and `M.WIDE_FACTOR` off it, so the constants are carried across as well as the
callables. Note what that makes them: COPIES of the bindings. Nothing assigns to them today --
checked -- but a future ablation that sets `M.WIDE_FACTOR` would move this module's copy and
not the one the package reads. Patch `chladni.mps.multiplicity` instead, or wait for this file
to be deleted.
"""
from chladni.mps.multiplicity import (  # noqa: F401
    GAP_RATIO,
    MAX_BRANCHES,
    PROMINENCE,
    WIDE_FACTOR,
    Multiplicity,
    multiplicity,
    wide_bracket_for,
)

__all__ = ["multiplicity", "wide_bracket_for", "Multiplicity", "GAP_RATIO", "PROMINENCE",
           "WIDE_FACTOR", "MAX_BRANCHES"]
