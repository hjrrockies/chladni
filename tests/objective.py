"""The scored seam -- a re-export of `chladni.mps.objective`.

TEST SUPPORT ONLY, and TEMPORARY. `Objective` and `Declined` moved into the package, because
`MPSEigensolver.solve_interval` needs them; this shim exists so that the promotion changed no
test file. It goes when the candidates are repointed at `chladni.mps`.

RE-EXPORTS, NEVER REDEFINES. `test_solve.py` does `isinstance(out, Declined)` on a value
produced on the far side of this boundary, so a second class object with the same name here
would make that check silently false.
"""
from chladni.mps.objective import Declined, Objective  # noqa: F401

__all__ = ["Objective", "Declined"]
