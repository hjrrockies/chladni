"""Deterministic cost counters for before/after benchmarks.  (2026-11-03)

Wall time on a shared machine is noise (`notes/accuracy_audit_2026-11-03.md`); these count the
work instead. `install()` wraps, once per process:

  assemblies   `MPSPencil.matrices` -- one basis evaluation at one lam, the unit every search,
               refine and planner step pays (it is never cached);
  cauchy_nodes the quadrature nodes `integrals.cauchy.cauchy_data` evaluated -- certification,
               normalization and Hadamard work.

`snapshot()` returns the running totals; take the difference around a solve.
"""
_COUNTS = dict(assemblies=0, cauchy_calls=0, cauchy_nodes=0)
_INSTALLED = False


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    from chladni.integrals import cauchy
    from chladni.mps import pencil

    matrices = pencil.MPSPencil.matrices

    def counted_matrices(self, lam):
        _COUNTS["assemblies"] += 1
        return matrices(self, lam)
    pencil.MPSPencil.matrices = counted_matrices

    cd = cauchy.cauchy_data

    def counted_cd(family, coef, lam, quad):
        _COUNTS["cauchy_calls"] += 1
        _COUNTS["cauchy_nodes"] += len(quad.pts)
        return cd(family, coef, lam, quad)
    cauchy.cauchy_data = counted_cd
    _INSTALLED = True


def snapshot():
    return dict(_COUNTS)


def since(before):
    return {k: _COUNTS[k] - before[k] for k in _COUNTS}
