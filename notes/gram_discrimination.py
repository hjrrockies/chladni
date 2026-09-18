"""Does the interior Gram angle separate "one eigenvalue found twice" from "two eigenvalues"?

`notes/gram_angle.py` asked this on `rect(1+delta, 1)` and answered yes -- distinct <= 0.10,
duplicate >= 0.948 -- and `docs/design/search.md` took those numbers as Phase D's contract with
`tau_gram = 0.5`. That measurement carries a caveat recorded in `landscape_findings` section 15:
exact eigenvalues, ONE domain, one mode pair, low `lam`, and a domain whose modes are exactly
L2-orthogonal, which is what makes the distinct-pair floor pure sampling noise. The L-shape with
a corner-adapted basis was named as the case to check before anything relies on it.

This script checks it, on the whole roster, now that `MPSPencil.interior_image` exists.

Three tables:

    RECTANGLE    the (2,1)/(1,2) pair across six separations, distinct against same-well
    ROSTER       adjacent true eigenvalues on L_shape, disk and eq_tri -- the DISTINCT side
    DUPLICATE    two roots a MERGE LICENCE apart, which is the separation Phase D actually
                 sees, across the prec ladder

The third table is the one that matters and the one the earlier measurement did not make. A
probe at "two-thirds of a gap" -- `search.md`'s own phrasing -- is PAST THE MIDPOINT, where the
candidate has already turned into the neighbour's eigenfunction; that is not the duplicate case
and reading it as one makes the instrument look broken. Two located roots that are really the
same eigenvalue sit `prec*lam` apart, which is eight to fourteen orders closer.

Exploratory; the tables are `notes/landscape_findings.md` section 36 and the claims are pinned
by `tests/test_search_collapse.py`.

Run from `tests/`:  PYTHONPATH=. ../.venv/bin/python ../notes/gram_discrimination.py
"""
import warnings

import numpy as np

warnings.filterwarnings("ignore")

import hard_roster as hr                                         # noqa: E402
import rect_suite as rs                                          # noqa: E402

TAU_GRAM = 0.5
PRECS = (1e-6, 1e-8, 1e-10, 1e-12)


def gram(p, a, b):
    """`|<u(a), u(b)>|` on the pencil's own interior points."""
    return abs(float(p.interior_image(float(a)) @ p.interior_image(float(b))))


def rectangle_pair():
    """The case section 15 measured, re-measured through `interior_image`."""
    print("=== RECTANGLE: the (2,1)/(1,2) pair on rect(1+delta, 1) ===")
    print(f"{'case':>14} {'gap d':>10} {'DISTINCT':>9}   same well, at lam_i - f*d")
    for case in rs.family_B():
        t = np.unique(rs.complete_truth(case))
        near = t[np.argsort(abs(t - 5 * np.pi**2))[:2]]
        if near.size < 2:
            continue
        la, lb = float(min(near)), float(max(near))
        p, d = case.pencil, lb - la
        same = "  ".join(f"{f:.2f}:{gram(p, la, la - f * d):.4f}" for f in (0.33, 0.67, 1.0))
        print(f"{case.label:>14} {d:10.2e} {gram(p, la, lb):9.4f}   {same}")


def roster_distinct():
    """Adjacent TRUE eigenvalues: the distinct population, on three real domains."""
    print("\n=== ROSTER: genuinely distinct eigenvalues ===")
    worst = {}
    for name in ("L_shape", "disk", "eq_tri"):
        case = hr.ROSTER[name]()
        p = case.pencil
        t = np.unique(np.asarray(case.truth, dtype=float))
        t = t[(t > case.a) & (t < case.b)]
        print(f"\n  {name}: n_int={p.n_int}, sampling floor 1/sqrt(n_int)"
              f"={1 / np.sqrt(p.n_int):.3f}")
        print(f"  {'lam_i':>12} {'lam_j':>12} {'gap':>9} {'DISTINCT':>9}")
        vals = []
        for i in range(min(len(t) - 1, 6)):
            la, lb = float(t[i]), float(t[i + 1])
            v = gram(p, la, lb)
            vals.append(v)
            print(f"  {la:12.5f} {lb:12.5f} {lb - la:9.2e} {v:9.4f}")
        worst[name] = max(vals)
    print(f"\n  worst distinct reading per domain: "
          + ", ".join(f"{k} {v:.4f}" for k, v in worst.items()))
    return max(worst.values())


def roster_duplicate():
    """Two roots a MERGE LICENCE apart: the separation Phase D actually sees."""
    print("\n=== ROSTER: one eigenvalue found twice, at prec*lam apart ===")
    print(f"{'domain':>9} {'lam':>11} | " + "  ".join(f"{p:>8.0e}" for p in PRECS))
    vals = []
    for name in ("L_shape", "disk", "eq_tri"):
        case = hr.ROSTER[name]()
        p = case.pencil
        t = np.unique(np.asarray(case.truth, dtype=float))
        t = t[(t > case.a) & (t < case.b)]
        for la in t[:3]:
            la = float(la)
            row = [gram(p, la - 0.5 * pr * la, la + 0.5 * pr * la) for pr in PRECS]
            vals += row
            print(f"{name:>9} {la:11.5f} | " + "  ".join(f"{v:8.4f}" for v in row))
    return min(vals)


def midpoint_trap():
    """Why `search.md`'s "two-thirds of a gap" must not be read as the duplicate case."""
    print("\n=== the midpoint trap ===")
    case = hr.ROSTER["L_shape"]()
    p = case.pencil
    t = np.unique(np.asarray(case.truth, dtype=float))
    t = t[(t > case.a) & (t < case.b)]
    la, lb = float(t[0]), float(t[1])
    print(f"  L_shape, lam_i={la:.5f}, lam_j={lb:.5f}, gap={lb - la:.3e}")
    for f in (0.001, 0.01, 0.1, 0.33, 0.5, 0.67, 0.9):
        print(f"    |cos(u(lam_i), u(lam_i + {f:5.3f}*gap))| = {gram(p, la, la + f * (lb - la)):.4f}"
              + ("   <- past the midpoint: this is the NEIGHBOUR's mode" if f > 0.5 else ""))


if __name__ == "__main__":
    rectangle_pair()
    hi = roster_distinct()
    lo = roster_duplicate()
    midpoint_trap()
    print(f"\nSEPARATION: worst distinct {hi:.4f}  <  tau_gram {TAU_GRAM}  <  "
          f"weakest duplicate {lo:.4f}")
