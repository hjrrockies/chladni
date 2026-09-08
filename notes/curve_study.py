"""Study 1: how tension curves behave on real corner domains, as it bears on the search.

Reads the cached fine scans, refines every dip to get the FLOOR (the sampled minimum is
nowhere near it), and measures the four things the search actually turns on:

    the plateau           what tau_scan can be a fraction of
    the floor spread      what bounds achievable precision, per eigenvalue
    width / gap           what sets the scan stride
    recall vs stride      the same question, measured end to end instead of inferred

Exploratory; not tracked.
"""
import sys
import warnings

sys.path.insert(0, "notes")
sys.path.insert(0, "tests")
warnings.simplefilter("ignore")

import numpy as np
from scipy.optimize import minimize_scalar

import curve_lab as cl

CACHE = ("/private/tmp/claude-501/-Users-hringer-projects-python-chladni/"
         "50c64400-9259-42d4-80e9-db814c63727f/scratchpad")


def load(lbl):
    z = np.load(f"{CACHE}/scan_{lbl}.npz")
    return z["lams"], z["S"], float(z["area"]), float(z["gap"])


def local_min_idx(y):
    return np.nonzero((y[1:-1] < y[:-2]) & (y[1:-1] <= y[2:]))[0] + 1


def find_dips(lams, s1, frac=0.5):
    """Discrete minima that are dips. `frac` is relative to the PLATEAU, not the median."""
    plateau = np.median(s1)
    i = local_min_idx(s1)
    return i[s1[i] < frac * plateau], plateau


def refine(p, lams, i):
    """Brent on sigma_1**2 in the neighbouring-sample bracket. Returns (lam*, floor)."""
    lo, hi = lams[max(i - 1, 0)], lams[min(i + 1, len(lams) - 1)]
    r = minimize_scalar(lambda x: p.tension(x)**2, bounds=(lo, hi), method="bounded",
                        options={"xatol": 1e-13})
    return r.x, np.sqrt(max(r.fun, 0.0))


def half_width(lams, s1, i, plateau):
    """Half-width where sigma_1 crosses plateau/2, linearly interpolated."""
    lvl = 0.5 * plateau

    def cross(rng):
        prev = i
        for j in rng:
            if s1[j] >= lvl:
                t = (lvl - s1[prev]) / (s1[j] - s1[prev]) if s1[j] != s1[prev] else 0.0
                return lams[prev] + t * (lams[j] - lams[prev])
            prev = j
        return None

    L = cross(range(i - 1, -1, -1))
    Rt = cross(range(i + 1, len(s1)))
    if L is None or Rt is None:
        return np.nan
    return 0.5 * (Rt - L)


def stride_recall(lams, s1, dip_idx, frac, plateau):
    """Subsample at stride m; how many true dips survive as discrete minima?

    This is the scan-stride question asked directly. `per_gap` converts m to the units the
    design is stated in.
    """
    out = []
    for m in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48):
        sub_l, sub_s = lams[::m], s1[::m]
        if sub_s.size < 5:
            continue
        cand = local_min_idx(sub_s)
        cand = cand[sub_s[cand] < frac * plateau]
        found = sub_l[cand]
        # a true dip counts as found if some subsampled minimum is within one stride of it
        h = lams[m] - lams[0]
        hit = sum(1 for d in lams[dip_idx]
                  if found.size and np.min(np.abs(found - d)) <= h)
        out.append((m, h, hit / max(len(dip_idx), 1), len(found)))
    return out


def study(lbl, expr, win):
    lams, S, area, gap = load(lbl)
    s1 = S[:, 0]
    dip_idx, plateau = find_dips(lams, s1)
    dom = cl.domain(expr)
    p = cl.make(dom, win[0], win[1])

    locs, floors, widths = [], [], []
    for i in dip_idx:
        x, f = refine(p, lams, i)
        locs.append(x)
        floors.append(f)
        widths.append(half_width(lams, s1, i, plateau))
    locs, floors, widths = np.array(locs), np.array(floors), np.array(widths)
    obs_gap = np.diff(locs) if locs.size > 1 else np.array([np.nan])

    print(f"\n=== {lbl}   area {area:.3f}   Weyl gap {gap:.3f}   {len(dip_idx)} dips "
          f"in [{win[0]:g}, {win[1]:g}]")
    print(f"    plateau          {plateau:.4f}")
    print(f"    floor            min {np.nanmin(floors):.1e}  med "
          f"{np.nanmedian(floors):.1e}  max {np.nanmax(floors):.1e}   "
          f"SPREAD {np.nanmax(floors)/np.nanmin(floors):.1e}")
    print(f"    observed gap     min {np.nanmin(obs_gap):.3f}  med "
          f"{np.nanmedian(obs_gap):.3f}  max {np.nanmax(obs_gap):.3f}")
    print(f"    half-width/gap   min {np.nanmin(widths/np.nanmedian(obs_gap)):.3f}  "
          f"med {np.nanmedian(widths)/np.nanmedian(obs_gap):.3f}  "
          f"max {np.nanmax(widths/np.nanmedian(obs_gap)):.3f}")
    print(f"    stride:  {'per gap':>9} {'h':>8} {'recall':>7} {'n_found':>8}")
    for m, h, rec, nf in stride_recall(lams, s1, dip_idx, 0.5, plateau):
        print(f"             {np.nanmedian(obs_gap)/h:9.1f} {h:8.4f} {rec:7.3f} {nf:8d}")
    return dict(lbl=lbl, plateau=plateau, floors=floors, locs=locs, widths=widths,
                obs_gap=obs_gap, gap=gap)


if __name__ == "__main__":
    res = [study(lbl, expr, win) for lbl, expr, win, _ in cl.ROSTER]
    print("\n\n=== summary ===")
    print(f"{'domain':>9} {'plateau':>9} {'floor spread':>13} {'w/gap med':>10} "
          f"{'min gap/Weyl':>13}")
    for r in res:
        print(f"{r['lbl']:>9} {r['plateau']:9.4f} "
              f"{np.nanmax(r['floors'])/np.nanmin(r['floors']):13.1e} "
              f"{np.nanmedian(r['widths'])/np.nanmedian(r['obs_gap']):10.3f} "
              f"{np.nanmin(r['obs_gap'])/r['gap']:13.3f}")


# --- Study 2: can the Weyl count DETECT the misses a coarse stride causes? ----------------
#
# The stride table says six samples per gap is not enough on four of seven domains. A solver
# cannot know the local gap in advance -- Weyl gives the MEAN, and the minimum local gap runs
# 0.13 to 0.79 of it. So the stride has to be corrected in the loop, and the only signal
# available on a domain with no closed form is the three-term Weyl count. This asks whether
# that signal actually fires on the misses.

def weyl_vs_stride():
    from chladni.asymp import weyl_deficit
    print("\n\n=== Study 2: does the Weyl deficit see what a coarse stride misses?\n")
    print(f"{'domain':>9} {'per Weyl gap':>13} {'recall':>7} {'n_found':>8} "
          f"{'weyl deficit':>13} {'flags?':>7}")
    for lbl, expr, win, _ in cl.ROSTER:
        lams, S, area, gap = load(lbl)
        s1 = S[:, 0]
        dip_idx, plateau = find_dips(lams, s1)
        dom = cl.domain(expr)
        lam1 = lams[dip_idx[0]] if dip_idx.size else np.nan
        if win[0] > lam1:
            print(f"{lbl:>9}  window starts above lam_1; not a prefix, skipped")
            continue
        for m in (2, 4, 8, 12, 16, 24, 32):
            sub_l, sub_s = lams[::m], s1[::m]
            h = sub_l[1] - sub_l[0]
            cand = local_min_idx(sub_s)
            cand = cand[sub_s[cand] < 0.5 * plateau]
            found = sub_l[cand]
            hit = sum(1 for d in lams[dip_idx]
                      if found.size and np.min(np.abs(found - d)) <= h)
            rec = hit / max(len(dip_idx), 1)
            try:
                wd = float(weyl_deficit(np.sort(found), dom))
            except Exception:
                wd = np.nan
            flag = "" if np.isnan(wd) else ("MISS" if wd >= 0.5 else "ok")
            print(f"{lbl:>9} {gap/h:13.1f} {rec:7.3f} {len(found):8d} "
                  f"{wd:13.2f} {flag:>7}")
        print()


# --- Study 3: multiplicity on real corner domains, and the audit closed properly ----------
#
# Section 9.3 of the notes scored multiplicity 9 for 9 -- on the disk and the equilateral
# triangle, both smooth, both with plane-wave-ish bases. `H_shape` and `plus` have
# symmetry-protected doubles AND reentrant corners, which is the case the solver is for.
#
# Study 2 showed the Weyl deficit flagging MISS at recall 1.000 on four domains. That is not
# a false positive: sigma_1 shows one dip per LEVEL and Weyl counts WITH MULTIPLICITY, so the
# deficit was reporting exactly the multiplicity that had not been estimated yet. This closes
# the loop.

def multiplicity(p, lam, plateau, G=10.0, P=2.0, nb=6):
    """(C1) largest ratio gap, then (C2) each counted branch is a local min to prominence P.

    The wide bracket is taken at the scale where the higher branches have actually moved --
    a fixed fraction of the plateau's own width, not `eps/c`, because `eps` is unknown here.
    """
    s = p.tensions(lam)[:nb]
    s = s[np.isfinite(s)]
    if s.size < 2:
        return 1, np.nan
    r = s[1:] / np.maximum(s[:-1], 1e-300)
    j = int(np.argmax(r))                      # gap after index j
    kappa_c1 = j + 1 if r[j] >= G else s.size
    kappa = 0
    h = 0.02 * abs(lam)
    lo, hi = p.tensions(lam - h), p.tensions(lam + h)
    for i in range(min(kappa_c1, len(lo), len(hi))):
        if lo[i] > P * s[i] and hi[i] > P * s[i]:
            kappa += 1
        else:
            break
    return max(kappa, 1), r[j]


def multiplicity_audit():
    from chladni.asymp import weyl_deficit
    print("\n\n=== Study 3: multiplicity on corner domains, and the audit after it\n")
    print(f"{'domain':>9} {'dips':>5} {'sum mult':>9} {'mults':>26} "
          f"{'deficit dips':>13} {'deficit mult':>13}")
    for lbl, expr, win, _ in cl.ROSTER:
        lams, S, area, gap = load(lbl)
        s1 = S[:, 0]
        dip_idx, plateau = find_dips(lams, s1)
        dom = cl.domain(expr)
        p = cl.make(dom, win[0], win[1])
        locs, mults = [], []
        for i in dip_idx:
            x, _ = refine(p, lams, i)
            m, _ = multiplicity(p, x, plateau)
            locs.append(x)
            mults.append(m)
        locs, mults = np.array(locs), np.array(mults)
        unf = np.sort(np.repeat(locs, mults))
        try:
            d0 = float(weyl_deficit(np.sort(locs), dom))
        except Exception:
            d0 = np.nan
        try:
            d1 = float(weyl_deficit(unf, dom))
        except Exception:
            d1 = np.nan
        print(f"{lbl:>9} {len(locs):5d} {int(mults.sum()):9d} "
              f"{str(list(mults)):>26} {d0:13.2f} {d1:13.2f}")
