"""Tabulated eigenvalues: other people's numbers, with their accuracy attached.

None of these has a closed form. Every value came from an independent MPS implementation
(`lappy`), which is why they are admissible as reference here -- `conventions.md` forbids
regenerating ground truth from `chladni`, and these do not come from `chladni`.

What they are NOT is uniformly good. They run from 14 digits to under 3, and one entry is
known to be wrong and shipped anyway. Read `fn.digits` before choosing a tolerance and
`fn.notes` before trusting an individual value.
"""
from __future__ import annotations

import numpy as np

from ._core import _check_k, tabulated

__all__ = ["gww_eigs", "L_shape_eigs", "ellipse_eigs", "reg_ngon_eigs", "cut_square_eigs",
           "mushroom_eigs", "H_shape_eigs", "chevron_eigs", "iso_tri_eigs"]

_LAPPY = "lappy MPS solve; see lappy benchmarks/reference/ and TUNING_LOG.md"


def _cut(name, k, table, limit):
    k = _check_k(k)
    if k > limit:
        raise ValueError(f"only the first {limit} eigenvalues are available for {name}")
    return np.asarray(table, dtype=float)[:k]


@tabulated(12.0, _LAPPY + " (Driscoll's table; GWW1 and GWW2 solved separately at n_basis=320)",
    settings="mixed Fourier-Bessel + fundamental-solution basis, n_basis=320; only the 120- and 240-degree corners are singular")
def gww_eigs(k):
    """The first 25 Dirichlet eigenvalues shared by the Gordon-Webb-Wolpert isospectral pair.

    The whole point of the pair is that these are the same for both drums, so one table
    serves `GWW1()` and `GWW2()`.
    """
    return _cut("gww", k, np.sort([
        2.53794399980, 9.20929499840, 14.3138624643, 20.8823950433, 24.6740110027,
        3.65550971352, 10.5969856913, 15.871302620, 21.2480051774, 26.0802400997,
        5.17555935622, 11.5413953956, 16.9417516880, 22.2328517930, 27.3040189211,
        6.53755744376, 12.3370055014, 17.6651184368, 23.7112974848, 28.1751285815,
        7.24807786256, 13.0536540557, 18.9810673877, 24.4792340693, 29.5697729132]), 25)


@tabulated(14.0, _LAPPY + " (n_basis=240; agrees to ~1e-13 over the first 10)",
    settings="PURE Fourier-Bessel at the single reentrant corner -- no fundamental "
             "solutions, no mixed basis, against lappy's n_basis=240. Reproduced in "
             "chladni with 1400 boundary and 500 interior collocation points; tension at "
             "the first five tabulated eigenvalues: 1e-08 at 40 terms, 4e-13 to 2e-12 at "
             "60, and 2e-15 to 1e-14 at 80. So 80 terms clears the table's 14 digits and "
             "60 reaches about 12.5. Note lam_3 = 2*pi**2 hits 3e-15 by 40 terms alone, "
             "because it has NO SINGULAR PART at the corner -- not because it vanishes "
             "there, which every Dirichlet eigenfunction does. With omega = 3*pi/2 the "
             "corner expansion runs in r**(2*l/3), singular unless l is a multiple of 3, "
             "and this mode has zero amplitude on all the rest. It is simply the UNIT "
             "SQUARE'S first eigenvalue, pi**2*(1+1), tiling: L_shape is three unit cells "
             "on the integer grid, so sin(pi*x)*sin(pi*y) is an exact eigenfunction of it "
             "and is analytic everywhere -- see polyomino_eig and rect_eigs(1, 1, 1)")
def L_shape_eigs(k):
    """The first 25 Dirichlet eigenvalues of `L_shape()`, the standard corner benchmark.

    Reproduced independently by `chladni` to between 0.0 and 1.9e-10 relative over the first
    five -- recorded in `docs/design/reference.md`, not automated as a test.
    """
    return _cut("L_shape", k, [
        9.639723844021946, 15.197251926454308, 19.739208802178716,
        29.521481114144848, 31.912635957137752, 41.47450989021491,
        44.94848778135119, 49.34802200544678, 49.34802200544678,
        56.70960988738507, 65.37653570984583, 71.05775564851349,
        71.57267968033655, 78.95683520871486, 89.30166835196012,
        92.3069067630492, 97.38072264602184, 98.69604401089357,
        98.69604401089357, 101.60529408377867, 112.36860922562566,
        115.5201730946677, 128.30485721416164, 128.30485721416164,
        130.11902885096785], 25)


@tabulated(
    6.7, _LAPPY + " (a=2 at n_basis=240; a=3,4 from an earlier, less careful pass)",
    digits_by={(2.0, 1.0): 13.3, (3.0, 1.0): 6.7, (4.0, 1.0): 7.0},
    notes={
        (2.0, 1.0): {0: "CORRECTED relative to lappy, which ships 3.566726599853406 and "
                        "documents it as disputed. Three of lappy's own independent solves "
                        "(n_basis 160/240/320, tensions 3.5e-15 to 4.8e-15) give "
                        "3.56672660292886x, agreeing with each other to 7.5e-16 and "
                        "differing from its table by 8.62e-10 relative, while lam_2..4 agree "
                        "to 2.7e-13-4.4e-12 -- so it is that one entry and not the pipeline "
                        "behind it. Corroborated by the MPS tension at the tabulated value "
                        "sitting at 8.46e-10 for every basis tried, which is the signature "
                        "of a displaced reference rather than a basis limit. lappy left the "
                        "value unchanged deliberately; chladni takes the corroborated one, "
                        "because shipping a number known to be wrong is worse than shipping "
                        "the better of two that are already public."},
        (3.0, 1.0): {None: "not re-verified; about 6.7 digits"},
        (4.0, 1.0): {None: "not re-verified; at least 7 digits"},
    },
    settings="pure boundary fundamental-solution basis, n_basis=240; the ellipse has no corners at all. n_basis=480 was tried and did not improve it")
def ellipse_eigs(k, a=2.0, b=1.0):
    """The first 10 Dirichlet eigenvalues of `ellipse(a, b)`, for `(a, b)` in
    `{(2,1), (3,1), (4,1)}`.

    `digits` is 6.7, the worst case across the three parameter sets. The `a=2` set alone is
    13.3-14.4 digits EXCEPT its first entry, which is disputed -- see `notes`.
    """
    tbl = {
        (2.0, 1.0): [3.566726602928861, 6.275430620157517, 10.028401620452271,
                     11.736665434133736, 14.877304135375340, 15.923963976853598,
                     20.846866237831982, 21.023029232908417, 24.885731654879255,
                     27.080637209915011],
        (3.0, 1.0): [3.108128700342726, 4.595055082905808, 6.509341614142352,
                     8.881928158123285, 11.030611158910741, 11.734242285459748,
                     13.539805518163716, 15.080890472747482, 16.430451193912770,
                     18.931804562846718],
        (4.0, 1.0): [2.920251003362351, 3.934862295633269, 5.175723543179423,
                     6.659014663915840, 8.397619666159422, 10.401610755766617,
                     10.714369529583474, 12.505571121120370, 12.678776996497007,
                     14.503899387838459],
    }.get((float(a), float(b)))
    if tbl is None:
        raise ValueError(f"no table for (a, b) = ({a}, {b}); have (2,1), (3,1), (4,1)")
    return _cut("ellipse", k, tbl, 10)


@tabulated(
    2.8, _LAPPY + " (n_basis=120; every corner is singular for these N)",
    digits_by={5: 10.2, 6: 10.2, 7: 10.2, 8: 2.8},
    notes={8: {3: "about 2.8 digits: a genuinely under-resolved mode at n_basis=120, a near "
                  "neighbour of the doubled eigenvalue at 29.540564727244991"}},
    settings="mixed Fourier-Bessel + fundamental-solution, fs_frac=0.5, n_basis=120 (N=5) to 240 (N=6); every corner is singular for these N")
def reg_ngon_eigs(k, N):
    """The first 10 Dirichlet eigenvalues of `reg_ngon(N)` at UNIT CIRCUMRADIUS, `N = 5..8`.

    `digits` is 2.8, which is the single bad entry at `N = 8`; N=5,6,7 are 10.2 digits and
    the rest of N=8 is 8.8. Consult `notes` rather than applying 2.8 across the board.
    """
    tbl = {
        5: [7.957089389349433, 20.106281644908393, 20.106281644908393,
            35.654694217826744, 35.654694217826744, 41.313935812799116,
            55.700798288206641, 55.700798288206641, 64.521882950899112,
            64.521882950899112],
        6: [7.155339133926017, 18.131677865530712, 18.131677865530712,
            32.451857514400388, 32.451857514400388, 37.491352876797706,
            47.629365773857295, 52.637890139143394, 60.105112094166572,
            60.105112094166572],
        7: [6.735099196687083, 17.083807204748648, 17.083807204748648,
            30.643654921220552, 30.643654921220552, 35.397032810525708,
            47.100728432751367, 47.100728432751367, 57.017683247909424,
            57.017683247909424],
        8: [6.484933493724240, 16.456119030336588, 16.456119030336588,
            29.536810903327243, 29.540564727244991, 29.540564727244991,
            34.124475285748161, 45.529751487885697, 45.529751487885697,
            55.049743550878873],
    }.get(int(N))
    if tbl is None:
        raise ValueError(f"no table for N = {N}; have 5, 6, 7, 8")
    return _cut("reg_ngon", k, tbl, 10)


@tabulated(6.4, _LAPPY + " (n_basis=320; pure Fourier-Bessel, the other corners are right angles)",
           digits_by={0.25: 6.4, 0.5: 9.1},
           notes={0.25: {None: "about 6.4 digits"}, 0.5: {None: "about 9.1 digits"}},
    settings="pure Fourier-Bessel, n_basis=320; the arc leaves no singular corner, so the three right angles are all that is needed")
def cut_square_eigs(k, r=0.25):
    """The first 10 Dirichlet eigenvalues of `cut_square(r)`, for `r` in `{0.25, 0.5}`."""
    tbl = {
        0.25: [20.585592337197024, 49.422034168078973, 53.607796386908106,
               84.371488667211239, 99.437928066752903, 104.562833789125236,
               129.187535527364219, 146.994370017546117, 170.060990914169764,
               170.553669971897790],
        0.5: [28.081015645274604, 55.146340053352930, 72.407039289994984,
              100.899496342046007, 115.233911594759078, 139.429979543240051,
              153.042769322697268, 182.635738771611955, 191.634273757009765,
              202.708585647789022],
    }.get(float(r))
    if tbl is None:
        raise ValueError(f"no table for r = {r}; have 0.25, 0.5")
    return _cut("cut_square", k, tbl, 10)


@tabulated(11.3, _LAPPY + " (mixed Fourier-Bessel and fundamental-solution basis)",
    settings="mixed Fourier-Bessel + fundamental-solution, n_basis=240; two singular corners at the cap-stem junction")
def mushroom_eigs(k, a=1.0, b=1.0, r=1.5):
    """The first 10 Dirichlet eigenvalues of `mushroom(1, 1, 1.5)`, the only triple tabulated.

    Two 270-degree reentrant corners at the cap-stem junction.
    """
    if (float(a), float(b), float(r)) != (1.0, 1.0, 1.5):
        raise ValueError(f"no table for (a, b, r) = ({a}, {b}, {r}); have (1, 1, 1.5)")
    return _cut("mushroom", k, [
        5.497868889097452, 11.507908981960103, 13.363962538819910,
        18.067786790669235, 20.805793683510373, 25.550152545706617,
        29.124676104888923, 32.589926048677256, 34.194889649910934,
        41.911982648353757], 10)


@tabulated(
    7.8, _LAPPY + " (n_basis=320; four reentrant 270-degree corners), except index 6",
    notes={(): {6: "EXACT, and replaced by the closed form. lappy noted its tabulated value "
                   "19.739208802178766 was 'suspiciously close to 2*pi**2, likely a mode "
                   "that vanishes on the connecting web'. The value is right and the reason "
                   "is sharper: H_shape is a polyomino of seven unit cells, so "
                   "sin(pi*x)*sin(pi*y) vanishes on the WHOLE integer grid and is an exact "
                   "eigenfunction with lam = 2*pi**2 -- see polyomino_eig, which predicts "
                   "this entry and L_shape's third from a closed form."}},
    settings="mixed Fourier-Bessel + fundamental-solution, n_basis=320; four reentrant corners, the hardest of the non-excluded set")
def H_shape_eigs(k):
    """The first 10 Dirichlet eigenvalues of `H_shape()`. Keyed in `notes` by the empty tuple,
    since the shape takes no parameters."""
    return _cut("H_shape", k, [
        7.733088853276283, 8.551726848574045, 13.927633223859791,
        13.931597881917137, 14.305229961336140, 17.706735223095627,
        2 * np.pi**2, 24.788709817219047, 26.370982345700579,
        26.423167326708011], 10)


@tabulated(
    3.0, _LAPPY + " (h1=1,h2=2 from an earlier independent table; the rest at n_basis=160)",
    digits_by={(1.0, 2.0): 12.0, (1.0, 1.5): 5.0, (2.0, 3.0): 3.0, (2.0, 4.0): 3.0},
    notes={(1.0, 2.0): {None: "12 digits: an earlier, independently-derived table"},
           (1.0, 1.5): {None: "about 5-7 digits"},
           (2.0, 3.0): {None: "about 3-4 digits: the sharpest corner is near 11 degrees and "
                              "genuinely needs far more basis than is practical"},
           (2.0, 4.0): {None: "about 3-4 digits, same cause"}},
    settings="mixed Fourier-Bessel + fundamental-solution, n_basis=160 (120 for the FB-order study); the sharpest corner reaches ~11 degrees and dominates the basis at 75%")
def chevron_eigs(k, h1=1.0, h2=2.0):
    """The first 10 Dirichlet eigenvalues of `chevron(h1, h2)`.

    `digits` is 3.0, the worst of the four pairs. Only `(1, 2)` is good; consult `notes`.
    `(1, 1.25)` was excluded outright as too hard.
    """
    tbl = {
        (1.0, 2.0): [39.66587536762846, 77.66316267381548, 81.88608149069968,
                     111.42970385691103, 120.59489370950362, 152.06601346806502,
                     161.16007983417921, 179.80395817996902, 204.7047973867004,
                     205.98199724200455],
        (1.0, 1.5): [113.734766669427927, 189.448519906295729, 214.171972707017147,
                     272.053387695407821, 283.244291179057882, 348.908624511603080,
                     355.061155405508714, 428.693725570045387, 438.828078645126823,
                     518.577130919274964],
        (2.0, 3.0): [64.708985605874176, 121.539374925198445, 130.253934421114678,
                     155.535111621981400, 173.830602900292064, 194.130498772824097,
                     214.340324596339258, 226.620355923715977, 253.734091427685115,
                     263.032759596165249],
        (2.0, 4.0): [24.427905596255293, 43.891947613356443, 56.871718718195481,
                     63.509810642063812, 77.629441744211917, 79.004363501607074,
                     96.362085037989488, 98.349053725219605, 112.019515962249514,
                     139.714706538389208],
    }.get((float(h1), float(h2)))
    if tbl is None:
        raise ValueError(f"no table for (h1, h2) = ({h1}, {h2}); "
                         f"have (1,2), (1,1.5), (2,3), (2,4)")
    return _cut("chevron", k, tbl, 10)


@tabulated(
    7.8, _LAPPY + " (n_basis=120; h=16, 20 from an earlier, less careful solve); h=1 is a "
                  "closed form, not a table",
    digits_by={0.5: 10.8, 1.0: 15.6, 2.0: 12.0, 4.0: 11.8, 8.0: 11.3, 16.0: 7.8, 20.0: 12.0},
    notes={1.0: {None: "CLOSED FORM, not tabulated. Base 2 and height 1 puts the vertices at "
                       "(-1,0), (0,1), (1,0), which is the right isosceles triangle with "
                       "legs sqrt(2) -- so its spectrum is pi**2 (m**2+n**2)/2 for m > n >= "
                       "1. lappy's table agrees to 3e-15 and is superseded here."},
           16.0: {None: "not re-verified; about 7.8 digits"},
           20.0: {None: "not re-verified; about 12 digits"}},
    settings="Fourier-Bessel with fundamental solutions mixed in (non-integer pi/angle ratio), n_basis=120; h=1 is superseded by a closed form here")
def iso_tri_eigs(k, h=1.0):
    """The first 10 Dirichlet eigenvalues of `iso_tri(h)` -- base 2, height `h`.

    `h` in `{0.5, 1, 2, 4, 8, 16, 20}`. The verified heights are 10.8-13.0 digits; `digits`
    reports 7.8, the worst case, which is the unverified `h = 16`.
    """
    if float(h) == 1.0:
        from .closed_form import iso_right_tri_eigs
        return iso_right_tri_eigs(k, np.sqrt(2.0))
    tbl = {
        0.5: [67.349455544425652, 111.036404225384885, 151.473895577570715,
              199.539129893408813, 221.691797716339380, 253.997396523095404,
              298.689466549538338, 325.281558433280395, 357.434635325396926,
              394.821748879388792],
        # h=1.0 is handled above by the closed form; see notes.
        2.0: [11.456820359432427, 25.694945637505505, 27.759101056346598,
              44.337951419766441, 49.884782473352665, 50.842728850066379,
              68.247856785464919, 74.672366637382709, 81.112315003885087,
              81.320389608320156],
        4.0: [6.726526574153285, 12.333521955030589, 19.071767666600952,
              19.074704193994812, 27.005508461745229, 29.098013503262695,
              36.056833854046154, 37.246116430239411, 39.954321850759705,
              46.498064800315710],
        8.0: [4.719098710033834, 7.227058278292141, 9.950620918236773,
              12.946772679734305, 15.010378933747058, 16.233055784524762,
              19.816898996168046, 20.000479851124531, 23.701929385100062,
              24.972774558019974],
        16.0: [3.741313107211302, 4.988436354293714, 6.231679923092005,
               7.517545607747248, 8.861653817098958, 10.271173428254006,
               11.749965646537081, 12.884145973648904, 13.300328814508363,
               14.923728877388257],
        20.0: [3.538204270133983, 4.552162970620473, 5.539932740921296,
               6.544007125902493, 7.578949068369423, 8.651562362007823,
               9.765571848342804, 10.923228519465912, 12.125988718163306,
               12.4252704009599],
    }.get(float(h))
    if tbl is None:
        raise ValueError(f"no table for h = {h}; have 0.5, 1, 2, 4, 8, 16, 20")
    return _cut("iso_tri", k, tbl, 10)
