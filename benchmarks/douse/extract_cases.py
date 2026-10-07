"""Extract the douse acceptance-benchmark shapes into `cases.json`.

Run from the chladni root, under the douse venv (douse and lappy importable):

    /Users/hringer/projects/python/douse/.venv/bin/python benchmarks/douse/extract_cases.py

The case list is `docs/design/douse_failure_cases.md` (ids T1.x, T2.x). Every shape is read from
douse's stored data -- JSONL ledgers, `cache.npz`, `start_shape` -- or from a literal copied from
a source file, with that file:line cited next to it. douse and lappy are READ-ONLY here: this
script opens their files for reading and imports their modules, nothing else (bytecode writing is
switched off so the imports leave no `.pyc` behind).

Parameters become vertices through douse's own `douse.polygon.polygon_vertices(p, perim)`; every
case is checked for perimeter (when built from `p`), counter-clockwise orientation and simplicity,
and every case whose `p` is a ledger row's endpoint `x` is checked against the row's stored
`edges` and `angles`. A record located by line number is also checked by content, so a
mis-numbered citation fails loudly instead of silently extracting the wrong shape.
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True        # import douse/lappy modules without writing into their trees

PY = Path('/Users/hringer/projects/python')
DOUSE = PY/'douse'
LAPPY = PY/'lappy'
LED = DOUSE/'benchmarks/run/ratio_multistart'
PRE = LED/'pre_rerun_2026-08-27'
RERUN = LED/'rerun'
OUT = Path(__file__).resolve().parent/'cases.json'

sys.path.insert(0, str(DOUSE))        # so `benchmarks` resolves to douse's package

import numpy as np                                                   # noqa: E402

from douse.polygon import param_vector, polygon_vertices             # noqa: E402
import benchmarks.ratio_multistart as RM                             # noqa: E402
import benchmarks.kink_sweep as KS                                   # noqa: E402

PERIM = 4.0                 # `VertexPolygon(N, perim=4.0)`: ratio_multistart.py:77, ratio_stall.py:32,
                            # missing_mode_cache.py:41, ngon_basis_probe.py:72, kink_sweep.py:57
PREC_LEDGER = 1e-13         # SpectrumEvaluator default prec, douse/evaluate.py:146
PERIM_TOL = 1e-12           # relative, for shapes built from p
EDGE_TOL = 1e-9             # stored ledger edges are lappy's, to full float precision
ANGLE_TOL = 1e-7            # degrees


# ---------------------------------------------------------------------------------------------
# Readers

def rel(path):
    """`path` relative to the douse or lappy root, for `source` strings."""
    path = Path(path)
    for root, tag in ((DOUSE, ''), (LAPPY, 'lappy/')):
        try:
            return tag + str(path.relative_to(root))
        except ValueError:
            pass
    return str(path)


def line(path, n):
    """Record on 1-based line `n` of a JSONL file."""
    with open(path) as fh:
        for i, text in enumerate(fh, 1):
            if i == n:
                return json.loads(text)
    raise IndexError(f'{path} has fewer than {n} lines')


def expect(cond, msg):
    if not cond:
        raise AssertionError(msg)


def floats(x):
    return [float(v) for v in np.asarray(x, dtype=float).ravel()]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def git_head(repo):
    try:
        return subprocess.run(['git', '-C', str(repo), 'rev-parse', '--short', 'HEAD'],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:                                  # noqa: BLE001 -- provenance only
        return '?'


# ---------------------------------------------------------------------------------------------
# Geometry and checks

def interior_angles_deg(v):
    """Interior angle at each vertex of a CCW polygon, in degrees (reflex > 180)."""
    a = np.roll(v, 1) - v                              # toward the previous vertex
    b = np.roll(v, -1) - v                             # toward the next vertex
    return np.degrees(np.mod(np.angle(a/b), 2*np.pi))


def signed_area(v):
    return 0.5*float(np.sum(v.real*np.roll(v.imag, -1) - np.roll(v.real, -1)*v.imag))


def is_simple(v):
    """No two non-adjacent edges intersect (touching or collinear overlap counts as a hit)."""
    n = len(v)
    P, Q = v, np.roll(v, -1)                           # edge i runs P[i] -> Q[i]

    def cross(a, b):
        return a.real*b.imag - a.imag*b.real

    i, j = np.triu_indices(n, k=2)
    keep = ~((i == 0) & (j == n - 1))                  # edges 0 and n-1 share vertex 0
    i, j = i[keep], j[keep]
    if len(i) == 0:
        return True
    p, r, q, s = P[i], Q[i] - P[i], P[j], Q[j] - P[j]
    rxs = cross(r, s)
    qp = q - p
    scale = np.max(np.abs(v))**2
    with np.errstate(divide='ignore', invalid='ignore'):
        t = cross(qp, s)/rxs
        u = cross(qp, r)/rxs
    proper = (np.abs(rxs) > 1e-14*scale) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    # parallel: a hit only if collinear and overlapping
    col = (np.abs(rxs) <= 1e-14*scale) & (np.abs(cross(qp, r)) <= 1e-14*scale)
    rr = np.abs(r)**2
    t0 = (qp.real*r.real + qp.imag*r.imag)/rr
    t1 = t0 + (s.real*r.real + s.imag*r.imag)/rr
    overlap = col & (np.maximum(t0, t1) >= 0) & (np.minimum(t0, t1) <= 1)
    return not bool(np.any(proper | overlap))


def geometry(v, tag):
    v = np.asarray(v, dtype=complex)
    area = signed_area(v)
    expect(area > 0, f'{tag}: signed area {area} is not positive (not CCW)')
    expect(is_simple(v), f'{tag}: polygon is not simple')
    edges = np.abs(np.roll(v, -1) - v)
    ang = interior_angles_deg(v)
    expect(abs(ang.sum() - 180*(len(v) - 2)) < 1e-8,
           f'{tag}: interior angles sum to {ang.sum()}, not {180*(len(v) - 2)}')
    return dict(interior_angles_deg=floats(ang), area=area, perimeter=float(edges.sum()),
                min_edge=float(edges.min()))


def verts_from_p(p, tag, perim=PERIM):
    v = polygon_vertices(np.asarray(p, float), perim)
    P = float(np.abs(np.roll(v, -1) - v).sum())
    expect(abs(P - perim) <= PERIM_TOL*perim, f'{tag}: perimeter {P!r} != {perim}')
    return v


def check_against_row(v, row, tag):
    """Stored `edges`/`angles` (sorted, by lappy) reproduced from our vertices."""
    msgs = []
    if 'edges' in row and isinstance(row['edges'], list):
        e = np.sort(np.abs(np.roll(v, -1) - v))
        d = float(np.max(np.abs(e - np.sort(row['edges']))))
        expect(d < EDGE_TOL, f'{tag}: stored edges differ by {d:.2e}')
        msgs.append(f'edges {d:.1e}')
    if 'angles' in row and isinstance(row['angles'], list):
        a = np.sort(interior_angles_deg(v))
        d = float(np.max(np.abs(a - np.sort(row['angles']))))
        expect(d < ANGLE_TOL, f'{tag}: stored angles differ by {d:.2e} deg')
        msgs.append(f'angles {d:.1e} deg')
    return ('row\'s stored ' + ' and '.join(msgs) + ' reproduced (max abs diff)') if msgs else ''


def vlist(v):
    return [[float(z.real), float(z.imag)] for z in np.asarray(v, dtype=complex)]


def plan(p, tag, vertices=None):
    if p is None and vertices is None:
        return None
    v = verts_from_p(p, tag + ' plan_at') if vertices is None else np.asarray(vertices, complex)
    geometry(v, tag + ' plan_at')
    return dict(p=None if p is None else floats(p), vertices=vlist(v))


CASES = []


def add(id, tier, kinds, *, p=None, vertices=None, perim=PERIM, plan_at=None, source,
        n_eigs=None, prec=None, returned=None, note, truth=None, notes='', row=None):
    expect(p is not None or vertices is not None, f'{id}: no shape')
    if vertices is None:
        v = verts_from_p(p, id, perim)
    else:
        v = np.asarray(vertices, dtype=complex)
    g = geometry(v, id)
    extra = check_against_row(v, row, id) if row is not None else ''
    notes = '; '.join(s for s in (notes, extra) if s)
    if truth is not None:
        truth = dict(values=floats(truth['values']), kind=truth['kind'], source=truth['source'])
    CASES.append(dict(
        id=id, tier=tier, kinds=list(kinds), N=len(v), perim=float(perim) if perim else None,
        p=None if p is None else floats(p), vertices=vlist(v), plan_at=plan_at,
        source=source, n_eigs_requested=n_eigs, prec_requested=prec,
        lappy_returned=None if returned is None else floats(returned), lappy_note=note,
        claimed_truth=truth, **g, notes=notes))
    return v


def close(a, b, rtol=1e-12):
    return np.allclose(np.asarray(a, float), np.asarray(b, float), rtol=rtol, atol=0)


# =============================================================================================
# Tier 1

def t1_1():
    path = DOUSE/'benchmarks/run/missing_mode/cache.npz'
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d['meta']))
    src = rel(path)
    p_bad, p_good = d['p_bad'], d['p_good']
    # catalogue values, douse_failure_cases.md:37, :47
    expect(close(p_bad, [1.02885517693507, 0.4136014913917629]), 'T1.1 p_bad')
    expect(close(p_good, [0.9291503563674081, 0.4300043720047623]), 'T1.1 p_good')
    fem_bad = d['fem_bad']
    # the same shape is LED/N3_k2.jsonl L8 refusals[0]
    ref = line(LED/'N3_k2.jsonl', 8)['refusals'][0]
    expect(close(ref['p'], p_bad, 0), 'T1.1: LED/N3_k2 L8 refusals[0] is not p_bad')
    add('T1.1-bad', 1, 'abc', p=p_bad, plan_at=plan(p_bad, 'T1.1-bad'),
        source=f'{src} [p_bad, lams_bad, fem_bad]; same p as LED/N3_k2.jsonl L8 refusals[0]',
        n_eigs=4, prec=PREC_LEDGER, returned=d['lams_bad'],
        note=('21.9/25.0/133.1 deg triangle, plan built at the shape: returned true indices 1,3,4,7 '
              'at tension 2.8e-14, no warning; F 2.7530 vs true 2.1346; Weyl deficit +2.02. '
              'Same shape at n_eigs=6 raises UnresolvedSpectrum (spurious value, tension 5.07e-4).'),
        truth=dict(values=fem_bad, kind='FEM P1 upper bound',
                   source=f'{src} fem_bad (conforming P1, {meta["bad"]["n_dof"]} dofs, '
                          f'fem_hfrac={meta["fem_hfrac"]})'),
        notes=(f'cache stores fem_bad with {len(fem_bad)} values; the catalogue says "44" -- the '
               f'44-value list is meta["bad_n6"]["fem"] (same mesh, same values to 1e-12, three '
               f'more). Stored edges/angles in meta["bad"].'))
    trail = meta['bracket']['trail']
    hit = [t for t in trail if close(t['p'], [0.9402286697638149, 0.42818182971442903], 0)]
    expect(len(hit) == 1, 'T1.1: bisection shape not in meta.bracket.trail')
    t = hit[0]
    add('T1.1-bisection', 1, 'ac', p=t['p'], plan_at=plan(t['p'], 'T1.1-bisection'),
        source=f'{src} meta["bracket"]["trail"] entry t={t["t"]}',
        n_eigs=4, prec=PREC_LEDGER, returned=t['lams'],
        note=('on the bisection toward the clean partner: lam_1 returned twice (88.0218/88.0219), '
              f'true lam_4 dropped; Weyl deficit {t["weyl"]:+.2f} and FEM deficit '
              f'{t["fem_deficit"]} -- both count statistics fooled.'),
        truth=None,
        notes=('no FEM eigenvalues stored for this shape (only fem_deficit=0). A simple lam_1 is '
               'a theorem, so the returned pair 88.0218/88.0219 is certainly wrong.'))
    add('T1.1-good', 1, [], p=p_good, plan_at=plan(p_good, 'T1.1-good'),
        source=f'{src} [p_good, lams_good, fem_good]',
        n_eigs=4, prec=PREC_LEDGER, returned=d['lams_good'],
        note='clean partner (control): complete spectrum, FEM deficit 0, tension 2.3e-14.',
        truth=dict(values=d['fem_good'], kind='FEM P1 upper bound',
                   source=f'{src} fem_good ({meta["good"]["n_dof"]} dofs)'),
        notes='control case, not a failure.')


def t1_2():
    path = LED/'pre_plateau_fix/N5_k2.jsonl'
    row = line(path, 1)
    expect(row['start'] == 0 and close(row['lams'][:1], [21.57523253952739]), 'T1.2 row')
    # plan_at: the regular pentagon in the VertexPolygon gauge, built the way douse builds it
    # everywhere (param_vector(regular(N), PERIM): ratio_stall.py:88, ngon_basis_probe.py:77-80,
    # tests/test_convexity_and_sampling.py:50); `regular` is ratio_multistart.py:81-82.
    p_reg = param_vector(RM.regular(5), PERIM)
    p_start = param_vector(RM.start_shape(5, 0), PERIM)
    cert = line(LED/'N5_k2.jsonl', 5)
    expect(cert['start'] == 5 and close(cert['lams'][0], 21.575211135401954), 'T1.2 cert row')
    ce = np.sort(cert['edges'])
    v0 = verts_from_p(row['x'], 'T1.2')
    dedge = float(np.max(np.abs(np.sort(np.abs(np.roll(v0, -1) - v0)) - ce)))
    add('T1.2', 1, 'ac', p=row['x'],
        plan_at=plan(p_reg, 'T1.2'),
        source=f'{rel(path)} L1 (start 0) field x',
        n_eigs=4, prec=PREC_LEDGER, returned=row['lams'],
        note=('recorded run (track_set, frozen plan) returned one copy of the double 68.8536 and '
              'jumped to lam_5=88.46 at tension 1.6e-10, deficit +0.683; FEM deficit +1 at '
              '1984-49842 dofs. A full solve on a plan frozen at the REGULAR pentagon returns '
              '[21.5752, 21.5752, 39.4871, 68.8537]: an invented double lam_1, F 1.8302 vs 3.1913, '
              'passing every guard (tension 2.9e-8, Weyl -0.79, FEM deficit 0).'),
        truth=dict(values=cert['lams'], kind='certified lappy',
                   source=(f'{rel(LED/"N5_k2.jsonl")} L5 (start 5, verified, '
                           f'worst_sigma {cert["worst_sigma"]:.1e}) -- the values quoted as '
                           f'"certified" in LAPPY_WISHLIST.md:323, HANDOFF.md:503, docs/todo.md:712. '
                           f'CAUTION: this is a NEARBY shape, not this one (sorted edges differ by '
                           f'{dedge:.1e}); trust it to ~5-6 digits here, and the multiplicity '
                           f'pattern [1,1,2]')),
        row=row,
        notes=('HOW THE PLAN WAS FROZEN is not recorded. The sources say only "plan at the REGULAR '
               'pentagon" (LAPPY_WISHLIST.md:338, docs/todo.md:777); lappy cold_start Leg 3 swept a '
               'frozen regular-pentagon plan and could not reproduce. plan_at here is the exact '
               'regular pentagon of perimeter 4 in the VertexPolygon gauge, param_vector(regular(5), '
               '4.0). The other candidate is the run\'s own start shape (this row has n_replans=0, so '
               'its plan was frozen at start_shape(5, 0), a jittered regular pentagon): p = '
               f'{floats(p_start)}. Note a frozen plan maps corner blocks by vertex index, and the '
               'labelling of the regular pentagon relative to x is a choice (all labellings are '
               'congruent for the regular pentagon itself).'))


def t1_3():
    p = param_vector(RM.start_shape(3, 4), PERIM)
    p_cat = [1.1756103887576692, 0.23218091730644705]           # douse_failure_cases.md:67
    expect(close(p, p_cat, 0), f'T1.3: start_shape(3,4) {p} != catalogue {p_cat}')
    repro = DOUSE/'benchmarks/diagnostics/repro_N3_k2_s4.jsonl'
    r1 = line(repro, 1)
    expect(close(r1['p'], p, 0), 'T1.3 repro L1 p')
    spectra = [('repro L1 (n_eigs=4, clean)', r1['lams'])]
    by_n = {}
    for k, ln in ((1, 3), (2, 1), (3, 1), (4, 1), (5, 4)):
        r = line(PRE/f'N3_k{k}.jsonl', ln)
        expect(r['start'] == 4, f'T1.3 pre N3_k{k} L{ln}')
        refs = r.get('refusals') or []
        if refs:
            expect(close(refs[0]['p'], p, 0), f'T1.3 pre N3_k{k} refusal p')
            spectra.append((f'pre_rerun/N3_k{k}.jsonl L{ln} refusals[0]', refs[0]['lams']))
            by_n[k + 2] = refs[0]['lams']
    # consensus: cluster all stored values (rel 1e-9), keep the cluster median
    allv = np.sort(np.concatenate([np.asarray(s, float) for _, s in spectra]))
    groups, cur = [], [allv[0]]
    for x in allv[1:]:
        if abs(x - cur[-1]) <= 1e-9*x:
            cur.append(x)
        else:
            groups.append(cur)
            cur = [x]
    groups.append(cur)
    consensus = [float(np.median(g)) for g in groups]
    cat = [248.653419474, 344.590171803, 430.380005708, 518.615801753, 611.337416106,
           705.749663843, 807.007609817, 887.472441245, 1020.39768]   # douse_failure_cases.md:71-72
    expect(len(consensus) == len(cat) and np.allclose(consensus, cat, rtol=2e-9, atol=0),
           f'T1.3 consensus {consensus} != catalogue {cat}')
    spread = max((max(g) - min(g))/np.mean(g) for g in groups)
    add('T1.3', 1, 'ad', p=p, plan_at=plan(p, 'T1.3'),
        source=('benchmarks.ratio_multistart.start_shape(3, 4) -> param_vector(., 4.0); equals '
                'catalogue p and repro_N3_k2_s4.jsonl L1 p bit-for-bit'),
        n_eigs=4, prec=PREC_LEDGER, returned=by_n[4],
        note=('11.2/16.3/152.6 deg spike, fresh plan: result depends on n_eigs. n_eigs=4 drops '
              '430.38; =5 drops lam_1 248.65; =6 drops 248.65 and 344.59; =7 drops 248.65 and '
              '887.47. Later code raises EigensolverFailure ("f has too many local minima").'),
        truth=dict(values=consensus, kind='consensus',
                   source=('cluster medians of every stored spectrum at this p: '
                           + '; '.join(s for s, _ in spectra)
                           + f'. Max relative spread within a cluster {spread:.1e}. Values above '
                             '887.47 rest on one record each (1020.39768 only on N3_k5).')),
        notes=('catalogue cites repro_N3_k2_s4.jsonl for the 9-value reference, but that file holds '
               'only the first 4 (L1); values 5-9 come from the pre_rerun N3_k3..k5 refusal records '
               '(p == plan_at). lappy_returned is the n_eigs=4 refusal. Others: '
               + '; '.join(f'n_eigs={n}: {[round(x, 4) for x in s]}' for n, s in sorted(by_n.items()))
               + '. N3_k1 L3 (n_eigs=3) is a crash row with no shape.'))


def t1_4():
    ref = load_module('lappy_reference_values',
                      LAPPY/'benchmarks/suite/run/reference_values.py').REFERENCE
    src = 'lappy/benchmarks/suite/run/reference_values.py'
    for n, tag in ((6, 'hex'), (8, 'oct')):
        # lappy/lappy/geometry.py:2155-2158 reg_ngon(n): unit CIRCUMRADIUS, CCW from (1, 0)
        theta = np.linspace(0, 2*np.pi, n + 1)[:-1]
        v = np.exp(1j*theta)
        e = ref[f'reg_ngon_{n}']
        add(f'T1.4-{tag}', 1, 'c', vertices=v, perim=None, plan_at=None,
            source='lappy.geometry.reg_ngon(%d) (lappy/lappy/geometry.py:2155-2158), unit circumradius' % n,
            n_eigs=len(e['eigs']), prec=None, returned=None,
            note=('reference-pipeline failure, not a cold solve: symsolve.solve_sym reducing D%d by '
                  'its proper subgroup D2 under-counted a sector -- reg_ngon_6 listed the double '
                  '32.45185751 once; reg_ngon_8 was missing lam_1 = 6.4849335. Still open in '
                  'lappy\'s per-sector path.' % n),
            truth=dict(values=e['eigs'], kind='certified lappy',
                       source=f'{src} REFERENCE["reg_ngon_{n}"] (corrected, full-domain, '
                              f'certified_digits={e["certified_digits"]})'),
            notes=(f'unit circumradius, perimeter {2*n*np.sin(np.pi/n):.15g}, NOT 4. lappy\'s '
                   'full-domain cold path measured correct multiplicities here (lappy '
                   'benchmarks/cold_start/run/NOTEBOOK.md Leg 3).'))


def t1_5():
    # lappy/benchmarks/cold_start/truth.py:26-35 rect_of_ratio(ratio, perim=4.0); ratios are the
    # literal 1.633 and 2.0 (cold_start_audit.py:64, probe.py:48) -- 1.633 is NOT sqrt(8/3).
    for ratio, k, dup in ((1.633, 12, '94.0813 (lam_5) returned twice'),
                          (2.0, 22, '72.171482 (lam_3) returned twice')):
        H = PERIM/(2*(1 + ratio))
        L = ratio*H
        v = np.array([0, L, L + 1j*H, 1j*H], dtype=complex)
        m = np.arange(1, 60)[:, None]
        nn = np.arange(1, 60)[None, :]
        exact = np.sort((np.pi**2*(m**2/L**2 + nn**2/H**2)).ravel())[:k]
        add(f'T1.5-r{ratio:g}', 1, 'c', vertices=v, plan_at=None,
            source=('lappy/benchmarks/cold_start/truth.py:26-35 rect_of_ratio(%g); failure in '
                    'lappy/docs/WORK_ORDER_solver_reliability.md:54-62' % ratio),
            n_eigs=k, prec=None, returned=None,
            note=(f'lappy cold path (Eigenproblem(dom).solve({k}), default precision, weyl_audit on): '
                  f'{dup} after a false Weyl alarm re-scan; fixed in lappy (MERGE_RTOL).'),
            truth=dict(values=exact, kind='closed form',
                       source='pi^2 (m^2/L^2 + n^2/H^2), H = 4/(2(1+r)), L = r H'),
            notes=('full returned arrays are not stored; the work order quotes the first 7 at '
                   'r=1.633: [23.5203 42.7641 74.8371 74.8375 94.0813 94.0813 119.7392]. The '
                   '74.8371/74.8375 pair is a near-double because 1.633 is close to sqrt(8/3).'
                   if ratio == 1.633 else 'full returned array not stored.'))


# =============================================================================================
# Tier 2

def t2_1():
    path = PRE/'N6_k1_s2_second_pass.json'
    row = json.load(open(path))[0]
    x_cat = [0.796102460003105, 1.0430649234606624, 0.3045509941484931, -0.2964947023033227,
             0.114354480800215, 0.862628738987791, 1.2240136818145162, 0.6424560268310093]
    expect(close(row['x'], x_cat, 0) and row['start'] == 2, 'T2.1 first shape')
    add('T2.1-pass1', 2, 'a', p=row['x'], plan_at=None, source=f'{rel(path)}[0] field x',
        n_eigs=3, prec=PREC_LEDGER, returned=row['lams'], row=row,
        note=('N6_k1 start 2: true lam_2 = 41.49929 absent at tension 1.3e-9; 92 iterations on '
              'lam_2/lam_1 reported 2.6203 > PPW 2.5387. Fresh ratio 2.4293.'),
        truth=None,
        notes=('plan_at not stored (n_replans=1). tests/test_spectral_guard.py:20 holds the same '
               'x rounded to 12 digits; the JSON value is used.'))
    path = RERUN/'N6_k1_start2.jsonl'
    row = line(path, 1)
    expect(close(row['lams'][2], 58.19482523360248), 'T2.1 rerun')
    add('T2.1-rerun', 2, 'ab', p=row['x'], plan_at=None, source=f'{rel(path)} L1 field x',
        n_eigs=3, prec=PREC_LEDGER, returned=row['lams'], row=row,
        note=('rerun shape: 58.1948 returned at tension 0.62 (not an eigenvalue) under '
              'worst_sigma 1.3e-9; Weyl 0.470 -- one dropped plus one spurious balances the count. '
              'The origin of spurious_tol.'),
        truth=None, notes='plan_at not stored.')


def t2_2():
    path = LED/'N6_k2.jsonl'
    row = line(path, 8)
    expect(row['start'] == 1 and row['lams'][1] == row['lams'][3] == 34.467345990537844, 'T2.2')
    expect(row['x'] == row['plan_at'], 'T2.2 x != plan_at')
    add('T2.2', 2, 'c', p=row['x'], plan_at=plan(row['plan_at'], 'T2.2'),
        source=f'{rel(path)} L8 (start 1) fields x, plan_at', row=row,
        n_eigs=4, prec=PREC_LEDGER, returned=row['lams'],
        note=('bit-identical TRIPLE 34.467345990537844 on an asymmetric hexagon (15.5..251.1 deg), '
              'verified=True; F 1.090 vs cell 3.198; tension 3.3e-7, deficit 0.23 -- passed every '
              'guard. Suspected, from implausibility only.'),
        truth=None)


def t2_3():
    path = LED/'N5_k4.jsonl'
    row = line(path, 8)
    expect(row['start'] == 5 and close(row['lams'][0], 20.33765341129639), 'T2.3 endpoint')
    add('T2.3-endpoint', 2, 'c', p=row['x'], plan_at=plan(row['plan_at'], 'T2.3'),
        source=f'{rel(path)} L8 (start 5) fields x, plan_at', row=row,
        n_eigs=6, prec=PREC_LEDGER, returned=row['lams'],
        note=('double ground state lam_1 = lam_2 = 20.33765341129639/20.337653411564297 (rel gap '
              '1.3e-11) and lam_3 = lam_4 = 48.6884 on an irregular pentagon; F 3.9255 vs cell '
              '5.0433; verified=True, excluded only on worst_sigma 4.9e-5. Wrong by theorem '
              '(simple lam_1).'),
        truth=None)
    path = PRE/'N5_k4.jsonl'
    pre = line(path, 4)
    expect(pre['start'] == 5 and len(pre['refusals']) == 4, 'T2.3 pre L4')
    for j, what in ((0, 'drops the ~51.35 mode (deficit 1.31); fresh plan (p == plan_at)'),
                    (3, 'also collapses the 100.5/101.0 pair (deficit 3.05); plan frozen at '
                        'refusals[0].p')):
        r = pre['refusals'][j]
        add(f'T2.3-pre-ref{j}', 2, 'ac', p=r['p'], plan_at=plan(r['plan_at'], f'T2.3-pre-ref{j}'),
            source=f'{rel(path)} L4 (start 5) refusals[{j}] fields p, plan_at',
            n_eigs=6, prec=PREC_LEDGER, returned=r['lams'],
            note=f'pre-rerun refusal: {what}; deficit {r["deficit"]:.4f}.', truth=None,
            notes=f'from_full_solve={r["from_full_solve"]}.')


def t2_4():
    src = DOUSE/'benchmarks/run/ratio_stall'
    # ratio_stall stores p (VertexPolygon gauge, perim 4: ratio_stall.py:32, 75, 85), no plan.
    for f, ln, n_eigs, note in (
            ('N5.jsonl', 4, 3, 'lam_2 duplicated as a bit-identical pair 45.5625 (ratio 2.614 > PPW); '
                               'true lam_2 gone'),
            ('N5.jsonl', 5, 3, 'lam_2 duplicated, 47.9100 x2 (ratio 2.724 > PPW); true lam_2 gone'),
            ('N5.jsonl', 28, 3, 'from iterate 17 (line 18) the 44.4468/44.4470 pair has lost a '
                                'member: returned [17.6709, 44.4477, 76.8476]; the run stalled at '
                                '2.51530 against the regular pentagon\'s 2.52684'),
            ('N4.jsonl', 25, 3, 'near-double 49.42243851/49.42243861 (gap 2.1e-9) on a non-square '
                                'quadrilateral; run then stopped on 40 failed backtracks')):
        r = line(src/f, ln)
        N = len(r['p'])//2 + 2
        expect(r['it'] == ln - 1, f'T2.4 {f} L{ln}')
        add(f'T2.4-N{N}-L{ln}', 2, 'c', p=r['p'], plan_at=None,
            source=f'{rel(src/f)} L{ln} (it={r["it"]}) field p',
            n_eigs=n_eigs, prec=PREC_LEDGER, returned=r['lams'], note=note, truth=None,
            notes=('ratio_stall stores p (not vertices), perimeter 4; no plan stored. '
                   'Representative of lines 18-28 (all have the pair collapsed; L28 is the '
                   'stalled endpoint).' if ln == 28 else
                   'ratio_stall stores p (not vertices), perimeter 4; no plan stored.'))


def t2_5():
    p = param_vector(RM.start_shape(6, 5), PERIM)
    p_cat = [1.6312897734785867, 0.9117377758429616, 0.6231789069106254, 0.2840791568565929,
             0.2333524495837802, 0.904778806970504, 0.6638056912940102, 0.17752397373425044]
    expect(close(p, p_cat, 0), 'T2.5: start_shape(6, 5) != catalogue p')    # :144-146
    add('T2.5', 2, 'abcde', p=p, plan_at=plan(p, 'T2.5'),
        source='benchmarks.ratio_multistart.start_shape(6, 5) -> param_vector(., 4.0); equals catalogue p',
        n_eigs=5, prec=PREC_LEDGER, returned=None,
        note=('reflex hexagon (203 deg): fails at every k. k=3 (n_eigs=5): UnresolvedSpectrum at '
              'the start (deficit 5.21), later a spurious value at 5.02e-4. k=2: one copy of the '
              'near-double 77.41 dropped, F inflated 37%. k=1: lam_1 ~ 39 dropped entirely.'),
        truth=None,
        notes=('plan_at = the shape (the fresh plan a start uses). k=2 and k=1 failures are at '
               'run endpoints, not this start: pre_rerun/N6_k2.jsonl L6 x and pre_rerun/N6_k1.jsonl '
               'L8 x (pre-rerun plans not stored).'))


def t2_6():
    specs = [
        ('T2.6-N6k3-L4-ref0', PRE/'N6_k3.jsonl', 4, 0, 7, 5, 'ac',
         'convex sliver; full solve on the frozen plan drops lam_5 = 94.7242 (deficit 1.98); a '
         'FRESH plan agrees bit-for-bit (HANDOFF.md:657-667), so not only staleness. lappy\'s '
         'missed-modes audit recovers it.'),
        ('T2.6-N5k5run3-ref0', RERUN/'N5_k5_contested_run3.jsonl', 2, 0, 1, 7, 'ac',
         'frozen plan returns [20.2672, 42.6339, 59.1257, 77.2296, ...] at deficit +0.51; a plan '
         'built AT this shape returns [20.2672, 20.2672, 42.6339, 59.1257, ...] at -0.73 '
         '(douse/evaluate.py:464-469).'),
        ('T2.6-N5k4-L7-ref0', PRE/'N5_k4.jsonl', 7, 0, 3, 6, 'a',
         'near-degenerate pentagon (start 3): ~50.04 dropped, deficit 1.32; fresh plan.'),
        ('T2.6-N5k4-L7-ref3', PRE/'N5_k4.jsonl', 7, 3, 3, 6, 'a',
         'same start, plan frozen at refusals[0].p: ~50.04 dropped, deficit 1.32.'),
        ('T2.6-N3k2-L8-ref0', PRE/'N3_k2.jsonl', 8, 0, 6, 4, 'ac',
         'one copy of the near-double 125.166/125.170 lost; deficit 0.84.'),
        ('T2.6-N3k5-L6-ref0', PRE/'N3_k5.jsonl', 6, 0, 5, 7, 'ac',
         'one copy of the near-double 177.458172/177.458199 lost; deficit 0.68.'),
        ('T2.6-N3k4-L8-ref0', LED/'N3_k4.jsonl', 8, 0, 4, 6, 'a',
         'thin triangle, deficit +4.64: about four modes missing.'),
    ]
    for cid, path, ln, j, start, n_eigs, kinds, note in specs:
        row = line(path, ln)
        expect(row['start'] == start, f'{cid}: start {row["start"]} != {start}')
        r = row['refusals'][j]
        frozen = r['p'] != r['plan_at']
        extra = []
        if cid == 'T2.6-N6k3-L4-ref0':
            expect(close(r['lams'][3], 93.71285099174577), cid)
            a = interior_angles_deg(verts_from_p(r['p'], cid))
            v = verts_from_p(r['p'], cid)
            e = np.abs(np.roll(v, -1) - v)
            extra.append(f'max interior angle {a.max():.1f} deg, edge ratio min/max '
                         f'{e.min()/e.max():.3f} (catalogue: 171.8 deg, 0.27)')
            extra.append('94.7242 itself is not stored; it is quoted in HANDOFF.md:665 and '
                         'LAPPY_WISHLIST.md:211')
        if cid == 'T2.6-N5k5run3-ref0':
            expect(close(r['lams'][1], 42.63392683824144), cid)
            extra.append('CAUTION: douse/evaluate.py:469 and the catalogue read this as "lam_1 is '
                         'DOUBLE and the frozen plan returned one copy". lam_1 of a connected '
                         'domain is simple, so the at-shape result [20.2672, 20.2672, ...] is the '
                         'one that is wrong (an invented duplicate, as in T1.2); the frozen-plan '
                         'set may instead be missing a copy of the near-double at ~122.8 (it '
                         'returns 122.8017/122.8211). Unsettled -- see T2.7.')
            extra.append('rerun/N5_k5_contested.jsonl L2 is the same record (the two rerun files '
                         'differ only in "seconds")')
        if cid == 'T2.6-N3k4-L8-ref0':
            expect(close(r['p'], [1.0352995627987067, 0.3312315145786663], 0), cid)
        if 'N3k2' in cid or 'N3k5' in cid:
            same = all(rr['p'] == r['p'] and rr['lams'] == r['lams'] for rr in row['refusals'])
            extra.append(f'all {len(row["refusals"])} refusals on this row are '
                         + ('identical' if same else 'NOT identical'))
        add(cid, 2, kinds, p=r['p'], plan_at=plan(r['plan_at'], cid),
            source=f'{rel(path)} L{ln} (start {start}) refusals[{j}] fields p, plan_at',
            n_eigs=n_eigs, prec=PREC_LEDGER, returned=r['lams'],
            note=note + f' Stored deficit {r["deficit"]:.4f}.', truth=None,
            notes='; '.join([('FROZEN plan (plan_at != p)' if frozen else 'plan_at == p (fresh plan)'),
                             f'from_full_solve={r["from_full_solve"]}'] + extra))


def t2_7():
    path = LED/'N5_k5.jsonl'
    recs = []
    with open(path) as fh:
        for ln, text in enumerate(fh, 1):
            row = json.loads(text)
            for j, r in enumerate(row.get('refusals') or []):
                recs.append((r['deficit'], ln, row['start'], j, r))
    recs.sort(key=lambda t: t[0])
    lo, hi = recs[0], recs[-1]
    mid = recs[len(recs)//2]
    for tag, (dfc, ln, start, j, r) in (('lowest', lo), ('median', mid), ('highest', hi)):
        add(f'T2.7-{tag}', 2, 'ac', p=r['p'], plan_at=plan(r['plan_at'], f'T2.7-{tag}'),
            source=f'{rel(path)} L{ln} (start {start}) refusals[{j}] fields p, plan_at',
            n_eigs=7, prec=PREC_LEDGER, returned=r['lams'],
            note=(f'N5_k5 (lam_6/lam_1) contested cell: Weyl refusal at deficit {dfc:.4f} '
                  '(threshold 0.5), near-double at ~122.5-123.5. Either a Weyl false positive at '
                  'k >= 6 or a lost copy; reruns give F 6.0617/6.0698 vs certified 6.0656540842.'),
            truth=None,
            notes=(f'{tag} of the {len(recs)} refusal records in this file (deficits '
                   f'{lo[0]:.4f}-{hi[0]:.4f}, all >= 0.5, so the lowest IS the one nearest 0.5; '
                   'the third pick is the median instead). from_full_solve='
                   f'{r["from_full_solve"]}; plan_at == p: {r["p"] == r["plan_at"]}. The '
                   'catalogue\'s "0.485-0.546" is the range of the rows\' worst_weyl_deficit, '
                   'not of refusal deficits; stored refusals number 48 across LED, pre_rerun and '
                   'the two (identical) rerun files.'))


def t2_8():
    for N, i, extra in ((5, 2, ''), (6, 1, ' (N6_k1 start 1 crashed twice)')):
        p = param_vector(RM.start_shape(N, i), PERIM)
        add(f'T2.8-N{N}s{i}', 2, 'd', p=p, plan_at=plan(p, f'T2.8-N{N}s{i}'),
            source=f'benchmarks.ratio_multistart.start_shape({N}, {i}) -> param_vector(., 4.0)',
            n_eigs=None, prec=PREC_LEDGER, returned=None,
            note=('EigensolverFailure: f has too many local minima (lappy/opt.py:212) at this start '
                  f'shape{extra}. Not deterministic.'),
            truth=None, notes='crash rows store no shape and no k-specific n_eigs; R-.')
    # kink_sweep.py:66-69 pentagon(h); frozen_plan_crossing (kink_sweep.py:114-133) plans at
    # h_freeze = 0.03 and realizes at h = 0 -> IndexError in basis_plan.realize.
    v0 = KS.pentagon(0.0)
    vf = KS.pentagon(0.03)
    L, H = KS.L_RECT, KS.H_RECT                         # 2(L + H) = 4 already (kink_sweep.py:58-60)
    m = np.arange(1, 30)[:, None]
    n = np.arange(1, 30)[None, :]
    exact = np.sort((np.pi**2*(m**2/L**2 + n**2/H**2)).ravel())[:4]
    add('T2.8-kink', 2, 'd', vertices=v0, plan_at=plan(None, 'T2.8-kink', vertices=vf),
        source=('benchmarks.kink_sweep.pentagon(0.0) (kink_sweep.py:66-69); plan_at = pentagon(0.03), '
                'the h_freeze of frozen_plan_crossing (kink_sweep.py:114)'),
        n_eigs=4, prec=1e-12, returned=None,
        note=('a plan frozen at a pentagon (h = 0.03) realized at its flat-vertex limit raises '
              'IndexError in basis_plan.realize (one block per corner; the flat vertex is not one). '
              'Works at |h| = 1e-8. Deterministic.'),
        truth=dict(values=exact, kind='closed form',
                   source='h = 0 is the sqrt(8/3) rectangle of perimeter 4: pi^2 (m^2/L^2 + n^2/H^2)'),
        notes=('vertices as stored by kink_sweep (lappy Polygon, not the VertexPolygon gauge), so p '
               'is null. Vertex 1 is FLAT (180 deg). The (1,2)/(3,1) modes cross here, so lam_3 = '
               'lam_4 is an exact double.'))


def t2_9():
    # tests/test_derivative.py:53 SEED_SHAPE, stored as vertices; the test maps it through
    # param_vector(SEED_SHAPE, 4.0) (test_derivative.py:66), which rescales to perimeter 4.
    seed = np.array([0.0, 1.3, 1.1 + 0.9j, 0.15 + 0.8j])
    p_seed = param_vector(seed, PERIM)
    add('T2.9-seed', 2, 'e', vertices=seed, perim=None, plan_at=None,
        source='douse/tests/test_derivative.py:53 SEED_SHAPE (vertices as stored)',
        n_eigs=None, prec=None, returned=None,
        note=('tension floor limits finite differences: target 1e-12 gives a floor 1.15e-10 and '
              'FD error 3.7e-4; target 1e-14 a floor 1.02e-11.'),
        truth=None,
        notes=(f'perimeter as stored is not 4; the test solves param_vector(SEED_SHAPE, 4.0) = '
               f'{floats(p_seed)} (perimeter 4, same shape up to similarity).'))
    path = LED/'pre_plateau_fix/N4_k3.jsonl'
    row = line(path, 1)
    lams_test = [23.6302578136, 55.3495724946, 57.3573811503, 101.2178122523, 101.2178122638]
    expect(np.allclose(row['lams'][:5], lams_test, rtol=0, atol=6e-11), 'T2.9 N4 optimum')
    add('T2.9-N4opt', 2, 'e', p=row['x'], plan_at=None,
        source=(f'{rel(path)} L1 (start 0) field x -- its lams match the literal in '
                'tests/test_cluster_opt.py:439-440 to the 10 digits printed there'),
        n_eigs=len(row['lams']), prec=PREC_LEDGER, returned=row['lams'], row=row,
        note=('N=4 lam_4/lam_1 optimum: top pair 101.2178122523/101.2178122638 split 1.1e-10 '
              'relative, below the basis resolution; the lost cluster cost 8 of the first 10 runs '
              '(trust radius collapsed to 5.68e-15).'),
        truth=None,
        notes=('the test itself stores no shape; this is the ledger row whose lams it quotes. '
               'Starts 1 and 4 (L2, L5) reach congruent endpoints with the pair bit-identical.'))
    path = LED/'N5_k3.jsonl'
    row = line(path, 6)
    expect(row['start'] == 5 and close(row['fun'], 4.375460168029941), 'T2.9 N5_k3 s5')
    pre = line(PRE/'N5_k3.jsonl', 7)
    expect(pre['x'] == row['x'], 'T2.9 N5_k3 s5: LED L6 and pre_rerun L7 differ')
    add('T2.9-N5k3s5', 2, 'e', p=row['x'], plan_at=None,
        source=f'{rel(path)} L6 (start 5) field x (identical to pre_rerun/N5_k3.jsonl L7)',
        n_eigs=5, prec=PREC_LEDGER, returned=row['lams'], row=row,
        note=('tracked F 4.441676010 disagrees with a fresh solve 4.375460168 (gap 6.6e-2) at the '
              'endpoint; trust ratios -9e7 to -7.5e11; trust radius collapsed.'),
        truth=None,
        notes=('catalogue cites tests/test_discontinuity.py and LED/detector/; neither stores a '
               'shape (detector/N5_k3_s5.json is a summary). The shape is this endpoint row, whose '
               'fun equals the fresh value 4.375460168 (douse/cluster_opt.py:1112, '
               'benchmarks/diagnostics/detector_validation.py:3). The tracked 4.441676010 is not '
               'stored with a shape.'))
    # regular N-gons at perimeter 4 in the VertexPolygon gauge: ngon_basis_probe.py:77-80
    probe = [json.loads(t) for t in open(DOUSE/'benchmarks/run/ngon_basis/probe.jsonl')]
    ref = load_module('lappy_reference_values',
                      LAPPY/'benchmarks/suite/run/reference_values.py').REFERENCE
    for N in (5, 8, 12, 16):
        p = param_vector(RM.regular(N), PERIM)
        default = [r for r in probe if r['N'] == N and 'prec' not in r]
        hiprec = [r for r in probe if r['N'] == N and r.get('prec') == 1e-13]
        d_lo = (min(r['digits'] for r in default), max(r['digits'] for r in default))
        d_hi = (min(r['digits'] for r in hiprec), max(r['digits'] for r in hiprec))
        hp = np.asarray(hiprec[0]['lams'])
        expect(all(np.array_equal(r['lams'], hiprec[0]['lams']) or
                   np.allclose(r['lams'], hp, rtol=1e-10) for r in hiprec), f'ngon {N} hiprec')
        if f'reg_ngon_{N}' in ref:
            e = ref[f'reg_ngon_{N}']
            scale = (2*N*np.sin(np.pi/N)/PERIM)**2      # unit circumradius -> perimeter 4
            truth_vals = np.asarray(e['eigs'])[:len(hp)]*scale
            dev = float(np.max(np.abs(truth_vals - hp)/truth_vals))
            truth = dict(values=truth_vals, kind='certified lappy',
                         source=(f'lappy/benchmarks/suite/run/reference_values.py REFERENCE'
                                 f'["reg_ngon_{N}"] (certified_digits={e["certified_digits"]}), '
                                 f'unit circumradius scaled to perimeter 4 by (2N sin(pi/N)/4)^2; '
                                 f'agrees with the probe\'s prec=1e-13 solve to {dev:.1e} rel'))
        else:
            truth = dict(values=hp, kind='high-precision lappy',
                         source=(f'benchmarks/run/ngon_basis/probe.jsonl, prec=1e-13 rows '
                                 f'(check_precision digits {d_hi[0]:.2f}-{d_hi[1]:.2f}); not '
                                 'certified, no independent check stored'))
        add(f'T2.9-ngon{N}', 2, 'e', p=p, plan_at=plan(p, f'T2.9-ngon{N}'),
            source=('param_vector(regular(%d), 4.0), as ngon_basis_probe.regular_domain '
                    '(ngon_basis_probe.py:77-80); data benchmarks/run/ngon_basis/probe.jsonl' % N),
            n_eigs=4, prec=None, returned=default[0]['lams'],
            note=(f'default prec gives {d_lo[0]:.2f}-{d_lo[1]:.2f} digits (check_precision), flat '
                  f'in n_cap and target; prec=1e-13 gives {d_hi[0]:.2f}-{d_hi[1]:.2f}.'),
            truth=truth,
            notes='prec_requested null = lappy/douse default; the failure is the default\'s accuracy.')
    path = LED/'N5_k3_convex.jsonl'
    row = line(path, 1)
    expect(row['lams'][3] == row['lams'][4] and row['nit'] == 1, 'T2.9 N5_k3_convex L1')
    add('T2.9-N5k3convex', 2, 'ce', p=row['x'], plan_at=plan(row['plan_at'], 'T2.9-N5k3convex'),
        source=f'{rel(path)} L1 (start {row["start"]}) fields x, plan_at', row=row,
        n_eigs=5, prec=PREC_LEDGER, returned=row['lams'],
        note=('near-triangle (173.9 and 179.6 deg vertices) "certified" at F 2.3455 after one '
              'iteration (cell 4.4824) on a bit-identical double 124.64827962507786. Low confidence.'),
        truth=None)


SKIPPED = [dict(
    id='T1.6', reason=('right_trapezoid and H_shape (lappy suite) already exist in chladni\'s '
                       'domain library; scan-resolution drops at low ppl: right_trapezoid dropped '
                       '44.9484877814 at k=10 (ppl 5), H_shape dropped 14.30522996 at k=7, 9 '
                       '(ppl 10). Reference lappy/benchmarks/suite/run/reference_values.py.'),
    domains=['right_trapezoid', 'H_shape'])]


def main():
    for f in (t1_1, t1_2, t1_3, t1_4, t1_5, t2_1, t2_2, t2_3, t2_4, t2_5, t2_6, t2_7, t2_8, t2_9):
        f()
    ids = [c['id'] for c in CASES]
    expect(len(ids) == len(set(ids)), 'duplicate case ids')
    out = dict(
        generated_by=(f'benchmarks/douse/extract_cases.py (douse {git_head(DOUSE)}, lappy '
                      f'{git_head(LAPPY)}); case ids from docs/design/douse_failure_cases.md'),
        perim_convention=('VertexPolygon(N, perim=4.0): p = [x_1..x_{N-2}, y_1..y_{N-2}], '
                          'vertices = douse.polygon.polygon_vertices(p, 4.0) (last vertex at the '
                          'origin, closing edge on +x). Cases with p null store vertices from their '
                          'source as-is (perim null where the source is not perimeter-4). '
                          'Vertices are CCW; angles in degrees.'),
        cases=CASES, skipped=SKIPPED)
    OUT.write_text(json.dumps(out, indent=1, allow_nan=False) + '\n')
    print(f'{len(CASES)} cases, {len(SKIPPED)} skipped -> {OUT}')
    for c in CASES:
        print(f'  {c["id"]:22s} N={c["N"]:2d} perim={c["perimeter"]:.15f} area={c["area"]:.6f} '
              f'min_edge={c["min_edge"]:.4f} max_angle={max(c["interior_angles_deg"]):.2f}')


if __name__ == '__main__':
    main()
