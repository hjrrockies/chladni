# The Lipschitz-Cone Method

A global search for approximate Laplacian eigenvalues in the GSVD formulation of the
method of particular solutions.

---

## 1. Problem

Given matrix families $A(\lambda) \in \mathbb{C}^{m_1 \times n}$ and
$B(\lambda) \in \mathbb{C}^{m_2 \times n}$, let
$\sigma_1(\lambda) \le \sigma_2(\lambda) \le \cdots$ be the generalized singular values
of the pair. A point $\lambda^\*$ is an **approximate eigenvalue** if $\sigma_1$ has a
near-zero local minimum there, and its **multiplicity** is the largest $\kappa$ such
that $\sigma_1, \dots, \sigma_\kappa$ all have near-zero local minima at $\lambda^\*$.

The task is to find every approximate eigenvalue in $[a, b]$, with multiplicities, using
as few GSVD evaluations as possible. Each evaluation returns the entire vector
$\sigma(\lambda)$, and every component of it should be treated as usable signal.

Two tolerances are supplied by the user: `tol_rel`, the desired relative precision on
$\lambda$, and `tol_merge`, the separation below which distinct eigenvalues may be
combined into one with conserved total multiplicity.

---

## 2. The local model

The objective is not a generic function. It sits at an $O(1)$ plateau almost everywhere
and dips into narrow, nearly V-shaped wells whose walls have comparable slopes. Near an
approximate eigenvalue,

$$\sigma_1(\lambda) \;\approx\; \sqrt{\varepsilon^2 + c^2 (\lambda - \lambda^\*)^2},$$

where $\varepsilon = \sigma_1(\lambda^\*)$ is the **floor** — set by how well the
particular-solution basis represents the true eigenfunction, so neither zero nor at the
level of machine epsilon — and $c$ is the wall slope. Three consequences drive the whole
design.

**Far field** ($c|\delta| \gg \varepsilon$). The function is $\approx c|\delta|$: a clean
cone with a Lipschitz kink at the bottom. Non-smooth, so interpolation applied to
$\sigma_1$ directly will misbehave.

**Near field** ($c|\delta| \lesssim \varepsilon$). The cone rounds off into a smooth
quadratic bottom of width $\sim \varepsilon / c$.

**Squaring removes the kink.** For a simple smallest generalized singular value,
$\sigma_1^2$ is an eigenvalue of an analytic Hermitian family and is therefore analytic.
The model says it equals $\varepsilon^2 + c^2\delta^2$ — an exact parabola in the far
field and a good one near the floor. **Always minimize $f = \sigma_1^2$, never
$\sigma_1$.**

### 2.1 Resolution limit

Let $\eta$ be the absolute noise in a computed generalized singular value, roughly
$u\|\cdot\|$ for the backend GSVD. Two evaluations are distinguishable only when
$\sigma_1(\lambda) - \varepsilon \gtrsim \eta$, giving

$$\boxed{\;\delta_{\mathrm{res}} \;\approx\; \frac{\sqrt{2\varepsilon\eta + \eta^2}}{c}
\;\approx\; \frac{\sqrt{2\varepsilon\eta}}{c}\;}$$

the geometric mean of floor and noise, divided by the slope. There is good news hidden
here: if the basis is excellent, so that $\varepsilon \approx \eta$, the cone is sharp
and $\delta_{\rm res} \approx \eta/c$ — full precision, escaping the usual $\sqrt{u}$
barrier that applies to smooth minima. The algorithm estimates $\delta_{\rm res}$ per
eigenvalue and refuses to promise more than that, whatever the user requested.

### 2.2 Work in $k = \sqrt{\lambda}$

Weyl's law makes eigen-wavenumbers asymptotically equispaced with density
$|\Omega|k/2\pi$ in two dimensions, and the wall slope $c$ is far more nearly constant in
$k$ than in $\lambda$. A single global Lipschitz constant is then meaningful across a
wide range, which is what makes the scan cheap. Convert back to $\lambda$ only for
reporting, remembering $\delta k / k = \tfrac12\, \delta\lambda/\lambda$.

---

## 3. Phase A — global scan by Lipschitz branch and bound

The difficulty is not finding dips but *certifying their absence*. Since
$\sigma_1 \ge 0$ there is no sign change to bracket, so completeness has to come from a
slope bound rather than from an intermediate value theorem.

### 3.1 The envelope

Given samples $s_i = \sigma_1(k_i)$ at the ends of an interval of width $h$, and a
Lipschitz constant $L$, no interior point can have

$$\sigma_1 \;<\; \hat s \;=\; \tfrac12\big(s_i + s_{i+1} - Lh\big),$$

with the bound attained at

$$\hat k \;=\; \tfrac12 (k_i + k_{i+1}) \;+\; \frac{s_i - s_{i+1}}{2L}.$$

This is the Piyavskii–Shubert lower envelope. The procedure is:

- If $\hat s > \tau_{\rm scan}$, the interval is **certified free** of approximate
  eigenvalues. Move it to the attic (§3.3) rather than discarding it.
- Otherwise evaluate at $\hat k$ and split, pushing both children onto a priority queue
  keyed by $\hat s$, smallest first.
- Stop subdividing when the isolation criterion (§3.4) is met; the interval then becomes
  a candidate bracket for Phase B.

### 3.2 What this actually buys

On the plateau $s_i \approx S = O(1)$, so an interval survives pruning only when
$h \gtrsim 2(S - \tau_{\rm scan})/L$. Away from dips the scan therefore strides forward
in steps of order $S/c$.

It is worth being precise about the comparison, because it is easy to overstate. That
stride is the **cone half-width**, which is comparable to the eigenvalue spacing — and it
is the same scale a uniform grid needs in order to see a discrete local minimum. So the
gain over a well-chosen uniform grid is a modest constant, not orders of magnitude. The
large factor $S/\tau$ applies only against a grid fine enough to resolve dip *floors*,
which is a straw man; nobody builds one.

The genuine advantages of branch and bound over a uniform grid are different:

1. **Adaptivity.** Effort concentrates automatically where $\sigma_1$ is small, and the
   stride widens where the plateau is flat, without the user tuning a spacing per
   subinterval or grading by Weyl density by hand.
2. **An explicit certificate.** Every region not returned was ruled out by a recorded
   numerical bound, which makes the completeness claim auditable and — via the attic —
   revisable.
3. **Continuity with refinement.** The final subdivision hands Phase B a bracket that is
   already tight and already has its endpoints evaluated, so no work is repeated.

Against that, the scan is inherently **sequential**: each new point depends on all
previous ones. If GSVDs can be run concurrently, a uniform grid submitted as one batch
may win on wall-clock even while losing on evaluation count. See §7.

### 3.3 Calibrating $L$, and the attic

Set $L$ from data, not from theory.

Initialize with a short uniform pilot pass — 20 to 40 points — and take
$\hat L = \gamma \max_i |\Delta s / \Delta k|$ with $\gamma \approx 2$. Then make it
self-correcting:

- Every new evaluation supplies a new secant slope. If any observed slope exceeds
  $\hat L / \gamma$, raise $\hat L$.
- Keep every pruned interval in an **attic**, tagged with the $\hat s$ that justified
  pruning. When $\hat L$ increases, re-test the attic and revive anything no longer
  certified.

Revival costs no function evaluations, and it converts "the Lipschitz constant was
underestimated" from a silent missed eigenvalue into a few extra evaluations. Pruning is
the only irreversible act in the algorithm, and the attic is what makes it reversible.

Once the first genuine dip is resolved in Phase B, the fitted parabola yields $c$
directly, and $\hat L$ becomes trustworthy.

Choose $\tau_{\rm scan}$ generously — say $10\,\tau_{\rm accept}$ — and filter later. A
dip whose floor turns out worse than expected must not be thrown away during the scan.

### 3.4 Free multiplicity signal, and the isolation criterion

Each GSVD returns the whole vector $\sigma(k)$; cache all of it. The count

$$c(k) \;=\; \#\{\, j : \sigma_j(k) \le \tau_{\rm scan} \,\}$$

is a lower bound on how many branches are near $k$, at no extra cost. Where
$c(k) \ge 2$, the interval may hold several distinct minima, so subdivide past the point
where a single-minimum bracket would suffice. Where $c(k) = 1$ throughout, one
refinement suffices and no cluster search is needed.

A sharper cluster detector is available and worth using. With two branches
$v_i(k) = \sqrt{\varepsilon^2 + c^2 (k - k_i)^2}$ and everything else large,

$$\sigma_1 = \min(v_1, v_2), \qquad \sigma_2 = \max(v_1, v_2).$$

So $\sigma_1$ has two minima with a local max between them, while $\sigma_2$ has exactly
one minimum, located at the crossing point $\tfrac12(k_1 + k_2)$. **The offset between
the $\sigma_2$ minimum and the nearest $\sigma_1$ minimum estimates $d/2$**, where
$d = |k_1 - k_2|$ is the separation — so a cluster can be measured before it is resolved.
A genuine double eigenvalue instead gives $\sigma_1 \equiv \sigma_2$ with coincident
minima. Use this to decide how far to subdivide.

**Isolation criterion.** Hand an interval to Phase B when its width falls below
$\alpha S / \hat L$ with $\alpha \approx \tfrac14$, *and* the sampled $c(k)$ never
exceeded 1 inside. Otherwise keep splitting.

---

## 4. Phase B — local refinement

Within a bracket $k_L < k_M < k_R$ satisfying $f(k_M) < \min(f(k_L), f(k_R))$, minimize
$f = \sigma_1^2$ using **Brent's method**: parabolic interpolation safeguarded by golden
section.

In the far field the three points lie on the exact parabola
$\varepsilon^2 + c^2\delta^2$, so the first interpolation step lands essentially on
$\lambda^\*$. Convergence typically takes 3–6 evaluations. The golden-section safeguard
engages only when two branches cross, which is exactly the clustered case where
$\sigma_1^2$ genuinely has a kink.

The parabola fit does triple duty. Its vertex abscissa is $\lambda^\*$, its vertex
ordinate is $\varepsilon^2$, and its curvature is $2c^2$. Every refinement therefore
returns the local floor and slope for free, feeding $\delta_{\rm res}$, the multiplicity
thresholds, and $\hat L$.

**Termination.** Stop when the bracket width falls below

$$\max\big(\texttt{tol\_rel} \cdot k, \;\; \delta_{\rm res}\big), \qquad
\delta_{\rm res} = \sqrt{2\varepsilon\eta}\,/\,c$$

using the current fit, or when the last three values of $f$ agree to within $\eta$
(stagnation — the search is inside the rounded bottom and further evaluations sample only
noise). Report the attained precision with each eigenvalue, and record which of the two
bounds was active, so that no digits are promised that do not exist.

**Retain a wide bracket.** Save the pair of evaluated points on either side of
$\lambda^\*$ at distance of order $\varepsilon/c$ — that is, the bracket from an early
iteration, before it collapsed. Phase C needs points where the higher $\sigma_j$ have
actually moved; the final bracket of width $\delta_{\rm res}$ is useless for that. This
costs nothing, since the evaluations were already made.

---

## 5. Phase C — multiplicity

Implement the definition literally, using cached vectors only. No new GSVDs are needed.

At $\lambda^\*$, with wide bracket $\lambda_- < \lambda^\* < \lambda_+$, declare index $j$
**active** when both conditions hold.

**(C1) Smallness with a gap.** $\sigma_j(\lambda^\*) \le \tau_{\rm accept}$ and the
largest relative gap in the low end of the spectrum falls after $j$, i.e.
$\sigma_{j+1}/\sigma_j \ge G$ with $G$ in the range 10 to 100. A gap test is far more
robust than any absolute threshold, because the floor $\varepsilon$ varies between
eigenvalues by orders of magnitude.

**(C2) Verified local minimum.**
$\sigma_j(\lambda^\*) < \min\big(\sigma_j(\lambda_-), \sigma_j(\lambda_+)\big) - \eta$.

The multiplicity $\kappa$ is the number of active indices, which by ordering is a
contiguous block $1, \dots, \kappa$.

**Why (C2) matters.** Consider two simple eigenvalues separated by $d$ with
$\delta_{\rm res} < d \lesssim \varepsilon/c$. At $\lambda_1^\*$ the second branch
contributes $\sigma_2 \approx cd$, which is small — a pure threshold test would report
$\kappa = 2$ at $\lambda_1^\*$ and again at $\lambda_2^\*$, and any subsequent merge would
double count. But near $\lambda_1^\*$ the function $\sigma_2$ is the *other* cone's
descending wall: monotone, not a local minimum. Condition (C2) rejects it, both points
report $\kappa = 1$, and the total is right.

This scheme extends to arbitrary multiplicity without special cases, which pairwise
$\sigma_1$/$\sigma_2$ logic does not: a triple eigenvalue needs $\sigma_3$, and three
close simple eigenvalues place $\sigma_2$ minima at two different midpoints.

---

## 6. Phase D — clustering and conservation

Sort the candidates and group by single linkage using the merge radius

$$\rho(\lambda) = \max\big(\texttt{tol\_merge} \cdot \lambda, \;\; \delta_{\rm res}(\lambda)\big).$$

Then distinguish two regimes, because the conservation rule differs between them. Apply
bottom-up: collapse first, then merge.

### 6.1 Unresolvable cluster (diameter $\le \delta_{\rm res}$)

The members are not separate findings. They are the same well located from different
brackets, or genuinely indistinguishable eigenvalues. Collapse to the single member with
the smallest $\sigma_1$ and **recount** $\kappa$ at that point using the Phase C rule.

Do **not** sum here. Summation is precisely the double-counting failure: $\kappa$
coincident eigenvalues produce $\kappa$ small values of $\sigma_j$ at one well, so each
duplicate discovery would independently report all $\kappa$ of them.

### 6.2 Tolerance cluster (diameter $\le$ `tol_merge`$\cdot\lambda$ but $> \delta_{\rm res}$)

The members were separately resolved and independently verified as distinct local
minima, and (C2) guaranteed that each count includes only branches minimal *at that
point*. Total multiplicity is therefore the **sum** of member multiplicities. Place the
merged eigenvalue at the $\sigma_1$-weighted centroid and report the cluster diameter as
its uncertainty.

Applied in this order, the procedure conserves total multiplicity in both regimes and
never double counts.

---

## 7. Pseudocode

```text
input:  [a,b], tol_rel, tol_merge, tau_accept, GSVD oracle sigma(.)
setup:  [A,B] = [sqrt(a), sqrt(b)];  tau_scan = 10 * tau_accept
        cache: memoized  k -> full sigma vector

# --- pilot: calibrate L and the plateau level S ---
sample ~30 points uniformly in [A,B]
Lhat = 2 * max secant slope of sigma_1
S    = median sigma_1

# --- Phase A: Lipschitz branch and bound ---
Q = intervals of the pilot grid;  attic = {}
while Q nonempty:
    I = pop interval with smallest envelope value s_hat
    if s_hat > tau_scan:
        attic.add(I, s_hat);  continue
    if width(I) < alpha * S / Lhat and maxcount(I) <= 1:
        candidates.add(I);  continue
    evaluate sigma at k_hat(I)          # one GSVD, full vector cached
    update Lhat from new secants
    if Lhat grew: revive attic entries no longer certified
    push both children of I

# --- Phase B: refine each candidate ---
for I in candidates:
    (lam_star, eps, c) = brent_min(
        f      = sigma_1^2,
        brack  = I,
        stop   = max(tol_rel * k, sqrt(2*eps*eta)/c)
                 or stagnation of last three f-values within eta )
    save wide bracket (lam_minus, lam_plus) at distance ~ eps/c

# --- Phase C: multiplicity from cached vectors, no new GSVDs ---
for each lam_star:
    kappa = #{ j : sigma_j(lam_star) <= tau_accept
                   and sigma_{j+1}/sigma_j >= G
                   and sigma_j(lam_star) < min(sigma_j at wide bracket) - eta }

# --- Phase D: cluster, collapse, merge ---
collapse clusters of diameter <= delta_res      -> recount kappa at representative
merge    clusters of diameter <= tol_merge*lam  -> sum kappa

return (lambda, kappa, attained_precision, floor eps) for each
```

---

## 8. Reducing the evaluation count further

**Two-level basis.** Cost per evaluation scales roughly as $n^3$. Run Phase A with a
reduced basis of size $n/2$: eight times cheaper per point, and the coarser basis has a
*larger* floor $\varepsilon$, hence *wider* dips — easier to catch, not harder. Then run
Phase B at full $n$, seeded by the coarse brackets, which are within
$O(\varepsilon_{\rm coarse}/c)$ of the true minima. This is usually the single largest
saving available.

**Never discard a vector.** One GSVD yields every $\sigma_j$. Multiplicity, cluster
detection, and slope calibration should all cost zero additional evaluations. Memoize by
$k$, since branch and bound will re-request points.

**Parallelism changes the objective.** With $p$ concurrent GSVDs the metric becomes
wall-clock rounds, not evaluation count. Pop the $p$ most promising intervals per round
and evaluate as a batch; the pilot pass and the independent Phase B refinements
parallelize trivially. If $p$ is large, a graded uniform grid may be the better scan
after all — the branch-and-bound advantage is adaptivity, and adaptivity is worth less
when points are nearly free.

**Weyl cross-check.** Compare $\sum \kappa$ over $[a,b]$ against

$$N(b) - N(a) \;\approx\; \frac{|\Omega|}{4\pi}(b-a) \;-\; \frac{|\partial\Omega|}{4\pi}\big(\sqrt b - \sqrt a\big).$$

A shortfall strongly suggests $\hat L$ was too small or $\tau_{\rm scan}$ too tight. It
costs nothing and catches the one failure mode the method cannot detect internally.

---

## 9. Limitations

**Pruning is the only correctness risk.** It fails exactly when the true local Lipschitz
constant exceeds $\hat L$ — a dip narrower than the model allows. Attic revival makes
this self-healing whenever the violation is *observed* somewhere, but a dip that is
narrow everywhere and never sampled is invisible to any finite-evaluation method. The
honest output is therefore conditional:

> No $\lambda \in [a,b]$ with $\sigma_1(\lambda) \le \tau_{\rm scan}$ exists outside the
> returned set, **given** Lipschitz constant $\hat L$.

Report $\hat L$, the largest observed secant slope, the number of attic revivals, and the
Weyl discrepancy as diagnostics. A large margin between $\hat L$ and the largest observed
slope means the scan was conservative and probably safe; a small margin means the
constant was being pushed and the run deserves a rerun with larger $\gamma$.

**Avoided crossings.** The $\min/\max$ picture in §3.4 assumes branches cross
transversally. Strictly, $\sigma_1^2$ and $\sigma_2^2$ are eigenvalues of a Hermitian
analytic family in one parameter, so crossings are non-generic and what actually occurs
is an avoided crossing. For near-degenerate MPS branches the eigenfunctions are nearly
orthogonal and the coupling is exponentially small, so the gap falls far below $\eta$ and
the crossing picture holds numerically. If a pair ever does couple appreciably, $\sigma_1$
shows a single wide flat bottom instead of two minima, and no amount of refinement
recovers two eigenvalues. This is a limit of the formulation, not of the search.

**Resolution is not tolerance.** Below $\delta_{\rm res}$ the discrete structure of the
sampled minima is determined by rounding, not by the mathematics. Refining past that
point does not merely stall — it can *spuriously separate* a degenerate pair into two
simple eigenvalues. The termination rule in §4 and the collapse rule in §6.1 exist
specifically to prevent this.

---

## 10. Parameter summary

| Symbol | Meaning | Typical setting |
|---|---|---|
| $\eta$ | absolute noise in a computed $\sigma$ | $u \cdot \|\cdot\|$ from backend |
| $\varepsilon$ | dip floor, $\sigma_1(\lambda^\*)$ | measured per eigenvalue |
| $c$ | cone wall slope | from parabola curvature |
| $S$ | plateau level of $\sigma_1$ | median of pilot samples |
| $\hat L$ | Lipschitz constant in $k$ | $\gamma \times$ max secant, $\gamma \approx 2$ |
| $\tau_{\rm accept}$ | "near zero" threshold | user supplied |
| $\tau_{\rm scan}$ | scan threshold | $10\,\tau_{\rm accept}$ |
| $\alpha$ | isolation width factor | $\approx 1/4$ |
| $G$ | multiplicity gap ratio | 10 to 100 |
| $\delta_{\rm res}$ | resolution limit | $\sqrt{2\varepsilon\eta}\,/\,c$ |
