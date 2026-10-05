# Regularization for high-precision and certifiable MPS eigenvalue calculations

*Review of `penalty_prec_limit_explained.md` and a recommended computational strategy. Written 2026-09-23.*

## Executive summary

The original note gives a strong and accessible explanation of why a one-parameter, order-1 Tikhonov penalty can fail when the requested relative eigenvalue precision approaches `1e-14` in double precision.

Its central conclusion is sound, with one important qualification:

> An order-1 penalty has a precision-dependent bias constraint and a machine-precision-dependent noise constraint. At sufficiently tight requested precision, no value of its single parameter satisfies both constraints.

The numerical crossover near `1e-13` is not universal. It depends on the basis size and scaling, arithmetic precision, matrix assembly, well slopes, cluster geometry, and the search algorithm. The structural conflict, however, is general.

For calculations requested at relative precision as small as `1e-14`, especially when certification may be requested, the recommended approach is:

1. Use smooth column equilibration.
2. Use the order-2 spectral penalty rather than order-1 Tikhonov for discovery and refinement.
3. Protect all LAPACK GSVD calls by first orthonormalizing the complete regularized stack with an unpivoted thin QR factorization.
4. Choose the small order-1 stabilization parameter from the arithmetic noise scale, not directly from `prec`.
5. Choose the order-2 transition parameter by continuation and monitor movement of the inferred eigenvalue, not just the height of the tension floor.
6. Use higher precision near candidate eigenvalues when `prec` approaches `1e-14`.
7. Remove the regularizer from the final verification and certify using the original Helmholtz field, continuous residual bounds, and a spectral inclusion theorem.

Regularization should be treated as a stable candidate-generation mechanism, not as the certificate itself.

---

## 1. Assessment of the original note

The original document succeeds particularly well in four respects:

- It explains the MPS tension in terms accessible to readers outside the subject.
- It distinguishes a genuine penalty from a coordinate change or hard truncation.
- It identifies the competing upper and lower constraints on the order-1 parameter `tau`.
- It separates proved floor estimates from the heuristic model of rounding noise.

The measured comparison between truncation flicker and Tikhonov noise is persuasive. The explanation that truncation gives zero penalty to retained signal directions and effectively infinite penalty to discarded directions, whereas order-1 Tikhonov uses one finite parameter for both jobs, is the key conceptual point.

Several statements should nevertheless be sharpened before the note is treated as definitive.

## 2. Recommended corrections and qualifications

### 2.1 The floor alone does not determine eigenvalue accuracy

The statement that the solver can locate a minimum only to within roughly the floor is incomplete. The tension and the parameter `lam` have different units, and localization depends on the slope of the well.

Near a simple eigenvalue, a useful local model is

$$
\sigma_\tau(\lambda)
\approx
\sqrt{F_\tau^2+S^2(\lambda-\lambda_*)^2},
$$

where:

- $$F_\tau$$ is the regularized floor;
- $$S$$ is the slope of the unregularized well wall;
- $$\lambda_*$$ is the eigenvalue.

The width of the rounded region is therefore approximately

$$
\Delta\lambda_{\mathrm{flat}}
\sim
\frac{F_\tau}{S}.
$$

The empirical rule

$$
\tau\lesssim 0.05\,\mathrm{prec}
$$

may be correct for the measured family, but the constant incorporates the well slopes, the scale of the eigenvalues, cluster geometry, and details of the search algorithm.

A more precise formulation is:

> The penalty rounds each well over a parameter interval whose width is approximately the floor divided by the wall slope. In the measured test family, reliable localization and cluster separation required `tau` to be no more than approximately `0.05 * prec`.

### 2.2 Requested accuracy and spectral separation are different

The note should distinguish three requirements:

1. localization accuracy for an isolated eigenvalue;
2. resolution of two or more nearby eigenvalues;
3. successful certification by the slope-based search algorithm.

A requested relative precision of `1e-14` does not imply that neighboring eigenvalues are separated by `1e-14`. Conversely, a cluster can be tighter than the requested accuracy.

For an eigenvalue estimate $$\widehat\lambda$$, the relevant localization target is

$$
|\widehat\lambda-\lambda_*|
\leq
\mathrm{prec}\,|\lambda_*|.
$$

Resolution of a cluster additionally depends on the actual eigenvalue gaps and the shapes of the interacting wells.

### 2.3 The outer QR does not literally disable LAPACK rank decisions

The statement that the QR implementation stops LAPACK from making an internal rank decision is too strong.

Let the complete regularized stack be

$$
Z=QR,
$$

and partition the rows of $$Q$$ into the two blocks passed to the GSVD. The combined GSVD input then has orthonormal columns:

$$
Q_1^*Q_1+Q_2^*Q_2=I.
$$

`xGGSVD3` still runs its internal rank-revealing preprocessing. The advantage is that its total-rank decision is now applied to an orthonormal stacked matrix rather than to the ill-conditioned original pencil. Consequently, the reported total numerical rank should robustly satisfy

$$
K+L=n.
$$

A better statement is:

> The outer QR makes LAPACK's internal total-rank decision numerically harmless by presenting it with a stacked pair having orthonormal columns.

The implementation should still verify that `K + L == n` on every evaluation.

### 2.4 The `1e-14` impossibility claim applies to order-1 regularization

The statement that “nothing fixes `1e-14`” should be narrowed to:

> No choice of the single order-1 penalty parameter fixes the `1e-14` case under the observed upper and lower bounds.

This avoids contradicting the promising order-2 results.

### 2.5 The arithmetic noise scale should be expressed cautiously

After unit column normalization,

$$
\|C\|_2\leq\|C\|_F=\sqrt n.
$$

Thus

$$
u\sqrt n
$$

is a conservative matrix-scale estimate, where $$u$$ is the unit roundoff. It is not automatically an accurate estimate of $$u\|C\|_2$$.

The effective perturbation also includes constants and errors from:

- evaluation of the Helmholtz basis functions;
- assembly and scaling of the collocation matrices;
- QR factorization;
- GSVD preprocessing;
- triangular solves;
- the search algorithm's differencing and interpolation.

The lower regularization scale is therefore better written as

$$
\tau_{\min}
\sim
C_{\mathrm{noise}}u\|C\|,
$$

with $$C_{\mathrm{noise}}$$ calibrated empirically and checked across basis sizes and arithmetic precisions.

### 2.6 The floor estimate is an upper bound

For the order-1 quotient

$$
\sigma_\tau(\lambda)
=
\min_{c\neq0}
\frac{\sqrt{\|A_Bc\|^2+\tau^2\|c\|^2}}
{\|A_Ic\|},
$$

evaluating the quotient at a candidate eigenfunction coefficient vector $$c^*$$ gives

$$
\sigma_\tau(\lambda_*)
\leq
\sqrt{\epsilon_{\mathrm{basis}}^2+\tau^2\rho^2},
\qquad
\rho=\frac{\|c^*\|}{\|A_Ic^*\|}.
$$

This does not prove that the actual floor equals $$\tau\rho$$, nor does it provide a matching lower bound. The observed proportionality between floor and `tau` is empirical.

### 2.7 Define the noise measurement

Because the lower-bound argument depends on the measured relative noise, the note should state exactly how that noise was computed. Possibilities include:

- deviation from a local smooth fit;
- repeated evaluation at the same parameter;
- comparison across BLAS/LAPACK libraries;
- variation of finite-difference slope estimates;
- neighboring-sample residuals after removal of the expected well shape.

The observation that noise is multiplicative along a well wall should also be labeled as empirical rather than as a general property of Tikhonov-regularized GSVDs.

### 2.8 Prefer adapted bases to indiscriminately smaller bases

“Smaller bases” should be broadened to “better-adapted or adaptively sized bases.” Reducing the number of basis functions may improve conditioning but can also increase representation error and raise the basis-limited floor.

---

## 3. Why order-1 regularization reaches a precision limit

After column equilibration, define

$$
C(\lambda)=
\begin{bmatrix}
A_B(\lambda)\\
A_I(\lambda)
\end{bmatrix}.
$$

The order-1 penalized tension is

$$
\sigma_\tau(\lambda)
=
\min_{c\neq0}
\frac{
\sqrt{\|A_Bc\|^2+\tau^2\|c\|^2}
}
{\|A_Ic\|}.
$$

If

$$
C=U\Sigma V^*,
\qquad
\Sigma=\operatorname{diag}(s_1,\ldots,s_n),
$$

then the penalty has the whitened spectral weight

$$
g_j=\frac{\tau}{s_j},
$$

and the corresponding smooth retention factor is

$$
f_j
=
\frac{1}{1+g_j^2}
=
\frac{s_j^2}{s_j^2+\tau^2}.
$$

This creates two competing requirements.

### Upper constraint: signal bias and well rounding

A true eigenfunction direction is charged by the penalty. The regularized floor therefore grows approximately linearly with `tau` in the measured problems. Excessive well rounding impairs localization and cluster resolution.

Schematically,

$$
\tau
\lesssim
C_{\mathrm{bias}}\,\mathrm{prec},
$$

where $$C_{\mathrm{bias}}$$ is problem- and search-dependent.

### Lower constraint: arithmetic noise

Directions with $$s_j$$ near the arithmetic noise floor contain little reliable information. If `tau` is too small, the minimizer can use these directions and the tension curve becomes noisy.

Schematically,

$$
\tau
\gtrsim
C_{\mathrm{noise}}u\|C\|.
$$

The lower scale is controlled primarily by the arithmetic and basis evaluation, not by the requested eigenvalue precision.

### The squeeze

A usable order-1 parameter requires

$$
C_{\mathrm{noise}}u\|C\|
\lesssim
\tau
\lesssim
C_{\mathrm{bias}}\,\mathrm{prec}.
$$

As `prec` decreases, the upper bound shrinks while the lower bound remains approximately fixed. Eventually the interval is empty.

This is the structural reason an order-1 penalty can work at `prec = 1e-12` but fail at `prec = 1e-14` in double precision.

---

## 4. Recommended regularization: an order-2 spectral penalty

The goal is to separate the two functions performed by the order-1 parameter:

- stabilize the unresolved numerical nullspace;
- avoid charging well-resolved eigenfunction directions.

Let

$$
C=
\begin{bmatrix}
A_B\\
A_I
\end{bmatrix}.
$$

First compute an unpivoted thin QR factorization

$$
\begin{bmatrix}
C\\
\tau_1I
\end{bmatrix}
=Q_1R_1.
$$

Then define the order-2 penalized tension as the smallest generalized singular value of

$$
\left(
\begin{bmatrix}
A_B\\
\tau_1I\\
\tau_2^2R_1^{-*}
\end{bmatrix},
A_I
\right).
$$

For real matrices, $$R_1^{-*}=R_1^{-T}$$.

A coordinate-free expression for its squared numerator penalty is

$$
P
=
\tau_1^2I
+
\tau_2^4(C^*C+\tau_1^2I)^{-1}.
$$

Thus

$$
\sigma_{\tau_1,\tau_2}(\lambda)
=
\min_{c\neq0}
\frac{
\sqrt{
\|A_Bc\|^2+c^*Pc
}
}
{\|A_Ic\|}.
$$

This form makes clear that the penalty is independent of QR sign conventions and varies smoothly with the matrices for fixed positive $$\tau_1$$.

### Spectral action

In the right singular-vector coordinates of $$C$$, the squared weight in direction $$j$$ is

$$
g_j^2
=
\left(\frac{\tau_1}{s_j}\right)^2
+
\frac{\tau_2^4}{s_j^2(s_j^2+\tau_1^2)}.
$$

When $$s_j\gg\tau_1$$,

$$
g_j^2
\approx
\left(\frac{\tau_1}{s_j}\right)^2
+
\left(\frac{\tau_2}{s_j}\right)^4.
$$

If $$\tau_1$$ is small enough that the first term is negligible in the transition region, then

$$
g_j
\approx
\left(\frac{\tau_2}{s_j}\right)^2.
$$

This is “order 2” in the spectral-filter sense. It is not a second-derivative Tikhonov operator.

The order-2 regime is approximately

$$
\tau_1
\ll
s_j
\ll
\frac{\tau_2^2}{\tau_1}.
$$

Below $$s_j\sim\tau_1$$, the regularization saturates and the small order-1 component protects the factorization.

### Why this helps

For a well-resolved signal direction with $$s_j\gg\tau_2$$, the order-2 charge is much smaller than an order-1 charge at the same transition scale:

$$
\left(\frac{\tau_2}{s_j}\right)^2
\ll
\frac{\tau_2}{s_j}.
$$

For an unresolved direction with $$s_j\ll\tau_2$$, the penalty grows more rapidly and suppresses it strongly.

The two parameters therefore have different roles:

- $$\tau_1$$ protects the QR factorization and bounds the triangular inverse;
- $$\tau_2$$ sets the smooth spectral transition between retained and suppressed directions.

---

## 5. Stable realization with QR and GSVD

Do not pass the ill-conditioned regularized pencil directly to `xGGSVD3`.

Form the full stack

$$
Z=
\begin{bmatrix}
A_B\\
\tau_1I\\
\tau_2^2R_1^{-*}\\
A_I
\end{bmatrix},
$$

and compute an unpivoted thin QR factorization

$$
Z=Q_2R_2.
$$

Partition the rows as

$$
Q_2=
\begin{bmatrix}
Q_N\\
Q_D
\end{bmatrix},
$$

where $$Q_N$$ contains the rows associated with the numerator and $$Q_D$$ contains the rows associated with $$A_I$$.

Compute the GSVD of

$$
(Q_N,Q_D).
$$

Because

$$
Q_N^*Q_N+Q_D^*Q_D=I,
$$

the combined GSVD input has orthonormal columns. In exact arithmetic this has the same generalized singular values as the original regularized pair, since the two pairs differ only by the common invertible right factor $$R_2^{-1}$$.

### Implementation rules

- Use unpivoted QR. Parameter-dependent pivot changes can introduce avoidable discontinuities.
- Do not form $$R_1^{-1}$$ explicitly. Apply $$R_1^{-*}$$ through triangular solves.
- Verify `K + L == n` after every `xGGSVD3` call.
- Monitor

  $$
  \frac{\|Z-Q_2R_2\|}{\|Z\|}
  \quad\text{and}\quad
  \|Q_2^*Q_2-I\|.
  $$

- Recompute original residuals after back-mapping coefficient vectors. A stable GSVD of the orthonormal blocks does not guarantee that an individual coefficient vector is well determined.
- Track invariant subspaces rather than individual vectors at crossings or within clusters.

---

## 6. Choosing the parameters

### 6.1 Choose `tau1` from the arithmetic scale

Do not choose $$\tau_1$$ primarily from `prec`. Its job is numerical stabilization.

Use a rule of the form

$$
\tau_1
=
C_1u\,S_C,
$$

where $$S_C$$ is a smoothly varying matrix scale. Candidates include

$$
S_C=\|C\|_2
$$

if cheaply estimated, or the conservative bound

$$
S_C=\|C\|_F=\sqrt n
$$

after unit column normalization.

Choose $$C_1$$ by requiring:

- stable QR residual and orthogonality;
- repeatability under small perturbations and alternative BLAS/LAPACK implementations;
- `K + L == n` in every GSVD call;
- no reappearance of multiplicative wall noise;
- bounded and accurate triangular solves.

The same smooth scaling rule should be used throughout a parameter scan.

### 6.2 Choose `tau2` by continuation

Choose $$\tau_2$$ from the desired spectral transition, then reduce it geometrically, for example:

$$
\tau_2,
\qquad
\frac{\tau_2}{2},
\qquad
\frac{\tau_2}{4}.
$$

For each value, refine the candidate eigenvalue and monitor the movement

$$
\Delta_\tau
=
\left|
\widehat\lambda_{\tau_2}
-
\widehat\lambda_{\tau_2/2}
\right|.
$$

An acceptance criterion should require

$$
\Delta_\tau
\ll
\mathrm{prec}\,|\widehat\lambda|.
$$

The exact safety factor should be part of the certification budget.

Do not accept a parameter merely because the regularized floor is low. The primary observable is the stability of the inferred eigenvalue location.

### 6.3 Extrapolation is diagnostic, not certification

For a smooth, isolated, nondegenerate minimum, the regularization bias may admit an expansion such as

$$
\widehat\lambda_{\tau_2}
=
\lambda_*+a\tau_2^4+\cdots.
$$

This can support extrapolation and error diagnostics. It should not be assumed universally, especially near crossings, clusters, changing active subspaces, or search bookkeeping transitions.

---

## 7. Precision strategy for `prec` near `1e-14`

A requested relative precision of `1e-14` leaves only a small margin above double-precision unit roundoff. That margin must absorb errors from basis evaluation, matrix assembly, QR, GSVD, triangular solves, interpolation, and residual evaluation.

A robust workflow is therefore adaptive in arithmetic precision.

### Stage 1: discovery in double precision

- Scan the spectral interval using the order-2 regularized tension.
- Locate candidate wells and candidate clusters.
- Track subspaces continuously across the scan.
- Reject obvious numerical artifacts using parameter continuation.

### Stage 2: local refinement in higher precision

Near each candidate:

1. reevaluate the basis functions and assemble the matrices in extended or arbitrary precision;
2. repeat column equilibration in that precision;
3. reduce $$u$$ and therefore the stabilization scale $$\tau_1$$;
4. repeat the $$\tau_2$$ continuation;
5. refine the candidate interval until its width is below the requested tolerance budget.

Increasing arithmetic precision lowers the noise-controlled regularization bound. This is more reliable than trying to extract a certifiable `1e-14` answer from a double-precision noise floor.

### Stage 3: unregularized verification

For the final candidate coefficient vector $$c$$, discard the penalty and recompute

$$
\frac{\|A_Bc\|}{\|A_Ic\|}
$$

in the certification precision.

Also verify stability under:

- decreasing $$\tau_2$$;
- basis enrichment or a better-adapted basis;
- collocation refinement;
- increased arithmetic precision;
- modest changes in sampling layout.

---

## 8. Certification

Regularized collocation values alone do not constitute a mathematical certificate.

### 8.1 Numerical certification within the implementation

If “certification” means a strong internal numerical acceptance test, require at least:

1. an interval $$I$$ containing the candidate with

   $$
   \operatorname{width}(I)
   \leq
   \mathrm{prec}\,|\widehat\lambda|;
   $$

2. stability of the interval under decreasing $$\tau_2$$;
3. stability under increased arithmetic precision;
4. stability under basis and collocation refinement;
5. `K + L == n` for every GSVD evaluation;
6. small QR residual and loss of orthogonality;
7. a small original, unregularized boundary-to-interior residual;
8. explicit cluster handling when an individual eigenvalue cannot be separated reliably.

This is a high-confidence numerical result, but it is not necessarily a rigorous mathematical enclosure.

### 8.2 Rigorous spectral certification

For a rigorous certificate, replace sampled residual evidence by bounds on continuous norms of the represented Helmholtz field.

The candidate field is

$$
u_c(x;\lambda)
=
\sum_jc_j\phi_j(x;\lambda),
$$

and it solves the Helmholtz equation exactly, up to basis-evaluation arithmetic. Certification should bound quantities such as

$$
\|u_c\|_{L^2(\partial\Omega)}
$$

and

$$
\|u_c\|_{L^2(\Omega)}
$$

or the corresponding norms required by the chosen MPS inclusion theorem.

Possible tools include:

- validated quadrature;
- interval arithmetic for Bessel and trigonometric basis evaluations;
- analytic interpolation or derivative bounds between collocation points;
- rigorous boundary parametrization bounds;
- an MPS spectral inclusion theorem converting tension into distance from the spectrum.

A spectral inclusion theorem may prove that the interval contains at least one eigenvalue. To prove that it contains exactly one eigenvalue, combine the inclusion with an independent eigenvalue count, bracketing theorem, or rigorous lower and upper bounds.

For a tight cluster, the honest certificate may enclose the cluster without assigning a separate certified interval to every member.

---

## 9. Proposed acceptance logic

For each candidate well or cluster:

1. **Discover** it with order-2 regularization.
2. **Refine** it at several decreasing values of $$\tau_2$$.
3. **Escalate precision** if regularization movement, wall noise, or residual uncertainty consumes too much of the requested error budget.
4. **Check representation stability** under basis adaptation or enrichment.
5. **Check sampling stability** under collocation refinement.
6. **Verify the unregularized field** in the final arithmetic precision.
7. **Construct a spectral inclusion interval** using continuous residual bounds.
8. **Count eigenvalues** in the interval if uniqueness is required.
9. **Certify a cluster instead of individual members** when separation cannot be justified.
10. **Decline explicitly** if the error budget cannot be closed.

The error budget should distinguish at least:

- regularization bias;
- arithmetic error;
- basis representation error;
- collocation-to-continuous-norm error;
- search and interpolation error;
- spectral conditioning and cluster ambiguity.

---

## 10. Suggested revisions to key claims in the original note

### Original idea

> The solver can locate the minimum only to within roughly the floor.

### Suggested replacement

> The penalty rounds a well over a parameter interval whose width is approximately the tension floor divided by the well-wall slope. For the measured test family, reliable localization and separation required `tau` to be no more than approximately `0.05 * prec`.

### Original idea

> The QR implementation stops LAPACK from making its own internal rank decision.

### Suggested replacement

> The QR implementation presents LAPACK with a stacked GSVD input having orthonormal columns. LAPACK still performs its internal preprocessing, but its total-rank decision is then numerically protected and should return `K + L = n`.

### Original idea

> Nothing fixes `1e-14`.

### Suggested replacement

> No choice of the single order-1 penalty parameter fixes the measured `1e-14` cases: values small enough to avoid excessive well rounding fall into the arithmetic-noise regime, while values large enough to suppress that noise bias or merge the wells. Higher-order filters or higher-precision arithmetic can change this conclusion.

### Original idea

> With 126 unit-norm columns, `eps * ||A||` is about `eps * sqrt(126)`.

### Suggested replacement

> With 126 unit-norm columns, `eps * sqrt(126)` is a conservative Frobenius-norm scale because `||A||_2 <= ||A||_F = sqrt(126)`. The effective perturbation should be measured and may include dimension- and algorithm-dependent constants.

---

## 11. Final recommendation

Use the order-2 penalty as the default smooth regularizer for high-accuracy MPS searches:

$$
P
=
\tau_1^2I
+
\tau_2^4(C^*C+\tau_1^2I)^{-1}.
$$

Realize it through triangular solves and an outer unpivoted QR, then apply `xGGSVD3` only to the resulting orthonormal row blocks.

Choose $$\tau_1$$ from the arithmetic noise scale and choose $$\tau_2$$ by continuation. Judge convergence by movement of the inferred eigenvalue or cluster interval, not only by the tension floor.

For requested relative precision near `1e-14`, perform final assembly and refinement in higher precision. Use the regularized solve only to obtain stable candidates. Construct the final certificate from the unregularized Helmholtz field, continuous residual bounds, and an appropriate spectral inclusion and counting argument.

In short:

> **Order-2 regularization for smooth discovery; higher precision for the final window; unregularized, continuous residual bounds for certification.**
