"""Helmholtz-exact particular solutions: the families an MPS eigensolver evaluates.

The spectral parameter is the wavenumber `k`, real and positive, in
`laplacian(u) + k**2 * u = 0`. `lam = k**2` is the user-facing eigenvalue and belongs to the
`mps` layer, converted at one seam. See docs/design/conventions.md.

Matrices are `(n_pts, n_cols)`, float64. Gradients are one complex array, `du_dx + 1j*du_dy`.

Columns arrive PRE-SCALED: a returned column is `psi_j * exp(-sigma_j)` where `sigma_j`
depends on `(family, column, k)` and never on the evaluation points. That is what makes a
downstream unit-norm rescale compose without double-counting -- the log scales simply add --
and it is what lets a Fourier-Bessel column at a sharp corner exist at all rather than
underflowing to zeros.
"""
