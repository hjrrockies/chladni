*Status: scope note. The layer it describes is not written yet.*

# Overview
`chladni` is going to rely on accurate boundary integration for two key parts of the MPS solve:
1. Eigenfunction orthonormalization using Rellich identities
2. Eigenvalue derivatives w.r.t. shape perturbation using Hadamard formulas

In both cases, we need to integrate the Cauchy data (values, normal and tangent derivatives) against a weight function. The Cauchy data of an eigenfunction is generically singular in the presence of corners. This means that `chladni` needs highly robust adaptive boundary quadrature, tailored to the actual singularities and the desired precision for the calculations. This all needs to be served up as a convenient container for consumption by an MPS eigensolver.