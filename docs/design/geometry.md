# Overview
The `geometry` module of `chladni` is the first layer of the system. It gives the user access to convenient functions for creating segments, boundaries, and domains. These geometric objects can be queried to obtain points sampled from the boundary, points sampled from the interior of a domain, tangent and normal vectors to the boundary, and other quantities like area, perimeter, and diameter. Segments and boundaries come with attached boundary conditions, which defaults to `dir` for Dirichlet. Individual segments always have a single boundary condition to be enforced, which is `dir` for Dirichlet, `neu` for Neumann, or a float value for the convex-combination form of Robin boundary conditions: `(1-bc_param)*u + bc_param*du_dN = 0`.

# Design principles
- Positive/counter-clockwise orientation: Segments understood to be taken counter-clockwise (CCW), boundaries consisting of multiple segments store the segments in CCW order, and outward-pointing unit normal vectors are generated with that assumption.
- Adaptive algorithms: Approximations (such as reparameterization with respect to arclength, or computing the area of a domain) should be done adaptively.
- User-specified precision: Approximations should also adapt to working relative precision.
- Lazy evaluation: Quantities like area and perimeter should be computed once, but only lazily.
- Pythonic sugar: Pythonic design should include "syntatic sugar" to make manipulating the geometry easy.