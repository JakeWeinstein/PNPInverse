# Mesh, Domain Decomposition, and Multi-Scale Strategies for PNP Solvers

## 1. The Core Challenge: EDL Resolution at Scale

The electric double layer (EDL) in a PNP system has a characteristic thickness on the order of the Debye length, which in aqueous electrolytes is typically 0.1--10 nm. The potential and ion concentrations decay exponentially from the electrode surface over this scale, creating steep gradients that demand fine mesh resolution. Meanwhile, the bulk transport domain may span micrometers or more. This multi-scale gap (3--4 orders of magnitude) is the central meshing challenge.

**Key insight from the literature**: the stiffness matrix conditioning in PNP problems depends more on the exponential variation of concentrations across the domain than on the mesh size alone. The transformed (self-adjoint, Slotboom) formulation of Nernst-Planck is always associated with an ill-conditioned stiffness matrix due to exponential factors varying by many orders of magnitude from the electrode to the bulk. The primitive (non-self-adjoint) formulation is preferred in practice because it avoids these conditioning problems.
- Source: [Lu et al., PNP FEM for biomolecular diffusion-reaction, PMC2922884](https://pmc.ncbi.nlm.nih.gov/articles/PMC2922884/)

## 2. Practical Mesh Sizing Rules

### 2.1 Elements Per Debye Length

There is no single universal formula, but the literature converges on these guidelines:

- **Minimum spacing < Debye length**: The mesh spacing in the EDL region must be smaller than the extrinsic Debye length. A nano-scale PNP study used minimum grid spacing of ~0.5 nm for a Debye length of ~0.8 nm (roughly 1.5 elements per Debye length as a bare minimum for qualitative results).
  - Source: [Bhatt et al., Nano-scale PNP solution, PMC9974139](https://pmc.ncbi.nlm.nih.gov/articles/PMC9974139/)

- **COMSOL diffuse double layer model**: mesh parameters are explicitly "dependent on the Debye length to make sure the mesh is always well resolved." COMSOL uses a two-level sizing: a global `h_max` for the bulk and a much smaller `h_max_surf` at the electrode boundary.
  - Source: [COMSOL Diffuse Double Layer Documentation](https://doc.comsol.com/5.6/doc/com.comsol.help.models.chem.diffuse_double_layer/diffuse_double_layer.html)

- **Rule of thumb for research codes**: 5--10 elements within the first Debye length from the electrode, with total domain coverage of at least 10--20 Debye lengths to capture the full EDL-to-bulk transition. For high applied potentials where the EDL compresses (depletion zone), finer resolution is needed because the effective gradient length shrinks below the equilibrium Debye length.

- **Smaller Debye lengths need more mesh points**: the nondimensionalized Debye length (epsilon) directly controls how many spatial mesh points are required. As epsilon decreases, the number of required mesh points increases.
  - Source: [vssimex_pnp GitHub, adaptive time-stepping for PNP](https://github.com/daveboat/vssimex_pnp)

### 2.2 Geometric Mesh Grading Formula

For 1D problems (or the normal direction in a boundary layer mesh), geometric grading is standard:

Given domain [0, L] with boundary layer at x = 0, define N elements with a grading ratio r (each successive element is r times the previous):

```
h_1 = smallest element (at boundary)
h_k = h_1 * r^(k-1),   k = 1, ..., N

Total length: L = h_1 * (r^N - 1) / (r - 1)
```

So:
```
h_1 = L * (r - 1) / (r^N - 1)
```

**Practical values**:
- Grading ratio r = 1.1 to 1.3 for smooth transitions (1.2 is a common default)
- First element size h_1 should be ~lambda_D / 5 to ~lambda_D / 10
- Typically 20--40 elements in the graded region capture the full EDL

For example, with lambda_D = 1 nm, L_domain = 10 um, r = 1.2, and h_1 = 0.1 nm:
- N = log(1 + L*(r-1)/h_1) / log(r) ~ 63 elements in the graded direction
- This is far more efficient than a uniform mesh which would need ~100,000 elements

## 3. Adaptive Mesh Refinement (AMR) for PNP

### 3.1 A Posteriori Error Estimators

An adaptive finite element method for the nonlinear steady-state PNP equations has been developed where spatial adaptivity targets both geometrical singularities and boundary layer effects. The approach uses a posteriori error estimates to drive refinement.
- Source: [Adaptive FE approximation for steady-state PNP, Springer](https://link.springer.com/article/10.1007/s10444-022-09938-2)

Effective refinement criteria for PNP include:
- **Gradient-based**: Refine where |grad(phi)| or |grad(c_i)| exceeds a threshold. Most of the concentration gradient is near the electrodes, within 5--10% of the total domain thickness.
  - Source: [Hermes2d Nernst-Planck example](https://www.hpfem.org/hermes-doc/hermes-examples/html/src/hermes2d/examples/nernst-planck/timedep-adapt.html)
- **Residual-based**: Classical element-residual estimators measuring how well the discrete solution satisfies the PDE locally.
- **Goal-oriented (adjoint-based)**: Refine to minimize error in a specific quantity of interest (e.g., electrode current). This is the most efficient strategy when you care about a specific output.

### 3.2 Multi-Mesh Strategy

A key insight from the Hermes2d PNP example: different unknowns benefit from different meshes. The concentration field C needs much finer mesh near electrodes than the electric potential phi, because "the gradient of C near the boundaries will be higher than gradients of phi." Using separate meshes for different variables (multi-meshing) avoids over-refining where it is not needed.
- Source: [Hermes2d Nernst-Planck adaptive example](https://www.hpfem.org/hermes-doc/hermes-examples/html/src/hermes2d/examples/nernst-planck/timedep-adapt.html)

## 4. hp-Adaptivity and Spectral Methods for Boundary Layers

### 4.1 Exponential Convergence with hp-FEM

For singularly perturbed problems (which PNP is, with the Debye length as the small parameter), hp-FEM achieves exponential convergence that is robust with respect to the perturbation parameter:

- **p-version** on a fixed mesh: convergence rate O(sqrt(log W) / W) in the energy norm, where W is the polynomial degree.
- **hp-version** with geometric grading: convergence rate O(exp(-W / log W)), which is exponential -- far superior to any fixed-order method.
- Source: [p and hp spectral element methods for boundary layer problems, arXiv:2409.14426](https://arxiv.org/html/2409.14426)

The critical requirement for exponential convergence: one layer of "needle elements" of width O(p * epsilon) must be placed near the boundary (where epsilon is the perturbation parameter, i.e., the Debye length in PNP).
- Source: [Schwab & Suri, hp FEM for reaction-diffusion, SIAM J. Numer. Anal.](https://dx.doi.org/10.1137/S0036142997317602)

### 4.2 Geometric Mesh for hp-FEM (Schwab-Suri Framework)

The Schwab-Suri geometric mesh on [0, 1] refined toward x = 0 uses a grading parameter sigma in (0, 1):

```
x_j = sigma^(n-j),   j = 0, 1, ..., n
```

with x_0 = sigma^n (smallest element at boundary), x_n = 1. The element sizes grow geometrically away from the boundary by factor 1/sigma. Combined with polynomial degree p increasing linearly from the boundary, this yields the exponential convergence rate.
- Source: [Schwab & Suri, Boundary Layer Approximation by Spectral/hp Methods](https://www.math.uh.edu/~hjm/june1995/p00501-p00508.pdf)

**Relevance to PNP**: The Debye length plays the role of epsilon. A geometric mesh with sigma ~ 0.5 and 3--5 layers of refined elements, combined with polynomial degree p = 3--5, can resolve the EDL far more efficiently than uniform h-refinement.

## 5. Extended Finite Element Method (XFEM) for PNP

A mesh-independent approach: XFEM augments the standard FE space with enrichment functions derived from asymptotic analysis of the boundary layer structure. This captures steep gradients on coarse meshes without refinement.

- An XFEM for the Nernst-Planck-Poisson equations uses enrichment functions based on the known exponential decay structure of the EDL solution. This captures the steep gradient near the boundary even with coarse discretization in 1D and 2D.
- Source: [An extended FEM for the Nernst-Planck-Poisson equations, ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0167273824000791)

**Practical implication**: If you know the analytical form of the boundary layer (which for PNP is approximately exp(-x/lambda_D)), you can enrich the FE basis with this function. The mesh no longer needs to resolve the layer explicitly. This is particularly attractive when the Debye length is orders of magnitude smaller than the domain.

## 6. Stabilized Discretizations: Scharfetter-Gummel and EAFE

An alternative to mesh refinement is to use discretizations that are inherently stable on coarse meshes:

### 6.1 Scharfetter-Gummel (Exponential Fitting)

The Scharfetter-Gummel scheme finds the exact solution of the 1D drift-diffusion equation with constant coefficients on each element. This "exponential fitting" means the discrete solution captures exponentially varying carrier densities even on coarse meshes, requiring considerably fewer mesh points than standard methods.
- Source: [Scharfetter-Gummel discretization, UTDallas](https://personal.utdallas.edu/~frensley/minitech/ScharfGum.pdf)

### 6.2 Edge-Averaged Finite Element (EAFE)

EAFE generalizes Scharfetter-Gummel to unstructured meshes in 2D/3D. It stabilizes continuous linear FE discretizations for convection-dominated problems and satisfies a discrete maximum principle -- preventing spurious oscillations even when the mesh Peclet number is large. The EAFE scheme produces stable numerical solutions on meshes where standard Galerkin would oscillate wildly.
- Source: [EAFE for convection-diffusion, arXiv:2402.13347](https://arxiv.org/html/2402.13347v1)
- Source: [PyEAFE on PyPI](https://pypi.org/project/pyeafe/)

**Key point**: EAFE/Scharfetter-Gummel stabilization reduces but does not eliminate the need for mesh refinement in the EDL. The Poisson equation still needs resolution of the charge density, and accuracy of derived quantities (current, reaction rates) still benefits from refined meshes. However, the stabilization dramatically improves robustness at moderate resolution.

## 7. Moving Mesh / r-Adaptivity for PNP

### 7.1 Flux-Based Moving Mesh

A recent (2024) flux-based moving mesh method for PNP equations uses a monitor function based on the ionic flux to guide mesh point redistribution. The method:
- Keeps the total number of mesh points fixed
- Redistributes them to concentrate points where the flux is largest (i.e., in the EDL)
- Uses an adaptive step-size control algorithm
- Preserves energy dissipation properties of the PNP system
- Outperforms both traditional moving mesh FEM and fixed mesh FEM
- Source: [Flux-based moving mesh for PNP, ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0021999124004182)

### 7.2 r-Adaptivity in Firedrake

The **Movement** library provides mesh movement (r-adaptivity) methods in Firedrake. This relocates mesh vertices without changing topology -- useful for tracking a moving EDL front during transient simulations or parameter continuation.
- Source: [Firedrake mesh adaptation tools](https://github.com/mesh-adaptation)

## 8. Domain Decomposition: Separating EDL from Bulk

### 8.1 Two-Domain Approach

The idea: solve the full PNP equations only in a thin EDL domain (a few Debye lengths from the electrode) and use simplified electroneutral Nernst-Planck or Ohmic transport in the bulk. This avoids resolving the Debye length across the entire domain.

- Homogenization of PNP equations for ion transport in charged porous media shows that effective ionic diffusivities become tensors related to microstructure, and the effective permittivity depends on the ratio of Debye screening length to macroscopic length. This theoretical framework supports the domain decomposition approach.
  - Source: [Homogenization of PNP, SIAM J. Appl. Math.](https://epubs.siam.org/doi/10.1137/140968082)

- Multi-region solvers using OpenFOAM employ domain decomposition where each region is solved separately with an outer coupling loop.
  - Source: [Electrochemical transport on pore-scale, Springer](https://link.springer.com/article/10.1007/s00366-023-01828-5)

### 8.2 Practical Implementation

For a 1D PNP problem with electrode at x = 0:
1. **EDL domain**: [0, 10*lambda_D] -- solve full PNP with fine geometric mesh
2. **Bulk domain**: [10*lambda_D, L] -- solve electroneutral Nernst-Planck with coarse mesh
3. **Interface matching**: enforce continuity of concentrations, potential, and fluxes at the domain boundary

This decomposition reduces the total DOF dramatically and avoids the ill-conditioning from having both nm-scale and um-scale elements in one system.

## 9. Firedrake-Specific AMR and Mesh Tools

### 9.1 MeshHierarchy (Geometric Multigrid)

Firedrake's `MeshHierarchy` creates uniform refinement hierarchies via bisection:
```python
mesh = IntervalMesh(N, L)
hierarchy = MeshHierarchy(mesh, 4)  # 4 levels of uniform refinement
```
This is primarily for multigrid solvers, not adaptive refinement. It uniformly refines, so it does not help with non-uniform grading.
- Source: [Firedrake geometric multigrid demo](https://www.firedrakeproject.org/demos/geometric_multigrid.py.html)

### 9.2 Netgen Integration

Firedrake integrates with Netgen for mesh generation and supports a solve-mark-refine loop for adaptive refinement. Netgen can generate graded meshes with boundary layers.
- Source: [Firedrake Netgen integration](https://www.firedrakeproject.org/demos/netgen_mesh.py.html)

### 9.3 Animate (Anisotropic Metric-Based Adaptation)

The **Animate** toolkit enables anisotropic mesh adaptation in Firedrake using a Riemannian metric framework. Users define a metric field that controls element shape, orientation, and size. The metric is piecewise linear and continuous, with DOFs at mesh vertices. This is the most powerful mesh adaptation tool available in Firedrake.
- Source: [Animate GitHub](https://github.com/mesh-adaptation/animate)

### 9.4 Goalie (Goal-Oriented Adaptation)

**Goalie** implements goal-oriented mesh adaptation using adjoint error estimation. For PNP, you could define the goal functional as the electrode current and adapt the mesh to minimize error in that quantity specifically.
- Source: [Goalie, mesh-adaptation GitHub](https://mesh-adaptation.github.io/)

### 9.5 Movement (r-Adaptivity)

The **Movement** library provides mesh movement methods that reposition vertices without changing mesh topology.
- Source: [Movement, mesh-adaptation GitHub](https://mesh-adaptation.github.io/)

### 9.6 Building a Graded Mesh Manually in Firedrake

Since Firedrake's `IntervalMesh` creates uniform meshes, a geometrically graded mesh must be constructed manually:
```python
import numpy as np
from firedrake import Mesh
# Geometric grading: h_1 * r^(k-1) element sizes
r = 1.15  # grading ratio
N = 50    # number of elements
h1 = lambda_D / 5  # first element size
x = np.zeros(N + 1)
for k in range(1, N + 1):
    x[k] = x[k-1] + h1 * r**(k-1)
# Create mesh from coordinates using plex
```
Or use Gmsh/Netgen to generate the graded mesh externally and import it.

## 10. How Other Codes Handle the Multi-Scale EDL Problem

### 10.1 COMSOL

- Uses boundary layer meshing with physics-controlled mesh sizing tied to the Debye length
- Two-level element sizing: global `h_max` for bulk, refined `h_max_surf` at electrode
- Predefined parameter files ensure mesh scales with Debye length across different electrolyte concentrations
- Source: [COMSOL diffuse double layer](https://doc.comsol.com/5.6/doc/com.comsol.help.models.chem.diffuse_double_layer/diffuse_double_layer.html)

### 10.2 DUNE/PDELab

- Supports local mesh refinement with a variety of techniques
- PNP solver uses EAFE stabilization with monolithic Newton for the full PNP+NS system
- Built on FEniCS with PyEAFE module for edge-averaged finite element stabilization
- Source: [PDELab PNP solver](https://pdelab.github.io/)

### 10.3 MOOSE (Idaho National Lab)

- Discussion of PNP implementation highlights that mesh resolution on the order of nanometers is required, limiting the spatial domain that can be simulated
- Source: [MOOSE PNP discussion](https://github.com/idaholab/moose/discussions/22144)

## 11. Is the Problem Mesh-Related or Algebraic?

**Both, but they are coupled**. The convergence failure at high anodic potentials likely has contributions from:

### 11.1 Mesh-Related Issues
- **Depletion zone thinning**: At high potentials, the EDL structure changes dramatically. Ion depletion creates near-zero concentrations that compress the effective boundary layer to sub-Debye-length scales. A mesh designed for the equilibrium Debye length may be too coarse for the depleted state.
- **Exponential concentration ratios**: With dimensionless potential ~4 V_T, concentrations vary by exp(4) ~ 55x within the EDL. At higher potentials, this becomes exp(10) ~ 22,000x, creating extreme gradients.

### 11.2 Algebraic/Jacobian Issues
- **Condition number scales exponentially with potential**: The stiffness matrix conditioning is dominated by the exponential Boltzmann factor exp(z*phi/V_T). For z*phi/V_T = 10, this factor is ~22,000, directly degrading the condition number.
- **Near-zero concentrations**: In depletion zones, concentrations approach machine epsilon, causing division-by-zero-like behavior in the Jacobian (many PNP formulations involve 1/c_i terms in the drift coefficient).
- **Formulation matters more than mesh**: Using the primitive formulation (not the Slotboom/exponential transformation) avoids the worst conditioning problems.
  - Source: [Lu et al., PMC2922884](https://pmc.ncbi.nlm.nih.gov/articles/PMC2922884/)

### 11.3 Recommendation

For the specific problem described (failure at ~4 V_T above ground with depletion zone stiffness):

1. **First check**: Is the mesh resolving the depleted EDL? At high anodic potential, the effective gradient length may be lambda_D / sqrt(exp(z*phi/V_T)), requiring much finer resolution than the equilibrium Debye length suggests. Add 5--10 elements within this compressed layer.

2. **Then address algebraic issues**: Use log-concentration variables (c_i = exp(u_i)) to avoid near-zero concentrations and improve Jacobian conditioning. Combine with EAFE/Scharfetter-Gummel stabilization.

3. **Consider XFEM enrichment**: If the analytical EDL profile is known, enriching the FE basis with the exponential boundary layer function can resolve the layer without mesh refinement.

4. **Use continuation on the mesh too**: Start with a coarse problem, refine the mesh as the continuation parameter increases, using the coarse solution projected onto the fine mesh as the initial guess.

## 12. Summary of Actionable Strategies

| Strategy | Effort | Impact | Firedrake Support |
|----------|--------|--------|-------------------|
| Geometric mesh grading (r=1.15--1.25) | Low | High | Manual or via Gmsh/Netgen |
| EAFE/Scharfetter-Gummel stabilization | Medium | High | Via PyEAFE or manual implementation |
| Log-concentration transformation | Low | High | Direct in variational form |
| Animate metric-based adaptation | Medium | High | Yes (Animate package) |
| Goal-oriented AMR (Goalie) | High | Very High | Yes (Goalie package) |
| hp-adaptivity with geometric grading | High | Very High | Limited (manual p-refinement) |
| XFEM enrichment | High | Very High | Manual enrichment in FE space |
| Domain decomposition (EDL + bulk) | High | High | Manual implementation |
| Moving mesh (flux-based) | Medium | Medium | Movement package |
| Mesh + parameter co-continuation | Low | Medium | Manual in continuation loop |

## Sources

- [Lu et al., PNP FEM for biomolecular systems, PMC2922884](https://pmc.ncbi.nlm.nih.gov/articles/PMC2922884/)
- [Bhatt et al., Nano-scale PNP solution, PMC9974139](https://pmc.ncbi.nlm.nih.gov/articles/PMC9974139/)
- [Adaptive FE for steady-state PNP, Springer](https://link.springer.com/article/10.1007/s10444-022-09938-2)
- [Flux-based moving mesh for PNP, ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0021999124004182)
- [XFEM for Nernst-Planck-Poisson, ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0167273824000791)
- [hp spectral methods for boundary layers, arXiv:2409.14426](https://arxiv.org/html/2409.14426)
- [Schwab & Suri, hp FEM for reaction-diffusion, SIAM](https://dx.doi.org/10.1137/S0036142997317602)
- [Scharfetter-Gummel scheme, UTDallas](https://personal.utdallas.edu/~frensley/minitech/ScharfGum.pdf)
- [EAFE for convection-diffusion, arXiv](https://arxiv.org/html/2402.13347v1)
- [PyEAFE on PyPI](https://pypi.org/project/pyeafe/)
- [COMSOL diffuse double layer](https://doc.comsol.com/5.6/doc/com.comsol.help.models.chem.diffuse_double_layer/diffuse_double_layer.html)
- [PDELab PNP solver](https://pdelab.github.io/)
- [Homogenization of PNP, SIAM](https://epubs.siam.org/doi/10.1137/140968082)
- [Firedrake geometric multigrid](https://www.firedrakeproject.org/demos/geometric_multigrid.py.html)
- [Firedrake Netgen integration](https://www.firedrakeproject.org/demos/netgen_mesh.py.html)
- [Animate toolkit, GitHub](https://github.com/mesh-adaptation/animate)
- [Goalie/Movement/Animate docs](https://mesh-adaptation.github.io/)
- [Firedrake mesh adaptation overview](https://github.com/mesh-adaptation)
- [Hermes2d Nernst-Planck adaptive](https://www.hpfem.org/hermes-doc/hermes-examples/html/src/hermes2d/examples/nernst-planck/timedep-adapt.html)
- [vssimex_pnp, GitHub](https://github.com/daveboat/vssimex_pnp)
- [MOOSE PNP discussion](https://github.com/idaholab/moose/discussions/22144)
- [Electrochemical transport pore-scale, Springer](https://link.springer.com/article/10.1007/s00366-023-01828-5)
