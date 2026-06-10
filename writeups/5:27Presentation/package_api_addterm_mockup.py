"""
HYPOTHETICAL PACKAGE-API MOCKUP (POWER-USER / ADD-A-TERM) -- NOT REAL, NOT RUNNABLE.

Companion to package_api_mockup.py for the funder / electrochemist talk.
The `pnpbv` package does not exist; this illustrates a hypothetical advanced
layer -- importing the default PNP + steric weak-form residual and adding
custom terms to it when no built-in class covers the desired physics.

This raw-residual `WeakForm` layer is not even in the tabled design doc
(.plans/pnpbv-migration), which only sketched composable objects; it is shown
as a possible future extension. Source-term signs are illustrative
("per the F = 0 convention"). The only real code today is Forward/bv_solver/.
Slide artifact only.
"""

import numpy as np
import pnpbv as pb                              # hypothetical -- does not exist
from pnpbv.formulations import pnp_steric       # default PNP + electromigration + Bikerman-steric residual
from ufl import grad, inner                      # the residual IS a UFL form -- standard FEM pieces

# Assumes `prob` from package_api_mockup.py (species + reactions + electrode + formulation).

# 1. Import the default weak form for this problem (diffusion + migration + steric, already assembled)
res = pnp_steric(prob)            # a WeakForm object terms can be added to

# 2. Exposes the symbolic context in *physical* concentrations
#    (formulation-consistent: the log-c / mu_H variables stay internal)
c, w   = res.fields, res.tests    # c["H+"], w["H+"], ...  concentrations + matching test functions
dx, ds = res.dx, res.ds           # 2D bulk measure ; boundaries: ds.electrode, ds.bulk

# 3. ADD CUSTOM TERMS -- each is one UFL expression added to the residual (F = 0)
#    Example: homogeneous peroxide decomposition  H2O2 -> 1/2 O2 + H2O  (first-order, rate k_d)
k_d = 1e-3
res.add( -k_d       * c["H2O2"] * w["H2O2"] * dx )    # sink on H2O2
res.add( +0.5 * k_d * c["H2O2"] * w["O2"]   * dx )    # matched source on O2   (signs per the F=0 convention)
#   forced convection in one line (e.g. RDE):  res.add( inner(u_flow, grad(c["O2"])) * w["O2"] * dx )

# 4. Adjust a boundary condition if required (terms are the focus, but BCs are editable too)
res.set_bc( "H+", pb.Robin(flux=j_H, on=ds.electrode) )   # swap/add a BC: Robin / Dirichlet / NoFlux

# 5. Same solver, same continuation + failed-step recovery -- the new term rides along
solver = pb.Solver.from_weak_form(res, mesh=pb.graded_rectangle(Nx=8, Ny=80, beta=3.0))
sol = solver.solve_grid(v_rhe=np.linspace(-0.4, 0.8, 16))
