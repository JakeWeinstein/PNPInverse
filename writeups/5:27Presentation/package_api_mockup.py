"""
HYPOTHETICAL PACKAGE-API MOCKUP -- NOT REAL, NOT RUNNABLE.

The `pnpbv` package does not exist. A design doc was drafted
(.plans/pnpbv-migration/PLAN.md, pnpbv_port/SCOPING.md) but was NEVER
implemented -- this mockup only illustrates what that hypothetical API could
look like, for the funder / electrochemist talk slide "Where this could go:
a Python package." The only real code today is the research solver under
Forward/bv_solver/.

Design intent: a clean separation between model specification (species,
reactions, terms, boundary conditions) and the numerical machinery
(formulation, meshing, continuation, convergence). Counterions are modeled
as analytic species via an `analytic` flag (folding the design doc's separate
`AnalyticCounterion` class into `Species`). Slide artifact only.
"""

import pnpbv as pb              # hypothetical -- does not exist
import numpy as np

K0_2E, K0_4E = 1e-7, 1e-7       # measured / fitted kinetics (placeholders)

# ---- Species: dynamic (PDE-solved) and analytic (Boltzmann) share ONE type ----
#      concentrations in mol/L ; D in m^2/s
O2   = pb.Species("O2",    z=0,  D=2.0e-9,  c_bulk=1.2e-3)                # dissolved reactant (O2-saturated)
H2O2 = pb.Species("H2O2",  z=0,  D=1.0e-9,  c_bulk=0.0)                   # 2e- product
H    = pb.Species("H+",    z=+1, D=9.31e-9, c_bulk=1e-4, role="proton")   # pH 4

K    = pb.Species("K+",    z=+1, c_bulk=0.2, analytic=True,               # supporting cation -> Boltzmann:
                  steric=pb.Bikerman(a=3.0e-10))                          #   no transport DOF, finite size
SO4  = pb.Species("SO4--", z=-2, c_bulk=0.1, analytic=True,
                  steric=pb.Bikerman(a=3.0e-10))

# ---- Reactions: EXPLICIT stoichiometry via reactant/product maps ----
r2e = pb.BVReaction("ORR_2e", reactants={O2: 1, H: 2}, products={H2O2: 1},
                    n_e=2, E_eq=0.695, alpha_a=0.5, alpha_c=0.5, k0=K0_2E, log_rate=True)  # 2e- (wanted)
r4e = pb.BVReaction("ORR_4e", reactants={O2: 1, H: 4}, products={},       # O2 + 4 H+ + 4 e- -> 2 H2O
                    n_e=4, E_eq=1.23,  alpha_a=0.5, alpha_c=0.5, k0=K0_4E, log_rate=True)  # 4e- (loss)

# ---- Compose the problem: model specification ends here ----
prob = pb.Problem(
    species     = [O2, H2O2, H, K, SO4],     # dynamic + analytic in one list, split by the flag
    reactions   = [r2e, r4e],
    electrode   = pb.SternElectrode(C_stern=0.20, l_eff=100e-6),   # double-layer -> Robin BC
    formulation = "logc_muh",
    initializer = pb.DebyeBoltzmannIC(),
)

# ---- Solve: numerical machinery (2D mesh, formulation, continuation, recovery) ----
solver = pb.Solver(prob, mesh=pb.graded_rectangle(Nx=8, Ny=80, beta=3.0))   # 2D graded rectangle
sol = solver.solve_grid(v_rhe=np.linspace(-0.4, 0.8, 16))                   # continuation auto-selected

# ---- Query: electrochemical quantities ----
sol.current_density                  # disk i-V
sol.reaction_currents["ORR_2e"]      # per-pathway current -> selectivity
sol.surface_pH


# ================= extensibility =================

# Change stoichiometry / kinetics -> edit the maps, nothing else re-derived
r4e = pb.BVReaction("ORR_4e", reactants={O2: 1, H: 4}, products={}, n_e=4, E_eq=1.20, k0=K0_4E)

# Add a mechanism still under model-selection -> compose one more object
# (flagged experimental, because Phase 6b identifiability is still open)
import pnpbv.experimental as pbe
moh = pbe.physics.cation_hydrolysis_with_langmuir_cap(cation=K, proton=H, pKa0=4.3, gamma_max=5.6e-6)
prob = pb.Problem(species=[O2, H2O2, H, K, SO4], reactions=[r2e, r4e],
                  adsorbates=[moh], electrode=pb.SternElectrode(C_stern=0.20, l_eff=100e-6))
# same solver, no re-derivation
