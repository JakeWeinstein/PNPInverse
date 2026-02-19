import copy

import firedrake.adjoint as adj

from Utils.generate_noisy_data_robin import generate_noisy_data_robin
from Helpers.Infer_RobinBC_from_data_helpers import make_objective_and_grad

params = {
    "snes_type": "newtonls",
    "snes_max_it": 100,
    "snes_atol": 1e-8,
    "snes_rtol": 1e-8,
    "snes_linesearch_type": "bt",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "robin_bc": {
        "kappa": [0.8, 0.8],
        "c_inf": [0.01, 0.01],
        "electrode_marker": 1,
        "concentration_marker": 3,
        "ground_marker": 3,
    },
}

D_true = [1.0, 1.0]
n_species = len(D_true)
z_vals = [1 if i % 2 == 0 else -1 for i in range(n_species)]
a_vals = [0.0 for _ in range(n_species)]
c0_vals = [0.1 for _ in range(n_species)]

#                   (n_species, order, dt, t_end, z_vals, D_vals, a_vals, phi_applied, c0_vals, phi0, params)
solver_params_gen = [n_species, 1, 1e-2, 0.1, z_vals, D_true, a_vals, 0.05, c0_vals, 5, params]

# Data generation should not be taped.
with adj.stop_annotating():
    data = generate_noisy_data_robin(solver_params_gen)

clean_c = list(data[:n_species])
phi_vec = data[n_species]

noisy_c = list(data[n_species + 1 : n_species * 2 + 1])
phi_noisy = data[-1]

kappa_guess = [1.0 for _ in range(n_species)]
params_guess = copy.deepcopy(params)
params_guess["robin_bc"]["kappa"] = kappa_guess

solver_params = [n_species, 1, 1e-2, 0.1, z_vals, D_true, a_vals, 0.05, c0_vals, 5, params_guess]

rf = make_objective_and_grad(solver_params, clean_c, phi_vec)

kappa_lb = 1e-8
bounds = [[kappa_lb for _ in range(n_species)], [None for _ in range(n_species)]]
kappa_opt = adj.minimize(
    rf,
    "L-BFGS-B",
    bounds=bounds,
    tol=1e-8,
    options={"disp": True},
)

if not isinstance(kappa_opt, (list, tuple)):
    kappa_opt = [kappa_opt]
kappa_opt_vals = [float(k.dat.data[0]) for k in kappa_opt]

print("kappa =", ", ".join(str(v) for v in kappa_opt_vals))
