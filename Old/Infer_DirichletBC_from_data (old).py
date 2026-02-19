from Utils.generate_noisy_data import generate_noisy_data
from Utils.forsolve import *
import numpy as np
from Helpers.Infer_DirichletBC_from_data_helpers import *
import firedrake.adjoint as adj

params = {
            'snes_type': 'newtonls',
            'snes_max_it': 100,
            'snes_atol': 1e-8,
            'snes_rtol': 1e-8,
            'snes_linesearch_type': 'bt',
            'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'
        }

D_true = [1.0, 1.0]
n_species = len(D_true)
z_vals = [1 if i % 2 == 0 else -1 for i in range(n_species)]
a_vals = [0.0 for _ in range(n_species)]
phi0_true = 1
c0_vals = [0.1 for _ in range(n_species)]

#                   (n_species, order, dt, t_end, z_vals, D_vals, a_vals, phi_applied, c0_vals, phi0, params)
solver_params_gen = [n_species, 1, 1e-2, 0.1, z_vals, D_true, a_vals, 0.05, c0_vals, phi0_true, params]

# data generation should not be taped
with adj.stop_annotating():
    data = generate_noisy_data(solver_params_gen)

clean_c = list(data[:n_species])
phi_vec = data[n_species]

noisy_c = list(data[n_species + 1 : n_species * 2 + 1])
phi_noisy = data[-1]

phi0_guess = 10

solver_params = [n_species, 1, 1e-2, 0.1, z_vals, D_true, a_vals, 0.05, c0_vals, phi0_guess, params]

rf = make_objective_and_grad(solver_params, clean_c,phi_vec)

phi0_lb = 1e-8
phi0_opt = adj.minimize(
    rf,
    "L-BFGS-B",
    bounds=(phi0_lb, None),
    tol=1e-8,
    options={"disp": True},
)

phi0_opt_value = phi0_opt.dat.data[0]

print(f"phi0 = {phi0_opt_value}")
