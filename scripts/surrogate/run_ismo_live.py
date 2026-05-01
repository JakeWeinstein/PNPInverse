#!/usr/bin/env python
"""Run ISMO with real Firedrake PDE solver.

Usage:
    source ../venv-firedrake/bin/activate
    python scripts/surrogate/run_ismo_live.py 2>&1 | tee StudyResults/ismo/run.log
"""
from __future__ import annotations

import os
import sys

# Fix libomp conflict between Firedrake/PETSc and PyTorch on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
os.chdir(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts._bv_common import (
    setup_firedrake_env,
    I_SCALE,
    FOUR_SPECIES_CHARGED,
    SNES_OPTS_CHARGED,
    make_bv_solver_params,
)
setup_firedrake_env()

import numpy as np
from Forward.steady_state import SteadyStateConfig
from Forward.bv_solver import make_graded_rectangle_mesh
from Surrogate.ensemble import load_nn_ensemble
from Surrogate.ismo import run_ismo, ISMOConfig, AcquisitionStrategy

# ── Constants (match overnight_train_v11.py exactly) ──
DT = 0.5
MAX_SS_STEPS = 100
T_END = DT * MAX_SS_STEPS
MESH_NX = 8
MESH_NY = 200
MESH_BETA = 3.0

# ── Build PDE solver config ──
print("Setting up PDE solver...", flush=True)

base_sp = make_bv_solver_params(
    eta_hat=0.0, dt=DT, t_end=T_END,
    species=FOUR_SPECIES_CHARGED,
    snes_opts=SNES_OPTS_CHARGED,
)

steady = SteadyStateConfig(
    relative_tolerance=1e-4,
    absolute_tolerance=1e-8,
    consecutive_steps=4,
    max_steps=MAX_SS_STEPS,
    flux_observable="total_species",
    verbose=False,
)

mesh = make_graded_rectangle_mesh(Nx=MESH_NX, Ny=MESH_NY, beta=MESH_BETA)

pde_solver_kwargs = {
    "base_solver_params": base_sp,
    "steady": steady,
    "observable_scale": -I_SCALE,
    "mesh": mesh,
    "mesh_params": (MESH_NX, MESH_NY, MESH_BETA),
}

# ── Load training data ──
print("Loading training data...", flush=True)
training_data = np.load("data/surrogate_models/training_data_merged.npz")
all_params = training_data["parameters"]
all_cd = training_data["current_density"]
all_pc = training_data["peroxide_current"]
phi_applied = training_data["phi_applied"]

# Use train/test split
split = np.load("data/surrogate_models/split_indices.npz")
train_idx = split["train_idx"]
test_idx = split["test_idx"]

params_train = all_params[train_idx]
cd_train = all_cd[train_idx]
pc_train = all_pc[train_idx]

# Target: use first test sample (synthetic parameter recovery)
target_cd = all_cd[test_idx[0]]
target_pc = all_pc[test_idx[0]]
true_params = all_params[test_idx[0]]

print(f"Training samples: {len(params_train)}", flush=True)
print(f"Target from test sample 0: k0=[{true_params[0]:.4e},{true_params[1]:.4e}] "
      f"alpha=[{true_params[2]:.4f},{true_params[3]:.4f}]", flush=True)

# ── Load surrogate ──
print("Loading NN ensemble surrogate...", flush=True)
surrogate = load_nn_ensemble(
    "data/surrogate_models/nn_ensemble/D1-default",
    n_members=5,
)

# ── Configure & run ISMO ──
output_dir = "StudyResults/ismo"
os.makedirs(output_dir, exist_ok=True)

config = ISMOConfig(
    max_iterations=5,
    samples_per_iteration=30,
    total_pde_budget=200,
    convergence_rtol=0.05,
    convergence_atol=1e-4,
    surrogate_type="nn_ensemble",
    acquisition_strategy=AcquisitionStrategy.OPTIMIZER_TRAJECTORY,
    warm_start_retrain=False,  # warm-start needs weight correction (use ismo_retrain.py for proper warm-start)
    retrain_epochs=2000,
    output_dir=output_dir,
    verbose=True,
)

print(f"\nTrue parameters: {true_params}", flush=True)
print(f"Voltage grid: {len(phi_applied)} points\n", flush=True)

result = run_ismo(
    surrogate=surrogate,
    target_cd=target_cd,
    target_pc=target_pc,
    training_params=params_train,
    training_cd=cd_train,
    training_pc=pc_train,
    phi_applied=phi_applied,
    bounds_k0_1=(1e-6, 1.0),
    bounds_k0_2=(1e-7, 0.1),
    bounds_alpha=(0.1, 0.9),
    pde_solver_kwargs=pde_solver_kwargs,
    config=config,
)

print(f"\n{'='*60}", flush=True)
print(f"ISMO COMPLETE", flush=True)
print(f"  Converged: {result.converged}", flush=True)
print(f"  Reason: {result.termination_reason}", flush=True)
print(f"  Iterations: {result.n_iterations}", flush=True)
print(f"  PDE evals: {result.total_pde_evals}", flush=True)
print(f"  Final params: {result.final_params}", flush=True)
print(f"  Final loss: {result.final_loss:.6e}", flush=True)
print(f"  True params: {tuple(true_params)}", flush=True)
print(f"  Wall time: {result.total_wall_time_s:.1f}s", flush=True)
print(f"{'='*60}", flush=True)
