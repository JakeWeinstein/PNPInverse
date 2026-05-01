#!/usr/bin/env python3
"""Benchmark: autograd vs finite-difference gradient computation time.

Compares wall-clock time for gradient evaluation using PyTorch autograd
(single forward + backward) versus central finite differences (2*N forward
evaluations for N parameters) on the NN ensemble surrogate.

Usage
-----
    python scripts/benchmark_autograd_vs_fd.py

Requires a trained NN ensemble at the default path.  Adjust
``ENSEMBLE_DIR`` below if your ensemble lives elsewhere.
"""

from __future__ import annotations

import time

import numpy as np

ENSEMBLE_DIR = "data/surrogate_models/nn_ensemble/D3-deeper"
N_WARMUP = 5
N_ITERS = 100


def benchmark() -> None:
    """Run autograd vs FD gradient benchmark and print results."""
    from Surrogate.ensemble import load_nn_ensemble
    from Surrogate.objectives import SurrogateObjective

    print(f"Loading ensemble from: {ENSEMBLE_DIR}")
    model = load_nn_ensemble(ENSEMBLE_DIR)

    n_eta = model.n_eta
    rng = np.random.default_rng(42)
    target_cd = rng.standard_normal(n_eta) * 0.01
    target_pc = rng.standard_normal(n_eta) * 0.001

    x = np.array([-3.0, -5.0, 0.3, 0.5])

    # --- Autograd path ---
    obj_auto = SurrogateObjective(model, target_cd, target_pc, secondary_weight=1.0)
    assert obj_auto._use_autograd, "Expected autograd to be enabled for NN ensemble"

    # Warmup
    for _ in range(N_WARMUP):
        obj_auto.objective_and_gradient(x)

    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        obj_auto.objective_and_gradient(x)
    t_autograd = (time.perf_counter() - t0) / N_ITERS

    # --- FD path (force disable autograd) ---
    obj_fd = SurrogateObjective(model, target_cd, target_pc, secondary_weight=1.0)
    obj_fd._use_autograd = False

    # Warmup
    for _ in range(N_WARMUP):
        obj_fd.objective_and_gradient(x)

    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        obj_fd.objective_and_gradient(x)
    t_fd = (time.perf_counter() - t0) / N_ITERS

    # --- Verify gradients match ---
    _, g_auto = obj_auto.objective_and_gradient(x)
    _, g_fd = obj_fd.objective_and_gradient(x)
    rel_err = np.max(np.abs(g_auto - g_fd) / np.maximum(np.abs(g_fd), 1e-12))

    # --- Report ---
    print(f"\nBenchmark results ({N_ITERS} iterations):")
    print(f"  FD:            {t_fd * 1000:.2f} ms/eval")
    print(f"  Autograd:      {t_autograd * 1000:.2f} ms/eval")
    print(f"  Speedup:       {t_fd / t_autograd:.1f}x")
    print(f"  Max rel error: {rel_err:.2e}")
    print()

    if t_fd / t_autograd >= 4.0:
        print("  PASS: >= 4x speedup achieved")
    else:
        print(f"  NOTE: speedup is {t_fd / t_autograd:.1f}x (target was >= 4x)")

    if rel_err < 1e-3:
        print("  PASS: gradient relative error < 0.1%")
    else:
        print(f"  WARN: gradient relative error {rel_err:.2e} exceeds 0.1%")


if __name__ == "__main__":
    benchmark()
