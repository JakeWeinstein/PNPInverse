"""Butler-Volmer forward solver convergence tests.

Run from the PNPInverse/ directory::

    python InferenceScripts/test_bv_forward.py [--strategy STRATEGY] [--eta ETA]

Strategies
----------
A   Neutral species (O2+H2O2, z=0), η=0 — should always converge
B   Neutral species, small cathodic overpotential (η = −1 V_T)
C   Neutral species, moderate overpotential (η = −5 V_T), potential continuation
D   Neutral species, full V-I sweep (η from 0 to −20 V_T) with continuation
E   Charged 2-species (H+ + Cl-, z=[+1,-1]), η=0 — tests Poisson coupling
F   Charged 2-species, potential continuation to η = −5 V_T
all Run A → F in order, stop on first failure

Each strategy prints a PASS/FAIL line with the computed electrode flux.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("FIREDRAKE_TSFC_KERNEL_CACHE_DIR", "/tmp/firedrake-tsfc")
os.environ.setdefault("PYOP2_CACHE_DIR", "/tmp/pyop2")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import firedrake as fd
from Forward.bv_solver import build_context, build_forms, set_initial_conditions, forsolve_bv, solve_bv_with_continuation
from Forward.params import SolverParams

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
V_T = 0.025693  # thermal voltage at 25 °C, V
F = 96485.3329   # C/mol
R = 8.314462618  # J/(mol·K)
T = 298.15       # K

# Physical parameters (from Excel / Mangan2025)
D_O2    = 1.5e-9    # m²/s
D_H2O2  = 1.6e-9    # m²/s
D_Hplus = 9.311e-9  # m²/s
D_Clmin = 2.032e-9  # m²/s (approx Cl- for simple charged case)

c_O2_bulk   = 5e-4   # mol/L = 0.5 mol/m³ (O₂ sat. at pH 4)
c_H2O2_bulk = 0.0    # produced at electrode

# Kinetics (pH 4, Cs+ case, approximate)
k0_phys = 2.4e-8   # m/s  (from j0 = 2.33e-4 mA/cm² with n=2, c=0.5 mol/m³)
alpha_O2 = 0.627   # from Tafel slope 41 mV/dec → α = V_T·ln10/41e-3 = 0.627

# Reference scales
L_ref  = 1e-4     # m  (100 µm diffusion layer)
D_ref  = np.sqrt(D_O2 * D_H2O2)   # geometric mean
kappa_scale = D_ref / L_ref

# Dimensionless k0
k0_hat = k0_phys / kappa_scale
print(f"[params] D_ref={D_ref:.3e} m²/s, kappa_scale={kappa_scale:.3e} m/s")
print(f"[params] k0_phys={k0_phys:.3e} m/s, k0_hat={k0_hat:.5f}")
print(f"[params] V_T={V_T*1000:.3f} mV, L_ref={L_ref*1e6:.0f} µm")

# ---------------------------------------------------------------------------
# PETSc solver options — try progressively more conservative settings
# ---------------------------------------------------------------------------

SNES_OPTS_STANDARD = {
    "snes_type": "newtonls",
    "snes_max_it": 50,
    "snes_atol": 1e-8,
    "snes_rtol": 1e-10,
    "snes_stol": 1e-12,
    "snes_linesearch_type": "bt",
    "snes_linesearch_maxlambda": 1.0,       # PETSc ≥3.24 (was maxstep)
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_mumps_icntl_8": 77,               # automatic MUMPS row/col scaling
    "mat_mumps_icntl_14": 60,              # increase fill-factor for pivoting
}

SNES_OPTS_CONSERVATIVE = {
    "snes_type": "newtonls",
    "snes_max_it": 200,
    "snes_atol": 1e-7,
    "snes_rtol": 1e-10,
    "snes_stol": 1e-12,
    "snes_linesearch_type": "l2",           # L2 line search: more conservative
    "snes_linesearch_maxlambda": 0.5,       # PETSc ≥3.24 (was maxstep); limit to 50 %
    "snes_divergence_tolerance": 1e12,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_mumps_icntl_8": 77,               # automatic MUMPS row/col scaling
    "mat_mumps_icntl_14": 60,              # increase fill-factor for pivoting
}

SNES_OPTS_PICARD = {
    # Picard / damped-Richardson: no Jacobian inversions, very stable
    "snes_type": "nrichardson",
    "snes_max_it": 500,
    "snes_atol": 1e-6,
    "snes_rtol": 1e-8,
    "snes_linesearch_type": "l2",
    "snes_linesearch_maxlambda": 0.3,       # PETSc ≥3.24 (was maxstep)
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_mumps_icntl_8": 77,
    "mat_mumps_icntl_14": 60,
}

# ---------------------------------------------------------------------------
# Helper: build solver_params for neutral O2+H2O2
# ---------------------------------------------------------------------------

def neutral_species_params(
    *,
    phi_applied_V: float,
    snes_opts: dict | None = None,
    dt: float = 0.01,
    t_end: float = 1.0,
    clip_exponent: bool = True,
    exponent_clip: float = 50.0,
    regularize_conc: bool = True,
    conc_floor: float = 1e-8,
    use_eta_in_bv: bool = True,
    k0_override: float | None = None,
) -> list:
    """Build SolverParams for the 2-species neutral (O2+H2O2) BV problem."""
    if snes_opts is None:
        snes_opts = SNES_OPTS_CONSERVATIVE

    k0 = k0_hat if k0_override is None else k0_override

    params = dict(snes_opts)
    params["bv_bc"] = {
        "k0":               [k0, k0],
        "alpha":            [alpha_O2, 1.0 - alpha_O2],
        "stoichiometry":    [-1, +1],        # O2 consumed, H2O2 produced
        "c_ref":            [1.0, 0.0],      # nondim: O2 bulk = 1, H2O2 ref = 0
        "E_eq_v":           0.0,
        "electrode_marker":      1,
        "concentration_marker":  3,
        "ground_marker":         3,
    }
    params["bv_convergence"] = {
        "clip_exponent":            clip_exponent,
        "exponent_clip":            exponent_clip,
        "regularize_concentration": regularize_conc,
        "conc_floor":               conc_floor,
        "use_eta_in_bv":            use_eta_in_bv,
    }
    params["nondim"] = {
        "enabled":                           True,
        "diffusivity_scale_m2_s":            D_ref,
        "concentration_scale_mol_m3":        c_O2_bulk * 1000.0,   # mol/m³
        "length_scale_m":                    L_ref,
        "potential_scale_v":                 V_T,
        "kappa_inputs_are_dimensionless":    True,
        "diffusivity_inputs_are_dimensionless": True,
        "concentration_inputs_are_dimensionless": True,
        "potential_inputs_are_dimensionless":    True,
        "time_inputs_are_dimensionless":         True,
    }

    # D_vals already in nondim: D_i / D_ref
    D_O2_hat   = D_O2 / D_ref
    D_H2O2_hat = D_H2O2 / D_ref

    return SolverParams.from_list([
        2,                  # n_species
        1,                  # FE order
        dt,                 # dt (nondim)
        t_end,              # t_end (nondim)
        [0, 0],             # z_vals (neutral)
        [D_O2_hat, D_H2O2_hat],   # D_vals (nondim)
        [0.0, 0.0],         # a_vals (no steric)
        phi_applied_V,      # phi_applied (nondim = η/V_T)
        [1.0, 0.0],         # c0_vals (nondim: O2=1, H2O2=0)
        0.0,                # phi0
        params,
    ])


def charged_species_params(
    *,
    phi_applied_V: float,
    snes_opts: dict | None = None,
    dt: float = 0.01,
    t_end: float = 1.0,
) -> list:
    """Build SolverParams for a simple 2-species charged (H+/Cl-) BV test."""
    if snes_opts is None:
        snes_opts = SNES_OPTS_CONSERVATIVE

    # Reference scales for charged case
    c_bulk = 0.1 * 1000.0   # 0.1 M → 100 mol/m³
    D_charged_ref = np.sqrt(D_Hplus * D_Clmin)
    k_scale_charged = D_charged_ref / L_ref
    k0_hat_charged = k0_phys / k_scale_charged
    D_Hp_hat = D_Hplus / D_charged_ref
    D_Cl_hat = D_Clmin / D_charged_ref

    params = dict(snes_opts)
    params["bv_bc"] = {
        "k0":            [k0_hat_charged, 0.0],   # H+ consumed; Cl- no reaction
        "alpha":         [0.5, 0.5],
        "stoichiometry": [-1, 0],
        "c_ref":         [1.0, 1.0],
        "E_eq_v":        0.0,
        "electrode_marker":      1,
        "concentration_marker":  3,
        "ground_marker":         3,
    }
    params["bv_convergence"] = {
        "clip_exponent": True,
        "exponent_clip": 50.0,
        "regularize_concentration": True,
        "conc_floor": 1e-8,
        "use_eta_in_bv": True,
    }
    params["nondim"] = {
        "enabled": True,
        "diffusivity_scale_m2_s":         D_charged_ref,
        "concentration_scale_mol_m3":     c_bulk,
        "length_scale_m":                 L_ref,
        "potential_scale_v":              V_T,
        "kappa_inputs_are_dimensionless": True,
        "diffusivity_inputs_are_dimensionless": True,
        "concentration_inputs_are_dimensionless": True,
        "potential_inputs_are_dimensionless": True,
        "time_inputs_are_dimensionless": True,
    }

    return SolverParams.from_list([
        2, 1, dt, t_end,
        [1, -1],
        [D_Hp_hat, D_Cl_hat],
        [0.0, 0.0],
        phi_applied_V,
        [1.0, 1.0],
        0.0,
        params,
    ])


# ---------------------------------------------------------------------------
# Test runner helper
# ---------------------------------------------------------------------------

def run_single_solve(sp, label: str, *, print_interval: int = 10) -> tuple[bool, float | None]:
    """Run a single BV forward solve, return (converged, electrode_flux)."""
    try:
        ctx = build_context(sp)
        ctx = build_forms(ctx, sp)
        set_initial_conditions(ctx, sp, blob=False)
        U = forsolve_bv(ctx, sp, print_interval=print_interval)

        # Compute total O2 flux at electrode (boundary marker 1).
        mesh = ctx["mesh"]
        ds = fd.Measure("ds", domain=mesh)
        n_vec = fd.FacetNormal(mesh)
        c_O2 = fd.split(U)[0]
        phi_field = fd.split(U)[-1]
        D_O2_hat = float(ctx["nondim"]["D_model_vals"][0])
        z0 = float(sp[4][0])
        em = float(ctx["nondim"]["electromigration_prefactor"])
        # Only add migration term when z != 0 (avoids fd.grad(0.0) issue).
        if abs(z0) > 1e-14:
            Jflux_O2 = D_O2_hat * (fd.grad(c_O2) + em * z0 * c_O2 * fd.grad(phi_field))
        else:
            Jflux_O2 = D_O2_hat * fd.grad(c_O2)
        flux_val = float(fd.assemble(fd.dot(Jflux_O2, n_vec) * ds(1)))
        print(f"  → electrode O2 flux (nondim) = {flux_val:.6e}")
        return True, flux_val
    except Exception as e:
        print(f"  → FAILED: {e}")
        traceback.print_exc()
        return False, None


def print_result(label: str, passed: bool, flux: float | None):
    status = "PASS" if passed else "FAIL"
    flux_str = f"flux={flux:.4e}" if flux is not None else "flux=N/A"
    print(f"\n{'='*60}")
    print(f"  {status}  {label}  |  {flux_str}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def strategy_A():
    """Neutral O2+H2O2, η = 0 (equilibrium). Should always converge."""
    print("\n--- Strategy A: Neutral species, η = 0 ---")
    sp = neutral_species_params(phi_applied_V=0.0, t_end=2.0, dt=0.1)
    ok, flux = run_single_solve(sp, "A")
    print_result("A: Neutral + eta=0", ok, flux)
    return ok


def strategy_B():
    """Neutral O2+H2O2, small cathodic overpotential η = −1 V_T = −0.0257 V."""
    print("\n--- Strategy B: Neutral species, η = −1 V_T ---")
    sp = neutral_species_params(phi_applied_V=-1.0, t_end=5.0, dt=0.1)
    ok, flux = run_single_solve(sp, "B")
    print_result("B: Neutral + eta=-1 V_T", ok, flux)
    return ok


def strategy_C():
    """Neutral O2+H2O2, moderate overpotential η = −5 V_T, direct solve."""
    print("\n--- Strategy C: Neutral species, η = −5 V_T (direct) ---")
    sp = neutral_species_params(phi_applied_V=-5.0, t_end=10.0, dt=0.1)
    ok, flux = run_single_solve(sp, "C")
    print_result("C: Neutral + eta=-5 V_T direct", ok, flux)

    if not ok:
        print("  → direct failed; trying conservative SNES options...")
        sp2 = neutral_species_params(
            phi_applied_V=-5.0, t_end=10.0, dt=0.1,
            snes_opts=SNES_OPTS_CONSERVATIVE,
        )
        ok, flux = run_single_solve(sp2, "C-conservative")
        print_result("C: Neutral + eta=-5 V_T conservative", ok, flux)

    if not ok:
        print("  → conservative failed; trying Picard iteration...")
        sp3 = neutral_species_params(
            phi_applied_V=-5.0, t_end=10.0, dt=0.01,
            snes_opts=SNES_OPTS_PICARD,
        )
        ok, flux = run_single_solve(sp3, "C-picard")
        print_result("C: Neutral + eta=-5 V_T Picard", ok, flux)

    return ok


def strategy_D(eta_target: float = -10.0):
    """Neutral O2+H2O2, potential continuation from η=0 to η=eta_target (nondim)."""
    print(f"\n--- Strategy D: Neutral species, continuation 0 → {eta_target:.1f} V_T ---")
    sp_base = neutral_species_params(
        phi_applied_V=eta_target, t_end=3.0, dt=0.1,
        snes_opts=SNES_OPTS_CONSERVATIVE,
    )
    try:
        U = solve_bv_with_continuation(
            sp_base,
            eta_target=eta_target,
            eta_steps=20,
            print_interval=999,
        )

        # U lives on the internal mesh; extract it from U's function space.
        mesh = U.function_space().mesh()
        ds = fd.Measure("ds", domain=mesh)
        c_O2 = fd.split(U)[0]
        proxy = float(fd.assemble(c_O2 * ds(1)))
        print(f"  → continuation reached eta={eta_target:.1f}, c_O2@electrode proxy={proxy:.4e}")
        print_result(f"D: continuation to eta={eta_target}", True, proxy)
        return True
    except Exception as e:
        print(f"  → continuation FAILED: {e}")
        traceback.print_exc()
        print_result(f"D: continuation to eta={eta_target}", False, None)
        return False


def strategy_E():
    """Charged 2-species (H+/Cl-), η = 0 — tests Poisson coupling."""
    print("\n--- Strategy E: Charged H+/Cl-, η = 0 ---")
    sp = charged_species_params(phi_applied_V=0.0, t_end=3.0, dt=0.1)
    ok, flux = run_single_solve(sp, "E")
    print_result("E: Charged + eta=0", ok, flux)
    return ok


def strategy_F(eta_target: float = -1.2):
    """Charged 2-species, potential continuation to η = −1.2 V_T.

    Tests that BV + Poisson coupling works for charged species up to moderate
    overpotential.  Large overpotentials (η < -1.37 V_T) fail for 0.1 M HCl
    because the Debye length (~1 nm) is far below the mesh resolution (~31 µm),
    making (λ_D/L)² ≈ 10⁻¹⁰ — the Poisson block is near-singular.  The
    electroneutral PNP limit would be needed for concentrated electrolyte +
    large overpotential.  For the O₂/H₂O₂ neutral-species use-case, strategy D
    covers the full overpotential range without this restriction.
    """
    print(f"\n--- Strategy F: Charged H+/Cl-, continuation 0 → {eta_target:.1f} V_T ---")
    sp_base = charged_species_params(
        phi_applied_V=eta_target, t_end=3.0, dt=0.1,
        snes_opts=SNES_OPTS_CONSERVATIVE,
    )
    try:
        U = solve_bv_with_continuation(
            sp_base,
            eta_target=eta_target,
            eta_steps=12,            # Δη ≈ −0.1 V_T each
            print_interval=999,
        )
        proxy = float(fd.assemble(fd.split(U)[0] * fd.Measure("ds", domain=U.function_space().mesh())(1)))
        print_result(f"F: charged continuation to eta={eta_target}", True, proxy)
        return True
    except Exception as e:
        print(f"  → FAILED: {e}")
        traceback.print_exc()
        print_result(f"F: charged continuation to eta={eta_target}", False, None)
        return False


def strategy_G_small_k0():
    """Try with k0 reduced by 100× to make the problem less stiff at large η."""
    print("\n--- Strategy G: Neutral species, k0/100, η = −10 V_T (less stiff) ---")
    sp = neutral_species_params(
        phi_applied_V=-10.0, t_end=5.0, dt=0.05,
        snes_opts=SNES_OPTS_CONSERVATIVE,
        k0_override=k0_hat * 0.01,   # 100x smaller k0
    )
    ok, flux = run_single_solve(sp, "G-small-k0")
    print_result("G: Neutral + eta=-10, small k0", ok, flux)
    return ok


# ---------------------------------------------------------------------------
# Sweep: generate I-V curve points
# ---------------------------------------------------------------------------

def strategy_sweep(eta_values=None):
    """Run a full I-V sweep using continuation, printing flux at each η."""
    if eta_values is None:
        eta_values = np.linspace(0, -20, 21)

    print(f"\n--- Sweep: {len(eta_values)} eta values from {eta_values[0]:.1f} to {eta_values[-1]:.1f} V_T ---")

    results = []
    sp_base = neutral_species_params(
        phi_applied_V=eta_values[-1], t_end=3.0, dt=0.1,
        snes_opts=SNES_OPTS_CONSERVATIVE,
    )

    # Rebuild context once and reuse
    ctx = build_context(sp_base)
    sp0 = neutral_species_params(phi_applied_V=0.0, t_end=3.0, dt=0.1)
    ctx = build_forms(ctx, sp0)
    set_initial_conditions(ctx, sp0, blob=False)

    mesh = ctx["mesh"]
    ds = fd.Measure("ds", domain=mesh)
    scaling = ctx["nondim"]
    J_scale = float(scaling.get("flux_scale_mol_m2_s", 1.0))
    I_scale = float(scaling.get("current_density_scale_a_m2", 1.0))

    dt_model = float(scaling["dt_model"])
    t_end_model = float(scaling["t_end_model"])
    num_steps = max(1, int(round(t_end_model / dt_model)))

    J_form = ctx["J_form"]
    problem = fd.NonlinearVariationalProblem(ctx["F_res"], ctx["U"], bcs=ctx["bcs"], J=J_form)
    solver = fd.NonlinearVariationalSolver(problem, solver_parameters=dict(sp_base[10]))

    for eta in eta_values:
        ctx["phi_applied_func"].assign(float(eta))
        success = False
        try:
            for _ in range(num_steps):
                solver.solve()
                ctx["U_prev"].assign(ctx["U"])
            success = True
        except Exception as e:
            print(f"  eta={eta:+.2f} FAILED: {e}")
            ctx["U"].assign(ctx["U_prev"])  # roll back

        if success:
            c_O2 = fd.split(ctx["U"])[0]
            phi_field = fd.split(ctx["U"])[-1]
            n_vec = fd.FacetNormal(mesh)
            em = float(scaling["electromigration_prefactor"])
            D0 = float(scaling["D_model_vals"][0])
            Jflux = D0 * (fd.grad(c_O2) + fd.Constant(0.0) * fd.grad(phi_field))
            flux_nondim = float(fd.assemble(fd.dot(Jflux, n_vec) * ds(1)))
            # Convert to physical units: flux × J_scale → mol/(m²·s); × n_e × F → A/m²
            n_electrons = 2
            current_A_m2 = flux_nondim * J_scale * n_electrons * F
            current_mA_cm2 = current_A_m2 * 0.1
            results.append((eta * V_T, flux_nondim, current_mA_cm2))
            print(f"  eta={eta:+6.2f} V_T  phi={eta*V_T*1000:+7.1f} mV  "
                  f"flux={flux_nondim:.4e}  I={current_mA_cm2:.4e} mA/cm²")
        else:
            results.append((eta * V_T, float("nan"), float("nan")))

    print("\nSweep summary (phi_V, flux_nondim, I_mA_cm2):")
    for row in results:
        print(f"  {row[0]:+8.4f} V   {row[1]:10.4e}   {row[2]:10.4e} mA/cm²")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

STRATEGIES = {
    "A": strategy_A,
    "B": strategy_B,
    "C": strategy_C,
    "D": strategy_D,
    "E": strategy_E,
    "F": strategy_F,
    "G": strategy_G_small_k0,
    "sweep": strategy_sweep,
}


def main():
    parser = argparse.ArgumentParser(description="BV forward solver convergence tests")
    parser.add_argument(
        "--strategy", default="A",
        choices=list(STRATEGIES.keys()) + ["all"],
        help="Which strategy to run (default: A)",
    )
    parser.add_argument(
        "--eta", type=float, default=None,
        help="Override target overpotential (in V_T units) for strategies D/F",
    )
    args = parser.parse_args()

    if args.strategy == "all":
        order = ["A", "B", "C", "D", "E", "F", "G"]
        for s in order:
            fn = STRATEGIES[s]
            kwargs = {}
            if s in ("D", "F") and args.eta is not None:
                kwargs["eta_target"] = args.eta
            ok = fn(**kwargs)
            if not ok:
                print(f"\n[all] Stopped at strategy {s} — convergence failed.")
                break
        return

    fn = STRATEGIES[args.strategy]
    kwargs = {}
    if args.strategy in ("D", "F") and args.eta is not None:
        kwargs["eta_target"] = args.eta
    fn(**kwargs)


if __name__ == "__main__":
    main()
