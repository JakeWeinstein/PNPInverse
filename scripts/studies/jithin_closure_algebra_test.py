"""Test whether Jithin's closure-form Bikerman algebra produces the cliff.

Pure-algebra diagnostic (no spatial PDE).  At each prescribed Ψ_OHP:

1. Compute would-be ideal Boltzmann pile-up at OHP for each species.
2. Apply Jithin's closure form:  c_k = (c_k_bulk/θ_b) · exp(−z_k βΨ) · θ_OHP
   where θ_OHP = 1 / (1 + Σ a_k · (c_k_bulk/θ_b) · exp(−z_k βΨ))
3. Plug c_O2(OHP) into a Tafel BV: j = -j₀ · (c_O2*/c_O2^b) · 10^(|η|/A_T)

This is a degenerate test that ignores spatial transport (no Levich limit,
no g_k flux supply).  But it directly probes the algebraic structure of
his closure: if c_O2(OHP) drops sharply when cations saturate the OHP via
the shared θ factor, BV current drops → cliff.

Outputs ``StudyResults/jithin_closure_algebra_test/iv_curve.json`` and
a plot ``iv_curve.png``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Constants
F = 96485.33
R_GAS = 8.3145
T_K = 298.15
N_A = 6.02214076e23
V_T = R_GAS * T_K / F
BETA = 1.0 / V_T

# Jithin Fig 4.36 parameters
E0 = 0.695
A_TAFEL = 0.0262
J0 = 1e-15

C_O2_B = 0.25
C_H2O2_B = 1e-6
C_HP_B = 10.0
C_CS_B = 190.0
C_SO4_B = 100.0

V_O2 = 0.064e-27
V_H2O2 = 0.16638e-27
V_HP = 0.175616e-27
V_CS = 0.28489e-27
V_SO4 = 0.43552e-27

A_O2 = V_O2 * N_A
A_H2O2 = V_H2O2 * N_A
A_HP = V_HP * N_A
A_CS = V_CS * N_A
A_SO4 = V_SO4 * N_A

Z_O2 = 0
Z_H2O2 = 0
Z_HP = 1
Z_CS = 1
Z_SO4 = -2

THETA_B = 1.0 - (
    A_O2 * C_O2_B + A_H2O2 * C_H2O2_B + A_HP * C_HP_B
    + A_CS * C_CS_B + A_SO4 * C_SO4_B
)
print(f"θ_b = {THETA_B:.4f}")
assert THETA_B > 0

A_K_O2 = C_O2_B / THETA_B
A_K_H2O2 = C_H2O2_B / THETA_B
A_K_HP = C_HP_B / THETA_B
A_K_CS = C_CS_B / THETA_B
A_K_SO4 = C_SO4_B / THETA_B


def closure_at_ohp(psi_ohp_v: float):
    """Returns (theta_OHP, c_O2_OHP, c_H_OHP, c_Cs_OHP)."""
    psi_beta = psi_ohp_v / V_T
    # phi_clamp on the Boltzmann factors so we don't overflow
    # Using a soft clamp at ±50 (eta of ±1.3 V)
    psi_clamped = np.clip(psi_beta, -50.0, 50.0)

    exp_h = np.exp(-Z_HP * psi_clamped)
    exp_cs = np.exp(-Z_CS * psi_clamped)
    exp_sm2 = np.exp(-Z_SO4 * psi_clamped)

    # Sum a · A_k · exp(-z·Ψ)
    sum_a_rhs = (
        A_O2 * A_K_O2 + A_H2O2 * A_K_H2O2
        + A_HP * A_K_HP * exp_h
        + A_CS * A_K_CS * exp_cs
        + A_SO4 * A_K_SO4 * exp_sm2
    )
    theta = 1.0 / (1.0 + sum_a_rhs)

    c_o2 = A_K_O2 * theta
    c_h = A_K_HP * exp_h * theta
    c_cs = A_K_CS * exp_cs * theta
    return theta, c_o2, c_h, c_cs


def bv_current(c_o2_ohp: float, v_electrode_v: float) -> float:
    eta = v_electrode_v - E0
    if eta >= 0:
        return 0.0
    return -J0 * (c_o2_ohp / C_O2_B) * 10.0 ** (abs(eta) / A_TAFEL)


def main() -> int:
    out_dir = Path(__file__).resolve().parents[2] / "StudyResults" / "jithin_closure_algebra_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sweep V_RHE (we ignore Stern for this algebra test: V_electrode = Ψ_OHP)
    v_rhe = np.linspace(-0.40, +0.55, 96)
    records = []
    print(f"{'V_RHE':>8}  {'θ_OHP':>10}  {'c_O₂(OHP)':>14}  "
          f"{'c_H⁺(OHP)':>14}  {'c_Cs⁺(OHP)':>14}  {'j_mA/cm²':>11}")
    for v in v_rhe:
        theta, c_o2, c_h, c_cs = closure_at_ohp(v)
        j = bv_current(c_o2, v)
        j_mA_cm2 = j * 0.1
        if abs(v - round(v, 2)) < 0.005 or v in (v_rhe[0], v_rhe[-1]):
            print(
                f"{v:+8.3f}  {theta:10.4e}  {c_o2:14.4e}  "
                f"{c_h:14.4e}  {c_cs:14.4e}  {j_mA_cm2:+11.4e}"
            )
        records.append({
            "v_rhe": float(v),
            "theta_OHP": float(theta),
            "c_O2_OHP_mol_m3": float(c_o2),
            "c_H_OHP_mol_m3": float(c_h),
            "c_Cs_OHP_mol_m3": float(c_cs),
            "j_A_m2": float(j),
            "cd_mA_cm2": float(j_mA_cm2),
        })

    report = {
        "label": "jithin_closure_algebra_only_test",
        "config": {
            "theta_b": float(THETA_B),
            "j0_A_m2": float(J0),
            "A_tafel_V_dec": float(A_TAFEL),
            "E0_V": float(E0),
            "ignored": [
                "spatial transport / Levich",
                "Stern voltage drop (Ψ_OHP = V_electrode here)",
                "g_k flux-supply integrals",
            ],
        },
        "records": records,
    }
    (out_dir / "iv_curve.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out_dir / 'iv_curve.json'}")

    v_arr = np.array([r["v_rhe"] for r in records])
    j_arr = np.array([r["cd_mA_cm2"] for r in records])
    theta_arr = np.array([r["theta_OHP"] for r in records])
    c_o2_arr = np.array([r["c_O2_OHP_mol_m3"] for r in records])

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 9.5), sharex=True)

    axes[0].plot(v_arr, j_arr, "o-", color="C0", markersize=3)
    axes[0].axhline(0.0, color="k", linewidth=0.5)
    axes[0].set_ylabel("j (mA/cm²)")
    axes[0].set_title(
        "Closure-only algebra test — does Jithin's Bikerman closure at "
        "the OHP\nproduce a cliff?  (no spatial PDE, no Levich, no "
        "Stern)"
    )
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(v_arr, c_o2_arr, "o-", color="C2", markersize=3)
    axes[1].axhline(C_O2_B, color="C2", linestyle=":",
                    label=f"bulk c_O₂ = {C_O2_B} mol/m³")
    axes[1].set_ylabel("c_O₂(OHP) (mol/m³)")
    axes[1].set_yscale("log")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(v_arr, theta_arr, "o-", color="C3", markersize=3)
    axes[2].set_ylabel("θ(OHP) = 1 − Σa·c")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Ψ_OHP = V_electrode (V vs RHE)")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "iv_curve.png", dpi=150)
    print(f"wrote {out_dir / 'iv_curve.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
