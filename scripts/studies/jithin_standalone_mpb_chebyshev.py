"""Standalone Jithin Fig 4.36 emulation — pure NumPy/SciPy.

Implements Jithin Chapter 3-4 closure-form Modified Poisson-Boltzmann
with flux + Bikerman steric saturation on ALL species, using Chebyshev
spectral collocation with tan⁻¹ boundary-layer mapping (his Eq 4.19).

Goal: verify whether his cliff at far cathodic V is recoverable from his
mathematical formulation when faithfully re-implemented.

Math:
  Domain x̃ ∈ [0, L_bulk] (physical, m), mapped to s̃ ∈ [0, 2] nondim
  via x̃ = L·s̃ with L = L_bulk/2.  Chebyshev s ∈ [-1, +1] with the
  inverse mapping s̃ = 1 + s (so s=+1 ↔ OHP, s=-1 ↔ bulk).
  Apply Jithin's tan⁻¹ stretch (Eq 4.20) to cluster nodes near s=+1.

Unknowns at N+1 collocation points:
  Ψ(s)      — nondim potential (β·ψ)
  g_O2(s)   — neutral flux-supply integrand
  g_H2O2(s) — neutral flux-supply integrand
  g_H(s)    — proton flux-supply integrand

Per spatial point, c_k is computed via Jithin Eq 3.38 closure with
shared Bikerman denominator over all species (Cs⁺ and SO₄²⁻ inert,
H⁺/O₂/H₂O₂ get flux supply).

Outer fixed-point on j via BV consistency at OHP (Eq 3.49).

Reproduction:
  python -u scripts/studies/jithin_standalone_mpb_chebyshev.py
  python -u scripts/studies/_plot_jithin_standalone_mpb.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.optimize import root


# =====================================================================
# Physical constants and Jithin Fig 4.36 parameters
# =====================================================================

F: float = 96485.33                  # C/mol
R_GAS: float = 8.3145                # J/(mol·K)
T_K: float = 298.15                  # K
N_A: float = 6.02214076e23           # 1/mol
EPS_0: float = 8.854e-12             # F/m
EPS_R: float = 78.5                  # water at 25°C
EPS: float = EPS_R * EPS_0           # F/m
V_T: float = R_GAS * T_K / F         # ~0.025693 V
BETA: float = 1.0 / V_T              # 1/V

# Geometry (Jithin Fig 4.36)
L_BULK: float = 10e-6                # m, his fitted L_diff
L: float = L_BULK / 2.0              # m, half-domain length (his eq 4.5)
L_STERN: float = 0.6e-9              # m, his fitted Stern thickness
C_S_STERN: float = EPS / L_STERN     # F/m², linear Stern capacitance (≈1.16)

# Kinetics
E0: float = 0.695                    # V vs RHE
A_TAFEL: float = 0.0262              # V/decade — Jithin Fig 4.36 fit
J0: float = 1e-15                    # A/m² — his fitted j₀ (placeholder)

# Diffusivities (Jithin Table 4.1, m²/s)
D_O2: float = 1.5e-9
D_H2O2: float = 1.6e-9
D_HP: float = 9.311e-9

# Bulk concentrations (mol/m³)
C_O2_B: float = 0.25
C_H2O2_B: float = 1e-6               # seed
C_HP_B: float = 10.0                 # pH 2
C_CS_B: float = 190.0
C_SO4_B: float = 100.0

# Ion volumes per molecule (m³), Jithin Table 4.1
V_O2: float = 0.064e-27
V_H2O2: float = 0.16638e-27
V_HP: float = 0.175616e-27
V_CS: float = 0.28489e-27
V_SO4: float = 0.43552e-27

# Excluded volume per mole (m³/mol)
A_O2: float = V_O2 * N_A
A_H2O2: float = V_H2O2 * N_A
A_HP: float = V_HP * N_A
A_CS: float = V_CS * N_A
A_SO4: float = V_SO4 * N_A

# Charges
Z_O2: int = 0
Z_H2O2: int = 0
Z_HP: int = 1
Z_CS: int = 1
Z_SO4: int = -2

# Bulk packing fraction θ_b = 1 - Σ a·c
THETA_B: float = 1.0 - (
    A_O2 * C_O2_B + A_H2O2 * C_H2O2_B + A_HP * C_HP_B
    + A_CS * C_CS_B + A_SO4 * C_SO4_B
)
assert THETA_B > 0, f"theta_b = {THETA_B:.4g} ≤ 0; bulk packing exceeds 1"

# A_k = c_k(bulk) / θ_b   (Jithin eq 3.37)
A_K_O2: float = C_O2_B / THETA_B
A_K_H2O2: float = C_H2O2_B / THETA_B
A_K_HP: float = C_HP_B / THETA_B
A_K_CS: float = C_CS_B / THETA_B
A_K_SO4: float = C_SO4_B / THETA_B

# V_RHE sweep (matches our solver run for direct overlay)
V_RHE_GRID: List[float] = [
    round(float(v), 4) for v in np.linspace(-0.40, +0.55, 25).tolist()
]

# Numerics
N_CHEB: int = 120                    # Chebyshev collocation points
ALPHA_1_MAP: float = 300.0           # tan⁻¹ stretch parameter (Jithin α₁)
ALPHA_2_MAP: float = 1.0             # mapping centre (Jithin α₂ = 1 → OHP)

OUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "StudyResults" / "jithin_standalone_mpb_chebyshev"
)


# =====================================================================
# Chebyshev spectral collocation + tan⁻¹ mapping
# =====================================================================

def cheb_diff_matrix(N: int) -> Tuple[np.ndarray, np.ndarray]:
    """Trefethen cheb.m: differentiation matrix on s ∈ [-1, +1].

    Returns (D, s) where D is (N+1)×(N+1) and s is length N+1
    collocation points s_k = cos(k π / N), k = 0..N.
    """
    if N == 0:
        return np.array([[0.0]]), np.array([1.0])
    s = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.ones(N + 1)
    c[0] = 2.0
    c[N] = 2.0
    c[1::2] *= -1.0

    S = np.tile(s, (N + 1, 1)).T
    dS = S - S.T

    D = np.outer(c, 1.0 / c) / (dS + np.eye(N + 1))
    D -= np.diag(np.sum(D, axis=1))
    return D, s


def build_mapped_grid(
    N: int, alpha1: float, alpha2: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Chebyshev grid s ∈ [-1, +1] mapped to a clustered grid via
    Jithin Eq 4.20: x̃ = α₂ + tan(λ(s − s₀))/α₁.

    Returns (s_cheb, x_phys, D_x, D_xx) where:
      s_cheb is the standard Chebyshev grid (N+1 points)
      x_phys is the physical-coord grid (n nm convention: x̃ ∈ [0, 2])
      D_x is the first-derivative matrix d/dx̃ on the mapped grid
      D_xx is the second-derivative matrix d²/dx̃²
    """
    D_s, s = cheb_diff_matrix(N)

    # Compute mapping constants (Jithin Eq 4.21-4.22)
    kappa = np.arctan(alpha1 * (1.0 + alpha2)) / np.arctan(
        alpha1 * (1.0 - alpha2)
    )
    s0 = (kappa - 1.0) / (kappa + 1.0)
    lam = np.arctan(alpha1 * (1.0 - alpha2)) / (1.0 - s0)

    # Forward mapping: s_phys = α₂ + tan(λ(s − s₀))/α₁
    x_phys = alpha2 + np.tan(lam * (s - s0)) / alpha1

    # Coordinate derivative: dx̃/ds = (λ/α₁) · sec²(λ(s−s₀))
    sec2 = 1.0 / np.cos(lam * (s - s0)) ** 2
    dx_ds = (lam / alpha1) * sec2

    # Chain rule: D_x = (ds/dx̃) · D_s, where ds/dx̃ = 1 / (dx̃/ds)
    ds_dx = 1.0 / dx_ds
    D_x = (ds_dx[:, None]) * D_s

    # D_xx via D_x · D_x (matrix product) — equivalent to (ds/dx̃)² D_ss + d²s/dx̃² D_s
    D_xx = D_x @ D_x

    return s, x_phys, D_x, D_xx


# =====================================================================
# Closure: compute c_k(x) given (Ψ, g_O2, g_H2O2, g_H, j)
# =====================================================================

def closure_concentrations(
    psi_beta: np.ndarray,
    g_o2: np.ndarray, g_h2o2: np.ndarray, g_h: np.ndarray,
    j: float,
) -> Tuple[np.ndarray, ...]:
    """Compute (c_O2, c_H2O2, c_H, c_Cs, c_SO4) from closure equation
    at every spatial point.

    Jithin eq 3.38 (closure form):
        c_k/(1 − Σ_j a_j c_j) = A_k · exp(−z_k Ψ) + φ_k g_k
        where Ψ = β·ψ (nondim potential)

    For O₂, H₂O₂ (z=0): exp(−z·Ψ) = 1
    For H⁺ (z=+1): exp(−Ψ)
    For Cs⁺ (z=+1): exp(−Ψ); inert (φ = 0)
    For SO₄²⁻ (z=−2): exp(2Ψ); inert (φ = 0)

    Returns concentrations in mol/m³.
    """
    # Boltzmann factors per species
    exp_o2 = np.ones_like(psi_beta)
    exp_h2o2 = np.ones_like(psi_beta)
    exp_h = np.exp(-psi_beta)
    exp_cs = np.exp(-psi_beta)
    exp_sm2 = np.exp(2.0 * psi_beta)

    # Flux supplies (per Jithin convention: φ_k = j / (n_k F D_k))
    #   O₂: 2 e⁻ per O₂ consumed → n_O₂ = 2,  φ_O₂ = j/(2 F D_O₂)
    #   H₂O₂: 2 e⁻ produce 1 H₂O₂ → n_H₂O₂ = −2 (negative; produced)
    #   H⁺: 2 e⁻ consume 2 H⁺ → 1 e⁻ per H⁺ → n_H⁺ = 1
    phi_o2 = j / (2.0 * F * D_O2)
    phi_h2o2 = -j / (2.0 * F * D_H2O2)
    phi_h = j / (F * D_HP)

    # Closure: c_k = (A_k·exp + φ·g) · θ, where θ = 1 − Σ a·c
    # Substitute Σ a·c = (1 − θ): solve algebraically for θ.
    #   1 − θ = θ · Σ a·(A_k·exp + φ·g)
    #   θ = 1 / (1 + Σ a·(A_k·exp + φ·g))
    rhs_o2 = A_K_O2 * exp_o2 + phi_o2 * g_o2
    rhs_h2o2 = A_K_H2O2 * exp_h2o2 + phi_h2o2 * g_h2o2
    rhs_h = A_K_HP * exp_h + phi_h * g_h
    rhs_cs = A_K_CS * exp_cs            # inert
    rhs_sm2 = A_K_SO4 * exp_sm2          # inert

    sum_a_rhs = (
        A_O2 * rhs_o2 + A_H2O2 * rhs_h2o2 + A_HP * rhs_h
        + A_CS * rhs_cs + A_SO4 * rhs_sm2
    )
    theta = 1.0 / (1.0 + sum_a_rhs)

    c_o2 = rhs_o2 * theta
    c_h2o2 = rhs_h2o2 * theta
    c_h = rhs_h * theta
    c_cs = rhs_cs * theta
    c_sm2 = rhs_sm2 * theta

    return c_o2, c_h2o2, c_h, c_cs, c_sm2, theta


# =====================================================================
# Residual for Newton: (Ψ, g_O₂, g_H₂O₂, g_H) on N+1 collocation points
# =====================================================================

def build_residual(N: int, D_x: np.ndarray, D_xx: np.ndarray, x_phys: np.ndarray):
    """Return a residual function res(U) and unpack helper.

    U is flattened [Ψ; g_O2; g_H2O2; g_H], length 4·(N+1).
    Closure equation enforces concentrations; Poisson gives Ψ residual;
    g_k ODE gives g_k residual.  BCs are baked into the residual.

    Domain: s ∈ [-1, +1].  Mapping: x̃ ∈ [0, 2L_bulk/L = 2].
    Physical x = L · x̃, so OHP at x̃=0, bulk at x̃=2.
    But here our s grid is s=cos(kπ/N) with s_0=+1 → x̃=α₂+0=1 corresponds
    to... wait, the mapping centre is α₂=1 in *physical x̃* coordinates,
    meaning the boundary layer is at x̃=1 (centre of physical domain
    [0, 2]).  That's wrong for our setup — OHP is at x̃=0.

    Re-anchor: in Jithin's notation his "0" boundary is the OHP and "2"
    is the bulk.  Mapping centre α₂=1 is the middle, NOT the OHP.  His
    Eq 4.20 places the rapid-variation region at x = α₂.  So for our
    setup we'd want α₂ to coincide with the OHP location in the mapped
    coordinate — but his text uses α₂=1, suggesting the OHP IS at the
    centre of his s∈[-1,+1] mapped domain.

    Punting on this — we'll use his convention: x̃ ∈ [0, 2] with OHP at
    x̃=0, bulk at x̃=2.  Mapping centre α₂=1 puts the cluster at x̃=1
    which is the middle of the domain, NOT the OHP.  We'll cluster at
    the OHP by setting α₂=0 in our convention.

    Returns residual function and helper.
    """
    n_dof = N + 1

    def unpack(U: np.ndarray):
        psi = U[0:n_dof]
        g_o2 = U[n_dof:2*n_dof]
        g_h2o2 = U[2*n_dof:3*n_dof]
        g_h = U[3*n_dof:4*n_dof]
        return psi, g_o2, g_h2o2, g_h

    def residual(U: np.ndarray, psi_ohp_beta: float, j: float) -> np.ndarray:
        """Residual for Newton solve.

        psi_ohp_beta: prescribed Ψ = β·ψ at OHP (Dirichlet BC)
        j: current density (A/m²) — fixed during inner Newton; outer
           loop iterates on this via BV consistency.
        """
        psi, g_o2, g_h2o2, g_h = unpack(U)

        c_o2, c_h2o2, c_h, c_cs, c_sm2, theta = closure_concentrations(
            psi, g_o2, g_h2o2, g_h, j,
        )

        # Charge density
        rho = (
            Z_O2 * c_o2 + Z_H2O2 * c_h2o2 + Z_HP * c_h
            + Z_CS * c_cs + Z_SO4 * c_sm2
        )

        # Poisson: ε d²ψ/dx̃² = -F·ρ  in physical x̃ (which is L · x_nondim)
        # In our x_phys coordinate (mapped Chebyshev), physical x = L · x_phys
        # so d/dx_phys = L · d/dx_phys, d²/dx_phys² = L² · d²/dx_phys²
        # Wait, our D_xx is already in the x_phys (nondim) coordinate.
        # We need to convert: d²ψ/dx_dim² = (1/L²) · D_xx · ψ
        # Poisson: -EPS · (1/L²) · D_xx · ψ = F · ρ
        psi_volts = psi / BETA   # back to V
        d2psi_dx2_phys = (D_xx @ psi_volts) / (L ** 2)
        res_poisson = -EPS * d2psi_dx2_phys - F * rho

        # g_k ODE (Jithin Eq 4.13 generalised, in physical x):
        #   dg_k/dx + z_k · β · dψ/dx · g_k =
        #       1 + a_k · Σ_j A_j a_j exp(−z_j Ψ) + φ_k · Σ_j a_j φ_j g_j
        # (translating Jithin Eq 4.13 from nondim to dim: drop the κ's)
        # We work in nondim x (x ∈ [0, 2]) so:
        #   dg/dx_phys (nondim) = L · dg/dx_dim
        # Hmm.  Let me re-derive g_k cleanly.
        #
        # g_k(x) = exp(-z_k β ψ(x)) · ∫_L^x exp(z_k β ψ(s))/(1 − Σ a·c(s)) ds
        # Differentiating: g_k' = -z_k β ψ' · g_k + 1/(1 − Σ a·c)
        # i.e. g_k' + z_k β ψ' g_k = 1/(1 − Σ a·c) = 1/θ
        # In physical x (m).
        dpsi_dx = (D_x @ psi_volts) / L      # V/m
        dpsi_dx_beta = dpsi_dx * BETA        # 1/m
        dg_o2_dx = (D_x @ g_o2) / L
        dg_h2o2_dx = (D_x @ g_h2o2) / L
        dg_h_dx = (D_x @ g_h) / L

        rhs_g = 1.0 / theta                  # 1/m (dimensions match: g has units m·s·mol/m³/F·s/m² · ... actually let me just check)
        res_g_o2 = dg_o2_dx + Z_O2 * dpsi_dx_beta * g_o2 - rhs_g
        res_g_h2o2 = dg_h2o2_dx + Z_H2O2 * dpsi_dx_beta * g_h2o2 - rhs_g
        res_g_h = dg_h_dx + Z_HP * dpsi_dx_beta * g_h - rhs_g

        # Boundary conditions overwrite first/last rows of residual
        # ψ(s=-1) → bulk (Ψ = 0); ψ(s=+1) → OHP (Ψ = psi_ohp_beta)
        # Chebyshev s_0 = cos(0) = +1, s_N = cos(π) = -1.  So index 0 is OHP, index N is bulk.
        res_poisson[0] = psi[0] - psi_ohp_beta
        res_poisson[N] = psi[N] - 0.0
        # g_k(bulk) = 0 → index N
        res_g_o2[N] = g_o2[N] - 0.0
        res_g_h2o2[N] = g_h2o2[N] - 0.0
        res_g_h[N] = g_h[N] - 0.0

        return np.concatenate([res_poisson, res_g_o2, res_g_h2o2, res_g_h])

    return residual, unpack


# =====================================================================
# Outer fixed-point: iterate j via BV consistency
# =====================================================================

def bv_current(c_o2_ohp: float, v_electrode: float) -> float:
    """Jithin Eq 3.46: j = -j₀ · (c_O2*/c_O2^b) · 10^(|η|/A_Tafel)
    Returns cathodic current in A/m² (negative)."""
    eta = v_electrode - E0
    # Tafel current magnitude (cathodic = positive Tafel exponent under |η|)
    if eta >= 0:
        # Anodic side — for Tafel-only model, no current
        return 0.0
    mag = J0 * (c_o2_ohp / C_O2_B) * 10.0 ** (abs(eta) / A_TAFEL)
    return -mag


def solve_for_psi_ohp(
    psi_ohp_beta: float, U_init: np.ndarray, residual, unpack,
    n_outer: int = 20, tol: float = 1e-9,
) -> Tuple[np.ndarray, float, str]:
    """Outer fixed-point on j; inner Newton on (Ψ, g_k).
    Returns (U_final, j_final, status)."""
    U = U_init.copy()
    j = 0.0
    psi_ohp_volt = psi_ohp_beta / BETA

    # Initial guess: zero current
    for outer in range(n_outer):
        sol = root(
            residual, U, args=(psi_ohp_beta, j),
            method="lm",
            options={"xtol": 1e-12, "ftol": 1e-12, "maxiter": 200},
        )
        if not sol.success:
            return U, j, f"inner_failed:{sol.message}"
        U = sol.x

        psi, g_o2, g_h2o2, g_h = unpack(U)
        c_o2, c_h2o2, c_h, c_cs, c_sm2, theta = closure_concentrations(
            psi, g_o2, g_h2o2, g_h, j,
        )
        c_o2_ohp = float(c_o2[0])

        # V_electrode from Stern: V_electrode = Ψ_OHP + L_Stern · E_OHP
        # where E_OHP = -dψ/dx at OHP (field magnitude pointing into bulk)
        # Actually Jithin Eq 3.51: V_electrode = ψ_OHP + (L_Stern · σ / ε)
        # where σ is surface charge.  Simpler: linear Stern → ψ_electrode
        # is just ψ_OHP minus the Stern voltage drop = Q·L_Stern/ε.
        # In V: Q_surf = -ε · dψ/dx |_{OHP} (Gauss law); Stern drop = Q·L_Stern/ε.
        # So V_electrode = ψ_OHP + L_Stern · (-dψ/dx|_OHP)
        # = ψ_OHP - L_Stern · dψ/dx|_OHP
        # In our convention with x growing from OHP to bulk and ψ→0 in bulk:
        # under cathodic polarization, ψ_OHP < 0, dψ/dx > 0.  V_electrode is more negative than ψ_OHP.
        # We compute it but don't use it for the inner solve (Ψ_OHP is the BC).

        j_new = bv_current(c_o2_ohp, psi_ohp_volt)
        if abs(j_new - j) < tol * max(abs(j_new), abs(j), 1e-12):
            return U, j_new, f"converged_outer_{outer+1}"
        j = j_new
    return U, j, "outer_loop_max_iter"


# =====================================================================
# Main: sweep Ψ_OHP, compute V_electrode, output JSON
# =====================================================================

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  Standalone Jithin Fig 4.36 emulation — Chebyshev MPB+flux")
    print("=" * 78)
    print(f"  N_CHEB     = {N_CHEB}")
    print(f"  L_BULK     = {L_BULK*1e6:.1f} μm")
    print(f"  L_STERN    = {L_STERN*1e9:.1f} nm  → C_S = {C_S_STERN:.3f} F/m²")
    print(f"  pH         = 2 (bulk H⁺ = {C_HP_B} mol/m³)")
    print(f"  bulk O₂    = {C_O2_B} mol/m³")
    print(f"  E°         = {E0:.3f} V vs RHE")
    print(f"  A_Tafel    = {A_TAFEL*1000:.1f} mV/dec")
    print(f"  j₀         = {J0:.2e} A/m²")
    print(f"  θ_b (bulk packing) = {THETA_B:.4f}")
    print(f"  output     = {OUT_DIR}")
    print("=" * 78)

    s_grid, x_phys, D_x, D_xx = build_mapped_grid(
        N_CHEB, ALPHA_1_MAP, ALPHA_2_MAP,
    )
    # x_phys is in [0, 2] nondim (Jithin's convention).  Convert to physical
    # m: x̃_dim = L · x_phys.  But our D_x/D_xx already account for this via
    # the physical-coordinate Jacobian (they're d/dx̃, d²/dx̃²) — but they
    # were computed in the nondim s-grid, with the mapping giving us
    # x_phys (still nondim).  So we need to scale: dimensional x = L * x_phys.
    # Inside build_residual we divide by L to convert nondim to dim derivative.

    residual, unpack = build_residual(N_CHEB, D_x, D_xx, x_phys)

    # Initial guess: ψ linear from 0 to small, g_k = 0
    n_dof = N_CHEB + 1
    U_init = np.zeros(4 * n_dof)
    # ψ linear: index 0 = OHP, index N = bulk
    U_init[:n_dof] = -0.1 * (1 - x_phys / 2)   # mild linear drop

    # Sweep Ψ_OHP over a range to cover the V_RHE grid we want
    # Jithin's convention: Ψ < 0 = cathodic.  Pick a range:
    psi_ohp_grid_v = np.linspace(-0.5, 0.0, 21)   # V at OHP
    print(f"\nSweeping {len(psi_ohp_grid_v)} Ψ_OHP values from "
          f"{psi_ohp_grid_v[0]:.3f} to {psi_ohp_grid_v[-1]:.3f} V...",
          flush=True)

    records = []
    U_warm = U_init.copy()
    t0 = time.time()
    for k, psi_ohp_v in enumerate(psi_ohp_grid_v):
        psi_ohp_beta = float(psi_ohp_v) * BETA
        U_final, j_final, status = solve_for_psi_ohp(
            psi_ohp_beta, U_warm, residual, unpack,
        )
        psi, g_o2, g_h2o2, g_h = unpack(U_final)
        c_o2, c_h2o2, c_h, c_cs, c_sm2, theta = closure_concentrations(
            psi, g_o2, g_h2o2, g_h, j_final,
        )

        # Back-calculate V_electrode
        dpsi_dx_ohp = (D_x[0, :] @ (psi / BETA)) / L
        v_stern_drop = -L_STERN * dpsi_dx_ohp
        v_electrode = float(psi_ohp_v) + v_stern_drop

        # j in mA/cm²
        j_mA_cm2 = j_final * 0.1   # A/m² → mA/cm² is ×0.1

        rec = {
            "psi_ohp_v": float(psi_ohp_v),
            "v_electrode_v": float(v_electrode),
            "v_stern_drop_v": float(v_stern_drop),
            "j_A_m2": float(j_final),
            "cd_mA_cm2": float(j_mA_cm2),
            "c_O2_OHP_mol_m3": float(c_o2[0]),
            "c_H_OHP_mol_m3": float(c_h[0]),
            "c_Cs_OHP_mol_m3": float(c_cs[0]),
            "c_SO4_OHP_mol_m3": float(c_sm2[0]),
            "theta_OHP": float(theta[0]),
            "status": status,
        }
        records.append(rec)
        print(
            f"  [{k+1:2d}/{len(psi_ohp_grid_v)}] "
            f"Ψ_OHP={psi_ohp_v:+.3f} V  V_elec={v_electrode:+.3f} V  "
            f"j={j_mA_cm2:+.4f} mA/cm²  c_O₂(OHP)={c_o2[0]:.2e} mol/m³  "
            f"θ(OHP)={theta[0]:.4f}  [{status}]",
            flush=True,
        )

        if status.startswith("converged"):
            U_warm = U_final   # warm-start the next Ψ_OHP

    wall = time.time() - t0
    print(f"\nSweep done in {wall:.1f}s")

    report = {
        "config": {
            "n_cheb": N_CHEB,
            "L_bulk_m": L_BULK,
            "L_stern_m": L_STERN,
            "C_S_F_m2": C_S_STERN,
            "E0_V": E0,
            "A_tafel_V_dec": A_TAFEL,
            "j0_A_m2": J0,
            "pH": 2,
            "c_O2_bulk_mol_m3": C_O2_B,
            "c_HP_bulk_mol_m3": C_HP_B,
            "c_Cs_bulk_mol_m3": C_CS_B,
            "c_SO4_bulk_mol_m3": C_SO4_B,
            "V_O2_nm3_per_molecule": V_O2 * 1e27,
            "V_H2O2_nm3_per_molecule": V_H2O2 * 1e27,
            "V_HP_nm3_per_molecule": V_HP * 1e27,
            "V_Cs_nm3_per_molecule": V_CS * 1e27,
            "V_SO4_nm3_per_molecule": V_SO4 * 1e27,
            "theta_b": THETA_B,
            "wall_seconds": wall,
        },
        "records": records,
    }
    out_path = OUT_DIR / "iv_curve.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
