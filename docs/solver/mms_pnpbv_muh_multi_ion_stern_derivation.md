# MMS Derivation — `logc_muh` + multi-ion shared-θ Bikerman + Stern Robin + parallel R2e/R4e BV

**Date:** 2026-05-13 (post-GPT-critique revision 2026-05-14, session 38)
**Scope:** byte-faithful Method-of-Manufactured-Solutions design for the
"no-speculative-physics" production stack used by
`scripts/studies/solver_demo_slide15_no_speculative_cs.py` and exercised by
`StudyResults/solver_demo_slide15_no_speculative_cs/`.
**Status:** derivation only; no source-builder code yet. Verdict APPROVED
after a 4-round adversarial GPT critique loop
(`docs/handoffs/CHATGPT_HANDOFF_38_mms-logc-muh-multi-ion-stern/`).

The "no-speculative-physics" subset of production is what the demo exercises:
formulation `logc_muh`; three dynamic species (O₂, H₂O₂, H⁺) with physical
hard-sphere `a_nondim`; two analytic Bikerman counterions (Cs⁺ and SO₄²⁻)
under the multi-ion shared-θ closure; Stern Robin BC at the electrode
(`C_S = 0.20 F/m²` after the two-stage anchor); parallel
R2e + R4e Butler–Volmer in log-rate form (Ruggiero 2022); `exponent_clip = 100`
and `u_clamp = 100`. Phase-6 speculative knobs
(`enable_water_ionization`, `lambda_hydrolysis`, Singh σ-pKa) are all off.

The existing MMS (`tests/test_mms_convergence.py`,
`scripts/verification/mms_bv_3sp_logc_boltzmann.py`) covers the older
**logc + single ClO₄⁻ counterion + electrode-Dirichlet** stack. The four
genuinely new pieces here are:

1. **muh substitution** `u_H → μ_H − em·z_H·φ` everywhere `u_H` appears
   (NP flux for H, BV cathodic-conc-factors, Poisson source `z_H·c_H`).
2. **Spatially-varying η** at the electrode because Stern-on uses
   `eta_raw = phi_applied − phi − E_eq` (`forms_logc_muh.py:393–394`).
3. **Shared-θ multi-ion Bikerman closure** for both Cs⁺ and SO₄²⁻,
   contributing to Poisson and to μ_steric simultaneously.
4. **Stern Robin boundary source** `g_S` on the φ-equation at the electrode.

Everything else (UnitSquareMesh sweep, `U_prev = U_manuf` with large dt,
side-wall geometry, expected L2 rate ≈ 2 / H1 rate ≈ 1 for CG1) carries
over unchanged. Source-quadrature degree initial candidate is
`SRC_QUAD_DEGREE = 8`, to be pinned by a one-time sweep in §7.

---

## 1. Primary unknowns and the muh transform

Mixed function space (`forms_logc_muh.py:299–308`):

  W = V_O2 × V_H2O2 × V_muH × V_φ

Primary unknowns:

  U = (u_O2, u_H2O2, μ_H, φ),    z = (0, 0, +1, ·)

with `em · z_H = +1` in the production scaling. Note **`U[h_idx]` is μ_H,
not u_H** (`forms_logc_muh.py:294`).

### Reconstruction (`forms_logc_muh.py:319–340`)

Define `clamp_u(x) := min(max(x, −U_CLAMP), +U_CLAMP)` with `U_CLAMP = 100`.

  u_expr_O2     = u_O2
  u_expr_H2O2  = u_H2O2
  u_expr_H      = μ_H − em · z_H · φ                   (muh reconstruction)

  c_O2     = exp(clamp_u(u_O2))
  c_H2O2   = exp(clamp_u(u_H2O2))
  c_H      = exp(clamp_u(μ_H − em · z_H · φ))           (muh)

  c_H,prev = exp(clamp_u(μ_H,prev − em · z_H · φ_prev))   ← **uses φ_prev**, not φ
                                                          (documented landmine,
                                                           `forms_logc_muh.py:174`)

### Time-stepping made negligible in MMS

We set `U_prev = U_manuf` (interpolated onto the FE space) and `dt = 1e15`.
The discrete time term `(c(U_h) − c(U_prev,h))/dt` is therefore O(h²/dt)
≈ O(10⁻¹⁵·h²) at u_exact, far below the FE discretization error of interest
(O(h²) for L2). It is not literally zero (U_prev,h is the CG1 projection
of the manufactured field, not the analytic field), but the contribution
is utterly negligible.

The φ_prev-vs-φ landmine (`forms_logc_muh.py:174`) is also neutralised
because φ_prev = φ at u_exact ⇒ c_H,prev = c_H at u_exact. The
**adjoint/Jacobian** sensitivity to φ_prev is not exercised by this MMS;
that's covered separately by
`tests/test_logc_muh_formulation.py::TestTransformResidualEquivalence`.

---

## 2. Production residual — exact UFL the MMS source must cancel

We mirror the production residual term-by-term in continuum and inject
the strong-form residual at `u_exact` as a forcing on `F_res`. We
**do not** use `ufl.replace` on F_res — a manufactured residual built
from `replace` would cancel both sides of any wiring bug symmetric across
the replace, exactly the cross-check this test must preserve.

### 2.1 Multi-ion shared-θ Bikerman closure (`boltzmann.py:91–268`)

Counterions `k ∈ {Cs⁺, SO₄²⁻}` with `z_k, c_k^bulk, a_k^ctn, phi_clamp_k`.

Per-ion exponential (`boltzmann.py:224–228`, each ion gets its **own** clamp):

  clamp_φ,k(φ) := min(max(φ, −phi_clamp_k), +phi_clamp_k)
  q_k(φ)       := exp(−z_k · clamp_φ,k(φ))

Bulk packing constant (`boltzmann.py:159`):

  A_dyn,bulk := Σ_i  a_i^dyn · c0_i
  A_an,bulk  := Σ_k  a_k^ctn · c_k^bulk
  θ_b        := 1 − A_dyn,bulk − A_an,bulk                      (> 0 by config validation)

Shared denominator at local φ (`boltzmann.py:241`):

  D(φ)  :=  θ_b  +  Σ_k  a_k^ctn · c_k^bulk · q_k(φ)

Free-dyn factor and floor (`boltzmann.py:205–206`):

  A_dyn(c_dyn) := Σ_i  a_i^dyn · c_i
  free_dyn     := max(1 − A_dyn, free_dyn_floor),  free_dyn_floor = 1e−10

Per-ion concentration expression (`boltzmann.py:254`):

  c_k^ster(φ; c_dyn)  :=  c_k^bulk · q_k(φ) · free_dyn / D(φ)

Per-ion packing contribution (`boltzmann.py:257`):

  P_k(φ; c_dyn) := a_k^ctn · c_k^ster(φ; c_dyn)

Per-ion charge contribution (`boltzmann.py:258`):

  ρ_k(φ; c_dyn) := z_k · c_k^ster(φ; c_dyn)

These are consumed by `forms_logc_muh.py:445–449` (μ_steric build) and
`:656–660` (Poisson source). The MMS source builder composes its own
UFL for these expressions independently from the production bundles
— see §7 "Independence policy".

### 2.2 Steric chemical potential (`forms_logc_muh.py:435–453`)

  θ_inner(φ; c_dyn) := 1 − A_dyn(c_dyn) − z_scale · Σ_k P_k(φ; c_dyn)
  packing            := max(θ_inner, packing_floor),  packing_floor ≈ 1e−8
  μ_steric           := − ln(packing)

At MMS runtime, `z_scale = 1.0` (the runtime z-ramp is an orchestrator
knob, not part of the residual we are verifying).

### 2.3 Nernst–Planck flux (interior, `forms_logc_muh.py:465–505`)

Production uses the **mobility flux** form `Jprod = +D · c · ∇(...)`
(`forms_logc_muh.py:499`), NOT the physical flux `N = −D · c · ∇(...)`.
The physical flux is `N = −Jprod`. All references to "J_i" in this
document mean Jprod (the production mobility flux).

For i ∈ {O₂, H₂O₂} (non-mu species, z_i = 0):

  ideal_grad_i := ∇u_i + em · z_i · ∇φ  =  ∇u_i           (since z_i = 0)
  J_i           := D_i · c_i · (ideal_grad_i + ∇μ_steric)

For i = H (mu species):

  ideal_grad_H := ∇μ_H                                    (∇u_H + em·z_H·∇φ = ∇μ_H)
  J_H           := D_H · c_H · (∇μ_H + ∇μ_steric)

Interior weak-form contribution (per species):

  F_res ∋  +∫_Ω  [(c_i − c_i,old)/dt] · v_i · dx  +  ∫_Ω  J_i · ∇v_i · dx

In MMS, `c_i,old ≈ c_i^ex` at u_exact ⇒ the time term is O(h²/dt) ≈ negligible.

### 2.4 Butler–Volmer (log-rate, parallel R2e + R4e)

η builder, **Stern-on branch** (`forms_logc_muh.py:393–402`):

  eta_raw_j(x,y)  =  φ_app^model  −  φ(x,y)  −  E_eq_j^model
  eta_scaled_j(x,y) = bv_exp_scale · eta_raw_j(x,y)
  eta_j(x,y)        = clamp(eta_scaled_j(x,y), ±exponent_clip)

In production scaling, both `phi_applied_model` and `E_eq_j_model` are
already stored in V_T units and `bv_exp_scale = 1.0`. We will not assume
this in the MMS source — use the `scaling['bv_exponent_scale']` Constant
verbatim.

Restriction to the electrode (y=0): η_j is **spatially varying** through
`φ(x, 0)`. This is the first non-trivial divergence from the existing MMS,
which assumed η constant on `ds(electrode)`.

Log-rate cathodic / anodic exponents (`forms_logc_muh.py:548–582`):

  log_cath_j(x,y) = ln(k0_j) + u_exprs[cat_j]
                  + Σ_f  power_f · (u_exprs[sp_f] − ln c_ref_f^nondim)
                  − α_j · n_e,j · η_j

  log_anod_j(x,y) = ln(k0_j) + u_exprs[anod_j] + (1 − α_j) · n_e,j · η_j     (if reversible AND anodic_species defined)
                 OR ln(k0_j) + ln c_ref,j^model + (1 − α_j) · n_e,j · η_j     (if reversible AND c_ref_model > 1e−30 AND anodic_species None — c_ref-anchored anodic)
                 OR  −∞  (irreversible ⇒ anodic = 0 by construction)

  R_j(x,y) = exp(log_cath_j) − exp(log_anod_j)

Critical: `u_exprs[h_idx] = μ_H − em·z_H·φ` (`forms_logc_muh.py:331–333`),
not raw `μ_H`. **This is the central muh substitution to test.**

For the demo's two reactions:

R2e (reversible, cathodic species O₂, anodic species H₂O₂, c_ref_R2e = 1.0):

  log_cath_R2e = ln(k0_R2e) + u_O2 + 2 · (μ_H − em·z_H·φ − ln c_H^ref) − α_R2e · n_e,R2e · η_R2e
  log_anod_R2e = ln(k0_R2e) + u_H2O2 + (1 − α_R2e) · n_e,R2e · η_R2e
  R_R2e       = exp(log_cath_R2e) − exp(log_anod_R2e)

R4e (irreversible, cathodic species O₂, c_ref_R4e = 0):

  log_cath_R4e = ln(k0_R4e) + u_O2 + 4 · (μ_H − em·z_H·φ − ln c_H^ref) − α_R4e · n_e,R4e · η_R4e
  R_R4e       = exp(log_cath_R4e)

Stoichiometry (script `:194,208`):

  s_O2   = (−1, −1)         (consumed by both R2e and R4e)
  s_H2O2 = (+1,  0)         (produced by R2e only)
  s_H    = (−2, −4)         (consumed by both, with R4e taking more)

Electrode contribution (`forms_logc_muh.py:613–615`):

  F_res ∋  −  Σ_j  Σ_i  s_{ij} · R_j · v_i · ds(electrode_marker)

After IBP this puts the natural BC at `Jprod_i · n = Σ_j s_{ij} · R_j`
at the electrode. For O₂ with s = −1 and R > 0 (cathodic), Jprod_O2 · n
= −R < 0 ⇒ Jprod_O2,y > 0 ⇒ ∂_y(u_O2 + μ_steric) > 0 at the electrode,
i.e. c_O2 increases toward the bulk (depletion at the surface). ✓

### 2.5 Poisson (`forms_logc_muh.py:638–660`)

  F_res ∋  +∫_Ω  ε_coeff · ∇φ · ∇w · dx
         −  ∫_Ω  charge_rhs · (Σ_i  z_i · c_i) · w · dx          (only z_H = 1 contributes; z_O2 = z_H2O2 = 0)
         −  ∫_Ω  z_scale · charge_rhs · (Σ_k  ρ_k(φ; c_dyn)) · w · dx

with `ε_coeff = scaling['poisson_coefficient']`,
`charge_rhs = scaling['charge_rhs_prefactor']`, `z_scale = 1.0` at MMS runtime.

### 2.6 Stern Robin (`forms_logc_muh.py:666–668`)

  F_res ∋  −  C_S^model · (φ_app^model − φ) · w · ds(electrode_marker)

with `C_S^model = float(scaling['bv_stern_capacitance_model'])`. The
production nondim conversion is in `:238–254`; for MMS we read the
nondim value off the ctx and use it as-is.

### 2.7 Boundary conditions imposed at build time

Concentrations:
- Bulk (y=1): Dirichlet `u_i = ln c0_i` for i ∈ {O₂, H₂O₂},
  `μ_H = ln c0_H` (since φ_bulk = 0).
- Electrode (y=0): no Dirichlet on c_i; production attaches BV-flux
  Neumann implicitly via the `−Σ_j s_{ij} R_j · v_i · ds(electrode)` term.

φ:
- Bulk (y=1): Dirichlet `φ = 0`.
- Electrode (y=0): **no Dirichlet** in the Stern-on branch
  (`forms_logc_muh.py:836` only attaches `bc_phi_electrode` in the
  Stern-off branch). The Stern Robin in §2.6 sets the natural BC.

Side walls (x=0, x=1): natural zero-flux for all fields (no Dirichlet,
no explicit ds term ⇒ implicit natural BC `J_i · n = 0` for each
species and `∇φ · n = 0` for Poisson).

---

## 3. Manufactured solution

### 3.1 Field shapes

Choose parameters `δ_O2, δ_H2O2, δ_H` ∈ (0, 1), `α₀, α₁, γ` ∈ ℝ:

  c_i^ex(x, y)   =  c0_i · [1 + δ_i · cos(πx) · (1 − y)²],     i ∈ {O₂, H₂O₂, H}
  φ^ex(x, y)    =  (1 − y) · [α₀ + α₁ · cos(πx)]  +  γ · y · (1 − y) · cos(πx)
  μ_H^ex(x, y) =  ln c_H^ex(x, y)  +  em · z_H · φ^ex(x, y)
  u_O2^ex      =  ln c_O2^ex
  u_H2O2^ex   =  ln c_H2O2^ex
  u_H^ex,recon =  μ_H^ex − em · z_H · φ^ex  =  ln c_H^ex          (muh consistency identity)

### 3.2 BC verification

**Bulk (y = 1):**
  c_i^ex(x, 1) = c0_i  ⇒ u_i^ex(x, 1) = ln c0_i  ✓
  φ^ex(x, 1) = 0  ✓
  μ_H^ex(x, 1) = ln c0_H + em·z_H·0 = ln c0_H  ✓
  ⇒ all bulk Dirichlets met by the manufactured shape; **no bulk BC override needed**.

**Electrode (y = 0):**
  c_i^ex(x, 0) = c0_i · [1 + δ_i · cos(πx)]      (BV-flux Neumann ⇒ source g_i^elec, §4.2)
  φ^ex(x, 0)  = α₀ + α₁ · cos(πx)                 (Stern Robin ⇒ source g_S, §4.4)

The Stern Robin is **non-trivially exercised** because φ^ex(x,0) ≠ φ_app^model
in general (we'll set α₀ ≠ φ_app^model = V_RHE/V_T = 21.4).

**Side walls (x = 0, x = 1):**
  ∂_x c_i^ex  =  − c0_i · δ_i · π · sin(πx) · (1 − y)²
              =  0  at x = 0, 1.   ✓
  ∂_x φ^ex  =  − π · sin(πx) · [(1 − y) · α₁ + γ · y · (1 − y)]
            =  0  at x = 0, 1.   ✓
  ⇒ natural zero-flux BCs (`J_i · n = 0`, `∇φ · n = 0`) auto-satisfied;
  **no side-wall source needed**.

### 3.3 Reconstructed exact closure quantities

All evaluated by direct UFL composition at the manufactured fields:

  A_dyn^ex(x,y)     = Σ_i  a_i^dyn · c_i^ex
  q_k^ex(x,y)      = exp(−z_k · φ^ex)                    (clamp identity by §5)
  D^ex(x,y)         = θ_b + Σ_k  a_k^ctn · c_k^bulk · q_k^ex
  c_k^ster,ex(x,y) = c_k^bulk · q_k^ex · (1 − A_dyn^ex) / D^ex      (free_dyn floor identity by §5)
  P_k^ex(x,y)      = a_k^ctn · c_k^ster,ex
  ρ_k^ex(x,y)      = z_k · c_k^ster,ex
  θ_inner^ex(x,y) = 1 − A_dyn^ex − Σ_k  P_k^ex
  μ_steric^ex(x,y) = − ln(θ_inner^ex)                     (packing floor identity by §5)

  J_i^ex          := D_i · c_i^ex · (∇u_i^ex + ∇μ_steric^ex),  i ∈ {O₂, H₂O₂}
  J_H^ex          := D_H · c_H^ex · (∇μ_H^ex + ∇μ_steric^ex)

  η_j^ex(x,y)     := bv_exp_scale · (φ_app^model − φ^ex − E_eq_j^model)    (clip identity by §5)

  R_R2e^ex(x,y)   :=  exp[ ln k0_R2e + u_O2^ex + 2(u_H^ex,recon − ln c_H^ref) − α_R2e · n_e,R2e · η_R2e^ex ]
                  −  exp[ ln k0_R2e + u_H2O2^ex                              + (1 − α_R2e) · n_e,R2e · η_R2e^ex ]

  R_R4e^ex(x,y)   :=  exp[ ln k0_R4e + u_O2^ex + 4(u_H^ex,recon − ln c_H^ref) − α_R4e · n_e,R4e · η_R4e^ex ]

---

## 4. Source terms — what we subtract from F_res

Pattern: each production residual `+∫ X · v dx` requires source
`−∫ X|_{u_exact} · v dx`; each production boundary `+∫ Y · v ds`
requires source `−∫ Y|_{u_exact} · v ds`. We use the divergence
theorem to convert weak-form flux terms into strong-form sources
on the interior (and inherit the natural-BC residual on the boundary).

### 4.1 NP interior source (i ∈ {O₂, H₂O₂, H})

The production residual contains `+∫ J_i · ∇v_i · dx`. IBP gives:

  ∫ J_i · ∇v_i · dx  =  −∫ (∇·J_i) · v_i · dx  +  ∫ J_i · n · v_i · ds(∂Ω)

At u_exact the interior strong-form residual is `−∇·J_i^ex`. Subtract:

  S_c_i(x,y)  :=  −∇·J_i^ex(x,y)
  F_res ←  F_res  −  ∫_Ω  S_c_i · v_i · dx_q

with `dx_q = fd.dx(degree=SRC_QUAD_DEGREE)`. Initial candidate
`SRC_QUAD_DEGREE = 8`; final value pinned by the sweep in §7.

The ds(∂Ω) IBP byproduct splits as:
- Bulk (y=1): w-component of `v_i` test functions is Dirichlet-zeroed
  (`u_i, μ_H` Dirichlets at bulk) ⇒ no ds contribution.
- Side walls: `J_i^ex · n` ≡ 0 by §3.2 ⇒ no ds contribution.
- Electrode (y=0): handled in §4.2.

### 4.2 NP electrode boundary source

The production residual at the electrode is

  +∫ J_i · n · v_i · ds(elec)  (IBP byproduct from §4.1, **not** explicitly written
                                  but built up by Newton from the weak form)
  −∫ Σ_j s_{ij} · R_j · v_i · ds(elec)  (explicit BV term, line :615)

At u_exact, the boundary residual is

  R_i^elec,ex(v_i) = ∫_{y=0}  [ J_i^ex · n − Σ_j s_{ij} · R_j^ex ]  ·  v_i  ·  ds

Subtract that integrand:

  g_i^elec(x)  :=  J_i^ex · n |_{y=0}  −  Σ_j  s_{ij} · R_j^ex |_{y=0}
  F_res ←  F_res  −  ∫_{y=0}  g_i^elec · v_i · ds_q(electrode_marker)

**Mobility vs physical flux:** production uses the mobility form
`Jprod = +D · c · ∇(...)` (no leading minus, see `forms_logc_muh.py:499`).
The physical Nernst–Planck flux is `N = −Jprod`. All `J_i^ex` in this
section are mobility flux. The natural BC at u_exact reads
`Jprod_i · n = Σ_j s_{ij} · R_j` (mobility flux equals stoichiometric
production rate at the surface), which in physical terms means
`N_i · n = − Σ_j s_{ij} · R_j` (physical flux equals minus stoichiometric
rate, i.e. species are produced into the electrolyte at rate +s·R when
s > 0).

With outward normal `n = (0, −1)` at y = 0, the explicit hand expression is:

  J_i^ex,y(x, y)        = +D_i · c_i^ex · ∂_y(u_i^ex + μ_steric^ex)              (i ∈ {O₂, H₂O₂}; z_i = 0)
  J_i^ex · n |_{y=0}   = J_i^ex,y(x, 0) · (−1)
                       = −D_i · c_i^ex|_{y=0} · ∂_y(u_i^ex + μ_steric^ex)|_{y=0}

  J_H^ex,y(x, y)        = +D_H · c_H^ex · ∂_y(μ_H^ex + μ_steric^ex)
  J_H^ex · n |_{y=0}   = −D_H · c_H^ex|_{y=0} · ∂_y(μ_H^ex + μ_steric^ex)|_{y=0}

(Implementation uses `fd.dot(J_i^ex, fd.FacetNormal(mesh))` and lets
Firedrake handle the outward-normal sign automatically; the manual
expression above is for derivation transparency only.)

`R_j^ex|_{y=0}` is **x-dependent** because η_j^ex depends on φ^ex(x, 0)
in the Stern-on branch (§2.4) and the cathodic factors depend on
`u_H^ex,recon(x, 0) = ln c_H^ex(x, 0) = ln c0_H + ln(1 + δ_H cos(πx))`.

### 4.3 Poisson interior source

Production residual on dx:

  +∫ ε_coeff · ∇φ · ∇w · dx
  −∫ charge_rhs · (Σ_i  z_i · c_i) · w · dx
  −∫ z_scale · charge_rhs · (Σ_k  ρ_k(φ; c_dyn)) · w · dx

IBP on the Laplacian:

  ∫ ε_coeff · ∇φ · ∇w · dx  =  −∫ ε_coeff · ∇²φ · w · dx  +  ∫ ε_coeff · ∇φ · n · w · ds(∂Ω)

Interior strong-form residual at u_exact:

  S_φ(x,y)  :=  −ε_coeff · ∇²φ^ex
              −  charge_rhs · (z_H · c_H^ex)
              −  z_scale · charge_rhs · Σ_k  z_k · c_k^ster,ex
  F_res ←  F_res  −  ∫_Ω  S_φ · w · dx_q

For our chosen `φ^ex = (1−y)(α₀ + α₁ cos πx) + γ · y(1−y) · cos πx`:

  ∂_x φ^ex     =  −π · sin(πx) · [(1−y) · α₁ + γ · y · (1−y)]
  ∂²_x φ^ex   =  −π² · cos(πx) · [(1−y) · α₁ + γ · y · (1−y)]
  ∂_y φ^ex     =  −(α₀ + α₁ cos πx)  +  γ · (1 − 2y) · cos(πx)
  ∂²_y φ^ex   =  −2γ · cos(πx)

  ∇²φ^ex      =  ∂²_x φ^ex  +  ∂²_y φ^ex
              =  −π² · cos(πx) · [(1−y) · α₁ + γ · y · (1−y)]  −  2γ · cos(πx)

(Implementation: use UFL `fd.div(fd.grad(phi_exact))` instead of the
hand-derived expression — same result, less risk of transcription error.)

The ds(∂Ω) IBP byproduct splits as:
- Bulk (y=1): φ is Dirichlet (w-component zeroed) ⇒ no ds contribution.
- Side walls: ∂_x φ^ex = 0 ⇒ ∇φ^ex · n = 0 ⇒ no ds contribution.
- Electrode (y=0): handled in §4.4.

### 4.4 Stern Robin boundary source

The production φ-equation residual on `ds(elec)` is

  R_φ^elec(w; U)  =  ∫_{y=0}  [ ε_coeff · ∇φ · n  −  C_S^model · (φ_app^model − φ) ]  ·  w  ·  ds

  where the ε_coeff · ∇φ · n term is the IBP byproduct from the
  ε_coeff · ∇φ · ∇w · dx Laplacian term (`forms_logc_muh.py:644`) and the
  −C_S^model · (φ_app^model − φ) term is the explicit Stern Robin
  (`:668`).

At u_exact, subtract:

  g_S(x)  :=  ε_coeff · (∇φ^ex · n) |_{y=0}  −  C_S^model · (φ_app^model − φ^ex|_{y=0})
  F_res ←  F_res  −  ∫_{y=0}  g_S · w · ds_q(electrode_marker)

Concrete pieces:

  (∇φ^ex · n) |_{y=0}  =  −  ∂_y φ^ex(x, 0)
                         =  −  [ −(α₀ + α₁ cos πx)  +  γ · (1 − 2·0) · cos(πx) ]
                         =  (α₀ + α₁ cos πx)  −  γ · cos(πx)
                         =  α₀  +  (α₁ − γ) · cos(πx)

  φ^ex(x, 0)  =  α₀ + α₁ · cos(πx)

  g_S(x)  =  ε_coeff · [α₀ + (α₁ − γ) · cos(πx)]
            −  C_S^model · [φ_app^model − α₀ − α₁ · cos(πx)]

**Sign check:** the production weak form has `−C_S^model · (φ_app − φ) · w · ds`,
so at u_exact the residual is
`[ε_coeff · ∇φ^ex · n − C_S^model · (φ_app − φ^ex)] · w · ds`. Subtracting
this integrand cancels the residual exactly. ✓

(Implementation: use `fd.dot(fd.grad(phi_exact), fd.FacetNormal(mesh))` and
let Firedrake handle the outward-normal sign.)

---

## 5. Clip and floor safety at the chosen test voltage

Test voltage: **V_RHE = +0.55 V**, matching the demo's anchor.

In production scaling, `phi_applied_model = V_RHE / V_T ≈ 21.41` and
`E_eq_j^model = E_eq_j / V_T` (so `E_eq_R2e^model ≈ 27.04`,
`E_eq_R4e^model ≈ 47.86`), with `bv_exp_scale = 1.0`. (Read these off
`scaling` at MMS runtime rather than hard-coding.)

### 5.1 BV `exponent_clip = 100`

**Recommended envelope:** `α₀ = α₁ = γ = 0.5`, so φ^ex(x,0) =
α₀ + α₁ cos(πx) ∈ [0, 1] (γ vanishes at y = 0 because of the y(1−y)
factor).

  η_R2e_raw^ex(x,0) = 21.41 − φ^ex(x,0) − 27.04   ∈   [−6.63, −5.63]
  η_R4e_raw^ex(x,0) = 21.41 − φ^ex(x,0) − 47.86   ∈   [−27.45, −26.45]

**Broad envelope** (sanity check at `|α₀|, |α₁|, |γ| ≤ 1`, φ^ex(x,0) ∈
[−2, +2]):

  η_R2e_raw^ex(x,0)  ∈  [−7.63, −3.63]
  η_R4e_raw^ex(x,0)  ∈  [−28.45, −24.45]

All inside ±100 ⇒ **clip identity at u_exact**. ✓

### 5.2 `u_clamp = 100`

  |u_O2^ex|, |u_H2O2^ex|, |ln c_H^ex| are O(|ln c0_i| + 1) ≪ 100.   ✓

### 5.3 Counterion `phi_clamp_k = 50`

Production clamps `φ` to ±phi_clamp_k BEFORE multiplying by `−z_k`
(`boltzmann.py:224–228`). The identity condition is therefore
`max|φ^ex| < phi_clamp_k`:

  max|φ^ex|  ≤  1.25  ≪  50.   ✓

⇒ q_k^ex = exp(−z_k · φ^ex) is the unclamped expression at u_exact.

### 5.4 `packing_floor = 1e−8` and `free_dyn_floor = 1e−10`

Demo dynamic-species hard-sphere `a_i^dyn`: 1.49e−5, 2.42e−5, 6.65e−5
(O₂, H₂O₂, H⁺ at r = 1.70, 2.00, 2.80 Å). With `C_SCALE = C_O2 = 1.2 mol/m³`
(`_bv_common.py:133`):

  c0_O2_nondim  = 1.0                                     (since C_SCALE = C_O2)
  c0_H2O2_nondim ≈ small seed (negligible)
  c0_H_nondim    = C_HP / C_SCALE = 0.1 mol/m³ / 1.2 mol/m³ ≈ 0.0833       (pH 4 ⇒ [H⁺] = 1e−4 M = 0.1 mol/m³)

Worst-case A_dyn^ex over the recommended envelope (δ_i = 0.3):

  A_O2^ex,max ≈ a_O2 · 1.3 · c0_O2 = 1.49e−5 · 1.3 · 1.0 ≈ 1.94e−5
  A_H^ex,max  ≈ a_H · 1.3 · c0_H = 6.65e−5 · 1.3 · 0.0833 ≈ 7.2e−6
  ⇒  A_dyn^ex ≲ 3e−5

free_dyn = 1 − A_dyn^ex ≈ 0.99997, close to 1 but **not identical**.
`free_dyn_floor = 1e−10` is inactive. The source builder MUST keep
(1 − A_dyn^ex) exactly; do NOT simplify to 1.

Production counterion constants from `scripts/_bv_common.py:707–708`:

  A_CSPLUS_HAT = 3.23e−5    (Cs⁺ r = 2.2 Å; nondim hard-sphere)
  A_SO4_HAT    = 4.20e−5    (SO₄²⁻ r = 2.4 Å; nondim hard-sphere)
  C_CSPLUS_HAT = 200 mol/m³ / 1.2 mol/m³ ≈ 166.7        ([Cs⁺] = 0.2 M = 200 mol/m³)
  C_SO4_HAT    = 100 mol/m³ / 1.2 mol/m³ ≈ 83.3         ([SO₄²⁻] = 0.1 M = 100 mol/m³)

  ⇒  a_Cs · c_Cs   ≈ 5.38e−3
  ⇒  a_SO4 · c_SO4 ≈ 3.50e−3
  ⇒  A_an,bulk    ≈ 8.88e−3
  ⇒  θ_b           ≈ 0.991

(These are quoted for reference only. The MMS source builder reads the
production values at runtime via `_get_bv_boltzmann_counterions_cfg(params)`;
do NOT hardcode.)

At **recommended envelope** (φ^ex(x,0) ∈ [0, 1]), worst case φ = +1
(max sulfate attraction):

  q_SO4 = exp(+2) ≈ 7.39,    q_Cs = exp(−1) ≈ 0.368
  D     = 0.991 + 3.50e−3 · 7.39 + 5.38e−3 · 0.368 ≈ 1.019
  P_SO4 = a_SO4 · c_SO4 · q_SO4 / D ≈ 3.50e−3 · 7.39 / 1.019 ≈ 0.0254
  P_Cs  = a_Cs · c_Cs · q_Cs / D ≈ 5.38e−3 · 0.368 / 1.019 ≈ 0.00194
  Σ P_k ≈ 0.0273
  θ_inner^ex ≳ 0.973

All three floors (`packing_floor`, `free_dyn_floor`, ion `phi_clamp`) are
**identity at u_exact** for the recommended envelope.

**Runtime invariant** (asserted in source builder before solve, via a
discrete-min check, NOT an integrand-clip integral):

```python
# Approach A (recommended): assemble an indicator and assert near-zero measure
mesh = ctx["mesh"]
small_region = fd.assemble(
    fd.conditional(theta_inner_ex < 1e-7, fd.Constant(1.0), fd.Constant(0.0))
    * fd.dx(domain=mesh)
)
vol_omega = fd.assemble(fd.Constant(1.0) * fd.dx(domain=mesh))
assert small_region < 1e-12 * vol_omega
```

`fd.assemble(fd.min_value(theta_inner, threshold) * dx)` is **NOT** a
valid global-min check — it's the integral of a clipped field.

### 5.5 Parameter envelope and K0_R4e_factor

  δ_O2 = δ_H2O2 = δ_H = 0.30       (matches existing MMS pattern)
  α₀ = 0.5,  α₁ = 0.5,  γ = 0.5    ⇒ |φ^ex| ≤ 1.25  (safely inside §5.1–4)
  V_RHE = +0.55 V                  ⇒ η_R2e ∈ [−6.63, −5.63], η_R4e ∈ [−27.45, −26.45]

**K0_R4e_factor = 1e−18** (primary, matches the demo's smallest factor).

Rationale: at K0_R4e_factor = 1 (i.e. k0_R4e = k0_R2e), the cathodic
log-rate exponents differ by

  log_cath_R4e − log_cath_R2e = (α_R4e·n_e,R4e − α_R2e·n_e,R2e)·|η_R4e|
                              ≈ 0.5·4·26.5 − 0.627·2·5.6
                              ≈ 53 − 7
                              ≈ 46

so R4e/R2e ~ e^46 ≈ 10^20. R4e overwhelmingly dominates and the
absolute residual magnitude (~k0·c·e^53) overwhelms Newton's relative
tolerance ability to drive residual to the discretization-error floor.
At K0_R4e_factor = 1e−18 the cathodic log-rate difference becomes

  log_cath_R4e − log_cath_R2e ≈ −18·ln(10) + 46 ≈ +5.1

so R4e/R2e ≈ e^5.1 ≈ 165: R4e is dominant but not catastrophic, the
operator-of-test exercises both reaction branches meaningfully, and
residual magnitudes are tractable for Newton's `rtol`.

**Runtime invariants** (asserted in source builder before solve):

```python
# R4e dominant-but-not-catastrophic in boundary L2 norm
mesh = ctx["mesh"]
ds_e = fd.ds(ctx['bv_settings']['electrode_marker'], domain=mesh)
R2e_norm = float(fd.assemble(R_R2e_ex**2 * ds_e))**0.5
R4e_norm = float(fd.assemble(R_R4e_ex**2 * ds_e))**0.5
R_ratio = R4e_norm / max(R2e_norm, 1e-300)
assert 10 < R_ratio < 1e5,  (
    f"R4e/R2e = {R_ratio:.3e} outside finite window — "
    f"K0_R4e_factor likely mis-set"
)
```

Lower bound `R_ratio > 10` ensures R4e is non-trivially present
(catches K0_R4e_factor = 0 or sign flip). Upper bound `R_ratio < 1e5`
ensures the source magnitude stays in a regime where Newton's `rtol`
can drive the residual to the discretization-error floor.

**Stern-off cross-check is NOT included in this plan.** At
φ_app^model ≈ 21.4, SO₄²⁻ has q_SO4 = exp(2 · 21.4) = exp(42.8) ≈ 4·10^18,
which saturates the shared-θ closure: P_SO4 → 1 − A_dyn ≈ 1, θ_inner → 0,
and `packing_floor` activates. This contradicts the "clamps/floors
inactive at u_exact" premise. Testing the Stern-off path requires
either a small-V_RHE design (V_RHE ≲ 0.1 V keeps the closure
unsaturated) or a separate saturation-active MMS with different
convergence-rate expectations. Both are out of scope here.

---

## 6. Convergence-test coverage by source-term group

| Source piece | Catches a bug in |
|---|---|
| `−∇·J_i` for i ∈ {O₂, H₂O₂} | μ_steric inclusion in NP flux (`:499`); shared-θ contributions of Cs⁺/SO₄²⁻ to packing (`:447–449`); D_i and c_i^ex scaling |
| `−∇·J_H` | muh reconstruction in mass-balance (∇u_H replaced by ∇μ_H, `:473`); μ_steric coupling for proton |
| `R_R2e, R_R4e` via `u_exprs[h]` | muh substitution in BV (`:551–563`); cathodic_conc_factors powers (2 vs 4); α·n_e signs; reversible-vs-irreversible branch; ln(k0_j) sign |
| Spatial η^ex(x,0) | Stern-on branch of `_build_eta_clipped` (`:393–394`); φ-coupling into BV that **only** happens when use_stern=True |
| Poisson interior `z_H·c_H + Σ_k z_k c_k^ster` | shared-θ closure numerator `(1 − A_dyn)` (`:254`); per-ion clamp inside D(φ) (`:224–227`); sign on `z_scale·charge_rhs` (`:658–660`) |
| `g_S` (Stern Robin) | sign of `−C_S·(φ_app − φ)` (`:668`); **use** of the ctx-stored `bv_stern_capacitance_model` in the residual; IBP byproduct from Laplacian on `ds(elec)` |

The MMS does **NOT** cover (by design):
- **Inactive clamp/floor branches**: `u_clamp`, ion `phi_clamp_k`,
  `exponent_clip`, `free_dyn_floor`, `packing_floor` are identity at
  u_exact (see §5). Bugs in the clamp branches (e.g. wrong sign in
  `fd.min_value`) are not exercised. A separate clip-activation MMS
  would be needed (see §8).
- **Runtime mutation of ctx values**: voltage continuation via
  `ctx['phi_applied_func'].assign(...)`, `set_stern_capacitance_model`,
  K0 ramp, `boltzmann_z_scale` ramp. The test asserts no mutation
  fires between source build and solve, so bugs in mutators are not
  caught.
- **Stern coefficient nondim conversion** at form-build time
  (`forms_logc_muh.py:238–254`): the source builder reads the converted
  `bv_stern_capacitance_model` value, so an off-by-factor in the
  conversion would change BOTH residual and source equally.
- **c_ref-anchored anodic branch** (`forms_logc_muh.py:574–580`): both
  R2e (has anodic_species) and R4e (irreversible) bypass this branch.
- **Legacy per-species `bv_c_ref_model_vals` path** (`:617–636`): gated
  off by `use_reactions=True` (i.e. `bv_reactions` list non-empty).
- **IC machinery** (`debye_boltzmann` composite-ψ + multispecies-γ in
  `picard_ic.py`): MMS supplies its own IC via `U.assign(U_manuf)`.
- **Anchor / continuation logic** (`solve_anchor_with_continuation`,
  `solve_grid_with_anchor`): single solve, no continuation.
- **Phase-6 speculative physics**: already off by construction.

---

## 7. Implementation notes for the source builder

(Out of scope for this derivation; deferred to a follow-up plan / PR.
This section captures the design constraints the source-builder
implementation must obey.)

### Reuse from `scripts/verification/mms_bv_3sp_logc_boltzmann.py`

- `compute_rates`, `plot_convergence`: keep verbatim.
- `make_sp_production`: adapt to muh + multi-ion + Stern factory call.
- `run_mms` outer loop: adapt for muh-aware interpolation and error
  norms (see below).
- `_ufl_l2_error`, `_ufl_h1_error`: still take `(u_ufl, u_h, mesh)` —
  call sites must pass `μ_H^ex_func` / `U.sub(h_idx)` for the proton
  (NOT `u_H^ex_func`).
- `_build_manufactured_source`: rewrite for the four new pieces.

### muh-aware test harness

- **U_manuf interpolation**: build `U_manuf` by interpolating each
  primary unknown onto the FE space:
  - `U_manuf.sub(O₂_idx) ← Interpolator(u_O2^ex, V_O2)`
  - `U_manuf.sub(H₂O₂_idx) ← Interpolator(u_H2O2^ex, V_H2O2)`
  - `U_manuf.sub(h_idx) ← Interpolator(μ_H^ex, V_muH)`  ← **NOT u_H^ex**
  - `U_manuf.sub(phi_idx) ← Interpolator(φ^ex, V_phi)`
- **Initial guess**: `U.assign(U_manuf); U_prev.assign(U_manuf)`.
  Places Newton inside the basin of u_exact.
- **Error norms on primary unknowns**:
  - O₂: `||U.sub(O₂_idx) − u_O2^ex_func||_{L2, H1}`
  - H₂O₂: `||U.sub(H₂O₂_idx) − u_H2O2^ex_func||_{L2, H1}`
  - H: `||U.sub(h_idx) − μ_H^ex_func||_{L2, H1}`  ← **proton error is on μ_H**
  - φ: `||U.sub(phi_idx) − φ^ex_func||_{L2, H1}`
- **Concentration diagnostics** (for plot annotations only): compute
  `c_H_h = exp(U.sub(h_idx) − em·z_H·U.sub(phi_idx))`, not `exp(U.sub(h_idx))`.

### Independence policy

The MMS source builder composes its own UFL for `c_k^ster, P_k, ρ_k, D(φ)`
from `_get_bv_boltzmann_counterions_cfg(params)` directly, mirroring
`boltzmann.py:91–268` algebraically but **without** consuming the
production-side `steric_boltz` bundle expressions. This preserves the
source-residual independence the MMS design relies on (any wiring bug
symmetric across both sides would otherwise cancel and give a passing
convergence rate against a buggy operator).

The source builder still references live ctx Function/Constant objects
for parameters that can be ramped at runtime (`phi_applied_func`,
`stern_coeff_const`, `bv_k0_funcs`, `bv_alpha_funcs`, `boltzmann_z_scale`)
SO THAT scalar metadata stays in lockstep with the residual side. But
the closure expressions themselves are independently composed.

### Stack-invariant asserts (source builder)

Before injecting sources into F_res, assert (using **correct dict
lookups** verified against `forms_logc_muh.py:194,196,219,641` and
`scripts/_bv_common.py`):

```python
from Forward.bv_solver.config import _get_bv_convergence_cfg, _get_bv_boltzmann_counterions_cfg
from Nondim.transform import _get_nondim_cfg

mesh       = ctx['mesh']
scaling    = ctx['nondim']
conv_cfg   = _get_bv_convergence_cfg(solver_params)
nondim_cfg = _get_nondim_cfg(solver_params)
counter_cfg = _get_bv_boltzmann_counterions_cfg(solver_params)
snes_opts  = solver_params[10]   # the params dict carrying SNES_OPTS_CHARGED

# reactions list non-empty (= use_reactions True; gates BV log-rate path)
bv_rxns = scaling.get('bv_reactions', [])
assert isinstance(bv_rxns, (list, tuple)) and len(bv_rxns) == 2

# log-rate BV (gates the muh-aware u_exprs[h] substitution path)
assert bool(conv_cfg.get('bv_log_rate', False)) is True

# Stern-on branch (gates spatially-varying η at electrode AND g_S source)
csm = scaling.get('bv_stern_capacitance_model')
assert csm is not None and float(csm) > 0

# Poisson source enabled (lives on nondim_cfg, NOT conv_cfg)
assert not bool(nondim_cfg.get('suppress_poisson_source', False))

# speculative physics off
from Forward.bv_solver.water_ionization import is_water_ionization_enabled
from Forward.bv_solver.cation_hydrolysis import is_cation_hydrolysis_enabled
assert not is_water_ionization_enabled(conv_cfg)
assert not is_cation_hydrolysis_enabled(conv_cfg)

# species count
assert ctx['n_species'] == 3

# exactly two bikerman counterions
bikerman = [e for e in counter_cfg if e.get('steric_mode') == 'bikerman']
assert len(bikerman) == 2

# no Γ-augmented mixed space
indices = ctx.get('mixed_space_indices')
assert indices is None or indices.gamma_index is None

# time term negligible
assert float(scaling.get('dt_model', 0.0)) >= 1e12

# packing-floor margin (discrete-min via indicator)
small_region_measure = fd.assemble(
    fd.conditional(theta_inner_ex < 1e-7, fd.Constant(1.0), fd.Constant(0.0))
    * fd.dx(domain=mesh)
)
vol_omega = fd.assemble(fd.Constant(1.0) * fd.dx(domain=mesh))
assert small_region_measure < 1e-12 * vol_omega

# R4e dominant-but-not-catastrophic in boundary L2 norm vs R2e
ds_e = fd.ds(ctx['bv_settings']['electrode_marker'], domain=mesh)
R2e_norm = float(fd.assemble(R_R2e_ex**2 * ds_e))**0.5
R4e_norm = float(fd.assemble(R_R4e_ex**2 * ds_e))**0.5
R_ratio  = R4e_norm / max(R2e_norm, 1e-300)
assert 10 < R_ratio < 1e5

# SNES tolerances tight enough to drive Newton to discretization-error floor
assert float(snes_opts.get('snes_atol', 1e-7)) <= 1e-5
assert float(snes_opts.get('snes_rtol', 1e-8)) <= 1e-7
```

### Live continuation policy

For the convergence-rate test, **no continuation setter or runtime
mutation (voltage assign via `ctx['phi_applied_func'].assign(...)`,
`set_stern_capacitance_model`, K0 ramp, `boltzmann_z_scale` ramp) fires
between `_build_manufactured_source(ctx)` and `solve(F == 0, U)`.** This
keeps the source UFL and residual UFL both at their build-time state;
mutator behavior is explicitly out of scope (see §6 "does NOT cover").

### Quadrature degree

Source integrands include `c_i · ∂_y(ln(packing))`, where `packing` is a
rational function of `exp(−z·φ^ex)` with multiple `q_k`'s. Initial
candidate degree = 8.

**One-time sweep** before pinning the production degree: run at degree
∈ {6, 8, 10, 12, 16} on N = 32 (mid-mesh). Record L2 error per primary
unknown for each degree. Pin `SRC_QUAD_DEGREE` to the smallest value at
which the L2 error plateaus within 1% of the degree-16 value. Document
the chosen degree and the sweep results in the test file's docstring.

### Test file layout

`tests/test_mms_logc_muh_multi_ion_stern.py`, mirroring
`tests/test_mms_convergence.py`. Classes:

- `TestMMSConvergence`: UnitSquareMesh sweep N ∈ {8, 16, 32, 64}; assert
  L2 slope ≥ 1.8 and H1 slope ≥ 0.8 per primary unknown
  (`u_O2, u_H2O2, μ_H, φ`), R² > 0.99. K0_R4e_factor = 1e−18.
- `TestMMSProductionGradedMesh`: single-solve recovery on the demo mesh
  (Nx=8, Ny=80, β=3); thresholds sized ~6× above baseline.

(No `TestSternOffSanity`, no `TestMMSConvergence_K0R4eSecondary` — both
removed during the GPT critique loop. See `docs/handoffs/CHATGPT_HANDOFF_38_*/FINAL_REVISION.md` for the issue ledger.)

---

## 8. What's deliberately deferred

- **Time-dependent verification**: this MMS verifies the steady-state
  spatial operator. Time-stepping is exercised by setting U_prev=U_manuf
  and dt = 1e15, so `(c − c_old)/dt` is O(h²/dt) ≈ negligible. A
  separate time-MMS would manufacture a c(x,y,t) and test temporal
  accuracy — orthogonal scope.
- **Adjoint Jacobian check**: a Taylor-test on the linearisation around
  u_exact would catch sign errors in the muh transform that are
  symmetric across the residual but break the Jacobian. Recommended
  follow-up, but not part of the convergence-rate MMS.
- **Clip-activation MMS**: mirror of `mms_voltage_sweep.py` adapted to
  muh + Stern + multi-ion; pushes V_RHE high enough to engage
  `exponent_clip` and quantify the convergence-rate degradation.
  Orthogonal to baseline-correctness verification here.
- **Bikerman-saturation regime**: at high cathodic drive the multi-ion
  closure saturates (`free_dyn → 0`, `packing → packing_floor`).
  Testing the floor branch is a separate MMS design problem; the
  existing single-counterion attempt is xfail in
  `tests/test_mms_steric_boltzmann_convergence.py` for this reason.
- **c_ref-anchored anodic branch** (`forms_logc_muh.py:574–580`): not
  exercised by R2e or R4e. A separate single-reaction MMS with
  `reversible=True, anodic_species=None, c_ref_model=1.0` would cover
  this code path.
- **Legacy per-species `bv_c_ref_model_vals` path** (`:617–636`): gated
  off by `use_reactions=True` (reactions list non-empty). Not in scope
  for the production stack.
- **Runtime mutation of ctx['phi_applied_func']** (voltage continuation),
  `set_stern_capacitance_model`, K0 ramp, `boltzmann_z_scale` ramp:
  verified by separate end-to-end runs (e.g. `phase6b_v10b` calibration).
- **Clamp-branch behavior**: `u_clamp`, ion `phi_clamp_k`,
  `exponent_clip`, `free_dyn_floor`, `packing_floor` are all identity
  at u_exact. Bugs in the clamp branches (wrong `fd.min_value` sign,
  wrong threshold) are not caught by this MMS.
- **K0_R4e_factor = 1 regime**: removed from this convergence-rate plan
  because R4e/R2e ~ e^46 ≈ 10^20 puts the operator-of-test in a
  R4e-only-dominant regime whose Newton-tolerance and quadrature-margin
  characteristics need separate pilot analysis.
- **Stern-off path**: at production V_RHE = 0.55 V, the Stern-off
  Dirichlet `φ(x,0) = φ_app^model ≈ 21.4` saturates the SO₄²⁻ closure
  (q_SO4 = exp(42.8) ≈ 4·10^18 ⇒ θ_inner → 0 ⇒ packing_floor active).
  Needs either a separate small-V_RHE design or a saturation-active MMS.
