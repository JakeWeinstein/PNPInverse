# Critique session 38 — Round 1

## Section 1: Context bundle

### Why this exists

I'm designing a Method-of-Manufactured-Solutions (MMS) convergence test for a production Poisson–Nernst–Planck + Butler–Volmer (PNP-BV) forward solver. The existing MMS (`tests/test_mms_convergence.py` + `scripts/verification/mms_bv_3sp_logc_boltzmann.py`) covers an older, simpler stack: `formulation='logc'`, single ClO₄⁻ analytic-Boltzmann counterion, Dirichlet on φ at the electrode. The current production stack is different on four axes, each of which is a way for the source builder to silently pass a buggy operator:

1. **`formulation='logc_muh'`**: the proton's primary unknown is the electrochemical potential `μ_H = u_H + em·z_H·φ`, not `u_H = ln c_H`. Concentration is reconstructed as `c_H = exp(μ_H − em·z_H·φ)`.
2. **Spatially-varying η at the electrode**: in the Stern-on branch (production), `eta_raw = phi_applied − phi − E_eq`. The existing MMS assumed η is spatially constant on `ds(electrode)`.
3. **Shared-θ multi-ion Bikerman closure**: two analytic counterions (Cs⁺ and SO₄²⁻) share a single denominator in the closure, and both contribute to (a) Poisson source and (b) the dynamic species' steric chemical potential μ_steric.
4. **Stern Robin BC at the electrode**: `F_res ∋ −C_S·(φ_app − φ)·w·ds(elec)`. No Dirichlet on φ at the electrode in the Stern-on branch.

Failure mode I most want to catch: an MMS that "passes" against a buggy production operator because the source builder makes the same sign/transform mistake the operator makes, so both cancel.

### Stack under test (no-speculative-physics subset of production)

From `scripts/studies/solver_demo_slide15_no_speculative_cs.py` + `summary.json`:

- formulation `logc_muh`; primary unknowns U = (u_O2, u_H2O2, μ_H, φ).
- 3 dynamic species: O₂ (z=0), H₂O₂ (z=0), H⁺ (z=+1), all with **physical** hard-sphere `a_nondim` (1.49e−5, 2.42e−5, 6.64e−5).
- 2 analytic Bikerman counterions in multi-ion shared-θ closure: Cs⁺ (z=+1), SO₄²⁻ (z=−2).
- `em·z_H = +1` in the production nondim scaling.
- Stern Robin at electrode, C_S = 0.20 F/m² (after nondim conversion).
- Parallel BV reactions:
  - R2e (reversible, irrev cathodic only — wait, no, R2e IS reversible): O₂ + 2H⁺ + 2e⁻ → H₂O₂; E°=0.695 V; α=0.627; n_e=2; cathodic_species=O₂; anodic_species=H₂O₂; c_ref=1.0; stoichiometry (O₂, H₂O₂, H⁺) = (−1,+1,−2); cathodic_conc_factor: species=H⁺, power=2, c_ref_nondim=C_HP_HAT.
  - R4e (irreversible): O₂ + 4H⁺ + 4e⁻ → 2H₂O; E°=1.23 V; α=0.5; n_e=4; cathodic_species=O₂; anodic_species=None; c_ref=0.0 (signals "no anodic anchor"); stoichiometry (O₂, H₂O₂, H⁺) = (−1, 0, −4); cathodic_conc_factor: species=H⁺, power=4, c_ref_nondim=C_HP_HAT.
- log-rate BV (`bv_log_rate=True`).
- `exponent_clip=100`, `u_clamp=100`.
- `enable_water_ionization=False`, `lambda_hydrolysis=0`.

Test voltage chosen: `V_RHE = +0.55 V` (anchor in the demo). In production scaling `phi_applied_model = V_RHE / V_T ≈ 21.41`, `E_eq_j^model = E_eq_j / V_T`, and `bv_exp_scale = 1.0`. (Concretely: V_T = 0.0257 V at T=298 K. So η_R2e = (0.55 − 0.695)/V_T ≈ −5.64. η_R4e = (0.55 − 1.23)/V_T ≈ −26.46.)

### Production residual — relevant code snippets (the ground truth)

**Multi-ion shared-θ closure** (`Forward/bv_solver/boltzmann.py:91–268`, key lines):

Line 155–159 (bulk packing constant):
```python
A_dyn_bulk = sum(a * c for a, c in zip(a_dyn_floats, c0_dyn))
A_an_bulk = sum(
    float(e["a_nondim"]) * float(e["c_bulk_nondim"]) for _, e in bikerman
)
theta_b = 1.0 - A_dyn_bulk - A_an_bulk
```

Line 204–207 (free-dyn factor and floor):
```python
A_dyn_local = sum(a_dyn_funcs[i] * ci[i] for i in range(len(ci)))
free_dyn_floor = fd.Constant(1e-10)
free_dyn = fd.max_value(fd.Constant(1.0) - A_dyn_local, free_dyn_floor)
theta_b_const = fd.Constant(theta_b)
```

Line 215–242 (per-ion build + shared denominator):
```python
per_ion_q = []
for k_idx, (j, entry) in enumerate(bikerman):
    z_k = int(entry["z"])
    c_k = float(entry["c_bulk_nondim"])
    a_k = float(entry["a_nondim"])
    phi_clamp_k = float(entry["phi_clamp"])
    z_k_const = fd.Constant(float(z_k))
    c_k_const = fd.Constant(c_k)
    a_k_const = fd.Constant(a_k)
    phi_clamped_k = fd.min_value(
        fd.max_value(phi, fd.Constant(-phi_clamp_k)),
        fd.Constant(phi_clamp_k),
    )
    q_k = fd.exp(-z_k_const * phi_clamped_k)
    per_ion_q.append({...})

denom = theta_b_const + sum(p["a_const"] * p["c_const"] * p["q"]
                            for p in per_ion_q)
```

Line 252–268 (bundle: c_steric, packing_contribution, charge_density):
```python
for p in per_ion_q:
    c_steric_k = p["c_const"] * p["q"] * free_dyn / denom
    bundles.append(StericBoltzmannBundle(
        c_steric_expr=c_steric_k,
        packing_contribution=p["a_const"] * c_steric_k,
        charge_density=p["z_const"] * c_steric_k,
        z_scale=z_scale,
        ...
    ))
```

**μ_steric build** (`Forward/bv_solver/forms_logc_muh.py:435–453`):
```python
if steric_active:
    packing_floor = float(conv_cfg.get("packing_floor", 1e-8))
    A_dyn = sum(steric_a_funcs[j] * ci[j] for j in range(n))
    if water_ion_enabled and float(water_bundle.a_oh_const) != 0.0:
        A_dyn = A_dyn + water_bundle.a_oh_const * water_bundle.c_oh_expr
    if steric_boltz:
        z_scale_shared = steric_boltz[0].z_scale
        packing_total = sum(b.packing_contribution for b in steric_boltz)
        theta_inner = (
            fd.Constant(1.0) - A_dyn - z_scale_shared * packing_total
        )
    else:
        theta_inner = fd.Constant(1.0) - A_dyn
    packing = fd.max_value(theta_inner, fd.Constant(packing_floor))
    mu_steric = -fd.ln(packing)
```

**NP fluxes** (`forms_logc_muh.py:465–505`):
```python
F_res = 0
for i in range(n):
    c_i = ci[i]
    c_old = ci_prev[i]
    v = v_list[i]

    if i in mu_species:
        # mu-species flux: D*c*grad(mu) (+ steric activity).  In muh,
        # the proton's μ_H is the primary, so ∇μ_H_ideal = ∇ui[i].
        ideal_grad = fd.grad(ui[i])
    else:
        # log-c flux: D*c*(grad(u) + em*z*grad(phi))
        drift = fd.Constant(em) * z[i] * phi
        ideal_grad = fd.grad(u_exprs[i]) + fd.grad(drift)

    # ... water_ion_enabled branch (skipped, off in our stack) ...

    if steric_active:
        Jflux = D[i] * c_i * (ideal_grad + fd.grad(mu_steric))
    else:
        Jflux = D[i] * c_i * ideal_grad

    F_res += ((c_i - c_old) / dt_const) * v * dx
    F_res += fd.dot(Jflux, fd.grad(v)) * dx
```

**Reconstruction** (`forms_logc_muh.py:319–340`):
```python
def _u_expr(i: int, ui_split, phi_var):
    if i in mu_species:
        return ui_split[i] - fd.Constant(em) * z[i] * phi_var
    return ui_split[i]

u_exprs      = [_u_expr(i, ui,      phi)      for i in range(n)]
u_prev_exprs = [_u_expr(i, ui_prev, phi_prev) for i in range(n)]

ci      = [fd.exp(fd.min_value(fd.max_value(u_exprs[i],      _NEG_U_CLAMP_C), _U_CLAMP_C))
           for i in range(n)]
ci_prev = [fd.exp(fd.min_value(fd.max_value(u_prev_exprs[i], _NEG_U_CLAMP_C), _U_CLAMP_C))
           for i in range(n)]
```

**η build** (`forms_logc_muh.py:388–405`):
```python
def _build_eta_clipped(E_eq_const):
    if use_stern:
        eta_raw = phi_applied_func - phi - E_eq_const
    elif conv_cfg["use_eta_in_bv"]:
        eta_raw = phi_applied_func - E_eq_const
    else:
        eta_raw = phi - E_eq_const
    eta_scaled = bv_exp_scale * eta_raw
    if conv_cfg["clip_exponent"]:
        clip_val = fd.Constant(float(conv_cfg["exponent_clip"]))
        return fd.min_value(fd.max_value(eta_scaled, -clip_val), clip_val)
    return eta_scaled
```

For our run (use_stern=True): `eta_raw = phi_applied - phi - E_eq` (φ-dependent!).

**BV log-rate** (`forms_logc_muh.py:548–615`):
```python
if bv_log_rate:
    log_cathodic = (
        fd.ln(k0_j) + u_exprs[cat_idx]
        - alpha_j * n_e_j * eta_j
    )
    for factor in rxn.get("cathodic_conc_factors", []):
        sp_idx = factor["species"]
        power = fd.Constant(float(factor["power"]))
        c_ref_log = fd.ln(fd.Constant(
            max(float(factor["c_ref_nondim"]), 1e-12)
        ))
        log_cathodic = log_cathodic + power * (u_exprs[sp_idx] - c_ref_log)
    cathodic = fd.exp(log_cathodic)

    if rxn["reversible"] and rxn["anodic_species"] is not None:
        anod_idx = rxn["anodic_species"]
        log_anodic = (
            fd.ln(k0_j) + u_exprs[anod_idx]
            + (fd.Constant(1.0) - alpha_j) * n_e_j * eta_j
        )
        anodic = fd.exp(log_anodic)
    elif rxn["reversible"] and float(rxn["c_ref_model"]) > 1e-30:
        c_ref_j_log = fd.ln(fd.Constant(float(rxn["c_ref_model"])))
        log_anodic = (
            fd.ln(k0_j) + c_ref_j_log
            + (fd.Constant(1.0) - alpha_j) * n_e_j * eta_j
        )
        anodic = fd.exp(log_anodic)
    else:
        anodic = fd.Constant(0.0)

R_j = cathodic - anodic
bv_rate_exprs.append(R_j)

stoi = rxn["stoichiometry"]
for i in range(n):
    if stoi[i] != 0:
        F_res -= fd.Constant(float(stoi[i])) * R_j * v_list[i] * ds(electrode_marker)
```

For R4e (irreversible, c_ref_model=0): the anodic branches all evaluate to 0 ⇒ R_j = cathodic only. **But wait** — read the elif: `rxn["reversible"] and float(rxn["c_ref_model"]) > 1e-30`. For R4e `reversible=False`, so this elif is short-circuited and anodic = 0. ✓

**Poisson** (`forms_logc_muh.py:638–660`):
```python
eps_coeff = fd.Constant(float(scaling["poisson_coefficient"]))
charge_rhs = fd.Constant(float(scaling["charge_rhs_prefactor"]))
F_res += eps_coeff * fd.dot(fd.grad(phi), fd.grad(w)) * dx
if not suppress_poisson_source:
    F_res -= charge_rhs * sum(z[i] * ci[i] * w for i in range(n)) * dx
    if water_ion_enabled:
        F_res -= charge_rhs * (
            fd.Constant(-1.0) * water_bundle.c_oh_expr
        ) * w * dx
    if steric_boltz:
        z_scale_shared = steric_boltz[0].z_scale
        charge_density_total = sum(b.charge_density for b in steric_boltz)
        F_res -= (
            z_scale_shared * charge_rhs * charge_density_total * w * dx
        )
```

**Stern Robin** (`forms_logc_muh.py:666–668`):
```python
if use_stern:
    stern_coeff = fd.Constant(float(stern_capacitance_model))
    F_res -= stern_coeff * (phi_applied_func - phi) * w * ds(electrode_marker)
```

**BC attachment** (`forms_logc_muh.py:836`-ish, omitted from quote):
- φ Dirichlet at electrode: ONLY when `use_stern=False`.
- φ Dirichlet at bulk: always (`φ = 0`).
- u_i Dirichlet at bulk: always (`u_i = ln c0_i`).
- μ_H Dirichlet at bulk: always (`μ_H = ln c0_H + em·z_H·0 = ln c0_H`).
- No Dirichlet on c_i at electrode (BV-flux Neumann implicit via the `−Σ s·R·v·ds` term).
- No Dirichlet on side walls (natural zero-flux).

### Existing MMS as reference

The existing single-counterion logc MMS at `scripts/verification/mms_bv_3sp_logc_boltzmann.py` is the template to extend. Key warning in its docstring (lines 44–50):

> Important: we deliberately do NOT use `ufl.replace` to mirror the discrete F_res at U_manuf — that would make U_manuf an exact discrete solution by construction, hiding any wiring bug that's symmetric on both sides of the subtraction. By writing the source against an independent continuum statement of the PDE, any inconsistency in F_res's discrete implementation shows up as a convergence-rate violation.

This is the design intent I'm inheriting. The new derivation must avoid `ufl.replace`.

### Things I most want you to pressure-test

1. **Sign on g_S (Stern Robin source)** — my derivation has

   `g_S(x) = ε_coeff·(∇φ^ex·n)|_y=0 − C_S^model·(φ_app^model − φ^ex|_y=0)`

   The production residual at the electrode for the φ equation has TWO ds(elec) contributions: the IBP byproduct from the Laplacian `+∫ ε·∇φ·n·w·ds` (implicit) and the explicit `−∫ C_S·(φ_app − φ)·w·ds`. At u_exact the residual on ds(elec) is the sum of these two evaluated at φ_exact. Subtracting that as a source kills it. Is this correct in sign and content?

2. **The IBP decomposition for ∇·J_i**. Production residual has `+∫ J_i·∇v_i·dx`. After IBP this is `−∫ (∇·J_i)·v·dx + ∫ J_i·n·v·ds(∂Ω)`. The ds(∂Ω) splits into:
   - bulk (Dirichlet on u_i, μ_H ⇒ v_i = 0) ⇒ no contribution
   - side walls (∂_x c_i^ex = 0 by manufactured choice) ⇒ J_i^ex·n = 0 ⇒ no contribution
   - electrode: this is where the explicit BV term `−∫ Σ_j s_ij · R_j · v_i · ds(elec)` lives

   So at u_exact the electrode-ds residual is `∫ [J_i^ex·n − Σ_j s_ij · R_j^ex] · v_i · ds(elec)`. I subtract that as `g_i^elec`. Is this correct? In particular, does the J_i^ex·n term get the right sign with my outward-normal convention?

3. **muh consistency**. I define `μ_H^ex(x,y) = ln c_H^ex(x,y) + em·z_H·φ^ex(x,y)`. Then `u_H^ex,recon = μ_H^ex − em·z_H·φ^ex = ln c_H^ex` and `c_H^ex = exp(u_H^ex,recon) = c_H^ex` ✓. The NP flux for H uses `ideal_grad_H = ∇μ_H^ex` directly (because `μ_H` is the primary), which by construction equals `∇(ln c_H^ex) + em·z_H·∇φ^ex`. Is this consistent with the production residual's `fd.grad(ui[i])` for the proton, where `ui[i]` is the raw μ_H in the mixed-function split? Specifically: in production, the residual's NP flux for H is `D_H · c_H · (∇μ_H + ∇μ_steric)` with `c_H = exp(clamp(μ_H − em·z_H·φ))` reconstructed. At u_exact, `c_H = c_H^ex`, so `J_H = D_H · c_H^ex · (∇μ_H^ex + ∇μ_steric^ex)` — exactly what my source builder uses. ✓ But this means **any sign error in the production muh reconstruction** would be symmetric on both sides and the MMS misses it. The MMS catches the wrong thing for an `em·z_H` sign flip in the residual: the residual would build flux from `D_H · c_H^WRONG · ∇μ_H`, where `c_H^WRONG = exp(μ_H + em·z_H·φ)` ≠ `c_H^ex`. The discrete solver would converge to some other u that makes this internally consistent, NOT to u_exact, ⇒ convergence-rate violation. So MMS DOES catch sign flips in the reconstruction. Right?

4. **R2e/R4e log-rate construction**. Cathodic of R2e:
   ```
   log_cath_R2e = ln(k0_R2e) + u_O2^ex + 2·(u_H^ex,recon − ln c_H^ref) − α_R2e · n_e,R2e · η_R2e^ex
   ```
   This uses `u_H^ex,recon = μ_H^ex − em·z_H·φ^ex`. The power is 2 (R2e has 2 protons consumed per pass, cathodic_conc_factor power=2 in the config). The exponent on η has the minus sign for the cathodic direction. Anodic of R2e:
   ```
   log_anod_R2e = ln(k0_R2e) + u_H2O2^ex + (1 − α_R2e) · n_e,R2e · η_R2e^ex
   ```
   `R_R2e = exp(log_cath) − exp(log_anod)`. Stoichiometry contribution to F_res:
   ```
   F_res ∋ −s_{O2,R2e} · R_R2e · v_O2 · ds(elec)    with s_{O2,R2e} = −1
         + −s_{H2O2,R2e} · R_R2e · v_H2O2 · ds(elec)  with s_{H2O2,R2e} = +1
         + −s_{H+,R2e}  · R_R2e · v_H · ds(elec)     with s_{H+,R2e}  = −2
   ```
   Does this correctly express "O₂ is consumed by R2e at rate R_R2e ⇒ negative flux to surface"? In the BV convention `R_j > 0` means cathodic dominance ⇒ O₂ reduced ⇒ O₂ surface flux from the bulk should be **into** the electrode ⇒ `J_O2·n > 0` at outward-normal at electrode. The residual `−s_{O2}·R·v·ds = −(−1)·R·v·ds = +R·v·ds`, which sets `J_O2·n = +R_R2e` ⇒ inflow into electrode. Sign-consistent.

5. **Stoichiometry rows**. (s_O2, s_H2O2, s_H) = (−1, +1, −2) for R2e and (−1, 0, −4) for R4e. R4e is O₂ + 4H⁺ + 4e⁻ → 2H₂O, so H₂O₂ stoichiometric coefficient is 0 ✓. Right?

6. **Normal at y=0**. UnitSquareMesh in Firedrake has the unit square `[0,1] × [0,1]`. At y=0 the outward unit normal is `n = (0, −1)`. So `(∇φ^ex·n)|_y=0 = −∂_y φ^ex|_y=0`. In §4.4 I have `(∇φ^ex·n)|_y=0 = α₀ + (α₁ − γ)·cos(πx)`. Let me re-derive: `φ^ex = (1−y)(α₀ + α₁ cos πx) + γ·y(1−y) cos πx`. `∂_y φ^ex = −(α₀ + α₁ cos πx) + γ·(1−2y)·cos πx`. At y=0: `∂_y φ^ex|_y=0 = −(α₀ + α₁ cos πx) + γ·cos πx`. Then `−∂_y φ^ex|_y=0 = α₀ + α₁ cos πx − γ cos πx = α₀ + (α₁ − γ)·cos πx` ✓. OK that's right.

7. **bv_exp_scale and units**. I've assumed `bv_exp_scale=1.0` because `phi_applied_model` and `E_eq_j_model` are already in V_T units. Confirm from `scaling["phi_applied_model"]` — production reads `phi_applied / V_T` into this field and uses it directly in `eta_raw`. The factor 1/V_T is then absorbed.

8. **Convergence-test coverage table**. Is anything in the production residual NOT exercised by the MMS as derived? E.g. does `scaling["bv_c_ref_model_vals"]` get read anywhere I'm missing? Does the implicit `mixed_space_indices` layout (`indices.species_slice`, `indices.phi_index`) get exercised correctly?

9. **Anything else** I haven't thought of. Be aggressive about edge cases — clamp identities, free_dyn_floor at coarse mesh, water_ionization-OFF dead code paths I might have missed, packing_floor at u_exact for the muh proton reconstruction.

## Section 2: Artifact under review

```markdown
# MMS Derivation — `logc_muh` + multi-ion shared-θ Bikerman + Stern Robin + parallel R2e/R4e BV

**Date:** 2026-05-13
**Scope:** byte-faithful Method-of-Manufactured-Solutions design for the
"no-speculative-physics" production stack used by
`scripts/studies/solver_demo_slide15_no_speculative_cs.py` and exercised by
`StudyResults/solver_demo_slide15_no_speculative_cs/`.
**Status:** derivation only; no source-builder code yet.

The "no-speculative-physics" subset of production is what the demo exercises:
formulation `logc_muh`; three dynamic species (O₂, H₂O₂, H⁺) with physical
hard-sphere `a_nondim`; two analytic Bikerman counterions (Cs⁺ and SO₄²⁻)
under the multi-ion shared-θ closure; Stern Robin BC at the electrode
(`C_S = 0.20 F/m²` after the two-stage anchor); parallel
R2e + R4e Butler–Volmer in log-rate form (Ruggiero 2022); `exponent_clip = 100`
and `u_clamp = 100`. Phase-6 speculative knobs
(`enable_water_ionization`, `lambda_hydrolysis`, Singh σ-pKa) are all off.

The existing MMS (`tests/test_mms_convergence.py`,
`scripts/verification/mms_bv_3sp_logc_boltzmann.py`) covers
**logc + single ClO₄⁻ counterion + Dirichlet at electrode**. The four
genuinely new pieces here are:

1. **muh substitution** `u_H → μ_H − em·z_H·φ` everywhere `u_H` appears
   (NP flux for H, BV cathodic-conc-factors, Poisson source `z_H·c_H`).
2. **Spatially-varying η** at the electrode because Stern-on uses
   `eta_raw = phi_applied − phi − E_eq` (`forms_logc_muh.py:393–394`).
3. **Shared-θ multi-ion Bikerman closure** for both Cs⁺ and SO₄²⁻,
   contributing to Poisson and to μ_steric simultaneously.
4. **Stern Robin boundary source** `g_S` on the φ-equation at the electrode
   — not present in the existing MMS, which uses Dirichlet there.

Everything else (UnitSquareMesh sweep, `U_prev = U_manuf` with large dt,
side-wall geometry, `SRC_QUAD_DEGREE = 8`, expected L2 rate ≈ 2 / H1 rate
≈ 1 for CG1) carries over unchanged.

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

### Time-stepping eliminated in MMS

We set `U_prev = U_manuf` and `dt = 1e15`, so the discrete time term
`(c − c_old)/dt` is exactly 0 at u_exact. The φ_prev-vs-φ landmine is
neutralised because φ_prev = φ at u_exact ⇒ c_H,prev = c_H. The
**adjoint/Jacobian** sensitivity to φ_prev is not exercised by this MMS;
that's a separate test (residual-equivalence, which already exists at
`tests/test_logc_muh_formulation.py::TestTransformResidualEquivalence`).

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

These bundles live on `ctx['boltzmann_bundles']` and are consumed by both
`forms_logc_muh.py:445–449` (μ_steric build) and `:656–660` (Poisson source).

### 2.2 Steric chemical potential (`forms_logc_muh.py:435–453`)

  θ_inner(φ; c_dyn) := 1 − A_dyn(c_dyn) − z_scale · Σ_k P_k(φ; c_dyn)
  packing            := max(θ_inner, packing_floor),  packing_floor ≈ 1e−8
  μ_steric           := − ln(packing)

At MMS runtime, `z_scale = 1.0` (the runtime z-ramp is an orchestrator
knob, not part of the residual we are verifying).

### 2.3 Nernst–Planck flux (interior, `forms_logc_muh.py:465–505`)

For i ∈ {O₂, H₂O₂} (non-mu species, z_i = 0):

  ideal_grad_i := ∇u_i + em · z_i · ∇φ  =  ∇u_i           (since z_i = 0)
  J_i           := D_i · c_i · (ideal_grad_i + ∇μ_steric)

For i = H (mu species):

  ideal_grad_H := ∇μ_H                                    (∇u_H + em·z_H·∇φ = ∇μ_H)
  J_H           := D_H · c_H · (∇μ_H + ∇μ_steric)

Interior weak-form contribution (per species):

  F_res ∋  +∫_Ω  [(c_i − c_i,old)/dt] · v_i · dx  +  ∫_Ω  J_i · ∇v_i · dx

In MMS, `c_i,old = c_i^ex` at u_exact ⇒ the time term vanishes identically.

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

  log_anod_j(x,y) = ln(k0_j) + u_exprs[anod_j] + (1 − α_j) · n_e,j · η_j     (if reversible, anodic species defined)
                 OR ln(k0_j) + ln c_ref,j^model + (1 − α_j) · n_e,j · η_j     (if reversible, c_ref-anchored anodic)
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
- Electrode (y=0): no Dirichlet on c_i; production attaches BV-flux Neumann
  implicitly via the `−Σ_j s_{ij} R_j · v_i · ds(electrode)` term.

φ:
- Bulk (y=1): Dirichlet `φ = 0`.
- Electrode (y=0): **no Dirichlet** in the Stern-on branch
  (`forms_logc_muh.py:836` only attaches `bc_phi_electrode` in the
  Stern-off branch). The Stern Robin in §2.6 sets the natural BC.

Side walls (x=0, x=1): natural zero-flux for all fields (no Dirichlet, no
explicit ds term ⇒ `∇·n = 0` implicit BC).

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
  ⇒ natural zero-flux BCs auto-satisfied; **no side-wall source needed**.

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

with `dx_q = fd.dx(degree=SRC_QUAD_DEGREE)`, `SRC_QUAD_DEGREE = 8`.
Implementation builds `J_i^ex` as a UFL expression and uses
`fd.div(J_i^ex)` directly; quadrature degree ensures the non-polynomial
integrands (`ln`, `exp`, `cos`, division by D) contribute negligibly
to convergence.

The ds(∂Ω) IBP byproduct splits as:
- Bulk (y=1): w-component of `v_i` test functions is Dirichlet-zeroed
  (`u_i,μ_H` Dirichlets at bulk) ⇒ no ds contribution.
- Side walls: `J_i^ex · n` ≡ 0 by §3.2 ⇒ no ds contribution.
- Electrode (y=0): handled in §4.2.

### 4.2 NP electrode boundary source

The production residual at the electrode is

  +∫ J_i · n · v_i · ds(elec)  (IBP byproduct, **not** explicitly written
                                  but built up by Newton from the weak form)
  −∫ Σ_j s_{ij} · R_j · v_i · ds(elec)  (explicit BV term, line :615)

At u_exact, the boundary residual is

  R_i^elec,ex(v_i) = ∫_{y=0}  [ J_i^ex · n − Σ_j s_{ij} · R_j^ex ]  ·  v_i  ·  ds

Subtract that integrand:

  g_i^elec(x)  :=  J_i^ex · n |_{y=0}  −  Σ_j  s_{ij} · R_j^ex |_{y=0}
  F_res ←  F_res  −  ∫_{y=0}  g_i^elec · v_i · ds_q(electrode_marker)

with outward normal `n = (0, −1)` at y = 0:

  J_i^ex · n |_{y=0}  =  −  J_i^ex,y(x, 0)
                       =  −  [ −D_i · c_i^ex · ∂_y(u_i^ex + μ_steric^ex) ]|_{y=0}      (i ∈ {O₂, H₂O₂})
                       =  +  D_i · c_i^ex|_{y=0} · ∂_y(u_i^ex + μ_steric^ex)|_{y=0}

  J_H^ex · n |_{y=0}  =  +  D_H · c_H^ex|_{y=0} · ∂_y(μ_H^ex + μ_steric^ex)|_{y=0}

(Implementation uses `fd.dot(J_i^ex, fd.FacetNormal(mesh))` and lets
Firedrake handle the sign automatically; the manual expression above is
for derivation transparency.)

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
`E_eq_R4e^model ≈ 47.86`), with `bv_exp_scale = 1.0`. (We will read all
three off `scaling` at MMS runtime rather than hard-coding.)

### 5.1 BV `exponent_clip = 100`

Choose envelope: `|α₀|, |α₁|, |γ| ≤ 1`. Then |φ^ex| ≤ 2.

  eta_raw_R2e^ex(x,0) = 21.41 − [α₀ + α₁ cos(πx)] − 27.04   ∈   [−30.4, −5.6]
  eta_raw_R4e^ex(x,0) = 21.41 − [α₀ + α₁ cos(πx)] − 47.86   ∈   [−51.3, −26.5]

Both stay inside ±100 ⇒ **clip identity at u_exact**. ✓

### 5.2 `u_clamp = 100`

  |u_O2^ex|, |u_H2O2^ex|, |ln c_H^ex| are O(|ln c0_i| + 1) ≪ 100.   ✓

### 5.3 Counterion `phi_clamp_k = 50`

  |z_k · φ^ex| ≤ 2 · |φ^ex| ≤ 4 ≪ 50  (z_SO4 = −2 dominates).   ✓

⇒ q_k^ex = exp(−z_k · φ^ex) is the unclamped expression at u_exact.

### 5.4 `packing_floor = 1e−8` and `free_dyn_floor = 1e−10`

At the demo's physical hard-sphere `a_i^dyn` (1.5e−5, 2.4e−5, 6.6e−5) and
`c0_dyn` (O(1) in nondim), `A_dyn^ex ≪ 1e−3` ⇒ `free_dyn = 1 − A_dyn` is
O(1). At the demo's counterion `a_k^ctn` (Marcus radii Cs⁺ ≈ 3.3 Å,
SO₄²⁻ ≈ 3.0 Å) and `c_k^bulk` (Cs⁺ 0.2 M, SO₄²⁻ 0.1 M after `C_SCALE`),
`A_an,bulk ≈ O(10⁻³)` ⇒ θ_b ≈ 1 − O(10⁻³) > 0.99.

  D^ex(x,y) = θ_b + Σ_k a_k^ctn · c_k^bulk · q_k^ex(x,y)

is positive and O(1) over our envelope (max |z_k φ| ≈ 4 ⇒ max exp ≈ 55).

  c_k^ster,ex ≤ c_k^bulk · 55 · 1 / 0.99 ≈ 55 · c_k^bulk

  P_k^ex ≤ a_k^ctn · 55 · c_k^bulk ≈ 55 · A_an,bulk_k ≈ O(10⁻²)

  θ_inner^ex = 1 − A_dyn^ex − Σ_k P_k^ex ≈ 1 − O(10⁻²) > 0.95

⇒ all three floors (`packing_floor`, `free_dyn_floor`, ion `phi_clamp`)
are **identity at u_exact**. The discrete operator still enforces them,
which is what we want: any clamp/floor-induced model error shows up as a
convergence-rate degradation.

### 5.5 Parameter envelope (recommended)

  δ_O2 = δ_H2O2 = δ_H = 0.30       (matches existing MMS pattern)
  α₀ = 0.5,  α₁ = 0.5,  γ = 0.5    ⇒ |φ^ex| ≤ 1.25  (safely inside §5.1–4)
  V_RHE = +0.55 V                  ⇒ η_R2e ≈ −5.6, η_R4e ≈ −26.5

Optional Stern-off cross-check: C_S^model → None (Dirichlet at electrode).
With Stern off, `eta_raw_j = φ_app^model − E_eq_j^model` is **spatially
constant** on the electrode (different branch in `_build_eta_clipped`),
and the φ Dirichlet BC at y=0 must be set to `phi_applied_func` (which
won't match φ^ex(x,0) unless we also override the manufactured φ shape).
For the convergence test, this is a separate run with an alternate
manufactured φ that equals φ_app at y=0; the source builder skips g_S
and the η_j build follows the `eta_raw = φ_app − E_eq` branch.

---

## 6. Convergence-test coverage by source-term group

| Source piece | Catches a bug in |
|---|---|
| `−∇·J_i` for i ∈ {O₂, H₂O₂} | μ_steric inclusion in NP flux (`:499`); shared-θ contributions of Cs⁺/SO₄²⁻ to packing (`:447–449`); D_i and c_i^ex scaling |
| `−∇·J_H` | muh reconstruction in mass-balance (∇u_H replaced by ∇μ_H, `:473`); μ_steric coupling for proton |
| `R_R2e, R_R4e` via `u_exprs[h]` | muh substitution in BV (`:551–563`); cathodic_conc_factors powers (2 vs 4); α·n_e signs; reversible-vs-irreversible branch; ln(k0_j) sign |
| Spatial η^ex(x,0) | Stern-on branch of `_build_eta_clipped` (`:393–394`); φ-coupling into BV that **only** happens when use_stern=True |
| Poisson interior `z_H·c_H + Σ_k z_k c_k^ster` | shared-θ closure numerator `(1 − A_dyn)` (`:254`); per-ion clamp inside D(φ) (`:224–227`); sign on `z_scale·charge_rhs` (`:658–660`) |
| `g_S` (Stern Robin) | C_S^model nondim conversion (`:238–254`); sign on `(φ_app − φ)` (`:668`); coupling to phi_applied_func continuation in `set_stern_capacitance_model` |

The MMS does **not** catch bugs in:
- IC machinery (`debye_boltzmann` composite-ψ + multispecies-γ in `picard_ic.py`).
- Anchor / continuation logic (`solve_anchor_with_continuation`,
  `solve_grid_with_anchor`).
- Warm-start floor, two-stage Stern bump, K0 continuation ladder.
- Clip *behaviour* (clips inactive by design here).
- Phase-6 speculative physics (already off in this study by definition).

---

## 7. Implementation notes for the source builder

(Out of scope for this derivation; deferred to a follow-up plan / PR.)

Key reusables from `scripts/verification/mms_bv_3sp_logc_boltzmann.py`:
- `make_sp_production` — adapt to muh + multi-ion + Stern factory call.
- `compute_rates`, `_ufl_l2_error`, `_ufl_h1_error`, `plot_convergence`,
  `run_mms` outer-loop — keep verbatim.
- `_build_manufactured_source` — rewrite for the four new pieces.

New helpers needed:
- `_build_multi_ion_shared_theta_source(phi_exact, counterions_cfg, a_dyn, c_dyn_ex)`:
  composes `q_k^ex, D^ex, c_k^ster,ex, ρ_k^ex, P_k^ex` as UFL expressions.
- Stern Robin source builder: takes `phi_exact, phi_app_model, C_S_model,
  eps_coeff, n_vec` and returns `g_S`.
- muh-aware BV source builder: takes the reaction-config list and
  substitutes `μ_H^ex − em·z_H·φ^ex` for `u_H^ex` everywhere.

Test file: `tests/test_mms_logc_muh_multi_ion_stern.py`, mirroring
`tests/test_mms_convergence.py` layout. Classes:
- `TestMMSConvergence`: UnitSquareMesh sweep N ∈ {8, 16, 32, 64};
  assert L2 slope ≥ 1.8 and H1 slope ≥ 0.8 per primary unknown
  (`u_O2, u_H2O2, μ_H, φ`), R² > 0.99.
- `TestMMSProductionGradedMesh`: single-solve recovery on the demo mesh
  (Nx=8, Ny=80, β=3); thresholds sized ~6× above baseline.
- Optional `TestSternOffSanity`: parametrized Stern-off run as
  cross-check (see §5.5).

---

## 8. What's deliberately deferred

- **Time-dependent verification**: this MMS verifies the steady-state
  spatial operator. Time-stepping is exercised by setting U_prev=U_manuf
  and dt large, so `(c − c_old)/dt = 0` identically. A separate
  time-MMS would manufacture a c(x,y,t) and test temporal accuracy
  — orthogonal scope.
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
  Testing the floor branch is a separate MMS design problem;
  the existing single-counterion attempt is xfail in
  `tests/test_mms_steric_boltzmann_convergence.py` for this reason.
```

## Section 3: Critique prompt

You are an adversarial reviewer. Be critical. Be argumentative.
Find every hole: missing steps, wrong algebra, untested assumptions,
edge cases not addressed, implicit dependencies, claims without
evidence, off-by-one errors, sign errors, dimensional errors. Don't
be polite — if something is wrong, say so. Concision over hedging.

For each issue, state:
  - WHAT is wrong (specific, not vague — name the line or symbol)
  - WHY it matters (what breaks downstream if uncorrected)
  - WHAT to do (concrete fix, or what evidence would close the gap)

Number your issues. After all issues, end your response with exactly
one of these lines, no other text after it:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN

Use APPROVED only when there are no issues you would block on.
Minor nitpicks alone do not justify ISSUES_REMAIN — call them out
but still verdict APPROVED. Use ISSUES_REMAIN whenever any of your
issues are genuinely blocking.
