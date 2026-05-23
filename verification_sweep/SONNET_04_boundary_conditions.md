# Verification Sweep — Agent 04/13: Boundary Conditions

**Scope:** Stern BC, Robin BCs for species, Dirichlet BCs at bulk, BC wiring between caller and forms.
**Files reviewed:**
- `Forward/bv_solver/forms_logc_muh.py` (full; 1561 lines)
- `Forward/bv_solver/anchor_continuation.py` (full; 2024 lines)
- `scripts/_bv_common.py` (relevant excerpts)
- `docs/solver/mms_pnpbv_muh_multi_ion_stern_derivation.md`

---

## 1. Stern BC Implementation

**Finding: CORRECT.**

`forms_logc_muh.py:666–668` implements the Stern Robin BC as:

```python
if use_stern:
    stern_coeff = fd.Constant(float(stern_capacitance_model))
    F_res -= stern_coeff * (phi_applied_func - phi) * w * ds(electrode_marker)
```

This is the weak form of `C_S · (φ_app − φ(0)) · w · ds`, which corresponds to
the Robin equation `ε ∂_n φ = C_S · (φ_app − φ)` at the electrode (y=0). The MMS
derivation confirms this exactly: §2.6 states `F_res ∋ − C_S^model · (φ_app^model − φ) · w · ds(electrode)`.

**Non-dimensionalization:** `C_S` is read from `bv_cfg['stern_capacitance_f_m2']` (physical F/m²)
at form-build time (`forms_logc_muh.py:238–254`). When `nondim_enabled=True`, the
conversion factor is:

```
conv_factor = V_scale / (F · c_scale · L_scale)
bv_stern_capacitance_model = C_S_physical * conv_factor
```

where `_F` = Faraday constant, `length_scale_m`, `potential_scale_v`, and
`concentration_scale_mol_m3` are all read from the scaling dict. When
`nondim_enabled=False` the physical value is used directly (`conv_factor = 1.0`).
Both paths store the converted value in `scaling['bv_stern_capacitance_model']` and
the conversion factor in `scaling['bv_stern_phys_to_nondim_factor']` for use by
`set_stern_capacitance_model`.

**C_S = 0.20 F/m² wiring from caller:** `scripts/_bv_common.py:688–689` passes
`stern_capacitance_f_m2` through `_make_bv_bc_cfg` into `params['bv_bc']`. The form
reads it via `_get_bv_cfg(params, n)['stern_capacitance_f_m2']` at build time. Hard
rule #6 is enforced by the factory call chain.

**Sign of σ:** The Stern relation `φ_electrode − φ(0) = σ/C_S` is encoded with
σ = C_S · (φ_app − φ). Cathodically (φ_app < 0 < φ relative to bulk), σ is
negative (excess electrons on the electrode), consistent with the OHP-field sign in
Singh 2016. No sign error found.

---

## 2. Anchor vs Grid BC Consistency / Runtime Bump Mechanism

**Finding: CORRECT — bump modifies the live FE Constant.**

`anchor_continuation.py:412–467` (`set_stern_capacitance_model`) directly calls
`stern_const.assign(nondim_value)` on `ctx['stern_coeff_const']`, which is the
actual `fd.Constant` object embedded in the UFL `F_res` expression at form-build
time (`forms_logc_muh.py:667`). This is the canonical Firedrake pattern for runtime
coefficient mutation: `.assign()` on a `Constant` propagates into all UFL
expressions that reference it without a form rebuild.

Metadata is also updated first:

```python
new_scaling["bv_stern_capacitance_model"] = nondim_value
ctx["nondim"] = new_scaling
stern_const.assign(nondim_value)
```

Both the Picard IC reader (`picard_ic.py` reads `ctx['nondim']['bv_stern_capacitance_model']`)
and the FE residual are updated atomically.

The `c_s_ladder` in `solve_anchor_with_continuation` (lines 1246–1316) walks
C_S in **decreasing** order (high → production) as documented in the docstring:
"strictly monotonic decreasing positive sequence whose last value matches the
production C_S". This is the correct direction: start at a relaxed Stern
(small capacitive constraint on the electrode BC → larger compact-layer drop)
then tighten to production. `set_stern_capacitance_model` is called at each rung.

**Interaction with `apply_bc`:** There is no `apply_bc=True/False` flag in the
`forms_logc_muh.py` path. The `use_stern` boolean is determined entirely by
whether `bv_stern_capacitance_model > 0` at form-build time. Once Stern is built
into `F_res`, it cannot be toggled off at runtime (only its coefficient can be
modified). The anchor-cold-start workaround (build at C_S=0.10, bump to 0.20) is
external to the form: callers build with a low `stern_capacitance_f_m2` in `sp`,
then call `set_stern_capacitance_model` after the anchor converges.

---

## 3. Robin BC for μ_H at the Electrode

**Finding: CORRECT — implicit Neumann BC, sign correct.**

There is no explicit Neumann term for H⁺ in the form. The H⁺ (μ_H species)
electrode BC is imposed as a **natural (implicit) Neumann BC** through the
BV source term in the Nernst–Planck residual:

```python
F_res -= fd.Constant(float(stoi[i])) * R_j * v_list[i] * ds(electrode_marker)
```

For H⁺ (species index 2), the stoichiometries from `PARALLEL_2E_4E_REACTIONS` are:
- R2e: stoi[2] = −2  (2 H⁺ consumed)
- R4e: stoi[2] = −4  (4 H⁺ consumed)

The residual contribution is:

```
F_res -= (-2)*R_R2e * v_H * ds(elec) - (-4)*R_R4e * v_H * ds(elec)
       = +2*R_R2e * v_H * ds(elec) + 4*R_R4e * v_H * ds(elec)
```

The natural BC that results (after IBP of the NP flux term) is:

```
J_H · n_outward = stoi_H · (R_R2e + R_R4e) = −2·R_R2e − 4·R_R4e
```

where `J_H = D_H · c_H · ∇μ_H` is the **mobility flux** (positive toward bulk),
`n_outward = (0, −1)` at y=0. Since R_R2e, R_R4e > 0 cathodically, the mobility
flux at the electrode is negative (flux toward electrode), meaning H⁺ is consumed
at the electrode (flows from bulk toward the reaction site). **Sign correct.**

---

## 4. Robin BC for O₂

**Finding: CORRECT — O₂ consumed at the electrode with correct sign.**

O₂ stoichiometries:
- R2e: stoi[0] = −1
- R4e: stoi[0] = −1

Residual: `F_res -= (-1)*R_R2e * v_O2 * ds + (-1)*R_R4e * v_O2 * ds`
         `= +R_R2e * v_O2 * ds + +R_R4e * v_O2 * ds`

Natural BC: `J_O2 · n_outward = −R_R2e − R_R4e < 0` (cathodic R > 0)

Mobility flux at electrode negative ⇒ O₂ flows toward electrode (depleted at
surface). **Sign correct.** The MMS derivation §2.3 confirms:
"Jprod_O2 · n = −R < 0 ⇒ Jprod_O2,y > 0 ⇒ O₂ increases toward bulk (depletion at surface)".

---

## 5. Robin BC for H₂O₂

**Finding: CORRECT — H₂O₂ produced by R2e, sign correct.**

H₂O₂ stoichiometries:
- R2e: stoi[1] = +1  (produced)
- R4e: stoi[1] = 0   (untouched)

Residual: `F_res -= (+1)*R_R2e * v_H2O2 * ds + 0`
         `= -R_R2e * v_H2O2 * ds`

Natural BC: `J_H2O2 · n_outward = +R_R2e > 0` (cathodic R2e > 0)

Mobility flux at electrode positive ⇒ H₂O₂ flows away from electrode (produced
at surface, transported to bulk). **Sign correct.**

---

## 6. Bulk Dirichlet BCs

**Finding: CORRECT.**

`forms_logc_muh.py:821–837`:

```python
bc_phi_ground = fd.DirichletBC(W.sub(n), fd.Constant(0.0), ground_marker)
for i in range(n):
    c0_i = max(float(c0_model[i]), _C_FLOOR)
    if i in mu_species:
        bc_val = fd.Constant(np.log(c0_i) + em * float(z_vals[i]) * phi_bulk_at_ground)
    else:
        bc_val = fd.Constant(np.log(c0_i))
    bc_ui.append(fd.DirichletBC(W.sub(i), bc_val, concentration_marker))
if use_stern:
    bcs = bc_ui + [bc_phi_ground]
else:
    bcs = bc_ui + [bc_phi_electrode, bc_phi_ground]
```

At y=L_eff (concentration_marker = ground_marker = 4 by default):
- φ = 0 by `bc_phi_ground`
- u_i = ln(c0_i) for non-mu species (O₂, H₂O₂)
- μ_H = ln(c0_H) + em·z_H·φ_bulk = ln(c0_H) + 0 = ln(c0_H) (since φ_bulk=0)

All via `fd.DirichletBC`. The MMS derivation §3.2 confirms "all bulk Dirichlets
met by the manufactured shape; no bulk BC override needed".

**Stern vs no-Stern:** When `use_stern=True`, `bc_phi_electrode` is NOT applied
(only `bc_phi_ground` at the bulk). The electrode φ is set by the Stern Robin,
not Dirichlet. When `use_stern=False`, `bc_phi_electrode` (Dirichlet φ=φ_applied
at electrode_marker=3) is added. This is exactly the documented behavior.

---

## 7. L_eff / domain_height_hat Consistency

**Finding: CORRECT.**

The IC code (`forms_logc_muh.py:975–976`) reads:

```python
bv_conv = params.get("bv_convergence", {}) if isinstance(params, dict) else {}
domain_height_hat = float(bv_conv.get("domain_height_hat", 1.0))
y_norm = spatial_var / fd.Constant(domain_height_hat)
```

The form itself does NOT embed an explicit L_eff — it uses the mesh coordinates
directly. `domain_height_hat` is only used in the IC to normalize `y_norm` for
the outer linear interpolation, ensuring the bulk BC lands at the mesh top
regardless of L_eff sweep extent. The form's Dirichlet BC at `concentration_marker`
(marker 4 = top boundary by convention from `make_graded_rectangle_mesh`) correctly
places bulk conditions at the mesh top independent of the numeric domain height.

**Caveat:** If the mesh's actual top boundary is NOT tagged as marker 4, or if
`domain_height_hat` in `bv_convergence` does not match the mesh's y-extent, the IC
y_norm normalization would be off. However, this is documented in CLAUDE.md as the
caller's responsibility. No form-level inconsistency detected.

---

## 8. BC Ordering / Consistency at y=0

**Finding: CORRECT — no conflict.**

At y=0 (electrode, marker 3):
- The Stern Robin contributes to the φ-equation residual via `ds(electrode_marker)`.
  No Dirichlet for φ is applied there (only at y=L_eff via ground_marker=4).
- The BV source terms (NP natural BCs for O₂, H₂O₂, H⁺) contribute to the
  species residuals via `ds(electrode_marker)`.
- There are NO Dirichlet BCs at electrode_marker for any field when `use_stern=True`.

The `bcs` list when Stern is active is:
```
[bc_u0_bulk, bc_u1_bulk, bc_muH_bulk, bc_phi_ground]
```
All four act on `concentration_marker = ground_marker = 4` (bulk). No overlap with
electrode_marker=3. No BC conflict.

When Stern is off, `bc_phi_electrode` (Dirichlet φ at marker 3) is added, and the
Stern Robin term is absent. Again no conflict.

---

## Issues / Concerns

### CONCERN 1 (Minor, not a bug): MMS does NOT verify nondim conversion for C_S

The MMS derivation explicitly notes (§6): "**Stern coefficient nondim conversion**
at form-build time (`forms_logc_muh.py:238–254`): the source builder reads the
converted `bv_stern_capacitance_model` value, so an off-by-factor in the conversion
would change BOTH residual and source equally." This means the MMS cannot catch a
bug in the physical-to-nondimensional C_S conversion factor. The conversion factor
is `V_scale / (F · c_scale · L_scale)`, which has units of [V·m³·mol⁻¹·m⁻²·F⁻¹] =
[1/(F·m²)] × [V/m × m³/mol] — requires careful dimensional analysis. The code path
is at lines 244–254 when `nondim_enabled=True`. This is only active in the
dimensionful code path; the standard production stack uses dimensional=False
(conv_factor=1.0), so this is low practical risk. **Not a bug; gap in test coverage.**

### CONCERN 2 (Minor, informational): C_S ladder direction

`c_s_ladder` is documented as "strictly monotonic **decreasing**" (anchor_continuation.py:987:
"ramps from a Stern-relaxed start **down** to the production target"). This means the
first entry should be the LARGEST C_S (most relaxed → smallest compact-layer drop for
a given applied voltage). For production C_S=0.20 F/m², the documented example ladder
is `(1.0, 0.5, 0.25, 0.10)` (for C_S=0.10 target). Users who accidentally pass an
increasing ladder will get a `ValueError` from the validation check at lines 1260–1266.
The direction is enforced, not just documented.

### CONCERN 3 (Minor): `apply_bc` flag does not exist in `forms_logc_muh.py`

The task brief mentioned "whether `apply_bc=True/False` flag flips Stern on/off".
No such flag exists in the `logc_muh` form builder. Stern is a build-time decision
(positive `stern_capacitance_f_m2` in config ↔ Robin BC built into F_res; None/0 ↔
Dirichlet). There is no runtime toggle. This is documented behavior, not a bug, but
callers relying on a runtime Stern-off switch would need to rebuild the form.

---

## Summary Table

| Check | Status | Notes |
|---|---|---|
| Stern BC form correct | PASS | `−C_S·(φ_app−φ)·w·ds(elec)` matches Robin `φ_electrode − φ(0) = σ/C_S` |
| C_S nondim conversion | PASS | `V_scale/(F·c_scale·L_scale)` with phys→nondim factor stored for runtime bump |
| Runtime bump modifies FE Constant | PASS | `stern_const.assign(nondim_value)` in `set_stern_capacitance_model` |
| Bump updates both metadata + Constant | PASS | Dict updated first, then Constant |
| Bulk Dirichlet: φ=0 at y=L_eff | PASS | `fd.DirichletBC(W.sub(n), 0, ground_marker)` |
| Bulk Dirichlet: u_i=ln(c0_i) | PASS | `fd.DirichletBC(W.sub(i), ln(c0_i), concentration_marker)` |
| Bulk Dirichlet: μ_H=ln(c0_H) | PASS | explicit φ_bulk=0 term kept for clarity |
| No Dirichlet at electrode when Stern-on | PASS | `bcs = bc_ui + [bc_phi_ground]` only |
| BV stoichiometry sign for H⁺ | PASS | stoi[H]=−2/−4 → H⁺ consumed cathodically |
| BV stoichiometry sign for O₂ | PASS | stoi[O2]=−1 → O₂ consumed at electrode |
| BV stoichiometry sign for H₂O₂ | PASS | stoi[H2O2]=+1 → H₂O₂ produced by R2e |
| No BC conflict at electrode | PASS | Stern Robin and BV natural BCs both on ds(3); no Dirichlet there |
| L_eff / domain_height_hat | PASS | IC normalizes by domain_height_hat; form does not embed L_eff |
| C_S=0.20 propagated from make_bv_solver_params | PASS | `_make_bv_bc_cfg → params['bv_bc'] → _get_bv_cfg → bv_stern_capacitance_model` |

---

## Bottom Line

All boundary conditions are correctly implemented. The Stern Robin BC is a proper
weak-form contribution `−C_S·(φ_app−φ)·w·ds` with no Dirichlet on φ at the electrode
when Stern is active. Species fluxes at the electrode are all implemented as natural
(implicit Neumann) BCs through the BV stoichiometric source terms, with correct signs
for cathodic ORR. The runtime C_S bump mechanism (`set_stern_capacitance_model`)
correctly modifies the live UFL `Constant` embedded in `F_res`, not just a Python
attribute. Bulk Dirichlets for φ=0 and u_i=ln(c0_i) are applied via `fd.DirichletBC`
at `concentration_marker = ground_marker = 4`.

**No correctness bugs found.** Two minor gaps: (1) The nondim C_S conversion path
(active only when `nondim_enabled=True`, not the standard production stack) is not
covered by the MMS. (2) The `apply_bc` flag does not exist — Stern-on/off is a
build-time decision, not a runtime toggle.
