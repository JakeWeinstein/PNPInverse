# SONNET_02: Butler-Volmer Reactions + Topology

## Bottom line

FLAG-MINOR — All production BV mechanics (reactions list, log-rate formulation, eta-clip, u_clamp, stoichiometry, current assembly) are correctly implemented, but `clipping_conventions.md` contains stale threshold arithmetic keyed to the legacy sequential `E_eq_R2=1.78 V` rather than the production parallel `E_eq_R2e=0.695 V`, and the `_bv_common.py` comment on hard rule 4 still cites the legacy values; neither is a runtime bug but both will mislead future readers.

---

## Verified claims

1. **Reactions list — PASS.** `PARALLEL_2E_4E_REACTIONS` in `scripts/_bv_common.py:882–911` contains exactly two reactions: R2e (`E_eq_v=0.695 V`, `n_electrons=2`, `reversible=True`, cathodic O₂ idx 0, anodic H₂O₂ idx 1, stoich `[-1,+1,-2]`, H⁺ factor `power=2`) and R4e (`E_eq_v=1.23 V`, `n_electrons=4`, `reversible=False`, cathodic O₂ idx 0, anodic None, stoich `[-1,0,-4]`, H⁺ factor `power=4`). All values match Ruggiero 2022 §1.

2. **Log-rate BV formulation — PASS.** `forms_logc_muh.py:548–582` implements cathodic as `exp(ln(k0) + u_cat + Σ power*(u_sp − ln c_ref) − alpha*n_e*eta_j)` and anodic as `exp(ln(k0) + u_anod + (1−alpha)*n_e*eta_j)`. Signs are correct. The `η_scaled` clip is applied inside `_build_eta_clipped` (lines 388–403) **before** `alpha*n_e` multiplication, satisfying hard rule 2.

3. **`_build_eta_clipped` — PASS.** `forms_logc_muh.py:388–403`: constructs `eta_scaled = bv_exp_scale * eta_raw`, then `fd.min_value(fd.max_value(eta_scaled, -clip_val), clip_val)`. With `exponent_clip=100` (config default `config.py:138`), both R2e and R4e are fully unclipped everywhere on the production grid `V_RHE ∈ [−0.5, +1.0]` V. Computed thresholds: R2e cathodic unclip at V = 0.695 − 100·V_T ≈ −1.87 V (well below −0.5 V floor); R4e at ≈ −1.34 V. At the convergence floor V = −0.5 V, η_scaled for R2e ≈ −46.5, unclipped. At V = +1.0 V, η_scaled for R4e ≈ −8.9, unclipped.

4. **`u_clamp=100` — PASS.** `forms_logc_muh.py:326–340`: `_U_CLAMP` reads `conv_cfg["u_clamp"]` (default 30.0 in form code, but `config.py:138,145` sets default to 100.0 and `_bv_common.py:455` passes `u_clamp=100.0`). Applied to `u_exprs[i]` (which for the muh proton species is the reconstructed `log(c_H) = mu_H − em·z_H·phi`, NOT raw mu_H — correct per module docstring lines 33–35). Applied once in the bulk `ci` reconstruction; not applied again in the BV boundary residual (log-rate uses `u_exprs[cat_idx]` directly — no double-clamping).

5. **Stoichiometry signs in species source — PASS.** `forms_logc_muh.py:612–615`: `F_res -= Constant(stoi[i]) * R_j * v_list[i] * ds(electrode_marker)`. For R2e: O₂ stoi = −1 (consumed, sink on residual), H₂O₂ stoi = +1 (produced, source), H⁺ stoi = −2 (consumed). For R4e: O₂ stoi = −1 (consumed), H₂O₂ stoi = 0 (untouched), H⁺ stoi = −4 (consumed). No H₂O₂ reduction term is present in the two-reaction parallel list (correct: R4e goes to water without releasing free peroxide per Ruggiero §1).

6. **Electrode current assembly — PASS.** `observables.py:106–121`: total current is `Σ_j (n_e_j / N_ELECTRONS_REF) * R_j * ds` where `N_ELECTRONS_REF=2`. For parallel 2e+4e: R2e contributes weight 1.0, R4e contributes weight 2.0 — correctly accounts for 4e channel consuming twice the electron count per reaction event.

7. **Hard rule 4 — production code clean of legacy sequential values.** `forms_logc_muh.py` contains no references to `E_EQ_R1_V=0.68`, `E_EQ_R2_V=1.78`, `k0_hat_r1`, or `k0_hat_r2` in the `use_reactions` branch. The legacy per-species fallback branch at line 617 is only reached when `bv_reactions` is absent from `bv_bc`.

8. **`bv_reactions` takes precedence over legacy bundle — PASS.** `_make_bv_bc_cfg` (`_bv_common.py:619–635`): when `bv_reactions is not None`, `reactions_out` is built from that list; the legacy `reaction_1`/`reaction_2` construction (which would use `E_eq_r1=0.68`, `E_eq_r2=1.78`) is bypassed entirely. Per `_bv_common.py:606–616` docstring: "When set, takes precedence over the legacy `k0_hat_r{1,2}, alpha_r{1,2}, E_eq_r{1,2}` keyword bundle". The concern from CLAUDE.md about `k0_hat_r1`/`k0_hat_r2` silently overriding is addressed by the precedence guard.

---

## Discrepancies / issues

### MINOR — Stale unclip-threshold arithmetic in `clipping_conventions.md` — `docs/solver/clipping_conventions.md:85–89,348`

The doc states "R2 unclips at V_RHE > +0.495 V" (clip=50) and "V_RHE > −0.789 V" (clip=100) using `E_eq_2 = 1.78 V`. This was accurate for the **legacy sequential topology** where R2 had `E_eq=1.78 V`, but the production parallel stack uses `E_eq_R2e = 0.695 V`. At clip=100, the true production R2e unclip threshold is `0.695 − 100·V_T ≈ −1.874 V` — well below the entire production grid, confirming R2e is always unclipped at clip=100. The table at line 348 also cites `E_eq=1.78` for "R2". The narrative conclusion (production grid is unclipped at clip=100) is still correct, but the arithmetic derivation is wrong. Any reader using the threshold formula for the production parallel reactions will get the wrong numbers.

**Runtime impact:** None — the clip calculation in `_build_eta_clipped` uses the per-reaction `E_eq_j` directly from the reactions list; it does not read this doc.

**Fix:** Update the threshold table and "R2 unclips" narrative to cite R2e (E°=0.695 V) and R4e (E°=1.23 V) with clip=100 thresholds of −1.87 V and −1.34 V respectively.

### MINOR — `_bv_common.py:118` comment on Hard Rule 4 still cites legacy values — `scripts/_bv_common.py:118`

The comment reads: `# "Use physical E_eq (R1 = 0.68 V, R2 = 1.78 V vs RHE), never E_eq = 0."` followed immediately by `E_EQ_R1_V = 0.68` and `E_EQ_R2_V = 1.78`. This citation of CLAUDE.md hard rule 4 is for the sequential topology. The parallel production constants (`E_EQ_R2E_V=0.695`, `E_EQ_R4E_V=1.23`) are defined right below (lines 123–124) and marked as the correct Ruggiero 2022 values. The legacy constants are only used by `make_bv_solver_params` when `bv_reactions=None` (legacy fallback path). The comment should distinguish that the cited values are the legacy sequential-topology defaults, not the production parallel values.

**Runtime impact:** None — `PARALLEL_2E_4E_REACTIONS` uses `E_EQ_R2E_V` and `E_EQ_R4E_V` directly.

### MINOR (known) — Legacy `u_clamp` default in form code vs factory default — `forms_logc_muh.py:326`

`forms_logc_muh.py:326` reads `conv_cfg.get("u_clamp", 30.0)` — the local fallback default is 30, not 100. The config layer (`config.py:145,166`) and factory (`_bv_common.py:455`) both default to 100, so any correctly-built `conv_cfg` will carry 100. A manually-constructed `conv_cfg` that omits `u_clamp` would silently get 30 instead of 100. This matches the pattern in `forms_logc.py` and is a known legacy issue (documented in `clipping_conventions.md` §2), not introduced here.

---

## Open / unverified

- **`n_e` in the BV log-rate is part of the exponent, not a separate current multiplier** — confirmed at `forms_logc_muh.py:555` (`alpha_j * n_e_j * eta_j`). The current-density observable then re-weights by `n_e_j / N_ELECTRONS_REF` in `observables.py:115`. There is a potential double-counting concern: the stoichiometry loop at line 612 applies the per-species source term directly from `R_j` (mole-flux units), while the observable uses `n_e_j * R_j` (charge units). This split is physically correct only if `R_j` is understood as a mole-flux rate (mol/m²/s nondim) and the observable separately converts to current. This author believes the convention is consistent (the stoichiometry source is correct as-is; the observable applies the Faraday factor via `I_SCALE * n_e/2` externally), but this cross-check between stoichiometry source and observable current was not fully traced through `I_SCALE` in this review. A separate observer-vs-source consistency check is warranted.

- **Picard IC R2e/R4e validation.** `picard_ic.py:1227–1229` validates that `bv_reactions[0].n_electrons==2` and `bv_reactions[1].n_electrons==4`. Correct. Not traced further in this review.

- **`alpha` range validation** (`config.py:518–519`): validates `0 < alpha <= 1`. Both ALPHA_R2E=0.627 and ALPHA_R4E=0.5 pass. ALPHA_R4E is flagged as a placeholder in comments — no issue for correctness review.
