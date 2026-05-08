# Round 5 — Counterreply (final under round cap)

## Section 1 — Per-issue acknowledgment

### Issue 1 — Confused absolute φ vs diffuse-layer ψ

**Accept fully.** You're right. The H⁺ Boltzmann shift is *relative
to the outer potential*. Existing code at `picard_ic.py:459`:

```python
log_H_rxn = log(max(H_o, 1e-300)) - psi_D + log_gamma
```

uses `psi_D` (the diffuse-layer drop), NOT absolute `phi`.

For the spatial IC the matched-asymptotic decomposition is:

```text
phi(y) = phi_outer(y) + psi(y)              (sum, not separate)

where:
  phi_outer(y)  = bulk-region absolute potential, slowly varying;
                  at the OHP-side edge of the outer region this is
                  the Picard's phi_o = log(H_o/c_clo4_bulk)
                  (legacy 1:1) or _solve_outer_phi_multiion(...)
                  (new multi-ion).
  psi(y)        = diffuse-layer drop, fast variation across λ_D;
                  ψ(0) = ψ_D at the OHP, ψ(y → bulk) = 0.

Boltzmann shift uses ψ:
  log_c_i_seed(y) = log(c_i_outer(y)) - z_i · psi(y) + log_gamma_psi(y)

Poisson primary variable / mu_H uses absolute φ:
  phi_init(y)  = phi_outer(y) + psi(y)       (FE Function on phi space)
  mu_H_init(y) = u_H_init(y) + em·z_H·phi_init(y)
```

For `logc_muh` formulation specifically, `mu_H = u_H + em·z_H·phi`
and `em·z_H = 1` ⇒ `mu_H = u_H + phi`. Substituting
`u_H = log(H_outer) − ψ + log_gamma_psi` gives `mu_H = log(H_outer) +
log_gamma_psi + (phi_outer)`, which is the cancellation
`mu_H_init = 2·log(H_outer) − log(c_clo4_bulk)` cited in
`forms_logc_muh.py:769-776` — but only when `phi_outer ≡ φ_o = log(H_o/c_clo4_bulk)`.
For multi-ion, `phi_outer ≡ _solve_outer_phi_multiion(...)`, and
the cancellation gives a different (multi-ion-correct) closed form.

**Fix to plan:** Phase 2.4 explicitly:

1. Solve Picard scalars (R_list, c_s, H_o, ψ_D, ψ_S, γ_s) → φ_o
   from `_solve_outer_phi_multiion` (Issue 2 below).
2. Build spatial `psi(y)` profile from linearized-Debye seed:
   `psi(y) = psi_D · exp(-y/lambda_eff)`. (Or from BKSA composite
   form for single-ion legacy path; preserved for byte-equivalent
   regression.)
3. Build spatial `phi_outer(y)` profile (slow): linear interp
   from `phi_o` at OHP-side edge to `0` at bulk.
4. Construct `phi_init(y) = phi_outer(y) + psi(y)`.
5. Construct `log_c_i_seed(y) = log(c_i_outer(y)) − z_i·psi(y) +
   log_gamma_psi(y)` for every species.
6. Interpolate to FE Functions.
7. For `logc_muh`: `mu_H_init(y) = u_H_init(y) + em·z_H·phi_init(y)`.

This properly separates the slow φ_outer evolution from the fast ψ
diffuse-layer profile. The Boltzmann shift uses ψ; the Poisson
primary uses φ; mu_H gets both correctly via the existing
formulation.

### Issue 2 — O₂/H₂O₂ outer aren't bulk; interpolate from Picard surface

**Accept.** In the matched-asymptotic IC, the dynamic species
transition from Picard's surface values `(O_s, P_s, H_o)` at y=0
toward bulk `(O_b, P_b, H_b)` at y=mesh_top, over a length scale
set by diffusion (typically L_REF, much longer than λ_D).

The `_solve_outer_phi_multiion` therefore takes
`c_dyn_outer(y)`, not just `c_dyn_b`. At the diffuse-layer edge
(y = O(λ_D), small relative to L_REF), `c_dyn_outer` is essentially
at the surface value `O_s`/`P_s`/`H_o`, not bulk — because the
diffusion-layer profile hasn't relaxed yet.

For the spatial IC build:

```python
# y_outer_at_ohp = small distance from y=0; effectively the surface side
# y_outer_at_bulk = large distance; bulk side
def c_dyn_outer_profile(y):
    # Linear interp from surface to bulk over L_REF.
    frac = min(y / L_REF, 1.0)
    O_y = (1 - frac) * O_s + frac * O_b
    P_y = (1 - frac) * P_s + frac * P_b
    H_y = (1 - frac) * H_o + frac * H_b
    return [O_y, P_y, H_y]

def phi_outer_profile(y):
    # Solve _solve_outer_phi_multiion at each y using c_dyn_outer(y).
    # In practice the analytic ions vary much faster than the dyn species
    # within the diffusion layer, so phi_outer(y) is dominated by the
    # outer-region balance with H_o(y), not c_O2(y).
    ...
```

**Fix to plan:** Phase 2.4 evaluates `_solve_outer_phi_multiion` at
multiple y points (typically just the OHP-side edge and bulk; linear
interp between for spatial IC) using the local `c_dyn_outer(y)`
values, NOT just `c_dyn_bulk`. For Picard's scalar use (which is
the OHP-side edge only), this reduces to taking `c_dyn_outer = (O_s,
P_s, H_o)` rather than `(O_b, P_b, H_b)`.

### Issue 3 — Local λ_eff with finite-difference dρ/dφ

**Accept (capitulating from R4 defense).** GPT's pushback is
well-founded: with Stern dominating ~93% of the drop, the IC's
Stern split is sensitive to `λ_eff`. A 20× error from bulk-vs-local
linearization could put `ψ_D` on the wrong branch of the Stern
nonlinearity.

**Fix:** `effective_debye_length` computes local `λ_eff` from
`-dρ/dφ_outer` via finite difference:

```python
def effective_debye_length_local(
    *, phi_o: float, ions: list, theta_b: float,
    z_dyn: list[int], c_dyn_outer: list[float], a_dyn: list[float],
    poisson_coeff: float, dphi: float = 1e-4,
) -> float:
    """Local λ_eff = sqrt(eps / |dρ/dφ|_outer) via finite difference."""
    def rho_at(phi):
        denom = theta_b + sum(
            ion["a_nondim"] * ion["c_bulk_nondim"]
            * math.exp(-ion["z"] * phi) for ion in ions
        )
        A_dyn = sum(a * c for a, c in zip(a_dyn, c_dyn_outer))
        ck = [ion["c_bulk_nondim"] * math.exp(-ion["z"] * phi)
              * (1 - A_dyn) / denom for ion in ions]
        rho_dyn = sum(z * c for z, c in zip(z_dyn, c_dyn_outer))
        rho_an = sum(ion["z"] * c for ion, c in zip(ions, ck))
        return rho_dyn + rho_an
    drho_dphi = (rho_at(phi_o + dphi) - rho_at(phi_o - dphi)) / (2 * dphi)
    inv_lambda_sq = max(-drho_dphi, 1e-30) / poisson_coeff
    return math.sqrt(1.0 / inv_lambda_sq)
```

Cost: 2 evaluations of the closure per call. Negligible.

Bulk linearization remains as a sanity-check / fallback when
`phi_o` is near 0 (where local and bulk agree).

### Issue 4 — `multi_ion_enabled=False` with len>1 footgun

**Accept.** Hard validation in `make_bv_solver_params`:

```python
def make_bv_solver_params(...):
    ...
    n_counterions = len(boltzmann_counterions or [])
    if n_counterions > 1 and not multi_ion_enabled:
        raise ValueError(
            f"make_bv_solver_params: len(boltzmann_counterions)={n_counterions}"
            f" requires multi_ion_enabled=True. The single-counterion legacy"
            f" code path will silently use boltzmann_counterions[0] only,"
            f" dropping the others. To use multiple analytic counterions"
            f" (e.g. Cs+ and SO4--), pass multi_ion_enabled=True."
        )
    if n_counterions == 1 and multi_ion_enabled:
        # Permit (the multi-ion code path reduces to single-ion correctly),
        # but warn the user once that they're paying for new code without
        # benefit.
        warnings.warn(
            "multi_ion_enabled=True with len(boltzmann_counterions)==1: "
            "the new multi-ion code path is engaged but reduces to "
            "single-ion. For byte-equivalent legacy behavior, pass "
            "multi_ion_enabled=False.",
            stacklevel=2,
        )
```

This makes the failure mode loud, not silent.

### Issue 5 — Pass C contradiction

**Accept.** Removing Pass C from "Done" criterion entirely:

> **"Done" acceptance:**
> - Pass A (pure-2e): ≥ 15/25 V_RHE converged with non-zero gross R_2e.
> - Pass B (pure-4e): ≥ 15/25 V_RHE converged.
> - Pass D (mixed reduced K0_R4e ladder): ≥ 15/25 V_RHE converged
>   with non-zero R_2e AND R_4e contributions, at the chosen ladder
>   factor.
> - Pass C (mixed literature K0): NOT REQUIRED for done. Pass C is
>   exploratory only — runs once at anchor, reports what the bare
>   literature placeholder produces (expected: R_4e dominates, R_2e
>   invisible). Used to inform M4 calibration scope post-fast-
>   realignment.
> If any of {A, B, D} ≥ 15/25 is unreachable in the time budget, ask
> the user before lowering.

## Section 2 — Updated plan summary

Cumulative R2 + R3 + R4 + R5 changes. Final form:

- **Phase 0** Git checkpointing.
- **Phase 1.1** Disabled-rxn `ln(k0)` guard.
- **Phase 1.2** Audit existing `picard_outer_loop_general`;
  instrument with `verbose` flag; identify failure mode (Picard /
  reconstruction / Newton).
- **Phase 1.3** Centralized disabled-rxn helper + strict topology
  predicate (classifies from nominal config).
- **Phase 2.1** Multi-steric Bikerman closed-form generalization in
  `boltzmann.py` (drop `len > 1` rejection; shared-theta closure).
- **Phase 2.2** Hard-sphere `a_nondim` from physical hydrated radii
  (Cs⁺ ≈ 3.23e-5, SO₄²⁻ ≈ 4.20e-5).
- **Phase 2.3** New `Forward/bv_solver/multi_ion.py`:
  - `CounterionConfig` dataclass.
  - `build_counterion_ctx()` (single producer of theta_b).
  - `_solve_outer_phi_multiion()` consuming
    `c_dyn_outer` (R5 Issue 2) AND full a_dyn (R4 Issue 1).
  - `compute_surface_gamma_multiion()` with outer anchors (R3 Issue 3).
  - `effective_debye_length_local()` from FD `-dρ/dφ` (R5 Issue 3).
  - Hard validation: `len(counterions) > 1 ⇒ multi_ion_enabled=True
    required` (R5 Issue 4).
- **Phase 2.4** Spatial IC properly separates `phi_outer(y)` from
  `psi(y)` (R5 Issue 1):
  - `psi(y) = psi_D · exp(-y/lambda_eff_local)` (linearized-Debye).
  - `phi_outer(y) = phi_o · (1 − y/L_REF)` (linear, slow).
  - `phi_init(y) = phi_outer(y) + psi(y)`.
  - `log_c_i_seed(y) = log(c_i_outer(y)) − z_i·psi(y) +
    log_gamma_psi(y)` for all species.
  - `c_dyn_outer(y)` interpolates from Picard `O_s/P_s/H_o` at OHP
    edge to bulk over L_REF (R5 Issue 2).
  - `mu_H_init(y) = u_H_init(y) + em·z_H·phi_init(y)` for `logc_muh`
    formulation.
  - Stern split: linear-Debye fallback with `lambda_eff_local`
    (R3 Issue 5; R5 Issue 3).
  - Single-ion legacy path: preserved via the `multi_ion_enabled=False`
    branch — byte-equivalent regression for ClO₄⁻ + sequential
    runs.
- **Phase 3** Anchor V=+0.55. Driver runs Pass A (pure-2e), Pass B
  (pure-4e), Pass C (mixed literature K0; exploratory only), Pass D
  (mixed K0_R4e LADDER {1e-12, 1e-15, 1e-18, 1e-21, 1e-24}).
- **Phase 4** Smoke gate per-pass:
  - Pass A: ≥ 5/25 V_RHE converged with gross R_2e ≠ 0.
  - Pass D: ≥ 5/25 V_RHE converged with R_2e and R_4e both ≠ 0
    at chosen ladder factor.
  - Pass B: anchor + warm-walk (1/25 OK).
  - Pass C: anchor + warm-walk (1/25 OK; reports R_4e domination).
- **Phase 5** Fix breaks:
  - 5a. Picard non-convergence: omega → 0.3 / 0.2; max_iters → 100.
  - 5b. Newton non-convergence: continuation in I (ramp from
    legacy ClO4 bulk concentrations toward Ruggiero values);
    continuation in `a_nondim` (start at 0, NOT A_DEFAULT — R2
    Issue 10); mesh refinement Ny 200→400→800.
  - 5c. Anchor relocation: V scan {+0.55, +0.50, +0.45, +0.40,
    +0.35, +0.30, +0.20, 0.0}.
  - 5d. `a_nondim` calibration: reduce hard-sphere values until
    `theta_b > 0` and pack-fraction stays bounded across EDL.
  - 5e. Legacy regression: leave broken; user fixes later.
  - 5f. Concrete legacy warm-start (R3 Issue 11):
    `_legacy_warmstart_to_target` per the spelled-out operation.
- **"Done" criterion:**
  - Pass A: ≥ 15/25 V_RHE converged.
  - Pass B: ≥ 15/25 V_RHE converged.
  - Pass D: ≥ 15/25 V_RHE converged at chosen ladder factor.
  - Pass C: NOT required for done (R5 Issue 5).
  - If A/B/D ≥ 15/25 is unreachable in the time budget, ask user
    before lowering.

Estimated total: 7-12 days. R5 fixes are clarifications + a
finite-difference helper; no new module beyond R4's `multi_ion.py`.

## Section 3 — Continued critique prompt

This is round 5 of 5 (cap reached). Final review:

Review the updated plan and my responses to your earlier issues.
Push back on responses where I defended poorly — name which point.
Raise any new issues the updated plan creates. Re-issue any earlier
issue you don't think I addressed. Same numbered format and same
verdict line at the end:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN

If you verdict ISSUES_REMAIN at round 5, the plan auto-revises with
all accepted-issue fixes and any unresolved points are documented
in `FINAL_REVISION.md` for the user to address in implementation.
