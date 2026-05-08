# Round 3 — Counterreply

## Section 1 — Per-issue acknowledgment

### Issue 1 — SO₄²⁻ blows up at +0.55 V; "Cs steric / SO₄ ideal" is wrong-physics

**Accept the diagnosis. Reject the proposed remedy. Adopt a third
path GPT didn't list: full multi-steric Bikerman with closed-form
shared-theta ansatz.**

You're right that SO₄²⁻ ideal explodes at +0.55 V. With z=-2, c_b =
100 mol/m³, ψ_D ≈ 1.5-2 (after Stern eats 90% of the drop per Issue
5), the surface c_SO4 = 100·exp(2·1.8) ≈ 3650 mol/m³. Heading
toward saturation density (~50000 mol/m³ for hard spheres) but not
catastrophic — yet. As we walk toward V=0, ψ_D grows; sulfate
saturates first.

**However:** I checked
`docs/steric_analytic_clo4_reduction_handoff.md` — the Bikerman
chemical-potential derivation generalizes to N analytic ions in
closed form, contrary to the "couples" comment in
`boltzmann.py:159-165`. Re-derivation:

```text
For each analytic ion k (steric, no flux at boundary):
  μ_k = ln(c_k) + z_k·φ - ln(θ) = const = ln(c_b_k) - ln(θ_b)
  ⇒ c_k(φ) = c_b_k · exp(-z_k·φ) · θ(φ)/θ_b   ≡ B_k(φ)·θ(φ)

with B_k(φ) = c_b_k · exp(-z_k·φ) / θ_b. Then:
  θ(φ) = 1 - A_dyn(φ) - Σ_k a_k · c_k(φ)
       = 1 - A_dyn(φ) - θ(φ) · Σ_k a_k · B_k(φ)
  ⇒ θ(φ) · [1 + Σ_k a_k·B_k(φ)] = 1 - A_dyn(φ)
  ⇒ θ(φ) = (1 - A_dyn(φ)) · θ_b / (θ_b + Σ_k a_k·c_b_k·exp(-z_k·φ))

Then:
  c_k(φ) = c_b_k · exp(-z_k·φ) · (1 - A_dyn(φ))
                     / (θ_b + Σ_{k'} a_{k'}·c_b_{k'}·exp(-z_{k'}·φ))
```

**This IS closed form** — same denominator for every analytic ion;
just summed over all entries. The single-ion case
(`boltzmann.py:215-225`) is the K=1 special case. Multi-ion is the
exact same structure with the sum in the denominator. No coupled
local NL solve required.

The "couples" comment in `boltzmann.py:159-165` describes what
mathematicians call "shared denominator" but isn't an iteration —
the closed-form is single-pass. The author may have been worried
about a different ansatz (per-ion theta) which IS iterative.

**Fix to plan:** Replace the "Cs steric / SO₄ ideal" hack with the
proper multi-steric closed-form. Implementation:

1. `boltzmann.py`: replace the `len(bikerman) > 1` rejection with
   a generalized loop. The numerator becomes
   `c_b_k · exp(-z_k·φ_clamped) · (1 - A_dyn_local)` per ion;
   the denominator becomes
   `θ_b + Σ_k a_k·c_b_k·exp(-z_k·φ_clamped)` shared across all
   ions. Returns a list of `StericBoltzmannBundle` instead of
   one. Caller (forms_logc{,_muh}.py) sums each bundle's
   `charge_density` into the Poisson residual and each bundle's
   `packing_contribution` into the dynamic-species `theta`.
2. The bulk pack-fraction check generalizes to
   `θ_b = 1 - A_dyn_bulk - Σ_k a_k · c_b_k > 0`.
3. The double-counting guard generalizes to "no analytic entry
   matches any dynamic species (z, c_b)".

Effort: ~1 day. The closed-form structure makes this a code
generalization, not a derivation.

This obsoletes my Issue 8 fast-path (Cs steric + SO₄ ideal).

### Issue 2 — `_solve_outer_phi_multiion` ideal vs steric

**Accept.** With multi-steric closure, the outer-region
electroneutrality residual becomes:

```text
H_o + Σ_k z_k · c_b_k · exp(-z_k·φ_o) · (1 - A_dyn_outer)
                  / (θ_b + Σ_{k'} a_{k'}·c_b_{k'}·exp(-z_{k'}·φ_o)) = 0
```

In the outer region, A_dyn_outer ≈ A_dyn_bulk = a_H · H_o (with
a_O2 = a_H2O2 = 0). So:

```text
H_o + (1 - a_H·H_o) · Σ_k z_k · c_b_k · exp(-z_k·φ_o)
                  / (θ_b + Σ_{k'} a_{k'}·c_b_{k'}·exp(-z_{k'}·φ_o)) = 0
```

Bisect on φ_o; ~30 iter at 1e-12 tolerance. Single-ion case
(K=1) reduces to a slight modification of the existing
`log(H_o/c_clo4_bulk)` form — actually NOT byte-equivalent
because the existing form ignores the steric (1-a_H·H_o)/θ
factor. (The legacy bug is acceptable for a 1:1 dilute system but
has been silently wrong this whole time.)

**Fix:** Add `_solve_outer_phi_multiion` in `picard_ic.py`. ~50 LOC
including bisection + bracketing. Default to single-ion legacy
form when `len(boltzmann_counterions) == 1` AND `a_H == 0` (for
byte-equivalent regression).

### Issue 3 — γ uses bulk anchors but should use outer

**Accept.** The activity coefficient at the OHP should reflect the
local concentrations at the OHP, not the bulk reservoir. Each
analytic ion's outer-region concentration is:

```text
c_k_outer = c_b_k · exp(-z_k·φ_o) · θ_outer / θ_b
```

with θ_outer from the closure at φ_o, A_dyn_outer.

Then γ_s (at the OHP, after the Boltzmann shift through ψ_D):

```text
γ_s = 1 / (1 + a_H·H_o·(e^(-ψ_D)−1)
             + Σ_k a_k·c_k_outer·(e^(-z_k·ψ_D)−1))
```

NOT `c_b_k`. This is consistent with the existing
`compute_surface_gamma` having `c_cl_anchor` as a separate
parameter (defaulting to `c_clo4_bulk` for analytic ClO₄⁻ but `H_o`
for synthesised-4sp).

**Fix:** Add `compute_surface_gamma_multiion(H_o, ψ_D, a_H,
ions: list[dict])` where each entry is `{"z": ±k, "a": a_nondim,
"c_outer": c_k_outer}`. Caller computes c_outer for each ion using
the closure with the current φ_o iterate. Single-ion case (K=1)
preserved by the existing `compute_surface_gamma`.

### Issue 4 — Don't change `poisson_coefficient`

**Accept.** Confirmed: `poisson_coefficient` is the residual's
nondim base (the prefactor on `Σ z_i² c_i` in PNP). Modifying it
double-counts ionic strength.

**Fix:** Use `λ_eff = sqrt(poisson_coefficient / Σ z_i² c_b_i_nondim)`
**only** in IC and Stern helpers. Add a helper:

```python
def effective_debye_length(poisson_coeff: float, bulk_concs: list,
                           charges: list) -> float:
    sum_zsq_c = sum(int(z)**2 * float(c)
                    for z, c in zip(charges, bulk_concs))
    if sum_zsq_c <= 0:
        return math.sqrt(max(poisson_coeff, 1e-30))
    return math.sqrt(poisson_coeff / sum_zsq_c)
```

`bulk_concs` includes both dynamic species AND analytic
counterions; charges accordingly. Use this `λ_eff` in
`solve_stern_split` (replacing the hardcoded `λ_D =
sqrt(poisson_coeff)`) and any IC linearized-Debye seed.

### Issue 5 — Stern dominates; can't disable at IC

**Accept.** GPT's capacitance arithmetic is right:

- C_diffuse ≈ ε/λ_D = (78·8.85e-12)/(0.55e-9) ≈ 1.25 F/m²
- C_Stern = 0.10 F/m²  (`stern_capacitance_f_m2`)
- C_total = (1/C_d + 1/C_S)^-1 = 1/(0.8 + 10) = 0.0926 F/m²
- ψ_S/ψ_total = C_total/C_Stern = 0.926
- Stern absorbs ~93% of the applied drop.

Disabling Stern at IC leaves Newton to reconstruct the dominant
piece of the EDL structure. Catastrophic for convergence.

**Fix to plan:** Use the **existing linearized-Debye Stern fallback**
already in `picard_ic.py:265-270`:

```python
denom = eps_nondim + stern_coeff_nondim * lambda_D
psi_D = stern_coeff * full_drop * lambda_D / denom
psi_S = full_drop - psi_D
```

This is sign-correct in the small-|ψ_D| regime. For multi-ion,
substitute `λ_D → λ_eff` (Issue 4 fix). At I=0.3 M, λ_eff ≈
0.55 nm, much smaller than the legacy ClO₄⁻ value, so the linearized
form gives a different ψ_D distribution but stays bounded.

The full BKSA nonlinear Stern split for 2:1 multi-ion is deferred —
the linear-Debye fallback gives the right asymptotic behavior for
small ψ_D and Newton handles the rest.

### Issue 6 — Disabled-reaction topology classification

**Accept.** Topology must be classified from the nominal config
(2-rxn parallel for `PARALLEL_2E_4E_REACTIONS`). Disabled reactions
contribute zero rows / zero rates but DON'T change the topology
hint.

**Fix:**

```python
def _classify_topology(reactions: list, h_idx: int) -> str:
    if _is_parallel_2e_4e(reactions, h_idx):    # strict, ignores enabled flag
        return "parallel_2e_4e"
    if _is_sequential_2e_h2o2(reactions):
        return "sequential_2e_h2o2"
    return "general"
```

In the form builder's per-rxn loop, `_is_reaction_disabled(rxn)`
gates only the rate computation (`R_j = fd.Constant(0.0)`). In the
Picard's `_assemble_n_reaction_system`, disabled rxns produce
`M[j,j] = 1`, `b[j] = 0` so the linear solve gives `R_j = 0`.

Pure-2e probe is then "parallel topology + R_4e disabled," not "1-rxn
sequential" — preserves topology dispatch.

### Issue 7 — Spatial IC still hardcodes `ln(H_outer/c_clo4_bulk)`

**Accept and propagate Issue 8 fix.** The problem flows from
overloading `c_clo4_bulk` as the single-counterion anchor. After
the Issue 8 refactor (structured counterion context), the spatial
IC interpolation in `_try_debye_boltzmann_ic*` rebuilds φ(y) as:

```text
φ(y=0) = φ_surface (post-Stern, from solve_stern_split)
φ(y) interpolated via composite-ψ profile (BKSA / linearized-Debye)
       using λ_eff and the multi-ion outer ψ_o
```

The `ln(H_outer/c_clo4_bulk)` line is replaced by the multi-ion
`_solve_outer_phi_multiion(H_o, ions_list)`.

For the dynamic species (O₂, H₂O₂), the spatial seed remains
linear-in-y between surface (Picard `c_s`) and bulk (`c_b`). For
H⁺, the Boltzmann-shifted seed uses the multi-ion ψ_D / λ_eff
profile.

**Fix to plan:** Phase 2.4 (linearized-Debye spatial IC) explicitly
removes `c_clo4_bulk` from the spatial IC code path. Both
`forms_logc.py` and `forms_logc_muh.py` get the same treatment.

### Issue 8 — `c_clo4_bulk` overloaded throughout adapters

**Accept (this is the load-bearing structural refactor).** Verified:

- `forms_logc_muh.py:873`:
  `c_clo4_bulk = max(float(counterions[0]["c_bulk_nondim"]), 1e-300)`
- Same line in `forms_logc.py`.
- Then `c_clo4_bulk` is passed verbatim into
  `picard_outer_loop_general`, `solve_stern_split`,
  `compute_surface_gamma`, and the spatial IC interpolation. With
  `[Cs, SO4]` it's whichever happens to be at index 0.

**Fix:** Refactor to a structured `counterion_ctx` dict:

```python
counterion_ctx = {
    "ions": [
        {"label": "Cs+",  "z": +1, "c_bulk_nondim": C_CSPLUS_HAT,
         "a_nondim": A_CSPLUS_HAT, "steric_mode": "bikerman"},
        {"label": "SO4--","z": -2, "c_bulk_nondim": C_SO4_HAT,
         "a_nondim": A_SO4_HAT, "steric_mode": "bikerman"},
    ],
    "theta_b": 1 - A_dyn_bulk - sum(a_k * c_k for ...),
}
```

Then:

- `picard_outer_loop_general` consumes `counterion_ctx`, computes
  φ_o + γ_s using the multi-ion helpers.
- `solve_stern_split` consumes `counterion_ctx` (or, equivalently,
  the precomputed `λ_eff`).
- Spatial IC interpolation consumes `counterion_ctx` for the
  composite-ψ profile.
- Single-ion legacy path: build `counterion_ctx` from
  `[counterions[0]]` for byte-equivalent regression.

Effort: ~1 day. Touches signatures across `picard_ic.py`,
`boltzmann.py`, `forms_logc.py`, `forms_logc_muh.py`. Mostly
mechanical.

### Issue 9 — "Done" criterion lowered without approval

**Accept.** Restoring "Done" to ≥ 15/25 V_RHE converged across
Pass A + Pass B + Pass C + Pass D (Issue 10). Phase 4 acceptance
(intermediate gate) becomes ≥ 5/25 on Pass A only as a "structural
plumbing works" smoke test; final acceptance is the original 15/25
across the full pass set.

If 15/25 turns out to be unreachable in the time budget, will ASK
the user before lowering, not silently downgrade.

### Issue 10 — Pure-2e doesn't exercise mixed parallel; mixed pass yields invisible R_2e

**Accept.** Add Pass D explicitly:

> **Pass D — Mixed with reduced K0_PHYS_R4E.** Run with
> `K0_PHYS_R4E = 1e-15 × K0_PHYS_R2E` so the BV-factor advantage
> for R_4e (~1e15 at +0.55 V from Issue 17) is roughly canceled.
> Yields R_2e and R_4e at comparable orders of magnitude → both
> channels structurally exercise the parallel-topology code path
> with inspectable peroxide curve. **Label as "non-physical k0
> ratio; structural validation only; not page-15 interpretable."**

Pass C (mixed at literature K0) remains as a "what does the bare
literature placeholder produce" run, expected R_4e-dominated.

Phase 4 acceptance updated to: Pass A ≥ 5/25 V_RHE converged AND
Pass D ≥ 5/25 V_RHE converged with non-zero gross R_2e and
non-zero R_4e contribution to disk current. Pass B and Pass C
plumbing-tested at the same anchor V (one V each).

### Issue 11 — Legacy warm-start fallback underspecified

**Accept.** Concrete operation:

```python
def _legacy_warmstart_to_target(
    *, sp_legacy, sp_target, v_anchor, mesh_args
):
    """Solve legacy ctx at v_anchor; copy compatible state into
    a freshly built target ctx."""
    # Step 1: build legacy ctx (ClO4 + sequential + 3sp logc_muh)
    ctx_legacy = build_ctx(sp_legacy, mesh_args)
    set_initial_conditions(ctx_legacy, sp_legacy)  # debye_boltzmann
    solve_steady_state(ctx_legacy, v_anchor)
    # Step 2: build target ctx (Cs+/SO4 + parallel + 3sp logc_muh)
    ctx_target = build_ctx(sp_target, mesh_args)  # same mesh, same n_species,
                                                  # same formulation
    # Step 3: copy U state subfunction-by-subfunction.
    # Both stacks use THREE_SPECIES_LOGC_BOLTZMANN: indices [O2,H2O2,H+,phi]
    # are aligned (and mu_H_idx = 2 for both).
    for src_sub, dst_sub in zip(ctx_legacy["U"].subfunctions,
                                ctx_target["U"].subfunctions):
        dst_sub.dat.data[:] = src_sub.dat.data_ro
    ctx_target["U_prev"].assign(ctx_target["U"])
    # Step 4: solve target residual at v_anchor
    solve_steady_state(ctx_target, v_anchor)
    return ctx_target
```

Failure modes (and what they mean):

- Step 4 cold-fails: target residual doesn't accept the legacy U
  state because (e.g.) the multi-ion analytic counterions push the
  bulk Poisson balance off.
- Step 4 succeeds: warm-walk from this ctx via C+D.

**Fix to plan:** §5 (fallback list) item 2 spells this out. ~1 day
to wire (mostly the per-subfunction copy + ensuring mesh +
formulation match).

## Section 2 — Updated plan summary

Plan changes (cumulative with R2 changes; will apply to artifact
after VERDICT APPROVED):

- **Phase 0** — unchanged.
- **Phase 1.1** — unchanged.
- **Phase 1.2** — Audit + instrument existing
  `picard_outer_loop_general` (R2 fix).
- **Phase 1.3** — Centralized disabled-reaction helper + strict
  topology predicate, classifying from NOMINAL config (R3 Issue 6).
- **Phase 2.1** — Multi-steric Bikerman closed-form generalization
  in `boltzmann.py` (drop `len > 1` rejection; both Cs⁺ and SO₄²⁻
  steric). R3 Issue 1.
- **Phase 2.2** — Hard-sphere `a_nondim` from physical hydrated
  radii (R2 Issue 9). Cs⁺ a≈3.23e-5, SO₄²⁻ a≈4.20e-5.
- **Phase 2.3** — Refactor `c_clo4_bulk` overload → structured
  `counterion_ctx` (R3 Issue 8). Add `effective_debye_length()`
  helper (R3 Issue 4) + `_solve_outer_phi_multiion()` (R3 Issue 2)
  + `compute_surface_gamma_multiion()` with outer anchors (R3 Issue 3).
- **Phase 2.4** — Spatial IC: linear-Debye seed for phi(y) using
  `λ_eff` and multi-ion ψ_o; remove `c_clo4_bulk` from spatial IC
  path (R3 Issue 7). Stern split via existing linearized-Debye
  fallback with `λ_eff` (R3 Issue 5).
- **Phase 3** — Anchor V = +0.55 (R2 Issue 16). Run Pass A pure-2e,
  Pass B pure-4e, Pass C mixed at literature K0, Pass D mixed at
  reduced K0_R4e (R3 Issue 10).
- **Phase 4** — Acceptance (smoke gate): Pass A ≥ 5/25 converged
  AND Pass D ≥ 5/25 converged with non-zero R_2e and R_4e
  contributions (R3 Issue 9).
- **Phase 5** — Fix breaks. Continuation in `a_nondim` from 0
  (R2 Issue 10). Concrete legacy warm-start operation (R3 Issue 11).
- **"Done" criterion** — Pass A + B + C + D combined ≥ 15/25 V_RHE
  converged. If unreachable, ask user before lowering.

Estimated total: 7-12 days (was 5-10). The structural refactor
(R3 Issues 1, 2, 3, 4, 5, 7, 8) adds load but is required.

## Section 3 — Continued critique prompt

Review the updated plan and my responses to your earlier issues.
Push back on responses where I defended poorly — name which point.
Raise any new issues the updated plan creates. Re-issue any earlier
issue you don't think I addressed. Same numbered format and same
verdict line at the end:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN
