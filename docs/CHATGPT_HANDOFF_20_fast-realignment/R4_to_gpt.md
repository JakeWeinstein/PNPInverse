# Round 4 — Counterreply

## Section 1 — Per-issue acknowledgment

### Issue 1 — Outer-φ formula silently zeros `a_O2`/`a_H2O2`

**Accept.** Verified `scripts/_bv_common.py:242`:

```python
THREE_SPECIES_LOGC_BOLTZMANN = SpeciesConfig(
    ...
    a_vals_hat=[A_DEFAULT] * 3,   # all three species get A_DEFAULT = 0.01
    ...
)
```

Production config has `a_O2 = a_H2O2 = a_H = 0.01`. My
`A_dyn_outer = a_H·H_o` was wrong — it must include all dynamic
species:

```text
A_dyn_outer = a_O2 · c_O2_outer + a_H2O2 · c_H2O2_outer + a_H · H_o
            = a_O2 · O_b + a_H2O2 · P_b + a_H · H_o    (in outer region)
```

(Picard's outer-region surface concs `O_s, P_s, H_o` represent
neutral-species depletion at the OHP, but for the `outer-region`
(just outside the diffuse layer) the dynamic-species concentrations
are essentially bulk-valued — except for H⁺ which Picard updates as
`H_o` per the proton-balance equation. Neutrals: bulk.)

Quick numerical check at bulk values (C_O2_HAT=1, C_H2O2_HAT=0, H_o
≈ C_HP_HAT=0.0833):
`A_dyn_outer = 0.01·(1 + 0 + 0.0833) = 0.01083`, so
`(1 - A_dyn_outer) = 0.989`. The error from my zeroing was ~1%,
small but nonzero. Fix anyway — multi-ion code shouldn't hardcode
species-specific assumptions.

**Fix:** `_solve_outer_phi_multiion` takes the full dynamic-species
list `(z_dyn, c_dyn_outer, a_dyn)` instead of just `H_o, a_H`. Uses
the closure formula consistently:

```python
def _solve_outer_phi_multiion(
    *, c_dyn_outer: list[float], a_dyn: list[float], z_dyn: list[int],
    ions: list[dict],   # analytic counterions: z, a, c_b
    theta_b: float, ...
) -> float:
    A_dyn_outer = sum(a * c for a, c in zip(a_dyn, c_dyn_outer))
    def residual(phi_o):
        # Multi-steric concentration of each analytic ion
        denom = theta_b + sum(
            ion["a"] * ion["c_b"] * math.exp(-ion["z"] * phi_o)
            for ion in ions
        )
        c_k = [ion["c_b"] * math.exp(-ion["z"] * phi_o) * (1 - A_dyn_outer)
                  / denom for ion in ions]
        # Outer-region electroneutrality: Σ_dyn z_i c_i + Σ_k z_k c_k = 0
        rho_dyn = sum(z * c for z, c in zip(z_dyn, c_dyn_outer))
        rho_an  = sum(ion["z"] * c for ion, c in zip(ions, c_k))
        return rho_dyn + rho_an
    # bisect on phi_o
    ...
```

### Issue 2 — Legacy byte-equivalence condition wrong

**Accept.** Production config has `a_H = 0.01` (not 0), so
`len==1 AND a_H==0` never fires for the existing single-counterion
ClO₄⁻ Bikerman runs. My condition would silently route them
through the new multi-ion code (same math in the K=1 case, but
different code path → loss of byte-equivalent regression
guarantee).

**Fix:** Explicit opt-in via a `multi_ion_enabled` feature flag in
`make_bv_solver_params`. Default `False` ⇒ legacy single-counterion
code path (no signature change for existing callers). New callers
pass `multi_ion_enabled=True` AND a multi-ion `boltzmann_counterions`
list to engage the new code.

```python
def make_bv_solver_params(
    ...,
    boltzmann_counterions: Optional[Sequence[Dict[str, Any]]] = None,
    multi_ion_enabled: bool = False,   # NEW: explicit opt-in
    ...
):
    ...
```

In adapters (`forms_logc_muh.py:870` etc.):

```python
if not params["bv_bc"].get("multi_ion_enabled", False):
    # Legacy path: take counterions[0], use single-ion helpers
    c_clo4_bulk = float(counterions[0]["c_bulk_nondim"])
    ...
else:
    # New multi-ion path: build counterion_ctx, use multi-ion helpers
    counterion_ctx = build_counterion_ctx(counterions, ...)
    ...
```

This guarantees the existing 25/25 V_RHE production stack
(`peroxide_window_3sp_bikerman_muh.py`) gets byte-equivalent
results no matter what; only callers that explicitly opt in to
multi-ion engage the new code.

### Issue 3 — `effective_debye_length` should use local dρ/dφ at outer

**Defend with documented approximation.** GPT is technically right:
the local screening at φ_o is

```text
λ_eff_local² = ε / (-dρ/dφ |_outer)
```

where the derivative includes the steric denominator. For the
multi-steric closure, this expands to a sum involving each ion's
`z² · c_k(φ_o)` plus a correction for the shared denominator
derivative `dD/dφ`.

**For the IC seed only**, I argue the bulk linearization
`λ_eff_bulk = sqrt(ε / Σ z² c_b)` is "good enough":

1. The IC's purpose is to give Newton a reasonable spatial profile
   to start from, not a physics-perfect EDL. Newton handles the
   full nonlinearity.
2. At the outer region (just outside the diffuse layer), c_k(φ_o)
   ≠ c_b but is still O(c_b). For modest |φ_o| ≤ ~3 (which is
   what we expect post-Stern at I=0.3 M with C_Stern dominating
   per Issue 5), `c_k(φ_o) / c_b = exp(-z·φ_o)·θ_outer/θ_b`, which
   is O(1) for z=±1 and O(e^±6) ≈ 400 for z=±2 at |φ_o|=3.
3. Sulfate's contribution to `λ_eff_local` then differs from bulk
   by up to ~20× depending on sign of φ_o. This IS a real error.
4. But: the linear-Debye Stern split (Issue 5 fix) is itself only
   accurate for small ψ_D, which co-occurs with |φ_o| small. So the
   regime where the bulk-vs-local discrepancy matters is the same
   regime where the linearized split is already breaking down.
   The "fast" plan accepts both approximations together.

**Fix to plan:** Use bulk-Σz²c form in `effective_debye_length`,
documented as "outer-region linearization; valid for moderate φ_o ≲
2-3. If Phase 4 / 5 surfaces a large discrepancy
(λ_eff_phys vs. measured EDL thickness > 2× off, or Newton
non-convergence localized to large-φ_o V_RHE points), upgrade to the
local form." Add a TODO comment in the helper docstring.

GPT may push back here; if it does, I'll accept the local form. But
for the fast plan, I'd rather not make every helper fully correct
before getting an end-to-end run.

### Issue 4 — Spatial IC for neutral species needs γ shift

**Accept.** Verified: `picard_ic.py:457`:

```python
log_O_rxn = math.log(max(O_s, 1e-300)) + log_gamma
log_P_rxn = math.log(max(P_s, 1e-300)) + log_gamma
log_H_rxn = math.log(max(H_o, 1e-300)) - psi_D + log_gamma
```

`log_gamma` is added to ALL species, including neutrals (O₂, H₂O₂),
because the Bikerman chemical potential is `μ_i = ln(c_i) + z_i·φ -
ln(θ)` for every species — the `-ln(θ)` term applies regardless of
charge.

For the spatial IC at point y in [0, mesh_top]:

```text
c_i(y) = c_i_outer · θ(y) / θ_outer    (for any species i)
       = c_i_outer · γ_psi(y)         where γ_psi(y) = θ(y)/θ_outer

θ(y)   = (1 - A_dyn(y)) · θ_b
       / (θ_b + Σ_k a_k · c_b_k · exp(-z_k · φ(y)))
```

For dynamic species at outer (just outside diffuse layer), c_O2 ≈
O_b, c_H2O2 ≈ P_b. Spatial seed:

```python
# Neutral species (all share theta-shift)
log_c_i_seed(y) = log(c_i_outer) + log(theta(y) / theta_outer)
                = log(c_i_outer) + log_gamma_psi(y)

# Charged species H+ (additional Boltzmann shift)
log_c_H_seed(y) = log(H_outer) - phi(y)·z_H + log_gamma_psi(y)
                = log(H_outer) - phi(y)         (z_H = +1)
                + log_gamma_psi(y)
```

**Fix to plan:** Phase 2.4 builds `gamma_psi(y)` from the multi-ion
shared-theta closure with the spatial φ(y) profile (linear-Debye
seeded). All dynamic species pick up `log_gamma_psi(y)` in their
spatial IC. H⁺ additionally gets `-φ(y)·z_H = -φ(y)`.

### Issue 5 — Pass D fixed factor balances near anchor only

**Accept.** Quantification: at V_RHE, the BV-factor ratio
`R_4e/R_2e ∝ exp(2·η_2e − 2·η_4e) = exp(2·(E_2e − E_4e)) = exp(2·(0.695−1.23))
= exp(-1.07) = 0.34` ... wait that's in physical V. In nondim with
V_T=0.025693, exp(2·(-1.07)/0.025693)·...

Hmm let me redo:
`log(R_4e/R_2e) = (-α_4e·n_4e·η_4e) - (-α_2e·n_2e·η_2e)`
                = `α_2e·n_2e·η_2e − α_4e·n_4e·η_4e`
                = `1.254·(V−0.695)/V_T − 2.0·(V−1.23)/V_T`
                = `(V/V_T)·(1.254 − 2.0) + (−0.695·1.254 + 1.23·2.0)/V_T`
                = `−0.746·V/V_T + (1.589)/V_T`
                = `(1.589 − 0.746·V) / V_T`

At V=+0.55: `(1.589 − 0.410)/0.0257 = 1.179/0.0257 = 45.86` →
exp(45.86) ≈ 8.3e19.

At V=0: `1.589/0.0257 = 61.83` → exp(61.83) ≈ 7.5e26.

At V=-0.4: `(1.589+0.298)/0.0257 = 73.43` → exp(73.43) ≈ 8.0e31.

So the BV ratio R_4e/R_2e ranges from 8e19 (at +0.55 V) to 8e31
(at -0.4 V) across the page-15 grid. A fixed K0 ratio that balances
at +0.55 V leaves R_4e dominating by ~12 orders at V=-0.4 V.

**Fix to plan:** Pass D becomes a ladder, not a fixed factor:

```python
PASS_D_K0_RATIO_LADDER = [1e-12, 1e-15, 1e-18, 1e-21, 1e-24]
```

Acceptance: the smallest ratio that gives non-zero R_2e and R_4e
contributions to disk current at ≥3 of 5 spaced V points (V ∈
{-0.40, -0.20, 0, +0.20, +0.55}). Document the chosen ratio as
"non-physical, structural validation only."

### Issue 6 — "Done" criterion ambiguous (combined vs per-pass)

**Accept.** Restate per-pass:

> **Phase 4 acceptance (smoke gate, must pass before Phase 5):**
> - Pass A (pure-2e): ≥ 5/25 V_RHE converged with non-zero gross R_2e.
> - Pass D anchor + warm-walk (mixed reduced K0_R4e): ≥ 5/25 V_RHE
>   converged with non-zero R_2e AND R_4e contributions.
> - Pass B (pure-4e): anchor + warm-walk single-V smoke (1/25 OK).
> - Pass C (mixed literature K0): anchor + warm-walk single-V smoke
>   (1/25 OK).
>
> **"Done" acceptance (single criterion):**
> - Pass A: ≥ 15/25 V_RHE converged.
> - Pass D: ≥ 15/25 V_RHE converged with at least one ladder factor.
> - Pass B: ≥ 15/25 V_RHE converged.
> - Pass C: anchor + warm-walk plumbing only (≥ 5/25 OK; structurally
>   diagnostic, not page-15 quantitative).
> If any of {A, B, D} ≥ 15/25 is unreachable in the time budget, ask
> the user before lowering. Pass C never gates done.

### Issue 7 — `counterion_ctx["theta_b"]` risks duplication

**Accept.** Single source of truth via a derived helper:

```python
# In `Forward/bv_solver/multi_ion.py` (new module):
@dataclass(frozen=True)
class CounterionConfig:
    z: int
    c_bulk_nondim: float
    a_nondim: float
    steric_mode: str  # "ideal" | "bikerman"
    label: str = ""

def build_counterion_ctx(
    counterions: list[CounterionConfig | dict],
    a_dyn: list[float],
    c_dyn_bulk: list[float],
    z_dyn: list[int],
) -> dict:
    """Build the canonical counterion context with derived theta_b."""
    ions = [_normalize(e) for e in counterions]
    A_dyn_bulk = sum(a * c for a, c in zip(a_dyn, c_dyn_bulk))
    A_an_bulk = sum(ion["a_nondim"] * ion["c_bulk_nondim"]
                    for ion in ions if ion["steric_mode"] == "bikerman")
    theta_b = 1.0 - A_dyn_bulk - A_an_bulk
    if theta_b <= 0:
        raise ValueError(f"theta_b <= 0 (got {theta_b})")
    return {
        "ions": ions,
        "z_dyn": list(z_dyn),
        "c_dyn_bulk": list(c_dyn_bulk),
        "a_dyn": list(a_dyn),
        "theta_b": theta_b,    # derived once; downstream reads it
    }
```

All downstream code (`boltzmann.py`, `picard_ic.py`,
`forms_logc.py`, `forms_logc_muh.py`) reads `ctx["theta_b"]` and
`ctx["ions"]`; never recomputes. If we need a derived quantity
(e.g. λ_eff, or local-charge-density), add a method to the helper.
No silent duplication.

**Fix to plan:** Phase 2.3 introduces this `multi_ion.py` module
with `CounterionConfig` + `build_counterion_ctx()` as the single
producer of `counterion_ctx`. All consumers use it.

## Section 2 — Updated plan summary

Cumulative changes (R2 + R3 + R4):

- **Phase 0** unchanged.
- **Phase 1.1** unchanged.
- **Phase 1.2** unchanged (audit existing Picard).
- **Phase 1.3** unchanged (disabled-rxn helper + strict topology;
  classify from nominal).
- **Phase 2.1** Multi-steric Bikerman closed-form (R3 Issue 1).
- **Phase 2.2** Hard-sphere `a_nondim` (R2 Issue 9).
- **Phase 2.3** NEW `Forward/bv_solver/multi_ion.py` with
  `CounterionConfig` + `build_counterion_ctx()` (R4 Issue 7).
  Includes `_solve_outer_phi_multiion` with full A_dyn (R4 Issue 1),
  `compute_surface_gamma_multiion` with outer anchors (R3 Issue 3),
  `effective_debye_length` (bulk Σz²c form; R4 Issue 3 with
  documented approximation). Adapter sites
  (`forms_logc{,_muh}.py:870`) gate on
  `params["bv_bc"]["multi_ion_enabled"]` for backward compat
  (R4 Issue 2).
- **Phase 2.4** Spatial IC: linear-Debye φ(y) seed using `λ_eff`
  and multi-ion ψ_o; spatial gamma_psi(y) from shared-theta closure
  applied to ALL dynamic species (R4 Issue 4); `c_clo4_bulk`
  removed from spatial IC code path (R3 Issue 7); Stern split via
  linearized-Debye fallback with `λ_eff` (R3 Issue 5).
- **Phase 3** Anchor V=+0.55. Pass A (pure-2e), B (pure-4e), C
  (mixed literature K0), D (mixed K0_R4e LADDER per R4 Issue 5).
- **Phase 4** acceptance per-pass smoke gate (R4 Issue 6).
- **Phase 5** Continuation in `a_nondim` from 0; concrete
  legacy warm-start (R3 Issue 11).
- **"Done"** per-pass criterion (R4 Issue 6).

Total estimate: 7-12 days. The R4 fixes are mostly clarifications
+ one new module (`multi_ion.py`); they don't add days beyond R3.

## Section 3 — Continued critique prompt

Review the updated plan and my responses to your earlier issues.
Push back on responses where I defended poorly — name which point.
Raise any new issues the updated plan creates. Re-issue any earlier
issue you don't think I addressed. Same numbered format and same
verdict line at the end:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN
