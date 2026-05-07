# Factory Correctness Report: `scripts/_bv_common.py`

Verifier: Claude Sonnet 4.6 (subagent)
Date reviewed: 2026-05-05
File: `/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/scripts/_bv_common.py` (562 lines)

---

## Executive Summary

The factory is **largely correct** for the production stack. All production flag
paths wire correctly and the critical `exponent_clip=100` default is present in
the right location (`_make_bv_convergence_cfg`, line 274). Three issues warrant
attention: one WARNING on `phi_clamp=50` in the ideal counterion preset, one
WARNING on the E_eq defaults silently being 0 (not the physical 0.68/1.78 V),
and one NOTE on a `conc_floor` value mismatch between factory and
`_default_bv_convergence_cfg`. None of these affect the production stack when
called exactly as CLAUDE.md specifies, but they create landmines for callers
who rely on defaults or reuse the ideal counterion preset.

---

## Issue 1 — WARNING: `phi_clamp=50.0` in `DEFAULT_CLO4_BOLTZMANN_COUNTERION`

**SEVERITY:** warning

**LOCATION:** `_bv_common.py` lines 407–411

```python
DEFAULT_CLO4_BOLTZMANN_COUNTERION: Dict[str, Any] = {
    "z": -1,
    "c_bulk_nondim": C_CLO4_HAT,
    "phi_clamp": 50.0,          # <-- this
}
```

**DESCRIPTION:**
`phi_clamp` controls the symmetric clamp on the Boltzmann exponent inside
`add_boltzmann_counterion_residual` and the ideal-counterion path. The value
`50.0` is the *old* historical default that predates the 2026-05-04 clip raise.
At `phi_clamp=50`, the ideal counterion Boltzmann exponent `exp(-z*phi)` is
clamped at `|phi|=50` (V/V_T). Because `z=-1` for ClO4-, the expression is
`exp(+phi)` in the anodic direction, so the clamp bites when `phi > 50` —
which corresponds to a physical potential of approximately `50 * 0.02569 V ≈
+1.28 V` above the reference. This is *outside* the production grid ceiling
of `+1.0 V_RHE`, so the ideal counterion clamp does not activate during normal
production sweeps. However, it creates a silent inconsistency:

- The steric counterion (`DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC`) inherits
  `phi_clamp=50` via `**DEFAULT_CLO4_BOLTZMANN_COUNTERION` (line 421).
- `config.py:_get_bv_boltzmann_counterions_cfg` falls back to `phi_clamp=50.0`
  as its own default (line 226).
- Multiple study scripts (e.g. `peroxide_window_muh_test.py`,
  `anodic_cold_start.py`) use `DEFAULT_CLO4_BOLTZMANN_COUNTERION` directly
  with `phi_clamp=50`.

None of this bites the production forward solve at V_RHE ≤ +1.0 V (phi < 50),
but it is a latent inconsistency. A more defensible value would be 100 to
match the raised `exponent_clip`. Consider raising
`DEFAULT_CLO4_BOLTZMANN_COUNTERION["phi_clamp"]` to `100.0` for consistency
with the `exponent_clip` and `u_clamp` values.

**EVIDENCE:** Lines 407–424 in `_bv_common.py`; `config.py` line 226 shows the
parser default is also 50.0.

---

## Issue 2 — WARNING: `E_eq_r1=0.0, E_eq_r2=0.0` factory defaults (physical values not baked in)

**SEVERITY:** warning

**LOCATION:** `_bv_common.py` lines 313–314 (`_make_bv_bc_cfg`) and lines
444–445 (`make_bv_solver_params`)

**DESCRIPTION:**
Both `_make_bv_bc_cfg` and `make_bv_solver_params` default `E_eq_r1=0.0` and
`E_eq_r2=0.0`. CLAUDE.md rule 4 says "Use physical `E_eq` (R1 = 0.68 V, R2 =
1.78 V vs RHE), never `E_eq = 0`." The factory does not enforce this — it is
entirely the caller's responsibility.

Empirically, only two scripts in the codebase actually pass the physical
values:

- `scripts/plot_iv_curve_unified.py` (line 167: `E_EQ_R1, E_EQ_R2 = 0.68, 1.78`).
- `scripts/studies/shape_match_v2.py` (lines 37–38).

Every other production-relevant script that uses `make_bv_solver_params` (e.g.
`peroxide_window_3sp_bikerman_muh.py`, `peroxide_window_stern_test.py`,
`anodic_cold_start.py`, `ic_refinement_study.py`) does NOT pass `E_eq_r1` or
`E_eq_r2`, silently defaulting to 0. This means their I-V curves are shifted
— the onset of R1 is placed at 0 V rather than +0.68 V vs RHE, and R2 at 0 V
rather than +1.78 V vs RHE. This is a systematic physics error in all runs that
don't explicitly set these.

The factory should either:
1. Expose `E_EQ_R1 = 0.68` and `E_EQ_R2 = 1.78` as module-level named
   constants so callers can import and use them, OR
2. Change the factory defaults to the physical values (breaking change for any
   intentional E_eq=0 test cases, but more correct), OR
3. At minimum, add a loud `warnings.warn` when the caller omits `E_eq_r1`/
   `E_eq_r2` with a non-zero formulation.

**EVIDENCE:** Lines 313–314 and 444–445 of `_bv_common.py`. Script-level grep
confirms most callers omit these kwargs. The `test_per_reaction_eeq.py` test
exercises both zero and physical E_eq in a comparison, confirming the factory
correctly routes the value when provided — the problem is purely the default.

---

## Issue 3 — NOTE: `conc_floor` mismatch between factory and config fallback

**SEVERITY:** note

**LOCATION:** `_bv_common.py` line 276 vs `config.py` line 113

**DESCRIPTION:**
`_make_bv_convergence_cfg` (the factory helper) writes `"conc_floor": 1e-12`.
`_default_bv_convergence_cfg` in `config.py` (the fallback used when
`params["bv_convergence"]` is absent) writes `"conc_floor": 1e-8`. Likewise,
`_get_bv_convergence_cfg` in `config.py` (the active parser) defaults to `1e-8`
(line 132: `float(raw.get("conc_floor", 1e-8))`).

Since any call to `make_bv_solver_params` explicitly writes `"conc_floor":
1e-12` into the params dict, the parser reads it back correctly and uses 1e-12.
So for any script using the factory, the effective value is 1e-12 as the factory
intends. The discrepancy only matters if `params["bv_convergence"]` is
constructed manually or falls back to the config default (e.g. in unit tests
that pass a minimal params dict). There is no operational impact on production
runs from the factory path, but the internal inconsistency is worth aligning.

**EVIDENCE:** `_bv_common.py` line 276; `config.py` lines 113 and 132.

---

## Correctness Arguments for All Key Requirements

### 1. Flag wiring — CORRECT

`make_bv_solver_params` (lines 515–560) routes all production flags through
`_make_bv_convergence_cfg` and `_make_bv_bc_cfg`, which write directly into
`params["bv_convergence"]` and `params["bv_bc"]` respectively. The SolverParams
object is built with `params` at index 10 (line 548–560).

| Flag | Written by | Written to | Key |
|---|---|---|---|
| `formulation="logc_muh"` | `_make_bv_convergence_cfg` line 280 | `params["bv_convergence"]["formulation"]` | `"logc_muh"` |
| `log_rate=True` | `_make_bv_convergence_cfg` line 278 | `params["bv_convergence"]["bv_log_rate"]` | `True` |
| `boltzmann_counterions=[...]` | `_make_bv_bc_cfg` line 398 | `params["bv_bc"]["boltzmann_counterions"]` | list of dicts |
| `stern_capacitance_f_m2=0.10` | `_make_bv_bc_cfg` line 400 | `params["bv_bc"]["stern_capacitance_f_m2"]` | `0.1` |
| `initializer="debye_boltzmann"` | `_make_bv_convergence_cfg` line 281 | `params["bv_convergence"]["initializer"]` | `"debye_boltzmann"` |

`config.py:_get_bv_convergence_cfg` (lines 124–157) reads these exact keys with
correct types. `config.py:_get_bv_boltzmann_counterions_cfg` (lines 164–263)
reads `params["bv_bc"]["boltzmann_counterions"]`. `config.py:_get_bv_cfg`
(lines 14–62) reads `stern_capacitance_f_m2`. Wiring is intact end-to-end.

### 2. `exponent_clip` default — CORRECT (clip=100)

`_make_bv_convergence_cfg` line 274 sets `"exponent_clip": 100.0` with an
inline comment citing `docs/clipping_conventions.md` and the 2026-05-04 raise.
`config.py:_get_bv_convergence_cfg` line 131 also defaults to `100.0`. Neither
path defaults to 50. The old clip=50 is gone from the production code path.

### 3. `THREE_SPECIES_LOGC_BOLTZMANN` species preset — CORRECT

- `n_species=3`: O2 (idx 0), H2O2 (idx 1), H+ (idx 2). ClO4- is excluded from
  the dynamic NP system. Correct.
- `z_vals=[0, 0, 1]`: O2 neutral, H2O2 neutral, H+ charge +1. Correct.
- Diffusivities: `[D_O2_HAT, D_H2O2_HAT, D_HP_HAT]` in the correct species
  order. Correct.
- `c0_vals_hat=[C_O2_HAT, H2O2_SEED_NONDIM, C_HP_HAT]`: O2 at 1.0
  (= 0.5/0.5), H2O2 seeded at 1e-4 (finite seed for log-c), H+ at 0.2
  (= 0.1/0.5). Correct.
- The 4th species (ClO4-) is intentionally absent and is handled analytically
  via `boltzmann_counterions`, consistent with CLAUDE.md.

### 4. `DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC` — CORRECT (aside from phi_clamp noted above)

- `z=-1`: Correct for ClO4-.
- `c_bulk_nondim=C_CLO4_HAT=0.2`: Correct (0.1 mol/m³ / 0.5 mol/m³ scale).
- `steric_mode='bikerman'`: Present and correct (line 422).
- `a_nondim=A_DEFAULT=0.01`: Present and required for bikerman (config.py raises
  if missing, line 242–245). Correct.
- `phi_clamp=50.0` (inherited from ideal counterion): See Issue 1 above.
- Bikerman theta_b consistency: `theta_b = 1 - A_dyn_bulk - a_b * c_b =
  1 - 0.012001 - 0.01*0.2 = 0.986 >> 0`. The Bikerman closure is well-posed
  and `boltzmann.py:build_steric_boltzmann_expressions` (line 183) validates
  this at build time. No cold-fail risk from theta_b.
- Double-counting guard: `boltzmann.py` lines 192–203 explicitly checks that
  the bikerman counterion does not duplicate any dynamic species by (z, c_bulk).
  ClO4- has z=-1, but no dynamic species has z=-1 in `THREE_SPECIES_LOGC_BOLTZMANN`
  (z_vals=[0,0,1]), so no double-count can occur.

### 5. Reaction stoichiometry — CORRECT

Species ordering: idx 0 = O2, idx 1 = H2O2, idx 2 = H+.

R1 (`stoichiometry_r1 = [-1, +1, -2]`):
- O2 consumed (−1), H2O2 produced (+1), H+ consumed (−2 per 2-electron step).
- `cathodic_species=0` (O2), `anodic_species=1` (H2O2). Correct.
- Reaction: O2 + 2H+ + 2e- → H2O2.

R2 (`stoichiometry_r2 = [0, -1, -2]`):
- O2 inert (0), H2O2 consumed (−1), H+ consumed (−2 per 2-electron step).
- `cathodic_species=1` (H2O2), `anodic_species=None` (irreversible). Correct.
- Reaction: H2O2 + 2H+ + 2e- → 2H2O (water product not tracked dynamically).

There are no off-by-one index errors. The stoichiometry arrays have length
`n_species=3`, which `config.py:_get_bv_reactions_cfg` line 293 validates
explicitly (`len(stoi) != n_species` raises).

H+ concentration factor (`cathodic_conc_factors`): species index 2 (H+) is
correctly identified (line 381), power=2, c_ref_nondim=C_HP_HAT=0.2. Applied
to both R1 and R2. Correct.

### 6. No deprecated callsites in `_bv_common.py` — CORRECT

`_bv_common.py` itself contains no calls to `add_boltzmann(ctx)` or to
`solve_grid_with_charge_continuation`. The factory correctly encodes
`boltzmann_counterions` into `params["bv_bc"]` so the dispatcher and forms
modules pick it up at build time — not as a post-hoc mutation. No double-count
risk within the factory itself.

**Caveat:** Several legacy study scripts (`v18_logc_lsq_inverse.py`,
`plot_iv_curves_3sp_true.py`, `v19_bv_clip_audit.py`, etc.) continue to use the
inline `add_boltzmann(ctx)` pattern. These scripts build `params["bv_bc"]`
manually *without* a `"boltzmann_counterions"` key, then call `add_boltzmann`
post-hoc. Since `boltzmann_counterions` is absent from their bv_bc block, the
config parser returns an empty list and the post-hoc mutation does not
double-count. The CLAUDE.md footgun applies only if a script both sets
`boltzmann_counterions` in params AND calls `add_boltzmann` — that combination
does not appear in any current script. However, these legacy scripts retain
a `phi_clamp=50` hardcode inside their local `add_boltzmann` helper, which is
consistent with their pre-raise era but inconsistent with the current production
default. They are marked non-operational in CLAUDE.md (v13–v24 study scripts).

### 7. Backward compatibility — CONDITIONALLY CORRECT

`FOUR_SPECIES_LOGC_DYNAMIC` is defined (lines 221–233) and is the correct 4sp
dynamic preset. `C_HP_HAT == C_CLO4_HAT == 0.2`, so bulk electroneutrality
holds (`z=+1` H+ at 0.2, `z=-1` ClO4- at 0.2). The stoichiometry correctly
marks ClO4- (idx 3) as inert in both R1 and R2.

The scope document mentioned `FOUR_SPECIES_LEGACY` — this name does not exist
in `_bv_common.py`. The only 4-species preset is `FOUR_SPECIES_LOGC_DYNAMIC`.
CLAUDE.md describes the "legacy 4-species concentration path" as backward compat
for v13/v15/v16, but those scripts are non-operational (marked as "historical
reference only"). The concentration backend itself was removed in the May 2026
cleanup. The 4sp dynamic preset is a *validation reference*, not the legacy
concentration-formulation path. No correctness issue — the naming differs from
the scope document's terminology.

`formulation` default in `make_bv_solver_params` is `"logc"` (line 449), not
`"concentration"`. This means new code that calls the factory without specifying
`formulation` gets log-c, not concentration. The concentration backend is
removed, so this is correct. Old scripts that relied on `formulation=
"concentration"` via the factory would receive validation errors from
`config.py:_validate_formulation` — but those scripts are marked non-operational
in CLAUDE.md.

---

## Summary Table

| Check | Result | Severity |
|---|---|---|
| `exponent_clip` default = 100 | PASS | — |
| `formulation` wired to `bv_convergence` | PASS | — |
| `log_rate` wired to `bv_log_rate` | PASS | — |
| `boltzmann_counterions` wired to `bv_bc` | PASS | — |
| `stern_capacitance_f_m2` wired to `bv_bc` | PASS | — |
| `initializer` wired to `bv_convergence` | PASS | — |
| Species z_vals correct (O2=0, H2O2=0, H+=+1) | PASS | — |
| ClO4- z=-1 in counterion dicts | PASS | — |
| ClO4- `steric_mode='bikerman'` in STERIC preset | PASS | — |
| `theta_b > 0` for bikerman closure | PASS (0.986) | — |
| Double-count guard in boltzmann.py | PASS | — |
| R1 stoichiometry indices | PASS | — |
| R2 stoichiometry indices | PASS | — |
| No deprecated Strategy B in factory | PASS | — |
| No inline `add_boltzmann` + `bv_bc.boltzmann_counterions` double-count | PASS | — |
| 4sp electroneutrality (C_HP_HAT == C_CLO4_HAT) | PASS | — |
| `phi_clamp=50` in ideal/steric counterion presets | FAIL (inconsistent) | WARNING |
| E_eq defaults = 0.0 (not physical 0.68/1.78 V) | FAIL (missing constants) | WARNING |
| `conc_floor` mismatch factory vs config fallback | FAIL (minor) | NOTE |
