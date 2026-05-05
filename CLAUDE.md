# CLAUDE.md — PNPInverse

Short operational guide for Claude when working in this repo. The
`README.md` has the research narrative and the full architecture; this
file is the project-specific conventions, invariants, and lessons that
are easy to get wrong. Read both.

## What this repo is

Research code for Poisson–Nernst–Planck / Butler–Volmer (PNP-BV) forward
simulation and inverse kinetic inference for a two-step ORR
(O₂ → H₂O₂ → H₂O) at pH 4. Targets are
`[log_k0_1, log_k0_2, alpha_1, alpha_2]`.

There are two coexisting production stacks (May 2026):

**Stable production** (used by the v13–v24 inverse scripts and the
direct-PDE FIM work):

- **3 dynamic species** (O₂, H₂O₂, H⁺) +
- **analytic Boltzmann counterion** for ClO₄⁻ (`steric_mode='ideal'`) +
- **log-concentration primary variables** (`u_i = ln c_i`,
  `formulation='logc'`) +
- **log-rate Butler–Volmer**.

Solver window V_RHE ∈ [−0.5, +0.6] V via the C+D orchestrator.

**New production target (Option 2b, 2026-05-04, 15/15 V_RHE [−0.5, +1.0]):**

- 3 dynamic species (O₂, H₂O₂, H⁺) +
- analytic Bikerman counterion for ClO₄⁻
  (`steric_mode='bikerman'`, residual-side closure plus
  Bikerman-consistent IC) +
- proton-electrochemical-potential primary variable for H⁺
  (`formulation='logc_muh'`, with `mu_H = u_H + em*z_H*phi` to keep
  Newton smooth in deep-ψ regions) +
- log-rate Butler–Volmer +
- finite Stern layer (`stern_capacitance_f_m2 ≈ 0.10`) +
- `initializer='debye_boltzmann'` IC (composite-ψ + multispecies-γ).

See `docs/4sp_bikerman_ic_option_2b_results.md` for the headline sweep
result; the inverse pipeline has not yet been migrated onto this stack.

Both stacks share the same dispatcher
(`Forward.bv_solver.{build_context, build_forms, set_initial_conditions}`)
selected through `bv_convergence.{formulation, initializer}` and
`bv_bc.{boltzmann_counterions[*].steric_mode, stern_capacitance_f_m2}`.
Defaults in `make_bv_solver_params` keep the legacy 4-species
concentration path for backward compatibility — v13/v15/v16 scripts
still rely on those defaults.

## Source-of-truth docs (read before opining on status)

| File | Purpose |
|---|---|
| `writeups/WeekOfApr27/PNP Inverse Solver Revised.pdf` | Forward-solver rebuild narrative |
| `docs/bv_solver_unified_api.md` | How to call the production stack |
| `docs/CONTINUATION_STRATEGY_HANDOFF.md` | Why C+D over A/B |
| `docs/CHATGPT_HANDOFF_10_LM_TIKHONOV_BASIN_GEOMETRY.md` | Current single-experiment inverse status |
| `docs/Next Steps After Basin Geometry.md` | The current next path (multi-experiment FIM) |
| `docs/noise_model_conventions.md` | FIM/inverse noise-model convention |
| `docs/clipping_conventions.md` | What log-rate did and didn't change about clipping; the +0.495 V R2 unclip threshold (relevant when `exponent_clip=50`) |
| `docs/4sp_bikerman_ic_option_2b_results.md` | New production target: bikerman + Stern + muh + debye_boltzmann IC = 15/15 V_RHE [−0.5, +1.0] |
| `docs/4sp_drop_boltzmann_investigation.md` | §11–§13: full investigation log behind the 2b fix (sign correction, IC γ, composite-ψ) |
| `docs/steric_analytic_clo4_reduction_handoff.md` | Derivation of the Bikerman analytic-counterion residual closure |
| `docs/Mangan2025_experimental_alignment.md` | Gap audit between current ClO₄⁻/3sp setup and the Mangan 2025 deck (sulfate / Cs⁺ / RRDE / IrOx) |
| `docs/forward_solver_test_coverage.md` | What the bv_solver test suite covers (and what it doesn't) |

Older `CHATGPT_HANDOFF_*` files are useful chronology, but newer docs
above supersede earlier optimism that the single-experiment inverse was
fully fixed.

## Environment

- **Activate `../venv-firedrake/bin/activate` from `PNPInverse/`.**
  Conda envs (e.g. `FireDrakeEnv`) are not the right ones for this
  project — they will not run Firedrake correctly here.
- Firedrake is installed separately into that venv; do not
  `pip install firedrake`.
- Useful cache settings before running PDE work:

  ```bash
  export MPLCONFIGDIR=/tmp
  export XDG_CACHE_HOME=/tmp
  export PYOP2_CACHE_DIR=/tmp/pyop2
  export FIREDRAKE_TSFC_KERNEL_CACHE_DIR=/tmp/firedrake-tsfc
  export OMP_NUM_THREADS=1
  ```

- Tests: `pytest -m "not slow"` for fast unit tests; `pytest -m slow`
  requires Firedrake.
- **Always stream test output to the console.** When writing or
  running tests, use flags that surface progress in real time
  (`pytest -s -vv`, `--log-cli-level=INFO`, `python -u`, and avoid
  buffering in `subprocess`/`tee` wrappers). Tests in this repo can
  stall for minutes inside a single Firedrake solve; without streamed
  output the user has no visibility into where it's hanging.

## Hard rules (lessons that cost real time)

1. **Always use adjoint gradients for inference.** Do not switch to
   derivative-free (Nelder–Mead, etc.) as a workaround when an
   adjoint-based optimizer fails. The fix is forward-solver robustness
   at perturbed parameters, not a different optimizer. Each
   derivative-free eval is a full charge-continuation solve (~20 s); it
   doesn't scale.

2. **Verify adjoints with cold-ramp FD, not warm-start FD**, especially
   near the R2 unclip threshold (V_RHE ∈ [+0.30, +0.50] V). Warm-start
   FD from a TRUE-parameter cache lands the perturbed solve in a
   metastable basin with exactly half the slope at small `h`. Use the
   `--fd-cold-ramp` flag in
   `scripts/studies/v19_lograte_extended_adjoint_check.py`. An
   "adjoint FAIL" based on warm-start FD is inconclusive until cold-ramp
   FD is run.

3. **Use C+D, not Strategy B, for the logc+Boltzmann stack.** That is:
   `Forward.bv_solver.solve_grid_per_voltage_cold_with_warm_fallback`,
   not `solve_grid_with_charge_continuation`. B converged 3/13 at
   production resolution (V26, 2026-05-01) because its Phase-1 V-sweep
   at z=0 hands bisection a mismatched species IC it can't recover from.
   B remains correct for the legacy 4sp concentration path only.

4. **The `eta_scaled` clip threshold depends on `exponent_clip`.** Code
   convention is `eta_scaled = (V_RHE − E_eq)/V_T`, clipped to
   ±`exponent_clip` *before* the α·n_e multiplication (see
   `forms_logc.py:_build_eta_clipped`).
   - At the **historical default `exponent_clip=50`** (used by V19–V24
     inverse runs): R2 unclips at V_RHE > +0.495 V.
   - At the **current default `exponent_clip=100`** (raised 2026-05-04
     because clip=50 sign-flips PC at V_RHE < −0.1 V — see
     `clip_observable_investigation.md` §5.2): R2 is unclipped over the
     entire production grid V_RHE ∈ [−0.5, +1.0] V; the formal R2
     unclip threshold is V_RHE > −0.79 V.
   Older handoffs (e.g. `docs/PNP Anodic Solver Handoff.md`) say
   ~+1.14 V — that's wrong even at clip=50 (it missed the α factor).
   Don't repeat the +1.14 V number without re-checking the clip.
   **Log-rate did NOT remove this clip**: it's structural (the final
   `exp(α·n_e·η)` would otherwise overflow). What log-rate eliminated
   is a separate `c_surf = exp(clamp(u, ±30))` clamp inside the BV
   residual; the corresponding default `u_clamp` is also 100 now.
   Full breakdown of the three distinct clips in
   `docs/clipping_conventions.md`.

5. **Default FIM noise model is `local_rel + abs_floor`**
   (`c_rel = 0.02`, defensible `sigma_abs` from instrument spec).
   Always also report `global_max` and pure `local_rel` as comparison.
   PC spans ~8 decades over V_RHE ∈ [−0.5, +0.6]; under `global_max`,
   the biggest-|y| voltage demotes everything else and produces
   artifact "weak-direction flips" that coincide with **degraded**
   `cond(F)`. Real ID gains require both rotation **and** non-degraded
   `cond(F)`. See `docs/noise_model_conventions.md`.

6. **Don't claim single-experiment four-parameter recovery is solved.**
   Current truthful statement: log-rate BV removed the old clipped-R2
   local Fisher singularity; α₁ and α₂ are robustly data-identifiable;
   joint (k0_1, k0_2) recovery from one CD+PC experiment is
   initialization-dependent and basin-bound. Tikhonov priors at
   literature-defensible `σ = log(3)` or `log(10)` shift within basins
   but do not cross them (a basin-crossing prior would need
   `σ ≲ log(1.15)` — physically indefensible). Next path is
   multi-experiment FIM screening — see
   `docs/Next Steps After Basin Geometry.md`.

7. **PC saturates at the mass-transport limit for V_RHE ≤ −0.30.**
   Negative-voltage grid extension alone (V23) gave WEAK PASS only
   (`cond(F)` ~2.5× better, weak direction still `log_k0_1`). Don't
   pitch grid extension as the route to break the weak direction.

8. **Solver convergence window is V_RHE ∈ [−0.5, +0.1] V at full z=1**
   for the legacy charge-continuation route; the production C+D
   orchestrator additionally extends to +0.6 V via warm-walk fallback
   on the stable production stack. With the **new production target
   (bikerman + muh + Stern 0.10 F/m² + debye_boltzmann IC)**, C+D
   reaches V_RHE = +1.0 V at 15/15 (cold ceiling +0.60 V; warm-walk to
   +1.00 V — see `docs/4sp_bikerman_ic_option_2b_results.md`). Plan
   voltage grids accordingly. Use physical `E_eq` (R1 = 0.68 V,
   R2 = 1.78 V vs RHE), never `E_eq = 0`.

9. **The IC and the residual must agree about steric saturation** for
   the bikerman closure to converge. `set_initial_conditions_debye_boltzmann_*`
   seeds composite-ψ + multispecies-γ on the analytic-bikerman path;
   the residual side picks up `build_steric_boltzmann_expressions`
   (in `Forward/bv_solver/boltzmann.py`) so the dynamic-species
   packing fraction *and* the Poisson source agree on the saturated
   counterion concentration. A bikerman IC without the matching
   residual (or vice-versa) cold-fails on the saturated manifold —
   that mismatch was the binding constraint behind §12's Option 2a
   detour. Don't reintroduce one half without the other. See
   `docs/steric_analytic_clo4_reduction_handoff.md`.

10. **The 4sp dynamic stack ceiling is unchanged by the 2b IC fix.**
    Both `formulation='logc'` and `formulation='logc_muh'` 4sp
    sweeps top out at the same 5/15 (no Stern) and 7/15 (Stern)
    counts as 2a′. The binding constraint at 4sp is the dynamic
    c_ClO₄ NP equation, not the IC; the 3sp + analytic-bikerman
    stack is the validation reference, and the 4sp dynamic stack
    is now an equivalence-test target (cathodic overlap agrees to
    ~10⁻⁹; +0.5 V edge to ~5·10⁻³). Don't pitch "go fully dynamic"
    as a fix for the anodic ceiling.

## Calling the production solver

Use the canonical factory + dispatcher; don't reinvent the flag wiring,
and don't add the inline `add_boltzmann(ctx)` helper while *also* setting
`bv_bc.boltzmann_counterions` — that double-counts the counterion.

**Stable production** (used by inverse scripts; ideal Boltzmann, no
Stern, linear IC):

```python
from scripts._bv_common import (
    make_bv_solver_params,
    THREE_SPECIES_LOGC_BOLTZMANN,
    DEFAULT_CLO4_BOLTZMANN_COUNTERION,
)
from Forward.bv_solver import solve_grid_per_voltage_cold_with_warm_fallback

sp = make_bv_solver_params(
    ...,
    species=THREE_SPECIES_LOGC_BOLTZMANN,
    formulation="logc",
    log_rate=True,
    boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION],
)
```

**New production target** (bikerman closure + Stern + muh + debye_boltzmann
IC; reaches V_RHE = +1.0 V):

```python
from scripts._bv_common import (
    make_bv_solver_params,
    THREE_SPECIES_LOGC_BOLTZMANN,
    DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC,   # steric_mode='bikerman'
)
from Forward.bv_solver import solve_grid_per_voltage_cold_with_warm_fallback

sp = make_bv_solver_params(
    ...,
    species=THREE_SPECIES_LOGC_BOLTZMANN,
    formulation="logc_muh",                      # mu_H = u_H + em*z_H*phi
    log_rate=True,
    boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC],
    stern_capacitance_f_m2=0.10,                 # 10 µF/cm² compact layer
    initializer="debye_boltzmann",               # composite-ψ + multispecies-γ
)
```

Reference driver: `scripts/studies/peroxide_window_3sp_bikerman_muh.py`.

Other unified-API gotchas (full list in
`docs/bv_solver_unified_api.md`):

- `set_initial_conditions(ctx, sp, blob=True)` is **silently ignored**
  in log-c mode (no blob IC for `u_i = ln c_i`).
- `validate_solution_state` needs `is_logc=...` when called on a log-c
  context. Orchestrators pass it for you; direct callers should set
  `is_logc=ctx.get("logc_transform", False)`. For the muh formulation
  also pass `mu_species=ctx.get('mu_species')` and
  `em=ctx['nondim'].get('electromigration_prefactor', 1.0)` so
  per-dof `log(c_i)` is recovered from `mu_i − em·z_i·phi` before
  the physics checks run.
- `H2O2_SEED_NONDIM = 1e-4` is the finite seed for `ln c_H2O2` at the
  bulk Dirichlet BC, not a physics tweak.
- The `debye_boltzmann` IC requires either a `synthesised_4sp`
  ClO₄⁻ counterion *or* a `boltzmann_counterions=[…]` entry with
  `steric_mode='bikerman'`. With `steric_mode='ideal'` the legacy
  tanh-Gouy-Chapman seed is used (byte-identical to pre-2b).

## Path conventions

- Run scripts from `PNPInverse/` (the directory containing this file),
  not from `Forward/` or `scripts/`.
- `scripts/Inference/` is **uppercase**. There is no
  `scripts/inference/` and no `scripts/bv/`. Older notes may reference
  paths that no longer exist.
- `StudyResults/` is part of the working research record, not a clean
  build-artifact directory. Check existing `summary.md` files before
  regenerating expensive studies.
- `archive/` is reference-only — not the active implementation surface.

## Workflow notes for this codebase

- **Plan before non-trivial forward-solver changes.** The legacy
  concentration `forms.py` was removed in the May 2026 cleanup (commit
  `e72163d`); the live backends are `Forward/bv_solver/forms_logc.py`
  (production) and `forms_logc_muh.py` (experimental muh variant).
  Edits to either can silently break the inverse pipeline through the
  dispatcher. Decide which validation scripts you'll run (`v24`,
  `v25`, the MMS scripts in `scripts/verification/`, and
  `scripts/studies/peroxide_window_3sp_bikerman_muh.py` for the
  bikerman+Stern+muh stack) before implementing.
- **Long-running studies cost minutes-to-hours.** Confirm with the
  user before regenerating expensive runs. Prefer `--aggregate-only`,
  `--skip-solves`, or cached extended-JSON modes when scripts expose
  them.
- **Adjoint tape hygiene:** wrap unannotated cold-ramp / continuation
  work in `with adj.stop_annotating():`, and only annotate the SS
  steps you actually want on the tape.
