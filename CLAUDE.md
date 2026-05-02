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

The active production forward stack (May 2026) is:

- **3 dynamic species** (O₂, H₂O₂, H⁺) +
- **analytic Boltzmann counterion** for ClO₄⁻ +
- **log-concentration primary variables** (`u_i = ln c_i`) +
- **log-rate Butler–Volmer**.

It is selected through three config flags via a single dispatcher.
Defaults remain the legacy 4-species concentration path for backward
compatibility — v13/v15/v16 scripts still rely on those defaults.

## Source-of-truth docs (read before opining on status)

| File | Purpose |
|---|---|
| `writeups/WeekOfApr27/PNP Inverse Solver Revised.pdf` | Forward-solver rebuild narrative |
| `docs/bv_solver_unified_api.md` | How to call the production stack |
| `docs/CONTINUATION_STRATEGY_HANDOFF.md` | Why C+D over A/B |
| `docs/CHATGPT_HANDOFF_10_LM_TIKHONOV_BASIN_GEOMETRY.md` | Current single-experiment inverse status |
| `docs/Next Steps After Basin Geometry.md` | The current next path (multi-experiment FIM) |
| `docs/noise_model_conventions.md` | FIM/inverse noise-model convention |

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

4. **R2 unclips at V_RHE > +0.495 V**, under the actual code convention
   `eta_scaled = (V_RHE − E_eq)/V_T` clipped to ±50 *before* the α·n_e
   multiplication (see `forms_logc.py:_build_eta_clipped`). Older
   handoffs (e.g. `docs/PNP Anodic Solver Handoff.md`) say ~+1.14 V —
   that's wrong (it missed the α factor). Don't repeat the +1.14 V
   number without re-checking the clip.

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
   orchestrator additionally extends to +0.6 V via warm-walk fallback.
   Plan voltage grids accordingly. Use physical `E_eq` (R1 = 0.68 V,
   R2 = 1.78 V vs RHE), never `E_eq = 0`.

## Calling the production solver

Use the canonical factory + dispatcher; don't reinvent the flag wiring,
and don't add the inline `add_boltzmann(ctx)` helper while *also* setting
`bv_bc.boltzmann_counterions` — that double-counts the counterion.

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

Other unified-API gotchas (full list in
`docs/bv_solver_unified_api.md`):

- `set_initial_conditions(ctx, sp, blob=True)` is **silently ignored**
  in log-c mode (no blob IC for `u_i = ln c_i`).
- `validate_solution_state` needs `is_logc=...` when called on a log-c
  context. Orchestrators pass it for you; direct callers should set
  `is_logc=ctx.get("logc_transform", False)`.
- `H2O2_SEED_NONDIM = 1e-4` is the finite seed for `ln c_H2O2` at the
  bulk Dirichlet BC, not a physics tweak.

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

- **Plan before non-trivial forward-solver changes.** Edits in
  `Forward/bv_solver/forms.py` (legacy) or `forms_logc.py` (production)
  can silently break the inverse pipeline through the dispatcher.
  Decide which validation scripts you'll run (`v24`, `v25`, the MMS
  scripts) before implementing.
- **Long-running studies cost minutes-to-hours.** Confirm with the
  user before regenerating expensive runs. Prefer `--aggregate-only`,
  `--skip-solves`, or cached extended-JSON modes when scripts expose
  them.
- **Adjoint tape hygiene:** wrap unannotated cold-ramp / continuation
  work in `with adj.stop_annotating():`, and only annotate the SS
  steps you actually want on the tape.
