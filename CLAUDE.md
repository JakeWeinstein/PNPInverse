# CLAUDE.md — PNPInverse

Short operational guide for Claude when working in this repo. The
`README.md` has the research narrative and the full architecture; this
file is the project-specific conventions, invariants, and lessons that
are easy to get wrong. Read both.

## What this repo is

Research code for Poisson–Nernst–Planck / Butler–Volmer (PNP-BV) forward
simulation and (eventually) inverse kinetic inference for a two-step
ORR (O₂ → H₂O₂ → H₂O) at pH 4.

The production forward stack (May 2026) is:

- **3 dynamic species** (O₂, H₂O₂, H⁺) +
- **analytic Bikerman counterion** for ClO₄⁻
  (`steric_mode='bikerman'`, residual-side closure plus
  Bikerman-consistent IC) +
- **proton electrochemical-potential primary variable**
  (`formulation='logc_muh'`, `mu_H = u_H + em·z_H·phi`) +
- **log-rate Butler–Volmer** +
- **finite Stern compact layer** (`stern_capacitance_f_m2 ≈ 0.10`) +
- **`debye_boltzmann` IC** (composite-ψ + multispecies-γ).

It reaches V_RHE = +1.0 V at 15/15 (cold ceiling +0.60 V; warm-walk to
+1.00 V) on the C+D orchestrator. Defaults in `make_bv_solver_params`
keep the legacy 4-species concentration path for backward compat —
v13/v15/v16 scripts still rely on those defaults — but new work
should opt in to the production stack via the flags above.

## Inverse status: paused

All inverse scripts in this repo are **legacy / non-operational**. No
inverse work is currently running. Held until the forward solver is
mature enough for a clean re-entry. When it resumes, start from
`docs/CHATGPT_HANDOFF_10_LM_TIKHONOV_BASIN_GEOMETRY.md`,
`docs/Next Steps After Basin Geometry.md`, and
`docs/noise_model_conventions.md`; treat the v13–v24 study scripts
(`scripts/studies/v*.py`) as historical reference only.

## Source-of-truth docs (read before opining on status)

| File | Purpose |
|---|---|
| `writeups/WeekOfApr27/PNP Inverse Solver Revised.pdf` | Forward-solver rebuild narrative |
| `writeups/ForwardSolverChangesMay26/forward_solver_changes_may2026.pdf` | May 2026 production-target writeup |
| `docs/bv_solver_unified_api.md` | How to call the production stack |
| `docs/4sp_bikerman_ic_option_2b_results.md` | Production-target reference sweep (15/15 V_RHE [−0.5, +1.0]) |
| `docs/4sp_drop_boltzmann_investigation.md` | §11–§13: the investigation log behind the production target |
| `docs/steric_analytic_clo4_reduction_handoff.md` | Derivation of the Bikerman analytic-counterion residual closure |
| `docs/clipping_conventions.md` | The three distinct BV-related clips and the `exponent_clip` 50 → 100 raise |
| `docs/Mangan2025_experimental_alignment.md` | Gap audit between the model and the Mangan 2025 deck |
| `docs/Ruggiero2022_JCatal_source_paper.md` | Peer-reviewed source paper underlying the Mangan deck (Mangan is co-author). Load-bearing experimental constants (sulfate not perchlorate, N=0.224, 1600 rpm, I=0.3 M) and the structural finding that the deck uses parallel 2e/4e ORR, not sequential R₀/R₁. PDF at `docs/Ruggiero2022_JCatal_manuscript.pdf` |
| `docs/seitz_mangan_data_folder_audit_2026-05-08.md` | Deep audit of the real experimental data drop at `data/EChem Reactor Modeling-Seitz-Mangan/`. Multi-document (2019→2025) confirmation: K₂SO₄ not ClO₄⁻, parallel 2e⁻ (0.67 V) + 4e⁻ (1.23 V) not sequential R₀+R₁, pH 1–6 sweep. Catalogues raw RRDE LSV / CP datasets, Yash's parallel 6-species PNP+BV reference code, and the missing `Tafel slope analysis cation-pH-Li-K-Cs.xlsx` to request |
| `docs/CONTINUATION_STRATEGY_HANDOFF.md` | Why C+D over A/B for the logc+counterion stack |
| `docs/forward_solver_test_coverage.md` | What the bv_solver test suite covers |

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
- **Always stream test output to the console** (`pytest -s -vv`,
  `--log-cli-level=INFO`, `python -u`, no buffering wrappers). Tests
  here can stall for minutes inside a single Firedrake solve.

## Hard rules (lessons that cost real time)

1. **Use C+D, not Strategy B**: call
   `Forward.bv_solver.solve_grid_per_voltage_cold_with_warm_fallback`,
   not `solve_grid_with_charge_continuation`. B's Phase-1 V-sweep at
   z=0 hands bisection a mismatched species IC it can't recover from
   on the logc+counterion stack (3/13 at production resolution).

2. **`exponent_clip` choice — clip=50 PC is fictitious; clip=100 is
   the only PC-trustworthy setting.** Convention:
   `eta_scaled = (V_RHE − E_eq)/V_T` clipped to ±`exponent_clip`
   *before* the α·n_e multiplication
   (`forms_logc.py:_build_eta_clipped`).
   - **clip=50** (historical default; V19–V24 runs): R2 unclips at
     V_RHE > +0.495 V. **Do not trust PC** — clipped R2 produces a
     fictitious peroxide current (sign-flipped at V_RHE < −0.1 V;
     magnitude artefacted across the cathodic grid). Don't compare
     clip=50 PC against experiment. See
     `docs/clip_observable_investigation.md` §5.2.
   - **clip=100** (current default, raised 2026-05-04): R2 unclips
     at V_RHE > −0.79 V — production grid fully unclipped, and the
     **only configuration where negative-voltage PC is trustworthy**.
     Some configs cold-fail more often here than at clip=50
     (no-Stern bikerman near V_RHE ≈ +0.1 V); recover with C+D
     warm-walk or Stern, not by lowering the clip.
   Older handoffs say ~+1.14 V — wrong (missed the α factor); ignore.
   Log-rate did *not* remove this clip (structural for
   `exp(α·n_e·η)`); it eliminated a *separate* `c_surf =
   exp(clamp(u, ±30))` clamp inside the BV residual. `u_clamp`
   default is also 100 now. Full breakdown in
   `docs/clipping_conventions.md`.

3. **The IC and the residual must agree about steric saturation.**
   `set_initial_conditions_debye_boltzmann_*` seeds composite-ψ +
   multispecies-γ; the residual side picks up
   `build_steric_boltzmann_expressions` so the dynamic-species
   packing fraction *and* the Poisson source agree on the saturated
   counterion concentration. A bikerman IC without the matching
   residual (or vice-versa) cold-fails on the saturated manifold.

4. **Solver convergence window with the production stack** is
   V_RHE ∈ [−0.5, +1.0] V via C+D (cold ceiling +0.60 V; warm-walk
   to +1.00 V — see
   `docs/4sp_bikerman_ic_option_2b_results.md`). Use physical `E_eq`
   (R1 = 0.68 V, R2 = 1.78 V vs RHE), never `E_eq = 0`.

5. **The 4sp dynamic stack ceiling is unchanged by the IC fix.**
   Both `formulation='logc'` and `'logc_muh'` 4sp sweeps top out at
   5/15 (no Stern) and 7/15 (Stern). Binding constraint is the
   dynamic c_ClO₄ NP equation, not the IC. The 4sp dynamic stack is
   a validation reference (cathodic overlap agrees to ~10⁻⁹;
   +0.5 V edge to ~5·10⁻³); don't pitch "go fully dynamic" as a fix
   for the anodic ceiling.

## Calling the production solver

Use the canonical factory + dispatcher; don't reinvent the flag wiring,
and don't add the inline `add_boltzmann(ctx)` helper while *also*
setting `bv_bc.boltzmann_counterions` — that double-counts the
counterion.

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
    formulation="logc_muh",
    log_rate=True,
    boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC],
    stern_capacitance_f_m2=0.10,
    initializer="debye_boltzmann",
)
```

Reference driver:
`scripts/studies/peroxide_window_3sp_bikerman_muh.py`. Full API in
`docs/bv_solver_unified_api.md`. Gotchas:

- `set_initial_conditions(ctx, sp, blob=True)` is silently ignored in
  log-c mode (no blob IC for `u_i = ln c_i`).
- `validate_solution_state` needs `is_logc=...` for log-c contexts; on
  the muh formulation also pass `mu_species=ctx.get('mu_species')`
  and `em=ctx['nondim'].get('electromigration_prefactor', 1.0)`.
- The `debye_boltzmann` IC requires either a `synthesised_4sp` ClO₄⁻
  counterion *or* a `steric_mode='bikerman'` entry; with
  `steric_mode='ideal'` it falls back to the tanh-Gouy-Chapman seed.
- `H2O2_SEED_NONDIM = 1e-4` is the finite seed for `ln c_H2O2` at the
  bulk Dirichlet BC, not a physics tweak.

## Path conventions

- Run scripts from `PNPInverse/` (the directory containing this file),
  not from `Forward/` or `scripts/`.
- `scripts/Inference/` is **uppercase** — no `scripts/inference/`
  or `scripts/bv/` exists.
- `scripts/studies/v*.py` are legacy inverse scripts; not operational.
- `StudyResults/` is part of the working research record, not a clean
  build-artifact directory. Check existing `summary.md` files before
  regenerating expensive studies.
- `archive/` is reference-only — not the active implementation surface.

## Workflow notes

- **Plan before non-trivial forward-solver changes.** Live backends
  are `forms_logc.py` and `forms_logc_muh.py` (concentration backend
  removed in the May 2026 cleanup). Decide which validation scripts
  you'll run (`scripts/verification/` MMS scripts,
  `scripts/studies/peroxide_window_3sp_bikerman_muh.py`) before
  implementing.
- **Long-running studies cost minutes-to-hours.** Confirm with the
  user before regenerating expensive runs.
- **Adjoint tape hygiene** (when inverse work resumes): wrap
  unannotated cold-ramp / continuation work in
  `with adj.stop_annotating():` and only annotate the SS steps you
  want on the tape.
