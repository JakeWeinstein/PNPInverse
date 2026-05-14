# Plan: pnpbv Migration (v0.1 → v1.0)

**Goal:** Fork the PNP-BV forward solver from PNPInverse's `Forward/bv_solver/` into a standalone, composable-objects Python package `pnpbv` over 8 milestones, preserving every CLAUDE.md Hard Rule (#1–#7), hitting every SCOPING.md reproduction target, and locking the output-logging schema under semver by v0.6.

**Date:** 2026-05-12
**Source-of-truth:** `pnpbv_port/SCOPING.md` (updated 2026-05-12 with Output Logging Requirements + Documentation Requirements sections)
**Research used:**
- `.research/cmk3-stern-capacitance/SUMMARY.md` — `C_S = 0.20 F/m²` citation chain (Bohra/Choi/CatINT/Kilic-Bazant), per-local-surface-element interpretation, RF ≈ 6000 implicit in fitted k₀
- `.research/pnp-bv-literature-provenance/SUMMARY.md` — Ruggiero 2022 parallel 2e/4e ORR provenance, Singh 2016 σ_S caveats
- `.research/pnp-convergence-strategies/SUMMARY.md` — log-c transform rationale, C+D vs Strategy B vs anchor-continuation tradeoffs

**Discussion context:** None (no prior CONTEXT.md for this migration).

## Approach

**Soft fork, forward-only, byte-equivalence first.** The first milestone (v0.1) ports the *math* — forms, anchor continuation, C+D dispatcher, multi-ion shared-θ, Stern, Bikerman, debye_boltzmann IC — into a new repo with **no API change**. The legacy reference run (`peroxide_window_3sp_bikerman_muh.py`) reproduces byte-equivalent (|Δj|/|j| ≤ 1e-10). Only once the math survives the move do composable objects land at v0.2. This sequencing means every later milestone has a numerical-equivalence anchor against the legacy stack, so refactors can't silently break physics.

**Composable abstractions evolve incrementally, each backed by a reproduction target.** v0.2 (composable objects), v0.3 (multi-ion + parallel BV + continuation strategies), v0.4 (HomogeneousReaction), v0.5 (SurfaceAdsorbate) each correspond to an existing PNPInverse research case that supplies the verification gate. The Phase 6α/6β lessons (CLAUDE.md Hard Rules) survive the migration because the underlying math survives — but each Rule is *also* mapped to a concrete enforcement point in `pnpbv` (see "CLAUDE.md Hard Rules → enforcement points" below).

**Logging schema lands at v0.1, grows additively v0.2–v0.5, freezes at v0.6.** PNPInverse will consume `results.parquet` programmatically from v0.7 onward, so schema stability matters. Additive-only evolution during the build-out + semver lock at v0.6 + a schema-regression test gives downstream consumers a stable contract. Doctest-in-CI starts at v0.2 (first public API example exists); docstring-contract lint runs in warn-mode at v0.1, fail-mode at v0.2.

**No `pyadjoint` in v1, but adjoint-friendly seams enforced from day one.** Immutable `frozen=True` dataclasses for `Species`, `AnalyticCounterion`, `BVReaction`, etc.; no in-place mutation in user-facing code; no untaped Firedrake ops in the hot path. A smoke `pyadjoint` tape-replay on the v0.1 reference case in v0.6 catches any landmines before v2 inverse work resumes.

**v0.4 and v0.5 mechanisms ship as experimental, excluded from the v1.0 stable surface.** Water ionization and cation hydrolysis are still in active research at PNPInverse (Phase 6β Step 10 Phase D verdict is `OUTCOME_C_NON_IDENTIFIABLE_flagged` as of 2026-05-12; Bikerman-`a` bridge runs in flight). Both mechanisms reproduce their legacy verification gates and are usable, but they live under `pnpbv.experimental.*` rather than `pnpbv.*`. Importing them emits an `ExperimentalWarning` on first use per process; their API stability is **not** guaranteed until they graduate post-v1 (graduation criteria: literature-calibrated parameter set + independent reviewer sign-off + Phase 6β identifiability resolved). The v1.0 stable surface is single counterion + multi-ion + parallel BV only — the math from v0.1–v0.3.

**v1.0 builds publication-ready artifacts but does NOT upload to PyPI.** Wheel + sdist build, `twine check dist/*` passes, local install from wheel succeeds in a fresh env, README hello-world runs. The `twine upload` command is documented in `docs/release/PYPI_PUBLISH_COMMAND.md` and ready to execute once the user authorizes publication. The GitHub Release at v1.0 includes the wheel + sdist as downloadable assets so external users can install via `pip install <release-asset-url>` without PyPI in the loop.

## Prerequisites

- **GitHub repo `pnpbv`** created (location TBD — see Open Questions).
- **Firedrake Docker image** identified and pinned (upstream image; specific tag in CI YAML).
- **Legacy paths inventory** confirmed: `Forward/bv_solver/{forms_logc,forms_logc_muh,anchor_continuation,*}.py` + IC/Stern/Bikerman helpers identified by absolute path.
- **Reproduction-target runs** in PNPInverse confirmed working at HEAD (so we have a current oracle). `peroxide_window_3sp_bikerman_muh.py`, `mangan_full_grid_csplus_so4.py`, `l_eff_transport_sweep_csplus_so4.py --enable-water-ionization`, `phase6b_v9_gate4_finite_hydrolysis_smoke.py` (post-v10a) must each run-to-completion before migration starts.
- **Phase 6β state acknowledged**: Phase D Step 10 verdict is `OUTCOME_C_NON_IDENTIFIABLE_flagged` (Δ_β alone cannot match deck). v10b kinetics + Bikerman `a` discrepancy investigation is open. Migration must NOT freeze any value currently under active investigation. See "Open Questions" #5.
- **PyPI namespace** `pnpbv` confirmed available.
- **License** picked (Apache-2.0 recommended for patent grant on numerical methods).

## Inter-Milestone DAG

```
v0.0 (pretask: snapshot legacy reference-run outputs as byte-equivalence oracle)
 │
 └── v0.1 (repo bootstrap + form/anchor/C+D port + reproduce ClO₄⁻)
      │
      └── v0.2 (composable objects: Species, AnalyticCounterion, BVReaction, SternElectrode, DebyeBoltzmannIC, Problem)
           │
           └── v0.3 (multi-ion + parallel BV + first-class continuation strategies)
                │
                ├── v0.4 (HomogeneousReaction + water ionization)   ─┐
                │                                                     │  parallel after v0.3
                └── v0.5 (SurfaceAdsorbate + cation hydrolysis)    ───┤
                     │                                                │
                     └── v0.6 (docs pass + schema lock)  ◄────── overlaps with v0.5 tail
                          │
                          └── v0.7 (PNPInverse cutover)
                               │
                               └── v1.0 (PyPI publish + tagged release)
```

Sequential total ≈ 12–15 wk; parallel optimization ≈ 6–8 wk (per SCOPING.md).

---

## Parallelization Overview

The migration is structured as a DAG of ~82 subplans across 9 milestones (v0.0 → v1.0), organized into **batches** within each milestone. Subplans in the same batch are independent and can run in parallel; batches are sequential within a milestone (governed by the dependency column in each milestone's Parallel Batches table); v0.4 and v0.5 can run in parallel after v0.3, and v0.6 can overlap with v0.5's tail.

| Milestone | Subplans | Peak parallelism | Notes |
|---|---|---|---|
| v0.0 (pretask) | 8 | 5 (4 fresh script runs + 1 StudyResults snapshot-copy) | Blocks everything after; oracle fixtures required by v0.1, v0.2, v0.3, v0.4, v0.5 byte-equivalence tests |
| v0.1 | 23 | 6 (Batch D — form/closure ports) | Largest milestone |
| v0.2 | 13 | 4 (Batch A records, Batch D docs/tests post-Solver) | |
| v0.3 | 7 | 4 (Batch A plumbing) | |
| v0.4 | 6 | 2 | Can run parallel with v0.5 |
| v0.5 | 6 | 3 | Can run parallel with v0.4 |
| v0.6 | 10 | 6 (Batch A docs+tests post-A1) | |
| v0.7 | 4 | 3 | |
| v1.0 | 5 | 2 | |
| **Total** | **~82** | **6 in v0.1 Batch D and v0.6 Batch A** | |

Each subplan corresponds to a `.md` file under `.plans/pnpbv-migration/subplans/` (to be created). Per-milestone batch tables list the subplan paths, the tasks each subplan covers (cross-references the milestone's "Tasks" table), and inter-batch dependencies. The orchestrator agent reads this master plan, generates each subplan file from its batch row, then dispatches parallel implementer agents wave-by-wave (see "Orchestration Notes" appendix).

---

## Pretask v0.0 — Legacy Oracle Snapshots

**Estimate:** 0.5 day (compute time dominates; 4 reference scripts × 5–30 min each + reproducibility re-run + tarball upload).
**Reproduction target:** The current `Forward.bv_solver` reference scripts at the pinned PNPInverse commit hash, **plus the existing `StudyResults/solver_demo_slide15_no_speculative_cs/` outputs** from the 2026-05-12 solver-works baseline — this is the *oracle*, not a target to match.
**Verification gate:** All 4 reference scripts produce non-empty fixture parquets + the existing `StudyResults/solver_demo_slide15_no_speculative_cs/` outputs are copied and reformatted (task 0.5b); reproducibility cross-check (task 0.6) passes bit-for-bit for the 4 fresh runs; commit hashes (one for fresh runs, one for the StudyResults source) + env documented; fixtures tarball downloadable from PNPInverse GitHub Releases.

**Why first.** Every later milestone has a verification gate that compares pnpbv output to legacy `Forward/bv_solver/` output. The legacy stack is a moving target (Phase 6β Step 10 Phase D is mid-investigation; Bikerman-`a` bridge runs in flight per CLAUDE.md Hard Rule #7). Freezing the oracle at a specific PNPInverse commit hash *before* migration starts decouples porting from ongoing research and lets CI byte-equivalence tests run without keeping legacy + new code coexisting in the env.

**Output.** Fixtures directory `oracle_fixtures/` (staged at `PNPInverse/.plans/pnpbv-migration/oracle_fixtures/`, mirrored into `pnpbv/tests/fixtures/legacy_oracle/` once the pnpbv repo exists) containing per-reference-script snapshots tagged with the source PNPInverse commit hash. CI later fetches these via a GitHub Release asset on the PNPInverse repo.

### Parallel Batches

| Batch | Subplan (to be created at `.plans/pnpbv-migration/subplans/…`) | Tasks | Depends on |
|---|---|---|---|
| A0 | `v0.0-A0-pin-commit-and-env.md` | 0.1 | — |
| A1 | `v0.0-A1-perchlorate-sequential-snapshot.md` | 0.2 | A0 |
| A2 | `v0.0-A2-csplus-so4-parallel-snapshot.md` | 0.3 | A0 |
| A3 | `v0.0-A3-water-ionization-snapshot.md` | 0.4 | A0 |
| A4 | `v0.0-A4-cation-hydrolysis-snapshot.md` | 0.5 | A0 |
| A5 | `v0.0-A5-solver-demo-slide15-snapshot.md` | 0.5b | A0 |
| B | `v0.0-B-reproducibility-cross-check.md` | 0.6 | A1, A2, A3, A4, A5 |
| C | `v0.0-C-readme-and-tarball.md` | 0.7, 0.8 | B |

**Peak parallelism:** 5 (A1–A5 — assumes ≥32 GB RAM available; serialize on memory-constrained machines; A5 is a snapshot-copy of existing `StudyResults/` files, not a script rerun, so it's cheap).
**Total subplans:** 8.

### Tasks (DAG)

| # | Task |
|---|------|
| 0.1 | Pin PNPInverse commit hash (current HEAD or a named tag); record env (Python version, Firedrake version, OS, `OMP_NUM_THREADS`, MPI rank count) in `oracle_fixtures/ORACLE_COMMIT.txt` |
| 0.2 | Run `scripts/studies/peroxide_window_3sp_bikerman_muh.py` at production resolution; capture per-voltage `j_total`, per-species OHP concentrations, Newton iters, final residuals, clip-activation counts, IC fields at the OHP, full BV params dump → `oracle_fixtures/v01_perchlorate_sequential/{results.parquet,ohp_concentrations.parquet,ic_fields.npz,config.yaml}` |
| 0.3 | Same for `scripts/studies/mangan_full_grid_csplus_so4.py` → `oracle_fixtures/v03_csplus_so4_parallel/` |
| 0.4 | Same for `scripts/studies/l_eff_transport_sweep_csplus_so4.py --enable-water-ionization` (Phase 6α 8/8) → `oracle_fixtures/v04_water_ionization/` |
| 0.5 | Same for `scripts/studies/phase6b_v9_gate4_finite_hydrolysis_smoke.py` (post-v10a Langmuir) → `oracle_fixtures/v05_cation_hydrolysis/` |
| 0.5b | **Snapshot existing `StudyResults/solver_demo_slide15_no_speculative_cs/`** (2026-05-12 solver-works baseline: Cs⁺/SO₄²⁻ no-speculative stack, 9 K0_R4e factors {1, 1e-6, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16, 1e-17, 1e-18} × 25 V_RHE all converged — see project memory `project_solver_demo_slide15_no_speculative_outcome.md` and the existing `summary.json` / per-factor subdirs `factor_*/`). Reformat the per-factor outputs into the canonical fixture schema → `oracle_fixtures/v_solver_demo_slide15_no_speculative_cs/{results.parquet,ohp_concentrations.parquet,ic_fields.npz,config.yaml}`. Preserve the K0_R4e factor sweep as a column in `results.parquet` (multiple `j_total` rows per `V_RHE`, keyed by `K0_R4e_factor` — 225 rows total). Also preserve `iv_curves.{pdf,png}` and `summary.json` verbatim. Record the source commit via `git log --format=%H -- 'StudyResults/solver_demo_slide15_no_speculative_cs/**' \| head -1` → `oracle_fixtures/v_solver_demo_slide15_no_speculative_cs/SOURCE_COMMIT.txt`. **Copy + reformat, NOT a script rerun** — the existing files are the snapshot. Supplemental oracle for v0.3-class stacks (multi-ion + parallel 2e/4e, no speculative physics) |
| 0.6 | Reproducibility cross-check: re-run each of 0.2–0.5 once and verify bit-for-bit match (rules out Firedrake JIT cache nondeterminism, MPI rank-order issues, RNG seed leaks). If any fixture differs across reruns, fix the root cause before shipping — every later byte-equivalence test would inherit the nondeterminism. **Task 0.5b is exempt from rerun** — it's a copy of existing artifacts; verify file checksums match the source StudyResults files instead |
| 0.7 | Write `oracle_fixtures/README.md` documenting commit hash, env, seeds, wallclock per run, and per-fixture column schema |
| 0.8 | Tar the fixtures dir; upload as a GitHub Release asset on the PNPInverse repo (e.g. tag `oracle-2026-05-12`) so pnpbv CI can fetch it reproducibly via `gh release download` |

### v0.0 Key Outputs
- `oracle_fixtures/` with 4 reference-case subdirs from fresh runs (v01, v03, v04, v05) + 1 snapshot-copy subdir (`v_solver_demo_slide15_no_speculative_cs/`) from existing StudyResults, each containing `results.parquet`, `ohp_concentrations.parquet`, `ic_fields.npz`, `config.yaml`.
- `oracle_fixtures/ORACLE_COMMIT.txt` + `oracle_fixtures/README.md`.
- GitHub Release asset on PNPInverse repo with the tarball.

### v0.0 Failure-Mode Triage
If task 0.6 fails (non-reproducible): identify the source of nondeterminism. Likely culprits in order: (a) Firedrake JIT cache (clear `~/.cache/pyop2` and `/tmp/firedrake-tsfc` between runs); (b) MPI rank ordering (pin `OMP_NUM_THREADS=1` and `mpiexec -n 1`); (c) RNG seed leak (grep for `np.random.seed` / `random.seed` in the script + transitive imports — the IC machinery has had this in the past). Do **not** ship fixtures from a nondeterministic baseline.

---

## Milestone v0.1 — Bootstrap + Port Forms + Reproduce ClO₄⁻ Sequential

**Estimate:** 1–2 wk.
**Reproduction target:** `scripts/studies/peroxide_window_3sp_bikerman_muh.py` (ClO₄⁻ single counterion, sequential R₁/R₂, Stern, `logc_muh`, debye_boltzmann IC).
**Verification gate:** byte-equivalent to legacy within |Δj_total|/max(|j_legacy|, 1e-30) ≤ 1e-10 across V_RHE ∈ [−0.5, +1.0] V at production resolution.

### Parallel Batches

| Batch | Subplan | Tasks | Depends on |
|---|---|---|---|
| A | `v0.1-A-repo-bootstrap.md` | 1.1, 1.2 | v0.0 |
| B1 | `v0.1-B1-nondim.md` | 1.3 | A |
| B2 | `v0.1-B2-numerical.md` | 1.4 | A |
| B3 | `v0.1-B3-logging-skeleton.md` | 1.17 | A |
| C | `v0.1-C-internal-forms-shared.md` | 1.5 | B1, B2 |
| D1 | `v0.1-D1-formulations-logc.md` | 1.6 | C |
| D2 | `v0.1-D2-formulations-logc-muh.md` | 1.7 | C |
| D3 | `v0.1-D3-internal-multi-ion.md` | 1.8 | C |
| D4 | `v0.1-D4-internal-stern.md` | 1.9 | C |
| D5 | `v0.1-D5-internal-bikerman.md` | 1.10 | C |
| D6 | `v0.1-D6-mesh.md` | 1.12 | A |
| E1 | `v0.1-E1-debye-boltzmann-ic.md` | 1.11 | D1, D2, D5 |
| E2 | `v0.1-E2-continuation-ladders.md` | 1.16 | B2 |
| E3 | `v0.1-E3-continuation-anchor.md` | 1.13 | D1–D5, E1 |
| E4 | `v0.1-E4-continuation-cold-warm.md` | 1.14 | E3 |
| E5 | `v0.1-E5-continuation-charge.md` | 1.15 | E3 |
| F1 | `v0.1-F1-solver-shim.md` | 1.18 | E3, E4, B3 |
| F2 | `v0.1-F2-mms-port.md` | 1.19 | D1, D2 |
| F3 | `v0.1-F3-legacy-test-port.md` | 1.20 | D1–D6, E1–E5 |
| G | `v0.1-G-ci-workflow.md` | 1.21 | F1, F2, F3 |
| H1 | `v0.1-H1-reference-example.md` | 1.22 | F1 |
| H2 | `v0.1-H2-byte-equivalence-test.md` | 1.23 | H1, v0.0 |
| H3 | `v0.1-H3-bikerman-provenance-hook.md` | 1.24 | B3, F1 |

**Peak parallelism:** 6 concurrent subplans (Batch D form/closure ports).
**Total subplans:** 23.

### Tasks (DAG)

| # | Task | Output | Depends on |
|---|------|--------|-----------|
| 1.1 | Create GitHub repo `pnpbv`; commit `pyproject.toml` (name=pnpbv, python>=3.10, dev/test/docs extras), `LICENSE` (Apache-2.0), `.gitignore` (Firedrake cache, `__pycache__`, `dist/`, `.pytest_cache`, `.venv/`), `CHANGELOG.md` with v0.1 entry, README skeleton | repo on GitHub, initial commit | — |
| 1.2 | Stub module tree: `pnpbv/{__init__.py,species.py,reactions/__init__.py,reactions/bv.py,reactions/homogeneous.py,reactions/adsorbate.py,physics/__init__.py,physics/water_ionization.py,physics/cation_hydrolysis.py,electrode.py,initializers/__init__.py,initializers/debye_boltzmann.py,formulations/__init__.py,formulations/logc.py,formulations/logc_muh.py,problem.py,mesh.py,continuation/__init__.py,continuation/anchor.py,continuation/cold_warm.py,continuation/charge.py,continuation/ladders.py,solver.py,solution.py,nondim.py,numerical.py,logging.py,_internal/__init__.py,_internal/forms.py,_internal/multi_ion.py,_internal/stern.py,_internal/bikerman.py,_internal/newton.py}` | empty stubs with module docstrings | 1.1 |
| 1.3 | Port `nondim.py`: `V_T`, `L_REF`, `C_SCALE` helpers (Firedrake-free; pure-Python so they can be imported without Firedrake) | `pnpbv/nondim.py` | 1.2 |
| 1.4 | Port `numerical.py`: `EXPONENT_CLIP=100`, `U_CLAMP=100`, `H2O2_SEED_NONDIM=1e-4`, Newton tolerances, default ladder schedules (`K0_LADDER_DEFAULT`, `KW_EFF_LADDER_DEFAULT`, `LAMBDA_HYDROLYSIS_LADDER_DEFAULT`). Every constant has a provenance comment per "Documentation Requirements" (cite CLAUDE.md Hard Rules where applicable) | `pnpbv/numerical.py` | 1.2 |
| 1.5 | Port `_internal/forms.py`: shared form-builder utilities (`_build_eta_clipped`, `build_steric_boltzmann_expressions`, etc.) from `Forward/bv_solver/`. **No math changes.** | `pnpbv/_internal/forms.py` | 1.3, 1.4 |
| 1.6 | Port `pnpbv/formulations/logc.py` from `Forward/bv_solver/forms_logc.py` | `pnpbv/formulations/logc.py` | 1.5 |
| 1.7 | Port `pnpbv/formulations/logc_muh.py` from `Forward/bv_solver/forms_logc_muh.py` | `pnpbv/formulations/logc_muh.py` | 1.5 |
| 1.8 | Port `_internal/multi_ion.py`: multi-ion shared-θ closure | `pnpbv/_internal/multi_ion.py` | 1.5 |
| 1.9 | Port `_internal/stern.py`: Stern compact-layer machinery (including `set_stern_capacitance_model` runtime-bump shim if used) | `pnpbv/_internal/stern.py` | 1.5 |
| 1.10 | Port `_internal/bikerman.py`: analytic Bikerman counterion residual closure | `pnpbv/_internal/bikerman.py` | 1.5 |
| 1.11 | Port `pnpbv/initializers/debye_boltzmann.py`: composite-ψ + multispecies-γ IC (keep name-based species lookups for v0.1; role-based introspection deferred to v0.2 per SCOPING.md Risk #1) | `pnpbv/initializers/debye_boltzmann.py` | 1.6, 1.7, 1.10 |
| 1.12 | Port `pnpbv/mesh.py`: `graded_rectangle`, `structured_rectangle` | `pnpbv/mesh.py` | 1.2 |
| 1.13 | Port `pnpbv/continuation/anchor.py` from `Forward/bv_solver/anchor_continuation.py` (functions `solve_anchor_with_continuation`, `extract_preconverged_anchor`, `solve_grid_with_anchor`) | `pnpbv/continuation/anchor.py` | 1.6, 1.7, 1.8, 1.9, 1.10, 1.11 |
| 1.14 | Port `pnpbv/continuation/cold_warm.py`: C+D dispatcher (`solve_grid_per_voltage_cold_with_warm_fallback`) | `pnpbv/continuation/cold_warm.py` | 1.13 |
| 1.15 | Port `pnpbv/continuation/charge.py`: charge continuation (Strategy B, kept for completeness even though it loses 3/13 on logc+counterion — see CLAUDE.md Hard Rule #1) | `pnpbv/continuation/charge.py` | 1.13 |
| 1.16 | Port `pnpbv/continuation/ladders.py`: `k0_ladder`, `kw_eff_ladder`, `lambda_hydrolysis` utilities (including `AdaptiveLadder` with `warm_start_floor` opt-in from step 9.5) | `pnpbv/continuation/ladders.py` | 1.4 |
| 1.17 | Implement `pnpbv/logging.py` skeleton: `set_level`, `RunLogger` class, per-run dir creation, `results.parquet` writer with **schema v0.1** = `{run_id, V_RHE, converged, newton_iters, final_residual, j_total, wall_time, clip_count, ic_residual_consistency_ok, bikerman_a_provenance_json}`, `results.json` mirror, `run.log` via Python logging, `config.yaml` dump, provenance header (pnpbv version, git SHA, Firedrake version, config hash, ISO timestamp). Schema version stored in parquet metadata key `pnpbv_schema_version = "0.1"` | `pnpbv/logging.py` | 1.1, 1.2 |
| 1.18 | Implement raw-function `pnpbv/solver.py` shim: thin wrapper around the v0.1 anchor + C+D functions. **No `Solver` / `Problem` classes yet** — accept legacy-style kwargs / params dict. Wires logging artifacts via `RunLogger`. Composable-objects API deferred to v0.2 | `pnpbv/solver.py` v0.1 shim | 1.13, 1.14, 1.17 |
| 1.19 | Port MMS verification suite: `scripts/verification/*` → `pnpbv/tests/mms/`. Adapt imports. Verify all rates match legacy | `pnpbv/tests/mms/test_*.py` | 1.6, 1.7 |
| 1.20 | Port legacy unit tests: `Forward/bv_solver/test_*.py` → `pnpbv/tests/unit/` + `pnpbv/tests/integration/`. Adapt imports. These are the byte-equivalence oracle | `pnpbv/tests/{unit,integration}/test_*.py` | 1.6–1.16 |
| 1.21 | CI workflow `.github/workflows/ci.yml`: pinned Firedrake Docker image (specific tag, not `latest`), runs `pytest -m "not slow"`, `ruff check`, `black --check`, `mypy --strict pnpbv/`. Separate nightly workflow for `pytest -m slow` (full Firedrake-using suite). Coverage report uploaded. **Docstring-contract lint in warn mode** (`pydocstyle` or `interrogate` — fails are reported but don't fail CI yet) | `.github/workflows/{ci,nightly}.yml` | 1.20 |
| 1.22 | Reference example `pnpbv/examples/v01_reference_perchlorate_sequential.py`: ported from `scripts/studies/peroxide_window_3sp_bikerman_muh.py`. Runs with `formulation='logc_muh'`, ClO₄⁻ counterion, sequential R₁/R₂, Stern, debye_boltzmann IC. Writes `results.parquet` | `pnpbv/examples/v01_reference_*.py` | 1.13, 1.14, 1.17, 1.18 |
| 1.23 | Byte-equivalence test `pnpbv/tests/integration/test_v01_reference.py` (marked `-m slow`): runs the same config under legacy `Forward.bv_solver` and `pnpbv`, asserts `np.allclose(j_total_pnpbv, j_total_legacy, rtol=1e-10, atol=1e-30)` across the V_RHE grid + per-species concentrations at the OHP within the same tolerance. Cross-check: BV exponent clip activation count matches legacy exactly | `pnpbv/tests/integration/test_v01_reference.py` | 1.20, 1.22 |
| 1.24 | Hard Rule #7 logging hook: `pnpbv.logging.log_bikerman_a_provenance()` records, per Species and AnalyticCounterion, whether `a` came from a physical Marcus/Stokes radius or from `A_DEFAULT = 0.01`. Captured in `results.parquet:bikerman_a_provenance_json` | logger callable + test | 1.17 |

### v0.1 Key Outputs
- `pnpbv` package importable from local `pip install -e .[dev]`.
- `pytest pnpbv/tests/` green; `pytest pnpbv/tests/integration -m slow` green.
- `pnpbv/examples/v01_reference_perchlorate_sequential.py` runs and writes `results.parquet` schema v0.1.
- Byte-equivalence test passes against `Forward.bv_solver` to ≤ 1e-10 rtol.
- CI workflow green on a PR-target branch (pinned Firedrake image).

### v0.1 Failure-Mode Triage
If byte-equivalence test fails: diff against legacy at the form-assembly level (compare `assemble(F_res)` vector elementwise). Most likely causes (in order): (a) constant-folding precision in `pnpbv/numerical.py` differs from legacy; (b) `_internal/forms.py` accidentally reordered terms in a way that changes floating-point summation order (use Kahan-summation or fixed reduction order); (c) Stern bump path differs (verify `set_stern_capacitance_model` reach order matches legacy); (d) IC seed values differ at machine precision (verify `H2O2_SEED_NONDIM` and `A_DEFAULT` are bit-identical).

---

## Milestone v0.2 — Composable Objects API

**Estimate:** 2–3 wk.
**Reproduction target:** v0.1 reference case re-expressed via composable-objects API.
**Verification gate:** byte-equivalent to v0.1 (rtol ≤ 1e-12 since the math is unchanged; only the call site changes).

### Parallel Batches

| Batch | Subplan | Tasks | Depends on |
|---|---|---|---|
| A1 | `v0.2-A1-species-record.md` | 2.1 | v0.1 |
| A2 | `v0.2-A2-bv-reaction-record.md` | 2.2 | v0.1 |
| A3 | `v0.2-A3-stern-electrode-record.md` | 2.3 | v0.1 |
| A4 | `v0.2-A4-solution-dataclasses.md` | 2.6 | v0.1 |
| B1 | `v0.2-B1-debye-boltzmann-ic-role-based.md` | 2.4 | A1, A3 |
| B2 | `v0.2-B2-problem-composition-root.md` | 2.5 | A1, A2, A3, B1 |
| C | `v0.2-C-solver-class.md` | 2.7 | B2, A4 |
| D1 | `v0.2-D1-reference-migration.md` | 2.8 | C |
| D2 | `v0.2-D2-byte-equivalence-test.md` | 2.9 | D1, v0.0 |
| D3 | `v0.2-D3-docstring-lint-fail-mode.md` | 2.10 | C |
| D4 | `v0.2-D4-doctest-enable.md` | 2.11 | C, D1 |
| D5 | `v0.2-D5-logging-schema-v0.2.md` | 2.12 | C |
| D6 | `v0.2-D6-init-reexports.md` | 2.13 | C |

**Peak parallelism:** 4 concurrent subplans (Batch A records, or Batch D post-Solver).
**Total subplans:** 13.

### Tasks

| # | Task | Depends on |
|---|------|-----------|
| 2.1 | Implement `pnpbv/species.py`: frozen dataclasses `Species(name, z, D, c_bulk, seed=None, role=None)`, `AnalyticCounterion(name, z, c_bulk, steric)`, `Bikerman(a)`, `IdealSteric()`. All `frozen=True` (adjoint-friendly per "Risks" #10 below) | 1.x |
| 2.2 | Implement `pnpbv/reactions/bv.py`: `BVReaction(name, reactants, products, n_e, E_eq, alpha_a, alpha_c, k0, log_rate=True)`. Reactants/products are `dict[Species, int]` for explicit stoich (object-keyed, not positional) | 2.1 |
| 2.3 | Implement `pnpbv/electrode.py`: `SternElectrode(C_stern, l_eff)`. Docstring includes the **full Bohra/Choi/CatINT citation chain** for `C_stern = 0.20 F/m²` per CLAUDE.md Hard Rule #6 and `.research/cmk3-stern-capacitance/SUMMARY.md` §10. No opinionated default for `C_stern` — user must supply | 2.1 |
| 2.4 | Implement `pnpbv/initializers/debye_boltzmann.py`: wrap v0.1 IC machinery in `DebyeBoltzmannIC` class. **Add role-based introspection** (`problem.species_by_role("proton")`) replacing hardcoded "H"/"O2" lookups. Add startup warning if IC steric mode disagrees with residual steric mode (CLAUDE.md Hard Rule #3) | 2.1 |
| 2.5 | Implement `pnpbv/problem.py`: `Problem(species, counterions, reactions, homogeneous=[], adsorbates=[], electrode, formulation, initializer)`. Composition root. `species_by_role(role)` helper for role-based introspection. Immutable (frozen dataclass) | 2.1, 2.2, 2.3, 2.4 |
| 2.6 | Implement `pnpbv/solution.py`: `Solution`, `SolutionDiagnostics` dataclasses. `Solution.current_density`, `Solution.reaction_currents`, `Solution.surface_pH`, `Solution.diagnostics` | 1.x |
| 2.7 | Implement `pnpbv/solver.py` v0.2: `Solver(problem, mesh)` class wrapping v0.1 raw-function interface. `Solver.solve_grid(v_rhe, continuation=None, log_level=..., out_dir=...)` returns `Solution`. Default continuation is `None` (auto-pick via heuristic stub — single-counterion → C+D; multi-ion → AnchorContinuation; deferred refinement to v0.3) | 2.5, 2.6, 1.x |
| 2.8 | Migrate `v01_reference_*.py` to composable API → `pnpbv/examples/v02_reference_perchlorate_sequential.py` | 2.7 |
| 2.9 | v0.2 byte-equivalence test: same physics, new API, rtol ≤ 1e-12 vs v0.1 reference | 2.8 |
| 2.10 | **Flip docstring-contract lint to fail mode in CI**. Public API (every symbol exported from `pnpbv/__init__.py`) must have docstring with type/units/default/range/meaning/citation/See-Also blocks per "Documentation Requirements" | 2.7 |
| 2.11 | **Enable doctest-in-CI**. README hello-world is the first doctest; runs in upstream Firedrake Docker image on every PR | 2.7 |
| 2.12 | Logging schema v0.2: additive — expose `SolutionDiagnostics` fields as parquet columns (`charge_imbalance`, `max_abs_c`, `surface_concentrations_json`). `pnpbv_schema_version = "0.2"` | 1.17, 2.6 |
| 2.13 | Update `pnpbv/__init__.py` to re-export composable-objects API: `Species, AnalyticCounterion, Bikerman, IdealSteric, BVReaction, SternElectrode, DebyeBoltzmannIC, Problem, Solver, Solution, SolutionDiagnostics, graded_rectangle, structured_rectangle` | 2.1–2.7 |

### v0.2 Key Outputs
- Composable-objects API works end-to-end on the v0.1 reference case.
- Role-based IC introspection lands (and v0.1 IC fields reproduce byte-equivalent under the new path).
- Docstring-contract lint enforced; doctest-in-CI enforced.

---

## Milestone v0.3 — Multi-ion + Parallel BV + Continuation Strategies

**Estimate:** 2 wk.
**Reproduction target:** `scripts/studies/mangan_full_grid_csplus_so4.py` (Cs⁺/SO₄²⁻ multi-ion + parallel 2e/4e ORR).
**Verification gate:** |Δj|/|j| ≤ 1e-8 across V_RHE grid (tolerance relaxed from v0.1's 1e-10 because multi-ion shared-θ + parallel BV introduces more nonlinear coupling; bit-equivalence not expected but numerical-noise band is).

### Parallel Batches

| Batch | Subplan | Tasks | Depends on |
|---|---|---|---|
| A1 | `v0.3-A1-multi-ion-plumbing.md` | 3.1 | v0.2 |
| A2 | `v0.3-A2-parallel-bv.md` | 3.2 | v0.2 |
| A3 | `v0.3-A3-continuation-strategy-classes.md` | 3.3 | v0.2 |
| A4 | `v0.3-A4-logging-schema-v0.3.md` | 3.8 | v0.2 |
| B1 | `v0.3-B1-auto-default-heuristic.md` | 3.4 | A3 |
| B2 | `v0.3-B2-object-keyed-k0-targets.md` | 3.5 | A2, A3 |
| C | `v0.3-C-reference-and-byte-equiv.md` | 3.6, 3.7 | A1, A2, A3, B1, B2, v0.0 |

**Peak parallelism:** 4 concurrent subplans (Batch A plumbing).
**Total subplans:** 7.

### Tasks

| # | Task | Depends on |
|---|------|-----------|
| 3.1 | Plumb multi-ion shared-θ closure through `Problem` (`counterions=[Cs, SO4]` triggers `multi_ion_enabled=True` in residual). Verify CLAUDE.md Hard Rule #1: anchor continuation auto-selected for multi-ion + Stern (C+D fails 13/13 around V ≈ +0.55 V on this stack) | 2.5 |
| 3.2 | Plumb parallel BV (multiple `BVReaction` on same electrode surface): r2e (`E°=0.695 V`) + r4e (`E°=1.23 V`), Ruggiero parallel-2e/4e per CLAUDE.md Hard Rule #4. Update Solution to expose `reaction_currents[r2e.name]` separately | 2.2 |
| 3.3 | Implement first-class continuation strategy classes: `AnchorContinuation(v_anchor, k0_ladder, kw_eff_ladder)`, `ColdWithWarmFallback`, `ChargeContinuation` in `pnpbv/continuation/`. Each is a frozen dataclass | 1.13, 1.14, 1.15 |
| 3.4 | Solver auto-default heuristic (`Solver._pick_default_continuation(problem)`): multi-ion + Stern → `AnchorContinuation`; single-counterion + Stern → `ColdWithWarmFallback`; minimal stack → `ChargeContinuation`. Unit tests with synthetic problems | 3.3 |
| 3.5 | Migrate `k0_targets={0: ...}` to object-keyed `{r2e: K0_R2E, r4e: K0_R4E}`. Continuation code looks up reactions by identity, not index. Add deprecation warning if integer-keyed dict supplied | 3.2, 3.3 |
| 3.6 | Reference example `pnpbv/examples/v03_reference_csplus_so4_parallel_2e_4e.py` | 3.1, 3.2, 3.3 |
| 3.7 | Verification gate test `pnpbv/tests/integration/test_v03_reference.py` (`-m slow`): reproduces `mangan_full_grid_csplus_so4.py` within rtol ≤ 1e-8 | 3.6 |
| 3.8 | Logging schema v0.3: additive — `j_per_reaction_json` column (map reaction name → current density), `continuation_trajectory_json` column (ordered list of rungs attempted, outcomes, fallback triggers), `ladder_telemetry_json`. `pnpbv_schema_version = "0.3"` | 2.12 |

### v0.3 Key Outputs
- Cs⁺/SO₄²⁻ + parallel 2e/4e reference case lands.
- Continuation strategies are first-class objects; auto-default heuristic works.
- Object-keyed `k0_targets`; positional binding deprecated.

---

## Milestone v0.4 — HomogeneousReaction + Water Ionization

**Estimate:** 2 wk.
**Reproduction target:** Phase 6α 8/8 sweep from `scripts/studies/l_eff_transport_sweep_csplus_so4.py --enable-water-ionization`.
**Verification gate:** All 8 voltages converge; |Δj|/|j| ≤ 1e-8 vs legacy.

### Parallel Batches

| Batch | Subplan | Tasks | Depends on |
|---|---|---|---|
| A1 | `v0.4-A1-homogeneous-reaction.md` | 4.1 | v0.3 |
| A2 | `v0.4-A2-water-ionization-factory.md` | 4.2 | A1 |
| A3 | `v0.4-A3-anchor-kw-ladder-integration.md` | 4.3 | A2, v0.3 anchor |
| A4 | `v0.4-A4-logging-schema-v0.4.md` | 4.6 | v0.3 |
| B1 | `v0.4-B1-reference-and-byte-equiv.md` | 4.4, 4.5 | A3, v0.0 |
| B2 | `v0.4-B2-tutorial-water-ionization.md` | 4.7 | B1 |

**Peak parallelism:** 2 concurrent subplans (A1 + A4 at start; A3 + (independent work) later).
**Total subplans:** 6.

### Tasks

| # | Task | Depends on |
|---|------|-----------|
| 4.1 | Implement `pnpbv/experimental/reactions/homogeneous.py`: `HomogeneousReaction(name, reactants, products, K_eq, kf_kw_factor)`. Importing anything under `pnpbv.experimental.*` emits an `ExperimentalWarning` on first use per process; `pnpbv.experimental.acknowledge_experimental()` silences for batch runs | 2.2 |
| 4.2 | Implement `pnpbv/experimental/physics/water_ionization.py`: reference factory `water_ionization(K_w=1e-14)`. Docstring cites K_w convention AND prominently states experimental status + the v1 stable-surface exclusion rationale (Phase 6α 8/8 reproduces; Phase 6β coupling investigation still open) | 4.1 |
| 4.3 | `AnchorContinuation` auto-enables `kw_eff_ladder` when an `experimental.HomogeneousReaction` instance is present in `Problem.homogeneous`. Default-off path (no homogeneous reactions) remains byte-equivalent to v0.3. **Stable continuation API itself is unaffected** — it reacts to whatever's in `Problem.homogeneous`, regardless of namespace | 3.3, 4.1 |
| 4.4 | Reference example `pnpbv/examples/experimental/v04_water_ionization.py` reproducing Phase 6α 8/8 | 4.2, 4.3 |
| 4.5 | Verification gate test (`-m slow`) | 4.4 |
| 4.6 | Logging schema v0.4: `kw_eff_ladder_rungs_json`, `water_ionization_active` (bool). `pnpbv_schema_version = "0.4"`. **Schema columns are part of the stable parquet contract** (consumers can rely on them existing); but the *semantics* of rows where `water_ionization_active=True` fall under the experimental-API exclusion until graduation | 3.8 |
| 4.7 | Tutorial 3 (water ionization) drafted into `docs/tutorials/experimental/03_water_ionization.ipynb`. **Top-of-notebook banner:** "EXPERIMENTAL — API not stable. May change or be removed in any minor release. Track graduation status in `CHANGELOG.md`." | 4.4 |

---

## Milestone v0.5 — SurfaceAdsorbate + Cation Hydrolysis

**Estimate:** 2–3 wk.
**Reproduction target:** Phase 6β v10a smoke (`phase6b_v9_gate4_finite_hydrolysis_smoke.py` post-v10a Langmuir cap).
**Verification gate:** |Δj|/|j| ≤ 1e-8 vs legacy; Γ field reproduces.

### Parallel Batches

| Batch | Subplan | Tasks | Depends on |
|---|---|---|---|
| A1 | `v0.5-A1-surface-adsorbate-record.md` | 5.1 | v0.3 |
| A2 | `v0.5-A2-cation-hydrolysis-factory.md` | 5.2 | A1 |
| A3 | `v0.5-A3-adsorbate-plumbing.md` | 5.3 | A1 |
| A4 | `v0.5-A4-logging-schema-v0.5.md` | 5.6 | v0.4 (or v0.3 if v0.4 not landed) |
| B1 | `v0.5-B1-reference-and-byte-equiv.md` | 5.4, 5.5 | A2, A3, v0.0 |
| B2 | `v0.5-B2-tutorial-cation-hydrolysis.md` | 5.7 | B1 |

**Peak parallelism:** 3 concurrent subplans (A2, A3, A4 after A1).
**Total subplans:** 6.

### Tasks

| # | Task | Depends on |
|---|------|-----------|
| 5.1 | Implement `pnpbv/experimental/reactions/adsorbate.py`: `SurfaceAdsorbate(name, reactants, products, pKa0, dpKa_dE, gamma_max)`. References both `Species` (stable) and `AnalyticCounterion` (stable) for cation-hydrolysis stoich. Same `ExperimentalWarning` machinery as v0.4 | 2.1, 2.2 |
| 5.2 | Implement `pnpbv/experimental/physics/cation_hydrolysis.py`: factory `cation_hydrolysis_with_langmuir_cap(cation, proton, pKa0, gamma_max)`. Docstring cites Singh 2016 JACS 138:13006 + Linsey 2025 deck slide 9 + `docs/phase6/CMK3_capacitance_literature.md` for `gamma_max` provenance (note: as of 2026-05-12 `gamma_max_nondim = 0.047` is v10b smoke baseline pending literature calibration — see SCOPING.md Risk #5). Docstring also prominently states experimental status: Phase 6β Step 10 Phase D verdict (2026-05-12) is `OUTCOME_C_NON_IDENTIFIABLE_flagged` — model not yet matched to deck; v1.0 stable-surface exclusion reflects this | 5.1 |
| 5.3 | Plumb adsorbate Γ field through `Problem.adsorbates` (the list itself is stable; only `SurfaceAdsorbate` instances must come from `pnpbv.experimental.*`) + `Solution.surface_coverages` (stable mapping interface; populated with experimental data when experimental adsorbates are present) | 5.1, 2.6 |
| 5.4 | Reference example `pnpbv/examples/experimental/v05_cation_hydrolysis.py` reproducing v10a smoke | 5.2, 5.3 |
| 5.5 | Verification gate test (`-m slow`) | 5.4 |
| 5.6 | Logging schema v0.5: `gamma_per_adsorbate_json` (experimental-feature column; same schema-vs-API distinction as v0.4 task 4.6), `surface_pH`, `sigma_S`, `F0`, all remaining `SolutionDiagnostics` fields (this is the **last additive change before v0.6 freeze**). `pnpbv_schema_version = "0.5"` | 4.6 |
| 5.7 | Tutorial 4 (cation hydrolysis) drafted into `docs/tutorials/experimental/04_cation_hydrolysis.ipynb` with the same EXPERIMENTAL top-of-notebook banner as Tutorial 3 | 5.4 |

---

## Milestone v0.6 — Documentation Pass + Schema Lock

**Estimate:** 1–2 wk (overlaps with v0.5 tail).
**Verification gate:** API reference builds and publishes; all six tutorials run as CI doctests; migration guide covers every legacy reference case; schema-regression test enforces semver lock.

### Parallel Batches

| Batch | Subplan | Tasks | Depends on |
|---|---|---|---|
| A1 | `v0.6-A1-sphinx-autodoc-setup.md` | 6.1 | v0.5 |
| A2 | `v0.6-A2-tutorials-1-through-6.md` | 6.2 | A1, v0.4, v0.5 |
| A3 | `v0.6-A3-migration-guide.md` | 6.3 | v0.5 |
| A4 | `v0.6-A4-physics-numerics-guide.md` | 6.4 | A1 |
| A5 | `v0.6-A5-schema-lock-test.md` | 6.6 | v0.5 |
| A6 | `v0.6-A6-pyadjoint-tape-replay.md` | 6.9 | v0.5 |
| A7 | `v0.6-A7-numerical-knob-audit.md` | 6.10 | v0.5 |
| B1 | `v0.6-B1-doctest-in-ci-full.md` | 6.5 | A1, A2 |
| B2 | `v0.6-B2-changelog-fleshout.md` | 6.7 | A1–A7 |
| B3 | `v0.6-B3-docs-publish-workflow.md` | 6.8 | A1 |

**Peak parallelism:** 6 concurrent subplans (Batch A2–A7 after A1).
**Total subplans:** 10.

### Tasks

| # | Task | Depends on |
|---|------|-----------|
| 6.1 | Sphinx + `myst-parser` + `autodoc` + `napoleon` setup. `docs/conf.py`, `docs/index.md`. Auto-generates API reference from docstrings | 2.10 |
| 6.2 | Write all six tutorials as Jupyter notebooks + CI-tested scripts: (1) single counterion (v0.1/v0.2), (2) multi-ion + parallel (v0.3), (3) water ionization (v0.4), (4) cation hydrolysis (v0.5), (5) custom HomogeneousReaction, (6) tuning a continuation strategy. Use **reduced grids** (Ny=20, 3–4 voltages) in CI to keep wallclock under control; full reproductions live in `tests/integration/` `-m slow` | 4.7, 5.7 |
| 6.3 | Write migration guide `docs/migration/legacy_to_pnpbv.md`: side-by-side old → new API for `peroxide_window_3sp_bikerman_muh.py`, `mangan_full_grid_csplus_so4.py`, `l_eff_transport_sweep_csplus_so4.py`, `phase6b_v9_gate4_finite_hydrolysis_smoke.py`. **Explicit "Experimental API migration" section**: water-ionization users import from `pnpbv.experimental.physics.water_ionization` and cation-hydrolysis users import from `pnpbv.experimental.physics.cation_hydrolysis_with_langmuir_cap`; both must call `pnpbv.experimental.acknowledge_experimental()` once at startup. No v1.0 stable equivalent exists | 2.13, 3.6, 4.4, 5.4 |
| 6.4 | Write `docs/physics_and_numerics_guide.md`: codifies CLAUDE.md Hard Rules #1–#7 as prose-with-citations. Sections: nondim conventions, IC↔residual coupling (Rule #3), clipping conventions (Rule #2, refs `docs/solver/clipping_conventions.md`), continuation-strategy auto-default heuristic (Rule #1), Ruggiero parallel-2e/4e (Rule #4), C_S literature anchor (Rule #6 — cite Bohra/Choi/CatINT chain), Bikerman `a_nondim` physicality (Rule #7), **§8 Experimental namespace rationale and graduation criteria** (water ionization + cation hydrolysis live under `pnpbv.experimental.*` because Phase 6β Step 10 Phase D landed `OUTCOME_C_NON_IDENTIFIABLE_flagged` on 2026-05-12 + Bikerman-`a` discrepancy still investigating; graduation requires literature-calibrated parameter set + independent reviewer sign-off + Phase 6β identifiability resolved) | 6.1 |
| 6.5 | Doctest-in-CI for all docstring/tutorial examples (Firedrake-using examples run in nightly CI only; Firedrake-free in PR CI) | 6.1, 6.2 |
| 6.6 | **Schema lock test** `pnpbv/tests/integration/test_schema_lock.py`: loads a fixture `results.parquet` per version v0.1–v0.5 and asserts the current writer can read each and produces a superset schema. Any column rename/removal/retype fails the test. Documents in CHANGELOG that incompatible changes after v0.6 require major-version bump | 5.6 |
| 6.7 | CHANGELOG.md fleshed out: per-version entries v0.1–v0.6 with breaking-change flags and before/after snippets | 6.3, 6.6 |
| 6.8 | Publish docs to GitHub Pages on tagged release; CI workflow `.github/workflows/docs.yml` | 6.1 |
| 6.9 | **`pyadjoint` tape-replay smoke** `pnpbv/tests/integration/test_adjoint_friendly.py`: instantiate the v0.1 reference Problem, wrap in `pyadjoint.ReducedFunctional` with a trivial functional, replay the tape, assert no exceptions. Validates "no in-place mutation in user-facing code" for v2 inverse work | 5.4 |
| 6.10 | Numerical-knob provenance comment audit: every default in `pnpbv/numerical.py` has a "why this value" comment with citation (per "Documentation Requirements"). Pre-commit hook checks for diffs vs CHANGELOG | 6.7 |

---

## Milestone v0.7 — PNPInverse Cutover

**Estimate:** 1 wk.
**Verification gate:** All SCOPING.md reproduction targets reproduce via PNPInverse adapter scripts importing `pnpbv`; deprecation warnings on legacy import.

### Parallel Batches

| Batch | Subplan | Tasks | Depends on |
|---|---|---|---|
| A1 | `v0.7-A1-pnpinverse-import-replacement.md` | 7.1 | v0.6 |
| A2 | `v0.7-A2-deprecation-warnings.md` | 7.2 | A1 |
| A3 | `v0.7-A3-pnpinverse-claude-md-update.md` | 7.4 | A1 |
| B | `v0.7-B-adapter-verification-and-phase6b-check.md` | 7.3, 7.5 | A1 |

**Peak parallelism:** 3 concurrent subplans (A2, A3, B after A1).
**Total subplans:** 4.

### Tasks

| # | Task | Depends on |
|---|------|-----------|
| 7.1 | In PNPInverse: replace `Forward/bv_solver/` imports with `import pnpbv as pb`. Reference scripts to adapt: `peroxide_window_3sp_bikerman_muh.py`, `mangan_full_grid_csplus_so4.py`, `l_eff_transport_sweep_csplus_so4.py` (uses experimental water ionization), `phase6b_v9_gate4_finite_hydrolysis_smoke.py` (uses experimental cation hydrolysis), `phase6b_v10a_phase_A2_v_kin.py` (experimental cation hydrolysis), `phase6b_step6_plumbing_ablation.py` (experimental cation hydrolysis), `phase6b_v10a_v_sweep_diagnostic.py` (experimental), `solver_demo_slide15_no_speculative_cs*.py`. **Experimental-feature scripts** must `from pnpbv.experimental.physics import …` and call `pnpbv.experimental.acknowledge_experimental()` once at module load to silence the per-call `ExperimentalWarning` in batch runs | 6.x |
| 7.2 | Add deprecation warnings to `Forward/bv_solver/__init__.py` (and key modules) pointing at `pnpbv` equivalents | 7.1 |
| 7.3 | Adapter-script verification: each reference script run via `pnpbv` reproduces its legacy output within the milestone-specified tolerance | 7.1 |
| 7.4 | Update PNPInverse CLAUDE.md "Calling the production solver" section to point at `pnpbv` API; preserve all Hard Rules | 7.1 |
| 7.5 | Phase 6β step 10 latest results reproduce via `pnpbv` (sanity check that the live investigation surface still works) | 7.3 |

---

## Milestone v1.0 — Publication-Ready Build + Tagged GitHub Release (No PyPI Upload)

**Estimate:** 1 wk.
**Verification gate:** Wheel + sdist built; `twine check dist/*` passes (PyPI metadata valid); local install from wheel succeeds in a fresh env (Firedrake installed separately per `docs/install.md`); README hello-world runs; GitHub Release published with wheel + sdist as downloadable assets; `docs/release/PYPI_PUBLISH_COMMAND.md` documents the one-command PyPI publish flow, **ready to execute on user authorization but NOT executed in v1.0**.

### Parallel Batches

| Batch | Subplan | Tasks | Depends on |
|---|---|---|---|
| A | `v1.0-A-version-bump-and-build.md` | 8.1, 8.2 | v0.7 |
| B | `v1.0-B-twine-check-and-publish-readiness.md` | 8.3, 8.4 | A |
| C | `v1.0-C-github-release-with-wheel-sdist-assets.md` | 8.5 | B |
| D | `v1.0-D-fresh-env-smoke-from-local-wheel.md` | 8.6 | A |
| E | `v1.0-E-optional-docker-conda.md` *(optional)* | 8.7 | C |

**Peak parallelism:** 2 concurrent subplans (B + D after A).
**Total subplans:** 5 (4 required + 1 optional).

### Tasks

| # | Task | Depends on |
|---|------|-----------|
| 8.1 | Bump version to 1.0.0 in `pyproject.toml`; create annotated git tag `v1.0.0` with release-notes summary | 7.x |
| 8.2 | Build wheel + sdist via `python -m build`; verify both artifacts exist in `dist/` with expected names (`pnpbv-1.0.0-py3-none-any.whl`, `pnpbv-1.0.0.tar.gz`) | 8.1 |
| 8.3 | Run `twine check dist/*` to validate PyPI-compatible metadata (long-description rendering, classifier validity, README rendering on PyPI page). **DO NOT** run `twine upload` in v1.0 | 8.2 |
| 8.4 | Document the PyPI publish command in `docs/release/PYPI_PUBLISH_COMMAND.md`: full `twine upload` invocation, account/token setup steps, TestPyPI dry-run procedure (`twine upload --repository testpypi dist/*` first; smoke test from TestPyPI), then real-PyPI command. File header states: **READY-TO-EXECUTE pending user authorization. Do not run without explicit user request.** | 8.3 |
| 8.5 | Create GitHub Release pointing at the `v1.0.0` tag, with release notes summarizing the v0.1→v1.0 journey and **wheel + sdist attached as downloadable release assets**. External users can `pip install <release-asset-url>` to consume without PyPI in the loop | 8.3 |
| 8.6 | Smoke test in fresh env from local wheel: create a clean Firedrake env, `pip install dist/pnpbv-1.0.0-py3-none-any.whl`, run README hello-world, confirm output matches expected. **Confirms wheel is installable without PyPI** | 8.2 |
| 8.7 | Optional: Docker image, conda-forge recipe (deferred — only build if external users request) | 8.5 |

---

## Cross-Cutting Concerns

### Logging schema evolution

| Version | Columns added (additive only) |
|---------|-------------------------------|
| 0.1 | `run_id`, `V_RHE`, `converged`, `newton_iters`, `final_residual`, `j_total`, `wall_time`, `clip_count`, `ic_residual_consistency_ok`, `bikerman_a_provenance_json` |
| 0.2 | `charge_imbalance`, `max_abs_c`, `surface_concentrations_json` |
| 0.3 | `j_per_reaction_json`, `continuation_trajectory_json`, `ladder_telemetry_json` |
| 0.4 | `kw_eff_ladder_rungs_json`, `water_ionization_active` |
| 0.5 | `gamma_per_adsorbate_json`, `surface_pH`, `sigma_S`, `F0` |
| 0.6 | **FROZEN under semver.** Any incompatible change → major version bump |

### Doctest-in-CI introduction
- **v0.1:** docstring-contract lint in **warn mode** only (no public-API examples yet to doctest beyond raw-function shim).
- **v0.2:** docstring-contract lint in **fail mode**; doctest-in-CI enabled (README hello-world is first doctest).
- **v0.3–v0.5:** new tutorials added, each doctested.
- **v0.6:** full doctest suite, Firedrake-using examples gated to nightly CI; Firedrake-free PR CI.

### CLAUDE.md Hard Rules → enforcement points in `pnpbv`

| Rule | Enforcement |
|------|-------------|
| #1 C+D for logc+counterion | `Solver._pick_default_continuation` chooses `ColdWithWarmFallback` for single-counterion + Stern, `AnchorContinuation` for multi-ion + Stern (anchor at v ≈ +0.55 V). `ChargeContinuation` exists but is never auto-selected for logc+counterion. Codified in v0.3 (3.4) + physics & numerics guide §1 |
| #2 `exponent_clip=100` | `pnpbv/numerical.py:EXPONENT_CLIP=100` with provenance comment (cites R2 unclip threshold at V_RHE > −0.79 V on production grid). Unit test asserts BV form uses clip **pre** α·n_e multiplication. Logged per-run via `clip_count` column |
| #3 IC↔residual steric agreement | `DebyeBoltzmannIC.__post_init__` validates steric mode against `Problem.counterions` Bikerman/Ideal at construction. Runtime warning + `results.parquet:ic_residual_consistency_ok` flag logged |
| #4 Ruggiero parallel-2e/4e E_eq | Documented in tutorial 2 + physics & numerics guide §4; SCOPING.md "Refactored" list explicitly removes the hardcoded `PARALLEL_2E_4E_REACTIONS` constant, so users construct via `BVReaction(...)` with full E_eq citation in their own code |
| #5 Check deck data first | Doc-only; physics & numerics guide §5 with reading-order pointer to `data/EChem Reactor Modeling-Seitz-Mangan/` |
| #6 `C_S = 0.20 F/m²` lit-anchored | `SternElectrode.C_stern` docstring includes full Bohra/Choi/CatINT/Kilic-Bazant citation chain per `.research/cmk3-stern-capacitance/SUMMARY.md` §10. No opinionated default — user must supply, and is pointed at the citation chain |
| #7 Bikerman `a_nondim` physical only for counterions | `Species.__init__` accepts optional `a`; `AnalyticCounterion` requires `a` whenever `steric=Bikerman(...)`. `pnpbv.logging.log_bikerman_a_provenance()` writes `bikerman_a_provenance_json` listing which species use physical radii vs `A_DEFAULT`. Tutorial 1 + physics & numerics guide §7 spell out the discrepancy |

### Reproduction targets → verification gates

| Milestone | Target script (in PNPInverse) | Tolerance |
|-----------|------------------------------|-----------|
| v0.1 | `peroxide_window_3sp_bikerman_muh.py` | `\|Δj\|/max(\|j_legacy\|, 1e-30) ≤ 1e-10` |
| v0.2 | Same, via composable API | `≤ 1e-12` (math unchanged) |
| v0.3 | `mangan_full_grid_csplus_so4.py` | `≤ 1e-8` (multi-ion nonlinearity) |
| v0.4 | `l_eff_transport_sweep_csplus_so4.py --enable-water-ionization` | All 8 voltages converge; `≤ 1e-8` |
| v0.5 | `phase6b_v9_gate4_finite_hydrolysis_smoke.py` (post-v10a Langmuir) | `≤ 1e-8`; Γ reproduces |
| v0.7 | All of the above + Phase 6β step 10 latest, via PNPInverse adapter scripts | Same tolerances per milestone |

---

## Success Criteria (plan-level)

- All 8 milestones land; each verification gate passes.
- `pip install pnpbv` works from PyPI in a fresh Firedrake env.
- PNPInverse uses `pnpbv` exclusively for forward solves (no remaining imports of `Forward.bv_solver`).
- Every CLAUDE.md Hard Rule (#1–#7) has a concrete enforcement point in `pnpbv` (lint, runtime check, or doc citation).
- `results.parquet` schema is semver-locked from v0.6 onward.
- Six tutorials run as CI doctests on every PR.
- Sphinx-generated API reference published to GitHub Pages.
- The v0.6 `pyadjoint` tape-replay smoke passes (validates adjoint-friendly seams for v2 inverse work).

---

## Risks and Mitigations

The first 5 risks are pre-existing in SCOPING.md ("Key Risks / Refactor Concerns"); risks 6–13 are new and surfaced during planning.

| # | Risk | Lik | Imp | Mitigation |
|---|------|-----|-----|------------|
| 1 | **IC↔residual ↔ continuation coupling** (SCOPING.md #1) | Med | High | Ship v0.2 with regression tests covering single-counterion ClO₄⁻ + ideal/Bikerman; multi-ion Cs⁺/SO₄²⁻ + Bikerman; with/without role-tagged proton. Reserve buffer time in v0.2 |
| 2 | **Adjoint-friendliness drift** (SCOPING.md #2) | Med | High | `frozen=True` dataclasses for all public records; no in-place mutation in user-facing code; v0.6 `pyadjoint` tape-replay smoke (Task 6.9) |
| 3 | **Positional → object-keyed reaction binding** (SCOPING.md #3) | Low | Med | v0.3 Task 3.5: deprecation warning for integer-keyed `k0_targets`; continuation code looks up by identity from v0.3 forward |
| 4 | **Convergence-strategy auto-default heuristic** (SCOPING.md #4) | Med | Med | v0.3 Task 3.4: unit tests with synthetic problems covering each branch of the heuristic |
| 5 | **Reference-factory parameter provenance** (SCOPING.md #5) | Med | Med | Every reference factory in `pnpbv/physics/` cites primary literature in docstring; `gamma_max` flagged as v10b smoke baseline pending literature calibration (Task 5.2 docstring note) |
| **6** | **Firedrake-Docker CI fragility** | Med | High | Pin a specific upstream Firedrake Docker tag in CI YAML (not `latest`); cache the image between runs (~3 GB pull is slow); manual-trigger workflow for testing image upgrades; document the pinned version + upgrade cadence in `CHANGELOG.md` and `.github/workflows/README.md` |
| **7** | **Doctest-in-CI cost for Firedrake examples** | High | Med | Split doctests: Firedrake-free (PR CI on every push) vs Firedrake-using (nightly CI + on release tag). Use reduced grids in tutorials (Ny=20, 3–4 voltages); full reproductions live in `tests/integration/` `-m slow` |
| **8** | **Results-parquet schema versioning friction** | Med | High | Additive-only policy enforced by schema-lock test (Task 6.6) loading fixture parquets per version v0.1–v0.5. Any rename/removal/retype fails CI; major-version bump policy in CHANGELOG. Schema version stored in parquet metadata `pnpbv_schema_version` |
| **9** | **Soft-fork drift — PNPInverse keeps moving during migration** | High | Med | Freeze numerics-affecting changes in `Forward/bv_solver/` during v0.1–v0.3 (cosmetic OK); communicate freeze in PNPInverse CLAUDE.md; nightly cross-check legacy-vs-pnpbv on the reference scripts from v0.4 onward; Phase 6β Step 10 Phase D currently mid-investigation — coordinate cut date with that workstream |
| **10** | **pyadjoint-readiness latent risk** | Med | High | (Combined with #2.) v0.6 Task 6.9 `pyadjoint` tape-replay smoke catches in-place mutations + untaped ops before v2 starts. Also: a CI lint rule forbidding `.assign(`, `+=`, `*=` on Firedrake `Function` objects in `pnpbv/` (allow only in `_internal/newton.py`) |
| **11** | **Numerical-defaults provenance drift** | Med | Med | Pre-commit hook (Task 6.10) checks `pnpbv/numerical.py` diffs vs CHANGELOG; CI fails if a default value changes without a corresponding CHANGELOG entry + regression test |
| **12** | **Tutorial-test wallclock cost** | Med | Med | Tutorials use reduced grids (Ny=20, 3–4 voltages, no parallel BV ladder in tutorial 1); full reproductions live in `tests/integration/` `-m slow` and run nightly. Cap PR-CI wallclock at 20 min |
| **13** | **Experimental → stable graduation path unclear** | Med | Med | Document graduation criteria in `docs/physics_and_numerics_guide.md` §8 (v0.6 task 6.4): literature-calibrated parameter set + independent reviewer sign-off + Phase 6β identifiability resolved. Add a graduation-tracking section to CHANGELOG. Until graduation, `pnpbv.experimental.*` API is allowed to break in any release without semver bump |

---

## Open Questions

1. **Repo location**: Where does `pnpbv` live on GitHub? `github.com/JakeWeinstein/pnpbv`? A new org? Affects PyPI publish permissions and CI billing.
2. **License**: **Apache-2.0** recommended (patent grant on numerical methods; common for scientific Python). Confirm.
3. **PyPI namespace `pnpbv`** — reserved or not? Since v1.0 does NOT upload to PyPI, this is deferred. Sub-question: do we want to upload a placeholder to reserve the name (small risk of squatting otherwise), or wait until we're ready to publish for real? Recommend: defer until publication is authorized — package is consumable via GitHub Release asset URLs in the meantime.
4. **Firedrake-Docker image pin policy**: pick a specific tag for v0.1 and document upgrade cadence (e.g., quarterly bump with explicit CI run). Or rolling `latest` with full backsliding accepted?
5. **Soft-fork timing**: Phase 6β Step 10 Phase D verdict is `OUTCOME_C_NON_IDENTIFIABLE_flagged` (2026-05-12); bridge runs in flight to disentangle Bikerman `a_nondim` discrepancy. Recommend: **wait for Phase 6β to reach a stable checkpoint** (~1–2 wk?) before starting v0.1, so the byte-equivalence oracle isn't a moving target. Alternative: start v0.1 in parallel and freeze the legacy snapshot at a specific commit hash for the oracle.
6. **`gamma_max` literature calibration**: v0.5 Task 5.2 docstring flags `gamma_max_nondim = 0.047` as v10b smoke baseline. Since v0.5 ships under `pnpbv.experimental.*` (excluded from v1.0 stable surface), the calibration timing is less urgent — the docstring flags the value as smoke-baseline-pending-graduation. **Graduation gate (post-v1):** literature-calibrated `gamma_max` + independent reviewer sign-off + Phase 6β identifiability resolved. Confirm this is the right gate.
7. **Does v0.7 deprecate `Forward/bv_solver/` immediately, or with a one-version grace period?** Hard deprecation is cleaner; grace period is friendlier to mid-flight research scripts. Recommend hard deprecation with a clear migration window flagged in PNPInverse CLAUDE.md.

---

## Orchestration Notes

This master plan is designed to be consumed by an **orchestrator agent** that delegates implementation to parallel subagents. Intended workflow:

1. **Pre-flight.** Orchestrator reads `PLAN.md` (this file). Confirms "Prerequisites" are met. Confirms PNPInverse commit hash for v0.0 oracle pinning. Resolves all 7 "Open Questions" with the user (or accepts written defaults). **Open Questions block the orchestrator** — do not silently guess.

2. **Subplan generation.** Orchestrator iterates through each milestone's "Parallel Batches" table. For each subplan that does not yet exist as a `.md` file in `.plans/pnpbv-migration/subplans/`, the orchestrator generates it from the corresponding rows in the milestone's "Tasks" table. Each generated subplan must be **self-contained**: scope statement, dependencies (with paths to upstream subplans/fixtures), file paths to touch, exit criteria, pointers to relevant CLAUDE.md Hard Rules (#1–#7) and research summaries (`.research/cmk3-stern-capacitance/`, `.research/pnp-bv-literature-provenance/`, `.research/pnp-convergence-strategies/`).

3. **Wave execution.** Milestones run sequentially in DAG order (v0.0 → v0.1 → v0.2 → v0.3 → {v0.4, v0.5 parallel} → v0.6 → v0.7 → v1.0). Within each milestone, the orchestrator computes wave assignments from the dependency table (topological sort), then dispatches **N parallel implementer agents** on the N subplans in each wave. Waits for the wave to converge (all subagents return success) before advancing.

4. **Verification gate at every milestone boundary.** Orchestrator runs the milestone's verification gate (per "Verification gate" header) before advancing to the next milestone. If a gate fails, the orchestrator **stops and surfaces the failure with a diff**; it does NOT proceed.

5. **Agent type recommendations** (orchestrator may override per subplan):
   - **Pure ports** (v0.1 Batch D form/closure moves; v0.2 record subplans): `general-purpose`. Mechanical work.
   - **Math-preserving ports** (v0.1 Batch C `_internal/forms.py` shared utilities, D1/D2 formulations): `general-purpose` + post-task `codex:rescue` spot-check that UFL hash matches legacy.
   - **Tests** (v0.1 F2/F3/H2, v0.2 D2, v0.3 C, v0.4 B1, v0.5 B1): `general-purpose`; pair with `sciai-verify` for the byte-equivalence test specifically.
   - **CI/infrastructure** (v0.1 G, v0.6 A1/B3): `general-purpose`.
   - **Docs and tutorials** (most of v0.6): `general-purpose`.
   - **Schema-lock test** (v0.6 A5): `general-purpose` + `sciai-verify` — semver contract is load-bearing for PNPInverse consumption.
   - **`pyadjoint` tape-replay smoke** (v0.6 A6): `general-purpose` + `codex:rescue` — Firedrake/pyadjoint integration is finicky.
   - **Optional plan-level adversarial review**: spawn `gpt-critique-loop` on this master plan once before v0.0 starts, and on high-risk subplans (form ports, byte-equiv tests, schema-lock test) before dispatching their implementers.

6. **State tracking.** Orchestrator maintains `.plans/pnpbv-migration/STATUS.md` updating per batch: status (`pending|in_progress|done|failed`), last attempt timestamp, implementer-agent commit SHA on success. Enables resumability after interruption.

7. **Failure handling.**
   - Routine port failure → retry once with the failure diagnostic appended to the subplan prompt.
   - **Byte-equivalence failure** (v0.1 H2, v0.2 D2, v0.3 C, v0.4 B1, v0.5 B1) → STOP the milestone. Surface the diff (specific voltage(s), specific fields, magnitude of mismatch). Do not proceed. Likely root causes in order: (a) constant-folding precision drift in `numerical.py`; (b) form-assembly order change in `_internal/forms.py` (Kahan/fixed-reduction summation may be needed); (c) Stern bump path divergence; (d) IC seed values differing at machine precision.
   - CI workflow failure → collect CI log, route to the most-recent implementer that touched the failing component.
   - Firedrake-Docker image breaking change → pin to last-known-good tag; file an upstream issue.

8. **Schema migration enforcement.** Each milestone v0.1–v0.5 bumps `pnpbv_schema_version` in `pnpbv/logging.py` per the "Logging schema evolution" table. The v0.6 A5 subplan locks the schema under semver. From v0.6 forward, any subplan that touches `pnpbv/logging.py` schema must also bump the major version, update fixtures, and add a CHANGELOG entry. Orchestrator enforces this by running the schema-lock test after every implementer-agent commit.

9. **Sensible defaults if user is unreachable on Open Questions** (still confirm where possible):
   - Q1 repo: `github.com/JakeWeinstein/pnpbv`
   - Q2 license: Apache-2.0
   - Q3 PyPI namespace: defer — v1.0 does not upload; revisit when publication is authorized
   - Q4 Firedrake-Docker pin: latest stable tag, quarterly bump cadence
   - Q5 soft-fork timing: wait for Phase 6β Step 10 to reach a stable checkpoint (~1–2 wk)
   - Q6 `gamma_max`: ship v0.5 (experimental) with smoke baseline `0.047`; lit-calibration is a graduation gate, not a v1.0 ship gate
   - Q7 deprecation: hard cutover at v0.7 with PNPInverse CLAUDE.md migration window note
