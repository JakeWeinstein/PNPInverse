# REPO_LAYOUT.md — where things go

One-page map of the PNPInverse repo after the 2026-06-15 cleanup. When adding a
file, put it where this says; that keeps the tree navigable.

## Source packages (live, imported by the production stack — do not reorganize)

| Package | Role |
|---|---|
| `Forward/` | Core production package: `bv_solver/`, `steady_state/`, `params.py`, solvers. |
| `Nondim/` | Nondimensionalization (imported by Forward). |
| `FluxCurve/` | Flux/observable curves (imported by Forward). |
| `Inverse/` | Inverse solver interface (imported by Forward; inverse *track* is paused). |
| `Surrogate/` | Surrogate-model code (paused track, still importable). |
| `calibration/` | Firedrake-free locked constants (`v10b.py`, etc.). |

## scripts/ — runnable code (NOT the test suite)

| Path | Contents |
|---|---|
| `scripts/_bv_common.py` | Shared constants/scales/presets. Imported by ~30 tests + most drivers. |
| `scripts/verification/` | MMS convergence engine — imported by `tests/test_mms_*`. Keep. |
| `scripts/studies/drivers/` | Re-runnable reference drivers (phase6b/phase7 calibration, mangan, l_eff, jithin reproductions, profile-likelihood). |
| `scripts/studies/plot/` | Plot generators (read results, write figures). |
| `scripts/studies/extract/` | Experimental-data digitizers (produce target JSON/CSV). |
| `scripts/studies/*.py` | Active/ad-hoc studies not yet promoted to a driver (e.g. current phase7p3 work). |
| `scripts/profile/` | Profiling harnesses. |

Rule of thumb: write-once-and-discard scripts don't belong here — git history is
the archive. A script that asserts an invariant belongs in `tests/`, not `scripts/`.

## tests/ — the pytest suite (flat, phase/feature-prefixed)

- Run fast: `pytest -m "not slow"`. Run Firedrake: `pytest -m slow` (needs the venv).
- Only custom marker is `slow` (declared in `pyproject.toml`).
- Tests import production constants from `scripts._bv_common` and a handful of
  `scripts/studies/drivers/` modules — those drivers are de-facto fixtures. If you
  move one, grep BOTH `scripts.studies.<mod>` (import) and
  `scripts/studies/.../<mod>.py` (path string) and update every hit.
- Known pre-existing failures (NOT from cleanup; see `tasks/repo_cleanup_plan.md`
  deferred reviews): `test_autograd_gradient` (5), `test_jithin_picard_closure`
  (10), `test_multistart::test_perfect_data_recovery` (1).

## StudyResults/ — the working output record

| Path | Contents |
|---|---|
| `StudyResults/phase7*`, `phase7p2*`, `phase7p3*`, `solver_demo_*`, `parallel_2e_4e` | ACTIVE phase-7 results at top level — scripts read/write these paths. |
| `StudyResults/inverse/` | Inverse version series (v14–v26) + master_inference. |
| `StudyResults/methodology/` | mms, diagnostics, iv_curves, peroxide_window. |
| `StudyResults/reproductions/` | jithin_*, yash_* simulator reproductions. |
| `StudyResults/_logs/` | Loose run logs. |
| `StudyResults/archive/` | Closed phases: `phase6b/`, `phase5/` (fast_realignment), `scratch/` (smoke/test probes), `legacy/`. |

Active phase-7 dirs are deliberately NOT bucketed, because active scripts hardcode
those paths. Bucket them only when phase 7 closes (and re-point the paths then).

## docs/

- Living source-of-truth: `docs/solver/` (API, clipping, continuation), `docs/phase6/`,
  `docs/phase7/`, `docs/papers/`. See the table in `CLAUDE.md`.
- `docs/handoffs/` keeps only HOT sessions (#26, #41–#45); closed #2–#40 are in
  `docs/handoffs/archive/`. Raw `.codex_log_*` transcripts are stripped (the
  distilled `*_from_gpt.md` / `FINAL_REVISION.md` are the record).

## Off-git / ignored

- `archive/` (225 MB old runs) moved to `~/PNPInverse-archive` (history retains it).
- `data/` (experimental drop) — gitignored, lives on disk only.
- `writeups/**/node_modules`, LaTeX build artifacts, `__pycache__`, `.DS_Store` — gitignored.

## tasks/

- `tasks/todo.md` — the **active science plan** (currently Phase 7.3). Do not clobber.
- `tasks/lessons.md` — accumulated lessons.
- `tasks/repo_cleanup_plan.md` — this cleanup's checklist + deferred follow-ups.
