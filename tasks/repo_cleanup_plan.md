# Repo Organization Plan — PNPInverse (parallel-subagent edition)

Status: **APPROVED PLAN, NOT YET EXECUTED** (drafted 2026-06-15; parallelized 2026-06-15).
Decisions locked with user:
- **Git size**: tidy working tree, **keep history** (no `git-filter-repo`; preserve all
  commit SHAs/clones). Move `archive/` off-git; `git rm` dead files (recoverable from history).
- **Reorg depth**: **full restructure** — split `scripts/studies/` into `drivers/ plot/
  extract/` and rewrite the importing tests.
- **Execution**: decomposed into **5 independent-scope workstreams dispatchable in parallel**
  (WS-A…E) + 1 serial integration pass (WS-F).
- This file is the checklist; `tasks/todo.md` is the active **Phase 7.3 science** plan — do
  NOT overwrite it.

---

## Parallel execution model

**Disjoint-ownership principle.** Each workstream OWNS one top-level subtree and may modify
**only** files inside it. No two workstreams share a writable file. The only repo-root files
that need editing (`.gitignore`, `CLAUDE.md`, `README.md`, new `REPO_LAYOUT.md`,
`tasks/lessons.md`) are assigned to exactly one owner (`.gitignore`→WS-C; records files→WS-F
integration only). This is what makes the five agents safe to run concurrently.

**Isolation mechanism (pick at dispatch time):**
- **Recommended — worktree per workstream.** Dispatch each agent with `isolation: "worktree"`
  on its own branch (`chore/cleanup-<area>`). Disjoint scopes ⇒ branches merge into
  `chore/repo-cleanup` with **zero conflicts** (they touch different files). Each workstream is
  an independently reviewable commit set. Cost: extra working-tree checkouts (repo is ~900 MB;
  object store is shared so incremental cost is the checkout, not full `.git`).
- **Fallback — fs-only fan-out, low disk.** Agents run in the shared tree but each restricted to
  its subtree and performing **filesystem mutations only** (`mv`/`rm`/`mkdir`) + emitting a
  manifest; they run **no `git` write commands** (no index race). WS-F replays `git add -A` and
  makes one commit per area. Use if disk for 5 worktrees is a concern.

**Wave graph (dependencies):**
```
WAVE 1 (all parallel, disjoint scopes, no cross-deps):
  WS-A  StudyResults/        ─┐
  WS-B  docs/                 │
  WS-C  writeups/ + root junk │──►  WAVE 2:  WS-F  integration
  WS-D  archive/  (off-git)   │            (pytest, records files,
  WS-E  scripts/ + tests/    ─┘             seam re-grep, PR)
```
WS-F is the **only** serialized step; it runs after A–E merge. Everything in Wave 1 is
independent — no Wave-1 stream reads or writes another's subtree.

**Shared-file ownership (hard rules):**
- `.gitignore` → **WS-C only**.
- `CLAUDE.md`, `README.md`, `REPO_LAYOUT.md`, `tasks/lessons.md` → **WS-F only** (they must
  reflect the *final* merged paths from A + E, so they can't be written in parallel).
- `.DS_Store` deletion → each agent removes them **only within its own subtree**; WS-C handles
  repo-root + any unowned location. Never run a repo-wide `find . -name .DS_Store -delete` from a
  scoped agent.

**Integration seam (the one latent cross-dependency).** Surviving `scripts/**` code could read a
`StudyResults/` path that WS-A moves. Handled without coupling the agents: WS-E flags any keeper
that reads a `StudyResults/` path; WS-F re-greps `StudyResults/` across surviving
`scripts/** Forward/**` after merge and patches stale read paths. (At plan time the only readers
— `_phase_D_*`, `_score_phase7_sweeps`, `mangan_p15_run_D_verdict`, `surrogate/resolve_missing` —
are all in WS-E's deletion set, so the expected patch count is ~0.)

**Dispatch (single fan-out message launches A–E concurrently):**

| WS | Branch | Agent type | Weight | Self-validates? |
|---|---|---|---|---|
| A | `chore/cleanup-studyresults` | general-purpose | medium | n/a (no code) |
| B | `chore/cleanup-docs` | general-purpose | light | n/a (no code) |
| C | `chore/cleanup-artifacts` | general-purpose | light | n/a (no code) |
| D | `chore/cleanup-archive` | general-purpose | light | n/a (no code) |
| E | `chore/cleanup-scripts-tests` | general-purpose | **heavy (long pole)** | **yes — `pytest -m "not slow"`** |
| F | `chore/repo-cleanup` (merge target) | orchestrator (this session) | serial | full `pytest` |

---

## Workstream scopes (at-a-glance)

| WS | Owns (may modify ONLY) | Forbidden | Depends on | Runs with |
|---|---|---|---|---|
| A | `StudyResults/**` | everything else | — | B,C,D,E |
| B | `docs/**` | everything else | — | A,C,D,E |
| C | `.gitignore`, `writeups/**`, root junk (`untitled folder/`, `Renders/`, `retrain*.txt`, `retrainlog.md`, root `.DS_Store`) | `docs/`, `StudyResults/`, `scripts/`, `tests/`, `archive/` | — | A,B,D,E |
| D | `archive/**` (+ external move target) | everything else | — | A,B,C,E |
| E | `scripts/**`, `tests/**` | everything else | — | A,B,C,D |
| F | `CLAUDE.md`, `README.md`, `REPO_LAYOUT.md`, `tasks/lessons.md`, merge/PR | new feature code | **A–E merged** | (serial) |

---

## WS-A — StudyResults taxonomy & junk  ·  branch `chore/cleanup-studyresults`

**Agent brief:** You may modify ONLY files under `StudyResults/`. Apply the taxonomy below with
`git mv` (worktree mode) or `mv` (fs-only mode). Do not touch any other directory. Commit as
`chore(cleanup): StudyResults taxonomy + junk`.

Target tree:
```
StudyResults/
  active/        phase7p3_*, phase7_fit
  phase7/        phase7p2_*, phase7_dual_pathway, phase7_p0_*, phase7_fit_adjoint
  inverse/       v14–v26, master_inference_*  (mostly already nested)
  reproductions/ jithin_*, yash_*, mangan_*, seitz_*
  methodology/   mms*, diagnostics, iv_curves, peroxide_window
  archive/
    phase5/      fast_realignment/
    phase6b/     all 36 phase6b_* + phase6b/
    legacy/      (existing, minus dup log)
  _logs/         ~50 loose *_run.log files
```
- [ ] Delete `StudyResults/legacy/surrogate_v12/run.log.saved` (31 MB dup of `run.log`).
- [ ] Delete stale `StudyResults/legacy/surrogate_v12/training_data_gapfill.npz.checkpoint.npz`.
- [ ] Delete 2 empty dirs (`inverse/v23_.../a0p30_0p55_k0low_ahigh/`,
      `phase6b_step10_phase_D_bridge_corrected_a/`) and any `StudyResults/**/.DS_Store`.
- [ ] `git mv` each group into the tree above.
- [ ] Smoke/test/scratch dirs (~21): delete, or `git mv` → `archive/scratch/` (default: archive).
- [ ] Output for WS-F: a list of any moved dir that a script might read (so F can re-grep).

## WS-B — docs archival  ·  branch `chore/cleanup-docs`

**Agent brief:** You may modify ONLY files under `docs/`. Commit as
`docs(cleanup): strip codex logs, archive closed handoffs`.
- [ ] Strip `.codex_log_R*.txt` (~11 MB) and `.codex_session_id` from `docs/handoffs/*` dirs
      (the distilled `R*_from_gpt.md`/`FINAL_REVISION.md` are the kept record).
- [ ] `git mv` closed handoffs #2–#40 → `docs/handoffs/archive/`; keep #26 + #41–#45 hot.
- [ ] De-collide duplicate numbers #12, #13, #19 (suffix `a`/`b`).
- [ ] `git mv` 4 loose top-level docs (`PHASE_6B_V9_*`, `phase6b_v9_post_gate4_plan.md`)
      → `docs/phase6/`.
- [ ] Delete `docs/.DS_Store`, `docs/papers/.DS_Store`.
- [ ] (Optional) fold `verification_sweep/` + `.verification/` audit reports into `docs/audits/`
      — but those dirs are OUTSIDE `docs/`, so leave them for WS-F to avoid scope bleed.

## WS-C — gitignore + build artifacts + root junk  ·  branch `chore/cleanup-artifacts`

**Agent brief:** You may modify ONLY `.gitignore`, `writeups/**`, and the named root-level junk.
Do not touch `docs/ StudyResults/ scripts/ tests/ archive/`. Commit as
`chore(cleanup): gitignore + de-track build artifacts + remove empty dirs`.
- [ ] Add to `.gitignore`: `**/node_modules/`, broaden `**/__pycache__/`.
- [ ] `git rm -r --cached writeups/5:27Presentation/build/node_modules` (7.6 MB, 270+ files).
- [ ] `git rm --cached` tracked LaTeX artifacts under `writeups/`
      (`*.aux *.fls *.fdb_latexmk *.log *.out *.toc`).
- [ ] Delete empty `untitled folder/`, `Renders/`; delete root `.DS_Store`.
- [ ] Move 3 stale retrain logs (`retrain_existing_models_log.txt`,
      `retrain_new_models_log.txt`, `retrainlog.md`) → `docs/legacy/` is WS-B's scope, so instead
      delete them here (recoverable from history) OR stage for WS-F. Default: `git rm`.

## WS-D — move `archive/` off-git  ·  branch `chore/cleanup-archive`

**Agent brief:** You may modify ONLY `archive/` (and create the external target). Keep history
(no filter-repo). Commit as `chore(cleanup): move 225 MB archive/ off-git`.
- [ ] Move `archive/` (225 MB) to the user-designated external location
      (default `~/PNPInverse-archive/`; confirm path before moving).
- [ ] `git rm -r archive/`.
- [ ] Leave a one-line breadcrumb file `archive/README.md`? NO — that re-creates the dir; instead
      record the new location in the WS-F integration notes for README/CLAUDE.md.

## WS-E — scripts + tests restructure / prune / promote  ·  branch `chore/cleanup-scripts-tests`

**THE LONG POLE — coupled scripts↔tests; runs as ONE agent (do not sub-split). Self-validates
with `pytest -m "not slow"` before committing.** You may modify ONLY `scripts/**` and `tests/**`.

### Invariants — locked modules (move only while editing the importing test in the same commit)
`scripts/__init__.py` and `scripts/studies/__init__.py` exist; new subdirs each need their own
`__init__.py`.

| Module (current path under scripts/) | Target subdir | Tests importing it (must update on move) |
|---|---|---|
| `_bv_common.py` | stay (lib) | ~30 tests — DO NOT MOVE |
| `verification/mms_bv_3sp_logc_boltzmann.py` | stay | test_mms_convergence, test_mms_steric_boltzmann_convergence |
| `verification/mms_pnpbv_muh_multi_ion_stern.py` | stay | test_mms_logc_muh_multi_ion_stern |
| `studies/run_multi_seed_v13.py` | drivers/ | test_diagnostic_metadata, test_multi_seed_aggregation |
| `studies/gradient_benchmark.py` | drivers/ | test_gradient_benchmark |
| `studies/phase6b_step10_phase_D_fit_eval.py` | drivers/ | test_phase6b_step10_phase_D_fit_eval |
| `studies/phase6b_step10_phase_D_orchestrate.py` | drivers/ | test_phase6b_step10_phase_D_orchestrate |
| `studies/phase6b_step6_plumbing_ablation.py` | drivers/ | test_phase6b_step6_plumbing_ablation(_slow), test_phase6b_v10b_calibration |
| `studies/phase6b_v10a_phase_A2_v_kin.py` | drivers/ | test_phase6b_step6_plumbing_ablation, test_phase6b_v10a_phase_A2_driver, test_phase6b_v10b_calibration |
| `studies/phase6b_v10a_v_sweep_diagnostic.py` | drivers/ | test_phase6b_v10a_phase_A2_driver, test_phase6b_v10a_v_kin_selection, test_phase6b_v10b_calibration |
| `studies/phase6b_v10b_cs_bracket.py` | drivers/ | test_phase6b_v10b_bracket_matrix |
| `studies/phase6b_v10b_gamma_kdes_matrix.py` | drivers/ | test_phase6b_v10b_bracket_matrix |
| `studies/solver_demo_slide15_dual_pathway_cs.py` | drivers/ | test_phase7p2_observables, test_phase7p3_c1_wiring, test_phase7p3_frame_byte |
| `studies/phase7p3_p0_2_onset_extractor.py` | drivers/ | test_phase7p3_onset_extractor |
| `studies/profile_likelihood_pde.py` | drivers/ | test_profile_likelihood |
| `studies/sensitivity_visualization.py` | plot/ | test_sensitivity_visualization |

**Also keep (CLAUDE.md-canonical, not test-imported):** `mangan_full_grid_csplus_so4`,
`l_eff_transport_sweep_csplus_so4`, `pass_a_*`, `peroxide_window_3sp_parallel_2e_4e_csplus_so4`,
`anchor_smoke_*`, `_run_jithin_*` drivers, `derive/*` extractors.

### E-1: promote verification scripts → tests (TDD: write → prove → remove)
- [ ] `test_picard_residual_consistency.py` (`slow`) from `picard_residual_consistency_csplus_so4.py`
      (Phase 5α GATE) + `_lowk0` as a parametrized case (Phase 5γ GATE v2); prove pass, `git rm` both.
- [ ] `test_adjoint_vs_fd.py` (`slow`) consolidating `v18_logc_adjoint_check.py` +
      `v19_adjoint_check_logc.py`; prove pass, `git rm` both.
- [ ] Promote assertion cores of `score_l_eff_sweep.py` + `phase7p3_p0_1_frame_byte_test.py`
      → `test_*_verdict.py`.
- [ ] `jithin_closure_algebra_test.py` → fast unit test.

### E-2: restructure + prune
- [ ] Create `scripts/studies/{drivers,plot,extract}/` each with `__init__.py`.
- [ ] `git mv` locked + canonical drivers → `drivers/`, plotters → `plot/`, `derive/*` → `extract/`.
- [ ] Rewrite imports in the ~10 test files in the table (`scripts.studies.X` →
      `scripts.studies.drivers.X`, etc.).
- [ ] `git rm` the ~120 dead one-offs by block: v18_* (21), v19_* (9), v20/v23/v25/v26_* (9),
      mis-named `test_*` scratch (~13), ClO4 `peroxide_window_4sp*`, `_phase_D_*` throwaways (11),
      `jithin_standalone_mpb_chebyshev.py`, debug/`*_smoke` probes.
- [ ] `git rm -r scripts/Inference/` (13) and `scripts/surrogate/` (17, minus `gradient_benchmark`
      already moved) — paused programs; restart point is a doc.
- [ ] **Validate:** `pytest -m "not slow"` green + `pytest --collect-only` clean (no import errors,
      no `test_*` scripts mis-collected). Do not commit until green.
- [ ] Flag for WS-F: any surviving script that reads a `StudyResults/` path.
- [ ] Commit: `refactor(cleanup): restructure scripts/studies; prune ~150 dead one-offs`.

## WS-F — integration  ·  branch `chore/repo-cleanup`  (serial, after A–E)

**Orchestrator (this session). Owns the records files. Runs after all five branches merge.**
- [ ] Merge `chore/cleanup-{studyresults,docs,artifacts,archive,scripts-tests}` → `chore/repo-cleanup`
      (disjoint scopes ⇒ expect zero conflicts; if `.gitignore` conflicts, only C touched it).
- [ ] Seam re-grep: `grep -rn StudyResults scripts Forward` over survivors; patch stale read paths
      using WS-A's moved-dir list + WS-E's flags.
- [ ] Full `pytest -m "not slow"` green; spot-run one promoted `slow` test.
- [ ] Optionally fold `verification_sweep/` + `.verification/` → `docs/audits/` (cross-scope, so
      done here not in B).
- [ ] Write `REPO_LAYOUT.md` (one-page "where things go").
- [ ] Update `CLAUDE.md` path conventions + reference-driver table to new `scripts/studies/`
      subpaths; record archive off-git location in `README.md`/`CLAUDE.md`.
- [ ] Append `tasks/lessons.md` (git history = the archive; tests import studies → fixtures;
      parallel cleanup = disjoint-subtree ownership).
- [ ] Open PR with per-workstream commits for review before merge to `main`.

---

## Deferred staleness reviews (not blocking; not assigned to a Wave-1 stream)
- `tests/`: v9-gate tests possibly superseded by v10b; `step6` fast/slow duplicate pair;
  `test_water_ionization_phase_6a.py` lone `@pytest.mark.skip`; 9 chained `xfail` in
  `test_jithin_picard_closure.py` — fix-or-remove. (Owned by a follow-up WS-E pass.)
- `bug_reports/` round1–4: mine `12/13_test_*` + `round2_06` for weak-tolerance test hardening
  (e.g. `test_multistart.py:332` `max_err < 1.0`). Keep as backlog.
