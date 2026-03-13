# Phase 8: Ablation Audit - Context

**Gathered:** 2026-03-12
**Status:** Ready for planning

<domain>
## Phase Boundary

Determine which v13 pipeline stages are necessary vs redundant, producing an empirically justified minimal pipeline specification. No new pipeline components — measurement and documentation only. Phase 7 baseline (20 seeds) is the reference. Output feeds Phase 9 experiments and Phase 11 pipeline build.

</domain>

<decisions>
## Implementation Decisions

### Ablation configurations
- Treat surrogate stages (S1-S3) as a single block — do not ablate individually
- 4 configurations, all using 20 seeds (0-19) paired with Phase 7 baseline:
  1. **Full v13** (S1-S5 → P1 → P2) — reuse Phase 7 results, no re-run needed
  2. **S4-only → P1 → P2** — multistart 20K LHS replaces cascade/joint warm-up
  3. **S4-only → P2** — skip P1, multistart straight to full cathodic PDE
  4. **Cold-start → P1 → P2** — no surrogate at all, default initial guess into PDE
- S4 multistart uses same 20K LHS grid as v13 (isolate warm-start variable, not grid size)
- S5 (best selection) is implicit logic, not a real stage — not ablated separately

### Statistical comparison method
- **Wilcoxon signed-rank test** on paired per-parameter relative errors across 20 seeds
- **p < 0.05** threshold: stage is "redundant" if removal does NOT significantly worsen results
- **Per-parameter** testing: separate Wilcoxon test for each of k0_1, k0_2, alpha_1, alpha_2 (4 p-values per ablation config)
- Report both **median error** (Wilcoxon) and **worst-case (max) error** (descriptive comparison)

### Justification criteria format (AUDT-03)
- **Table + narrative** in `StudyResults/v14/ablation/`
- Table columns: Stage | Status (justified/redundant/unjustified) | Criterion (literature/empirical/simplest) | Evidence summary
- 1-2 sentence narrative per stage explaining the verdict
- S4 (multistart warm-starting) gets **literature** credit from surrogate-assisted optimization literature
- S1-S3 need **empirical** justification from ablation results or are marked redundant
- S5 listed as "N/A: selection logic" — not a real stage
- P1/P2 assessed by ablation results

### Minimal pipeline spec output
- **Markdown spec document** at `StudyResults/v14/ablation/minimal_pipeline_spec.md`
- Lists only stages that survived ablation with justification for each
- Stages removed if Wilcoxon p >= 0.05 — clean statistical cut
- **Includes timing data** (wall-clock per stage per seed) for cost/benefit analysis
- If ALL surrogate stages are redundant (cold-start matches surrogate-warmed), spec recommends PDE-only pipeline
- Spec only states what survived — no Phase 9 recommendations (clean separation of concerns)
- Not a runnable script — blueprint consumed by Phase 9 experiments and Phase 11 pipeline build

### Claude's Discretion
- Ablation script architecture (new script vs modifying v13 with flags)
- How to implement the cold-start initial guess (zeros, parameter-space center, or other)
- Plot styling for ablation comparison figures
- Narrative tone and detail level in justification table
- Whether to generate comparison box plots or other visualizations

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `scripts/surrogate/Infer_BVMaster_charged_v13_ultimate.py`: Has `--skip-p1` flag already, `--pde-cold-start` flag, and `--strategy` flag for controlling which surrogate stages run — can likely be invoked with different flag combinations for each ablation config
- `scripts/surrogate/Infer_PDE_only_v14.py`: PDE-only variant with `--skip-warmup` flag for cold-start baseline
- `StudyResults/v14/multi_seed/seed_results.csv`: Phase 7 full v13 baseline results (20 seeds) — config 1 is already done
- `Surrogate/multistart.py`: MultiStartConfig/Result with 20K LHS grid — reusable for S4-only configs
- Phase 7 multi-seed wrapper pattern: sequential seed execution to avoid Firedrake/PETSc conflicts

### Established Patterns
- Frozen dataclass config + result pattern for all strategies
- `StudyResults/v14/` subdirectory organization with CSV + PNG + metadata.json
- Print-based logging with `[tag]` prefixes
- AUDT-04 metadata.json sidecar for justification (tool name, justification type, reference, rationale)

### Integration Points
- Ablation results consumed by Phase 9 (objective redesign) and Phase 11 (pipeline build)
- Minimal pipeline spec read by Phase 9 planner to know which stages are the baseline
- All outputs in `StudyResults/v14/ablation/` directory

</code_context>

<specifics>
## Specific Ideas

- Surrogate stages (S1-S3) treated as one block — the question is "does the cascade/joint warm-up before multistart help?" not "does S2 vs S3 matter?"
- Phase 7 baseline already exists for 20 seeds — no need to re-run full v13, just run the 3 new configs
- v13 script already has `--skip-p1` and `--pde-cold-start` flags that may cover some configs directly

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 08-ablation-audit*
*Context gathered: 2026-03-12*
