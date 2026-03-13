# Phase 12: Archive Old Results - Context

**Gathered:** 2026-03-12
**Status:** Ready for planning

<domain>
## Phase Boundary

Move old StudyResults directories to `archive/StudyResults/`, delete bad outputs and temporary artifacts, and keep current results (v13, v14, V&V) in their original locations. Requirements: ARCH-01, ARCH-02.

</domain>

<decisions>
## Implementation Decisions

### Keep vs Archive Criteria
- **Keep in place (7 dirs):** master_inference_v13, v14, inverse_verification, surrogate_fidelity, mms_convergence, pipeline_reproducibility, target_cache
- **Archive (~105 dirs):** Everything else in StudyResults/ moves to `archive/StudyResults/`
- **Delete outright:** v14_pde_only/ (bad output, ARCH-02), tmp/, tmp_replay_diag_smoke/, tmp_replay_diag_smoke2/, convergence_study_summary.txt, master_inference_v5_log.txt
- Loose .md summary files (phase3-6 summaries) get archived, not deleted

### Archive Structure
- Flat dump into `archive/StudyResults/` — no subdirectory grouping
- Original directory names preserved
- No manifest or README — directory names are self-documenting, git history tracks the move

### Deletion Policy
- Only v14_pde_only/ (required by ARCH-02) and temporary/debugging artifacts (tmp dirs, loose log files) are deleted
- Everything else of historical value is archived, not deleted

### Git Handling
- Two commits: (1) delete bad outputs and tmp artifacts, (2) move old dirs to archive
- Repo size is not a concern — goal is working directory cleanliness, not history reduction

### Claude's Discretion
- Exact ordering of move operations
- Any .gitkeep or similar handling for empty directories
- Verification steps within the plan

</decisions>

<specifics>
## Specific Ideas

No specific requirements — straightforward organizational task following roadmap definitions.

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- No code assets needed — this is a file-move/delete operation

### Established Patterns
- StudyResults/ is the standard output directory for all pipeline runs
- No existing archive/ directory — will be created fresh

### Integration Points
- Scripts may reference StudyResults/ paths — but only v13/v14/V&V results are referenced by current code
- writeups/vv_report/generate_figures.py reads from StudyResults/ (inverse_verification, surrogate_fidelity, mms_convergence, pipeline_reproducibility) — all in the keep set

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 12-archive-old-results*
*Context gathered: 2026-03-12*
