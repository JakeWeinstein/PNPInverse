---
phase: 12-archive-old-results
verified: 2026-03-12T05:30:00Z
status: passed
score: 5/5 must-haves verified
gaps: []
---

# Phase 12: Archive Old Results Verification Report

**Phase Goal:** Move old StudyResults to archive and remove bad outputs
**Verified:** 2026-03-12T05:30:00Z
**Status:** PASSED
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| #  | Truth                                                                      | Status     | Evidence                                                                    |
|----|----------------------------------------------------------------------------|------------|-----------------------------------------------------------------------------|
| 1  | ~96 old StudyResults directories exist under archive/StudyResults/ intact  | VERIFIED   | `ls archive/StudyResults/ | wc -l` = 100 (96 dirs + 4 .md files)          |
| 2  | StudyResults/v14_pde_only/ no longer exists in the repo                    | VERIFIED   | `test ! -d StudyResults/v14_pde_only` = PASS                               |
| 3  | Temporary artifacts (tmp dirs, loose log files) no longer exist            | VERIFIED   | All 4 tmp/log items confirmed absent from StudyResults/                     |
| 4  | Current results (v13, v14, V&V dirs) remain in StudyResults/               | VERIFIED   | All 7 keep-set directories present; StudyResults/ contains exactly 7 dirs   |
| 5  | Loose .md summary files are archived, not deleted                          | VERIFIED   | All 4 phase summary .md files found in archive/StudyResults/                |

**Score:** 5/5 truths verified

### Required Artifacts

| Artifact                              | Expected                          | Status     | Details                                      |
|---------------------------------------|-----------------------------------|------------|----------------------------------------------|
| `archive/StudyResults/`               | Archive location for old results  | VERIFIED   | 100 items present (96 dirs + 4 .md files)    |
| `StudyResults/master_inference_v13/`  | Kept v13 results (unchanged)      | VERIFIED   | Directory exists at expected path            |
| `StudyResults/v14/`                   | Kept v14 results (unchanged)      | VERIFIED   | Directory exists at expected path            |
| `StudyResults/inverse_verification/`  | Kept V&V results (unchanged)      | VERIFIED   | Directory exists at expected path            |
| `StudyResults/surrogate_fidelity/`    | Kept V&V results (unchanged)      | VERIFIED   | Directory exists at expected path            |
| `StudyResults/mms_convergence/`       | Kept V&V results (unchanged)      | VERIFIED   | Directory exists at expected path            |
| `StudyResults/pipeline_reproducibility/` | Kept V&V results (unchanged)   | VERIFIED   | Directory exists at expected path            |
| `StudyResults/target_cache/`          | Kept cache (unchanged)            | VERIFIED   | Directory exists at expected path            |

### Key Link Verification

| From                                        | To                                                                               | Via                     | Status   | Details                                                                                 |
|---------------------------------------------|----------------------------------------------------------------------------------|-------------------------|----------|-----------------------------------------------------------------------------------------|
| `writeups/vv_report/generate_figures.py`    | StudyResults/inverse_verification, surrogate_fidelity, mms_convergence, master_inference_v13 | file path references | WIRED  | Script resolves STUDY = ROOT / "StudyResults" and reads from all 4 kept subdirectories. No references to archive/ or v14_pde_only. |

### Requirements Coverage

| Requirement | Source Plan | Description                                              | Status    | Evidence                                                            |
|-------------|-------------|----------------------------------------------------------|-----------|---------------------------------------------------------------------|
| ARCH-01     | 12-01-PLAN  | Old StudyResults directories moved to archive/StudyResults/ | SATISFIED | 100 items in archive/StudyResults/ confirmed; sample dirs (master_inference_v2, bv_iv_curve) verified present |
| ARCH-02     | 12-01-PLAN  | StudyResults/v14_pde_only/ removed (bad output)          | SATISFIED | Directory absent from StudyResults/ and not present in archive     |

No orphaned requirements — REQUIREMENTS.md maps only ARCH-01 and ARCH-02 to Phase 12, and both are claimed in 12-01-PLAN frontmatter.

### Anti-Patterns Found

None. This phase involved only file system moves and deletions — no source code was created or modified.

### Human Verification Required

None required. All must-haves are verifiable via file system checks.

### Gaps Summary

No gaps. All 5 observable truths are verified, all 8 artifacts are present and in place, the key link from generate_figures.py to the kept StudyResults subdirectories is wired, and both requirements ARCH-01 and ARCH-02 are satisfied by concrete file system evidence.

Two commits were confirmed in git history at hashes 44cc0a0 (delete bad outputs) and 26fd04f (archive old results), matching the planned two-commit strategy.

StudyResults/ root contains exactly 7 directories and one .DS_Store file — no loose files, no temporary artifacts, no unexpected items.

---

_Verified: 2026-03-12T05:30:00Z_
_Verifier: Claude (gsd-verifier)_
