# Plan Review: Surrogate Pipeline Roadmap

**Reviewed:** 2026-03-16
**Plan file:** `.plans/surrogate-pipeline-roadmap/PLAN.md`
**Verdict:** APPROVE WITH MINOR REVISIONS

---

## Overall Assessment

This is a well-structured meta-plan that faithfully translates the user's request into an actionable roadmap. The phasing is logical, the dependency DAG is correct, parallelization opportunities are identified, and the research context is used extensively. The plan correctly treats each phase as a stub for a detailed implementation plan (consistent with the "plan for plans" framing). The risk table and open questions are honest and useful.

The issues below are all MINOR. None would derail execution.

---

## Issues

### MINOR-1: Incorrect file path for the BV solver (Phase 2d)

Phase 2d lists an input as `Forward/bv_solver.py`. The actual forward solver is a package at `Forward/bv_solver/` (containing `solvers.py`, `forms.py`, `config.py`, `mesh.py`, `nondim.py`). The detailed plan for Phase 2d will need the correct path to build the coarse-mesh wrapper.

**Suggested fix:** Change `Forward/bv_solver.py` to `Forward/bv_solver/` (the package directory).

### MINOR-2: Training data size inconsistency between research summary and plan

The research summary references "~200 training samples" in several places (Section 2.1, 2.3, decision guide), while the plan correctly uses 3,194 (the merged v9+v11 dataset). This is not an error in the plan itself, but when the Phase 2 detailed plans are written, implementers should be aware that the research summary's GP/PCE "sweet spot" recommendations were calibrated against ~200 samples, not 3,194. With 3,194 samples, exact GP becomes more expensive (O(N^3) ~ 3.2e10 operations for 3,194 points), making the plan's note about SVGP/KISS-GP more important than it might first appear.

**Suggested fix:** Add an explicit note in Phase 2a that 3,194 samples is above the typical exact-GP comfort zone and sparse methods (SVGP) should be the default assumption, not a fallback.

### MINOR-3: PCE / Sobol sensitivity analysis mentioned in open questions but not in any phase

The research summary highlights PCE as providing Sobol sensitivity indices directly from coefficients, which would be valuable for understanding k0_2's identifiability. Open Question 3 in the plan asks whether PCE should be Phase 2f. Given that the user explicitly asked to "identify problems in the training data set" and wants to understand k0_2 sensitivity, a quick Sobol/Morris screening in Phase 1 (using existing surrogates) would strengthen the data audit without adding a full new phase.

**Suggested fix:** Add a bullet to Phase 1 outputs: "Optional: Morris or variance-based sensitivity screening using existing NN ensemble to quantify k0_2's relative influence across the voltage range."

### MINOR-4: No explicit mention of the environment constraint

The user's memory file and project context indicate that the correct Python environment is `venv-firedrake` in the parent directory, not conda. Since multiple phases involve installing new packages (GPyTorch, DeepXDE/NeuralOperator, etc.), each detailed plan will need to account for this. A note at the roadmap level would prevent repeated rediscovery.

**Suggested fix:** Add to the top-level "Approach" section: "All work uses the `venv-firedrake` virtual environment in the parent directory. New dependencies (GPyTorch, etc.) must be installed there."

### MINOR-5: Phase 3 success criterion may be too aggressive

Phase 3 requires "at least one new surrogate achieves < 5% worst-case error (vs current 10.7%)." This is a 2x improvement and may not be achievable from architecture changes alone without data augmentation. If Phase 1 recommends data augmentation but the augmented data isn't ready by Phase 2, new surrogates will train on the same data and may hit similar error floors.

**Suggested fix:** Add a conditional: "If Phase 1 recommends data augmentation AND augmentation is completed before Phase 2, target < 5%. Otherwise, target improvement over current best on the same data, with the understanding that ISMO (Phase 4) will close the remaining gap."

### MINOR-6: Phase 2e dependency is understated

Phase 2e (autograd retrofit) is listed as depending on Phase 1 but the plan notes it "can start almost immediately." This is correct -- it has essentially no dependency on Phase 1 since it modifies the existing NN ensemble's gradient computation, not its training data. The dependency table should reflect this.

**Suggested fix:** Change Phase 2e's "Depends On" to "None (independent of Phase 1)" in the overview table.

---

## Alignment with User Request

| User Request | Plan Coverage | Notes |
|---|---|---|
| Meta-plan (plan for plans) | Covered | Each phase is a stub with "Key decisions for detailed planning" |
| Phase 1: identify training data problems (size, range, gaps, k0_2 focus) | Covered well | Phase 1 is thorough with specific metrics |
| Phase 2: train DeepONet, FNO, PEDS | Covered | Phases 2b, 2c, 2d |
| GP surrogate | Covered | Phase 2a |
| Iterative surrogate refinement | Covered | Phase 4 (ISMO) |
| Assume identifiability | Covered | Plan explicitly notes "take as given" per user instruction |
| PyTorch models use autograd | Covered | Phase 2e + all new surrogates |
| PDE optimization refinement | Covered | Phase 5 (trust-region + weight annealing) |
| Surrogate-PDE back-and-forth | Covered | Phase 4 (ISMO) and Phase 5 (trust-region) |
| Weight annealing | Covered | Phase 5 |

No scope drift detected. The plan does not add phases the user didn't ask for (identifiability analysis is correctly deferred per user instruction). Phase 6 (integration) is a natural capstone that the user didn't explicitly request but is implied.

---

## Research Alignment

The plan makes excellent use of the research context:

- ISMO for iterative refinement (Phase 4) -- directly from research
- GP via GPyTorch as recommended sweet spot (Phase 2a)
- PEDS with coarse solver (Phase 2d) -- from research
- DeepONet and FNO/DIFNO (Phases 2b, 2c) -- from research
- Trust-region surrogate-to-PDE transition (Phase 5) -- from research
- Autograd as "easy fix" gap (Phase 2e) -- from research

No approaches flagged as problematic in the research are pursued. The plan appropriately notes FNO's potential mismatch with the 22-point voltage grid (Phase 2c key decisions).

One research recommendation not incorporated: **regularization (Tikhonov)**. The research summary lists this as High Priority recommendation #3. The plan mentions it only in Phase 5 key decisions as a question. Given the research's emphasis, this could be elevated -- but it's reasonable to defer to the Phase 5 detailed plan.

---

## Summary

The plan is solid and ready for execution. The six minor issues above are refinements, not blockers. The most actionable ones are MINOR-1 (wrong file path -- will cause confusion) and MINOR-2 (GP scalability expectation setting). The plan faithfully captures the user's intent, uses research appropriately, and structures the work in a logical dependency order with good parallelization.
