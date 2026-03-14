---
phase: quick-1
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - scripts/Inference/Infer_BVMaster_charged_v13_ultimate.py
  - scripts/studies/run_multi_seed_v13.py
autonomous: true
requirements: [MOVE-01]
must_haves:
  truths:
    - "v13 pipeline script lives at scripts/Inference/Infer_BVMaster_charged_v13_ultimate.py"
    - "run_multi_seed_v13.py finds and invokes the v13 script at its new location"
    - "_ROOT still resolves to PNPInverse/ root (same directory depth maintained)"
    - "Old copy no longer exists in scripts/surrogate/"
  artifacts:
    - path: "scripts/Inference/Infer_BVMaster_charged_v13_ultimate.py"
      provides: "v13 master inference pipeline"
    - path: "scripts/studies/run_multi_seed_v13.py"
      provides: "Updated path reference to Inference/ instead of surrogate/"
  key_links:
    - from: "scripts/Inference/Infer_BVMaster_charged_v13_ultimate.py"
      to: "scripts/_bv_common.py"
      via: "_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR)) -- still 2 levels deep"
      pattern: "from scripts._bv_common import"
    - from: "scripts/studies/run_multi_seed_v13.py"
      to: "scripts/Inference/Infer_BVMaster_charged_v13_ultimate.py"
      via: "v13_script_path default_factory"
      pattern: "Inference.*Infer_BVMaster"
---

<objective>
Move the v13 master inference pipeline script from scripts/surrogate/ to a new scripts/Inference/ directory, updating all internal references and external callers.

Purpose: Organize inference scripts into their own dedicated directory, separating inference from surrogate training/building utilities.
Output: Script relocated with all path references updated and working.
</objective>

<execution_context>
@./.claude/get-shit-done/workflows/execute-plan.md
@./.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@scripts/surrogate/Infer_BVMaster_charged_v13_ultimate.py
@scripts/studies/run_multi_seed_v13.py
@scripts/_bv_common.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Move v13 script to scripts/Inference/ and update all references</name>
  <files>
    scripts/Inference/Infer_BVMaster_charged_v13_ultimate.py
    scripts/studies/run_multi_seed_v13.py
  </files>
  <action>
1. Create the `scripts/Inference/` directory.

2. Move (git mv) the v13 script:
   `git mv scripts/surrogate/Infer_BVMaster_charged_v13_ultimate.py scripts/Inference/Infer_BVMaster_charged_v13_ultimate.py`

3. Delete the stale __pycache__ entry:
   `rm -rf scripts/surrogate/__pycache__/Infer_BVMaster_charged_v13_ultimate.cpython-312.pyc`

4. Update the docstring usage examples inside the moved script. Replace all occurrences of `scripts/surrogate/Infer_BVMaster_charged_v13_ultimate.py` with `scripts/Inference/Infer_BVMaster_charged_v13_ultimate.py` (there are 6 occurrences in lines 20-33 of the docstring).

5. Update `scripts/studies/run_multi_seed_v13.py` line 69: change the path segment `"surrogate"` to `"Inference"` in the `v13_script_path` default_factory lambda inside the `MultiSeedConfig` dataclass.

IMPORTANT: Do NOT change the `_THIS_DIR` / `_ROOT` path resolution logic in the v13 script. The new location `scripts/Inference/` is still exactly 2 levels deep under PNPInverse/, so `_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))` still correctly resolves to the project root. The `_bv_common.py` docstring explicitly documents this "scripts are 2 levels deep" contract.

Do NOT touch any other scripts in scripts/surrogate/ -- they remain where they are (build_surrogate.py, train_nn_surrogate.py, etc. are surrogate-building tools, not inference).
  </action>
  <verify>
    <automated>cd /Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse && test -f scripts/Inference/Infer_BVMaster_charged_v13_ultimate.py && ! test -f scripts/surrogate/Infer_BVMaster_charged_v13_ultimate.py && grep -q "scripts/Inference" scripts/Inference/Infer_BVMaster_charged_v13_ultimate.py && grep -q '"Inference"' scripts/studies/run_multi_seed_v13.py && echo "PASS"</automated>
  </verify>
  <done>
    - scripts/Inference/Infer_BVMaster_charged_v13_ultimate.py exists with updated docstring paths
    - scripts/surrogate/Infer_BVMaster_charged_v13_ultimate.py no longer exists
    - run_multi_seed_v13.py references "Inference" not "surrogate" for v13_script_path
    - No other files modified
  </done>
</task>

</tasks>

<verification>
- `test -f scripts/Inference/Infer_BVMaster_charged_v13_ultimate.py` -- file exists at new location
- `! test -f scripts/surrogate/Infer_BVMaster_charged_v13_ultimate.py` -- old location cleaned up
- `grep -c "scripts/Inference" scripts/Inference/Infer_BVMaster_charged_v13_ultimate.py` -- docstring updated (should return 6)
- `grep "Inference" scripts/studies/run_multi_seed_v13.py` -- caller updated
- `python -c "import ast; ast.parse(open('scripts/Inference/Infer_BVMaster_charged_v13_ultimate.py').read()); print('syntax ok')"` -- script parses without error
</verification>

<success_criteria>
- v13 inference script relocated to scripts/Inference/ with correct internal path references
- All external callers (run_multi_seed_v13.py) updated to find the script at its new location
- _ROOT resolution still works (2-level depth maintained)
- No broken imports or path references
</success_criteria>

<output>
After completion, create `.planning/quick/1-make-scripts-inference-directory-and-mov/1-SUMMARY.md`
</output>
