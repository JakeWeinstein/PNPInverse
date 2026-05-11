1. **`k_des` detailed-balance rule is wrong.**  
WHAT: §3.3 says `k_des = k_hyd / 10^(pKa_eff)`.  
WHY: Singh pKa constrains hydrolysis/protonation equilibrium, i.e. something like `k_hyd/k_prot` after concentration and nondim scaling. It does not constrain MOH desorption to bulk. Using this would assign a fake literature value to the wrong kinetic branch.  
WHAT TO DO: Remove this route for `k_des`. If using pKa, use it only to audit `k_hyd/k_prot` consistency. Treat `k_des` as residence/desorption/mass-transfer unless a source explicitly reports that process.

2. **No nondimensionalization map for `k_des`.**  
WHAT: §3.3 talks about diffusion/Eyring physical rates but never defines how a physical `s^-1`, `m/s`, or mass-transfer coefficient maps to `k_des_nondim`.  
WHY: Any literature-derived `k_des` will be dimensionally arbitrary. `D/δ_OHP` is also not a first-order surface desorption rate as written.  
WHAT TO DO: Add the exact dimensional equation and scaling used by the residual before any `k_des` anchor can be accepted.

3. **Solver default `k_des` is not updated.**  
WHAT: D4 updates `SMOKE_KINETICS`, but `Forward/bv_solver/cation_hydrolysis.py` still has `raw_cfg.get("k_des", 1.0)`.  
WHY: Any caller enabling cation hydrolysis but omitting `k_des` silently uses the old smoke value.  
WHAT TO DO: Define `K_DES_NONDIM_V10B` in the solver/common layer and use it in the bundle default, docs, tests, and `__all__`.

4. **Step 6 A0 audit is hardcoded to the v10a A.2 JSON.**  
WHAT: `phase6b_step6_plumbing_ablation.py` compares A0 to `StudyResults/phase6b_v10a_phase_A2_v_kin/phase_a2_v_kin.json`.  
WHY: Running with `--out-subdir phase6b_v10b_step6...` still audits against the old baseline, so D6 cannot prove byte-equivalence to v10b A.2.  
WHAT TO DO: Add a CLI/config option for the A.2 baseline JSON, default it appropriately, and point v10b to `phase6b_v10b_phase_A2_v_kin`.

5. **D6 says compare `R_net`, but the existing audit does not.**  
WHAT: D6 requires byte-equivalence in `cd, R_net, σ_S, θ`; the current audit compares gamma/theta/sigma/cd and branch currents, not `R_net`.  
WHY: The stated contract is not actually tested.  
WHAT TO DO: Add `R_net_unperturbed` or the λ=1 `R_net` diagnostic to the audit keys and tests.

6. **`SMOKE_* = V10B_*` aliases destroy historical reproducibility.**  
WHAT: D3/D4 map `GAMMA_MAX_HAT_SMOKE` and `SMOKE_KINETICS` to v10b values.  
WHY: v9/v10a scripts and tests that import “smoke” will rerun with v10b physics while retaining historical names. That is provenance corruption, not compatibility.  
WHAT TO DO: Keep frozen historical `*_SMOKE` / `*_V10A` constants, introduce `*_V10B`, and update only v10b production drivers/defaults.

7. **Alias/export/test plan is incomplete.**  
WHAT: D3 does not mention updating `__all__`, module export tests, slow tests, and imports that currently assert `GAMMA_MAX_HAT_SMOKE`.  
WHY: New `GAMMA_MAX_HAT_V10B` can exist but remain absent from the public surface, while tests still validate the old surface.  
WHAT TO DO: Add explicit tests that both historical and v10b constants export correctly and that production defaults use v10b.

8. **The k_des bracket diagnostic contradicts itself.**  
WHAT: Phase D3 says loop `k_des` at `k_hyd_baseline = 1e-3`, then says use `θ(k_hyd=1e-1)` as the diagnostic.  
WHY: Those are different solves. Also, at cap saturation θ may be nearly insensitive to `k_des`; `R_net = k_des Γ` is more sensitive.  
WHAT TO DO: Run both baseline and cap-saturated rungs, or explicitly choose `k_hyd_route = 1e-1`; report θ and `R_net`.

9. **No coupled `Γ_max × k_des` sensitivity.**  
WHAT: Γ_max and k_des are swept independently.  
WHY: In `Γ_ss`, `k_des` and `F0/Γ_max` share the denominator; independent sweeps miss ridges and compensating parameter pairs.  
WHAT TO DO: Add at least a small 3×3 matrix around final values, evaluated at baseline and cap-saturated `k_hyd`.

10. **D5 regression tolerance can reject valid calibration.**  
WHAT: It requires v10b `cd` and `x_2e` within ±20% of v10a and says failures trigger step-4/5/6 rework.  
WHY: A real literature change to Γ_max/k_des may legitimately move selectivity/source strength. Selectivity gap is explicitly not a v10b pass criterion.  
WHAT TO DO: Split hard gates into convergence/mass-balance/invariant gates versus informative physics deltas. Do not escalate solely on expected physics movement.

11. **C_S sweep pass criteria are underdefined and likely brittle.**  
WHAT: D7 requires no sign flips and monotonic σ_S.  
WHY: Near-zero R4e with `K0_R4e_factor=1e-14` can flip from numerical noise, and σ_S monotonicity is not guaranteed in the coupled PNP-Stern solve.  
WHAT TO DO: Add magnitude floors, sign conventions, and a physically justified monotonic/continuity criterion.

12. **R3 contradicts D7.**  
WHAT: D7 requires 4/4 C_S rungs converge; risk R3 allows falling back to 3/4 with a note.  
WHY: The executor can “pass” a failed required rung without a rule change.  
WHAT TO DO: Decide now: either 4/4 is mandatory, or define the exact acceptable degraded-pass condition and required critique/escalation.

13. **C_S legacy audit is stale and too broad.**  
WHAT: §3.1 says 12 call-sites, but grep shows many more 0.10 occurrences across scripts/tests/docs.  
WHY: The listed audit will miss files and may globally mutate historical/legacy studies.  
WHAT TO DO: Start from a fresh grep, classify every occurrence, and defer non-v10b script migrations unless each has its own smoke verification.

14. **Updating legacy C_S scripts in Phase E is scope creep.**  
WHAT: Phase E changes many old drivers after v10b regression is done.  
WHY: This broadens blast radius after validation and is not covered by `pytest -k "phase6b or cation"`.  
WHAT TO DO: Move it to a separate cleanup task, or run per-script smoke tests and include them in DoD.

15. **D1 “engineering_choice=True flag” has no schema.**  
WHAT: The plan never says whether this flag lives in code constants, JSON metadata, docs only, or solver config.  
WHY: Downstream cannot programmatically distinguish literature anchors from priors.  
WHAT TO DO: Define a `v10b_calibration` metadata block in result JSON/docs, with per-parameter `value`, `source_type`, `engineering_choice`, `citation`, and `bracket`.

16. **“Data-constrained in Phase D” is not justified.**  
WHAT: D1 says engineering choices get a “data-constrained in Phase D” note.  
WHY: The locked chronology says Phase D is K-only Δβ fit, not necessarily Γ_max/k_des fitting.  
WHAT TO DO: Say “open parameter for Phase D only if Phase D scope is expanded”; otherwise keep it as post-v10b/open.

17. **Γ_max decision rule is too naive.**  
WHAT: “Any peer-reviewed source within factor ~3 locks the cited value.”  
WHY: A capacitance-derived cation packing number, hydrated-radius estimate, adsorbed MOH coverage, and OHP site density are not interchangeable.  
WHAT TO DO: Require mechanism/electrode/electrolyte compatibility and dimensional equivalence to Γ_MOH before locking.

18. **The plan asserts likely literature outcome without evidence.**  
WHAT: §3.2 says literature will likely confirm 1 monolayer and v10b is mostly citation tightening.  
WHY: That biases the search and weakens D1.  
WHAT TO DO: Remove the prediction. Let the decision tree decide.

19. **Result JSON provenance remains named `smoke_kinetics`.**  
WHAT: A.2, step 6, and v-sweep config payloads still write `"smoke_kinetics"`.  
WHY: v10b outputs will claim smoke kinetics even when using calibrated values.  
WHAT TO DO: Write `"v10b_kinetics"` and optionally duplicate `"smoke_kinetics"` only as a deprecated compatibility field.

20. **New bracket driver lacks test requirements.**  
WHAT: D7 creates `phase6b_v10b_cs_bracket.py`, but D8 only runs existing `phase6b or cation` tests.  
WHY: The new driver can have broken CLI/output schema and still pass unit tests if the long run is skipped or partial.  
WHAT TO DO: Add fast tests for parse/schema/target grid and require the actual driver JSON in D7.

VERDICT: ISSUES_REMAIN