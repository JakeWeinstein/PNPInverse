# Handoff 27 — Phase 6β parallel planning while Phase 6α sweep finishes

**Status basis:** Prepared after reading
`docs/CHATGPT_HANDOFF_26_phase6a_outcome_and_phase6b_scoping.md` and
re-inspecting local artifacts under
`StudyResults/fast_realignment_2026-05-08/`.  The handoff-26 warning
to re-read the run log was correct: the current output directory is a
mixed stale/new state.  Do not scope Phase 6β from the aggregate
`summary.json` / `verdict.json` in that directory without first
repairing or rerunning the Phase 6α sweep.

## 1. Artifact status — what is trustworthy

### Trustworthy Phase 6α artifacts

Only the two 100 µm per-combo files were refreshed by the Phase 6α run:

* `StudyResults/fast_realignment_2026-05-08/l_eff_transport_sweep/L100um_ratio_1e-18/iv_curve.json`
  * file timestamp inspected: May 9 2026, 21:07
  * 13/13 converged
  * cd at `V_RHE=-0.40`: `-0.7373038942816692 mA/cm²`
  * max surface pH proxy across grid: about `10.63`
  * no cathodic peak
* `StudyResults/fast_realignment_2026-05-08/l_eff_transport_sweep/L100um_ratio_1e-30/iv_curve.json`
  * file timestamp inspected: May 9 2026, 21:15
  * 13/13 converged
  * cd at `V_RHE=-0.40`: `-0.43989380672883305 mA/cm²`
  * max surface pH proxy across grid: about `10.63`
  * no cathodic peak

The `ratio=1e-18` case matches Handoff 26's reported combo-1 numbers,
so water self-ionization was active despite the older cosmetic config
serialization bug.

### Stale artifacts

The following are stale pre-Phase-6α or mixed outputs and should not be
used as final evidence:

* `StudyResults/fast_realignment_2026-05-08/l_eff_transport_sweep/summary.json`
  * timestamp inspected: May 9 2026, 18:12
  * predates the Phase 6α per-combo rewrites
* `StudyResults/fast_realignment_2026-05-08/l_eff_transport_sweep/verdict.json`
  * still reports old H+ Levich-limited values, e.g. `-0.089865...`
    mA/cm² at 100 µm and surface pH around 14
* `L66um_*`, `L21um_*`, and `L16um_*` `iv_curve.json`
  * timestamps inspected: May 9 2026, 17:49-18:12
  * these are old pre-water-ionization files

The Phase 6α run log
`StudyResults/fast_realignment_2026-05-08/l_eff_transport_sweep_phase6a_run.log`
ends shortly after starting the 66 µm case, so the run appears to have
stopped after the two 100 µm combos.

## 2. What the valid 100 µm Phase 6α data says

Water self-ionization did the intended first job: it broke the pure H+
Levich ceiling and lifted current substantially.  It did not produce
the experimental shape.

For `L_eff=100 µm`, `ratio=1e-18`:

| V_RHE | cd mA/cm² | surface pH proxy |
|---:|---:|---:|
| -0.4000 | -0.7373 | 10.6101 |
| -0.3208 | -0.5450 | 10.5539 |
| -0.2417 | -0.4553 | 10.5585 |
| -0.1625 | -0.4338 | 10.6331 |
| -0.0833 | -0.3127 | 10.5430 |
| -0.0042 | -0.1537 | 10.0935 |
| +0.0750 | -0.1014 | 9.4360 |
| +0.1542 | -0.0915 | 8.6768 |
| +0.2333 | -0.0901 | 7.8956 |
| +0.3125 | -0.0898 | 7.1242 |
| +0.3917 | -0.0890 | 6.3884 |
| +0.4708 | -0.0851 | 5.6984 |
| +0.5500 | -0.0661 | 5.0599 |

For `ratio=1e-30`, the deeply cathodic plateau is softer
(`-0.4399 mA/cm²`) but the same qualitative shape remains: no peak,
no decay, pH still around 10.6 under deepest cathodic load.

Interpretation:

* Phase 6α removed a numerical/physics gate but did not add the missing
  shape mechanism.
* The model still lacks the deck's cathodic peak around `+0.10 V_RHE`
  and decay at more cathodic potentials.
* The pH ceiling from water alone is too alkaline for the intended
  deck/data operating window.

## 3. Immediate operational next step

Finish Phase 6α cleanly before using smallest-L conclusions.

Recommended approach:

1. Do not rerun into the existing mixed
   `l_eff_transport_sweep/` directory unless it is deliberately archived
   or cleaned first.
2. Prefer a new output directory such as
   `StudyResults/fast_realignment_2026-05-08/l_eff_transport_sweep_phase6a/`.
3. Either add `--out-dir`, `--resume`, and/or `--only-l-eff` flags to
   `scripts/studies/l_eff_transport_sweep_csplus_so4.py`, or run a
   dedicated short script that only computes the missing 66/21/16 µm
   Phase 6α combos.
4. After the sweep is complete, rerun:
   * `scripts/studies/plot_l_eff_transport_sweep.py`
   * `scripts/studies/score_l_eff_sweep.py`
5. Treat any aggregate `summary.json` / `verdict.json` as invalid unless
   its timestamp is later than all eight Phase 6α per-combo files.

## 4. Planning can proceed in parallel, but keep this gate

The core Phase 6β question should not be "how do we add sulfate?"  It
should be:

> Does sulfate/bisulfate chemistry, when coupled to the current
> acid-form ORR law, actually create the observed peak and decay, or
> does it merely raise local H+ and amplify cathodic current?

This is the main caution.  The existing acid-form BV reactions include
`(c_H / c_H_ref)^n` cathodic concentration factors.  Under that law,
raising surface `c_H` from pH 10 toward pH 6-7 generally strengthens
the cathodic rate ceiling, not weakens it.  Sulfate buffering is still
physically relevant and likely required for local pH, but it is not
automatically a current-throttling mechanism in the present model.

Therefore Phase 6β should start with a scoping spike that validates the
sign and shape before committing to a full implementation.

## 5. Recommended Phase 6β scoping spike

### Goal

Create the cheapest defensible prototype that answers:

* Does a fast `HSO4- ⇌ SO4-- + H+` equilibrium closure reduce, cap, or
  amplify current under the existing acid-form BV rate law?
* Does it create any local minimum/peak in cd vs `V_RHE` near
  `+0.10 V_RHE`?
* Does it bring surface pH into the 4-7 or 4-9 target band without
  overshooting current magnitude?

### Suggested prototype order

1. Build a zero-DOF/local algebra check first.
   * Use the existing `c_H_surface_nondim` from the valid 100 µm Phase
     6α curves.
   * Apply sulfate/bisulfate equilibrium algebra offline.
   * Estimate the implied change to acid-form BV concentration factors.
   * If the sign is wrong, document that 6β alone cannot produce the
     deck peak under the current acid-form kinetics.
2. If the sign is plausible, implement a guarded form-level prototype.
   * Mirror the Phase 6α pattern: default-off config flag, helper
     module, no disabled-path drift, slow regression for disabled
     byte-equivalence.
   * Add a continuation ladder for buffer activation, analogous to
     `kw_eff_ladder`.
3. Run a small voltage subset first.
   * `L_eff=100 µm`, both ratios, maybe 5-7 voltages around
     `[-0.20, +0.25] V_RHE`.
   * Only then spend runtime on the full L sweep.

### Acceptance gates for the spike

* Disabled path matches Phase 6α.
* Phase 6α with buffer disabled reproduces the two valid 100 µm curves.
* Enabling buffer moves surface pH in the intended direction.
* Enabling buffer does not simply increase cathodic current everywhere.
* A peak or at least a slope reversal appears near the deck peak window;
  otherwise mark 6δ as required.

## 6. Implementation shape if 6β proceeds

Prefer a mass-conserving closure over an ad hoc H+ source.

Do not add `R_buf = k * (c_HSO4 - c_SO4 * c_H / K_a)` only to the H+
residual unless SO4/HSO4 bookkeeping is handled explicitly.  That risks
creating or destroying sulfate/proton equivalents without a matching
conserved variable.

A safer 6β.1 design should have:

* `enable_sulfate_buffering=False` default.
* A new helper module, likely `Forward/bv_solver/sulfate_buffering.py`.
* Config keys for:
  * `enable_sulfate_buffering`
  * `ka2_hat` or dimensional `Ka2` converted centrally
  * total sulfate/bisulfate inventory
  * activation/continuation factor
  * optional diffusivity/steric size for HSO4-
* Explicit statement of conserved variable(s), for example:
  * total sulfate if using algebraic speciation
  * proton condition coupled to sulfate acid-base state
* Poisson charge includes SO4-- and HSO4- separately.
* Bikerman packing includes the selected sulfate species consistently.
* Continuation in `solve_anchor_with_continuation`, analogous to the
  Phase 6α `kw_eff_ladder`.

Open architecture choice:

* Algebraic equilibrium closure is the fastest 6β.1 path and is likely
  chemically defensible because sulfate proton transfer is fast.
* Fully dynamic HSO4- NP species is more faithful but should be 6β.2
  unless the algebraic closure gives the wrong shape or breaks solver
  robustness.

## 7. Data targets and provenance

Use the local audit
`docs/seitz_mangan_data_folder_audit_2026-05-08.md` as the source map.

High-value direct data already present:

* `data/EChem Reactor Modeling-Seitz-Mangan/Brianna/0,1M K2SO4 data 8-15-19.xlsx`
  * full RRDE LSV at six pH values
  * columns include `E_disk (V vs RHE)`, `j_disk`, `j_ring`, `H2O2%`,
    `n_e`, `Overpotential`
* `data/EChem Reactor Modeling-Seitz-Mangan/Brianna/20201024 CP Experiment Data-Code/CP_data.csv`
  * cation/pH/hold-voltage CP summary
* `data/EChem Reactor Modeling-Seitz-Mangan/Brianna/20201024 CP Experiment Data-Code/Summary Data-Error.xlsx`
  * error bars for CP holds
* `data/EChem Reactor Modeling-Seitz-Mangan/Linsey/ButlerVolmer MATLAB/Brianna_ORR_Data.mat`
  * single `ORR_K2SO4_pH6_cyc1` curve

Important missing target:

* `Tafel slope analysis cation-pH-Li-K-Cs.xlsx`
  * not present in the local folder
  * Handoff/audit says Yash's plotting workflow used the Cs+ pH 4
    column from this workbook
  * without it, matching is against a derived figure rather than the
    underlying fit target

Deck-shape targets to keep explicit:

* cathodic onset around `+0.15 V_RHE`
* peak around `+0.10 V_RHE`
* plateau/left magnitude around `-0.18 mA/cm²`
* peak magnitude around `-0.40 mA/cm²` where applicable
* decay at more cathodic potentials
* cation identity changes peak height/shape
* surface pH should remain in the experimentally plausible operating
  window, roughly 4-9 and ideally 4-7 for the relevant deck comparison

## 8. When to scope Phase 6δ

If the 6β spike shows that sulfate buffering raises H+ and amplifies
acid-form ORR without producing a peak/decay, then Phase 6δ should move
from "maybe later" to "required for shape."

6δ candidate:

* alkaline-form ORR or local-pH-dependent BV stoichiometry/rate switch
* transition from acid-form H+-consuming rate law to alkaline OH-/water
  form as local pH rises
* this is more likely than sulfate alone to create cathodic decay past
  the peak under strong alkaline surface conditions

Do not start with full cation-OHP physics (6γ) until the sulfate/pH
and acid-vs-alkaline kinetic form questions are separated.  Cation
identity matters for the final deck story, but it is a second-order
shape modulator relative to getting the pH/kinetic-regime switch right.

## 9. Concrete next task list for Claude Code

1. Add non-destructive sweep controls to
   `scripts/studies/l_eff_transport_sweep_csplus_so4.py`:
   * `--out-dir`
   * optionally `--only-l-eff` / `--only-ratio`
   * optionally `--resume`
2. Rerun only the missing Phase 6α combos into a clean output directory.
3. Regenerate plot and verdict from the clean directory.
4. Write a short sulfate-buffer algebra spike:
   * input: valid Phase 6α 100 µm `c_H_surface_nondim` / pH curves
   * output: predicted sign of rate-factor change under sulfate/bisulfate
     equilibrium
5. Based on the spike:
   * if sign/shape is plausible, implement guarded 6β.1 algebraic closure
   * if sign is wrong, write Phase 6δ plan before touching solver forms

## 10. Commands useful for re-checking state

```bash
for f in StudyResults/fast_realignment_2026-05-08/l_eff_transport_sweep/*/iv_curve.json; do
  stat -f '%Sm %N' "$f"
done
```

```bash
jq -r '"\(.l_eff_m*1000000|round) um ratio \(.ratio): cd[-0.4]=\(.cd_mA_cm2[0]) pH[-0.4]=\(.surface_pH_proxy[0]) converged=\(.n_converged)/\(.n_total)"' \
  StudyResults/fast_realignment_2026-05-08/l_eff_transport_sweep/*/iv_curve.json
```

```bash
jq '.' StudyResults/fast_realignment_2026-05-08/l_eff_transport_sweep/verdict.json
```

If the first command shows mixed 18:xx and 21:xx timestamps, the
directory is still mixed and aggregate verdicts remain invalid.
