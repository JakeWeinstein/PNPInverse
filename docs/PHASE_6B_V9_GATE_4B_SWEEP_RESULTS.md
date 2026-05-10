# Phase 6β v9 Gate 4B — sensitivity sweep results + cache optimization

**Date:** 2026-05-10
**Driver:** `scripts/studies/phase6b_v9_gate4_finite_hydrolysis_smoke.py`
**Outputs:** `StudyResults/phase6b_v9_gate4_smoke/iv_curve.json`,
`u_warmstart_at_v_target.npz`
**Outcome:** ✅ **All 9 sweep combinations converged.** Architecture
verdict: PASS. Calibration verdict: open (smoke values are
intentionally tame for convergence; deck-magnitude calibration is
6β.2 work).

---

## Cache optimization

The Gate 4 smoke pipeline is three steps:

1. **Anchor** at V=+0.55 V (kw_eff ladder + k0 ladder, λ=0).
2. **Warm-walk** at λ=0 from +0.55 V → −0.40 V across 11 V_RHE
   points (this is the expensive one — ~7 min on Ny=80).
3. **λ ramp** at V=−0.40 V (0 → 0.25 → 0.50 → 0.75 → 1.0).

Steps 1+2 are independent of the cation-hydrolysis sweep
parameters because λ-modulation byte-zeros every cation-hydrolysis
contribution at λ=0. Without optimization, every (r_H_El, C_S, k_des)
combination re-paid ~9 min of anchor + walk; with 9 combinations,
that's ~80 min of redundant work.

**Optimization:** run anchor + walk **once** at baseline, snapshot U
at V=−0.40 V, and warm-start every combination's λ ramp from the
cached snapshot. Implementation has three pieces:

### 1. Live `r_H_El_pm` Function

`r_H_El` was previously baked into a Constant at form-build. Now
it's an R-space `Function` on the bundle, mutable via the new
accessor. This lets the r_H_El sweep run on a single ctx without
form rebuild.

* `Forward/bv_solver/cation_hydrolysis.py::CationHydrolysisBundle`
  gains `r_H_El_pm_func: Any`.
* `build_cation_hydrolysis_terms` constructs and assigns it.
* `_build_singh_2016_eq_4_pka_shift` uses it via UFL division
  (instead of a baked Python float in a Constant).
* `Forward/bv_solver/anchor_continuation.py::set_reaction_r_H_El_pm_model`
  is the new accessor (mirrors the kinetic-rate setter pattern).

### 2. `solve_lambda_ramp_from_warm_start`

New top-level orchestrator helper in `anchor_continuation.py`. Takes
a cached U snapshot + sp + λ ladder, builds a fresh ctx, restores U,
optionally re-converges via SS solve (for parameter changes that
shift the FE residual), then walks the λ ladder with outer Picard
for Γ at each rung.

```python
result = solve_lambda_ramp_from_warm_start(
    sp_at_voltage,
    mesh=mesh,
    U_warmstart=cached_snapshot,
    k0_targets={0: K0_HAT_R2E, 1: K0_HAT_R4E},
    lambda_hydrolysis_ladder=(0.0, 0.25, 0.50, 0.75, 1.0),
    parameter_overrides={...},   # optional, in addition to sp config
    reconverge_at_ss=True,
)
```

Returns the same `AnchorContinuationResult` shape as
`solve_anchor_with_continuation`, so caller plumbing stays uniform.

### 3. U snapshot persistence

The cache snapshot is `tuple(d.data_ro.copy() for d in U.dat)` — a
tuple of numpy arrays, one per mixed-space subspace. Driver pickles
via `np.savez` to `u_warmstart_at_v_target.npz` after the first walk
and re-loads on subsequent runs. Skips the 9-min walk entirely once
the file exists.

### Wall-time savings

| Path | Per-combo wall | Total for 9 combos |
|---|---|---|
| Cold (no cache) | ~12 min | ~108 min |
| Warm (cache + ramp) | ~37 s | ~9 min cache + 5.4 min sweep = **14 min** |
| Warm (snapshot pre-loaded) | ~37 s | **5.4 min sweep only** |

**~94 min saved on the first full sweep, ~99 min on every subsequent
sweep.**

---

## Sweep results

Configuration per combination:

* Stack: 4sp dynamic K2SO4 + sulfate analytic Bikerman + Phase 6α
  water ionization
* Mesh: Ny=80, L_eff = 16 µm
* Voltage: V_RHE = −0.40 V (Gate 2 SUCCESS target)
* Kinetic baselines: `k_hyd = 1e-3, k_prot = 1e-3, k_des = 1.0`
  (all nondim) — intentionally tame for convergence
* Singh form: `singh_2016_eq_4` with `r_H_El_pm = 200.98` (Cu prior)
* λ ladder: (0.0, 0.25, 0.50, 0.75, 1.0)
* Picard tolerance: 1e-4 relative on Γ; cap 8 iterations per rung

### Results table

```
label              r_H_El   C_S    k_des   converged   Γ_final    cd
                   (pm)    (F/m²) (nondim)             (nondim)  (mA/cm²)
─────────────────────────────────────────────────────────────────────────
baseline           200.98   0.10   1.0     True        0.5552    -5.532
r_H_El = 180       180.00   0.10   1.0     True        0.5703    -5.532
r_H_El = 195       195.00   0.10   1.0     True        0.5593    -5.532
r_H_El = 215       215.00   0.10   1.0     True        0.5462    -5.532
r_H_El = 250       250.00   0.10   1.0     True        0.5263    -5.532
C_S = 0.05         200.98   0.05   1.0     True        0.3131    -5.532
C_S = 0.20         200.98   0.20   1.0     True        1.3215    -5.532
k_des = 0.1        200.98   0.10   0.1     True        5.5519    -5.532
k_des = 10.0       200.98   0.10   10.0    True        0.0555    -5.532
```

All 9 converged in ≤ 45 s of λ-ramp wall; total sweep wall **5.4 min**.

---

## Interpretation

### 1. Picard formula sanity check (k_des sweep)

The closed-form Γ_ss prediction for the smooth-blend λ residual is

```
Γ_ss(λ=1) = k_hyd · ⟨c_M · 10^(−ΔpKa)⟩
            /  (k_des + k_prot · ⟨c_H⟩ / δ_OHP)
```

For the smoke baselines the `k_prot · ⟨c_H⟩ / δ_OHP` term is
negligible, so we expect `Γ × k_des = const`:

```
k_des = 0.1   →  Γ = 5.5519   →  Γ·k_des = 0.5552
k_des = 1.0   →  Γ = 0.5552   →  Γ·k_des = 0.5552
k_des = 10.0  →  Γ = 0.0555   →  Γ·k_des = 0.5552
```

Constant to 4 decimal places. **The Picard outer loop is correctly
landing on the analytic fixed point.**

### 2. r_H_El sensitivity (Singh Eq. 4 directionality)

```
r_H_El = 180   →  Γ = 0.570   (ΔΓ = +2.7%)
r_H_El = 195   →  Γ = 0.559   (ΔΓ = +0.7%)
r_H_El = 200.98 →  Γ = 0.555  (baseline)
r_H_El = 215   →  Γ = 0.546   (ΔΓ = -1.6%)
r_H_El = 250   →  Γ = 0.526   (ΔΓ = -5.2%)
```

Direction is correct: increasing r_H_El moves the geometric factor
`(1 − r_M-O² / r_H_El²)` toward zero, reducing the magnitude of the
Singh ΔpKa shift, reducing the hydrolysis enhancement, reducing Γ.

**Magnitude is small** — ~8% Γ swing across a 70 pm bracket, where
the Linsey 2025 deck slide 27 implies ~6 pKa unit drop = a
10⁶× Ka_M shift. With current `k_hyd = 1e-3 nondim` the system is
so far below saturation that even a 10⁶× shift in `pKa_factor`
maps to only a small change in Γ. This is the **calibration gap**
flagged in the v9 plan §"Risk callouts" as the load-bearing
Cu→carbon transferability question.

### 3. Stern sensitivity (the loudest knob)

```
C_S = 0.05  →  Γ = 0.313   (factor 0.56× baseline)
C_S = 0.10  →  Γ = 0.555   (baseline)
C_S = 0.20  →  Γ = 1.322   (factor 2.38× baseline)
```

Doubling C_S more than doubles Γ. Mechanism: σ_S ≈ C_S · ψ_S, so
doubling C_S doubles σ_S, doubles ΔpKa magnitude, exponentiates
into `10^(−ΔpKa)`, multiplies the forward branch of R_net. Stern
capacitance is the strongest lever in the sweep.

### 4. cd plateau is invariant across the sweep

`cd = −5.532 mA/cm²` to 4 decimals across **all 9 combinations.**
At V=−0.40 V on this stack, BV is mass-transport saturated
(Gate 2's H+ Levich floor at L_eff = 16 µm). Adding a hydrolysis
source at the OHP produces extra local H+, but BV consumes it as
fast as it's produced; the rate-limiting step remains the bulk H+
flux into solution.

This is a **clean re-confirmation of Gate 2's saturated regime**.
It is also a **calibration warning** for cation-series 6β.2:
* If we want hydrolysis to **move cd**, we need either
  (a) a less saturated V (closer to onset), or
  (b) much larger k_hyd to overwhelm bulk supply, or
  (c) thicker boundary layer (smaller L_eff) so the OHP H+ doesn't
      hand back to transport instantly.
* If we want hydrolysis to **move surface pH**, the local
  ⟨c_H⟩ at OHP is what matters — that's already moving in the
  expected direction (Γ depends on it via the Picard formula).

---

## Status against the plan §4B verdict criteria

| Criterion | Status |
|---|---|
| Newton converges at all (λ, C_S, k_des, r_H_El) combinations | ✅ 9/9 converged |
| λ=0 reproduces Gate 2 baseline | ✅ cd=−5.532 matches Gate 2 SUCCESS |
| Bikerman packing < 1.0 at every rung | ✅ no packing-overflow errors raised |
| Predicted-vs-realized `Δ ln R_4e` agreement < 30% | ⏳ not computed (cd plateau is invariant; metric undefined) |
| C_S sensitivity bounded | ✅ Γ ranges 0.31–1.32 across C_S 0.05–0.20 |
| r_H_El calibration target ΔpKa(K) ≈ −6 within 30% | ❌ smoke `k_hyd = 1e-3` produces ~8% Γ swing |
| Branch diagnostic (R4#5) bare vs Γ-corrected σ_S | ⏳ not run (production path uses bare σ_S as designed) |

**Architectural verdict: PASS** — the v9 machinery converges
robustly across the parameter sweep, the Picard outer loop hits the
analytic fixed point, and the Singh formula directionality is
correct.

**Calibration verdict: OPEN** — the smoke baselines were
intentionally tame to ensure convergence (the plan's stated
`k_hyd = 1e2 m/s` physical translates to ~5×10⁶ nondim, which
overwhelms BV transport and breaks Picard). To match the deck
slide 27 ΔpKa(K) ≈ −6 unit drop, calibration of (k_hyd, σ_S
unit-conversion, r_H_El for ORR-on-CMK-3-carbon) is required.
This is the load-bearing Cu→carbon transferability work that
**6β.2 should close** with the cation-series holdout (Cs⁺ /
Na⁺ / Li⁺) experimentally constraining the kinetic priors.

---

## Reproducing the run

```bash
cd /Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse
source ../venv-firedrake/bin/activate
export MPLCONFIGDIR=/tmp XDG_CACHE_HOME=/tmp PYOP2_CACHE_DIR=/tmp/pyop2 \
       FIREDRAKE_TSFC_KERNEL_CACHE_DIR=/tmp/firedrake-tsfc OMP_NUM_THREADS=1

# Quick smoke (single baseline combo, ~9 min cold or ~1 min if cache exists):
python -u scripts/studies/phase6b_v9_gate4_finite_hydrolysis_smoke.py --quick

# Full sweep (9 one-axis-at-a-time combos):
python -u scripts/studies/phase6b_v9_gate4_finite_hydrolysis_smoke.py

# Force fresh cache (delete the snapshot first):
rm StudyResults/phase6b_v9_gate4_smoke/u_warmstart_at_v_target.npz
python -u scripts/studies/phase6b_v9_gate4_finite_hydrolysis_smoke.py
```

Outputs:

* `StudyResults/phase6b_v9_gate4_smoke/iv_curve.json` — full
  sweep payload (per-combination Γ, cd, pc, Picard rung
  diagnostics)
* `StudyResults/phase6b_v9_gate4_smoke/u_warmstart_at_v_target.npz`
  — cached U snapshot at V=−0.40 V (5 numpy arrays, one per mixed-
  space subfunction)

---

## Pointers for next steps

1. **6β.1.b — full V_RHE grid at λ=1.** Re-extend the warm-walk to
   walk down through the V_RHE grid at λ=1 (cation hydrolysis
   active throughout) and record cd, pc, Γ per voltage. Needs
   either (a) caching at multiple voltages, or (b) the existing
   warm-walk pattern with λ=1 set in sp construction. Probably
   ~30 min wall.
2. **6β.2 — cation-series holdout.** Re-run the full sweep with
   `cation = "Cs+"`, `"Na+"`, `"Li+"` (each picks a different
   Singh Table S1 row). Calibrate `k_hyd` per cation against the
   deck's qualitative trend (Cs⁺ produces most local H+, Li⁺
   least).
3. **Calibration of `r_H_El_pm` for ORR-on-carbon.** Iterate the
   r_H_El sweep at *much higher* `k_hyd` (e.g. `1e0` to `1e+1`
   nondim) until Γ shifts produce a ~6-unit pKa swing inferable
   from surface pH movement. The Cu prior `r_H_El = 200.98 pm`
   is unlikely to be exactly right for carbon; the deck slide 27
   data is the calibration target.
4. **Lift the cd-plateau saturation.** Run the same sweep at
   V=−0.20 V (closer to onset, where BV isn't saturated) to
   measure whether cation hydrolysis shifts cd in the expected
   direction. This is the cleanest "did the physics matter"
   test, separate from the surface-pH question.
