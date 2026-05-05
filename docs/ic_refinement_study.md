# IC refinement study — how close is `debye_boltzmann` to SS?

**Date:** 2026-05-03
**Question:** Across the existing converging window
`V_RHE ∈ [-0.5, +0.6] V`, how much Newton work does each
initializer need to refine its initial guess to steady state, and
how does the new `debye_boltzmann` analytical IC compare to the
baseline `linear_phi`?

**Answer:** The `debye_boltzmann` IC is a **kinetic-regime tool, not
a global IC**. It pays off at `V_RHE ≥ +0.3 V` (1.5–6× fewer Newton
iterations on direct z=1 SS, and unblocks `V=+0.6 V` where
`linear_phi` direct-z=1 fails outright). Below that voltage, the
Picard outer loop oscillates against the H⁺ mass-transport limit
and the IC silently falls back to `linear_phi` — correct behavior,
but no benefit.

---

## 1. Methodology

`scripts/studies/ic_refinement_study.py` builds a fresh context with
each initializer, forces full charge coupling (`z=1`,
`boltzmann_z_scale=1`), and runs the production SS time-stepping loop
with no z-ramp. Per (V, initializer) it records:

- `picard_iters` — IC construction cost (`debye_boltzmann` only;
  `None` for `linear_phi`).
- `cd_IC`, `pc_IC` — observables assembled from the IC field
  *before* any Newton step. The static distance of the IC from a
  PDE solution.
- `first_step_snes_iters` — Newton iterations on the *first* SS
  time-step starting from the IC. The canonical "how close is the
  IC to SS at this dt" metric.
- `total_ss_steps` — transient time-steps until the CD plateau
  detector fires (or `MAX_SS_STEPS=200` cap).
- `total_snes_iters` — sum of Newton iterations across all SS
  time-steps.
- `converged_z1` — bool: did direct z=1 SS converge from this IC?

Truth reference: a single pass through the production C+D
orchestrator (`linear_phi` + standard z=0 SS + z-ramp + z=1 SS) at
every voltage. All 12/12 voltages converged in the truth pass
(3 cold + 9 warm-walk) at clip=100, Ny=200.

Configuration: `Ny=200`, `exponent_clip=100`, `dt_init=0.25`,
`ss_rel_tol=1e-4`, `ss_consec=4`, `MAX_SS_STEPS=200`. SNES options
match the production driver.

Artifacts: `StudyResults/ic_refinement_study/data.json` (full
records including per-step history) and
`StudyResults/ic_refinement_study/refinement.png` (4-panel plot).

## 2. Results — Newton work to reach SS from each IC

| V_RHE | linear_phi: first / total / steps | debye_boltzmann: picard / first / total / steps | speedup |
|---|---|---|---|
| -0.500 | FAIL | picard=50 fallback → FAIL | — |
| -0.400 | FAIL | picard=50 fallback → FAIL | — |
| -0.300 | FAIL | picard=50 fallback → FAIL | — |
| -0.200 | FAIL | picard=50 fallback → FAIL | — |
| -0.100 | FAIL | picard=50 fallback → FAIL | — |
| +0.000 | FAIL | picard=50 fallback → FAIL | — |
| +0.100 | FAIL | picard=50 fallback → FAIL | — |
| +0.200 | FAIL | picard=50 fallback → FAIL | — |
| **+0.300** | 24 / 53 / 8 | 21 / 23 / **31** / 9 | **1.7× total** |
| **+0.400** | 24 / 72 / 10 | 21 / 15 / **22** / 8 | **3.3× total** |
| **+0.500** | 24 / 76 / 13 | 21 / **5** / **12** / 9 | **6.3× total** |
| **+0.600** | **FAIL** | 21 / 8 / **11** / 5 | **unblocks** |

`first` = Newton iters on the first SS time-step.
`total` = sum of Newton iters across all SS time-steps until plateau.
`steps` = number of SS time-steps until plateau.
`FAIL` = direct z=1 SS diverged before plateau (Newton raised on a
time-step solve).

Key observations:

1. **Direct z=1 SS doesn't work for either IC at `V_RHE ≤ +0.2 V`**
   under clip=100 settings. Both fail with `cd_IC` already at
   non-physical magnitudes (see §3). This is consistent with
   `clip_observable_investigation.md`'s finding that clip=100 gives
   only 3/13 cold-converged voltages on the production grid; the
   z-ramp is doing real basin work that direct z=1 cannot replace.

2. **Where direct z=1 works at all (`V ≥ +0.3 V`),
   `debye_boltzmann` reduces total Newton iterations by 1.7×–6.3×.**
   The first-step Newton iterations drop from 24 (linear_phi) to as
   low as 5 (debye_boltzmann at V=+0.5 V) — Newton needs
   substantially less correction from a Boltzmann-aware IC.

3. **At `V=+0.6 V`, `linear_phi` direct-z=1 SS fails outright;
   `debye_boltzmann` succeeds in 11 total Newton iters across 5 SS
   steps.** This is the regime where the IC unblocks new voltages
   that the production orchestrator only reaches via warm-walk from
   the +0.5/+0.6 V cold anchors.

## 3. IC observable error — how far is the static IC from physical SS?

`cd_IC` is the BV current assembled from the IC field with no Newton
step. It is the answer Newton starts with at z=1 SS.

| V_RHE | linear_phi cd_IC | debye_boltzmann cd_IC | truth (orchestrator) |
|---|---|---|---|
| -0.500 | -3.34e+29 | -3.34e+29 (fallback) | ~-1.84e-1 |
| -0.300 | -1.39e+26 | -1.39e+26 (fallback) | ~-1.82e-1 |
| -0.100 | -5.80e+22 | -5.80e+22 (fallback) | ~-1.78e-1 |
| +0.000 | -1.18e+21 | -1.18e+21 (fallback) | ~-1.78e-1 |
| +0.200 | -4.92e+17 | -4.92e+17 (fallback) | ~-3e-2 |
| **+0.300** | -1.00e+16 | **-7.23e+05** | ~-3e-3 |
| **+0.400** | -2.05e+14 | **-6.14e+00** | ~+1e-3 |
| **+0.500** | -4.18e+12 | **-5.22e-05** | ~-1e-5 |
| **+0.600** | -8.52e+10 | **+1.83e-09** | ~+1e-8 |

**linear_phi everywhere is 10²⁰–10³⁰× off.** The IC has `c_H = bulk`
at the electrode (no Boltzmann depletion or enrichment) combined
with unclipped large cathodic eta. The cathodic exponent
`exp(-α·n·η) ~ exp(57)` saturates the BV rate, and with
`(c_H/c_ref)² = 1` the IC reports a rate ~10²⁵ times its physical
SS value. The IC observable is numerically meaningless; only the
z-ramp transient brings Newton out of this regime.

**debye_boltzmann at `V ≥ +0.3 V` closes the gap by 8–22 OOM.**
At V=+0.5 V the analytical IC's static `cd_IC = -5.22e-5` is
within ~5× of the converged truth `~-1e-5`. At V=+0.6 V the IC even
flips to the correct positive sign that matches the converged
anodic-edge value before Newton runs.

## 4. Why cathodic V (V_RHE ≤ +0.2 V) fails the Picard loop

At cathodic voltages, ORR is mass-transport limited:
`R1 + R2 ≈ D_H · H_b` (saturated). The Picard mass balance
`H_o = max(H_b - (R1+R2)/D_H, floor)` drives `H_o` to the FP floor
(`1e-300`) on the first iterate. Then:

```
phi_o = ln(H_o / c_ClO4_bulk) ≈ -690
psi_D = phi_applied - phi_o ≈ +670  (sign-flipped from physical)
H_s = H_o · exp(-psi_D) → underflows to floor
```

Next iterate: with `H_s` at the floor, `(H_s/c_ref)²` is `1e-1377`,
so `A1 ≈ 0`, the 2×2 RHS is dominated by `-B1·P_b`, R1 flips sign.
Then the mass balance recovers `H_o`, but the next iterate over-
corrects in the other direction. Under-relaxation ω=0.5 doesn't
damp this bistability. Picard hits the 50-iter cap and the IC falls
back to `linear_phi`.

The matched-asymptotic reduction in
`docs/PNP_BV_Analytical_Simplifications.md` §"Two-rate algebraic
outer model" assumes the outer mass balance has a stable interior
solution. This assumption holds when reactions are
**kinetically** (not mass-transport) limited — i.e., far from the
diffusion-limit asymptote. On the production grid, that's the
cathodic-onset / kinetic-control transition at `V ≈ +0.3 V` and up.
At V_RHE ≤ +0.2 V the system is mass-transport-limited (PC
saturates per `CLAUDE.md` rule #7), and the analytical reduction's
assumptions break.

## 5. Practical takeaway

- **Use `debye_boltzmann` for V_RHE ≥ +0.3 V** when cold-starting at
  full coupling. It cuts Newton iterations by 1.7×–6×, and unblocks
  V=+0.6 V at clip=100 where direct z=1 with `linear_phi` fails.

- **At V_RHE ≤ +0.2 V the IC is no-op** (Picard fails, fallback to
  `linear_phi`). That's correct behavior — the analytical reduction
  doesn't apply in the mass-transport-limited regime — but means the
  IC adds zero value here.

- **Direct z=1 SS is not a substitute for the z-ramp at low V** —
  it diverges at `V_RHE ≤ +0.2 V` for both ICs at clip=100. The
  z-ramp path remains the production fallback for those voltages
  (and the orchestrator uses it whenever the analytical IC isn't
  sufficient).

- **For the inverse pipeline**, the IC's value is mostly at the
  anodic edge of the production grid (V ∈ [+0.30, +0.66]) where
  perturbed-parameter cold solves see the most Newton stress. At
  cathodic V the inverse pipeline already converges quickly via
  warm-walk; the IC doesn't change that.

- **Improving cathodic-V coverage** would require either (a) a
  different Picard fixed-point that doesn't depend on H⁺
  mass-balance closure (e.g., µ_H = u_H + φ as the variable, since
  `μ_H` is smooth across the Debye layer at any voltage), or
  (b) a Stern-augmented model that limits the diffuse-layer drop and
  decouples the H⁺ depletion from the bulk supply. Both are
  documented in `docs/Peroxide Solver Convergence.md` and
  `docs/PNP_BV_Analytical_Simplifications.md` as longer-term work.

## 6. Pointers

- Study script: `scripts/studies/ic_refinement_study.py`
- Data: `StudyResults/ic_refinement_study/data.json`
  (per-V/per-IC records including per-step `snes_iters`, `dt`,
  `cd`, `pc` history)
- Plot: `StudyResults/ic_refinement_study/refinement.png`
  (4-panel: first-step SNES iters, total SNES iters, SS time-steps,
  IC observable error vs V; X markers flag non-converged direct-z=1
  attempts)
- IC implementation:
  `Forward/bv_solver/forms_logc.py:set_initial_conditions_debye_boltzmann_logc`
  and `_try_debye_boltzmann_ic`
- Orchestrator z-ramp bypass:
  `Forward/bv_solver/grid_per_voltage.py:_solve_cold` (lines ~309-360)
- Math: `docs/PNP_BV_Analytical_Simplifications.md` §"Two-rate
  algebraic outer model" and §"Voltage-specific analytical initial
  condition"
- Companion peroxide-window outcome:
  `docs/peroxide_window_investigation.md` and the new study
  `StudyResults/peroxide_window_pb_init_test/` (which confirmed
  the IC alone does not unblock V_RHE ≥ +0.68 V without Stern)
- Clip-threshold context: `docs/clip_observable_investigation.md`
  (clip=100 is the unclipped-everywhere reference; cold-start basin
  shrinks 10/13 → 3/13 vs clip=50, which is why direct z=1 SS fails
  at low V even with a perfect IC)
