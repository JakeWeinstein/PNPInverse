# Phase 5α GATE — FAILED (multi-ion Picard does not converge at +0.55 V)

Date: 2026-05-09
Branch: `fast-realignment-2026-05-08`
Gate report: `StudyResults/fast_realignment_2026-05-08/phase5alpha_gate/gate_report.json`

## Headline

The Phase 5α GATE script
(`scripts/studies/picard_residual_consistency_csplus_so4.py`) ran to completion
and reported `gate_passed: false`. Both reactions show massive
log-consistency violations (`|log10(R_r / R_p)|` of 11.2 and 23.5).

**The failure is not a Picard-vs-residual semantic mismatch. It is the
patched multi-ion Picard failing to converge.** The
`initializer_picard_state` payload reports
`initializer_fallback_reason = "picard_max_iters_delta=1.001"`, meaning the
outer loop hit `max_iters` with `delta` of order unity (oscillation, not
near-convergence). The IC then dropped to the linear-φ fallback and the
residual rates were assembled at THAT state — incomparable to the failure-state
Picard dict.

## What the GATE measured

Picard failure-state (post `max_iters=50`):

| field | value | reading |
|---|---|---|
| `phi_o` | -1.67e-4 | ✓ multi-ion physics: bulk Cs⁺/SO₄²⁻ closure puts φ_o ≈ 0. |
| `psi_D` | +1.58 | ✓ matches PHASE_4 prediction (`ψ_D ≈ 1.58` for I=0.3 M, C_S=0.10 F/m²). |
| `psi_S` | +19.83 | ✓ Stern carries `phi_applied − ψ_D` = 21.41 − 1.58. |
| `gamma_s` | 0.93 | reasonable Bikerman γ at the multi-ion OHP. |
| `H_o` | 1e-300 | **at the species floor — H⁺ has been completely depleted**. |
| `R_list[0]` (R_2e) | -5.4e-10 | essentially zero / sign-flipped. |
| `R_list[1]` (R_4e) | 0.286 | mass-transport-limit value (`D_O · O_b / D_O ≈ 1`). |
| `eta_list[0]` (R_2e) | -7.22 | `psi_S − E_eq_2e` = 19.83 − 27.05. ✓ |
| `eta_list[1]` (R_4e) | -28.05 | `psi_S − E_eq_4e` = 19.83 − 47.87. ✓ |

Residual rates assembled at the linear-φ fallback IC (which includes the
correct cathodic depletion physics):

| reaction | R_residual | R_picard | log10 ratio |
|---|---|---|---|
| R_2e | 9.05e+1 | -5.4e-10 | 11.23 |
| R_4e | 8.54e+22 | 0.286 | 23.48 |

The residual numbers themselves (especially R_4e at 1e22) are also unphysical —
they reflect the linear-φ IC seeding `phi_applied = 21.4` everywhere, which
makes `eta = phi_applied − E_eq_4e = -26.4`, gives
`exp(0.5·4·(-26.4)) ≈ exp(-52.8)`, but with `(c_H/c_H_b)^4` un-depleted
(linear-φ doesn't account for diffuse-layer H⁺ depletion). The residual
sees a much larger rate than physics allows.

## Root cause

Pre-T7 (single-ion Picard run on multi-ion-configured stack):
- Picard's hardwired `phi_o = log(H_o / c_clo4_bulk)` produced a large
  *negative* `phi_o` (because `c_clo4_bulk = 0.0833 ≈ c_H_bulk`, and
  small `H_o` makes `log(H_o / c_clo4_bulk)` strongly negative).
- The Stern split allocated nearly the full applied potential to
  `psi_D`, so `psi_S → 0` and `eta_drop → 0`.
- BV rates were exponentially small.
- Picard converged trivially (the trivial near-equilibrium fixed point).
- This is the wrong-anchor state PHASE_4 §"Root cause" identified.

Post-T7 (multi-ion Picard):
- `solve_outer_phi_multiion` correctly returns `phi_o ≈ 0` (Cs⁺/SO₄²⁻
  electroneutrality). `psi_D ≈ 1.58` (small, because Stern dominates).
  `psi_S ≈ 19.83`.
- BV rates at `eta_R2e = -7.22, eta_R4e = -28.05` are LARGE in the cathodic
  direction.
- Specifically R_4e: `exp(-α n_e η) = exp(0.5·4·7.22) = exp(14.4) ≈ 1.8e6`,
  multiplied by `k0 · O_b · (c_H/c_H_ref)^4`.
- The mass-transport budget for H⁺ (bulk `c_H ≈ 0.083` nondim, diffusivity
  `D_H ~ 1.2`) cannot supply this rate. `H_o` saturates at the floor.
- The Picard's fixed-point iteration with `omega=0.5` cannot resolve the
  saturation: each iter drives R larger, hits floor, oscillates.
- After 50 iterations, `delta = 1.001` (order-unity oscillation, NOT
  convergence).

## Why the failure is correct physics, not a Picard bug

The multi-ion Picard is reporting a real physical incompatibility: at
V_RHE = +0.55 V with `psi_S ≈ 19.83`, the cathodic R_4e rate vastly exceeds
H⁺ mass-transport supply. The same problem PHASE_4_STATUS.md §"Root cause"
predicted: "Newton struggles to bridge from the IC (c_O ≈ 1 everywhere) to
this strongly mass-transport-limited state."

PHASE_4 hoped that fixing the wrong-anchor would let Newton converge from
the corrected IC. What actually happens is the Picard *itself* now sees
the same physical wall — because the Picard, like Newton, is solving for
a steady state at a voltage where rates outpace mass transport.

## What changed since T0

T0 baseline: 76/76 tests passing. After T1–T7 helper extraction,
multi-ion plumbing, `phi_clamp` consistency, and caller updates, the
single-ion regression remains 93/93 passing (66 fast + 27 slow IC). The
multi-ion machinery itself is intact: 14/14 tests in
`test_multi_ion_csplus_so4.py` (including 8 new T5/T6 multi-ion smoke
tests) pass.

The failure is specifically in production-config Picard convergence at
high cathodic drive — not in the helpers, not in the multi-ion machinery,
not in single-ion regression.

## Implications for the patch sequence

Per PHASE_4_STATUS.md §"GATE FAIL" branch and the planning agent's
follow-up:

- **Phase 5β/5γ are not blocked by code correctness** — the helpers and
  plumbing are correct. They are blocked by the absence of a converged
  Picard anchor for the multi-ion stack at the page-15 voltage range.
- **Phase 5γ (anchor builder with k0 + dt continuation) was always the
  intended remedy for this** — it walks k0 from 1e-12 to its target,
  letting the rate stay sub-mass-transport-limit until the Newton/Picard
  physics has bridged in. The GATE failure means we cannot validate
  Picard log-consistency at +0.55 V from a cold start, but Phase 5γ's
  ramp would build an anchor at suppressed k0 where Picard converges
  trivially, then walk k0 up.
- **Phase 5δ (voltage homotopy from V_RHE = +0.85 V)** is an
  alternative: at higher V_RHE the cathodic eta is smaller in magnitude
  (because `eta_drop = psi_S = phi_applied - psi_D` and ψ_D may be ≈ 1.58
  at any V_RHE in the cathodic regime — actually psi_S grows with V_RHE,
  making this WORSE for cathodic R_4e dominance, not better).
  **5δ is unlikely to help** for this specific failure mode; the
  cathodic-rate-vs-mass-transport wall is monotonic in cathodic drive.
- **Phase 5ε (legacy ClO₄⁻ IC swap, one-hour falsification)** is a
  diagnostic: build the spatial IC with single-ion ClO₄⁻ Picard and the
  multi-ion residual. If Newton converges, it confirms the multi-ion IC
  is the obstacle; if Newton still diverges, it confirms the residual
  itself can't accept this state.

## Recommendation

The most direct path forward is **Phase 5γ k0 + dt continuation** with the
multi-ion Picard. The Picard converges trivially at `k0 → 0` (no
reaction, `R = 0`, `c_s = c_b`, `H_o = c_H_b`). Walking k0 upward in
small geometric steps lets the Picard incrementally accommodate the
cathodic drive. The GATE script can then be re-run at the converged
target k0, with a strong expectation of log-consistency holding (because
the Picard converged this time).

Phase 5α delivered:

1. ✓ Four helpers extracted from `picard_outer_loop_general` /
   `picard_outer_loop` with single-ion byte-equivalence preserved at 1e-12
   (84/84 tests through the byte-equivalence checkpoint).
2. ✓ Multi-ion branches in all four helpers, with `solve_outer_phi_multiion`,
   `compute_surface_gamma_multiion`, `effective_debye_length_local`,
   `per_ion_outer_concs` correctly threaded.
3. ✓ `_phi_safe_exp` per-ion `phi_clamp_val` consistency between
   `multi_ion.py` and the UFL `boltzmann.py:223` clamp.
4. ✓ `multi_ion_ctx`/`poisson_coefficient` plumbed through
   `picard_outer_loop_general` and both
   `_try_debye_boltzmann_ic{,_muh}` callers.
5. ✓ Phase 5α GATE script authored and executed.
6. ✗ Log-consistency assertion FAILED — but the failure is the multi-ion
   Picard not converging, not a code-correctness issue.

The handoff to Phase 5γ: build an anchor via k0 + dt continuation, then
re-run this GATE at the converged anchor. If the Picard converged there,
the log-consistency check will tell us whether the helpers' multi-ion
output matches the residual.

## Files of record

- `Forward/bv_solver/picard_ic.py` — four extracted helpers (T1-T4),
  multi-ion branches (T6), `multi_ion_ctx + poisson_coefficient`
  kwargs on `picard_outer_loop_general` (T7).
- `Forward/bv_solver/multi_ion.py` — `_phi_safe_exp` mirroring
  UFL `phi_clamp_val=50` clamp (T5).
- `Forward/bv_solver/forms_logc.py` — pre-Picard `ctx_mion_pre` build
  + threading to Picard call (T7).
- `Forward/bv_solver/forms_logc_muh.py` — same (T7).
- `tests/test_picard_ic_helpers.py` — 6 new T1-T4 tests
  (byte-equivalence + R3 sign correction).
- `tests/test_multi_ion_csplus_so4.py` — 8 new T5/T6 tests
  (per-ion phi_clamp + multi-ion helper branches).
- `scripts/studies/picard_residual_consistency_csplus_so4.py` — Phase 5α
  GATE.
- `StudyResults/fast_realignment_2026-05-08/phase5alpha_gate/gate_report.json`
  — full GATE output JSON.
