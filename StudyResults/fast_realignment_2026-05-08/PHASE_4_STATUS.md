# Phase 4 Status — Multi-Ion Convergence Wall

Date: 2026-05-08
Branch: `fast-realignment-2026-05-08`

## Where we are

Phases 0–3 landed cleanly:
- ✅ Phase 0: Git checkpoint + branch + tag.
- ✅ Phase 1.1: Disabled-reaction guard (10/10 tests pass on logc + logc_muh).
- ✅ Phase 1.2: Picard audit at +0.55 V — `picard_outer_loop_general`
  converges geometrically for parallel-2e/4e topology (no bug).
- ✅ Phase 1.3: `_is_reaction_disabled` + `_is_parallel_2e_4e` helpers,
  trivial-row treatment in `_assemble_n_reaction_system` (43/43 Picard
  tests pass).
- ✅ Phase 2.1: Multi-counterion shared-theta closed-form
  (19/19 closure-algebra + integration tests pass).
- ✅ Phase 2.2: Cs⁺/SO₄²⁻ hard-sphere constants in `_bv_common.py`
  (θ_b ≈ 0.991, electroneutrality + I=0.3 M verified).
- ✅ Phase 2.3: `Forward/bv_solver/multi_ion.py` (6/6 multi-ion smoke
  tests pass).
- ✅ Phase 2.4: Multi-ion spatial IC in `forms_logc{,_muh}.py`
  (single-counterion legacy path byte-equivalent: 12/12 tests pass on
  both backends).
- ✅ Phase 3: Driver `peroxide_window_3sp_parallel_2e_4e_csplus_so4.py`
  authored.

## Phase 4 result: anchor smoke at V_RHE=+0.55 V FAILED

The multi-ion stack (Cs⁺ + SO₄²⁻ + parallel-2e/4e + Stern + Bikerman)
does NOT converge from the multi-ion debye_boltzmann IC at the page-15
anchor. At V_RHE = 0 V (per plan §5c anchor relocation) it ALSO fails.

**Diagnostic 1**: legacy single-counterion ClO₄⁻ + Bikerman + Stern +
parallel-2e/4e DOES converge at V_RHE = +0.55 V (cold success in
33.7 s). So the parallel-2e/4e topology and Stern integration are
working; the regression is specifically in the multi-ion path.

**Diagnostic 2**: the multi-ion IC ITSELF converges
(`_try_debye_boltzmann_ic_muh` returns `ok=True, picard_iters=40`).
The IC produces sensible spatial fields:
  - `u_O2 ∈ [-0.073, 0]` (slight surface depletion)
  - `u_H2O2 ∈ [-9.28, -9.21]` (essentially at bulk seed)
  - `mu_H ∈ [-2.56, -2.49]` (bulk H⁺ everywhere)
  - `phi ∈ [0, 1.58]`           (Stern-compressed psi_D ≈ 1.58)

**Diagnostic 3**: Newton diverges from this IC at z=1. The orchestrator
falls back to linear-phi + z-ramp, but z=0 also diverges
(`PARTIAL z=0.000`, `snes_reason=-9` = `DIVERGED_LINE_SEARCH`).

## Root cause

The multi-ion physics CORRECTLY identifies that with I = 0.3 M and a
small Stern coefficient (0.10 F/m² → ψ_D ≈ 1.58 at +0.55 V), the
diffuse-layer drop is small. **This means H⁺ at the OHP is only mildly
depleted** (`c_H/c_H_b ≈ exp(-1.58) ≈ 0.21`).

Consequently `(c_H / c_H_ref)² ≈ 0.041` at the OHP, and the cathodic
2e rate at η_R2e = -7.23 (after Stern compression) is

    R_2e ≈ k₀ · c_O · 0.041 · exp(0.627 · 2 · 7.23)
         = 1.263e-3 · 1 · 0.041 · 8700 ≈ 0.45  (nondim)

A nondim rate of 0.45 implies `c_O_surface = c_O_b - R/D ≈ 1 - 0.45 =
0.55`. That's plausible but Newton struggles to bridge from the IC
(c_O ≈ 1 everywhere) to this strongly mass-transport-limited state.

**The legacy single-ClO₄⁻ stack works at +0.55 V because** the Picard's
1:1 closed form `phi_o = log(H_o / c_clo4_b)` gives `phi_o ≈ 0` (when
`c_clo4_b = c_H_b = 0.0833`, i.e. C_O2-rescaled) and the diffuse drop
collects the full 21.4 → 10.87 at this Stern. **That very high ψ_D
strongly depletes H⁺ at the OHP** (`c_H/c_H_b ≈ exp(-10.87) ≈ 1.9e-5`),
which collapses `(c_H/c_H_ref)²` by 11 orders of magnitude and turns
the cathodic rate into a trivially mass-transport-friendly value.

**That is an artefact of the single-counterion 1:1 closure**, not
physics: a real Cs₂SO₄ electrolyte at I=0.3 M does NOT have ψ_D = 11.
The multi-ion physics correctly gives ψ_D ≈ 1.58.

## Recommendation

Three escalation paths from the plan §5, in increasing scope:

1. **Phase 5b — Continuation in I.** Ramp from low ionic strength
   (`C_CSPLUS_HAT * α, C_SO4_HAT * α` for `α ∈ {0.01, 0.1, 0.5, 1.0}`)
   so the EDL transitions smoothly from "low-I large-ψ_D" to "I=0.3 M
   small-ψ_D" with Newton repair at each step. Within scope of fast
   realignment.

2. **Phase 5b — Continuation in `a_nondim`.** Ramp from `a_k = 0`
   (ideal Boltzmann, no Bikerman saturation) to literature values.
   Less likely to help here since the rate problem is the BV exponent
   not steric saturation, but cheap to try.

3. **Phase 5g — Smarter spatial IC.** Replace linear `phi_outer(y)`
   with `solve_outer_phi_multiion()` evaluated at several y nodes
   (the plan's note 2). Higher-fidelity IC closer to the SS solution
   would give Newton a better start. Out of fast-realignment scope.

4. **Outside the plan**: revisit the BV log-rate formulation. The
   current `eta_clipped` clip at 100 is much larger than the regime
   where `R_2e` becomes mass-transport-limited (`α·n_e·η ≈ 11` at
   +0.55 V). Lowering the clip (or adding a softplus-bounded BV form)
   would prevent unbounded rates during Newton iteration without
   distorting the physical SS.

## Time invested

- Phase 0: 5 min
- Phase 1.1-1.3: ~30 min (helpers + audit + tests)
- Phase 2.1-2.4: ~90 min (multi-ion machinery + IC refactor + tests)
- Phase 3: 30 min (driver)
- Phase 4 diagnostic: ~30 min
- **Total**: ~3 hours of the 7-12 day plan estimate.

The "fast" in the plan title was about not running M3a/M3b cascade.
The Newton convergence wall hit at the FIRST production sweep is
exactly the Phase 5 risk the plan acknowledged ("expected longest").

## Decision required

Per plan §4: "If any of {A, B, D} ≥ 15/25 is unreachable in the time
budget, ASK THE USER before lowering. Do not silently downgrade."

**Asking.**
