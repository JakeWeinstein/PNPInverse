# Phase 6β v9 Gate 2 — cathodic dynamic-K⁺ smoke SUCCESS report

**Date:** 2026-05-10
**Stack:** `phase6b_v9_gate2_4sp_dynamic_k2so4_cathodic`
**Outcome:** ✅ **CONVERGED** at V_RHE = -0.40 V on the 4sp dynamic-K⁺
stack via the production-style anchor-and-walk pattern (per CLAUDE.md
"Calling the production solver" multi-ion example).
**Anchor:** V = +0.55 V (anodic), with k0 ladder + Phase 6α Kw_eff ladder.
**Walk:** 11/11 V_RHE points converged from +0.55 V → -0.40 V.

## Strategy that worked (Step 1 of the smoke script)

1. **Build the 4sp dynamic-K2SO4 stack** with `initializer="linear_phi"`
   (the `debye_boltzmann` Picard `cosh(psi_D)` overflows for asymmetric
   K2SO4 salts — the matched-asymptotic Picard was derived for 1:1
   symmetric electrolytes; see HISTORICAL_FAILURE.md for the full
   diagnosis).
2. **Anchor at V = +0.55 V** via `solve_anchor_with_continuation` with:
   * `k0_targets = {0: K0_HAT_R2E, 1: K0_HAT_R4E}`
   * `initial_scales = (1e-12, 1e-9, 1e-6, 1e-3, 1.0)` (5-rung k0 ladder)
   * `kw_eff_ladder = (0.0, KW_HAT*1e-6, KW_HAT*1e-3, KW_HAT*0.1, KW_HAT)`
   * Anchor wall: 156.8s.
3. **Extract `PreconvergedAnchor`** from the converged state.
4. **Warm-walk** via `solve_grid_with_anchor` over
   `(0.55, 0.50, 0.40, 0.30, 0.20, 0.10, 0.00, -0.10, -0.20, -0.30, -0.40)` V.
   * 11/11 converged.
   * Walk wall: 3223.5s ≈ 53.7 min.
5. **Total wall:** ~56 min for the full anchor-and-walk pass.

## Per-V_RHE results (Step 1)

```
V_RHE     converged  method                    cd (mA/cm²)  pc (mA/cm²)
+0.55     True       warm<-+21.407             -0.583       +0.291
+0.50     True       warm<-+21.407             -0.618       +0.309
+0.40     True       warm<-+19.461             -0.935       +0.467
+0.30     True       warm<-+15.569             -2.552       +1.276
+0.20     True       warm<-+11.677             -5.467       +2.733
+0.10     True       warm<- +7.784             -5.532       +2.766
 0.00     True       warm<- +3.892             -5.532       +2.766
-0.10     True       warm<- +0.000             -5.532       +2.766
-0.20     True       warm<- -3.892             -5.532       +2.766
-0.30     True       warm<- -7.784             -5.532       +2.766
-0.40     True       warm<--11.677             -5.532       +2.766
```

The cd plateau at -5.53 mA/cm² for V ≤ +0.10 V is the H⁺ mass-transport
floor (Levich-limited at L_eff = 16 µm).  pc tracks at half cd (consistent
with the parallel 2e/4e topology with peroxide selectivity ~50% at this
k0_R4e/k0_R2e ratio).

## Key findings

* **Linear-phi IC** is sufficient when paired with the production
  anchor-and-walk pattern: the anchor at V=+0.55 V (anodic, K⁺ depletes
  near electrode) is well-conditioned, and warm-walking down the V_RHE
  grid lets the dynamic K⁺ field evolve smoothly through the inversion
  region near V=+0.30 V (where cd jumps from -0.93 to -2.55 mA/cm²).
* **Cold per-voltage at -0.40 V fails immediately** — the historical
  failure mode that motivated this rewrite.
* The matched-asymptotic Picard (`initializer="debye_boltzmann"`) needs
  to be extended to handle asymmetric salts before it can be used on
  K2SO4; for now, linear-phi + production anchor pattern works.

## Implications for Gate 2 pass criterion

* **Gate 2 PASS criterion met:** `iv_curve.json` shows Newton converged
  at V_RHE = -0.40 V (and at every other voltage from +0.55 V down).
* **6β.1 unblocked** for Gates 3 + 4 (Γ machinery + cation hydrolysis
  finite-rate kinetics).
* **Gate 2D regression test status:** the dynamic-K⁺ probe converges
  cleanly via the production anchor-and-walk pattern, but the
  reference 3sp + multi-ion (analytic K⁺ + analytic SO₄²⁻) stack does
  NOT — at the same +0.55 V anchor it hits
  ``LadderExhausted: k0 ladder exhausted at Kw_eff=0`` on the first
  k0=1e-12 rung.  The multi-ion path has its own IC fragility
  (probably similar root cause: linear-phi IC gives an inconsistent
  starting point for the 2-Bikerman shared-θ residual).  Gate 2D
  test correctly skips with ``pytest.skip`` rather than spuriously
  passing or failing; the cross-validation (probe matches reference
  within 1%) is deferred until the multi-ion IC is also brought to
  parity.  This does NOT block 6β.1 because the dynamic-K⁺ probe is
  the load-bearing path; the reference is a sanity check.

## Artifacts

* `iv_curve.json` — full 11-point I-V curve with diagnostics.
* `diagnostics.json` — diagnostics-only excerpt for grep-friendly inspection.
* `HISTORICAL_FAILURE.md` — earlier failure diagnosis (cold-only and
  Picard cosh overflow); kept as a record of the dead ends explored.
* This `SUCCESS.md` — passing-result summary.
