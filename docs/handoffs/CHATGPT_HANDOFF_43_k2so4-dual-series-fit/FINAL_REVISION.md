# Final revision — Critique session 43 (k2so4-dual-series-fit)

- Revised artifact: `~/.claude/plans/phase7p2-k2so4-dual-series-fit.md`
  (revised incrementally each round; final state includes everything)
- Final verdict: **APPROVED** (round 5 of cap 5)
- Rounds: R1 18 issues → R2 11 → R3 7 → R4 4 → R5 APPROVED
  (+3 non-blocking notes, folded in)
- Codex session: 019eb3ce-9820-7c31-b722-a32b79246f0e

## Headline outcomes of the loop

1. **OCP axis corrected (R1#1):** the 0.47 V GC-OCP component
   composes with the file's MEASURED Ag/AgCl→RHE calibration
   (+0.549 V), not the theoretical 0.574 V ⇒ central shift 1.019 V
   (was 1.044), with a 4-point refit bracket {0.969, 0.994, 1.044,
   1.069} covering both the calibration convention and ±50 mV on
   the unconfirmed 0.47 component. θ is OCP-conditional until
   confirmation.
2. **Optimizer integrity (R1#2/3, R2#1, R3#1, R4#1):** frozen
   pre-fit mask; failed solves RAISE and restart L-BFGS-B from the
   last valid checkpoint (no penalty values or stale gradients ever
   enter the optimizer); removed bins are retried at the polished θ
   and re-inclusion re-runs the polish and all downstream gates.
3. **Ring series redesigned (R2#4, R2#6):** the objective fits RAW
   baseline-corrected j_ring with the collection model on the MODEL
   side (j_ring_model = −pc_model·N·A_d/A_r), making N variants
   pure model-side refits; per-cycle ring baseline subtraction with
   σ propagation.
4. **pc_model = outer-boundary escape flux (R1#4)** with an
   explicit escape≡production test; form_cd written
   reaction-by-reaction with ledger-equality and isolated-reaction
   tests (R1#5).
5. **Statistics hardened (R1#8, R2#10, R3#7):** raw χ² on a fixed
   vector for all comparisons; ΔAIC removed; σ declared a
   conservative single-observation predictive scale (no absolute
   reduced-χ² semantics; "standardized residual score" naming);
   Δχ²=4 profile widths are "pragmatic", not CIs.
6. **Identifiability made rigorous (R1#9, R2#8, R3#4):** true
   re-optimized profiles for the 4e parameters AND for the derived
   4e current fraction (constraint tolerance 0.01, ladder extended
   to cross Δχ²=4 both sides), under dual vs pc-only objectives —
   this comparison IS the paper claim.
7. **Structural unknowns become refits, not rescores (R1#10/13,
   R2#5/7):** explicit refit-vs-rescore table (window edge, OCP,
   L_eff, N, c_H, c_O₂, weights = REFIT; ablations = RESCORE);
   air-saturation ruled out by a documented data-based
   impossibility check; O₂/rpm/catalyst/ring-hold all in the asks.
8. **Lock tiers (R3#3, R4#2):** computational lock vs paper-grade
   claims with a claim→blocking-ask table; cross-condition
   transferability additionally gated on the raw Tafel xlsx
   ("digitized-slide-conditional" until then).

## Ledger summary

Accepted: all 40 numbered issues across R1 (18), R2 (11), R3 (7),
R4 (4), plus all 3 R5 non-blocking notes. Defended: none — every
issue was either a genuine gap or an improvement worth taking
(notably R2#4's model-side-N design, which is strictly better than
the original target-rescaling).

Unresolved: none.
