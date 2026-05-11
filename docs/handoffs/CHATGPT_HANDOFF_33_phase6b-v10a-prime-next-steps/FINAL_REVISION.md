# FINAL_REVISION — Critique session 33

- **Topic**: phase6b-v10a-prime-next-steps
- **Rounds**: 4 (cap raised to 5 mid-session; APPROVED on R4)
- **Final verdict**: **APPROVED** (R4); 2 non-blocking nits folded in
- **Revised artifact**: `/Users/jakeweinstein/.claude/plans/sparkly-gilded-pasteur.md`
- **Session dir**: `docs/handoffs/CHATGPT_HANDOFF_33_phase6b-v10a-prime-next-steps/`

## Summary

22 issues raised across 4 rounds; **all 22 accepted, 0 defended, 0
unresolved**.  GPT verdict APPROVED on R4 after sharpening fixes
landed.  The plan changed substantially:

* Decoupled K0 factor for plateau-shift vs branch-pass-at-target-V.
* Single-run target moved from K0_R4e_factor = 1e-18 (project memory's
  selectivity sweet spot) to 1e-14 (V=−0.10 branch-pass probe; the
  V with highest |sensS| in v10a).
* Dense bracket sweep redesigned from `{1, 1e-6, 1e-12, 1e-18, 1e-24}`
  (5 points, misses V=−0.10 window) to `{1e-10, 1e-12, 1e-14, 1e-16,
  1e-18, 1e-20, 1e-22, 1e-24}` (8 points, log-uniform, covers
  V=−0.10 through V=−0.50 windows).
* Decision tree expanded from 3 outcomes to **7 explicit branches
  (Cases A–G)** with deterministic next actions, plus a
  precedence-ordered Step 0 (artifact check → cap-dominance check →
  Phase A.2).
* "F₀ drops with weaker R_4e → θ retreats" deleted (was wrong — F₀
  is uncoupled from K0_R4e).
* "C_S doubles σ_S" softened to "Stern-diffuse coupling is
  sublinear; σ_S(V) is observed, not predicted".
* Added Jensen-safe F₀ decomposition with `amplification_from_c_K`
  vs `amplification_from_singh` ratios; expected values calibrated
  to v10a baseline (K+ enrichment is already load-bearing,
  amplification rises from 0.31 → 4.39 across V_RHE).
* Added solver-faithful (log-space, clip-aware) R_4e decomposition
  labeled as scalar approximation.
* Pinned v10b-vs-v10c routing threshold: `denom_cap/total > 0.8 AND
  θ > 0.9 AND |sensS| < 0.10` at every σ<0 V.
* Added attribution disclaimer: v10a' is operational, not
  attribution-clean; combined C_S + K0 change is intentional and
  routes work regardless.

## Addressed (20)

### Round 1 — 8 issues, all accepted

| # | Issue | Fix landing in plan |
|---|---|---|
| R1.1 | `1e-18` claimed for 0.3 V plateau shift but algebra gives `1e-10`; project memory's `1e-18` is selectivity, not plateau onset | Implementation notes: decoupled "Plateau-shift K0 factor" (`e^(−78·ΔV)`) from "Branch-pass K0 factor at target V" (per-V from v10a extrapolation). Single-run target = `1e-14` (V=−0.10 probe). |
| R1.2 | `cd_ok ≠ non-transport-limited` for mixed/2e branches at parallel 2e/4e cathode | Decision tree Case B: `o2lev > 0.9 AND cd_ok=True` is a documented escalation trigger (transport-limited artifact). |
| R1.3 | α=0.378 only moves onset to V=0, not V=−0.30 | Implementation notes: recomputed α targets (V=−0.10 ⇒ 0.35, V=−0.30 ⇒ 0.30, V=−0.50 ⇒ 0.27); α demoted to alternative knob with narrower headroom. |
| R1.4 | c_H⁴ multiplier omitted from plateau-shift reasoning | R_4e log-space decomposition includes `n_e · <ln(c_H/c_H_ref)>` term. |
| R1.5 | "F₀ drops with weaker R_4e → θ retreats" is wrong (F₀ uncoupled from K0_R4e) | Risk #4 deleted. Replacement: v10b-routing threshold for cap-dominance regime. |
| R1.6 | F₀ growth is K+ enrichment, NOT Singh ΔpKa (`pka_shift_avg ~ 1e-5`) | F₀ decomposition emits `amplification_from_c_K` + `amplification_from_singh` ratios. |
| R1.7 | "C_S doubles σ_S magnitude" oversimplified (Stern-diffuse series) | Context paragraph softened; σ_S(V) at C_S=0.20 is observed, no pre-commit. |
| R1.8 | Bracket `{1, 1e-6, 1e-12, 1e-18, 1e-24}` misses V=−0.10 mixed window | Redesigned: `{1e-10, 1e-12, 1e-14, 1e-16, 1e-18, 1e-20, 1e-22, 1e-24}` (8 points, log-uniform). |

### Round 2 — 7 issues, all accepted

| # | Issue | Fix landing in plan |
|---|---|---|
| R2.1 | `1e-14` doesn't sit in V=−0.30 window (30× above upper edge) | Reframed: `1e-14` is V=−0.10 probe only; bracket covers V=−0.30 + V=−0.50. |
| R2.2 | `o2lev > 0.9` artifact branch had no deterministic action | Decision tree expanded to 7 cases (A–G), each with concrete next step. |
| R2.3 | "Genuine BV-controlled" too strong; `o2lev ≤ 0.9` is sanity only | Reworded to "passes O₂-transport sanity check (necessary, not sufficient)". |
| R2.4 | R_4e decomposition using raw `exp` would clip-mismatch the solver's bv_log_rate path | Switched to log-space; emits `eta_scaled_clipped_avg`, `exponent_clip_active`, mirrors solver's clipping. |
| R2.5 | F₀ decomposition uses Jensen-invalid `10^<−ΔpKa>` | Switched to boundary averages of factor products: `<c_K>`, `<10^−ΔpKa>`, `<c_K · 10^−ΔpKa>`. |
| R2.6 | "denominator_cap dominates" had no pinned threshold | Pinned: `>0.8 AND θ>0.9 AND |sensS|<0.10` at every σ<0 V. |
| R2.7 | Combined C_S + K0 change loses attribution | Added explicit attribution disclaimer; v10a' is operational, not attribution-clean; routing works regardless. |

### Round 3 — 5 issues, all accepted (cap-hit, auto-revise)

| # | Issue | Fix landing in plan |
|---|---|---|
| R3.1 | Expected `amplification_from_c_K ≈ 1` is wrong; v10a F₀/k_hyd already shows 0.31 → 4.39 (K+ enrichment is load-bearing) | Sanity checks updated: "K+ enrichment is already load-bearing in v10a — not a new effect from C_S=0.20." |
| R3.2 | R_4e log decomposition still not solver-faithful (η_raw with Stern includes phi_boundary; clip is on η/V_T before α·n_e) | Labeled as scalar approximation; emits boundary η_scaled avg/min/max; `log_R4e_measured` is authoritative. |
| R3.3 | Case F's "switch perturbation knob to φ" invalidates the same V_kin score | Case F: only increase ε from 0.05 to 0.10/0.15; φ-perturbation deferred as auxiliary diagnostic. |
| R3.4 | Case G's inference "absence of artifact flag → σ<0 + o2lev<0.9 holds" is invalid | Case G: inspect per-filter failure matrix (counts per filter); route based on actual blocking filter. |
| R3.5 | Routing precedence ambiguous when clean V_kin AND cap-dominated both apply | Step 0 precedence guards pinned: artifact check → cap-dominance check → Case A. |

### Round 4 — 2 non-blocking nits, both folded in. **VERDICT: APPROVED.**

| # | Issue | Fix landing in plan |
|---|---|---|
| R4.1 | R_4e log decomposition mismatch can come from boundary-averaging/covariance, not only missing Stern term | Sanity checks point 6: "decomposition is a scalar approximation over boundary-averaged nonlinear terms... Interpret only **order-of-magnitude** discrepancies as diagnostic of a missing residual term". |
| R4.2 | Risk #5 wording suggested v10c was automatic; decision tree Case D escalates to review | Risk #5 reworded: "v10c (C_S bracket) is **a possible next physical knob after review** — not an automatic next step.  The escalation may instead amend the locked rule itself." |

GPT R4 quote:
> "All prior material blockers are addressed: K0/α/c_H math is
> separated, C_S/cap routing is deterministic, bracket coverage
> now hits the relevant branch windows, transport artifacts are
> routed, and the decomposition language is appropriately cautious.
> **VERDICT: APPROVED**"

## Defended (0)

None.

## Unresolved (0)

None.  All 22 issues across 4 rounds incorporated.

## Notes for downstream readers

- The R1 critique was structurally devastating: the original plan's
  central claim (`K0_R4e factor = 1e-18` opens the σ_S<0 ∩ cd_ok
  intersection) was based on conflating two different conditions.
  The single-run target moved 4 orders of magnitude (1e-18 → 1e-14)
  as a result.
- R3.1's correction is load-bearing for v10b: the F₀ amplification
  is dominantly K+ enrichment, NOT Singh ΔpKa.  This means v10b
  calibration should anchor on Γ_max / k_des (which scale with
  c_K_avg via the cap denominator), not on Singh r_H_El.
- The plan now has 7 explicit decision-tree branches.  Test the
  branches during the v10a' run interpretation — if a real run
  hits a branch not in the tree, the tree is incomplete and needs
  amending before bracket-sweep launch.
