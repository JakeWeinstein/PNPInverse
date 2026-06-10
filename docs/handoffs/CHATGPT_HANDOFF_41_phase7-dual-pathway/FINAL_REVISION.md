# Final revision — Critique session 41 (phase7-dual-pathway)

- Revised artifact: `/Users/jakeweinstein/.claude/plans/ancient-squishing-pond.md` (full rewrite incorporating all accepted issues)
- Final verdict: **APPROVED** (round 3 of cap 5)
- Codex session: 019eaf7d-3355-7a91-96a3-16ee155631eb

## Addressed (accepted)

| Issue | Fix | Landed in plan |
|---|---|---|
| R1#1 impossible 10× cap-escape threshold (≥ 4e O₂ ceiling) | Gate → >3× H⁺ cap AND ≤1.05× O₂-4e ceiling | Phase 2 |
| R1#2 species-δ "≤10%" false (δ_H/δ_O₂≈1.70, δ_OH≈1.41) | Honest risk R5 + sweep-level L_eff sensitivity | Risk R5, Phase 5a |
| R1#3 + R2#1 + R3#1 bisulfate buffering (HSO₄⁻ ceiling ≈0.83 mA/cm² is load-bearing) | Phase 3d stress-test bracket (0.1 vs 1.1 mol/m³, trigger-only label), pre-5b decision rule, future-scope closure spec | Phase 3d, Risk R4 |
| R1#4 E_eq thermodynamic language overstated | "Formal onset/reference parameter"; empirical Tafel branch framing everywhere | Context + Conventions |
| R1#5 + R2#4 + R3#2 observable layer: sum + index fragility + lying flags | `reaction_sum` mode with role-based resolution (`produces_h2o2`/`consumes_h2o2` validated against stoichiometry, both-true forbidden) | Phase 1a/1d |
| R1#6 per-reaction 4e current off by 2× (verified in code) | Raw rate + electron-weighted current outputs; Σ=cd test | Phase 1d, 3b |
| R1#7 F2 validation gate vs stale I_lim (verified) | Route-aware I_lim constants (O₂-4e for cd, O₂-2e for pc) in phase7_wls.py | Phase 5a |
| R1#8 + R2#5 + R3#3 reversible acid R2e consumes H₂O₂; analytic dismissal insufficient | Reversibility kept (anchor physics) BUT anodic-share <1% promoted to gates: Phase 0b, every accepted sweep corner, final θ | Phase 0b, 5a, 5b |
| R1#9 volcano may be Tafel competition, not local pH | Mechanism reframed as M1/M2/M3 candidates; discriminating ablations added | Context, Phase 6 |
| R1#10 hydrolysis continuation under-specified | Explicit stage-2 sequence (kw@λ=0 → Stern bump → separate λ ramp); Kw+hydrolysis+Stern flagged untested | Phase 5b, Risk R1 |
| R1#11 R3 E_eq double-shift risk | Presets store unshifted E°; single-point shift; η-preservation test | Phase 1c |
| R1#12 + R2#6 fit underdetermined; hinges too loose | Hinges labeled sanity bounds; Step 1 hunts total-current panel; conditional rename of 5b to "shape calibration"; identifiability artifacts required | Step 1, Phase 5 |
| R1#13 + R2#7 k0-only sweep can falsely reject | 2D α probe (α_w2e {0.45,0.63} × α_w4e {0.35,0.50,0.65}) before negative verdict | Phase 5a |
| R1#14 digitization σ ≠ experimental uncertainty | Rendered-curve metric semantics; markers preferred; ≤40 pts; relative χ² only | Step 1 |
| R1#15 + R2#3 validation placement (dispatch bypassed — verified: anchor_continuation imports backend directly) | Shared `_validate_reactions_vs_convergence` helper called by both build_forms_* heads; 6-cell test matrix | Phase 1b |
| R1#16 + R2#9 no global balance diagnostics | Route ledger: E closure, O₂ mass closure, electron-current consistency, acid/water share | Phase 3b |
| R2#2 post-fit L_eff bracket hides compensation | Sensitivity at sweep level (best corners re-run at 21.7/26.2 µm) | Phase 5a |
| R2#8 HSO₄⁻ closure "same pattern" was glib | Future-scope spec rewritten: charge/migration/sulfate-pool/electroneutrality | Risk R4 |
| R3#4 water-route validation should bind active routes only | "Active" = proton_donor==water AND k0>0; k0=0 entries don't force Kw | Phase 1b |
| R3#5 top-2 L_eff corners may miss flat-landscape winners | Top-4 if losses within ~1σ | Phase 5a |

## Defended (sustained)

| Issue | Defense |
|---|---|
| R1#8 (partially) | Acid R2e stays reversible: microscopic reversibility near E_eq + anchor stability; instrumented with hard <1% anodic-share gates instead of surgery |
| R1#12 (partially) | Full profile-likelihood/Bayesian treatment out of v11 scope; replaced by simplex-spread + profile slices + conditional claim downgrade |
| R2#2 (partially) | Full refits at 3 films rejected on compute grounds; sweep-level repetition accepted as the honest middle |

## Unresolved

None — verdict APPROVED in round 3.
