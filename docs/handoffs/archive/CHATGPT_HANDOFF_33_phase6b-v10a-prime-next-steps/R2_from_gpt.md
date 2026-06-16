1. WHAT: Re point 8, the statement “`1e-14` sits in the V=−0.10 window AND inside the V=−0.30 window’s upper edge” is wrong. Your own V=−0.30 window is `[1e-18, 3e-16]`; `1e-14` is ~30× above the upper edge.

WHY: The single-run target is a V=−0.10 probe only. It is not a dual V=−0.10/V=−0.30 probe.

WHAT TO DO: Rewrite that line: `1e-14` targets the high-sensitivity V=−0.10 window; the dense bracket is what covers V=−0.30 and V=−0.50.

2. WHAT: The `o2lev > 0.9` artifact branch has no next action. It says “manual review” and “do NOT route to Phase A.2,” but does not say whether to run the bracket, adapt factor, or escalate.

WHY: The most likely first run, `1e-14`, may return a plausible V but with `o2lev > 0.9`. Without a deterministic next step, the plan stalls.

WHAT TO DO: Define it as: artifact candidate = not clean. If single run returns artifact, launch dense bracket. If dense bracket only returns artifacts, escalate as “locked current filter satisfied only by transport-limited mixed-branch plateaus.”

3. WHAT: “Otherwise the returned v_kin is genuine (BV-controlled)” is too strong. `o2lev <= 0.9` is a sanity pass, not proof of BV control.

WHY: A point can be near transport transition, cap-saturated, or branch-distorted without tripping `o2lev > 0.9`.

WHAT TO DO: Replace “genuine (BV-controlled)” with “passes the O2-transport sanity check.” Keep interpretation conditional on `o2lev`, θ/cap terms, and decompositions.

4. WHAT: The proposed `R_4e_decomposition` can be numerically wrong if it uses raw `exp(-αnη/V_T)`. The solver has `EXPONENT_CLIP = 100`, and at cathodic V the raw 4e exponent exceeds that. Also, the decomposition must use the exact overpotential expression used by the BV form, not a simplified table η if Stern/diffuse terms enter.

WHY: A raw exponent can overpredict R4e by many orders and falsely blame K0 or c_H.

WHAT TO DO: Emit log-space fields: `log_k0`, `log_bv_raw`, `log_bv_clipped`, `exponent_clip_active`, `n_e*log(c_H/c_ref)`, and `log_R4e_predicted`. Use the same clipped/log-rate path as the solver.

5. WHAT: The F0 decomposition uses averages incorrectly: `10**(-pka_shift_avg)` is not `<10**(-ΔpKa)>`, and `k_hyd * <c_K> * 10**(-<ΔpKa>)` is not `<k_hyd c_K 10**(-ΔpKa)>`.

WHY: Today ΔpKa is tiny, so the error is hidden. In v10b, if ΔpKa becomes O(1), this will misattribute amplification between K+ enrichment and Singh pKa.

WHAT TO DO: Emit boundary averages of the actual factors: `<c_K>`, `<pka_factor>`, `<c_K*pka_factor>`, `F0_total`. Label any “from_c_K_only” / “from_pka_only” values as counterfactual approximations.

6. WHAT: “`denominator_cap` dominates” remains undefined.

WHY: The v10b-vs-v10c route will be subjective. One reviewer will call 55% dominant; another will call only 90% dominant.

WHAT TO DO: Set a threshold now, e.g. `denominator_cap / denominator_total > 0.8` and `theta > 0.9` at all σ<0 branch-window candidates, plus low usable `|sensS|`. Pick the exact threshold, but make it explicit.

7. WHAT: Dropping all high-factor controls means the run no longer isolates the C_S=0.20 effect. You changed C_S and K0 together, then want to diagnose cap saturation.

WHY: If θ worsens, the combined run tells you the operational outcome but not whether C_S alone caused it. That weakens the v10b/v10c routing evidence.

WHAT TO DO: Either add one C_S=0.20, factor=1 diagnostic control if wall time permits, or explicitly state that v10a′ is operational, not attribution-clean. Do not over-interpret C_S×cap causality from the combined K0 run alone.

VERDICT: ISSUES_REMAIN