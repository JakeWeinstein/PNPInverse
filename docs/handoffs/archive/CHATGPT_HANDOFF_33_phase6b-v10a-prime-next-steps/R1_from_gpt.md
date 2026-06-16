1. WHAT: `1e-18` is not the factor for a 0.3 V plateau shift. With your own slope, `ln(1e-18)/78 = -0.53 V`, not `-0.3 V`. If old onset is `~+0.30 V`, `1e-18` predicts new onset around `-0.23 V`, not `-0.50 V`. To push `+0.30 -> -0.50 V` needs about `1e-27`, not `1e-18`.

WHY: Your expectation for where the recovered operable region lands is off by several tenths of a volt. That changes which V points and factor brackets are meaningful.

WHAT TO DO: Separate the two claims: `~1e-10` for a 0.3 V kinetic-onset shift; `~1e-18` for the prior mixed-selectivity regime. Do not claim `1e-18` means `V_plateau ≈ -0.5`.

2. WHAT: You are treating `cd_ok` as “not O2-transport-limited.” That is false for mixed/pure-2e branches. The driver itself notes `|cd|/I_lim_4e = (1 + x_4e)/2` at the O2 flux limit. So if `x_4e < 0.8`, `cd_ok` can pass while `o2lev ≈ 1`.

WHY: A K0_R4e reduction may “fix” `cd_ok` by lowering the electron count, not by moving the O2 plateau out of the selected V. Then the selected point can still be transport-limited.

WHAT TO DO: Keep the locked rule unchanged, but stop describing this as purely “plateau onset moved.” For interpretation, require/report `o2_flux_levich_ratio < 0.9` as a diagnostic sanity check, or explicitly label candidates with `cd_ok=true, o2lev≈1` as current-filter artifacts.

3. WHAT: `α_R4E = 0.38` only moves onset from `+0.30` to about `0.00 V` under your algebra. It does not move it into the observed `σ_S < 0` region, which starts at `≤ -0.10 V`. To target `-0.10 V`, α is closer to `0.35`; to target `-0.30 V`, closer to `0.30`.

WHY: The α alternative is being made to look cleaner than it is. The stated α change may still leave the empty intersection.

WHAT TO DO: If comparing α fairly, compute target α for the actual required overlap voltage, not just `V_new = 0`.

4. WHAT: You omitted the `c_H^4` dependency from the plateau-shift reasoning. `K0` and α are not the only rate-scale levers; surface H+ changes are a fourth-power multiplier.

WHY: K0-only onset estimates can be wrong by orders if the C_S/hydrolysis change shifts surface H+. A 10x H+ change is equivalent to `1e4` in K0, about 0.12 V at α=0.5.

WHAT TO DO: In the rerun, decompose plateau movement into `k0 factor`, `exp BV`, and `(c_H/c_ref)^4`. Do not infer K0 effectiveness from voltage exponent alone.

5. WHAT: The plan’s line “with weaker R4e, F0 at fixed V drops, so Γ saturation should retreat; expect θ < 0.7” is wrong. `F0 = k_hyd * c_K * 10^(-ΔpKa)` is not directly coupled to `K0_R4e`.

WHY: This is the biggest Focus 2 hole. Reducing 4e kinetics does not automatically desaturate Langmuir coverage. C_S=0.20 is more likely to increase `F0` through compact-layer/K+ enrichment, pushing θ upward.

WHAT TO DO: Delete that expectation. Treat `θ`, `denominator_cap/denominator_total`, `dRnet/dσ`, and `dRnet/dlogCs` as measured outcomes. If θ is high at all `σ<0` points, route to Γmax/k_des calibration rather than blaming K0.

6. WHAT: The stated C_S mechanism is partly wrong. The v10a JSON shows `pka_shift_avg` is tiny in the cathodic region, around `1e-6` to `1e-5`, not O(1). F0 growth is dominated by K+ enrichment, not Singh ΔpKa.

WHY: If you diagnose saturation as “C_S doubled ΔpKa, so F0 grew,” you will tune the wrong part of the model.

WHAT TO DO: Decompose `F0` into `c_K` and `10^(-ΔpKa)` contributions. Keep `pka_shift_avg` as a sanity check, but do not describe it as the current load-bearing amplifier.

7. WHAT: “C_S doubles σ_S magnitude at every V” is too strong. In a Stern/diffuse series coupling, increasing C_S changes the Robin split; σ may rise sublinearly and the zero crossing need not shift meaningfully.

WHY: The plan’s “abort_to_v10c unexpected because C_S bump should make σ more cathodic” is not justified. C_S changes magnitude more than sign structure.

WHAT TO DO: Make the C_S=0.20 σ-crossing an observed diagnostic, not an expectation. Do not rely on it to recover `σ<0`.

8. WHAT: The bracket `{1, 1e-6, 1e-12, 1e-18, 1e-24}` is too sparse. From the existing v10a records, approximate branch-pass windows are:
`-0.10 V: 7e-16 to 2.5e-13`
`-0.30 V: 1e-18 to 3e-16`
`-0.50 V: 1e-21 to 5e-19`

WHY: Your bracket misses the `-0.10 V` mixed-branch window entirely and only grazes `-0.30 V`. If the sensitivity-optimal valid point is near `-0.10 V`, the sweep can falsely conclude no good factor exists.

WHAT TO DO: Use an adaptive or denser log sweep in the cathodic branch-pass region, e.g. `{1e-12, 1e-14, 1e-16, 1e-18, 1e-20, 1e-22, 1e-24}`, with `1` and `1e-6` only as controls if needed. Better: after each run, set the next factor near `R2/R4` for the best `σ<0` candidate.

VERDICT: ISSUES_REMAIN