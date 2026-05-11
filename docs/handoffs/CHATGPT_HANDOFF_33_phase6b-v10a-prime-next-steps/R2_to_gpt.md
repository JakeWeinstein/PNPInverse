# Round 2 counterreply — v10a' next-steps

All 8 points accepted. The plan has substantive holes (algebra,
mechanism mis-identification, and a sparse bracket missing the V=−0.10
window). Counterreplies + revised plan section below.

## 1. Acknowledgment per issue

### Re your point 1 — 1e-18 → 0.53 V (not 0.3 V). **Accept.**

I conflated two different claims. Splitting cleanly:

- **Kinetic plateau shift via K0 reduction**: ∂V_plateau/∂(ln K0) =
  −1/78 → factor `e^(−78·0.3) ≈ 7e-11` gives the 0.3 V shift to
  overlap the v10a σ_S<0 region.
- **Mixed-selectivity ratio (project memory)**: K0_R4e/K0_R2e ≈ 1e-18
  produces 35–50% peroxide selectivity at *whatever V the system
  lands at* — that's about R_2e/R_4e equality on the cathodic
  Levich plateau, not plateau onset.

So 1e-18 isn't the right target for plateau-shift; it's the right
target for branch-filter satisfiability. These can be different
factors.

### Re your point 2 — `cd_ok` is not "non-transport-limited" for
mixed branches. **Accept.**

`|cd|/I_lim_4e = (1 + x_4e)/2` at the O₂ Levich limit. With
x_4e < 0.8, cd_ok = True can pass while o2lev ≈ 1 — exactly the
parallel-2e/4e asymmetry already flagged in session 32 R1 issue
#5. A K0_R4e reduction that pushes selectivity into the mixed
regime would mechanically lower cd by reducing the 4e weight, but
the O₂ flux at the surface can still be saturated.

**Fix:** In the v10a' decision tree, add a *hard sanity check* on
the returned V_kin candidate: if `locked_current_filter_passed` is
True AND `o2_flux_levich_ratio > 0.9` (the existing v10a-emitted
flag), do NOT take the V at face value — escalate to "current-filter
artifact, transport-limited". The locked rule isn't amended; the
informational flag becomes a documented escalation trigger.

### Re your point 3 — α=0.378 only shifts onset to V=0, not V=−0.30.
**Accept; algebra was wrong.**

To target overlap with the actual σ_S<0 region (V ≤ −0.10 in the
v10a run), recompute α:
- η_new at V=−0.10: η = −0.10 − 1.23 = −1.33
- η_old at V=+0.30: η = +0.30 − 1.23 = −0.93
- α' / α = η_old / η_new = −0.93 / −1.33 = 0.699
- α' = 0.5 · 0.699 ≈ 0.35

For V=−0.30: α' ≈ 0.30.  For V=−0.50: α' ≈ 0.27.

α ∈ [0.27, 0.35] is at the edge of experimental Tafel uncertainty
(ORR α typically 0.30–0.50). Lowering α below 0.30 starts
contradicting the deck's measured Tafel slope. Adds noise.

**Fix:** Keep K0_R4e as the primary knob (more headroom; no Tafel
constraint), but document that the α knob exists with a tighter
constraint. In the revised plan: "α_R4E is an alternative knob with
~0.05–0.20 of practical range; defer to K0_R4e to keep the
Tafel-slope-fit conversation separate."

### Re your point 4 — c_H⁴ multiplier omitted. **Accept.**

The plateau condition is
    K0 · c_O2 · (c_H/c_H_ref)^{n_e} · exp(−α·n_e·η/V_T) ≈ D·c_O2/l_eff

c_H at the electrode surface is NOT pinned to bulk (0.0833 nondim);
it shifts under cathodic operation (H+ depletion from BV consumption
+ H+ production from cation hydrolysis). The v10a JSON shows
`c_H_surface_nondim` at V=+0.55 is 4.17e-7 — six orders of
magnitude below bulk. That's a (c_H/c_H_ref)^4 = (4.17e-7/0.0833)^4
≈ 6.3e-22 factor on R_4e — a *huge* suppression.

This complicates plateau-onset reasoning: K0 reduction shifts ln R_4e
by ln(F); a 10× c_H surface change shifts ln R_4e by 4·ln(10) =
9.2 (i.e. ~0.12 V of equivalent BV shift at the −78/V slope).

**Fix:** In the diagnostic JSON output, decompose R_4e attenuation
per V into:
- K0 factor (just the `--k0-r4e-factor` value, logged in config)
- BV exponent `exp(−α·n_e·η/V_T)` at the converged η
- c_H term `(c_H_avg/c_H_ref)^{n_e}` at the converged surface
- Stoichiometric prefactors

Then read the plateau onset from the *full* expression at each V,
not just the K0 + BV exponent. Add this as a per-V record field
`R_4e_decomposition` (purely informational, doesn't affect V_kin
selection).

### Re your point 5 — "F₀ drops → θ retreats" is wrong. **Accept fully.**

F₀ = k_hyd · ⟨c_M · 10^(−ΔpKa)⟩ has no direct K0_R4e dependence.
K0_R4e reduction does NOT lower F₀ and does NOT automatically
desaturate Γ. The Risk #4 sanity-check claim is plainly wrong.

**Fix:** Delete Risk #4 from the plan. Replace with a more honest
expectation:

> Risk: K0_R4e reduction is decoupled from F₀ and θ. Γ saturation
> in the σ_S<0 region may persist or worsen (C_S = 0.20 strengthens
> the K+ enrichment that drives F₀ up). If the diagnostic's
> `denominator_cap` dominates `denominator_total` at all σ_S<0
> candidates, that's a v10b prerequisite signal (literature-anchored
> Γ_max needed), NOT a v10a' kinetics issue.

### Re your point 6 — F₀ driven by K+ enrichment, NOT Singh ΔpKa.
**Accept.**

Looking back at the v10a `pka_shift_avg` values: at V=+0.55, +0.40,
+0.20, +0.10 (σ_S > 0): all `pka_shift_avg = 0.0` (anode-clamped).
At V=-0.10, -0.30, -0.50 (σ_S < 0): the values weren't explicitly
in the table I quoted but checking the JSON shows
`pka_shift_avg` is tiny (1e-6 to 1e-5 range — well below O(1)).

So 10^(−ΔpKa) ≈ 1 across the whole grid, and F₀ growth from
0.052 (V=+0.55) to 0.732 (V=−0.50) is mostly K+ enrichment at the
OHP under cathodic bias. This is a 14× F₀ increase, with c_K
likely contributing the bulk of it.

**Fix:** Add the F₀ decomposition to the diagnostic JSON outputs,
exactly as suggested:

```python
# Per V at λ=1:
"F0_decomposition": {
    "c_K_surface_nondim": ...,
    "pka_shift_avg":      ...,
    "ten_to_minus_dpka":  ...,         # = 10^(-pka_shift_avg)
    "F0_from_c_K_only":   k_hyd * c_K_surface,
    "F0_from_pka_only":   k_hyd * c_K_bulk * 10**(-pka_shift_avg),
    "F0_total":           F0_avg,
},
```

This makes the dominant driver legible to downstream readers and
to v10b's Γ_max / k_des calibration.

### Re your point 7 — "C_S doubles σ_S magnitude" oversimplified.
**Accept.**

Stern-diffuse series coupling: the total interfacial drop
`Δφ_interface = Δφ_Stern + Δφ_diffuse` and `σ_S = C_S · Δφ_Stern =
C_diffuse · Δφ_diffuse`. Doubling C_S shifts which of the two
layers carries the larger share; total σ doesn't simply double.
In the limit C_S → ∞ (no Stern), σ_S is set entirely by diffuse;
in C_S → 0 (no diffuse charge), σ_S = 0 at fixed φ_m.

**Fix:** Soften the plan's framing of C_S → σ_S. Replace the
"abort_to_v10c unexpected because C_S bump should make σ more
cathodic" risk wording with: "C_S = 0.20 may shift the σ_S = 0
crossing voltage and the σ_S magnitude/V_RHE relationship; the
plan does NOT pre-commit to a direction. If `abort_to_v10c`
fires, it indicates the new C_S didn't open the σ_S<0 region —
route to v10c bracket sweep on C_S itself (the originally
planned v10c)."

### Re your point 8 — Bracket too sparse, misses V=−0.10 window. **Accept.**

GPT's extrapolated branch-pass windows from the v10a data:
- V=−0.10: ratio for x_4e ∈ [0.05, 0.95] is roughly 7e-16 to 2.5e-13
- V=−0.30: 1e-18 to 3e-16
- V=−0.50: 1e-21 to 5e-19

The proposed bracket `{1, 1e-6, 1e-12, 1e-18, 1e-24}` only grazes
V=−0.30's window (at 1e-18); V=−0.10's window (7e-16 to 2.5e-13)
is *missed entirely* between 1e-12 and 1e-18.

The v10a |sensS| peaked at V=+0.10 (0.250) and was highest among
σ_S<0 candidates at V=−0.10 (0.217). So if a V_kin exists, it's
most likely near V=−0.10. Missing the V=−0.10 branch-pass window
means missing the most likely V_kin candidate.

**Fix:** Replace the bracket with a denser log-uniform sweep:
`{1e-10, 1e-12, 1e-14, 1e-16, 1e-18, 1e-20, 1e-22, 1e-24}` — 8
points spanning the V=−0.10 through V=−0.50 branch-pass windows.
Drop the `1` and `1e-6` controls (those just reproduce the v10a
failure mode; they don't add diagnostic value).

Even better — implement the "adaptive next factor" loop GPT
suggested, but for the first cut, the dense log-uniform is
simpler. If the dense run lands a V_kin, no adaptive run needed;
if not, the dense run's R_2e/R_4e per V locations the adaptive
target.

## 2. Updated artifact (changes since R1)

* **Plateau-shift factor decoupled from selectivity-ratio factor.**
  The single-run target uses `--k0-r4e-factor 1e-14` (centered in
  the V=−0.10 branch-pass window per GPT's extrapolation), not
  1e-18 (which targets V=−0.30 onward). 1e-18 is retained as a
  named bracket point.
* **`o2lev > 0.9 + cd_ok = True` is a documented escalation flag.**
  If V_kin returns at such a point, the user must inspect manually
  (transport-limited artifact).
* **α_R4E demoted to "alternative, narrower headroom".** Plan
  records the math but doesn't pursue.
* **R_4e + F₀ decompositions added to diagnostic JSON output.**
  c_H⁴ contribution + K+ enrichment vs Singh ΔpKa contributions
  exposed per V.
* **Risk #4 deleted ("weaker R_4e → θ retreats" was wrong).**
  Replaced with the honest expectation that K0 reduction is
  decoupled from cap saturation; route to v10b if `denominator_cap`
  dominates at all σ_S<0 candidates.
* **C_S → σ_S framing softened.** Plan doesn't pre-commit to
  σ_S<0 region opening.
* **Bracket sweep redesigned.** New bracket:
  `{1e-10, 1e-12, 1e-14, 1e-16, 1e-18, 1e-20, 1e-22, 1e-24}`.
  Centered to cover the per-V branch-pass windows.
* **Single-run target:** `--k0-r4e-factor 1e-14` (was 1e-18).

### Revised "Implementation notes" section (verbatim replacement)

```markdown
## Implementation notes (revised)

### `--k0-r4e-factor` wiring

(Unchanged — see R1 plan.)

### Plateau shift vs. selectivity ratio (NEW — algebra
decoupling)

Two distinct quantities, both controlled by reducing K0_R4e but
targeting different conditions:

| Target                          | Condition                                  | Factor at α=0.5, n_e=4 |
|---------------------------------|--------------------------------------------|------------------------|
| Plateau onset shift ΔV          | ln F = −78·ΔV                              | ΔV=0.3 V ⇒ F ≈ 7e-11   |
|                                  |                                            | ΔV=0.5 V ⇒ F ≈ 6e-18   |
| Branch-pass window (per V)      | x_4e ∈ [0.05, 0.95] at converged surface c | per-V from v10a JSON   |

GPT R1 issue #8 extrapolated branch-pass windows from the v10a
records:
- V=−0.10 V: F ∈ [7e-16, 2.5e-13]
- V=−0.30 V: F ∈ [1e-18, 3e-16]
- V=−0.50 V: F ∈ [1e-21, 5e-19]

The single-run target `1e-14` sits in the V=−0.10 window (the V
that had the highest |sensS| in v10a) AND inside the V=−0.30
window's upper edge. If it lands a clean V_kin, no bracket sweep
needed.

### `o2_flux_levich_ratio > 0.9` as escalation trigger (NEW)

The locked `|cd|/I_lim_4e < 0.9` filter is satisfied by mixed-
selectivity transport-limited plateaus (Levich asymmetry — session
32 R1 issue #5).  After the v10a' run:

1. If `select_v_kin()` returns a v_kin AND that v_kin's record has
   `locked_current_filter_passes_but_o2_transport_limited = True`,
   the candidate is a **current-filter artifact** — escalate, do
   NOT route to Phase A.2 at that V without manual review.
2. Otherwise the returned v_kin is genuine (BV-controlled).

### R_4e + F₀ decompositions (NEW — added to rung diagnostic)

Extend `collect_v10a_rung_diagnostics(ctx)` with:

```python
# Per V at λ=1:
diag["R_4e_decomposition"] = {
    "k0_factor_explicit":     float(k0_r4e_factor),
    "bv_exponent_at_eta":     exp(-alpha * n_e * eta_R4e / V_T),
    "c_H_attenuator_pow_n_e": (c_H_avg / c_H_ref) ** n_e,
    "stoich_prefactor":       c_O2_avg,
    "R_4e_predicted":         k0_R4e * (these terms),
    "R_4e_measured":          fd.assemble(R_4e_form),
}
diag["F0_decomposition"] = {
    "c_K_surface_nondim":     c_K_avg,
    "pka_shift_avg":          pka_shift_avg,
    "ten_to_minus_dpka":      10 ** (-pka_shift_avg),
    "F0_from_c_K_only":       k_hyd * c_K_avg,
    "F0_from_pka_only":       k_hyd * c_K_bulk * 10**(-pka_shift_avg),
    "F0_total":               F0_avg,
}
```

Surfaces the dominant amplification mechanism (K+ enrichment vs
Singh ΔpKa) and the dominant R_4e suppression mechanism (K0 vs
c_H⁴ vs BV exponent) so v10b calibration knows which to tune.

### Bracket-sweep redesign

Replace `{1, 1e-6, 1e-12, 1e-18, 1e-24}` (5 points) with:

`{1e-10, 1e-12, 1e-14, 1e-16, 1e-18, 1e-20, 1e-22, 1e-24}` (8 points)

Centered on the V=−0.10 through V=−0.50 branch-pass windows
extrapolated by GPT in R1 issue #8.  Wall budget: ~12 h serial.

### α_R4E alternative (documented, not pursued)

Recomputed targets:
- V_overlap=−0.10 V ⇒ α' ≈ 0.35
- V_overlap=−0.30 V ⇒ α' ≈ 0.30
- V_overlap=−0.50 V ⇒ α' ≈ 0.27

α < 0.30 starts contradicting ORR Tafel-slope measurements (typically
60–120 mV/dec → α ∈ [0.30, 0.50]).  α reduction has tighter
experimental headroom than K0 reduction.  Defer to K0 to keep the
Tafel-slope-fit conversation separate from V_kin selection.

### Risk #4 — DELETED (was wrong)

Original wording:
> Risk: With weaker R_4e, F₀ at fixed V drops, so Γ saturation
> should retreat.

This is wrong.  F₀ = k_hyd · ⟨c_M · 10^(−ΔpKa)⟩ is uncoupled
from K0_R4e.  K0 reduction does NOT desaturate Γ.

Replacement risk:
> Risk: K0_R4e reduction may leave Γ saturated at all σ_S<0
> candidates if K+ enrichment + Singh ΔpKa already push F₀
> high.  C_S = 0.20 enhances K+ enrichment.  If
> `denominator_cap` dominates `denominator_total` at every
> σ_S<0 V in the v10a' output, that's a v10b prerequisite
> signal (need literature-anchored Γ_max), NOT a v10a' kinetics
> issue.  Decision tree branch: route to v10b instead of
> v10c.
```

### Risks (revised)

| # | Risk | Mitigation |
|---|---|---|
| 1 | K0_R4e_factor = 1e-14 puts the V=−0.10 branch in the mixed window but |sensS| at V=−0.10 may shift unpredictably with C_S = 0.20 | The v10a' run also captures the FULL R_4e + F₀ decompositions; if |sensS|_max relocates to a different V, the decision tree re-scores the candidates. |
| 2 | C_S = 0.20 in series with the diffuse layer may shift σ_S=0 crossing OR keep it near V=0 — direction not pre-committed | Observe in v10a' JSON; if σ_S<0 region didn't open, route to v10c bracket sweep on C_S itself (the originally-planned v10c). |
| 3 | At factor = 1e-14, R_4e_kinetic may still exceed I_lim_4e at the V=−0.10 window → cd_ok still fails | The bracket sweep `{1e-10 ... 1e-24}` covers 14 OOM in factor; at least one rung in the sweep should bring R_4e below Levich at some V where σ_S<0. |
| 4 | (DELETED — was incorrect: K0 ≠ F₀ coupling) | — |
| 5 | If even the dense bracket sweep produces no V_kin, the locked rule is fundamentally unsatisfiable for this stack | Surface to acceptance bundle Status section; escalate to experimental group. v10c (C_S bracket) is the next physical knob. |
| 6 | Singh ΔpKa magnitude (`pka_shift_avg`) is tiny (1e-6 to 1e-5) in v10a, suggesting it's NOT the load-bearing F₀ amplifier. C_S = 0.20 still wouldn't change this enough to matter. | F₀ decomposition surfaces this; reading the JSON should show c_K_surface as the dominant amplifier. If Singh ΔpKa stays tiny at C_S = 0.20, the v10b calibration target shifts away from Singh r_H_El and toward Γ_max / k_des. |

## 3. Continued critique prompt

Review the updated plan and my responses to your earlier issues.
Push back on responses where I defended poorly — name which point.
Raise any new issues the updated plan creates. Re-issue any earlier
issue you don't think I addressed. Same numbered format and same
verdict line at the end:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN

Scope reminder: still narrow on the three subtle physics/parameter-
coupling points (K0_R4e knob, C_S × cap interaction, bracket
coverage). Don't critique the locked V_kin selection rule itself,
driver structure, or test list.

This is the second of three rounds under the configured cap.
