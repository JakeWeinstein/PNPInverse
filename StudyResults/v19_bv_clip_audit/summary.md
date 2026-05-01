# V19 Stage 1 — BV exponent clip audit summary

**Script:** `scripts/studies/v19_bv_clip_audit.py`
**Date:** 2026-04-28
**Forward model:** `forms_logc.py` + 3 dynamic species (O2, H2O2, H+) + Boltzmann ClO4- background, H2O2 IC seeded at 1e-4.
**V_GRID:** [-0.10, 0.00, +0.10, +0.15, +0.20] V_RHE.
**Caps swept:** 50, 60, 70, 80, 100, None.
**Acceptance test (from `docs/PNP Anodic Solver Handoff.md` Stage 1):**

> If increasing the cap from 50 to 70/80/100 changes any of dPC/dα_2,
> weak eigenvector, sv_min, ridge_cos, then the failure is at least partly
> a clip artifact. Otherwise — and the solver fails badly without the cap —
> proceed to the FV/SG prototype.

---

## 1. Result table

```
  cap   n_V    sv_min      cond(F)    ridge_cos   weak_eigvec
   50    5    2.348e-02    2.03e+11    0.9999     [-0.00,-1.00,+0.00,+0.01]
   60    0    --- cold solve failed at z=0.000 on every voltage ---
   70    0    --- cold solve failed at z=0.000 on every voltage ---
   80    0    --- cold solve failed at z=0.000 on every voltage ---
  100    0    --- cold solve failed at z=0.000 on every voltage ---
 none    0    --- cold solve failed at z=0.000 on every voltage ---
```

`cap=50` reproduces the baseline FIM from
`docs/CHATGPT_HANDOFF_5_FIM_DEFINITIVE_NEXT_ANODIC.md` (Design B: sv_min=2.35e-2,
cond=2.03e+11, ridge_cos=1.000, weak_eigvec ≈ pure log_k0_2). Machinery confirmed.

`cap≥60` failed at the very first Newton solve of cold-start (`z=0`, ~0.2s
each). The continuation never even began. **No clip > 50 is reachable from a
cold start under this discretization.**

## 2. Verdict

The handoff's binary acceptance test fires the **second branch**: the solver
fails badly when the cap is loosened, and the FIM cannot be recomputed at
higher caps because no forward solve completes there.

We cannot conclude "the clip is *not* an artifact." We can only conclude:

> Under CG1 + log-c + 3sp + Boltzmann, the BV exponent clip is currently
> **structurally required at cap=50**. The discretization does not admit any
> wider exponent range from cold start — let alone the 1.14 V anodic frontier
> required for direct R2 voltage-shape sensitivity (cap≥73 needed; per
> mechanism analysis below).

Following the handoff plan, this points to **Stage 4 (1D FV/SG prototype)** as
the next high-confidence path. Stage 2 (log-rate BV + remove concentration
floors inside BV) and Stage 3 (cap/gamma_R2/voltage continuation) are still
worth trying as cheaper diagnostics before committing to FV/SG, because:

- Stage 2's log-rate evaluation directly attacks the `huge_exp × tiny_c` product
  that almost certainly drives the Newton blow-up here.
- Stage 3.1's cap continuation (warm-starting cap=51 from cap=50's converged
  state) could in principle reach higher caps even when cold-start cannot.

This audit does not by itself rule those paths out — it only rules out cold
starts at higher caps with the current formulation.

## 3. Mechanism (why cold solve at cap=60 failed instantly)

At V_RHE = -0.10 V the R2 overpotential is η_2 = -1.88 V, which gives
`eta_scaled = η_2 / V_T = -73.2`. Under each cap:

| cap   | clipped eta_scaled | exp(α_2·n_e·|eta_scaled|) at α_2=0.5, n_e=2 |
|-------|------------------:|--------------------------------------------:|
|  50   |             -50.0 | exp(50)  ≈ 5.18e21 |
|  60   |             -60.0 | exp(60)  ≈ 1.14e26 (220× cap=50) |
|  70   |             -70.0 | exp(70)  ≈ 2.51e30 (4.8e8× cap=50) |
| None  |             -73.2 | exp(73.2) ≈ 5.50e31 |

R2 rate at the electrode (cathodic):

```
r_2 = k0_2_HAT × c_H2O2_surf × (c_H_surf/c_ref)^2 × exp(α_2·n_e·|eta|)
    ≈ 5.26e-5 × c_H2O2_surf × 1.0 × exp(...)
```

At the cold-start IC, `c_H2O2 = 1e-4` everywhere (seeded). So at z=0 (Poisson
decoupled) Newton's first residual evaluation gives:

| cap   | r_2 magnitude at IC |
|-------|--------------------:|
|  50   | 5.26e-5 × 1e-4 × 5.18e21 ≈ 2.7e13 |
|  60   | 5.26e-5 × 1e-4 × 1.14e26 ≈ 6.0e17 |
|  70   | 5.26e-5 × 1e-4 × 2.51e30 ≈ 1.3e22 |

The Jacobian entry `dR_2/du_H2O2 = R_2` scales identically. Newton requires a
step `Δu_H2O2` that reduces this rate by 5+ orders of magnitude — i.e.,
`Δu_H2O2 ≈ -ln(220×) = -5.4` for cap=60 — far outside any line-search-safe
update. SNES line search saturates and Newton diverges before completing the
first iteration.

Notably, **z-ramp continuation does not help here** because the BV term is
present at z=0; the obstruction is in the BV reaction itself, not in the
Poisson coupling.

## 4. Verified-at-cap=50 baseline (sanity check)

Forward observables exactly match the baseline reported across the prior
handoff chain.

```
V_RHE   cd            pc           r1        r2        c_H2O2 [min, max]
-0.10  -1.7802e-01  +2.9058e-06  4.855e-1  4.855e-1  [8.05e-05, 1.00e-04]
+0.00  -1.7379e-01  +1.5288e-05  4.739e-1  4.740e-1  [6.25e-07, 1.00e-04]
+0.10  -1.6307e-01  +1.5388e-05  4.447e-1  4.448e-1  [5.01e-09, 1.00e-04]
+0.15  -1.4504e-01  +1.5394e-05  3.956e-1  3.956e-1  [4.76e-10, 1.00e-04]
+0.20  -7.8607e-02  +1.5414e-05  2.144e-1  2.144e-1  [5.41e-11, 1.00e-04]
```

`r1` and `r2` agree to 5 sig figs at every voltage in the working window —
confirming R2 is fully clipped (rates are slaved to `k0_2 × exp(50) × c_H2O2_surf`
times the same `(c_H/c_ref)^2`, and saturate at the same level as `r1`'s
clipped contribution since `α_1 ≈ α_2 ≈ 0.5–0.6`).

H2O2 minimum drops by 6 orders of magnitude from V=-0.10 (8e-5, near IC seed)
to V=+0.20 (5.4e-11) — heavy depletion at the electrode driven by the saturated
R2 sink. At still higher anodic V the depletion would push c_H2O2 below
floating-point underflow (~1e-300) before R2 unclips at V_RHE ≈ +1.14 V, which
is why no continuation pathway exists in this voltage regime under the current
formulation.

## 5. Per-V α_2 sensitivities at cap=50

```
V_RHE     dPC/dα_2       dr2/dα_2
-0.10    +2.78e-03      +7.59e-03
+0.00    +2.16e-05      +5.89e-05
+0.10    +1.67e-07      +4.38e-07
+0.15    +7.49e-09      +2.58e-08
+0.20    -1.39e-09      -1.07e-08
```

α_2 sensitivity decays by ~6 OOM across the 30 mV window. The remaining nonzero
values come entirely from the implicit `c_H2O2_surf(α_2)` coupling — direct
`∂R_2/∂α_2 = R_2 · n_e · |eta_scaled|` is identical at every V (because the
clip pins `|eta_scaled| = 50`), so the per-V variation we see reflects
H2O2 surface depletion with V, not voltage-shape information.

This matches the FIM weak eigenvector being log_k0_2-dominant: `r_2 ≈ k_0_2 ×
exp(50) × c_H2O2_surf × (c_H/c_ref)^2`, where only `k_0_2` enters as an
independent multiplicative factor in the saturated regime.

## 6. Outputs

- `convergence_by_cap.json` — z_achieved, converged flag, timing per (cap, V).
- `fim_by_cap.json` — full FIM, singular values, weak eigvec, raw S matrices,
  per-V dPC/dα_2 and dR2/dα_2 (cap=50 only; others have status=insufficient_voltages).
- `observables_by_cap.csv` — cd, pc, r1, r2 per (cap, V).
- `rates_by_cap.csv` — r1, r2, dPC/dα_2, dR2/dα_2.
- `h2o2_min_by_cap.csv` — surface c_H2O2 plus domain min/max.
- `run.log` — full stdout.

## 7. Recommended next steps

Per the handoff's staged plan:

1. **Stage 2 (cheap, ~1 day)** — log-rate BV evaluation in `forms_logc.py`,
   remove the BV concentration floor for log-c, add a catastrophic numerical
   guard (`if |log_rate_component| > 120: mark unreliable`). The current
   audit's failure mechanism is exactly the `huge_exp × tiny_c` product
   pathology that log-rate BV is designed to fix.

2. **Stage 3.1 (cheap diagnostic, hours)** — cap continuation: warm-start
   cap=51 from cap=50's converged state at V=-0.10 (where c_H2O2_surf is
   largest, hence the smallest residual jump under cap increase). If even
   cap=51 cannot be reached by warm-start, the cliff is essentially binary
   under CG1+log-c, and Stage 4 becomes unavoidable.

3. **Stage 4 (heavy, weeks)** — 1D FV/SG prototype in `Forward/fv_sg_solver/`
   if Stages 2–3 don't unstick the anodic frontier. The structural cause —
   CG1 not preserving positivity in the H2O2 depletion zone — only goes away
   with a positivity-preserving discretization.

The fastest informative next step is Stage 2 + Stage 3.1 in combination: do
log-rate BV first, then re-run this audit (and Stage 3.1 cap continuation) to
see whether log-rate evaluation alone makes higher caps reachable.
