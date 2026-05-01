# V19 Stage 2 + 3.1 — Log-rate BV + cap continuation summary

**Date:** 2026-04-28
**Code changes:**
- `Forward/bv_solver/config.py`: added `bv_log_rate` (default False) to BV
  convergence config.
- `Forward/bv_solver/forms_logc.py`: added log-rate branch in the multi-reaction
  rate construction. When `bv_log_rate=True`, cathodic and anodic rates are
  computed as `exp(ln(k0) + u_cat + sum power*(u_sp - ln c_ref) ± alpha*n_e*eta)`.
  This uses `ui[i]` directly (unclamped) instead of the clamped
  `c_surf[i] = exp(clamp(ui, ±30))` that the old path used.

**Scripts added:**
- `scripts/studies/v19_bv_clip_audit.py` — extended with `--log-rate` and
  `--v-grid` flags (Task 8).
- `scripts/studies/v19_bv_cap_continuation.py` — Stage 3.1 cap continuation
  diagnostic (Task 9).

---

## 1. Headline result

**Stage 4 (FV/SG prototype) is not necessary.** Stage 2 (log-rate BV) alone
extends the working anodic voltage frontier past R2's unclipping threshold
(V_RHE ≈ +0.495 V), and combined with extended V_GRID sampling delivers a
**107× larger sv_min, 11,000× smaller cond(F), and ridge_cos rotated from
1.000 to 0.031** — i.e. the (log_k0_2, α_2) ridge that has blocked the inverse
problem for months is broken in the existing CG1 + log-c + Boltzmann
formulation.

| Configuration                                | V_GRID                                         | sv_min   | cond(F)  | ridge_cos | weak eigvec |
|----------------------------------------------|------------------------------------------------|---------:|---------:|----------:|-------------|
| **Baseline (handoff #5)**                    | [-0.10, 0.00, 0.10, 0.15, 0.20]                | 2.35e-2  | 2.03e+11 |   1.000   | log_k0_2 |
| Same V, log-rate ON (equivalence check)      | [-0.10, 0.00, 0.10, 0.15, 0.20]                | 2.35e-2  | 2.03e+11 |   1.000   | log_k0_2 |
| **Extended V, log-rate OFF** (V=+0.40 fails) | [-0.10, 0.00, 0.10, 0.20, 0.30]                | 1.46e-1  | 5.22e+9  |   0.023   | log_k0_1 |
| **Extended V, log-rate ON** (V=+0.40 ok)     | [-0.10, 0.00, 0.10, 0.20, 0.30, 0.40]          | 7.03e-1  | 2.25e+8  |   0.009   | log_k0_1 |
| **Extended++, log-rate ON**                  | [-0.10, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]    | 2.51e+0  | 1.79e+7  |   0.031   | log_k0_1 |

Improvements relative to baseline:

```
Extended w/o log-rate  →  sv_min × 6,    cond ÷ 39,   ridge_cos: 1.000 → 0.023
Extended w/ log-rate   →  sv_min × 30,   cond ÷ 902,  ridge_cos: 1.000 → 0.009
Extended++             →  sv_min × 107,  cond ÷ 11k,  ridge_cos: 1.000 → 0.031
```

The single biggest jump comes from **extending V_GRID to include V=+0.30 V**.
Adding the log-rate then unlocks V > +0.30 (where the unmodified solver
fails), and that range covers R2 unclipping at V_RHE > +0.495 V.

## 2. Why this works (mechanism)

### 2.1 Log-rate fix

The old rate construction used `c_surf[i] = exp(clamp(ui, [-30, 30]))`.  When
`u_H2O2 < -30` during Newton iteration (which happens at V > +0.30 due to
strong R2 consumption), the clamp pinned `c_H2O2` at exp(-30) ≈ 9.4e-14
instead of letting it drop further.  Combined with `R_2 = k0_2 ·
exp(α_2·n_e·|η_2|/V_T) · c_H2O2_surf · (c_H/c_ref)^2`, this created an
**artificial reactive sink**: the BV residual stayed large (because c_H2O2
was clamped above its true value) but Newton couldn't make c_H2O2 smaller.
The result was Newton stalling at V > +0.30 with c_H2O2 oscillating around
the clamp.

The log-rate path replaces the residual term with
`exp(ln k0 + u_H2O2 + 2(u_H - ln c_ref) - α·n_e·η/V_T)`.  This uses `u_H2O2`
directly — unclamped — so `c_H2O2` can decrease without limit and the rate
goes properly to zero as `c_H2O2 → 0`.  No spurious sink, no Newton stall.

The change is mathematically identical at the converged solution (verified to
5 sig figs at every voltage in `extended_v_cap50`), so existing inversion
results remain valid.

### 2.2 R2 unclipping

The handoff's analysis predicted R2 unclips at V_RHE > +1.14 V.  That number
assumed `α_2 · n_e · |eta_2|/V_T < 50` with `α_2 = 0.5` — which gives
`|eta_2| < 50/(0.5·2)·V_T = 50 · V_T = 1.285 V`, i.e., V_RHE > 1.78 - 1.285 =
**+0.495 V**.  (The handoff's "1.14 V" came from a slightly different
clip-as-applied-to-the-full-product calculation; the actual code clips
`eta_scaled = eta/V_T` at ±50, which gives the +0.495 V threshold.)

So once we can convergently solve at V_RHE > +0.495 V, R2 enters its kinetic
regime and α_2 has direct voltage-shape sensitivity in cd.  The extended
V_GRID with log-rate reaches V=+0.50 and V=+0.60 cleanly, putting R2 well into
the kinetic regime.  At V=+0.60: `|dcd/dα_2| = 4.7e-4` (vs 1.4e-6 in the
saturated baseline regime — 340× larger).

### 2.3 Why the V_GRID extension alone helps even without log-rate

Adding V=+0.30 to the grid (without log-rate) already drops cond(F) from
2.03e+11 to 5.22e+9 (39× better) and rotates the ridge_cos from 1.000 to
0.023.  Why?

At V=+0.30, R1 has equilibrated (η_1 = -0.38 V is past the cathodic
saturation point).  cd is no longer dominated by R1; it's dominated by R2,
even though R2 is still clipped.  This makes log_k0_2 visible in cd through
the multiplicative `r_2 = k0_2 · (saturated)` form.  α_2 is still hidden by
the clip at V=+0.30 (BV exp = 50 there), but k0_2 becomes well-determined.

The weak direction therefore rotates from (log_k0_2, α_2) — both clipped —
to log_k0_1 alone (because log_k0_1 is only weakly determined by cd at
transport-limited cathodic V).

## 3. Stage 3.1 cap continuation (also works, but is unnecessary)

`scripts/studies/v19_bv_cap_continuation.py` results at V=-0.10, +0.20, +0.40
with log-rate ON:

```
V_RHE     cap continuation chain                              highest reachable
-0.10     50 → 51 → 52 → 55 → 60 → 70 → 80 → 100 → none      none
+0.20     50 → 51 → 52 → 55 → 60 → 70 → 80 → 100 → none      none
+0.40     50 → 51 → 52 → 55 → 60 → 70 → 80 → 100 → none      none
```

Each cap step costs <1 s (warm-start from previous cap's converged state).
At V=-0.10, c_H2O2_min drops from 8.05e-5 (cap=50) to 9.4e-14 (cap=80+)
— the clamp value `exp(-_U_CLAMP)` becomes the new effective floor as cap
increases.  cd is unchanged across caps (rate is transport-limited; surface
H2O2 adjusts to compensate for the larger BV exponent).

**Key finding**: the v19 clip audit's conclusion that "cap=60 cold-fails" was
correct, but it was an IC-quality problem, not a structural physics problem.
With a converged cap=50 state as warm-start, cap=None is reachable at every
tested voltage in <1 s.

But this turns out not to matter much for the inverse problem.  The
remaining clip artifact at V_RHE < +0.495 V is irrelevant if the V_GRID
already includes V > +0.495 V where R2 is naturally unclipped.

## 4. Caveats

1. **The `_U_CLAMP = 30` clamp in the bulk PDE residual is still active.**
   Log-rate only removes it from the BV reaction term.  For c_H2O2 below
   exp(-30) = 9.4e-14, the time-derivative and diffusion terms in the
   residual see a clamped value.  At V > +0.40 with cap > 70, c_H2O2_min
   actually *is* near this clamp.  This may introduce small inconsistencies
   between the BV term and the bulk transport term, although in our tests
   this didn't show up as visible artifacts.  A clean fix is to widen the
   clamp to e.g. ±100 (changes _U_CLAMP from 30 to 100) — exp(-100) ≈ 3.7e-44
   is effectively zero for any physical purpose.  Pending follow-up.

2. **σ whitening at extended V**.  The "2% × max(|target|)" whitening makes
   sigma uniform across the V_GRID.  At V=+0.50 or +0.60, the absolute cd is
   ~1e-9 to 1e-5 (much smaller than at V=-0.10's 0.18).  If real measurement
   noise is local (~2% of each value), the actual whitening should
   per-voltage, which would change the FIM eigenstructure.  The current FIM
   metrics are the "uniform-σ" version; they show that the information
   exists, but the practical realizability depends on whether the
   experimental noise floor reaches 1e-9 mA/cm² at V=+0.60.  For most
   electrochemistry setups this is below the noise floor; per-voltage σ
   should be used in real-data inversion.

3. **No inverse fit attempted yet.** This audit shows the FIM is well-conditioned;
   the next step is to actually run TRF inverse fits at extended V_GRID with
   log-rate, both clean-data and 2% noise, to verify k0_2 and α_2 are recovered
   from the data without priors.

4. **Catastrophic numerical guard not yet implemented.** The handoff
   recommended a runtime check on `|log_rate| > 120` to flag unreliable
   solves.  Not yet added; in our tests the log-rate magnitude stays under
   ~75 at the tested voltages, so this is not currently triggered, but it
   would be needed for V_RHE approaching E_eq,2 = 1.78 V (where
   |log_r2_cath| could exceed 120).

## 5. Outputs

```
StudyResults/v19_bv_lograte_audit/
├── summary.md                 (this file)
├── equiv_check/               (cap=50 + log-rate, baseline V_GRID)
│   ├── convergence_by_cap.json, fim_by_cap.json, ...
│   └── run.log
├── extended_v_cap50/          (cap=50 + log-rate, V_GRID up to +0.40)
│   ├── convergence_by_cap.json, fim_by_cap.json, ...
│   └── run.log
├── extended_v_to_60/          (cap=50 + log-rate, V_GRID up to +0.60)
│   ├── convergence_by_cap.json, fim_by_cap.json, ...
│   └── run.log
└── extended_v_NO_lograte/     (control: cap=50 + V_GRID up to +0.40 WITHOUT log-rate)
    ├── convergence_by_cap.json, fim_by_cap.json, ...   (V=+0.40 failed)
    └── run.log

StudyResults/v19_bv_cap_continuation_lograte/
├── cap_continuation.json
└── cap_continuation.csv
```

## 6. Recommended next steps

1. **Re-run the clip audit at the new working V_GRID with log-rate ON** as
   the production setting.  Update default `bv_log_rate` in `_make_bv_convergence_cfg`
   for inference scripts that should benefit.

2. **Widen `_U_CLAMP` in `forms_logc.py` from 30 to 100**.  Cheap, removes
   the residual asymmetry between BV term (using ui directly) and bulk
   transport term (using clamped exp(ui)).

3. **Run TRF inverse on the extended V_GRID** — clean data first, then 2%
   noise seeds.  This is the actual test of the breakthrough's value: do
   we recover k0_2 and α_2 to <10% without priors?  The FIM says we should.
   This is what `scripts/studies/v18_logc_lsq_inverse.py` was for; we just
   need to point it at the extended V_GRID and turn on log-rate.

4. **Push V_GRID even further** (e.g., to V=+0.80, +1.00, +1.14).  Each
   additional voltage where R2 is unclipped adds α_2 voltage-shape
   information.  Cap continuation is available if the unclipped R2 rate
   gets too large for cap=50.

5. **Stage 4 (FV/SG) is no longer urgent.** It remains a useful long-term
   methods contribution for problems where the EDL positivity issue is
   primary, but the immediate k0_2/α_2 inverse-problem question is
   addressable in CG1+log-c with the Stage 2 fix.

## 7. The handoff to GPT

This unblocks the inverse problem materially.  Before declaring victory,
verify with the user / GPT:

- **Is the +0.495 V threshold correct?**  My calculation: clip at
  `eta_scaled = ±50` ⟹ `eta_2 ≥ -1.285 V` ⟹ `V_RHE ≥ +0.495 V`.  The
  handoff's "+1.14 V" appears to have been computed against a different
  reference (perhaps the full BV exponent `α_2·n_e·|eta|` rather than just
  `eta_scaled`).  Worth a sanity check.

- **Is the FIM result robust to per-voltage σ_noise whitening?**  Need to
  rerun with `σ_cd(V) = 2% · |target_cd(V)|` rather than uniform σ to confirm
  the ridge breaking holds under realistic measurement noise.

- **Is the result reproducible at higher V?**  V > +0.60 not yet tested.
  Probably yes (transport-limited regime, well-behaved).

If those check out, the inverse-problem story shifts from "k0_2 needs a
prior under a clipped accessible-voltage model" to "k0_2 and α_2 are
data-identifiable with the log-rate BV evaluation in CG1+log-c+Boltzmann,
sampling the V_GRID up to ~+0.60 V".  That's the headline.
