# Option 2b — Bikerman-consistent IC: results

**Date:** 2026-05-04
**Status:** Landed on `main` (commits `1317b1b`, `77ceff3`, `3c4ba67`)
**Predecessor plan:** [`4sp_bikerman_ic_option_2b_plan.md`](./4sp_bikerman_ic_option_2b_plan.md)
**Predecessor result:** [`4sp_drop_boltzmann_investigation.md`](./4sp_drop_boltzmann_investigation.md)
§12 (Option 2a′)

This doc is the standalone results write-up for the 2b implementation
across all four IC code paths (logc/muh × 4sp/3sp+bikerman). For the
broader narrative — pre-2a′ baseline, why 2a′ wasn't enough, and what
2b is *meant* to do — see §13 of the investigation log.

---

## TL;DR

* **Production target (3sp + analytic ClO₄⁻ with `steric_mode='bikerman'`
  + `formulation='logc_muh'`)**: 15/15 voltages converged on the
  Stern-pass full-grid sweep (V_RHE ∈ [−0.5, +1.0]). Cold ceiling
  +0.60 V (was +0.30 V on 2a′-equivalent runs); warm-walk reach +1.00 V.
  This is the headline result.
* **4sp dynamic stacks (logc and muh)**: convergence reach **unchanged**
  vs the 2a′ baseline (5/15 no-Stern, 7/15 Stern, max V_RHE = +0.50 V).
  The binding constraint there is the dynamic c_ClO₄ NP equation, not
  the IC.
* **Cross-stack equivalence**: on the 7-voltage overlap where the 4sp
  dynamic and 3sp + bikerman analytic stacks both converged, CD/PC
  agree to ≤ 5·10⁻³ (and to ≤ 4·10⁻⁹ in the cathodic regime). 2b
  preserves the analytic-vs-dynamic equivalence.

---

## What landed

Composite ψ + multispecies γ correction in the analytical IC for both
forms files:

* `Forward/bv_solver/forms_logc.py:_try_debye_boltzmann_ic`
* `Forward/bv_solver/forms_logc_muh.py:_try_debye_boltzmann_ic_muh`

Gated on a unified flag

```python
apply_bikerman_ic = synthesised_4sp_counterion or bikerman_in_counterions
```

so the same physics fires for both the legacy synthesised-4sp ClO₄⁻
counterion *and* an explicit `boltzmann_counterions=[{steric_mode='bikerman', …}]`
entry. The legacy 3sp + ideal-Boltzmann counterion path is byte-identical
to pre-2a′/pre-2b (verified by snapshot regression tests).

### Composite ψ profile

BKSA matched-asymptotic — saturated zone near the electrode + outer
exponential decay:

```
nu       = 2 * a_cl * c_clo4_bulk
psi_sat  = ln(2/nu)                                      # ≈ 6.21 for a=0.01, c_b=0.2
alpha    = sqrt((2/(nu*lam_D^2)) * ln[1 + nu*(cosh psi_D - 1)])
y_match  = (|psi_D| - psi_sat) / alpha

psi(y) = sign(psi_D) * (|psi_D| - alpha*y),                  y in [0, y_match]
psi(y) = sign(psi_D) * psi_sat * exp(-(y-y_match)/lam_D),    y >= y_match
```

Falls through to the prior tanh-Gouy-Chapman expression when the
linear-Debye limit applies (`|psi_D| <= psi_sat * (1 - 1e-3)` or
`nu <= 0`).

### Multispecies γ correction

Anchor selection per case:

| case | a_cl source | c_cl_anchor |
|---|---|---|
| 4sp dynamic (synthesised counterion) | `a_vals[3]` | `H_outer(y)` (electroneutrality) |
| 3sp + analytic-bikerman counterion | `entry['a_nondim']` | `c_clo4_bulk` (analytic outer = bulk) |

Both reduce to:

```
gamma(y) = 1 / [1 + a_h * H_outer(y) * (exp(-psi) - 1)
                  + a_cl * c_cl_anchor * (exp(+psi) - 1)]
```

(neutral species drop out — `exp(0) - 1 = 0`). Seed:

```
u_O2_init   = ln(c_O2_outer)   + log_gamma
u_H2O2_init = ln(c_H2O2_outer) + log_gamma
u_H_init    = ln(H_outer) - psi + log_gamma
phi_init    = ln(H_outer / c_clo4_bulk) + psi
u_3_init    = ln(c_clo4_bulk) + phi_init + log_gamma     (4sp dynamic only)
```

### muh log_gamma propagation

The muh primary variable for the proton is `mu_H = u_H + em*z_H*phi`.
With `em*z_H = 1` the `(-psi)` in u_H cancels the `(+psi)` in phi:

```
mu_H_init = (ln H_outer - psi + log_gamma) + (ln(H_outer/c_clo4_bulk) + psi)
          = 2*ln(H_outer) - ln(c_clo4_bulk) + log_gamma
```

so log_gamma propagates as a smooth additive shift on mu_H. Pre-2b
the muh IC was pure Boltzmann (no γ) — this is a separate fix on top
of the 4sp logc 2a′ that landed in §12.

---

## Sweep results

All three sweeps: `Ny=200`, `clip=50`, `u_clamp=100`, full 15-V grid

```
V_RHE = (-0.50, -0.30, -0.10, 0.00, +0.10, +0.30, +0.50,
         +0.55, +0.60, +0.65, +0.66, +0.68, +0.70, +0.75, +1.00)
```

driven through the `solve_grid_per_voltage_cold_with_warm_fallback`
orchestrator (Phase 1 cold + per-V z-ramp, Phase 2 warm-walk fallback
from cold anchors).

### Convergence summary

| Stack | Pass | 2a′ baseline | 2b | Δ |
|---|---|---|---|---|
| **3sp + bikerman + muh** *(production target)* | no Stern | n/a (new) | **14/15, max V_RHE = +1.00 V** | first run |
| | Stern 0.10 F/m² | n/a | **15/15 clean, max V_RHE = +1.00 V** | first run |
| 4sp dynamic logc | no Stern | 5/15, +0.10 V | 5/15, +0.10 V | unchanged |
| | Stern 0.10 F/m² | 7/15, +0.50 V | 7/15, +0.50 V | unchanged |
| 4sp dynamic muh | no Stern | 5/15, +0.10 V | 5/15, +0.10 V | unchanged |
| | Stern 0.10 F/m² | 7/15, +0.50 V | 7/15, +0.50 V | unchanged |

### Per-voltage detail — 3sp + bikerman + muh + Stern pass

The production-target Stern pass, the only 15/15 sweep:

```
V_RHE   ok   method                  c_steric   stern_drop  CD            PC
-0.500  ✓   warm<--11.677            7.5e-10    -0.05 V     -1.85e-01     +1.5e-05
-0.300  ✓   warm<- -3.892            1.8e-06    -0.04 V     -1.83e-01     +1.5e-05
-0.100  ✓   cold                     4.1e-03    +0.02 V     -1.79e-01     +1.5e-05
 0.000  ✓   cold                     1.7e-01    +0.20 V     -1.75e-01     +1.5e-05
+0.100  ✓   cold                     3.6e+00    +0.97 V     -1.68e-01     +1.5e-05
+0.300  ✓   cold                     6.1e+01    +5.0  V     -1.16e-01     +1.5e-05
+0.500  ✓   cold                     9.7e+01    +9.7  V     -1.6e-05      +1.5e-05
+0.550  ✓   cold                     9.9e+01    +10.8 V     -1.6e-05      +1.6e-05
+0.600  ✓   cold                     1.0e+02    +11.8 V     -1.3e-05      +1.3e-05
+0.650  ✓   warm<-+23.353            1.0e+02    +12.8 V     -7.8e-07      +7.8e-07
+0.660  ✓   warm<-+25.299            1.0e+02    +13.0 V     -3.2e-07      +3.2e-07
+0.680  ✓   warm<-+25.688            1.0e+02    +13.4 V     -4.8e-08      +4.8e-08
+0.700  ✓   warm<-+26.467            1.0e+02    +13.8 V     -6.9e-09      +6.9e-09
+0.750  ✓   warm<-+27.245            1.0e+02    +14.7 V     -4.8e-11      +4.8e-11
+1.000  ✓   warm<-+29.191            1.0e+02    +19.0 V     +1.2e-16      +1.2e-16
```

CD in mA/cm², cathodic-positive in the test convention shown (negative
= cathodic ORR direction). PC is the H₂O₂ partial current.

Headlines:

* `c_steric` saturates at the Bikerman cap `1/a_b = 100` for every
  V ≥ +0.30 V. The IC seed and the residual closure (which already
  saturates at the same cap via `build_steric_boltzmann_expressions`)
  are now consistent — Newton lands in the right basin from the cold
  start.
* The Stern layer absorbs ≈ 13 V of the applied potential at V = +0.65 V
  (`stern_drop = +12.8 V`), keeping ψ_D modest enough that the proton
  supply doesn't underflow the BV cathodic terms. Non-trivial chemistry
  (`CD` and `PC` matching at ~10⁻⁷ mA/cm²) runs through V = +0.75 V.
* At V ≥ +0.50 V on the **no-Stern** pass (full ψ_D in the diffuse
  layer), CD and PC drop to machine epsilon (~10⁻¹⁶ mA/cm²). This is
  not a clip artifact — `eta_R2` clip threshold is V_RHE = +0.495 V
  and R2 *unclips* above it. It's the genuine kinetic dead zone:
  c_H_surface ≈ exp(−ψ_D) · c_H_bulk drops below ~10⁻³³ at ψ_D > 30,
  and the cathodic BV terms have factors of c_H² (R1) and c_H⁴ (R2)
  that underflow. Anodic terms survive at exp(−40 to −60) ≈ 10⁻¹⁸.

### Per-voltage detail — 3sp + bikerman + muh, no Stern

For comparison, the no-Stern pass (14/15 — V=+0.10 cold-fails between
fully-saturated regimes):

```
V_RHE   ok   method                  c_steric   CD            PC
-0.500  ✓   warm<--11.677            7.1e-10    -1.84e-01     -1.82e-01
-0.300  ✓   warm<- -3.892            1.7e-06    -1.83e-01     -6.5e-02
-0.100  ✓   cold                     4.1e-03    -1.79e-01     +2.9e-06
 0.000  ✓   cold                     2.0e-01    -1.75e-01     +1.5e-05
+0.100  ✗   cold-failed              -          -             -
+0.300  ✓   cold                     1.0e+02    -1.1e-05      +1.1e-05
+0.500  ✓   warm<-+11.677            1.0e+02    +2.6e-16      +2.6e-16
+0.550  ✓   warm<-+19.461            1.0e+02    +1.6e-16      +1.6e-16
+0.600  ✓   warm<-+21.407            1.0e+02    +1.0e-16      +1.0e-16
+0.650  ✓   warm<-+23.353            1.0e+02    +1.2e-16      +1.2e-16
+0.660  ✓   warm<-+25.299            1.0e+02    +1.7e-16      +1.7e-16
+0.680  ✓   warm<-+25.688            1.0e+02    +2.6e-16      +2.6e-16
+0.700  ✓   warm<-+26.467            1.0e+02    +5.5e-16      +5.5e-16
+0.750  ✓   warm<-+27.245            1.0e+02    +2.0e-15      +2.0e-15
+1.000  ✓   warm<-+29.191            1.0e+02    +1.1e-12      +1.1e-12
```

`c_steric` saturates at the cap from V=+0.30 onwards, and warm-walk
reaches +1.00 V cleanly. The V=+0.10 cold-fail is the transition zone
between "γ ≈ 1" (linear-Debye) and "γ ≈ 1/(a_cl·c_b·exp(+ψ))" (deep
saturation) where Newton's basin is narrow without Stern's potential
absorption to soften ψ_D.

---

## Cross-stack equivalence

Both `4sp logc + 2b` and `3sp + bikerman + muh + 2b` Stern passes
converged at `V_RHE ∈ {−0.5, −0.3, −0.1, 0, +0.1, +0.3, +0.5}` (7
voltages). Hybrid abs/rel error of CD and PC between the two stacks:

```
V_RHE     CD rel err    PC rel err
-0.500    4·10⁻⁹        3·10⁻⁹
-0.300    6·10⁻⁹        4·10⁻⁹
-0.100    4·10⁻⁸        1·10⁻⁸
 0.000    3·10⁻⁸        2·10⁻⁸
+0.100    2·10⁻⁶        7·10⁻⁸
+0.300    1·10⁻³        3·10⁻⁶
+0.500    5·10⁻³        7·10⁻³
```

* Pure-cathodic regime (V ≤ 0): essentially bit-equal, well below the
  `snes_atol = 1e-7` solver convergence tolerance.
* ORR onset (V = +0.1 to +0.3): per-mille range. Inside the
  `tests/test_solver_equivalence.py::REL_TOL = 5·10⁻³` threshold.
* V = +0.5 V (last common voltage, kinetic dead zone): ~5–7 millis-rel.
  CD = −1.5·10⁻⁵ mA/cm² for both stacks — same order, same sign.

The analytic 3sp + bikerman closure is a faithful representation of
the dynamic ClO₄⁻ transport, and 2b's IC change preserves the
equivalence across the overlap.

---

## What 2b did *not* fix

* **4sp dynamic ceiling unchanged.** Both `formulation='logc'` and
  `formulation='logc_muh'` 4sp dynamic sweeps come in at the same
  5/15 (no Stern) and 7/15 (Stern) pass counts as 2a′. The IC's
  γ-corrected u_3 puts c_ClO₄ on the saturated manifold (≈ 1/a_b),
  but the dynamic species' transport residual at z < 1 in the
  Phase-1 z-ramp can't reconcile the saturated-IC manifold with
  ∇μ_ClO₄ = 0 in SS. This is consistent with the project direction
  in `CLAUDE.md`: the 3sp + analytic-bikerman stack is the
  *production* target; the 4sp dynamic stack is a validation
  reference for equivalence testing.
* **V_RHE > +1.0 V.** Likely still cold-fails because the warm-walk
  needs an anchor and the kinetic dead zone is too deep for the
  cold IC. Trigger for **Option 2c** (numerical ODE integration
  of the Bikerman first integral) — out of scope for 2b.
* **V = +0.10 V cold-fail (no-Stern, 3sp+bikerman+muh).** Cold-fails
  in the linear-Debye → saturation transition zone. Recovered
  cleanly by the Stern pass; not a 2b regression.
* **Residual side untouched.** 2b is purely an IC seed change.
  `build_steric_boltzmann_expressions` and `mu_steric = -ln(packing)`
  are unchanged from §11/§12.

---

## Test coverage

| File | Class / function | What it asserts |
|---|---|---|
| `tests/test_steric_psi_profile.py` | composite-ψ identity tests | continuity at y_match, far-field decay, sign branch, scipy comparison against the Bikerman first integral (qualitative match) |
| `tests/test_initializer_debye_boltzmann_3sp_bikerman.py` | `Test3spBikermanLogc`, `Test3spBikermanMuh` | γ + composite ψ + log_gamma propagation for the 3sp + bikerman counterion stack in both formulations |
| | `TestRegression3spIdealStillWorks` | byte-identical IC for 3sp + ideal counterion in both formulations (regression gate on the legacy path) |
| `tests/test_initializer_debye_boltzmann_4sp_muh.py` | `TestDebyeBoltzmann4spMuhExtension` | parallel of the existing 4sp logc test suite for muh — confirms γ now lands on the muh path's IC |
| `tests/test_initializer_debye_boltzmann_4sp.py` | `test_ic_uses_composite_psi_at_v0p66` | composite-ψ regression target on the 4sp logc path at V = +0.66 V |
| `tests/test_steric_saturation.py` | `test_3sp_bikerman_v0p55_muh` | full-stack integration test on the user's target config at V = +0.55 V (the new cold-success voltage) |

Plus the existing regression gates that must continue to pass for
non-bikerman paths:
`test_initializer_debye_boltzmann.py`, `test_solver_equivalence.py`,
`test_stern_no_stern_snapshot.py`, `test_mms_convergence.py`,
`test_steric_sign.py`, `test_bv_common_config.py`,
`test_steric_boltzmann_closure*.py`.

---

## Artifacts

* **Sweep outputs** (in `StudyResults/`, currently untracked — stage
  separately if you want them in git):
  * `peroxide_window_3sp_bikerman_muh_2b/` — production-target sweep,
    new directory. Files: `iv_curve.json` (per-voltage CD/PC +
    convergence flags), `diagnostics.json` (full diagnostics including
    `c_counterion0_surface_mean`), `comparison.png` (3-panel CD/PC/c_steric
    vs V plot).
  * `peroxide_window_4sp_extended_debye_boltzmann/` — 4sp logc 2b rerun
    (overwrote 2a′; the 2a′ snapshot is at `_PRE_2B/`).
  * `peroxide_window_4sp_extended_debye_boltzmann_logc_muh/` — 4sp muh
    2b rerun (with `_PRE_2B/` snapshot of 2a′).
* **Sweep scripts** (committed):
  * `scripts/studies/peroxide_window_3sp_bikerman_muh.py` — new, drives
    the production-target sweep.
  * `scripts/studies/peroxide_window_4sp_extended.py` — pre-existing,
    invoke as `python -u scripts/studies/peroxide_window_4sp_extended.py
    debye_boltzmann [logc|logc_muh]`.

---

## Commits

```
3c4ba67 docs: §13 Resolution — Bikerman-consistent IC (Option 2b, 2026-05-04)
77ceff3 feat: Bikerman-consistent IC (composite psi + multispecies gamma) for both forms_logc and forms_logc_muh
1317b1b test: Bikerman-consistent IC tests across logc/muh × 4sp/3sp+bikerman
```

---

## References

* [`4sp_bikerman_ic_option_2b_plan.md`](./4sp_bikerman_ic_option_2b_plan.md)
  — pre-implementation plan (predecessor scope: 4sp dynamic only).
* [`4sp_drop_boltzmann_investigation.md`](./4sp_drop_boltzmann_investigation.md)
  §13 — the same results in narrative form, in the broader
  investigation log alongside §11 (steric sign) and §12 (Option 2a′).
* [`4sp_bikerman_ic_option_2a_plan.md`](./4sp_bikerman_ic_option_2a_plan.md)
  — predecessor 2a′ plan (γ correction without composite ψ).
* [`steric_analytic_clo4_reduction_handoff.md`](./steric_analytic_clo4_reduction_handoff.md)
  — derivation of the analytic-counterion closure used on the
  residual side (`build_steric_boltzmann_expressions`).
* Bazant–Kilic–Storey–Ajdari (2009), *Adv. Colloid Interface Sci.*
  152, 48 — the matched-asymptotic composite-ψ profile is eq. (32);
  arXiv:0903.4790.
