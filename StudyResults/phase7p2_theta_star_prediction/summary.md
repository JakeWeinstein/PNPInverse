# Stage 2 — pre-registered θ* transferability diagnostic (NO tuning)

θ* = (−3.683, −13.537, 0.550, 0.285) — the slide-15 Cs⁺/pH-4 adjoint
fit — evaluated unchanged at K⁺/pH 6.39, L_eff 15.4 µm, OCP 1.019 V,
on the 30 Stage-0 bin centers (physical iR axis). 30/30 converged.
Locked BEFORE any refit per the session-43 plan.

## Scores (raw χ², fixed vector — the baseline the fit must beat)

- raw χ² = 44,515 over 60 residuals; J = 1483.8/pt
- per-series standardized residual scores: disk 763.9, ring 719.9
  (both series misfit comparably; neither dominates)

## Feature metrics (data axis, recomputed observables)

| Feature | θ* prediction | Data | Δ |
|---|---|---|---|
| disk onset (|j|=0.05) | 0.365 V | 0.518 V | −0.152 V |
| ring onset (j_r=0.01) | 0.323 V | 0.490 V | −0.167 V |
| ring peak position | 0.161 V | 0.338 V | −0.18 V |
| ring peak height | 0.158 | 0.380 mA/cm²_ring | ×0.41 |
| plateau j_disk | −5.28 | −3.50 mA/cm² | +51 % |

## Reading (diagnostic only — see uncertainty framing)

The misfit is coherent, not random: every feature sits ~0.15–0.18 V
too cathodic, the plateau overshoots toward the 4e transport ceiling
(−5.28 vs ceiling −5.71; the data's −3.50 is NOT transport-saturated),
and the ring current is ~2.4× too small. A more anodic onset (larger
k0 and/or different α), a less 4e-dominated partition, and a plateau
below the ceiling are exactly the directions the disk series can now
constrain.

**Uncertainty framing (pre-registered):** θ*'s 4e parameters are
weakly identified on the peroxide-only slide-15 data, and θ* was fit
to a digitized rendered curve on the deck's (sheet-sign) iR axis. A
poor prediction here may reflect θ* uncertainty and axis-convention
mismatch rather than failed transferability of the water-route
kinetics; the Stage-5 re-prediction with θ_K (and the OCP/axis refit
variants) is the controlled comparison.

Files: `prediction.json` (full curves), `feature_metrics.json`,
`prediction_vs_data.png`.
