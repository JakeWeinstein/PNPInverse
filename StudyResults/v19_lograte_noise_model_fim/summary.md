# V19 — noise-model FIM audit

Re-whitening of existing S_cd, S_pc sensitivities at TRUE params under
multiple noise models.  Forward data: `StudyResults/v19_bv_lograte_audit/extended_v_to_60/`
(cap=50, log-rate ON, V_GRID=[-0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).

|cd| range: [1.824e-09, 1.780e-01]  (mA/cm^2 in I_SCALE units)

|pc| range: [2.704e-09, 1.544e-05]

## Results

| noise_model | sv_min | cond(F) | ridge_cos | weak_eigvec |
|---|---:|---:|---:|---|
| A_global_2pct_max | 2.510e+00 | 1.79e+07 | 0.031 | [+0.999,+0.031,-0.018,-0.001] |
| B_local_2pct | 6.690e+01 | 7.03e+05 | 0.056 | [+0.998,+0.056,-0.021,-0.003] |
| C_local_2pct_floor_1e-06 | 2.874e+00 | 1.32e+06 | 0.020 | [+1.000,+0.020,-0.022,-0.003] |
| C_local_2pct_floor_1e-07 | 5.218e+00 | 2.93e+07 | 0.011 | [+1.000,+0.011,-0.018,-0.001] |
| C_local_2pct_floor_1e-08 | 5.355e+00 | 1.07e+08 | 0.011 | [+1.000,+0.011,-0.018,-0.001] |
| C_local_2pct_floor_1e-09 | 6.060e+00 | 8.56e+07 | 0.011 | [+1.000,+0.011,-0.018,-0.001] |

## Interpretation rule (per GPT plan)

Pass: under realistic σ_abs the FIM still has cond ≤ ~1e8-1e9,
ridge_cos not near 1, weak eigvec not pure log_k0_2.

Note: σ_abs values are in the same units as cd, pc (mA/cm^2 in I_SCALE
convention).  At V=+0.60 V, |cd| ≈ 1.8e-9 — sub-1e-6 floors essentially
zero out the high-V signal and we expect the FIM to revert toward the
baseline single-experiment ridge.
