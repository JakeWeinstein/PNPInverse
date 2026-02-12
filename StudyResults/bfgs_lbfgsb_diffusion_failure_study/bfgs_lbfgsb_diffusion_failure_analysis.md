# BFGS vs L-BFGS-B Diffusion Failure Study

## Study Design
- Problem: `infer_D` only
- Methods: `BFGS`, `L-BFGS-B`
- Noise levels: `[0.0, 0.005, 0.02]`
- Seed for no-noise (`noise_std=0`): `20270400`
- Seeds for noisy runs (`noise_std>0`): `[20270401, 20270402, 20270403, 20270404, 20270405, 20270406, 20270407, 20270408, 20270409, 20270410]`
- Cases per run: `d_true in {[1,3],[0.5,2]}`, `d_guess in {[0.3,0.3],[10,10]}`

Total runs: **168**
Failed runs: **8**

## Summary by Method and Noise

| method | noise_std | success_rate | n_success/n_runs | median_time_s | median_peak_rss_mib | median_rel_error | median_objective |
|---|---:|---:|---:|---:|---:|---:|---:|
| BFGS | 0 | 1 | 4/4 | 10.59 | 338 | 6.215e-05 | 7.972e-14 |
| BFGS | 0.005 | 1 | 40/40 | 10.22 | 333.5 | 0.02416 | 1.276e-05 |
| BFGS | 0.02 | 1 | 40/40 | 9.911 | 334.9 | 0.09253 | 0.0002042 |
| L-BFGS-B | 0 | 0.75 | 3/4 | 5.057 | 316 | 0.0008752 | 2.801e-11 |
| L-BFGS-B | 0.005 | 0.875 | 35/40 | 5.105 | 320.6 | 0.02412 | 1.278e-05 |
| L-BFGS-B | 0.02 | 0.95 | 38/40 | 5.019 | 320.6 | 0.09403 | 0.0002039 |

## Across-Seed Stability for Noisy Cases (`noise_std > 0`)

| method | failures/runs | failure_rate | seeds_with_failures |
|---|---:|---:|---|
| BFGS | 0/80 | 0 | none |
| L-BFGS-B | 7/80 | 0.0875 | [20270401, 20270402, 20270404, 20270406, 20270407, 20270409, 20270410] |

## Failure Cases and Logged D Values

| method | noise_std | seed | d_true | d_guess | min_logged_d_pair | last_logged_d_pair | min_logged_d_component | reason |
|---|---:|---:|---|---|---|---|---:|---|
| L-BFGS-B | 0 | 20270400 | [1.0, 3.0] | [10.0, 10.0] | [0.23360679, 0.36898834] | [1.89295046, 2.31792262] | 0.2336 | DIVERGED_LINE_SEARCH |
| L-BFGS-B | 0.005 | 20270401 | [1.0, 3.0] | [10.0, 10.0] | [0.2348071, 0.36684491] | [1.90479801, 2.32004539] | 0.2348 | DIVERGED_LINE_SEARCH |
| L-BFGS-B | 0.005 | 20270402 | [1.0, 3.0] | [10.0, 10.0] | [0.03229418, 1.01357971] | [1.42543256, 2.19932283] | 0.03229 | DIVERGED_LINE_SEARCH |
| L-BFGS-B | 0.005 | 20270404 | [1.0, 3.0] | [10.0, 10.0] | [0.23570039, 0.36526913] | [1.91615451, 2.32437545] | 0.2357 | DIVERGED_LINE_SEARCH |
| L-BFGS-B | 0.005 | 20270406 | [1.0, 3.0] | [10.0, 10.0] | [0.04335572, 1.14466137] | [1.41816252, 2.20680239] | 0.04336 | DIVERGED_LINE_SEARCH |
| L-BFGS-B | 0.005 | 20270409 | [1.0, 3.0] | [10.0, 10.0] | [0.04891558, 1.19075142] | [1.42703783, 2.2072424] | 0.04892 | DIVERGED_LINE_SEARCH |
| L-BFGS-B | 0.02 | 20270407 | [1.0, 3.0] | [10.0, 10.0] | [0.23882922, 0.35987668] | [1.91540161, 2.29646049] | 0.2388 | DIVERGED_LINE_SEARCH |
| L-BFGS-B | 0.02 | 20270410 | [1.0, 3.0] | [10.0, 10.0] | [0.22627741, 0.38276322] | [1.88307652, 2.37398539] | 0.2263 | DIVERGED_LINE_SEARCH |

## What Failed D Values Have in Common

- Failures by method: `{'L-BFGS-B': 8}`
- Failures by noise: `{0.0: 1, 0.005: 5, 0.02: 2}`
- Failures by initial guess: `{(10.0, 10.0): 8}`
- Failures by true D: `{(1.0, 3.0): 8}`
- Logged minimum D component in failed runs: min=0.03229, median=0.2299, max=0.2388
- Logged minimum D component in successful runs: min=1.113e-19, median=0.3, max=0.3936

### Method-Level Comparison

| method | median min-D (fail) | median min-D (success) | fail frac(min-D<0.1) | success frac(min-D<0.1) |
|---|---:|---:|---:|---:|
| BFGS | nan | 0.3 | nan | 0.1905 |
| L-BFGS-B | 0.2299 | 0.3 | 0.375 | 0.3684 |

### Interpretation

- Forward failures are consistent with trial iterates driving one or both diffusion coefficients to very small values, which can increase stiffness and hurt nonlinear line-search robustness.
- If failures cluster under high initial guesses (e.g. `[10, 10]`), that suggests aggressive early steps in log-D space are part of the mechanism.
- The comparison against successful runs shows whether failed trajectories visit a low-D region that successful trajectories usually avoid.

