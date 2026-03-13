# Optimization Method Study

Noise std = 0 seeds: [20260211]
Noise std > 0 seeds: [20260211, 20260212, 20260213]

Total runs: 420
Failed runs: 24
Median time/memory/error statistics use converged runs only.

## Summary by Problem / Method / Noise

| problem | method | noise_std | success_rate | n_success/n_runs | median_time_s | median_peak_rss_mib | median_rel_error | median_objective |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| infer_D | BFGS | 0 | 1 | 6/6 | 9.871 | 330.3 | 7.093e-05 | 9.632e-14 |
| infer_D | BFGS | 0.005 | 1 | 18/18 | 9.278 | 326.9 | 0.04067 | 1.191e-05 |
| infer_D | BFGS | 0.02 | 1 | 18/18 | 9.149 | 334.3 | 0.1446 | 0.0001905 |
| infer_D | CG | 0 | 1 | 6/6 | 22.54 | 350 | 7.533e-05 | 9.933e-14 |
| infer_D | CG | 0.005 | 1 | 18/18 | 20.61 | 343.5 | 0.0406 | 1.191e-05 |
| infer_D | CG | 0.02 | 1 | 18/18 | 15.64 | 342.7 | 0.1446 | 0.0001905 |
| infer_D | L-BFGS-B | 0 | 0.8333 | 5/6 | 4.993 | 320.8 | 0.0008752 | 2.801e-11 |
| infer_D | L-BFGS-B | 0.005 | 0.9444 | 17/18 | 4.8 | 318.2 | 0.04059 | 1.191e-05 |
| infer_D | L-BFGS-B | 0.02 | 0.8889 | 16/18 | 4.641 | 316.1 | 0.1457 | 0.0001905 |
| infer_D | Newton-CG | 0 | 0.8333 | 5/6 | 7.389 | 333.8 | 0.325 | 7.259e-07 |
| infer_D | Newton-CG | 0.005 | 0.7778 | 14/18 | 8.545 | 342.4 | 0.3497 | 1.262e-05 |
| infer_D | Newton-CG | 0.02 | 0.7222 | 13/18 | 9.108 | 344 | 0.3525 | 0.0001914 |
| infer_D | SLSQP | 0 | 1 | 6/6 | 3.309 | 307 | 1.941 | 1.536e-05 |
| infer_D | SLSQP | 0.005 | 1 | 18/18 | 3.403 | 304.8 | 1.955 | 2.721e-05 |
| infer_D | SLSQP | 0.02 | 1 | 18/18 | 3.598 | 308.7 | 1.996 | 0.0002063 |
| infer_D | TNC | 0 | 0.8333 | 5/6 | 6.803 | 321.7 | 0.002536 | 2.801e-11 |
| infer_D | TNC | 0.005 | 0.7222 | 13/18 | 5.902 | 324.5 | 0.03685 | 1.191e-05 |
| infer_D | TNC | 0.02 | 0.7778 | 14/18 | 6.069 | 326.1 | 0.143 | 0.0001905 |
| infer_phi0 | BFGS | 0 | 1 | 4/4 | 3.797 | 309.7 | 2.588e-13 | 1.627e-26 |
| infer_phi0 | BFGS | 0.005 | 1 | 12/12 | 3.615 | 305 | 0.0001261 | 6.396e-06 |
| infer_phi0 | BFGS | 0.02 | 1 | 12/12 | 3.638 | 309.7 | 0.0005044 | 0.0001023 |
| infer_phi0 | CG | 0 | 1 | 4/4 | 3.687 | 308.3 | 3.753e-13 | 3.373e-26 |
| infer_phi0 | CG | 0.005 | 1 | 12/12 | 3.58 | 309 | 0.0001261 | 6.396e-06 |
| infer_phi0 | CG | 0.02 | 1 | 12/12 | 3.552 | 308.5 | 0.0005044 | 0.0001023 |
| infer_phi0 | L-BFGS-B | 0 | 1 | 4/4 | 3.528 | 305.9 | 4.091e-12 | 3.124e-23 |
| infer_phi0 | L-BFGS-B | 0.005 | 1 | 12/12 | 3.303 | 306 | 0.0001261 | 6.396e-06 |
| infer_phi0 | L-BFGS-B | 0.02 | 1 | 12/12 | 3.287 | 304.2 | 0.0005044 | 0.0001023 |
| infer_phi0 | Newton-CG | 0 | 1 | 4/4 | 5.922 | 328.8 | 7.601e-13 | 4.515e-25 |
| infer_phi0 | Newton-CG | 0.005 | 1 | 12/12 | 6.116 | 330.3 | 0.0001261 | 6.396e-06 |
| infer_phi0 | Newton-CG | 0.02 | 1 | 12/12 | 6.246 | 331 | 0.0005044 | 0.0001023 |
| infer_phi0 | SLSQP | 0 | 1 | 4/4 | 3.352 | 304.8 | 7.643e-13 | 3.979e-25 |
| infer_phi0 | SLSQP | 0.005 | 1 | 12/12 | 3.113 | 301.8 | 0.0001261 | 6.396e-06 |
| infer_phi0 | SLSQP | 0.02 | 1 | 12/12 | 3.111 | 299.4 | 0.0005044 | 0.0001023 |
| infer_phi0 | TNC | 0 | 1 | 4/4 | 3.292 | 306.9 | 6.426e-13 | 2.227e-25 |
| infer_phi0 | TNC | 0.005 | 1 | 12/12 | 3.924 | 314.1 | 0.0001261 | 6.396e-06 |
| infer_phi0 | TNC | 0.02 | 1 | 12/12 | 3.906 | 310.8 | 0.0005044 | 0.0001023 |

## Failure Log

| problem | method | noise_std | case | reason |
|---|---|---:|---|---|
| infer_D | L-BFGS-B | 0 | d_true=[1.0, 3.0], d_guess=[10.0, 10.0] | ConvergenceError: Nonlinear solve failed to converge after 25 nonlinear iterations.
Reason:
   DIVERGED_LINE_SEARCH |
| infer_D | TNC | 0 | d_true=[0.5, 2.0], d_guess=[0.3, 0.3] | ConvergenceError: Nonlinear solve failed to converge after 66 nonlinear iterations.
Reason:
   DIVERGED_LINE_SEARCH |
| infer_D | Newton-CG | 0 | d_true=[0.5, 2.0], d_guess=[10.0, 10.0] | ConvergenceError: Nonlinear solve failed to converge after 44 nonlinear iterations.
Reason:
   DIVERGED_LINE_SEARCH |
| infer_D | L-BFGS-B | 0.005 | d_true=[1.0, 3.0], d_guess=[10.0, 10.0] | ConvergenceError: Nonlinear solve failed to converge after 29 nonlinear iterations.
Reason:
   DIVERGED_LINE_SEARCH |
| infer_D | TNC | 0.005 | d_true=[1.0, 3.0], d_guess=[0.3, 0.3] | ConvergenceError: Nonlinear solve failed to converge after 46 nonlinear iterations.
Reason:
   DIVERGED_LINE_SEARCH |
| infer_D | TNC | 0.005 | d_true=[0.5, 2.0], d_guess=[0.3, 0.3] | ConvergenceError: Nonlinear solve failed to converge after 58 nonlinear iterations.
Reason:
   DIVERGED_LINE_SEARCH |
| infer_D | Newton-CG | 0.005 | d_true=[0.5, 2.0], d_guess=[10.0, 10.0] | ConvergenceError: Nonlinear solve failed to converge after 18 nonlinear iterations.
Reason:
   DIVERGED_LINE_SEARCH |
| infer_D | TNC | 0.005 | d_true=[1.0, 3.0], d_guess=[0.3, 0.3] | ConvergenceError: Nonlinear solve failed to converge after 61 nonlinear iterations.
Reason:
   DIVERGED_LINE_SEARCH |
| infer_D | TNC | 0.005 | d_true=[0.5, 2.0], d_guess=[0.3, 0.3] | ConvergenceError: Nonlinear solve failed to converge after 30 nonlinear iterations.
Reason:
   DIVERGED_LINE_SEARCH |
| infer_D | Newton-CG | 0.005 | d_true=[0.5, 2.0], d_guess=[10.0, 10.0] | ConvergenceError: Nonlinear solve failed to converge after 15 nonlinear iterations.
Reason:
   DIVERGED_LINE_SEARCH |
| infer_D | TNC | 0.005 | d_true=[0.5, 2.0], d_guess=[0.3, 0.3] | ConvergenceError: Nonlinear solve failed to converge after 80 nonlinear iterations.
Reason:
   DIVERGED_MAX_IT |
| infer_D | Newton-CG | 0.005 | d_true=[0.5, 2.0], d_guess=[10.0, 10.0] | ConvergenceError: Nonlinear solve failed to converge after 18 nonlinear iterations.
Reason:
   DIVERGED_LINE_SEARCH |
| infer_D | TNC | 0.02 | d_true=[0.5, 2.0], d_guess=[0.3, 0.3] | ConvergenceError: Nonlinear solve failed to converge after 26 nonlinear iterations.
Reason:
   DIVERGED_LINE_SEARCH |
| infer_D | L-BFGS-B | 0.02 | d_true=[0.5, 2.0], d_guess=[10.0, 10.0] | ConvergenceError: Nonlinear solve failed to converge after 76 nonlinear iterations.
Reason:
   DIVERGED_LINE_SEARCH |
| infer_D | TNC | 0.02 | d_true=[1.0, 3.0], d_guess=[0.3, 0.3] | ConvergenceError: Nonlinear solve failed to converge after 28 nonlinear iterations.
Reason:
   DIVERGED_LINE_SEARCH |
| infer_D | TNC | 0.02 | d_true=[0.5, 2.0], d_guess=[0.3, 0.3] | ConvergenceError: Nonlinear solve failed to converge after 80 nonlinear iterations.
Reason:
   DIVERGED_MAX_IT |
| infer_D | Newton-CG | 0.02 | d_true=[0.5, 2.0], d_guess=[0.3, 0.3] | ConvergenceError: Nonlinear solve failed to converge after 27 nonlinear iterations.
Reason:
   DIVERGED_LINE_SEARCH |
| infer_D | Newton-CG | 0.02 | d_true=[0.5, 2.0], d_guess=[10.0, 10.0] | SciPyConvergenceError: SciPy minimization failed because: Warning: Desired error not necessarily achieved due to precision loss. |
| infer_D | L-BFGS-B | 0.02 | d_true=[1.0, 3.0], d_guess=[10.0, 10.0] | ConvergenceError: Nonlinear solve failed to converge after 80 nonlinear iterations.
Reason:
   DIVERGED_MAX_IT |
| infer_D | TNC | 0.02 | d_true=[0.5, 2.0], d_guess=[0.3, 0.3] | ConvergenceError: Nonlinear solve failed to converge after 35 nonlinear iterations.
Reason:
   DIVERGED_LINE_SEARCH |
| infer_D | Newton-CG | 0.02 | d_true=[0.5, 2.0], d_guess=[10.0, 10.0] | ConvergenceError: Nonlinear solve failed to converge after 80 nonlinear iterations.
Reason:
   DIVERGED_MAX_IT |
| infer_D | Newton-CG | 0.005 | d_true=[1.0, 1.0], d_guess=[0.3, 0.3] | SciPyConvergenceError: SciPy minimization failed because: Warning: Desired error not necessarily achieved due to precision loss. |
| infer_D | Newton-CG | 0.02 | d_true=[1.0, 1.0], d_guess=[0.3, 0.3] | SciPyConvergenceError: SciPy minimization failed because: Warning: Desired error not necessarily achieved due to precision loss. |
| infer_D | Newton-CG | 0.02 | d_true=[1.0, 1.0], d_guess=[0.3, 0.3] | SciPyConvergenceError: SciPy minimization failed because: Warning: Desired error not necessarily achieved due to precision loss. |
