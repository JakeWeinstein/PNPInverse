# V21 — LM on G0 with 4 inits (clean data)

Per Path A in `docs/PNP Positive Voltage Extension Handoff.md` and Route 1(a) in
`docs/CHATGPT_HANDOFF_9_ADJOINT_RESOLVED_GRID_DONE.md`. Tests whether
Levenberg-Marquardt's damped Gauss-Newton step strategy escapes the Tafel-ridge
basins where TRF G0 stalled.

Setup: bv_log_rate=True, observables=CD+PC, regularization=none, init_cache=cold-solve at INIT,
σ=2%×max|target|, V_GRID=G0=[-0.10, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60].

Note: scipy.optimize.least_squares method='lm' has no bounds. An unphysical-parameter guard was
added to `compute_residuals_and_jacobian` (`scripts/studies/v18_logc_lsq_inverse.py`): when
α exits (0, 1] or log_k0 wanders >5 from TRUE, the function returns huge residuals (1e3) and
a back-pointing Jacobian instead of letting the forward solver raise.

## Results

| init | evals | wall | guards | cost | k0_1 err | k0_2 err | α_1 err | α_2 err | <5% count | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| plus20 | 27 | 3.5min | 3 | 1.99e+02 | -17.25% | -43.25% | +28.39% | +18.32% | 0 | xtol |
| minus20 | 24 | 2.2min | 5 | 5.34 | +709.77% | +12252.66% | -5.99% | -10.00% | 0 | xtol |
| k0high_alow | 22 | 1.7min | 6 | 8.06 | +1388.83% | +14481.96% | -7.80% | -10.44% | 0 | xtol |
| k0low_ahigh | 24 | 3.0min | 3 | 1.87e+02 | -63.75% | -61.99% | +29.42% | +18.27% | 0 | xtol |

## Comparison vs TRF G0 baseline (`StudyResults/v20_best_grid_trf_clean/`)

| init | TRF cost | LM cost | TRF <5% count | LM <5% count | verdict |
|---|---:|---:|---:|---:|---|
| plus20 | 141 | 199 | 1 | 0 | LM worse |
| minus20 | **0.011** | **5.34** | **3** | **0** | LM catastrophically worse |
| k0high_alow | 0.40 | 8.06 | 2 | 0 | LM catastrophically worse |
| k0low_ahigh | 143 | 187 | 0 | 0 | LM worse |

**Total <5% counts: TRF=6, LM=0.**

## Headline

LM **is decisively worse than TRF on every init**. The handoff #9 question
"Path A vs Path C first?" is answered: Path A failed.

Mechanism: along the Tafel ridge `k0 · exp(-α·n_e·η/V_T) = const`, log_k0 can
change a lot for a tiny α nudge while the rate (and so cd, pc) stays nearly
constant. TRF's bounds on log_k0 (±2 from TRUE) kept it parked on the ridge
near TRUE. LM has no bounds, and its damped Gauss-Newton step slid far along
the ridge:
- minus20 → k0_2 = 122× TRUE (log_k0_2 ≈ -3.04 vs TRUE -9.85)
- k0high_alow → k0_2 = 145× TRUE (log_k0_2 ≈ -2.87)

α stays in physical range across all inits (-10% to +30%) — the unphysical-
parameter guard prevented α excursions. log_k0 has no analogous "physically
required" bound, so LM rode the ridge.

All 4 inits terminated with `xtol` (status=3) at non-zero residual: LM is
**not stuck mid-step** — it converged to a stationary point of the cost on the
Tafel ridge. The line profiles from recovered→TRUE show TRUE has lower J than
the LM endpoint in every case: the basin TRUE lives in IS reachable, LM just
walked away from it.

## Reconciling LM and TRF

Both stall at xtol with non-zero residual. The difference is:
- **TRF on G0**: bounded; stays near TRUE; minus20 lands in TRUE's basin (3/4
  params <5%); other inits stall in nearby (k0, α) ridge points.
- **LM on G0**: unbounded; slides down ridges; ends 100-15000× TRUE in k0;
  α robust because the guard prevents excursion.

This reframes the inverse-problem bottleneck: **TRF's bounds were the implicit
prior keeping it near TRUE.** Without them, the data alone does not localize
log_k0. Adding an explicit log_k0 prior (Tikhonov) is now the obvious next step.

## Implications

- Route 1(a) [LM] from handoff #9: **dead**. LM's no-bounds nature is fatal
  for this Tafel-ridge problem.
- Route 1(b) [FIM-eigenbasis x_scale TRF]: still untested. scipy x_scale is
  diagonal-only; would need explicit reparameterization. Speculative; not
  recommended given how decisively LM failed.
- Route 1(c) [restart-with-perturbation]: still untested. Cheap to try, but
  doesn't address the underlying ridge geometry — the perturbed restart would
  likely also stall on the ridge.
- **Route 2 [Tikhonov on log_k0]**: now the obvious next step. The handoff #9
  suggestion of σ_log_k0 = log(3) (factor-of-3 prior) is the right starting
  point. With a weak prior anchoring log_k0 near TRUE, all 4 inits should
  converge to TRUE consistently. Defensible Bayesian framing.

## Files

- `scripts/studies/v18_logc_lsq_inverse.py` — added unphysical-parameter guard
  in `compute_residuals_and_jacobian` (lines 417-446 after edit).
- `lm_plus20/`, `lm_minus20/`, `lm_k0high_alow/`, `lm_k0low_ahigh/` — per-init
  result.json + history_partial.json + line_profile.json + true_curve.npz +
  targets.npz.
- `_master_run.log` — full sequential run log with all 4 inits + GUARD events.
- `_master_run_aborted.log` — first attempt where plus20 crashed unguarded.
