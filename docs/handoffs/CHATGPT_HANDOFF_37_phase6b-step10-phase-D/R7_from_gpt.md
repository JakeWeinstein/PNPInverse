One non-blocking implementation correction: `CationHydrolysisBundle` has `cation_params`, not `params`. So this part:

```python
if hasattr(bundle, "params") and isinstance(bundle.params, dict):
    bundle.params["beta_offset_pm2"] = float(val)
```

will no-op. If you want the same metadata mirror pattern as `set_reaction_r_H_El_pm_model`, update `bundle.cation_params` via `object.__setattr__` with a copied dict. The ctx metadata mirror and live `beta_offset_pm2_func` are the load-bearing pieces, so I would not block Phase D on this.

Everything else from v6 is now addressed: topology matches A.2, Stern and ablation baselines are separate, σ-grid construction is ordered correctly, and the acceptance gate remains locked.

VERDICT: APPROVED