# Infer BV Exchange Rate Constant (k0) from Synthetic Data

## Goal

Create an adjoint-based inference script that recovers the Butler-Volmer exchange
rate constant `k0` from synthetic PNP-BV solution data, following the same
`InferenceRequest → run_inverse_inference → InferenceResult` pattern used by
`Infer_RobinBC_from_data.py`.

## Why k0?

- **Most useful**: k0 is the key catalyst descriptor experimentalists extract from
  I-V curve fitting. It controls the onset/half-wave potential of the current-voltage
  relationship.
- **Well-conditioned**: k0 enters the BV rate expression **linearly**
  (`R_j = k0_j * [...]`), so the adjoint gradient w.r.t. log(k0) is clean.
- **Natural log-transform**: Use `log_k0` as the optimization variable (like `logD`
  for diffusion), giving unconstrained optimization over a positive parameter.
- **Per-reaction scalar**: One `fd.Function` in R-space per reaction — minimal
  control dimension.

## Blocking Issue

Currently in `build_forms()`, k0 is wrapped as `fd.Constant(float(...))` (lines 608,
646). This kills the Firedrake adjoint tape — `fd.Constant` constructed from a Python
float has no derivative path. We must promote k0 to `fd.Function(R_space)` so
`firedrake.adjoint` can differentiate through it.

This is the **exact same pattern** already used for diffusivity:
```python
# Existing (logD):
m[i] = fd.Function(R_space, name=f"logD{i}")   # ← control variable
m[i].assign(np.log(D_model_vals[i]))
D[i] = fd.exp(m[i])                            # ← used in weak form
```

We replicate it for k0:
```python
# New (log_k0):
log_k0_j = fd.Function(R_space, name=f"log_k0_{j}")
log_k0_j.assign(np.log(k0_model_val))
k0_j = fd.exp(log_k0_j)                        # ← used in weak form
```

## Implementation Steps

### Step 1: Modify `Forward/bv_solver.py` — promote k0 to fd.Function

In `build_forms()`, change both BV paths to use `fd.Function` for k0:

**Multi-reaction path** (lines 607-608):
```python
# BEFORE:
k0_j = fd.Constant(float(rxn["k0_model"]))

# AFTER:
log_k0_j = fd.Function(R_space, name=f"log_k0_{j}")
log_k0_j.assign(float(np.log(max(rxn["k0_model"], 1e-30))))
k0_j = fd.exp(log_k0_j)
```

**Legacy per-species path** (lines 645-646):
```python
# BEFORE:
k0_i = fd.Constant(float(scaling["bv_k0_model_vals"][i]))

# AFTER:
log_k0_i = fd.Function(R_space, name=f"log_k0_{i}")
log_k0_i.assign(float(np.log(max(scaling["bv_k0_model_vals"][i], 1e-30))))
k0_i = fd.exp(log_k0_i)
```

Collect `log_k0_funcs` list and store in `ctx`:
```python
ctx.update({
    ...
    "log_k0_funcs": log_k0_funcs,   # NEW — list of fd.Function(R_space)
    ...
})
```

**Risk**: Using `fd.Function` instead of `fd.Constant` for k0 may slightly change
Newton convergence behavior (UFL expression tree is different). Non-inference scripts
that don't touch k0 will see identical numerics since `fd.exp(log(k0))` == `k0`.

### Step 2: Add `bv_k0` target to `Inverse/parameter_targets.py`

Add `_build_bv_k0_target()` following the robin_kappa pattern:

```python
def _build_bv_k0_target() -> ParameterTarget:
    def apply_value_inplace(solver_params, value):
        # For multi-reaction config: set k0 in each reaction
        # For legacy config: set k0 in bv_bc["k0"] list
        params = solver_params[10]
        bv_cfg = params.get("bv_bc", {})
        reactions = bv_cfg.get("reactions")
        if reactions is not None:
            vals = ensure_sequence(value)
            for j, rxn in enumerate(reactions):
                rxn["k0"] = float(vals[j]) if j < len(vals) else float(vals[-1])
        else:
            n_species = int(solver_params[0])
            bv_cfg["k0"] = as_species_list(value, n_species, "bv_k0")

    def controls_from_context(ctx):
        return ctx["log_k0_funcs"]

    def estimate_from_controls(controls):
        return [float(np.exp(ctrl.dat.data[0])) for ctrl in ensure_sequence(controls)]

    # No bounds needed — log-space is unconstrained

    def eval_cb_post_factory(_ctx):
        def eval_cb_post(j, m):
            m_list = ensure_sequence(m)
            log_k0 = [float(v.dat.data[0]) for v in m_list]
            k0 = [float(np.exp(v)) for v in log_k0]
            log_str = ", ".join(f"{v:>12.6f}" for v in log_k0)
            k0_str = ", ".join(f"{v:>12.6e}" for v in k0)
            print(f"[opt] j={float(j):>14.6e}  log_k0=[{log_str}]  k0=[{k0_str}]")
        return eval_cb_post

    return ParameterTarget(
        key="bv_k0",
        description="Infer BV exchange rate constant(s) k0 in log-space.",
        objective_fields=("concentration",),  # BV k0 primarily affects concentration profiles
        apply_value_inplace=apply_value_inplace,
        controls_from_context=controls_from_context,
        estimate_from_controls=estimate_from_controls,
        default_bounds_factory=None,  # log-space is unconstrained
        eval_cb_pre_factory=None,
        eval_cb_post_factory=eval_cb_post_factory,
    )
```

Register in `build_default_target_registry()`:
```python
return {
    "diffusion": ...,
    "dirichlet_phi0": ...,
    "robin_kappa": ...,
    "bv_k0": _build_bv_k0_target(),    # NEW
}
```

### Step 3: Create `scripts/inference/Infer_BV_k0_from_data.py`

New inference script following the `Infer_RobinBC_from_data.py` pattern:

- Uses `ForwardSolverAdapter.from_module_path("Forward.bv_solver", solve_function_name="forsolve_bv")`
- Uses the `"bv_k0"` target
- Configures a simple 2-species neutral BV problem (z=[0,0]) with one reaction
- Sets `true_value` and `initial_guess` for k0 (in the same units as `bv_bc.reactions[j].k0`)
- Uses `build_default_solver_params(...)` with a `bv_bc` config block
- Prints estimated vs true k0

Key design choices:
- Use neutral species (z=0) to avoid Poisson stiffness complications
- Single reaction for simplicity
- Moderate overpotential (η ≈ -2 V_T) so BV has measurable effect
- Short time horizon (dt=0.01, t_end=0.1) matching existing inference scripts

### Step 4: Update `Inverse/__init__.py`

No change needed — the target is registered in `build_default_target_registry()` which
is already exported.

### Step 5: Verification

1. `python -c "from Inverse import build_default_target_registry; print(sorted(build_default_target_registry().keys()))"`
   → should include `"bv_k0"`
2. `python scripts/inference/Infer_BV_k0_from_data.py`
   → should run to completion, printing estimated k0 close to true value
3. `python scripts/verification/test_bv_forward.py --strategy A`
   → regression check that BV solver still passes with fd.Function k0

## Files Changed

| File | Change |
|------|--------|
| `Forward/bv_solver.py` | Promote k0 from `fd.Constant(float)` to `fd.exp(fd.Function(R_space))` in both BV paths; store `log_k0_funcs` in ctx |
| `Inverse/parameter_targets.py` | Add `_build_bv_k0_target()` and register as `"bv_k0"` |
| `scripts/inference/Infer_BV_k0_from_data.py` | New inference script |
