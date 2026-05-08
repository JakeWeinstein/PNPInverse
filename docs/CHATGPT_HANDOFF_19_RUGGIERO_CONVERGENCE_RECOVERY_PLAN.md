# CHATGPT Handoff 19 - Ruggiero Parallel-Topology Convergence Recovery Plan

Date: 2026-05-07
Status: Plan to hand to Claude Code after reading
`docs/ruggiero_realignment_plan.md` and checking the current M3a.2 code.

## Executive Position

The failed M3a.2 diagnostic does not yet prove the parallel 2e/4e model needs
the full M3a.3 Picard rewrite before anything can converge. The current
diagnostic setup has two avoidable convergence traps:

1. It falls back from the matched `debye_boltzmann` Picard IC to `linear_phi`.
   That leaves O2 and H+ essentially bulk-valued at the cathode while the BV
   factors are enormous.
2. The pure-channel passes set one `k0` exactly to zero while `log_rate=True`,
   but the forms build `ln(k0_j)` unconditionally. That can poison the residual
   and Jacobian before Newton even has a meaningful problem to solve.

Do not jump straight to the full generic Picard rewrite. First run the
following convergence rescue ladder. It is designed to answer one question at a
time: zero-k0 bug, linear-phi weakness, topology change, and then true 4e
stiffness.

## Key Diagnosis From The Plan

`docs/ruggiero_realignment_plan.md` says the parallel stack fails even at
`z=0`. That is important. If failure happens at `z=0`, the first-order problem
is not high-ionic-strength Poisson, Bikerman packing, multi-ion sulfate, or
Debye resolution. It is the reaction/diffusion boundary problem plus the IC.

That should keep the debug scope narrow:

- no C_O2 migration during this rescue;
- no multi-ion electrolyte;
- no sulfate;
- no OH-;
- no L_eff retune;
- no inverse/adjoint work.

## Stage 0 - Freeze The Baseline

Before changing convergence code, preserve these controls:

1. Legacy sequential, `debye_boltzmann`, `clip=50`, `V_RHE=0.0`:
   expected cold convergence and `cd ~= -0.175 mA/cm2`.
2. Legacy sequential, `debye_boltzmann`, `clip=100`, `V_RHE=0.0`:
   may cold-fail, but should be recoverable by warm walk.
3. Legacy sequential, `linear_phi`, `clip=50`, `V_RHE=0.0`:
   run this explicitly. If it fails, then the M3a.2 failure is partly just
   "linear_phi cannot initialize this BV problem," independent of parallel
   topology.

Record these in a new study output:

```text
StudyResults/parallel_2e_4e_convergence_rescue/
```

This makes the later claims falsifiable.

## Stage 1 - Make Disabled Reactions Safe In Log-Rate Mode

### Problem

`scripts/studies/peroxide_window_3sp_parallel_2e_4e.py` uses:

```python
("pure_2e_k04e_zero", 1.0, 0.0)
("pure_4e_k02e_zero", 0.0, 1.0)
```

but both `forms_logc.py` and `forms_logc_muh.py` build:

```python
fd.ln(k0_j)
```

inside the log-rate branch. If `k0_j == 0`, the form contains `ln(0)`.
That is not a legitimate disabled reaction. It is a singular expression.

### Code Change

In both:

- `Forward/bv_solver/forms_logc.py`
- `Forward/bv_solver/forms_logc_muh.py`

inside the `for j, rxn in enumerate(rxns_scaled):` loop, add an early disabled
reaction guard before any `fd.ln(k0_j)` is constructed.

Suggested behavior:

```python
k0_model_j = float(rxn["k0_model"])
k0_j = fd.Function(R_space, name=f"bv_k0_rxn{j}")
k0_j.assign(k0_model_j)
bv_k0_funcs.append(k0_j)

alpha_j = fd.Function(R_space, name=f"bv_alpha_rxn{j}")
alpha_j.assign(float(rxn["alpha"]))
bv_alpha_funcs.append(alpha_j)

if k0_model_j <= 0.0 or bool(rxn.get("enabled", True)) is False:
    R_j = fd.Constant(0.0)
    bv_rate_exprs.append(R_j)
    continue
```

This preserves reaction indexing and `bv_k0_funcs` length, but skips the
singular log-rate expression and stoichiometric residual contribution.

### Tests

Add unit or lightweight form-build tests:

- two-reaction log-rate config with second `k0=0` builds without NaN/exception;
- `reaction_index=1` observable assembles to zero for disabled reaction;
- legacy nonzero two-reaction config is unchanged.

Do this before any convergence interpretation. If exact-zero `k0` was poisoning
the pure-channel probes, this fix should immediately change the failure mode.

## Stage 2 - Replace "Pure Channel With Zero k0" By True Reduced Reaction Lists

Even after Stage 1, the cleanest pure-channel test is not "two reactions with
one zero." It is a reaction list containing only the active reaction.

Create a new probe script rather than overloading the existing one:

```text
scripts/studies/parallel_2e_4e_convergence_rescue.py
```

Run these cases at first:

```text
A. legacy sequential + debye_boltzmann + clip=50
B. legacy sequential + linear_phi      + clip=50
C. R2e-only          + linear_phi      + clip=50
D. R4e-only          + linear_phi      + clip=50
E. R2e-only          + linear_phi      + clip=100
```

Use a tiny voltage list first:

```text
V_RHE = [+0.45, +0.30, +0.00]
```

Do not start at `-0.40`. Start near the weak-reaction side and walk cathodic
only after an anchor exists.

### Interpretation

- If B fails and A passes: `linear_phi` is confirmed inadequate even for the
  legacy problem. Do not blame parallel topology yet.
- If C fails but B passes: the R2e-only topology or single-reaction setup has a
  bug.
- If C passes and D fails: R4e stiffness is the next target.
- If C and D pass separately but mixed fails: the issue is coupled flux
  competition, not basic form construction.

## Stage 3 - Cross-Topology Warm Start From A Fresh Legacy Solve

The plan mentions "load Run C state," but Run C only saved JSON diagnostics and
plots, not the Firedrake `U` state. So do not try to load it from disk.

Instead, in the rescue script:

1. Build and solve the legacy sequential stack at one voltage, preferably
   `V_RHE=0.0` or `+0.30`, using the known-good `debye_boltzmann` IC.
2. Snapshot the converged `U` arrays in memory.
3. Build the parallel context on the same mesh and same species/formulation.
4. Copy the legacy `U` data into the parallel context:

```python
for src, dst in zip(ctx_legacy["U"].dat, ctx_parallel["U"].dat):
    dst.data[:] = src.data_ro
ctx_parallel["U_prev"].assign(ctx_parallel["U"])
```

5. Solve the parallel residual at the same voltage.

This tests whether the failure is just bad IC distance. The old profiles are
not physically perfect for the parallel topology, but they are much closer than
`linear_phi`: O2/H+/phi already contain a cathodic reaction boundary layer.

### Cases To Try

Run in this order:

```text
1. R2e-only from legacy warm state
2. R2e + tiny R4e from legacy warm state
3. R2e + ramped R4e from legacy warm state
```

Do not use exact-zero `k0` after Stage 1 except as a disabled-reaction test.

## Stage 4 - Reaction-Strength Continuation For R4e

R4e is numerically severe with the current placeholder:

```text
E_eq = 1.23 V
n_e = 4
alpha = 0.5
k0_4e = k0_2e placeholder
```

At `V_RHE=0.0`, the cathodic exponent is roughly `exp(96)` before concentration
depletion. Equal `k0_4e` is not a gentle diagnostic.

Use the existing mutable `ctx["bv_k0_funcs"]` instead of rebuilding forms for
every continuation step. Build the mixed parallel form with a positive but tiny
R4e `k0`, solve, then assign larger `k0` values.

Suggested ladder for `k0_4e / K0_HAT_R4E`:

```text
1e-60, 1e-54, 1e-48, 1e-42, 1e-36, 1e-30,
1e-24, 1e-18, 1e-12, 1e-8, 1e-6, 1e-4, 1e-2, 1
```

Use larger jumps only after the first stable region is found. At each step:

```python
ctx["bv_k0_funcs"][1].assign(K0_HAT_R4E * scale)
run_ss(max_steps=...)
```

Acceptance for the first rescue is modest:

- reach at least `scale=1e-12` at one voltage;
- then tighten the ladder around the failure point;
- only demand `scale=1` after a usable continuation path is identified.

If this works, use the converged mixed state as the anchor for voltage warm-walk.

## Stage 5 - Voltage Continuation From The Weak-Reaction Side

Once any parallel state converges at one voltage, do not run the full page-15
grid cold. Walk from the easiest voltage outward.

Try anchor order:

```text
+0.45 -> +0.30 -> +0.20 -> +0.10 -> 0.00 -> -0.20 -> -0.32 -> -0.40
```

Use the same warm-start copying strategy between voltages or reuse
`solve_grid_per_voltage_cold_with_warm_fallback` only after the first full-z
anchor is found.

If `clip=100` fails early, get the path at `clip=50` first. Then rerun the same
anchor with `clip=100`. The final quantitative run still needs `clip=100`, but
the convergence debug does not need to start there.

## Stage 6 - Only If Needed: Minimal Parallel Picard, Not Full Multi-Ion Work

If Stages 1-5 fail, then M3a.3 is genuinely needed. But keep the first Picard
rewrite narrowly scoped to the 3-species parallel 2e/4e problem. Do not mix in
multi-ion or sulfate.

For fixed `H_o`, `psi_D`, `gamma_s`, and BV coefficients, the parallel two-rate
surface balance is still a small linear system, not a full generic nonlinear
solve.

Use:

```text
R2 = A2 * O_s - B2 * P_s
R4 = A4 * O_s

O_s = O_b - (R2 + R4) / D_O
P_s = P_b + R2 / D_P
H_o = H_b - (2*R2 + 4*R4) / D_H_effective
```

For fixed coefficients, substitute `O_s` and `P_s` into `R2`, `R4` and solve
the resulting 2x2 linear system. Then update `H_o`, Stern split, `psi_D`, and
gamma, with relaxation.

This is much smaller than the eventual arbitrary-N-reaction Picard and should
be enough for M3a.2/M3a.3.

### Acceptance

The minimal Picard is successful if:

- R2e-only converges at `V_RHE=0.0`;
- mixed R2e/R4e converges at one voltage with a small R4e scale;
- the Picard-seeded state reduces SNES iterations or changes failure from
  immediate cold-fail to later continuation failure.

Only after that should Claude generalize the Picard to arbitrary reaction
lists.

## Implementation Order

Do this in order:

1. Add disabled-reaction guard for log-rate forms.
2. Add tests for disabled `k0=0` reactions.
3. Write `parallel_2e_4e_convergence_rescue.py`.
4. Run Stage 0/2 controls on `V_RHE={+0.45,+0.30,0.0}`.
5. Implement fresh legacy-solve to parallel warm-start in the rescue script.
6. Add R4e k0 continuation using `ctx["bv_k0_funcs"][1].assign(...)`.
7. Try voltage continuation only after one mixed anchor converges.
8. If all fail, implement the minimal 2x2 parallel Picard.

## What Not To Do Yet

- Do not migrate `C_O2 = 0.5 -> 1.2` during this rescue. That changes scaling
  and invalidates the convergence diagnosis.
- Do not start M3b multi-ion work.
- Do not interpret exact-zero `k0` log-rate failures as physics.
- Do not require full `clip=100` success before identifying an anchor path.
- Do not start with `V_RHE=-0.40`; start near the weak-reaction side.
- Do not call a `linear_phi` failure a parallel-topology failure until legacy
  sequential + `linear_phi` has been tested.

## Bottom Line For Claude Code

The fastest plausible convergence path is:

```text
fix ln(0) disabled-reaction trap
  -> prove R2e-only can converge
  -> warm-start parallel from a freshly solved legacy state
  -> ramp R4e k0 from tiny positive values
  -> warm-walk voltage grid
  -> only then write the parallel Picard if needed
```

This sequence should take hours to a day to falsify. The full M3a.3 Picard
rewrite should be the fallback, not the first move.
