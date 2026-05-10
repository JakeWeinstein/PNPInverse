# PNP-BV Forward Solver: Continuation Strategy Handoff

This document is a self-contained handoff for evaluating continuation
strategies in the production PNP-BV forward solver after the
formulation rebuild described in
`writeups/WeekOfApr27/PNP Inverse Solver Revised.tex`.  The goal is to
pick the right orchestration to wire the new log-c + analytic-Boltzmann
+ log-rate BV stack into the main pipeline (`Forward.bv_solver` and
`FluxCurve.bv_point_solve`) so it works across the full target voltage
grid `V_RHE ∈ [−0.50, +0.60] V` without regressing v13's existing
4-species path.

It captures (a) the problem structure, (b) every continuation strategy
under consideration, (c) tradeoffs, and (d) what has already been shown
to work, with file-path references where relevant.

## 1. Problem structure

### Forward physics (dimensionless form)

Coupled Poisson–Nernst–Planck system on a graded 2D rectangular mesh
(`Forward/bv_solver/mesh.py:make_graded_rectangle_mesh`,
`Nx=8, Ny=200, beta=3`):

```
∂c_i/∂t = -∇·J_i,   J_i = -D_i (∇c_i + z_i c_i ∇φ)        (Nernst-Planck)
-ε∇²φ   = F Σ z_i c_i                                     (Poisson)
```

Two electrode reactions with multi-step Butler–Volmer at the electrode
surface:

```
R1: O₂ + 2H⁺ + 2e⁻ → H₂O₂        E_eq = 0.68 V (RHE)
R2: H₂O₂ + 2H⁺ + 2e⁻ → 2H₂O      E_eq = 1.78 V (RHE)
```

Inverse parameters: `(log k0_1, log k0_2, α_1, α_2)`.

### What changed in the Apr 27 rebuild (the production stack)

Three independent toggles, all stacked in production:

1. **3-species + analytic Boltzmann counterion (Change 1).**  ClO₄⁻
   is removed as a dynamic Nernst–Planck unknown and represented as
   `c_CLO4(x) = c_CLO4_bulk · exp(-z_CLO4·φ(x))` (a Poisson–Boltzmann
   reduction).  The remaining dynamic species are
   `[O₂, H₂O₂, H⁺]` with `z = [0, 0, +1]`.  The Boltzmann ion
   contributes `−charge_rhs · z · c_bulk · exp(−z·φ)` to the Poisson
   residual.  See `Forward/bv_solver/boltzmann.py`.

2. **Log-concentration formulation (Change 2).**  Primary unknown is
   `u_i = ln(c_i)`.  Forces positivity exactly (`c_i = exp(u_i) > 0`)
   and removes the surface-concentration clip in BV.  Backend lives in
   `Forward/bv_solver/forms_logc.py` (vs the legacy concentration
   backend in `Forward/bv_solver/forms.py`).

3. **Log-rate Butler–Volmer (Change 3).**  Each BV branch is computed
   as `r = exp(log r)` with the surface concentration entering
   *additively* inside the exponent (`log r = ln k0 + u_cat - α·n_e·η/V_T`
   for the cathodic branch).  This eliminates the phantom R₂ sink at
   high anodic η that the old "clip + multiply" form produced.
   Toggle: `params['bv_convergence']['bv_log_rate'] = True`.

### Target voltage grid

`V_RHE ∈ {−0.50, −0.40, −0.30, −0.20, −0.10, 0.00, +0.05, +0.10,
+0.20, +0.30, +0.40, +0.50, +0.60} V`.

Why this matters: the writeup's whole point is that the new stack
expands the converged window from the old `V_RHE ≤ +0.20 V` to the
full `V_RHE ≤ +0.60 V` grid, and the local Fisher information
condition number drops from `~2×10¹¹` to `~1.79×10⁷`.  Any
continuation orchestration we adopt has to actually deliver
convergence on this whole grid.

### Key boundary-condition / geometry facts

- Electrode at `y=0`, bulk at `y=1`.  Markers: `electrode=3,
  concentration=4 (Dirichlet bulk for c_i), ground=4 (Dirichlet 0
  for φ at bulk)`.  At the electrode, `φ` has Dirichlet BC at
  `phi_applied = V_RHE / V_T` (no Stern layer in this configuration).
- Bulk Dirichlet BCs: `c_O₂ = 1.0`, `c_H₂O₂ = 1e-4` (small seed for
  log-c finiteness), `c_H⁺ = 0.2`.  The Boltzmann counterion
  contributes `−c_CLO4·exp(φ)` at every interior point; at the bulk
  (`φ=0`) it equals `−c_CLO4_bulk = −0.2`, exactly canceling
  `c_H⁺_bulk = +0.2` so the bulk is electroneutral *at full charge*.
- Continuation proceeds in a "z-factor" applied uniformly to both
  the dynamic species' charges and (post-fix) the Boltzmann ion's
  charge:
  - `z_factor = 0`: dynamic species charges set to 0, Boltzmann
    contribution multiplied by 0.  Bulk is truly neutral.  PDE
    reduces to diffusion + BV at electrode, plus a trivial Poisson
    (no charge sources).
  - `z_factor = 1`: physical solution.  Dynamic species at their
    real charges; Boltzmann ion at full charge.

  This `boltzmann_z_scale` was just added (`Forward/bv_solver/
  boltzmann.py`); previously the analytic Boltzmann ion's charge was
  always at full magnitude regardless of the z-factor for dynamic
  species, which broke any orchestration that wanted a truly neutral
  z=0 intermediate.

## 2. Software architecture (post-patch)

```
Forward/bv_solver/
    config.py              # parses bv_bc, bv_convergence (formulation, log_rate),
                           #   and bv_bc.boltzmann_counterions
    forms.py               # concentration backend (legacy, v13 path)
    forms_logc.py          # log-concentration backend (production)
    boltzmann.py           # add_boltzmann_counterion_residual()
    dispatch.py            # build_context/build_forms/set_initial_conditions
                           #   route on params['bv_convergence']['formulation']
    grid_charge_continuation.py   # Phase 1 V-sweep at z=0, Phase 2 per-point z-ramp
    solvers.py             # forsolve_bv, solve_bv_with_continuation, ...

FluxCurve/bv_point_solve/
    __init__.py            # solve_bv_curve_points_with_warmstart
                           #   (sequential two-branch sweep, full-z, used by v13)
    forward.py             # cached fast-path forward solve
    predictor.py           # Lagrange predictor + bridge points
```

The dispatcher and `Forward.bv_solver.__init__` are now live and route
correctly:
- `params['bv_convergence']['formulation'] = 'concentration'` → forms.py (default)
- `params['bv_convergence']['formulation'] = 'logc'` → forms_logc.py

`bv_log_rate` and `boltzmann_counterions` are formulation-independent:
they're config flags consumed by both backends.  The standard
production stack is `(formulation='logc', log_rate=True, boltzmann_counterions=[ClO4-])`.

The FluxCurve pipeline (`solve_bv_curve_points_with_warmstart`) imports
the dispatcher, so v13 inverse code automatically uses the right
backend whenever the config says so.

## 3. Continuation strategies under consideration

We need an orchestration that takes a target voltage grid
`V_RHE_grid` and produces converged full-z PNP solutions at every
voltage.  The candidates differ in how they combine **voltage
continuation** (sweep V across the grid using neighbor solutions as
warm-start) and **z-continuation** (gradually turn on the species'
charge contribution from 0 to 1).

### Strategy A: single anchor + sequential V-warm-start at full z

1. Cold-solve at `V=0` with internal z-ramp (one expensive cold).
2. Sequential two-branch sweep at full z: anodic chain `V=0 → +0.05 → +0.10 → … → +0.60`, cathodic chain `V=0 → −0.05 → −0.10 → … → −0.50`.  Each step restores the previous V's full-z snapshot, jumps `phi_applied`, runs steady-state.
3. Optional: per-step substepping/bisection inside any ΔV step that fails (split ΔV into 4 substeps with up to 3 levels of bisection).

This is essentially what `FluxCurve.solve_bv_curve_points_with_warmstart`
already does (with an additional 3-point Lagrange predictor between
voltages and bridge points for large gaps), so it's already in the
main pipeline and now routes to the logc backend via the dispatcher.

### Strategy B: V-sweep at z=0, then per-V z-ramp

1. Phase 1: hold all charges at z=0 (dynamic species' z's plus
   Boltzmann z_scale, both at 0).  At z=0 the system is just
   diffusion + BV + Dirichlet BCs.  Sweep voltage across the whole
   grid using bridge points + bisection where ΔV is large; each new
   V warm-starts off the previous V's z=0 solution.
2. Phase 2: independently per voltage, ramp z from 0 to 1 starting
   from that voltage's Phase 1 z=0 solution.  Adaptive z-ramp
   (binary search for foothold + geometric acceleration toward 1.0)
   per `Forward/bv_solver/grid_charge_continuation.py`.

This is `solve_grid_with_charge_continuation` and is the canonical
shape used by the 4-species concentration formulation.  In the 4sp
case it covers `V_RHE ∈ [−0.50, +0.10] V`.

The `boltzmann_z_scale` patch we just shipped is required to make
this shape work for the 3sp+Boltzmann formulation (without it, Phase
1 at z=0 still has the Boltzmann ion contributing a full
`−c_CLO4` to the bulk, which is non-electroneutral and breaks the
V-sweep).

### Strategy C: per-V cold-start with internal z-ramp (no V continuation)

For each voltage in the grid independently:
1. Build a fresh context at the target V (mesh, forms, IC).
2. The IC sets `u_i = ln(c0_i)` for every species and a linear `φ`
   profile from `V/V_T` at the electrode to `0` at the bulk.
3. Set `z_consts = 0` (and `boltzmann_z_scale = 0` post-fix).
4. Run steady-state at the target V with everything decoupled.
5. Ramp z from 0 to 1 in 20 linear steps, checkpointing.

This is what `v18_logc_lsq_inverse.solve_cold` does.  Each voltage
is independent — failures don't cascade.

### Strategy D: warm-walk from nearest converged neighbor

After C anchors a cold-success block, walk **outward** to remaining
voltages by warm-starting from the nearest converged neighbor at
**full z** and substepping `phi_applied`:
1. Restore neighbor's full-z snapshot.
2. Set `z_consts = z_nominal` (full charge from the start, no z-ramp
   here).
3. March `phi_applied` from `V_anchor` to `V_target` in `n_substeps=4`
   linear substeps.  On any substep failure, recursively bisect
   (`bisect_depth=3`).
4. Final SS at `V_target`.

This is `v24_3sp_logc_vs_4sp_validation.solve_warm_3sp_step`.  D is
not used standalone; it always runs as a fallback after C's
anchors.

### Combinations

- **C + D (v24's shape):** primary cold-start everywhere; warm-walk
  fills gaps at the cathodic and anodic edges.  Most robust shown
  to date.
- **A with substepping (FluxCurve + per-step substep/bisect):**
  cheapest if no chain-failure point in the grid; uses bridge
  points and recovery already present in
  `FluxCurve.bv_point_solve`.
- **B with the boltzmann_z_scale fix:** the `grid_charge_continuation`
  shape we already have, made formulation-aware.  Untested for
  3sp+Boltzmann after the fix.

## 4. Tradeoffs

| Strategy | Cold-solve cost | Robustness to single failure | V-continuation at full z required? | Existing infra |
|---|---|---|---|---|
| A | 1 cold + N cheap warms | Chain fails downstream of any failed step | Yes (every step) | `FluxCurve.solve_bv_curve_points_with_warmstart` (already routes to logc) |
| B | 1 cold + N cheap warms (Phase 1) + N z-ramps (Phase 2) | Each Phase 2 z-ramp is independent; Phase 1 chain can fail | No (Phase 1 is decoupled at z=0) | `solve_grid_with_charge_continuation` (now also routes to logc; needs `boltzmann_z_scale` fix to be useful) |
| C | NV cold-solves | Each cold is independent; failures don't propagate | No | Hand-rolled in `v18_logc_lsq_inverse.solve_cold` and `v24.solve_cold_3sp` |
| D | Walks remaining gaps | Localized; failures only cost the residual gap | Yes (within the walk, with substepping) | Hand-rolled in `v24.solve_warm_3sp_step` |
| C+D | NV cold + small number of walks | Most robust shown empirically | Yes (in D) | v24 prototype only; no production wrapper |

### Specific concerns per strategy

- **A is cheapest if the chain doesn't break.**  Empirically untested
  for the production logc+Boltzmann+log-rate stack at the full
  `V_RHE ∈ [−0.50, +0.60]` grid.  For the 4sp concentration case
  this shape works in the inverse pipeline.  Risk: at large `|V|`
  the BV exponent changes a lot per ΔV, so even a 0.05 V step at
  full z may require substepping.  v24's experience with
  `n_substeps=4 + bisect_depth=3` between adjacent cold successes
  suggests per-step substepping is needed beyond ≈ `|V_RHE| > 0.30 V`.

- **B has a clean phase separation.**  Phase 1 at z=0 is genuinely
  decoupled (after the `boltzmann_z_scale` fix), so V-continuation
  is just diffusion + BV.  Phase 2's z-ramps are per-V independent.
  Risk: BV flux is V-dependent even at z=0 (the BV exponential is
  over `(V−E_eq)/V_T`, no z anywhere), so each Phase 1 step still
  has to handle a strongly V-sensitive boundary flux.  Whether this
  is worse or better than full-z V-continuation in A is empirically
  unknown for this formulation.  Quick smoke test (with the
  Boltzmann fix) on `V_RHE ∈ {−0.20, −0.10, 0.00}` still shows
  Phase 1 failing to bridge `V=0 → V=−0.10` at `Ny=80` (see
  `StudyResults/v25_main_pipeline_vs_standalone_logc/` after the
  in-progress fix).  The failure is in the SNES at intermediate
  bridge points, so it might be solvable with finer bridge spacing
  rather than reflecting a fundamental obstruction.

- **C is the most expensive but cleanly the most robust per-V.**
  Empirically: cold-only converges in `V_RHE ≳ −0.20 V` for the
  3sp+Boltzmann logc stack.  For voltages outside that range, C
  alone fails — needs D on top.

- **D requires anchors.**  Without C's cold successes, there's
  nothing to walk from.  A single anchor at V=0 (D from V=0) is
  what reduces to A.

- **C+D is the empirical winner.**  v24 demonstrates the full
  `V_RHE ∈ [−0.50, +0.10] V` grid with 8/8 PASS at the F2
  diffusion-limit tolerance (5% of `max|obs|`), and the wall time
  is acceptable (~74 s for 8 voltages on Ny=200).  Cost concern:
  scales as O(NV) cold-solves up front, which is wasteful in the
  easy regime where a warm-start would suffice.

## 5. What has been validated

### v18 (cold-only) baseline

`scripts/studies/v18_logc_lsq_inverse.py:solve_cold` — per-V
cold-start with internal z-ramp.  Used for the *initial target curve*
in the inverse problem.  No warm-start fallback for that step:
failures abort with `FATAL: cold solve at TRUE failed at V=…`.
Empirically converges in roughly `V_RHE ≳ −0.20 V`.

### v24 (C+D) validation

`scripts/studies/v24_3sp_logc_vs_4sp_validation.py` — explicitly
tested by the writeup.  Compares the production logc+Boltzmann+log-rate
stack against the 4sp standard PNP via `solve_grid_with_charge_continuation`
on the overlap window `V_RHE ∈ [−0.50, +0.10] V`:

```
                                       max     mean
|ΔCD| / max|CD|                      0.269%  0.201%
|ΔPC| / max|PC|                      0.104%  0.033%
8/8 voltages PASS at the 5% F2 tolerance.
```

Wall time: 4sp = 17.8 s, 3sp logc (cold + warm continuation) = 74.3 s.

### grid_charge_continuation (B) for 4sp

`Forward/bv_solver/grid_charge_continuation.py` — the canonical
4sp charge-continuation pipeline.  Empirically reaches z=1 across
`V_RHE ∈ [−0.50, +0.10] V` for the 4sp formulation.  Untested for
3sp+Boltzmann after the `boltzmann_z_scale` patch.

### FluxCurve (A) for 4sp

`FluxCurve.bv_point_solve.solve_bv_curve_points_with_warmstart` —
sequential two-branch warm-start at full z, with bridge points and
per-step recovery.  Used by v13 (`scripts/Inference/Infer_BVMaster_charged_v13_ultimate.py`)
and stable for the 4sp concentration formulation across the v13 cathodic
grid `eta_cathodic ∈ [−1.0, −46.5]` (in nondim units, ≈ `V_RHE ∈
[−1.20, +0.00] V` after E_eq).  Untested at full z for 3sp+Boltzmann
on the production grid.

### Adjoint validity

Per the writeup §"Adjoint validity": at three previously inaccessible
voltages `V_RHE ∈ {0.30, 0.40, 0.50}`, adjoint gradients agree with
cold-ramp finite differences to ~6 significant figures.  This means
whichever continuation strategy we pick must preserve the adjoint
machinery (annotation enabled around the SNES solves) at the
voltages we care about.

## 6. Open question for the reader

Which of these orchestrations should we wire as the canonical entry
point in `Forward.bv_solver` for the production 3sp+Boltzmann+logc+log-rate
stack on the full `V_RHE ∈ [−0.50, +0.60] V` grid?

Sub-questions:

1. **Is A with per-step substepping likely to work for this
   formulation?**  If yes, it's the cheapest option and we don't
   need to add anything to the pipeline beyond the dispatcher patch
   that's already in.  v13's FluxCurve pipeline already implements
   it; we'd just exercise it through the new backend.

2. **Is B (grid_charge_continuation with the boltzmann_z_scale fix)
   structurally sound for the new formulation?**  Phase 1 at z=0
   is now truly neutral; that should make it at least as well-conditioned
   as the 4sp case where this shape works.  But quick tests with
   small `Ny=80` still show Phase 1 V-sweep failing to bridge
   `V=0 → V=−0.10`, which is suspicious.  Is this a mesh-resolution
   artifact, a SNES tolerance issue, or a fundamental incompatibility
   with the log-c primary variable?

3. **Should we just productionize C+D since it is the only thing
   empirically shown to cover the full grid?**  Cost is acceptable
   (~75 s for 8 voltages at production resolution).  The wrapper
   would be ≈ 200 lines of Python around the dispatcher.

4. **Is there a hybrid worth exploring?**  E.g.:
   - "A with cold-start fallback at any V where the warm-start chain
     breaks" — start with A for cheapness, fall back to C only at
     the failing V, optionally followed by D from there.
   - "B with the option to fall back to per-V cold (C) if Phase 1
     fails to reach a particular V, plus D from cold successes."

Constraints to keep in mind:

- **Backward compatibility:** v13's FluxCurve-driven inverse must keep
  working with `formulation='concentration'` defaulted to the 4sp path.
  Whatever shape we pick for the new path must coexist.
- **Adjoint preservation:** the orchestration must support pyadjoint
  annotation around the final converged solve at each V (used for
  Jacobian computation in the inverse problem).  v18/v24 do this by
  re-solving a few annotated final SS steps after the unannotated
  cold/warm work.
- **CLI flags:** the user wants the new formulation toggleable via
  CLI flags on the production scripts.  Whatever orchestration we
  pick, the script-level surface should be `--formulation
  {logc,concentration} --log-rate/--no-log-rate
  --boltzmann-counterion/--no-boltzmann-counterion`.

A recommendation on which strategy to commit to (with a concrete
empirical test plan if needed) would unblock the rest of the patch.

## 7. Where the code lives (for grounding)

- Dispatcher (already patched): `Forward/bv_solver/dispatch.py`,
  `Forward/bv_solver/__init__.py`.
- Boltzmann residual (already patched, with `boltzmann_z_scale` knob):
  `Forward/bv_solver/boltzmann.py`.
- Config parsing: `Forward/bv_solver/config.py`.
- Concentration backend: `Forward/bv_solver/forms.py`.
- Log-c backend: `Forward/bv_solver/forms_logc.py`.
- B-shape orchestration: `Forward/bv_solver/grid_charge_continuation.py`.
- A-shape orchestration: `FluxCurve/bv_point_solve/__init__.py`.
- C+D reference (v24 prototype):
  `scripts/studies/v24_3sp_logc_vs_4sp_validation.py`.
- v18 cold-only reference:
  `scripts/studies/v18_logc_lsq_inverse.py`.
- Param factory with new toggles: `scripts/_bv_common.py`
  (`make_bv_solver_params(formulation=…, log_rate=…,
  boltzmann_counterions=…)`, `THREE_SPECIES_LOGC_BOLTZMANN`,
  `DEFAULT_CLO4_BOLTZMANN_COUNTERION`).
- In-progress parity test (currently failing on B, C+D not yet wired):
  `scripts/studies/v25_main_pipeline_vs_standalone_logc.py`.
- Apr 27 writeup: `writeups/WeekOfApr27/PNP Inverse Solver Revised.tex`.
