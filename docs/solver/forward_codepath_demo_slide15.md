# Forward solver codepath — `solver_demo_slide15_no_speculative_cs.py`

Visual call graph for the code touched when the Cs⁺/SO₄²⁻ baseline I-V
demo runs (the dataset plotted by
`plot_solver_demo_slide15_no_speculative_cs.py`).  Every leaf is a real
function with `file.py:line`; the right-hand annotations call out the
parts that make convergence possible.

```
══════════════════════════════════════════════════════════════════════════════
ENTRY  scripts/studies/solver_demo_slide15_no_speculative_cs.py
       main()  →  _run_one_factor()      ← loops over K0_R4e ∈ {1, 1e-6, 1e-12, 1e-18}
══════════════════════════════════════════════════════════════════════════════
       │
       ├──► setup_firedrake_env()                       (scripts/_bv_common.py)
       │
       ├──► make_graded_rectangle_mesh(Nx=8, Ny=80,     (bv_solver/mesh.py)
       │       beta=3.0, domain_height_hat=1.0)             ← mesh once per run
       │
       ├──► _make_sp(stern=0.10, k0_r4e_factor=f)       sp_anchor
       │      └─ make_bv_solver_params(...)             (scripts/_bv_common.py)
       │             • formulation = "logc_muh"
       │             • THREE_SPECIES_LOGC_BOLTZMANN
       │             • physical a_nondim (O2 r=1.7Å, H2O2 r=2.0Å, H+ r=2.8Å)
       │             • Cs+/SO4²⁻ Bikerman counterions
       │             • exponent_clip = 100      ← Hard Rule #2: clip η_scaled before α·n_e
       │
       ├──► _make_sp(stern=0.20, k0_r4e_factor=f)       sp_baseline (production C_S)
       │
       ▼
══════════════════════════════════════════════════════════════════════════════
STAGE 1 — ANCHOR BUILD          (cold Newton at V_RHE = +0.55 V, C_S = 0.10)
══════════════════════════════════════════════════════════════════════════════
       │
       │  solve_anchor_with_continuation(sp_anchor, mesh, k0_targets,
       │      initial_scales=(1e-12, 1e-9, 1e-6, 1e-3, 1.0),
       │      max_inserts_per_step=4, ic_at_target=True)
       │      └─ Forward/bv_solver/anchor_continuation.py:902
       │
       │  ┌──── 1.  build_context(sp, mesh)             dispatch.py:82
       │  │       └─ build_context_logc_muh             forms_logc_muh.py:152
       │  │            • mixed FE space  W = V_u × V_μH × V_φ
       │  │            • U  = Function(W) (Newton iterate)
       │  │            • U_prev for SER pseudo-time
       │  │
       │  ├──── 2.  build_forms(ctx, sp)                dispatch.py:89
       │  │       └─ build_forms_logc_muh               forms_logc_muh.py:166
       │  │            • build_model_scaling            nondim.py
       │  │            • _resolve_mu_h_index            (H+ for μ_H transform)
       │  │            • build_steric_boltzmann_expressions    ← Tresset closure
       │  │                 boltzmann.py:91                      with (1−A_dyn) for
       │  │                                                       Cs+ AND SO4²⁻
       │  │            • add_boltzmann_counterion_residual      boltzmann.py:272
       │  │            • assemble F_res = Poisson + NP + BV
       │  │            • J_form = derivative(F_res, U)
       │  │            • NonlinearVariationalProblem(F, U, bcs, J)
       │  │            • NonlinearVariationalSolver(problem, options=SNES_OPTS_CHARGED)
       │  │                 → ctx['_last_solver']  ← reused by Stern bump
       │  │
       │  ├──── 3.  set_initial_conditions(ctx, sp)     dispatch.py:96
       │  │       └─ set_ic_debye_boltzmann_logc_muh    forms_logc_muh.py:997
       │  │            └─ picard_outer_loop_general     picard_ic.py
       │  │                 • 2×2 scalar Picard on (ψ_S, ψ_D)
       │  │                 • composite-ψ + multispecies-γ seed
       │  │                 • falls back to linear_phi if Picard oscillates
       │  │
       │  └──── 4.  AdaptiveLadder(initial_scales)      anchor_continuation.py:719
       │
       │       ┌─ while not ladder.is_done():
       │       │     k0_scale = ladder.current_scale       ← starts at 1e-12 (kinetics off,
       │       │                                              diffusion-only Newton)
       │       │     for j, k_target in k0_targets:
       │       │         solve_reaction_k0_model(ctx, j,    anchor_continuation.py:246
       │       │             k0_scale * k_target)
       │       │     ok = run_ss(max_steps)              ← SER pseudo-time loop
       │       │           │
       │       │           └─ make_run_ss(...)            grid_per_voltage.py:135
       │       │                ┌─ for k in range(max_steps):
       │       │                │     solver.solve()       ← Firedrake SNES (Newton+LS)
       │       │                │     j_cd = assemble(of_cd)
       │       │                │     if |Δj_cd| < ss_rel_tol · |j_cd| + ss_abs_tol
       │       │                │        for ss_consec steps:
       │       │                │            return True  ← plateau detected = steady state
       │       │                │     dt *= grow if ok else shrink
       │       │                └─ return False if max_steps hit
       │       │     if ok:
       │       │         snapshot_U(U)                    ← save converged state
       │       │         ladder.record_success()          ← advance to next scale
       │       │     else:
       │       │         if ladder.record_failure_and_insert():
       │       │             restore_U(last_snap)
       │       │             ↻ retry at sqrt(prev_scale, failed_scale)  ← sqrt-midpoint
       │       │         else:
       │       │             raise LadderExhausted
       │       └─ return AnchorContinuationResult(converged=True, ctx=...)
       │
       ▼
══════════════════════════════════════════════════════════════════════════════
STAGE 2 — STERN BUMP LADDER     (C_S: 0.10 → 0.20 F/m², or → 100 F/m² no-Stern)
══════════════════════════════════════════════════════════════════════════════
       │
       │  for cs in _stern_bump_ladder(target):    ← (0.20, 0.50, 1.0, 2.0, 5.0, 10.0, 100.0)
       │      set_stern_capacitance_model(ctx, cs) anchor_continuation.py
       │          • mutates ctx['stern_capacitance_func']  ← shared FE Constant
       │          • residual sees new C_S without rebuild
       │      ctx['_last_solver'].solve()          ← reuse the anchor's solver
       │
       │  snapshot_U(ctx['U'])   →   PreconvergedAnchor(phi_eta, U_snapshot,
       │                                                 k0_targets, dof_count,
       │                                                 ladder_history)
       │
       ▼
══════════════════════════════════════════════════════════════════════════════
STAGE 3 — GRID WALK             (25 V_RHE points, [−0.40, +0.55] V)
══════════════════════════════════════════════════════════════════════════════
       │
       │  solve_grid_with_anchor(sp_baseline, anchor=PreconvergedAnchor,
       │      phi_applied_values=φ_grid, mesh,
       │      n_substeps_warm=8, bisect_depth_warm=5,
       │      per_point_callback=_grab)
       │      └─ Forward/bv_solver/grid_per_voltage.py:875
       │
       │  ┌─ sort grid by |V − V_anchor| ascending      ← walk outward from anchor
       │  │
       │  └─ for each target φ in sorted grid:
       │
       │       ┌─ _build_for_voltage(φ_target)          ← fresh ctx at target φ
       │       │    (same dispatch as Stage 1 steps 1–3, but at C_S = 0.20)
       │       │
       │       ├─ choose source = nearest converged neighbor
       │       │    (anchor first, then any previously-walked grid point)
       │       │
       │       ├─ restore_U(source_snapshot, ctx['U'], ctx['U_prev'])  ← warm-start
       │       │
       │       ├─ ctx['phi_applied_func'].assign(source_φ)       ← start at source
       │       │
       │       ├─ warm_walk_phi(ctx, solver, of_cd,              grid_per_voltage.py:218
       │       │     v_anchor_eta=src_φ, v_target_eta=tgt_φ,
       │       │     n_substeps=8, bisect_depth_warm=5)
       │       │     │
       │       │     └─ _march(v0=src, v1=tgt, depth=0):
       │       │          for v_sub in linspace(v0, v1, N+1)[1:]:
       │       │              paf.assign(v_sub)
       │       │              if not run_ss(max_steps):
       │       │                  if depth < bisect_depth:
       │       │                      _march(v_prev, midpoint, depth+1)   ← recursive
       │       │                      _march(midpoint, v_sub,  depth+1)      bisection
       │       │                  else:
       │       │                      restore_U; return False               ← give up
       │       │              v_prev = v_sub
       │       │          # final pin at target
       │       │          paf.assign(tgt)
       │       │          return run_ss(max_steps_final)
       │       │
       │       ├─ on success: snapshot_U → append (tgt_φ, snap) to source pool
       │       │              per_point_callback(orig_idx, tgt_φ, ctx)
       │       │                  └─ _build_bv_observable_form(ctx, mode=...)  observables.py:67
       │       │                       • mode="current_density"    → ∫ Σ_j (n_e/2)·R_j ds
       │       │                       • mode="gross_h2o2_current" → ∫ R_R2e ds only
       │       │                  fd.assemble(form) → cd_arr[i], pc_arr[i]
       │       │
       │       └─ on failure: mark not-converged; next grid point picks a different
       │                       source if available
       │
       ▼
══════════════════════════════════════════════════════════════════════════════
OUTPUT  StudyResults/solver_demo_slide15_no_speculative_cs/
        factor_{f:g}/iv_curve.json     ← cd_mA_cm2, pc_mA_cm2 per V, convergence flags
══════════════════════════════════════════════════════════════════════════════
```

## What makes convergence possible

The solver is layered defense.  Each item below is the line that
keeps the next-deepest Newton solve in basin.

| Layer | Mechanism | Where | Why it matters |
|---|---|---|---|
| 1 | **k0-scale ladder** (1e-12 → 1.0) | `AdaptiveLadder` @ `anchor_continuation.py:719` | Starts with kinetics off (Newton sees diffusion-only problem), ramps to production k0.  Sqrt-midpoint insertion recovers from failed rungs without restart. |
| 2 | **debye_boltzmann IC** | `picard_ic.py` → `set_ic_debye_boltzmann_logc_muh` | Seeds the EDL with the right composite ψ_S + ψ_D structure so Newton's initial residual is O(1) rather than O(1e26). |
| 3 | **`exponent_clip = 100`** | `forms_logc_muh.py:_build_eta_clipped` | Clamps η_scaled before α·n_e multiplication.  Without it `exp(α·n_e·η)` over/underflows and SNES gets Inf in F.  At clip=100 the production V band is fully unclipped. |
| 4 | **Bikerman analytic counterion closure** | `boltzmann.py:build_steric_boltzmann_expressions` | Cs⁺ and SO₄²⁻ never enter the Newton state vector — they're closed-form functionals of φ.  Eliminates the transport modes that would otherwise stall Newton in the Debye layer. |
| 5 | **Stern bump ladder** | `_stern_bump_ladder` in demo + `set_stern_capacitance_model` | C_S = 0.20 F/m² won't anchor cold; C_S = 0.10 will.  Builds at 0.10, then ramps to 0.20 via verified rungs without rebuilding forms. |
| 6 | **SER plateau detection** | `make_run_ss` @ `grid_per_voltage.py:135` | Newton solves alone don't tell you the BV system is steady — you need ∫current to stop drifting.  Plateau detector watches the surface flux observable and declares success when it stops moving. |
| 7 | **Warm-start from nearest neighbor** | `solve_grid_with_anchor` @ `grid_per_voltage.py:875` | Each grid point inherits the converged U from its closest already-walked neighbor.  Newton's basin radius in φ_applied is ~0.05 V; the anchor is up to 0.95 V from the cathodic edge, so the walk visits the points in distance order, not list order. |
| 8 | **Recursive φ-substep bisection** | `warm_walk_phi` @ `grid_per_voltage.py:218` | If the 8-substep warm walk fails between two grid points, the failing interval is halved and retried — up to `bisect_depth_warm=5` levels (32× refinement).  Catches the Frumkin cliff near V≈0 V where the Bikerman SO₄²⁻ saturation suddenly bites. |
| 9 | **`adj.stop_annotating()` wrapper** | demo script around solver calls | The forward demo doesn't need adjoint tape; wrapping prevents pyadjoint from recording 10k+ operations per solve, which would OOM the process. |

## Files touched (by stage)

```
Stage 1 (anchor)            Stage 2 (Stern bump)         Stage 3 (grid walk)
─────────────────           ──────────────────────        ─────────────────────
anchor_continuation.py      anchor_continuation.py        grid_per_voltage.py
  ├─ AdaptiveLadder            └─ set_stern_capacitance_   ├─ solve_grid_with_anchor
  ├─ solve_reaction_k0_           model                    ├─ warm_walk_phi
  │  model                                                 ├─ _march (bisection)
  └─ solve_anchor_with_                                    ├─ snapshot_U / restore_U
     continuation                                          └─ make_run_ss

dispatch.py                                                observables.py
  └─ build_context / build_forms                            └─ _build_bv_observable_form
     / set_initial_conditions
                                                           dispatch.py
forms_logc_muh.py                                           └─ same build_* for each
  ├─ build_context_logc_muh                                    voltage's fresh ctx
  ├─ build_forms_logc_muh
  ├─ set_ic_debye_boltzmann_logc_muh
  └─ _build_eta_clipped

boltzmann.py
  ├─ build_steric_boltzmann_expressions   ← Tresset Eq. (19) + (1−A_dyn)
  └─ add_boltzmann_counterion_residual

picard_ic.py
  └─ picard_outer_loop_general

multi_ion.py
  └─ shared-θ closure helpers

mesh.py
  └─ make_graded_rectangle_mesh

nondim.py
  └─ build_model_scaling
```

## Per-factor wall-time breakdown (typical)

```
Stage 1 anchor build:    20-60 s   (5 k0-ladder rungs × ~5 SS steps × Newton)
Stage 2 Stern bump:      10-30 s   (7 rungs × 1 Newton solve each)
Stage 3 grid walk:       2-5 min   (25 points × warm-walk × ~8 substeps × Newton)
                                    + occasional bisection at Frumkin cliff
Total per factor:        3-7 min
4 factors:               15-30 min
```

## Notes

* The plot script `plot_solver_demo_slide15_no_speculative_cs.py` reads
  only the `iv_curve.json` files this codepath writes — no solver code
  is exercised by plotting.
* The same call graph holds for the no-Stern variant (`--no-stern`),
  the only differences being `stern_final = 100.0` and the bump ladder
  going through more rungs.  Frumkin cliff at V≈0 V triggers more
  bisection events in the no-Stern run.
* This is **not** the inverse-solver codepath.  The inverse path
  (`scripts/Inference/`) is paused and uses different orchestrators
  (`Inference/lm_tikhonov.py`, etc.).
