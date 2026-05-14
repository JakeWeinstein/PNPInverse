## Q1

SEVERITY: note
LOCATION: scripts/studies/solver_demo_slide15_no_speculative_cs.py:454-463
TRIGGER: Stage 2 receives `SystemExit` or `KeyboardInterrupt` during a Stern bump.
EVIDENCE: The handler is `except Exception as exc` at scripts/studies/solver_demo_slide15_no_speculative_cs.py:460, not `except BaseException`; this catches normal solver/programming exceptions but does not catch `SystemExit` or `KeyboardInterrupt`.

SEVERITY: warning
LOCATION: scripts/studies/solver_demo_slide15_no_speculative_cs.py:454-466, scripts/studies/solver_demo_slide15_no_speculative_cs.py:502-514, Forward/bv_solver/anchor_continuation.py:459-467
TRIGGER: A future edit, harness, or manual intervention bypasses the `if bump_err is not None: return` path after a failed Stern bump.
EVIDENCE: `set_stern_capacitance_model(ctx_anchor, float(cs_target))` runs before `ctx_anchor["_last_solver"].solve()` at scripts/studies/solver_demo_slide15_no_speculative_cs.py:456-458; the setter updates `ctx["nondim"]` and then `stern_const.assign(nondim_value)` at Forward/bv_solver/anchor_continuation.py:459-467. The current code returns immediately on `bump_err` at scripts/studies/solver_demo_slide15_no_speculative_cs.py:464-498, so normal Stage 3 cannot see the failed state; if that guard were bypassed, `snapshot_U(ctx_anchor["U"])` and `PreconvergedAnchor(...)` at scripts/studies/solver_demo_slide15_no_speculative_cs.py:502-514 would use the same mutated context. UNVERIFIED whether Firedrake leaves partial Newton updates in `U` on failed `solve()`, but the demo has no restore-to-last-success path in this loop.

SEVERITY: note
LOCATION: scripts/studies/solver_demo_slide15_no_speculative_cs.py:456-462
TRIGGER: Stage 2 history reporting for successful and failed Stern bump targets.
EVIDENCE: `"ok"` is appended only after `ctx_anchor["_last_solver"].solve()` returns at scripts/studies/solver_demo_slide15_no_speculative_cs.py:457-459, and `"fail"` is appended only in the exception branch at scripts/studies/solver_demo_slide15_no_speculative_cs.py:460-462; no defect found in append placement.

## Q2

SEVERITY: note
LOCATION: scripts/studies/solver_demo_slide15_no_speculative_cs.py:105-118, scripts/studies/solver_demo_slide15_no_speculative_cs.py:160-165, scripts/_bv_common.py:157-158, scripts/_bv_common.py:373-378, Forward/bv_solver/forms_logc_muh.py:407-452
TRIGGER: Production `_make_sp` builds the dynamic O2/H2O2/H+ steric parameters for the `logc_muh` path.
EVIDENCE: The preset `THREE_SPECIES_LOGC_BOLTZMANN` uses `[A_DEFAULT] * 3` with `A_DEFAULT = 0.01` at scripts/_bv_common.py:157-158 and scripts/_bv_common.py:373-378, while the demo overrides with `a_vals_hat=[A_O2_PHYSICAL, A_H2O2_PHYSICAL, A_HP_PHYSICAL]` at scripts/studies/solver_demo_slide15_no_speculative_cs.py:160-165. The demo formula is `a_phys=(4/3)*pi*r^3*N_A`, `a_nondim=a_phys*C_SCALE` at scripts/studies/solver_demo_slide15_no_speculative_cs.py:105-113. For H+ r=2.80e-10 m: `a_phys=(4/3)*pi*(2.80e-10)^3*6.02214076e23=5.5374982976e-5 m^3/mol`; `a_hat=5.5374982976e-5*1.2=6.6449979571e-5`; `c_max_hat=1/a_hat=15048.9136`; `c_max_phys=C_SCALE/a_hat=1.2/6.6449979571e-5=18058.6963 mol/m^3`, matching the expected ~1.8e4 mol/m^3. The `logc_muh` form consumes these as `A_dyn=sum(a_i*c_i)` and `theta=1-A_dyn-...` at Forward/bv_solver/forms_logc_muh.py:407-452, so the nondimensionalization is consistent.

## Q3

SEVERITY: warning
LOCATION: scripts/studies/solver_demo_slide15_no_speculative_cs.py:301-322
TRIGGER: `_stern_bump_ladder` is called with `target <= STERN_ANCHOR`, especially `--stern-final 0.10` or a smaller custom value.
EVIDENCE: `STERN_ANCHOR = 0.10` at scripts/studies/solver_demo_slide15_no_speculative_cs.py:99 and `_stern_bump_ladder` returns `[float(target)]` for `target <= STERN_ANCHOR` at scripts/studies/solver_demo_slide15_no_speculative_cs.py:304-313, so `target == 0.10` returns `[0.10]` even though the docstring says the list excludes the 0.10 starting point at scripts/studies/solver_demo_slide15_no_speculative_cs.py:305-310; `target < 0.10` returns `[target]` and performs a direct downward step.

SEVERITY: note
LOCATION: scripts/studies/solver_demo_slide15_no_speculative_cs.py:301-322
TRIGGER: Production Stern target `0.20` and near-no-Stern target `100.0`.
EVIDENCE: For `target == 0.20`, the first verified rung `0.20` satisfies `rung >= target`, so the function appends `float(target)` and returns `[0.20]` at scripts/studies/solver_demo_slide15_no_speculative_cs.py:315-318. For `target == 100.0`, the loop appends rungs `0.20, 0.50, 1.0, 2.0, 5.0, 10.0` at scripts/studies/solver_demo_slide15_no_speculative_cs.py:315-319, then the `100.0` rung satisfies `rung >= target` and returns `[0.20, 0.50, 1.0, 2.0, 5.0, 10.0, 100.0]`; no defect found for the production `0.20` and documented `100.0` cases.

## Q4

SEVERITY: warning
LOCATION: Forward/bv_solver/dispatch.py:68-80, Forward/bv_solver/dispatch.py:82-120, Forward/bv_solver/config.py:82-95
TRIGGER: A caller bypasses config validation and passes raw/legacy solver parameters whose `bv_convergence.formulation` is an unrecognized string.
EVIDENCE: `_resolve_backend` returns `"logc_muh"` only for exact `"logc_muh"` and otherwise returns `"logc"` at Forward/bv_solver/dispatch.py:76-79; `build_context`, `build_forms`, and `set_initial_conditions` then use the log-c backend at Forward/bv_solver/dispatch.py:82-120. There is no `NotImplementedError` or dispatch-level raise. The separate config layer would reject unknown formulations at Forward/bv_solver/config.py:82-95, but dispatch.py by itself silently falls back to `"logc"`.

## Q5

SEVERITY: note
LOCATION: Forward/bv_solver/mesh.py:68-87, Forward/bv_solver/mesh.py:102-108
TRIGGER: `make_graded_rectangle_mesh(Nx=8, Ny=80, beta=3.0, domain_height_hat=1.0)`.
EVIDENCE: The mesh starts as `fd.RectangleMesh(Nx, Ny, 1.0, 1.0)` and then applies `coords[:, 1] = (coords[:, 1] ** beta) * domain_height_hat` at Forward/bv_solver/mesh.py:102-107, so the y nodes follow `y_i=(i/80)^3*1.0`; `y_80=(80/80)^3=1.0` equals `domain_height_hat`. The cell widths are `Delta y_i=(i^3-(i-1)^3)/80^3`; `Delta y_1=1/512000=1.953125e-6` and `Delta y_80=(80^3-79^3)/80^3=18961/512000=0.037033203125`, so the graded fine end is at y=0. The docstring marks boundary `3 = bottom (y=0, electrode)` and `4 = top (y=domain_height_hat, bulk)` at Forward/bv_solver/mesh.py:82-86; no defect found for the requested mesh.

## Q6

SEVERITY: note
LOCATION: scripts/studies/solver_demo_slide15_no_speculative_cs.py:502-514, Forward/bv_solver/grid_per_voltage.py:102-123, Forward/bv_solver/anchor_continuation.py:110-180, Forward/bv_solver/grid_per_voltage.py:1060-1113
TRIGGER: Code between Stage 2 success and the Stage 3 `solve_grid_with_anchor(...)` call.
EVIDENCE: After the Stage 2 loop, the only pre-Stage-3 state extraction is `U_post_bump = snapshot_U(ctx_anchor["U"])` and construction of `PreconvergedAnchor(...)` at scripts/studies/solver_demo_slide15_no_speculative_cs.py:502-514. `snapshot_U` is `tuple(d.data_ro.copy() for d in U.dat)` at Forward/bv_solver/grid_per_voltage.py:102-123, and `PreconvergedAnchor.__post_init__` validates scalars/tuples/NumPy arrays without Firedrake assignment or assembly at Forward/bv_solver/anchor_continuation.py:110-180. The Stage 3 driver itself wraps its loop, including callback invocation, in `with adj.stop_annotating():` at Forward/bv_solver/grid_per_voltage.py:1060-1113; no pyadjoint-recordable Firedrake write/solve/assemble was found between Stage 2 exit and Stage 3 entry in the code read.

## Q7

SEVERITY: warning
LOCATION: scripts/studies/solver_demo_slide15_no_speculative_cs.py:520-538, scripts/studies/solver_demo_slide15_no_speculative_cs.py:554-564
TRIGGER: A converged grid point reaches `_grab`, but current-density or peroxide-current observable construction/assembly raises.
EVIDENCE: `_grab` catches `Exception` around both observable assemblies and only prints at scripts/studies/solver_demo_slide15_no_speculative_cs.py:520-538. The point convergence status is computed independently from `grid_result.points[i].converged` at scripts/studies/solver_demo_slide15_no_speculative_cs.py:554-556, while `_to_json_list` converts non-finite `cd_arr`/`pc_arr` entries to `None` at scripts/studies/solver_demo_slide15_no_speculative_cs.py:560-564. This can produce `converged=True` and count the point in `n_converged` while the I-V current value is `null`, masking an observable correctness failure in the output curve.

## Other findings

SEVERITY: warning
LOCATION: scripts/studies/solver_demo_slide15_no_speculative_cs.py:186-187, scripts/studies/solver_demo_slide15_no_speculative_cs.py:600-609, Forward/bv_solver/anchor_continuation.py:176-180
TRIGGER: A custom CLI run passes `--factors 0`, a negative factor, `nan`, or `inf`.
EVIDENCE: `_parse_factor_list` only strips tokens and returns `tuple(float(tok)...)` without finite or positive validation at scripts/studies/solver_demo_slide15_no_speculative_cs.py:600-609. `_make_sp` computes `k0_r4e_target = float(K0_HAT_R4E) * float(k0_r4e_factor)` at scripts/studies/solver_demo_slide15_no_speculative_cs.py:186 and later stores it in `k0_targets` at scripts/studies/solver_demo_slide15_no_speculative_cs.py:243; `PreconvergedAnchor.__post_init__` requires `float(k) > 0.0` and raises only after Stage 2 work at Forward/bv_solver/anchor_continuation.py:176-180. Bad factors therefore fail late or enter solver setup before validation.

SEVERITY: warning
LOCATION: scripts/studies/solver_demo_slide15_no_speculative_cs.py:638-644, scripts/studies/solver_demo_slide15_no_speculative_cs.py:690-693, scripts/studies/solver_demo_slide15_no_speculative_cs.py:304-322, Forward/bv_solver/config.py:58-63, Forward/bv_solver/anchor_continuation.py:444-467
TRIGGER: A custom CLI run passes non-finite or zero `--stern-final` values.
EVIDENCE: `--stern-final` is parsed as `float` at scripts/studies/solver_demo_slide15_no_speculative_cs.py:638-644 and assigned without finite/positive validation at scripts/studies/solver_demo_slide15_no_speculative_cs.py:690-693. Config validation only rejects `< 0` at Forward/bv_solver/config.py:58-63, and the live setter only rejects `< 0` before assigning the nondimensional value at Forward/bv_solver/anchor_continuation.py:444-467. `_stern_bump_ladder(float("nan"))` would skip every `rung >= target` and `rungs[-1] < target` comparison at scripts/studies/solver_demo_slide15_no_speculative_cs.py:315-321, returning the verified rungs through `100.0` while the baseline solver params carry `stern_final_v=nan`; `target == 0.0` returns `[0.0]` at scripts/studies/solver_demo_slide15_no_speculative_cs.py:312-313 and attempts to mutate a live Robin Stern coefficient to zero rather than rebuilding the no-Stern Dirichlet form.

SEVERITY: warning
LOCATION: Forward/bv_solver/mesh.py:68-73, Forward/bv_solver/mesh.py:102-108
TRIGGER: A caller uses `make_graded_rectangle_mesh` with `beta <= 0`, `nan`, or `inf`.
EVIDENCE: `make_graded_rectangle_mesh` validates `domain_height_hat` at Forward/bv_solver/mesh.py:102 but performs no validation on `beta` before `coords[:, 1] = (coords[:, 1] ** beta) * domain_height_hat` at Forward/bv_solver/mesh.py:103-107. The docstring describes `beta > 1` clustering at Forward/bv_solver/mesh.py:76-78, but invalid beta values can collapse or corrupt y coordinates at the electrode endpoint.

VERDICT: CONCERNS FOUND
