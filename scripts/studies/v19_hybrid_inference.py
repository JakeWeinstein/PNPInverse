"""v19 hybrid inference: 4 free BV params from CD + PC with L-BFGS-B + pyadjoint.

Architecture (following v13's IC-cache + parallel pattern):

  Main process
    IC cache: dict[voltage_idx -> tuple of numpy arrays of converged U.dat]
    scipy L-BFGS-B with jac=True
    Each objective-call dispatches one task per voltage to the worker pool.

  Worker processes (ProcessPoolExecutor, spawn context)
    At init:  build mesh once per worker.
    Per task: build ctx (logc or 4sp by voltage), load cached IC, run a small
              number of pseudo-time steps to re-converge, assemble CD + PC,
              compute per-voltage objective + adjoint gradient w.r.t.
              (k0_1, k0_2, alpha_1, alpha_2). Return new converged U arrays
              so the main process can refresh the cache for the next BFGS iter.

Observables
  CD everywhere (physical in both regimes).
  PC only on 4sp voltages (V < V_THRESHOLD); logc PC is an artifact of the
  H2O2 seed x exponent-clip interaction.

Targets
  Loaded from StudyResults/v19_hybrid/iv_target.npz (produced by
  scripts/studies/v19_hybrid_forward.py). 2% Gaussian noise added.

Usage (from PNPInverse/):
    ../venv-firedrake/bin/python scripts/studies/v19_hybrid_inference.py
    ../venv-firedrake/bin/python scripts/studies/v19_hybrid_inference.py --workers 4
    ../venv-firedrake/bin/python scripts/studies/v19_hybrid_inference.py --fd-grad
"""
from __future__ import annotations
import os, sys, time, argparse, pickle, traceback
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(line_buffering=True)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

E_EQ_R1 = 0.68
E_EQ_R2 = 1.78
H2O2_SEED = 1e-4
V_THRESHOLD = 0.0       # logc at V>=0, 4sp at V<0
NOISE_PCT = 2.0
NOISE_SEED = 42
INIT_OFFSET_K0 = 1.2    # +20% initial guess on k0
INIT_OFFSET_ALPHA = 0.9 # -10% initial guess on alpha
MESH_NX = 8
MESH_NY = 200
MESH_BETA = 3.0

# BFGS iteration budget
MAXITER = 15


# ---------------------------------------------------------------------------
# Worker configuration (frozen, pickled to each worker once)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WorkerCfg:
    mesh_nx: int
    mesh_ny: int
    mesh_beta: float
    w_pc_threshold: float   # weight of PC observable (0 for logc, 1 for 4sp)


_WORKER_CFG: Optional[WorkerCfg] = None
_WORKER_MESH: Any = None


def _worker_init(cfg: WorkerCfg) -> None:
    """Called once per worker process. Pre-build the shared mesh."""
    global _WORKER_CFG, _WORKER_MESH
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ.setdefault("FIREDRAKE_TSFC_KERNEL_CACHE_DIR", "/tmp/firedrake-tsfc")
    os.environ.setdefault("PYOP2_CACHE_DIR", "/tmp/pyop2")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")
    _WORKER_CFG = cfg
    from Forward.bv_solver import make_graded_rectangle_mesh
    _WORKER_MESH = make_graded_rectangle_mesh(
        Nx=cfg.mesh_nx, Ny=cfg.mesh_ny, beta=cfg.mesh_beta,
    )


# ---------------------------------------------------------------------------
# Solver-path-specific helpers (run in the worker)
# ---------------------------------------------------------------------------

def _build_sp_logc(eta_hat, k0_r1, k0_r2, alpha_r1, alpha_r2):
    from scripts._bv_common import (
        D_O2_HAT, D_H2O2_HAT, D_HP_HAT, C_O2_HAT, C_HP_HAT,
        A_DEFAULT, N_ELECTRONS,
        _make_nondim_cfg, _make_bv_convergence_cfg, SNES_OPTS_CHARGED,
    )
    from Forward.params import SolverParams

    params = dict(SNES_OPTS_CHARGED)
    params["bv_convergence"] = _make_bv_convergence_cfg()
    params["nondim"] = _make_nondim_cfg()
    reaction_1 = {
        "k0": k0_r1, "alpha": alpha_r1,
        "cathodic_species": 0, "anodic_species": 1, "c_ref": 1.0,
        "stoichiometry": [-1, +1, -2], "n_electrons": N_ELECTRONS,
        "reversible": True, "E_eq_v": E_EQ_R1,
        "cathodic_conc_factors": [{"species": 2, "power": 2, "c_ref_nondim": C_HP_HAT}],
    }
    reaction_2 = {
        "k0": k0_r2, "alpha": alpha_r2,
        "cathodic_species": 1, "anodic_species": None, "c_ref": 0.0,
        "stoichiometry": [0, -1, -2], "n_electrons": N_ELECTRONS,
        "reversible": False, "E_eq_v": E_EQ_R2,
        "cathodic_conc_factors": [{"species": 2, "power": 2, "c_ref_nondim": C_HP_HAT}],
    }
    params["bv_bc"] = {
        "reactions": [reaction_1, reaction_2],
        "k0": [k0_r1] * 3, "alpha": [alpha_r1] * 3,
        "stoichiometry": [-1, -1, -1], "c_ref": [1.0, 0.0, 1.0],
        "E_eq_v": 0.0,
        "electrode_marker": 3, "concentration_marker": 4, "ground_marker": 4,
    }
    return SolverParams.from_list([
        3, 1, 0.25, 80.0,
        [0, 0, 1], [D_O2_HAT, D_H2O2_HAT, D_HP_HAT], [A_DEFAULT] * 3,
        eta_hat, [C_O2_HAT, H2O2_SEED, C_HP_HAT], 0.0, params,
    ])


def _build_sp_4sp(eta_hat, k0_r1, k0_r2, alpha_r1, alpha_r2):
    from scripts._bv_common import (
        FOUR_SPECIES_CHARGED, make_bv_solver_params, SNES_OPTS_CHARGED,
    )
    return make_bv_solver_params(
        eta_hat=eta_hat, dt=0.25, t_end=80.0,
        species=FOUR_SPECIES_CHARGED, snes_opts=SNES_OPTS_CHARGED,
        k0_hat_r1=k0_r1, k0_hat_r2=k0_r2,
        alpha_r1=alpha_r1, alpha_r2=alpha_r2,
        E_eq_r1=E_EQ_R1, E_eq_r2=E_EQ_R2,
    )


def _add_boltzmann(ctx):
    """Add Boltzmann ClO4- background term to F_res (logc path only)."""
    import firedrake as fd
    from scripts._bv_common import C_CLO4_HAT
    mesh = ctx["mesh"]; W = ctx["W"]; U = ctx["U"]
    scaling = ctx["nondim"]
    phi = fd.split(U)[-1]
    w = fd.TestFunctions(W)[-1]
    dx = fd.Measure("dx", domain=mesh)
    charge_rhs = fd.Constant(float(scaling["charge_rhs_prefactor"]))
    c_bulk = fd.Constant(C_CLO4_HAT)
    phi_cl = fd.min_value(fd.max_value(phi, fd.Constant(-50.0)), fd.Constant(50.0))
    ctx["F_res"] -= charge_rhs * fd.Constant(-1.0) * c_bulk * fd.exp(phi_cl) * w * dx
    ctx["J_form"] = fd.derivative(ctx["F_res"], U)
    return ctx


def _build_ctx(V_RHE, k0_1, k0_2, alpha_1, alpha_2):
    """Build context for the path that handles this voltage.

    Returns (ctx, sp, solver_type) where solver_type is 'logc' or '4sp'.
    Caller is responsible for choosing annotation state.
    """
    from scripts._bv_common import V_T
    from Forward.bv_solver.forms_logc import (
        build_context_logc, build_forms_logc, set_initial_conditions_logc,
    )
    from Forward.bv_solver.forms import (
        build_context, build_forms, set_initial_conditions,
    )
    phi_hat = V_RHE / V_T

    if V_RHE >= V_THRESHOLD:
        sp = _build_sp_logc(phi_hat, k0_1, k0_2, alpha_1, alpha_2)
        ctx = build_context_logc(list(sp), mesh=_WORKER_MESH)
        ctx = build_forms_logc(ctx, list(sp))
        ctx = _add_boltzmann(ctx)
        set_initial_conditions_logc(ctx, list(sp))
        solver_type = "logc"
    else:
        sp = _build_sp_4sp(phi_hat, k0_1, k0_2, alpha_1, alpha_2)
        ctx = build_context(list(sp), mesh=_WORKER_MESH)
        ctx = build_forms(ctx, list(sp))
        set_initial_conditions(ctx, list(sp))
        solver_type = "4sp"
    return ctx, sp, solver_type


def _run_ss(sol, U, Up, dt_const, obs_form, max_steps=60, rel_tol=1e-4, abs_tol=1e-8):
    """Pseudo-transient continuation to steady state (SER adaptive dt)."""
    import firedrake as fd
    import firedrake.adjoint as adj
    dt_val = 0.25; dt_const.assign(dt_val)
    prev_flux = None; prev_delta = None; sc = 0
    for _ in range(max_steps):
        try:
            sol.solve()
        except Exception:
            return False
        with adj.stop_annotating():
            Up.assign(U)
        fv = float(fd.assemble(obs_form))
        if prev_flux is not None:
            d = abs(fv - prev_flux)
            sv = max(abs(fv), abs(prev_flux), 1e-8)
            if d / sv <= rel_tol or d <= abs_tol:
                sc += 1
            else:
                sc = 0
            if prev_delta and d > 0:
                r = prev_delta / d
                dt_val = (min(dt_val * min(r, 4), 5.0) if r > 1
                          else max(dt_val * 0.5, 0.25))
                dt_const.assign(dt_val)
            prev_delta = d
        prev_flux = fv
        if sc >= 4:
            return True
    return False


def _worker_solve(task: Tuple) -> Dict[str, Any]:
    """Solve one voltage. Returns dict with J, grad, new_U_arrays, converged.

    Task:
      (voltage_idx, V_RHE, log_k0_1, log_k0_2, alpha_1, alpha_2,
       cached_U_arrays,  # tuple of np arrays or None for cold start
       target_cd, target_pc, w_pc)
    """
    global _WORKER_MESH, _WORKER_CFG
    try:
        (vi, V_RHE, log_k0_1, log_k0_2, alpha_1, alpha_2,
         cached_U, target_cd, target_pc, w_pc) = task

        import firedrake as fd
        import firedrake.adjoint as adj
        from scripts._bv_common import I_SCALE, SNES_OPTS_CHARGED
        from Forward.bv_solver.observables import _build_bv_observable_form

        k0_1 = float(np.exp(log_k0_1))
        k0_2 = float(np.exp(log_k0_2))

        t0 = time.time()

        # Clear tape, enable annotation for context build.
        tape = adj.get_working_tape()
        tape.clear_tape()
        adj.continue_annotation()

        ctx, sp, solver_type = _build_ctx(V_RHE, k0_1, k0_2, alpha_1, alpha_2)

        U = ctx["U"]; Up = ctx["U_prev"]
        zc = ctx["z_consts"]; n = ctx["n_species"]
        dt_const = ctx["dt_const"]; paf = ctx["phi_applied_func"]
        from scripts._bv_common import V_T
        paf.assign(V_RHE / V_T)

        # Base solver options (from SNES_OPTS_CHARGED) with mandatory error-on-diverge.
        sp_base = {k: v for k, v in dict(SNES_OPTS_CHARGED).items()
                   if k.startswith(("snes_", "ksp_", "pc_", "mat_"))}
        sp_base["snes_error_if_not_converged"] = True
        # Cold-start solver: conservative line search for deep cathodic cold z-ramp.
        sp_dict_cold = dict(sp_base)
        sp_dict_cold["snes_linesearch_maxlambda"] = 0.3
        # Warm-start solver: default line search + moderate conservative linesearch.
        # NOTE: NO snes_lag_jacobian -- params change each BFGS iter, so the old
        # Jacobian is stale. Cost is small because warm-started solves only need
        # a handful of Newton iterations per step.
        sp_dict_warm = dict(sp_base)
        sp_dict_warm["snes_linesearch_maxlambda"] = 0.5

        prob = fd.NonlinearVariationalProblem(
            ctx["F_res"], U, bcs=ctx["bcs"], J=ctx["J_form"])
        # Pick solver parameters based on whether we have a warm start.
        _sp = sp_dict_warm if cached_U is not None else sp_dict_cold
        sol = fd.NonlinearVariationalSolver(prob, solver_parameters=_sp)
        form_cd = _build_bv_observable_form(
            ctx, mode="current_density", reaction_index=None, scale=-I_SCALE)
        form_pc = _build_bv_observable_form(
            ctx, mode="peroxide_current", reaction_index=None, scale=-I_SCALE)

        achieved_z = 0.0
        ok = False

        warm_fail_reason = None
        if cached_U is not None:
            # WARM START: load cached IC, assign z=1 directly, take a few steps.
            try:
                for src_arr, dst in zip(cached_U, U.dat):
                    dst.data[:] = np.asarray(src_arr)
                Up.assign(U)
                z_nominal = [float(sp[4][i]) for i in range(n)]
                for i in range(n):
                    zc[i].assign(z_nominal[i])
                with adj.stop_annotating():
                    # 40 steps at deep cathodic can be needed when params drift
                    # by 20% from the seeded state.
                    ok = _run_ss(sol, U, Up, dt_const, form_cd,
                                 max_steps=40, rel_tol=1e-4)
                if ok:
                    achieved_z = 1.0
                else:
                    warm_fail_reason = "warm_ss_did_not_converge"
            except Exception as _e:
                ok = False
                achieved_z = 0.0
                warm_fail_reason = f"warm_ss_exc:{type(_e).__name__}:{_e}"

        if not ok:
            # COLD START: rebuild solver with cold-start params, reset U to ICs, z=0 then z-ramp.
            # Rebuild the solver because sol was constructed with sp_dict_warm.
            sol = fd.NonlinearVariationalSolver(prob, solver_parameters=sp_dict_cold)
            # Reset U to manufactured initial conditions (since warm-start may
            # have left U in a bad state).
            if V_RHE >= V_THRESHOLD:
                from Forward.bv_solver.forms_logc import set_initial_conditions_logc
                with adj.stop_annotating():
                    set_initial_conditions_logc(ctx, list(sp))
            else:
                from Forward.bv_solver.forms import set_initial_conditions
                with adj.stop_annotating():
                    set_initial_conditions(ctx, list(sp))
            with adj.stop_annotating():
                for zci in zc:
                    zci.assign(0.0)
                if not _run_ss(sol, U, Up, dt_const, form_cd, max_steps=100):
                    return {
                        "vi": vi, "J": 1e6, "grad": np.zeros(4),
                        "new_U": None, "ok": False,
                        "reason": f"z=0 cold start failed (warm={warm_fail_reason})",
                        "elapsed": time.time() - t0, "solver": solver_type,
                    }
                z_nominal = [float(sp[4][i]) for i in range(n)]
                for z_val in np.linspace(0, 1, 21)[1:]:
                    U_ckpt = tuple(d.data_ro.copy() for d in U.dat)
                    for i in range(n):
                        zc[i].assign(z_nominal[i] * z_val)
                    if _run_ss(sol, U, Up, dt_const, form_cd, max_steps=60):
                        achieved_z = z_val
                    else:
                        for src, dst in zip(U_ckpt, U.dat):
                            dst.data[:] = src
                        Up.assign(U)
                        break
            if achieved_z < 1.0 - 1e-3:
                return {
                    "vi": vi, "J": 1e6, "grad": np.zeros(4),
                    "new_U": None, "ok": False,
                    "reason": f"z-ramp stalled at z={achieved_z:.3f}",
                    "elapsed": time.time() - t0, "solver": solver_type,
                }

        # A few annotated steps at z=1 so the tape contains the final state.
        for _ in range(3):
            sol.solve()
            with adj.stop_annotating():
                Up.assign(U)

        # Capture converged state for next-iter warm start.
        new_U_arrays = tuple(np.asarray(d.data_ro, dtype=float).copy() for d in U.dat)

        # Build objective symbolically so the adjoint tape captures it.
        cd_sym = fd.assemble(form_cd)
        pc_sym = fd.assemble(form_pc)
        obj_sym = 0.5 * (cd_sym - float(target_cd)) ** 2 \
                  + 0.5 * float(w_pc) * (pc_sym - float(target_pc)) ** 2

        # Adjoint gradient w.r.t. (k0_1, k0_2, alpha_1, alpha_2).
        k0_funcs = list(ctx["bv_k0_funcs"])[:2]
        alpha_funcs = list(ctx["bv_alpha_funcs"])[:2]
        controls = [adj.Control(f) for f in k0_funcs + alpha_funcs]
        rf = adj.ReducedFunctional(obj_sym, controls)
        try:
            grad_raw = rf.derivative()
        except Exception as exc:
            return {
                "vi": vi, "J": 1e6, "grad": np.zeros(4),
                "new_U": new_U_arrays, "ok": False,
                "reason": f"adjoint failed: {type(exc).__name__}: {exc}",
                "elapsed": time.time() - t0, "solver": solver_type,
            }

        def _extract(g):
            if hasattr(g, "dat"):
                return float(g.dat[0].data_ro[0])
            if hasattr(g, "values"):
                return float(g.values()[0])
            return float(g)

        g_vals = [_extract(g) for g in grad_raw]
        # Chain rule: d/dlog(k0) = k0 * d/dk0
        grad_theta = np.array([
            k0_1 * g_vals[0],
            k0_2 * g_vals[1],
            g_vals[2],
            g_vals[3],
        ], dtype=float)

        return {
            "vi": vi, "J": float(obj_sym), "grad": grad_theta,
            "new_U": new_U_arrays, "ok": True, "reason": "",
            "cd_sim": float(cd_sym), "pc_sim": float(pc_sym),
            "elapsed": time.time() - t0, "solver": solver_type,
        }
    except Exception:
        tb = traceback.format_exc()
        return {
            "vi": task[0] if task else -1,
            "J": 1e6, "grad": np.zeros(4),
            "new_U": None, "ok": False,
            "reason": f"WORKER EXCEPTION:\n{tb}",
            "elapsed": 0.0, "solver": "?",
        }


# ---------------------------------------------------------------------------
# Main-process driver
# ---------------------------------------------------------------------------

class HybridInferenceDriver:
    def __init__(self, V_grid, target_cd, target_pc, noisy_cd, noisy_pc,
                 w_pc_mask, n_workers: int, verbose: bool = True):
        self.V_grid = V_grid
        self.target_cd = target_cd
        self.target_pc = target_pc
        self.noisy_cd = noisy_cd
        self.noisy_pc = noisy_pc
        self.w_pc_mask = w_pc_mask
        self.n_workers = n_workers
        self.verbose = verbose

        self.ic_cache: Dict[int, Tuple[np.ndarray, ...]] = {}
        self.eval_count = 0

        cfg = WorkerCfg(
            mesh_nx=MESH_NX, mesh_ny=MESH_NY, mesh_beta=MESH_BETA,
            w_pc_threshold=V_THRESHOLD,
        )
        if n_workers > 0:
            ctx = mp.get_context("spawn")
            self.pool = ProcessPoolExecutor(
                max_workers=n_workers, mp_context=ctx,
                initializer=_worker_init, initargs=(cfg,),
            )
            print(f"[driver] ProcessPool with {n_workers} workers (spawn)")
        else:
            self.pool = None
            _worker_init(cfg)  # initialize main process for in-line calls
            print("[driver] Sequential (no pool)")

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True)
            self.pool = None

    def _dispatch(self, theta: np.ndarray) -> List[Dict[str, Any]]:
        """Dispatch N voltage tasks and collect results."""
        tasks = []
        for i, V in enumerate(self.V_grid):
            tasks.append((
                int(i), float(V),
                float(theta[0]), float(theta[1]),
                float(theta[2]), float(theta[3]),
                self.ic_cache.get(i),
                float(self.noisy_cd[i]),
                float(self.noisy_pc[i]),
                float(self.w_pc_mask[i]),
            ))

        results: List[Optional[Dict[str, Any]]] = [None] * len(tasks)
        if self.pool is not None:
            future_map = {self.pool.submit(_worker_solve, t): t[0] for t in tasks}
            for fut in as_completed(future_map):
                res = fut.result()
                results[int(res["vi"])] = res
        else:
            for t in tasks:
                res = _worker_solve(t)
                results[int(res["vi"])] = res
        return [r for r in results if r is not None]

    def eval_obj_grad(self, theta: np.ndarray) -> Tuple[float, np.ndarray]:
        """Single scipy callback: returns (J_total, grad_total)."""
        self.eval_count += 1
        t0 = time.time()
        results = self._dispatch(theta)

        J_total = 0.0
        grad_total = np.zeros(4)
        n_ok = 0; n_fail = 0
        per_V_log = []

        for res in results:
            vi = int(res["vi"])
            if res["ok"] and res["new_U"] is not None:
                self.ic_cache[vi] = res["new_U"]
                n_ok += 1
                J_total += res["J"]
                grad_total += res["grad"]
                if "cd_sim" in res:
                    per_V_log.append(
                        f"V={self.V_grid[vi]:+.3f}({res['solver']}): "
                        f"J={res['J']:.3e} cd={res['cd_sim']:+.5f} "
                        f"t={res['elapsed']:.1f}s"
                    )
            else:
                n_fail += 1
                J_total += res["J"]
                per_V_log.append(
                    f"V={self.V_grid[vi]:+.3f}({res.get('solver','?')}) FAIL: {res.get('reason','?')}"
                )

        wall = time.time() - t0
        if self.verbose:
            print(f"\n[eval {self.eval_count}] theta=[{theta[0]:+.4f}, "
                  f"{theta[1]:+.4f}, {theta[2]:+.4f}, {theta[3]:+.4f}]")
            print(f"  k0_1={np.exp(theta[0]):.4e}, k0_2={np.exp(theta[1]):.4e}, "
                  f"alpha_1={theta[2]:.4f}, alpha_2={theta[3]:.4f}")
            print(f"  J_total={J_total:.6e}  "
                  f"||grad||={np.linalg.norm(grad_total):.3e}  "
                  f"ok={n_ok}/{n_ok+n_fail}  wall={wall:.1f}s")
            if self.eval_count <= 3:
                for line in per_V_log:
                    print(f"    {line}")
        return J_total, grad_total


# ---------------------------------------------------------------------------
# Target loading / noise / reporting
# ---------------------------------------------------------------------------

def _load_target(extended: bool = True):
    """Load the I-V target. Prefer the extended (v-chain + onset) target if available.

    Returns (data_dict, U_seed_per_V_or_None).
    """
    if extended:
        ext_npz = os.path.join(_ROOT, "StudyResults", "v19_extended", "target.npz")
        ext_pkl = os.path.join(_ROOT, "StudyResults", "v19_extended", "U_seed.pkl")
        if os.path.exists(ext_npz):
            data = np.load(ext_npz, allow_pickle=True)
            u_seed = None
            if os.path.exists(ext_pkl):
                with open(ext_pkl, "rb") as f:
                    u_seed = pickle.load(f)
            return data, u_seed, "extended"

    npz_path = os.path.join(_ROOT, "StudyResults", "v19_hybrid", "iv_target.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"Target not found: {npz_path}\n"
            "Run scripts/studies/v19_hybrid_forward.py or v19_extended_target.py.")
    data = np.load(npz_path, allow_pickle=True)
    return data, None, "basic"


def _report(best_theta, k0_1_true, k0_2_true, alpha_1_true, alpha_2_true,
            init_theta, J_final, n_evals, elapsed):
    k0_1_rec = np.exp(best_theta[0])
    k0_2_rec = np.exp(best_theta[1])
    a1_rec, a2_rec = best_theta[2], best_theta[3]

    k0_1_err = 100 * (k0_1_rec - k0_1_true) / k0_1_true
    k0_2_err = 100 * (k0_2_rec - k0_2_true) / k0_2_true
    a1_err = 100 * (a1_rec - alpha_1_true) / alpha_1_true
    a2_err = 100 * (a2_rec - alpha_2_true) / alpha_2_true

    print("\n" + "=" * 70)
    print("RECOVERY RESULTS (v13 CD+PC objective, hybrid forward, BFGS+adjoint)")
    print("=" * 70)
    print(f"  {'Param':>10} {'True':>14} {'Initial':>14} {'Recovered':>14} {'Err':>10}")
    print("-" * 70)
    init_k0_1 = np.exp(init_theta[0]); init_k0_2 = np.exp(init_theta[1])
    print(f"  {'k0_1':>10} {k0_1_true:>14.6e} {init_k0_1:>14.6e} "
          f"{k0_1_rec:>14.6e} {k0_1_err:>+9.2f}%")
    print(f"  {'k0_2':>10} {k0_2_true:>14.6e} {init_k0_2:>14.6e} "
          f"{k0_2_rec:>14.6e} {k0_2_err:>+9.2f}%")
    print(f"  {'alpha_1':>10} {alpha_1_true:>14.6f} {init_theta[2]:>14.6f} "
          f"{a1_rec:>14.6f} {a1_err:>+9.2f}%")
    print(f"  {'alpha_2':>10} {alpha_2_true:>14.6f} {init_theta[3]:>14.6f} "
          f"{a2_rec:>14.6f} {a2_err:>+9.2f}%")
    print("-" * 70)
    print(f"  Final J: {J_final:.6e}")
    print(f"  Function evals: {n_evals}")
    print(f"  Total wall time: {elapsed:.1f}s ({elapsed/60:.1f} min)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=-1,
                        help="Parallel workers (default: min(cpu_count-1, n_voltages))")
    parser.add_argument("--fd-grad", action="store_true",
                        help="Use scipy finite-difference grad (debug; slow)")
    parser.add_argument("--maxiter", type=int, default=MAXITER)
    parser.add_argument("--noise-seed", type=int, default=NOISE_SEED)
    parser.add_argument("--noise-pct", type=float, default=NOISE_PCT,
                        help="Gaussian noise std (%% of |target|). Set 0 for exact data.")
    args = parser.parse_args()

    from scripts._bv_common import setup_firedrake_env
    setup_firedrake_env()

    # ---- Load target I-V (prefer extended v-chain + onset if present) ----
    data, u_seed, target_kind = _load_target(extended=True)
    V_grid = np.asarray(data["V_RHE"])
    target_cd = np.asarray(data["cd"])
    target_pc = np.asarray(data["pc"])
    solver_used = np.asarray(data["solver_used"])
    print(f"[target] kind={target_kind}, {len(V_grid)} voltages")
    # Drop any NaN points (e.g. V=-0.5, -0.4 that 4sp could not converge)
    finite = np.isfinite(target_cd) & np.isfinite(target_pc)
    V_grid = V_grid[finite]
    target_cd = target_cd[finite]
    target_pc = target_pc[finite]
    solver_used = solver_used[finite]
    n_V = len(V_grid)

    k0_1_true = float(data["k0_r1"]); k0_2_true = float(data["k0_r2"])
    alpha_1_true = float(data["alpha_r1"]); alpha_2_true = float(data["alpha_r2"])

    rng = np.random.default_rng(args.noise_seed)
    noise_pct = float(args.noise_pct)
    if noise_pct > 0:
        noisy_cd = target_cd + rng.normal(0, noise_pct / 100, n_V) * np.abs(target_cd)
        noisy_pc = target_pc + rng.normal(0, noise_pct / 100, n_V) * np.abs(target_pc)
    else:
        noisy_cd = target_cd.copy()
        noisy_pc = target_pc.copy()

    # PC mask: 1 only where solver_used == '4sp' (physical PC); 0 on logc.
    w_pc_mask = np.array([1.0 if s == "4sp" else 0.0 for s in solver_used])
    n_pc_pts = int(w_pc_mask.sum())

    print("=" * 70)
    print("v19 HYBRID INFERENCE — CD + masked-PC with BFGS + pyadjoint")
    print("=" * 70)
    print(f"  Voltages: {n_V} ({sum(solver_used=='4sp')} 4sp, {sum(solver_used=='logc')} logc)")
    print(f"  Observables: CD on all {n_V}, PC on {n_pc_pts} 4sp points")
    print(f"  Noise: {noise_pct}% Gaussian (seed {args.noise_seed})")
    print(f"  True: k0_1={k0_1_true:.4e}, k0_2={k0_2_true:.4e}, "
          f"α1={alpha_1_true:.4f}, α2={alpha_2_true:.4f}")

    # Initial guess: +20% k0, -10% alpha (all 4 params)
    init_theta = np.array([
        np.log(k0_1_true * INIT_OFFSET_K0),
        np.log(k0_2_true * INIT_OFFSET_K0),
        alpha_1_true * INIT_OFFSET_ALPHA,
        alpha_2_true * INIT_OFFSET_ALPHA,
    ])
    print(f"  Init: k0_1={np.exp(init_theta[0]):.4e}, k0_2={np.exp(init_theta[1]):.4e}, "
          f"α1={init_theta[2]:.4f}, α2={init_theta[3]:.4f}")

    # Parallel pool sizing
    n_workers = args.workers
    if n_workers < 0:
        n_workers = min(n_V, max(1, (os.cpu_count() or 4) - 1))
    print(f"  Workers: {n_workers}")

    driver = HybridInferenceDriver(
        V_grid=V_grid, target_cd=target_cd, target_pc=target_pc,
        noisy_cd=noisy_cd, noisy_pc=noisy_pc, w_pc_mask=w_pc_mask,
        n_workers=n_workers,
    )

    # Pre-seed IC cache from target's converged U arrays, if available.
    # This lets every BFGS iter (including the first) warm-start at the
    # cathodic voltages where cold z=0 / z-ramp would diverge.
    if u_seed is not None:
        seed_V = u_seed.get("V_RHE", [])
        seed_U = u_seed.get("U_per_V", [])
        if len(seed_V) == len(seed_U) and len(seed_V) >= n_V:
            # map V_RHE back to voltage index in V_grid
            V_to_idx = {float(v): i for i, v in enumerate(V_grid.tolist())}
            seeded = 0
            for V_s, U_s in zip(seed_V, seed_U):
                key = float(V_s)
                if key in V_to_idx:
                    driver.ic_cache[V_to_idx[key]] = tuple(np.asarray(a) for a in U_s)
                    seeded += 1
            print(f"[driver] Pre-seeded IC cache with {seeded}/{n_V} voltages from target.")

    # ---- L-BFGS-B ----
    from scipy.optimize import minimize
    bounds = [
        (np.log(1e-6), np.log(1e0)),   # log_k0_1
        (np.log(1e-9), np.log(1e-2)),  # log_k0_2
        (0.1, 0.9),                    # alpha_1
        (0.1, 0.9),                    # alpha_2
    ]

    t_total = time.time()
    try:
        if args.fd_grad:
            # FD fallback
            res = minimize(
                lambda th: driver.eval_obj_grad(th)[0],
                init_theta, method="L-BFGS-B", bounds=bounds,
                options={"maxiter": args.maxiter, "disp": True, "ftol": 1e-10, "gtol": 1e-7},
            )
        else:
            res = minimize(
                driver.eval_obj_grad, init_theta, jac=True,
                method="L-BFGS-B", bounds=bounds,
                options={"maxiter": args.maxiter, "disp": True, "ftol": 1e-10, "gtol": 1e-7},
            )
    finally:
        driver.close()
    elapsed = time.time() - t_total

    _report(
        res.x, k0_1_true, k0_2_true, alpha_1_true, alpha_2_true,
        init_theta, float(res.fun), driver.eval_count, elapsed,
    )

    # Save
    out_dir = os.path.join(_ROOT, "StudyResults", "v19_hybrid_inference")
    os.makedirs(out_dir, exist_ok=True)
    out = {
        "theta_final": res.x.tolist(),
        "theta_init": init_theta.tolist(),
        "J_final": float(res.fun),
        "n_evals": driver.eval_count,
        "elapsed_s": elapsed,
        "true": {"k0_1": k0_1_true, "k0_2": k0_2_true,
                 "alpha_1": alpha_1_true, "alpha_2": alpha_2_true},
        "V_grid": V_grid.tolist(),
        "w_pc_mask": w_pc_mask.tolist(),
        "noise_seed": args.noise_seed, "noise_pct": NOISE_PCT,
    }
    import json
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_dir}/result.json")


if __name__ == "__main__":
    main()
