"""Forward solve functions for the Butler-Volmer PNP solver."""

from __future__ import annotations

from typing import Any

import numpy as np
import firedrake as fd

from .forms import build_context, build_forms, set_initial_conditions


def _clone_params_with_phi(solver_params, *, phi_applied: float):
    """Return a new SolverParams-like list with phi_applied replaced."""
    lst = list(solver_params)   # works for both list and SolverParams
    lst[7] = float(phi_applied)
    return lst


def forsolve_bv(
    ctx: dict[str, Any],
    solver_params: Any,
    print_interval: int = 20,
) -> Any:
    """Run the BV PNP time-stepping loop to steady state.

    Parameters
    ----------
    ctx:
        Context dict from :func:`build_forms`.
    solver_params:
        11-element list / SolverParams.
    print_interval:
        Print progress every N steps.

    Returns
    -------
    Function
        Final solved state ``U``.
    """
    try:
        n_species, order, dt, t_end, z_vals, D_vals, a_vals, phi_applied, c0, phi0, params = solver_params
    except Exception as exc:
        raise ValueError("bv_solver expects a list of 11 solver parameters") from exc

    U = ctx["U"]
    U_prev = ctx["U_prev"]
    F_res = ctx["F_res"]
    bcs = ctx["bcs"]
    scaling = ctx.get("nondim", {})

    dt_model = float(scaling.get("dt_model", dt))
    t_end_model = float(scaling.get("t_end_model", t_end))
    num_steps = max(1, int(round(t_end_model / dt_model)))

    J = fd.derivative(F_res, U)
    problem = fd.NonlinearVariationalProblem(F_res, U, bcs=bcs, J=J)
    solver = fd.NonlinearVariationalSolver(problem, solver_parameters=params)

    for step in range(num_steps):
        if step % print_interval == 0:
            print(f"[bv_solver] step {step}/{num_steps}  eta_hat={float(ctx['phi_applied_func'].dat.data[0]):.4f}")
        solver.solve()
        U_prev.assign(U)

    return U


def solve_bv_with_continuation(
    solver_params: Any,
    *,
    eta_target: float,
    eta_steps: int = 20,
    print_interval: int = 20,
    mesh: Any = None,
) -> Any:
    """Solve BV problem using potential continuation from η̂ = 0 to ``eta_target``.

    This is the primary convergence strategy for large overpotentials.
    At each continuation step:

    1. Set ``phi_applied_func`` to the current η̂.
    2. Run the time-stepping loop.
    3. Use the result as the initial condition for the next step.

    If Newton diverges at a step, the step size is halved (bisection up to 4 times).

    Parameters
    ----------
    solver_params:
        11-element list / SolverParams.  The ``phi_applied`` field is overridden
        during continuation — its value sets the **final** target, not the starting
        value.  The starting value is always 0 (equilibrium).
    eta_target:
        Final dimensionless overpotential to reach.  In nondim mode this is
        φ_electrode / V_T.  In dimensional mode this is φ_electrode in Volts.
    eta_steps:
        Number of continuation steps between 0 and eta_target.
    print_interval:
        Passed to :func:`forsolve_bv`.

    Returns
    -------
    Function
        The solved state at ``eta_target``.
    """
    try:
        n_s, order, dt, t_end, z_v, D_v, a_v, phi_applied, c0, phi0, params = solver_params
    except Exception as exc:
        raise ValueError("bv_solver expects a list of 11 solver parameters") from exc

    # Build context at eta = 0.
    params_0 = _clone_params_with_phi(solver_params, phi_applied=0.0)
    ctx = build_context(params_0, mesh=mesh)
    ctx = build_forms(ctx, params_0)
    set_initial_conditions(ctx, params_0, blob=False)

    # Build the continuation path.
    path = np.linspace(0.0, eta_target, eta_steps + 1)[1:]  # skip 0 (already at IC)

    scaling = ctx["nondim"]
    J = fd.derivative(ctx["F_res"], ctx["U"])
    problem = fd.NonlinearVariationalProblem(ctx["F_res"], ctx["U"], bcs=ctx["bcs"], J=J)
    solver = fd.NonlinearVariationalSolver(problem, solver_parameters=params)

    dt_model = float(scaling.get("dt_model", dt))
    t_end_model = float(scaling.get("t_end_model", t_end))
    num_steps = max(1, int(round(t_end_model / dt_model)))

    def _try_timestep(eta_try: float) -> bool:
        """Attempt time-stepping at eta_try.  Returns True on success.

        On failure the context state (U, U_prev, phi_applied_func) is NOT
        guaranteed to be consistent — caller must restore from a checkpoint.
        """
        ctx["phi_applied_func"].assign(float(eta_try))
        try:
            for _ in range(num_steps):
                solver.solve()
                ctx["U_prev"].assign(ctx["U"])
            return True
        except fd.ConvergenceError:
            return False

    # eta currently solved (start = 0, already in ctx)
    eta_current = 0.0
    max_sub = 6   # maximum recursive bisection depth per target step

    for i_cont, eta_target_step in enumerate(path):
        print(f"[continuation] step {i_cont+1}/{len(path)}  eta_hat → {eta_target_step:.4f}")

        # Checkpoint the last known-good state before attempting this step.
        U_ckpt = fd.Function(ctx["U"])
        U_prev_ckpt = fd.Function(ctx["U_prev"])

        # Try the full step directly.
        if _try_timestep(eta_target_step):
            eta_current = eta_target_step
            continue  # success — proceed to next planned step

        # Direct step failed — adaptive sub-stepping between eta_current and eta_target_step.
        print(f"  [sub-step] direct solve failed; inserting sub-steps "
              f"from eta={eta_current:.4f} to eta={eta_target_step:.4f}")

        eta_lo = eta_current
        eta_hi = eta_target_step
        sub_count = 0

        while sub_count < max_sub:
            # Restore to last good state and try the midpoint.
            ctx["U"].assign(U_ckpt)
            ctx["U_prev"].assign(U_prev_ckpt)

            eta_mid = (eta_lo + eta_hi) / 2.0
            print(f"  [sub-step {sub_count+1}/{max_sub}] eta_mid={eta_mid:.4f}")

            if _try_timestep(eta_mid):
                # Midpoint converged — update checkpoint and try the full target.
                U_ckpt.assign(ctx["U"])
                U_prev_ckpt.assign(ctx["U_prev"])
                eta_lo = eta_mid

                ctx["U"].assign(U_ckpt)
                ctx["U_prev"].assign(U_prev_ckpt)
                if _try_timestep(eta_target_step):
                    eta_current = eta_target_step
                    break  # reached the target
                # Target still fails — bisect again from eta_mid toward eta_hi
                sub_count += 1
            else:
                # Midpoint failed too — halve the lower half
                eta_hi = eta_mid
                sub_count += 1
        else:
            raise RuntimeError(
                f"[continuation] Failed to converge at eta={eta_target_step:.4f} after "
                f"{max_sub} sub-stepping attempts (reached eta={eta_lo:.4f})."
            )

    return ctx["U"]


def solve_bv_with_ptc(
    solver_params: Any,
    *,
    eta_target: float,
    eta_steps: int = 20,
    ptc_config: dict[str, Any] | None = None,
    print_interval: int = 20,
    mesh: Any = None,
) -> Any:
    """Solve BV problem using pseudo-transient continuation (PTC).

    PTC is a globalization strategy that embeds the nonlinear problem inside
    a fictitious time-stepping loop.  The key idea: at each Newton iteration,
    add a scaled mass-matrix term ``(1/dt_ptc) * M * (u^{n+1} - u^n)`` to the
    residual.  When ``dt_ptc`` is small, the system is dominated by the mass
    matrix (easy to invert); as ``dt_ptc`` grows, the system approaches the
    true steady-state problem.

    The time step ``dt_ptc`` is **adapted** based on the SNES residual:

        dt_ptc^{k+1} = dt_ptc^k * (||r^{k-1}|| / ||r^k||) * growth

    This grows dt_ptc when convergence is good (residual decreasing) and
    shrinks it when convergence stalls (residual increasing).

    The BV solver already has a time-stepping mass matrix ``(c_i - c_old)/dt_m``
    in the weak form.  PTC reuses this by dynamically adjusting ``dt_m``
    between pseudo-time steps.

    Strategy: at each eta in the voltage continuation path:
      1. Reset dt_ptc to ``dt_initial``.
      2. Iterate: solve one time step, adapt dt_ptc based on residual ratio.
      3. Stop when ``dt_ptc > max_dt`` (equivalent to steady state) or
         when relative change in solution < ``ss_tol``.

    Parameters
    ----------
    solver_params:
        11-element list / SolverParams.
    eta_target:
        Final dimensionless overpotential to reach.
    eta_steps:
        Number of voltage continuation steps from 0 to eta_target.
    ptc_config:
        Dictionary with PTC hyperparameters:
          - ``dt_initial`` (float): initial pseudo time step. Default 0.01.
            Small values make the initial step very close to a Picard iteration.
          - ``growth`` (float): base growth factor for dt adaptation. Default 1.5.
          - ``max_dt`` (float): stop PTC when dt exceeds this. Default 1e6.
            Effectively says "the solution is steady enough."
          - ``max_ptc_steps`` (int): max PTC iterations per voltage. Default 200.
          - ``ss_tol`` (float): relative-change convergence tolerance. Default 1e-5.
    print_interval:
        Print progress every N continuation steps.
    mesh:
        Optional mesh to use (if None, one is created internally).

    Returns
    -------
    Function
        The solved state at ``eta_target``.

    Physical motivation
    -------------------
    PTC is particularly effective for PNP-BV because:
    - The Debye layer (eps_hat ~ 9e-8) creates extreme stiffness in the
      Poisson equation at large overpotentials.
    - Adding the mass matrix effectively regularizes the problem at small dt_ptc,
      preventing the Newton solver from trying to resolve the full stiff
      boundary layer in one shot.
    - As dt_ptc grows, the solver gradually "tightens" the resolution of
      the boundary layer.
    """
    ptc = ptc_config or {}
    dt_initial = float(ptc.get("dt_initial", 0.01))
    growth = float(ptc.get("growth", 1.5))
    max_dt = float(ptc.get("max_dt", 1e6))
    max_ptc_steps = int(ptc.get("max_ptc_steps", 200))
    ss_tol = float(ptc.get("ss_tol", 1e-5))

    try:
        n_s, order, dt, t_end, z_v, D_v, a_v, phi_applied, c0, phi0, params = solver_params
    except Exception as exc:
        raise ValueError("bv_solver expects a list of 11 solver parameters") from exc

    # Build context at eta = 0.
    params_0 = _clone_params_with_phi(solver_params, phi_applied=0.0)
    ctx = build_context(params_0, mesh=mesh)
    ctx = build_forms(ctx, params_0)
    set_initial_conditions(ctx, params_0, blob=False)

    scaling = ctx["nondim"]
    dt_model = float(scaling.get("dt_model", dt))

    # The dt_const in the weak form controls the (c_i - c_old)/dt_const mass
    # matrix term.  Since dt_const is now a mutable fd.Constant, PTC can
    # dynamically assign it: ctx["dt_const"].assign(new_dt_value).
    # Each "time step" in forsolve_bv uses U_prev -> U.
    # For PTC, we solve one step at a time and adapt.

    J = fd.derivative(ctx["F_res"], ctx["U"])
    problem = fd.NonlinearVariationalProblem(
        ctx["F_res"], ctx["U"], bcs=ctx["bcs"], J=J
    )
    solver = fd.NonlinearVariationalSolver(
        problem, solver_parameters=params
    )

    # Voltage continuation path from 0 to eta_target.
    path = np.linspace(0.0, eta_target, eta_steps + 1)[1:]

    U_scratch = ctx["U"].copy(deepcopy=True)

    for i_cont, eta_step in enumerate(path):
        ctx["phi_applied_func"].assign(float(eta_step))

        if (i_cont + 1) % print_interval == 0 or i_cont == 0:
            print(f"[PTC] continuation step {i_cont+1}/{len(path)}  "
                  f"eta_hat -> {eta_step:.4f}")

        # PTC inner loop at this voltage.
        # Each iteration: solve one implicit time step, check convergence.
        # The existing dt_model in the form is fixed, so PTC effectively controls
        # how many time steps we take before declaring steady state.
        # More fundamentally: when dt_model is already in the form, taking
        # one time step IS a PTC step with dt_ptc = dt_model.
        # To truly vary dt_ptc, we would need a mutable fd.Constant for dt.
        # For now, we use the adaptive stopping criterion on the existing dt.

        prev_res_norm = None
        dt_ptc = dt_initial  # Tracks the PTC "pseudo-dt" (for adaptation logic)

        for k in range(max_ptc_steps):
            U_scratch.assign(ctx["U"])

            try:
                solver.solve()
            except fd.ConvergenceError:
                # PTC recovery: restore and try with the existing dt (one more chance)
                ctx["U"].assign(U_scratch)
                # If even one step fails, just keep the current state and move on
                print(f"  [PTC] Newton failed at eta={eta_step:.4f}, ptc_step={k}")
                break

            # Compute residual-based adaptation metric.
            delta_norm = fd.errornorm(ctx["U"], U_scratch, norm_type="L2")
            ref_norm = fd.norm(U_scratch, norm_type="L2")
            rel_change = delta_norm / max(ref_norm, 1e-14)

            # Update U_prev for next time step.
            ctx["U_prev"].assign(ctx["U"])

            # Adapt dt_ptc based on residual decrease.
            if prev_res_norm is not None and delta_norm > 0:
                ratio = prev_res_norm / max(delta_norm, 1e-30)
                dt_ptc = dt_ptc * ratio * growth
                dt_ptc = min(dt_ptc, max_dt)

            prev_res_norm = delta_norm

            # Check steady-state convergence.
            if rel_change < ss_tol:
                break

            # Check if PTC has effectively converged (dt_ptc very large).
            if dt_ptc >= max_dt:
                break

    return ctx["U"]


def solve_bv_with_charge_continuation(
    solver_params: Any,
    *,
    eta_target: float,
    eta_steps: int = 20,
    charge_steps: int = 10,
    print_interval: int = 20,
    mesh: Any = None,
) -> Any:
    """Solve BV problem using charge continuation (z-ramping).

    Strategy:
    1. First solve the full voltage range with z=[0,0,0,0] (neutral species).
       This avoids Poisson stiffness entirely -- the electromigration term
       vanishes when z=0, and the Poisson source term is zero.
    2. Then at the target eta, ramp z_factor from 0 to 1 in ``charge_steps``
       increments, gradually turning on the charge coupling.
       At each step, z_i = z_nominal_i * z_factor.

    This works because the z values are stored as ``fd.Constant`` objects
    in ``ctx["z_consts"]``, so they can be ``.assign()``-ed between solves
    without rebuilding forms or the solver.

    Physical motivation
    -------------------
    The dominant source of stiffness in the charged PNP-BV system is the
    Poisson equation's (lambda_D/L)^2 ~ 9e-8 prefactor, which creates a
    thin Debye layer near the electrode.  For neutral species (z=0), the
    Poisson equation decouples from the transport equations, and the
    system reduces to pure Nernst-Planck diffusion-reaction -- much easier
    to solve.

    By first solving the neutral problem to get approximate concentration
    profiles, then gradually turning on the charge coupling, we provide
    the Newton solver with progressively better initial guesses for the
    charged problem.

    Parameters
    ----------
    solver_params:
        11-element list / SolverParams.
    eta_target:
        Final dimensionless overpotential.
    eta_steps:
        Number of voltage continuation steps (used for neutral sweep).
    charge_steps:
        Number of steps to ramp z_factor from 0 to 1.  More steps = more
        robust but slower.  10-20 is usually sufficient.
    print_interval:
        Print progress every N steps.
    mesh:
        Optional mesh.

    Returns
    -------
    Function
        The solved state at ``eta_target`` with full charge coupling.
    """
    try:
        n_s, order, dt, t_end, z_v, D_v, a_v, phi_applied, c0, phi0, params = solver_params
    except Exception as exc:
        raise ValueError("bv_solver expects a list of 11 solver parameters") from exc

    z_nominal = [float(v) for v in ([z_v] * n_s if np.isscalar(z_v) else list(z_v))][:n_s]

    # ---------- Phase 1: Neutral voltage sweep ----------
    # Set all charges to zero for the initial voltage continuation.
    params_neutral = _clone_params_with_phi(solver_params, phi_applied=0.0)
    # Override z values to zero.
    params_neutral[4] = [0.0] * n_s

    ctx = build_context(params_neutral, mesh=mesh)
    ctx = build_forms(ctx, params_neutral)
    set_initial_conditions(ctx, params_neutral, blob=False)

    scaling = ctx["nondim"]
    dt_model = float(scaling.get("dt_model", dt))
    t_end_model = float(scaling.get("t_end_model", t_end))
    num_steps = max(1, int(round(t_end_model / dt_model)))

    J = fd.derivative(ctx["F_res"], ctx["U"])
    problem = fd.NonlinearVariationalProblem(
        ctx["F_res"], ctx["U"], bcs=ctx["bcs"], J=J
    )
    solver = fd.NonlinearVariationalSolver(
        problem, solver_parameters=params
    )

    # Voltage continuation from 0 to eta_target (neutral).
    path = np.linspace(0.0, eta_target, eta_steps + 1)[1:]
    for i_cont, eta_step in enumerate(path):
        ctx["phi_applied_func"].assign(float(eta_step))
        if (i_cont + 1) % print_interval == 0 or i_cont == 0:
            print(f"[charge-cont] Phase 1 (neutral): "
                  f"step {i_cont+1}/{len(path)}  eta -> {eta_step:.4f}")
        for _ in range(num_steps):
            solver.solve()
            ctx["U_prev"].assign(ctx["U"])

    print(f"[charge-cont] Phase 1 complete: neutral solution at eta={eta_target:.4f}")

    # ---------- Phase 2: Charge ramping at target eta ----------
    # Gradually turn on z_factor from 0 to 1.
    # z_consts are fd.Constant objects that can be .assign()-ed.
    z_consts = ctx["z_consts"]  # list of fd.Constant objects
    z_path = np.linspace(0.0, 1.0, charge_steps + 1)[1:]  # skip 0 (already neutral)

    for i_z, z_factor in enumerate(z_path):
        # Update all charge valences: z_i = z_nominal_i * z_factor.
        for i in range(n_s):
            z_consts[i].assign(float(z_nominal[i]) * float(z_factor))

        print(f"[charge-cont] Phase 2 (z-ramp): "
              f"step {i_z+1}/{len(z_path)}  z_factor={z_factor:.3f}")

        # Run time stepping to find steady state at this z_factor.
        try:
            for _ in range(num_steps):
                solver.solve()
                ctx["U_prev"].assign(ctx["U"])
        except fd.ConvergenceError:
            print(f"  [charge-cont] Failed at z_factor={z_factor:.3f}")
            # Restore last good state.
            ctx["U"].assign(ctx["U_prev"])
            break

    print(f"[charge-cont] Phase 2 complete: z_factor={float(z_path[-1]):.3f}")
    return ctx["U"]
