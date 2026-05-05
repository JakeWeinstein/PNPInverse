"""Anodic-biased cold solve at V=+1.0 V, then walk down toward E_eq_R1.

Direct test of "Option 1" from `docs/clip_observable_investigation.md`
§5.6: see whether the peroxide window V_RHE > +0.68 V is accessible at
all if we feed the solver an initial condition biased toward the
anodic-R1 basin (c_H2O2 heavily depleted at the electrode), instead of
cold-ramping from bulk like the production C+D orchestrator does.

Strategy:

  1. At V_RHE = +1.0 V, manually set
       u_O2(x,y) = 0                                # bulk (= ln c0_O2)
       u_H2O2(x,y) linear from -9.21 at y=1 (bulk BC)
                          to -25 at y=0 (heavily depleted at electrode)
       u_H(x,y) = ln(0.2) = -1.61                   # bulk
       phi(x,y) = eta_hat * (1 - y)                 # linear ramp
     and run Newton at SS (large dt) without z-ramp. If it converges,
     the SS basin is reachable from this IC.

  2. Walk *down* in V from the converged V=+1.0 snapshot:
        +0.95, +0.90, +0.85, +0.80, +0.75, +0.72, +0.70, +0.69,
        +0.685, +0.68
     Each step uses the previous SS as warm IC; walks toward E_eq_R1
     from the anodic side.

  3. At each converged V, capture CD/PC + surface concentrations.

  4. Plot the down-walk alongside the cathodic-side data already in
     StudyResults/peroxide_window_extension/.

Output: StudyResults/anodic_cold_start/.

Run from PNPInverse/ with `../venv-firedrake/bin/activate` active.
"""
from __future__ import annotations

import json
import os
import sys
import time
from math import log

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(line_buffering=True)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np


# ---------------------------------------------------------------------------
# Test plan
# ---------------------------------------------------------------------------
V_START = 1.00          # initial anodic-cold target
V_DOWN = [
    0.95, 0.90, 0.85, 0.80, 0.75, 0.72, 0.70,
    0.69, 0.685, 0.68,  # the wall from the cathodic side
    0.66, 0.64, 0.62,   # below the wall — should overlap with cathodic-side data
]

EXPONENT_CLIP = 100.0
MESH_NY = 200
U_CLAMP = 200.0   # widened past default 30 / MMS 100 to let the SS
                  # u_H+ depletion (~ -40) and u_H2O2 depletion settle
                  # without the bulk PDE coefficient being clipped.

# IC at V=+1.0: at high anodic phi the double-layer DEPLETES H+ (z=+1
# repels from positive electrode). Boltzmann/NP equilibrium gives
#   u_H+_surf ≈ u_H+_bulk - phi = -1.6 - 38.9 = -40.5
# c_H+ at electrode is tiny -> (c_H+/c_ref)^2 in R2's cathodic factor
# kills R2 (factor ~10^-34). R1_anodic (no H+ factor in production
# code) dominates, mass-transport limited at the H2O2 supply rate
# 1.5e-5; SS u_H2O2_surf ~ -14 (modest depletion).
IC_TRIES: list[tuple[float, float]] = [
    # (u_H+_at_electrode, u_H2O2_at_electrode)
    (-40.0, -14.0),   # SS estimate
    (-40.0, -10.0),
    (-35.0, -14.0),
    (-40.0, -20.0),
    (-30.0, -14.0),
    (-45.0, -14.0),
]

OUT_SUBDIR = "anodic_cold_start"


def main() -> None:
    from scripts._bv_common import (
        setup_firedrake_env,
        V_T, I_SCALE,
        K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2,
        THREE_SPECIES_LOGC_BOLTZMANN,
        DEFAULT_CLO4_BOLTZMANN_COUNTERION,
        SNES_OPTS_CHARGED,
        make_bv_solver_params,
    )
    setup_firedrake_env()

    E_EQ_R1, E_EQ_R2 = 0.68, 1.78

    import firedrake as fd
    import firedrake.adjoint as adj
    from Forward.bv_solver import (
        build_context, build_forms,
        make_graded_rectangle_mesh,
    )
    from Forward.bv_solver.observables import _build_bv_observable_form

    OUT_DIR = os.path.join(_ROOT, "StudyResults", OUT_SUBDIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Bulk concentration values (= production c0_HAT)
    c0_HAT = list(THREE_SPECIES_LOGC_BOLTZMANN.c0_vals_hat)  # [1.0, 1e-4, 0.2]
    u_bulk = [float(np.log(max(c, 1e-30))) for c in c0_HAT]
    n_species = THREE_SPECIES_LOGC_BOLTZMANN.n_species

    print("=" * 80)
    print("  Anodic cold-start probe (v2: widened u_clamp + H+ accumulation)")
    print("=" * 80)
    print(f"  V_start         = +{V_START} V    (eta_hat = {V_START/V_T:+.3f})")
    print(f"  V_down chain    = {V_DOWN}")
    print(f"  exponent_clip   = {EXPONENT_CLIP}")
    print(f"  u_clamp         = {U_CLAMP}")
    print(f"  IC (u_H+, u_H2O2) tries = {IC_TRIES}")
    print(f"  c0_HAT bulk     = {c0_HAT}")
    print(f"  u_bulk          = {[f'{u:+.3f}' for u in u_bulk]}")
    print()

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=int(MESH_NY), beta=3.0)

    snes_opts = {**SNES_OPTS_CHARGED}
    snes_opts.update({
        "snes_max_it":               400,
        "snes_atol":                 1e-7,
        "snes_rtol":                 1e-10,
        "snes_stol":                 1e-12,
        "snes_linesearch_type":      "l2",
        "snes_linesearch_maxlambda": 0.3,
        "snes_divergence_tolerance": 1e10,
    })

    def _make_sp(v_rhe: float):
        sp = make_bv_solver_params(
            eta_hat=v_rhe / V_T,
            dt=1e15, t_end=1e15,           # SS solve, no transient
            species=THREE_SPECIES_LOGC_BOLTZMANN,
            snes_opts=snes_opts,
            formulation="logc",
            log_rate=True,
            u_clamp=U_CLAMP,
            boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION],
            k0_hat_r1=K0_HAT_R1, k0_hat_r2=K0_HAT_R2,
            alpha_r1=ALPHA_R1,   alpha_r2=ALPHA_R2,
            E_eq_r1=E_EQ_R1,     E_eq_r2=E_EQ_R2,
        )
        new_opts = dict(sp.solver_options)
        new_bv = dict(new_opts["bv_convergence"])
        new_bv["exponent_clip"] = float(EXPONENT_CLIP)
        new_opts["bv_convergence"] = new_bv
        return sp.with_solver_options(new_opts)

    def _extract_snes(sp):
        return {k: v for k, v in sp.solver_options.items()
                if k.startswith(("snes_", "ksp_", "pc_", "mat_"))}

    def _set_anodic_ic(ctx, *, u_H_at_electrode: float, u_H2O2_at_electrode: float):
        """Anodic-biased IC: u_O2 at bulk, u_H+ linear from bulk at y=1 to
        ``u_H_at_electrode`` at y=0 (double-layer accumulation), u_H2O2
        linear from bulk at y=1 to ``u_H2O2_at_electrode`` at y=0
        (depleted by R2 + anodic R1); phi linear from eta_hat at y=0 to
        0 at y=1."""
        x, y = fd.SpatialCoordinate(ctx["mesh"])
        eta_hat_target = float(ctx["nondim"]["phi_applied_model"])
        U = ctx["U"]; U_prev = ctx["U_prev"]
        U.sub(0).interpolate(fd.Constant(u_bulk[0]) + 0 * x)              # u_O2 = bulk
        U.sub(1).interpolate(
            fd.Constant(u_bulk[1]) * y
            + fd.Constant(u_H2O2_at_electrode) * (1.0 - y)
        )                                                                  # u_H2O2 linear
        U.sub(2).interpolate(
            fd.Constant(u_bulk[2]) * y
            + fd.Constant(u_H_at_electrode) * (1.0 - y)
        )                                                                  # u_H+ linear (NEW)
        U.sub(3).interpolate(fd.Constant(eta_hat_target) * (1.0 - y))     # phi linear
        U_prev.assign(U)

    def _build_and_solve(sp, *, ic_function=None,
                         ic_u_H_at_electrode: float | None = None,
                         ic_u_H2O2_at_electrode: float | None = None):
        """Build ctx + Newton problem at sp's eta_hat. If ``ic_function`` is
        provided, copy it in as IC; else set anodic IC with given depletion.
        Returns (ctx, converged: bool, snes_iters: int, error_msg or None)."""
        ctx = build_context(sp, mesh=mesh)
        ctx = build_forms(ctx, sp)
        if ic_function is not None:
            ctx["U"].assign(ic_function)
            ctx["U_prev"].assign(ic_function)
        else:
            assert ic_u_H2O2_at_electrode is not None
            assert ic_u_H_at_electrode is not None
            _set_anodic_ic(
                ctx,
                u_H_at_electrode=ic_u_H_at_electrode,
                u_H2O2_at_electrode=ic_u_H2O2_at_electrode,
            )

        F_res = ctx["F_res"]; bcs = ctx["bcs"]; U = ctx["U"]
        J = fd.derivative(F_res, U)
        problem = fd.NonlinearVariationalProblem(F_res, U, bcs=bcs, J=J)
        solver = fd.NonlinearVariationalSolver(
            problem, solver_parameters=_extract_snes(sp)
        )
        try:
            solver.solve()
            return ctx, True, int(solver.snes.getIterationNumber()), None
        except fd.ConvergenceError as exc:
            return ctx, False, int(solver.snes.getIterationNumber()), str(exc)

    def _grab_observables(ctx):
        f_cd = _build_bv_observable_form(
            ctx, mode="current_density", reaction_index=None, scale=-I_SCALE)
        f_pc = _build_bv_observable_form(
            ctx, mode="peroxide_current", reaction_index=None, scale=-I_SCALE)
        cd = float(fd.assemble(f_cd))
        pc = float(fd.assemble(f_pc))
        # Surface fields
        elec = int(ctx["bv_settings"]["electrode_marker"])
        ds_e = fd.ds(domain=ctx["mesh"], subdomain_id=elec)
        area = float(fd.assemble(fd.Constant(1.0) * ds_e))
        sf = {"electrode_area_nondim": area}
        for i in range(n_species):
            u_mean = float(fd.assemble(ctx["U"].sub(i) * ds_e)) / area
            sf[f"u{i}_surface_mean"] = u_mean
            try:
                sf[f"c{i}_surface_mean"] = float(np.exp(u_mean))
            except OverflowError:
                sf[f"c{i}_surface_mean"] = float("inf")
        sf["phi_surface_mean"] = (
            float(fd.assemble(ctx["U"].sub(n_species) * ds_e)) / area
        )
        return cd, pc, sf

    # ---------------- Step 1: cold solve at V_START with anodic IC ----------------
    rows: list[dict] = []
    snapshot = None

    print("-" * 80)
    print(f"  STEP 1: anodic cold-start at V_RHE = +{V_START} V")
    print("-" * 80)
    sp_start = _make_sp(V_START)

    ctx_start = None
    iters_start = -1
    err_start = "untried"
    chosen_ic: tuple[float, float] | None = None
    with adj.stop_annotating():
        for (u_H_try, u_H2O2_try) in IC_TRIES:
            print(f"  trying u_H+={u_H_try:+.2f}, u_H2O2={u_H2O2_try:+.2f} at electrode ...")
            t0 = time.time()
            ctx_start, ok, iters_start, err_start = _build_and_solve(
                sp_start,
                ic_u_H_at_electrode=u_H_try,
                ic_u_H2O2_at_electrode=u_H2O2_try,
            )
            dt_solve = time.time() - t0
            if ok:
                chosen_ic = (u_H_try, u_H2O2_try)
                print(f"    OK in {iters_start} iters ({dt_solve:.1f}s)")
                break
            else:
                err_short = (err_start or "").splitlines()[0] if err_start else "?"
                print(f"    FAIL in {iters_start} iters ({dt_solve:.1f}s): {err_short}")

    if chosen_ic is None:
        print(f"\n[ERROR] All anodic IC attempts failed at V=+{V_START}")
        rows.append({
            "v_rhe": V_START,
            "method": "anodic-cold-failed",
            "newton_converged": False,
            "newton_iters": iters_start,
            "ic_tries": [list(t) for t in IC_TRIES],
            "snes_error": err_start,
        })
    else:
        cd, pc, sf = _grab_observables(ctx_start)
        print(f"  CD = {cd:+.6e},  PC = {pc:+.6e}")
        print(f"  surface u_O2={sf['u0_surface_mean']:+.3f}, "
              f"u_H2O2={sf['u1_surface_mean']:+.3f}, "
              f"u_H+={sf['u2_surface_mean']:+.3f}, "
              f"phi={sf['phi_surface_mean']:+.3f}")
        rows.append({
            "v_rhe": V_START,
            "method": f"anodic-cold (u_H+={chosen_ic[0]:+.1f}, u_H2O2={chosen_ic[1]:+.1f})",
            "newton_converged": True,
            "newton_iters": iters_start,
            "cd_mA_cm2": cd,
            "pc_mA_cm2": pc,
            "surface_fields": sf,
        })
        snapshot = ctx_start["U"].copy(deepcopy=True)

    # ---------------- Step 2: walk down ----------------
    if snapshot is not None:
        print()
        print("-" * 80)
        print(f"  STEP 2: walk down toward E_eq_R1 = +{E_EQ_R1} V")
        print("-" * 80)
        with adj.stop_annotating():
            for v in V_DOWN:
                t0 = time.time()
                sp_v = _make_sp(v)
                ctx_v, ok, iters, err = _build_and_solve(
                    sp_v, ic_function=snapshot
                )
                dt_solve = time.time() - t0
                if ok:
                    cd, pc, sf = _grab_observables(ctx_v)
                    print(f"  V=+{v:.3f}  OK in {iters} iters ({dt_solve:.1f}s)  "
                          f"CD={cd:+.4e}  PC={pc:+.4e}  "
                          f"u_H2O2_surf={sf['u1_surface_mean']:+.2f}")
                    rows.append({
                        "v_rhe": v, "method": "warm-down",
                        "newton_converged": True, "newton_iters": iters,
                        "cd_mA_cm2": cd, "pc_mA_cm2": pc,
                        "surface_fields": sf,
                    })
                    snapshot = ctx_v["U"].copy(deepcopy=True)
                else:
                    err_short = (err or "").splitlines()[0] if err else "?"
                    print(f"  V=+{v:.3f}  FAIL in {iters} iters ({dt_solve:.1f}s): {err_short}")
                    rows.append({
                        "v_rhe": v, "method": "warm-down-failed",
                        "newton_converged": False, "newton_iters": iters,
                        "snes_error": err,
                    })
                    break  # stop the chain

    # ---------------- Save ----------------
    csv_path = os.path.join(OUT_DIR, "iv_curve.csv")
    with open(csv_path, "w") as f:
        f.write("V_RHE,cd_mA_cm2,pc_mA_cm2,method,newton_iters\n")
        for r in rows:
            cd_s = f"{r['cd_mA_cm2']:.8e}" if r.get("newton_converged") else ""
            pc_s = f"{r['pc_mA_cm2']:.8e}" if r.get("newton_converged") else ""
            f.write(f"{r['v_rhe']:.4f},{cd_s},{pc_s},"
                    f"{r['method']},{r['newton_iters']}\n")
    json_path = os.path.join(OUT_DIR, "iv_curve.json")
    with open(json_path, "w") as f:
        json.dump({
            "config": {
                "V_START": V_START, "V_DOWN": V_DOWN,
                "EXPONENT_CLIP": EXPONENT_CLIP, "MESH_NY": MESH_NY,
                "IC_TRIES": [list(t) for t in IC_TRIES],
                "k0_hat_r1": float(K0_HAT_R1), "k0_hat_r2": float(K0_HAT_R2),
                "alpha_r1": float(ALPHA_R1), "alpha_r2": float(ALPHA_R2),
                "E_eq_r1": E_EQ_R1, "E_eq_r2": E_EQ_R2,
                "V_T": float(V_T), "I_SCALE": float(I_SCALE),
            },
            "rows": rows,
        }, f, indent=2)

    print()
    print(f"[OK] CSV  -> {csv_path}")
    print(f"[OK] JSON -> {json_path}")

    # ---------------- Plot combined with cathodic-side data ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Anodic-side rows (this run)
        anodic_v = [r["v_rhe"] for r in rows if r.get("newton_converged")]
        anodic_cd = [r["cd_mA_cm2"] for r in rows if r.get("newton_converged")]
        anodic_pc = [r["pc_mA_cm2"] for r in rows if r.get("newton_converged")]

        # Cathodic-side data from peroxide_window_extension run
        cath_path = os.path.join(
            _ROOT, "StudyResults", "peroxide_window_extension", "iv_curve.json"
        )
        cath_v: list[float] = []
        cath_cd: list[float] = []
        cath_pc: list[float] = []
        if os.path.isfile(cath_path):
            cath = json.load(open(cath_path))
            for v_, cd_, pc_ in zip(
                cath["v_rhe"], cath["cd_mA_cm2"], cath["pc_mA_cm2"]
            ):
                if cd_ is not None and pc_ is not None:
                    cath_v.append(v_); cath_cd.append(cd_); cath_pc.append(pc_)

        fig = plt.figure(figsize=(11, 11))
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.32)
        ax_cd = fig.add_subplot(gs[0])
        ax_pc = fig.add_subplot(gs[1], sharex=ax_cd)
        ax_pc_zoom = fig.add_subplot(gs[2], sharex=ax_cd)

        # CD
        if cath_v:
            ax_cd.plot(cath_v, cath_cd, "b-o", markersize=5,
                       label="cathodic-cold (warm-walk)")
        ax_cd.plot(anodic_v, anodic_cd, "g-s", markersize=6,
                   label="anodic-cold (this run)")
        ax_cd.axvline(E_EQ_R1, color="red", ls="--", lw=0.8, alpha=0.6,
                      label=f"E_eq_R1 = +{E_EQ_R1} V")
        ax_cd.set_ylabel("CD (mA/cm²)")
        ax_cd.set_title(
            f"Two-sided continuation: cathodic-cold + anodic-cold "
            f"(clip={int(EXPONENT_CLIP)})"
        )
        ax_cd.grid(True, alpha=0.3)
        ax_cd.legend(fontsize=9)

        # PC symlog
        if cath_v:
            ax_pc.plot(cath_v, cath_pc, "b-o", markersize=5,
                       label="cathodic-cold")
        ax_pc.plot(anodic_v, anodic_pc, "g-s", markersize=6,
                   label="anodic-cold")
        ax_pc.set_yscale("symlog", linthresh=1e-6)
        ax_pc.axvline(E_EQ_R1, color="red", ls="--", lw=0.8, alpha=0.6)
        ax_pc.axhline(0, color="k", ls=":", lw=0.6)
        ax_pc.set_ylabel("PC (mA/cm²) [symlog]")
        ax_pc.grid(True, which="both", alpha=0.3)
        ax_pc.legend(fontsize=9)

        # PC linear zoom
        if cath_v:
            ax_pc_zoom.plot(cath_v, cath_pc, "b-o", markersize=5,
                            label="cathodic-cold")
        ax_pc_zoom.plot(anodic_v, anodic_pc, "g-s", markersize=6,
                        label="anodic-cold")
        ax_pc_zoom.axvline(E_EQ_R1, color="red", ls="--", lw=0.8, alpha=0.6)
        ax_pc_zoom.axhline(0, color="k", ls=":", lw=0.6)
        ax_pc_zoom.set_xlabel("V vs RHE (V)")
        ax_pc_zoom.set_ylabel("PC (mA/cm²) [linear]")
        ax_pc_zoom.grid(True, alpha=0.3)
        ax_pc_zoom.legend(fontsize=9)

        png_path = os.path.join(OUT_DIR, "iv_curve_two_sided.png")
        plt.savefig(png_path, dpi=160, bbox_inches="tight")
        plt.close()
        print(f"[OK] PNG  -> {png_path}")
    except Exception as exc:
        print(f"[WARN] plot skipped: {exc}")


if __name__ == "__main__":
    main()
