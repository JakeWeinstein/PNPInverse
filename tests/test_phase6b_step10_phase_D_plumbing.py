"""Phase 6β step 10 — Phase D Δ_β plumbing regression tests.

Covers the D1 + D3 spec from
``~/.claude/plans/phase6b-step10-phase-D-deltaBeta-fit.md``:

* ``calibration.singh2016`` helper module is Firedrake-free.
* ``compute_beta_per_cation('K+')`` returns ``-45.608196 pm²`` exactly
  to 6 decimal places.  ``compute_beta_per_cation('K')`` raises
  ``ValueError`` (no alias support).
* ``SINGH_2016_CATION_PARAMS`` is the same Python object whether
  accessed through ``calibration.singh2016`` or ``scripts._bv_common``
  (three import-order permutations, subprocess-isolated).
* :func:`set_reaction_beta_offset_pm2_model` mirrors the live FE
  Function, ``ctx['bv_convergence']`` metadata, AND
  ``bundle.cation_params`` in lockstep.
* At Δ_β = 0 the cation-hydrolysis residual is byte-equivalent to the
  pre-D1 v10b path (machine precision via UFL ``x + 0 ≡ x``).
* At Δ_β = 1e6 pm² the ΔpKa expression assembles to
  ``(β_K_Cu + 1e6) · σ_singh`` to relative tolerance 1e-6 (residual +
  diagnostics agreement).

The driver-side tests (D3 items #1–10) live with the driver in
``tests/test_phase6b_step10_phase_D_fit_eval.py`` — this file is the
plumbing layer only.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ===================================================================
# D3 — calibration.singh2016 leaf module
# ===================================================================


def test_calibration_singh2016_firedrake_free():
    """``calibration.singh2016`` must not pull Firedrake or Forward.bv_solver
    into ``sys.modules``.  Verified by a clean-room subprocess so test
    ordering can never corrupt the check.
    """
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {_ROOT!r})
        import calibration.singh2016 as singh
        assert "firedrake" not in sys.modules, "calibration.singh2016 pulled firedrake"
        assert "Forward.bv_solver" not in sys.modules, "calibration.singh2016 pulled Forward.bv_solver"
        print("OK", singh.compute_beta_per_cation("K+"))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, (
        f"Firedrake-free import failed:\n"
        f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
    )
    assert result.stdout.startswith("OK"), result.stdout


def test_compute_beta_per_cation_K_plus_at_cu_default():
    """For K+ at the Cu default ``r_H_El_pm_Cu = 200.98``, β must
    equal -45.608196 pm² rounded to 6 decimal places (the canonical
    value used throughout the Phase D bracket construction).
    """
    from calibration.singh2016 import compute_beta_per_cation

    beta = compute_beta_per_cation("K+")
    assert round(beta, 6) == -45.608196, (
        f"β(K+) = {beta!r} does not match canonical -45.608196 at 6 dp"
    )


def test_compute_beta_per_cation_canonical_keys_all_cations():
    """``compute_beta_per_cation`` returns a float for every canonical
    charged-form cation key.  Numeric values are byte-stable from the
    Singh Table S1 + Cu r_H_El back-fit data.
    """
    from calibration.singh2016 import compute_beta_per_cation

    for cation in ("Li+", "Na+", "K+", "Rb+", "Cs+"):
        val = compute_beta_per_cation(cation)
        assert isinstance(val, float), f"β({cation}) not a float: {val!r}"


def test_compute_beta_per_cation_rejects_bare_element_alias():
    """``compute_beta_per_cation('K')`` (bare element, no charge sign)
    must raise ``ValueError``.  No alias support — the canonical keys
    are locked to the charged-form convention.
    """
    from calibration.singh2016 import compute_beta_per_cation

    with pytest.raises(ValueError, match="unknown cation"):
        compute_beta_per_cation("K")


def test_compute_beta_per_cation_rejects_lowercase_alias():
    from calibration.singh2016 import compute_beta_per_cation

    with pytest.raises(ValueError, match="unknown cation"):
        compute_beta_per_cation("k+")


def test_compute_beta_per_cation_override_r_h_el():
    """Caller-provided ``r_H_El_pm`` overrides the table default.  At
    r_H_El = r_M_O (Singh's geometric crossover) β must vanish exactly.
    """
    from calibration.singh2016 import (
        SINGH_2016_CATION_PARAMS,
        SINGH_R_O_PM,
        compute_beta_per_cation,
    )

    r_M = SINGH_2016_CATION_PARAMS["K+"]["r_M_pm"]
    r_M_O = r_M + SINGH_R_O_PM  # exact crossover
    val = compute_beta_per_cation("K+", r_H_El_pm=r_M_O)
    assert val == pytest.approx(0.0, abs=1e-20)


def test_compute_beta_per_cation_rejects_non_positive_r_h_el():
    """``r_H_El_pm <= 0`` raises (Singh's 1/r_H_El² diverges)."""
    from calibration.singh2016 import compute_beta_per_cation

    with pytest.raises(ValueError, match="r_H_El_pm must be positive"):
        compute_beta_per_cation("K+", r_H_El_pm=0.0)
    with pytest.raises(ValueError, match="r_H_El_pm must be positive"):
        compute_beta_per_cation("K+", r_H_El_pm=-1.0)


# ===================================================================
# D3 — SINGH_2016_CATION_PARAMS is-identity across import order
# ===================================================================


_IDENTITY_SCRIPT_TEMPLATE = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, {root!r})
    {first_import}
    {second_import}
    from calibration.singh2016 import SINGH_2016_CATION_PARAMS as A
    from scripts._bv_common import SINGH_2016_CATION_PARAMS as B
    assert A is B, "SINGH_2016_CATION_PARAMS aliases broke is-identity!"
    print("OK", id(A))
    """
)


def _run_identity_subprocess(first: str, second: str) -> str:
    script = _IDENTITY_SCRIPT_TEMPLATE.format(
        root=_ROOT,
        first_import=first,
        second_import=second,
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, (
        f"identity subprocess failed: stdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )
    return result.stdout


def test_singh_params_identity_singh2016_first():
    """Import calibration.singh2016 first, then scripts._bv_common —
    the cation params dict must be the SAME Python object.
    """
    out = _run_identity_subprocess(
        first="import calibration.singh2016",
        second="import scripts._bv_common",
    )
    assert out.startswith("OK"), out


def test_singh_params_identity_bv_common_first():
    """Import scripts._bv_common first (triggers the
    calibration.singh2016 import transitively), then re-import singh
    directly — still the same object.
    """
    out = _run_identity_subprocess(
        first="import scripts._bv_common",
        second="import calibration.singh2016",
    )
    assert out.startswith("OK"), out


def test_singh_params_identity_both_in_one_breath():
    """Import both modules in the same line via tuple form."""
    out = _run_identity_subprocess(
        first="import calibration.singh2016, scripts._bv_common",
        second="",
    )
    assert out.startswith("OK"), out


# ===================================================================
# D3 — Setter mirror contract (Firedrake-backed)
# ===================================================================


def _build_minimal_cation_hydrolysis_ctx():
    """Build a minimal Firedrake ctx with the cation_hydrolysis bundle.

    Helper for tests that need to exercise the offset setter and
    residual without invoking the full BV form-build path.
    """
    import firedrake as fd
    from Forward.bv_solver.cation_hydrolysis import (
        build_cation_hydrolysis_terms,
    )

    mesh = fd.IntervalMesh(2, 0.0, 1.0)
    R_space = fd.FunctionSpace(mesh, "R", 0)
    conv_cfg = {
        "enable_cation_hydrolysis": True,
        "cation_hydrolysis_config": {
            "k_hyd": 1e-3,
            "k_prot": 1e-3,
            "k_des": 1.0,
            "delta_ohp_hat": 4e-6,
            "gamma_max_nondim": 0.047,
            "pka_shift_form": "singh_2016_eq_4",
            "pka_shift_params": {
                "z_eff": 0.919, "r_M_pm": 138.0, "r_H_El_pm": 200.98,
                "A_pm": 620.32, "B": 17.154, "r_O_pm": 63.0,
                "anode_clamp": True,
            },
            "beta_offset_pm2": 0.0,
        },
    }
    ctx = {"bv_convergence": conv_cfg}
    roles = ["neutral", "neutral", "proton", "counterion"]
    z_vals = [0.0, 0.0, +1.0, +1.0]
    bundle = build_cation_hydrolysis_terms(
        ctx=ctx, conv_cfg=conv_cfg, z_vals=z_vals, roles=roles, h_idx=2,
        R_space=R_space,
    )
    return ctx, bundle, mesh, R_space


@pytest.mark.slow
def test_set_beta_offset_pm2_mirrors_ctx_and_diagnostics():
    """The setter must update three places in lockstep:
    1. The live FE Function ``bundle.beta_offset_pm2_func``.
    2. The metadata dict
       ``ctx['bv_convergence']['cation_hydrolysis_config']['beta_offset_pm2']``.
    3. The bundle's ``cation_params['beta_offset_pm2']`` mirror.
    """
    from Forward.bv_solver.anchor_continuation import (
        set_reaction_beta_offset_pm2_model,
    )

    ctx, bundle, _, _ = _build_minimal_cation_hydrolysis_ctx()

    # Initial state
    assert float(bundle.beta_offset_pm2_func) == 0.0
    assert (
        ctx["bv_convergence"]["cation_hydrolysis_config"]["beta_offset_pm2"]
        == 0.0
    )
    assert bundle.cation_params["beta_offset_pm2"] == 0.0

    # Positive value
    set_reaction_beta_offset_pm2_model(ctx, +12.345)
    assert float(bundle.beta_offset_pm2_func) == pytest.approx(12.345)
    assert (
        ctx["bv_convergence"]["cation_hydrolysis_config"]["beta_offset_pm2"]
        == 12.345
    )
    assert bundle.cation_params["beta_offset_pm2"] == 12.345

    # Negative value (Phase D bracket allows any sign)
    set_reaction_beta_offset_pm2_model(ctx, -45.608)
    assert float(bundle.beta_offset_pm2_func) == pytest.approx(-45.608)
    assert (
        ctx["bv_convergence"]["cation_hydrolysis_config"]["beta_offset_pm2"]
        == -45.608
    )
    assert bundle.cation_params["beta_offset_pm2"] == -45.608

    # Very large value (per the Phase D Stern bracket upper bound)
    set_reaction_beta_offset_pm2_model(ctx, 1.0e6)
    assert float(bundle.beta_offset_pm2_func) == 1.0e6


@pytest.mark.slow
def test_set_beta_offset_pm2_raises_on_missing_bundle():
    """Calling the setter on a ctx without a cation_hydrolysis bundle
    must raise ``ValueError`` (no silent no-op).
    """
    from Forward.bv_solver.anchor_continuation import (
        set_reaction_beta_offset_pm2_model,
    )

    with pytest.raises(ValueError, match="no 'cation_hydrolysis' bundle"):
        set_reaction_beta_offset_pm2_model({}, 0.0)


# ===================================================================
# D3 — Residual byte-equivalence at offset=0
# ===================================================================


def _assemble_pka_shift_avg(pka_shift_expr, mesh):
    """Helper: compute the per-domain average of a UFL ΔpKa expression."""
    import firedrake as fd

    num = float(fd.assemble(pka_shift_expr * fd.dx(domain=mesh)))
    denom = float(fd.assemble(fd.Constant(1.0) * fd.dx(domain=mesh)))
    return num / denom


@pytest.mark.slow
def test_beta_offset_zero_byte_equivalent_to_pre_d1():
    """At Δ_β = 0 the ``build_pka_shift`` UFL expression must match
    the pre-D1 path (where ``beta_offset_pm2_func=None``) to machine
    precision when evaluated at the same σ_S.
    """
    import firedrake as fd
    from Forward.bv_solver.cation_hydrolysis import build_pka_shift

    ctx, bundle, mesh, R_space = _build_minimal_cation_hydrolysis_ctx()

    # σ_S = -0.02 C/m² (cathodic; representative of V_kin conditions)
    sigma_S = fd.Constant(-0.02)

    # New path with offset Function (offset = 0)
    assert float(bundle.beta_offset_pm2_func) == 0.0
    pka_new = build_pka_shift(
        cation_params=bundle.cation_params,
        sigma_S=sigma_S,
        r_H_El_func=bundle.r_H_El_pm_func,
        beta_offset_pm2_func=bundle.beta_offset_pm2_func,
    )
    val_new = _assemble_pka_shift_avg(pka_new, mesh)

    # Pre-D1 path (no offset Function — fallback collapse)
    pka_old = build_pka_shift(
        cation_params=bundle.cation_params,
        sigma_S=sigma_S,
        r_H_El_func=bundle.r_H_El_pm_func,
    )
    val_old = _assemble_pka_shift_avg(pka_old, mesh)

    # `x + 0` in UFL → numerically identical to `x` at quadrature points.
    assert val_new == pytest.approx(val_old, rel=1e-14, abs=1e-30)


@pytest.mark.slow
def test_beta_offset_nonzero_residual_matches_analytic():
    """At Δ_β = +1e6 pm² the assembled ΔpKa value must equal
    ``(β_K_Cu + 1e6) · σ_singh`` analytically computed from the same
    σ_S, to relative tolerance 1e-6.
    """
    import firedrake as fd
    from calibration.singh2016 import compute_beta_per_cation
    from Forward.bv_solver.anchor_continuation import (
        set_reaction_beta_offset_pm2_model,
    )
    from Forward.bv_solver.cation_hydrolysis import build_pka_shift

    ctx, bundle, mesh, R_space = _build_minimal_cation_hydrolysis_ctx()

    # σ_S = -0.02 C/m² (cathodic)
    sigma_S_val = -0.02
    sigma_S = fd.Constant(sigma_S_val)

    # Apply Δ_β = +1e6 pm² (matches D3 #15 spec)
    delta_beta = 1.0e6
    set_reaction_beta_offset_pm2_model(ctx, delta_beta)

    pka_expr = build_pka_shift(
        cation_params=bundle.cation_params,
        sigma_S=sigma_S,
        r_H_El_func=bundle.r_H_El_pm_func,
        beta_offset_pm2_func=bundle.beta_offset_pm2_func,
    )
    val_assembled = _assemble_pka_shift_avg(pka_expr, mesh)

    # Analytic expectation: σ_singh = max(0, -σ_S) · 6.2415e-6 (counts/pm²)
    # ΔpKa = (β_K_Cu + Δ_β) · σ_singh
    counts_per_pm2_per_C_per_m2 = (1.0 / 1.602176634e-19) * 1.0e-24
    sigma_singh = max(0.0, -sigma_S_val) * counts_per_pm2_per_C_per_m2
    beta_K_Cu = compute_beta_per_cation("K+")
    expected = (beta_K_Cu + delta_beta) * sigma_singh

    assert val_assembled == pytest.approx(expected, rel=1e-6), (
        f"assembled ΔpKa {val_assembled!r} != expected {expected!r} "
        f"(σ_singh = {sigma_singh!r}, β_K_Cu + Δ_β = "
        f"{beta_K_Cu + delta_beta!r})"
    )


@pytest.mark.slow
def test_beta_offset_anodic_bias_still_clamps_to_zero():
    """Anode clamp must still gate the residual at anodic σ_S — Δ_β
    only scales the magnitude of an already-clamped expression.
    """
    import firedrake as fd
    from Forward.bv_solver.anchor_continuation import (
        set_reaction_beta_offset_pm2_model,
    )
    from Forward.bv_solver.cation_hydrolysis import build_pka_shift

    ctx, bundle, mesh, _ = _build_minimal_cation_hydrolysis_ctx()
    set_reaction_beta_offset_pm2_model(ctx, +1.0e6)
    sigma_S = fd.Constant(+1.0)  # anodic
    pka_expr = build_pka_shift(
        cation_params=bundle.cation_params,
        sigma_S=sigma_S,
        r_H_El_func=bundle.r_H_El_pm_func,
        beta_offset_pm2_func=bundle.beta_offset_pm2_func,
    )
    val = _assemble_pka_shift_avg(pka_expr, mesh)
    assert val == pytest.approx(0.0, abs=1e-20)


@pytest.mark.slow
def test_beta_offset_dispatch_through_parameter_overrides():
    """The ``beta_offset_pm2`` key in the ``_OVERRIDE_DISPATCH`` dict
    (consumed by ``solve_grid_with_anchor``'s parameter-override path)
    routes to :func:`set_reaction_beta_offset_pm2_model`.

    We verify the wiring by inspecting the source — full
    grid-level integration is covered separately by the driver's HARD
    A.2 reproduction gate.
    """
    import inspect

    from Forward.bv_solver import anchor_continuation as ac

    src = inspect.getsource(ac)
    assert '"beta_offset_pm2": set_reaction_beta_offset_pm2_model' in src, (
        "expected 'beta_offset_pm2' key in _OVERRIDE_DISPATCH dict"
    )
    assert "set_reaction_beta_offset_pm2_model" in ac.__all__, (
        "set_reaction_beta_offset_pm2_model missing from anchor_continuation __all__"
    )
