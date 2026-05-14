"""MMS convergence + broken-config tests for the logc_muh + Cs+/SO4(2-)
multi-ion + Stern Robin BV stack.

Verification target: the production stack used by
``scripts/studies/solver_demo_slide15_no_speculative_cs.py``.

Backs:
  - Derivation: ``docs/solver/mms_pnpbv_muh_multi_ion_stern_derivation.md``
  - Plan: ``.plans/mms-muh-multi-ion-stern/PLAN.md``
  - Source-builder: ``scripts/verification/mms_pnpbv_muh_multi_ion_stern.py``

Convergence is verified via log-log linear regression (``scipy.stats.linregress``):
  - L2 rate >= 1.8 (slack from CG1 theoretical 2.0)
  - H1 rate >= 0.8 (slack from CG1 theoretical 1.0)
  - R-squared > 0.99 for all fits (asymptotic regime)

Three test classes:

  - ``TestMMSConvergence``: UnitSquareMesh sweep N in (8, 16, 32, 64);
    per-primary-unknown rates and R^2 thresholds.

  - ``TestMMSProductionGradedMesh``: single-solve recovery on the demo
    graded rectangle (Nx=8, Ny=80, beta=3); per-field thresholds.

  - ``TestMMSAsserts``: 12 parametrized broken-config tests verifying
    the MMS invariant layers catch feature-flag drift, missing
    counterions, wrong reactions, etc.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

if os.environ.get("PYTEST_XDIST_WORKER") is not None:
    pytest.skip(
        "MMS multi-ion + Stern tests are not xdist-safe (shared Firedrake "
        "TSFC cache and StudyResults/ outputs). Run as "
        "`pytest -p no:xdist tests/test_mms_logc_muh_multi_ion_stern.py`.",
        allow_module_level=True,
    )

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from conftest import skip_without_firedrake


# ---------------------------------------------------------------------------
# Lazy imports — defer Firedrake-touching imports to fixture body so collection
# stays cheap and PyTorch-only test runs aren't poisoned by PETSc/MPI loads.
# ---------------------------------------------------------------------------

def _import_run_mms():
    from scripts.verification.mms_pnpbv_muh_multi_ion_stern import run_mms
    return run_mms


def _import_graded_verifier():
    from scripts.verification.mms_pnpbv_muh_multi_ion_stern import (
        verify_on_graded_production_mesh,
    )
    return verify_on_graded_production_mesh


def _import_make_sp():
    from scripts.verification.mms_pnpbv_muh_multi_ion_stern import (
        make_sp_production_muh,
    )
    return make_sp_production_muh


def _import_solve_helpers():
    from scripts.verification.mms_pnpbv_muh_multi_ion_stern import (
        _assert_prebuild_config_invariants,
        _assert_postbuild_static_ctx_invariants,
    )
    return {
        "prebuild": _assert_prebuild_config_invariants,
        "postbuild": _assert_postbuild_static_ctx_invariants,
    }


# ---------------------------------------------------------------------------
# Field labels
# ---------------------------------------------------------------------------

FIELD_NAMES = ["u_O2", "u_H2O2", "mu_H", "phi"]
SPECIES_LABELS = ["O2 (u)", "H2O2 (u)", "H+ (mu)", "phi"]


# ---------------------------------------------------------------------------
# Helper: log-log convergence rate assertion
# ---------------------------------------------------------------------------

def _assert_convergence_rate(h_vals, err_vals, expected_rate: float,
                             *, rate_tol: float = 0.2,
                             min_r_squared: float = 0.99,
                             label: str = "") -> tuple[float, float]:
    """Assert log-log slope and R^2.  Returns ``(slope, r_squared)``."""
    from scipy.stats import linregress

    log_h = np.log(np.array(h_vals, dtype=float))
    log_err = np.log(np.array(err_vals, dtype=float))
    slope, _, r_value, _, std_err = linregress(log_h, log_err)
    r_squared = float(r_value) ** 2

    assert r_squared > min_r_squared, (
        f"{label}: R^2 = {r_squared:.6f} < {min_r_squared} "
        f"(slope={slope:.4f}, std_err={std_err:.4f})"
    )
    assert slope >= expected_rate - rate_tol, (
        f"{label}: slope = {slope:.4f} < {expected_rate:.1f} - {rate_tol} "
        f"(R^2={r_squared:.6f}, std_err={std_err:.4f})"
    )
    return float(slope), r_squared


# ---------------------------------------------------------------------------
# Convergence-study test class
# ---------------------------------------------------------------------------

@skip_without_firedrake
@pytest.mark.slow
class TestMMSConvergence:
    """UnitSquareMesh sweep convergence study."""

    MESH_SIZES = (8, 16, 32, 64)
    OUTPUT_DIR = os.path.join(_ROOT, "StudyResults", "mms_logc_muh_multi_ion_stern")

    @pytest.fixture(scope="class")
    def mms_results(self):
        run_mms = _import_run_mms()
        results = run_mms(self.MESH_SIZES, verbose=True)
        if not all(results["newton_converged"]):
            failed = [
                int(N) for N, c in zip(results["N"], results["newton_converged"])
                if not c
            ]
            pytest.fail(
                f"Newton failed on meshes: {failed} "
                f"(reasons: {results['snes_reason']})"
            )
        assert len(results["N"]) == len(self.MESH_SIZES)
        return results

    def test_l2_convergence_rates(self, mms_results):
        """L2 rate >= 1.8 with R^2 > 0.99 for all 4 primary unknowns."""
        for field, label in zip(FIELD_NAMES, SPECIES_LABELS):
            _assert_convergence_rate(
                mms_results["h"],
                mms_results[f"{field}_L2"],
                expected_rate=2.0, rate_tol=0.2, min_r_squared=0.99,
                label=f"{label} L2",
            )

    def test_h1_convergence_rates(self, mms_results):
        """H1 rate >= 0.8 with R^2 > 0.99 for all 4 primary unknowns."""
        for field, label in zip(FIELD_NAMES, SPECIES_LABELS):
            _assert_convergence_rate(
                mms_results["h"],
                mms_results[f"{field}_H1"],
                expected_rate=1.0, rate_tol=0.2, min_r_squared=0.99,
                label=f"{label} H1",
            )

    def test_save_convergence_artifacts(self, mms_results):
        """Persist JSON + PNG artifacts under StudyResults/.

        Failure here is non-blocking for rate checks but documents the
        provenance for the convergence claim.
        """
        import json
        from datetime import datetime, timezone

        from scripts.verification.mms_pnpbv_muh_multi_ion_stern import (
            plot_convergence,
        )

        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

        payload = {
            "metadata": {
                "date": datetime.now(timezone.utc).isoformat(),
                "mesh_sizes": list(mms_results["N"]),
                "h_values": list(mms_results["h"]),
                "fields": FIELD_NAMES,
                "description": (
                    "MMS convergence — logc_muh + Cs+/SO4(2-) "
                    "multi-ion + Stern Robin BV"
                ),
            },
            "results": {
                k: list(v) if isinstance(v, list) else v
                for k, v in mms_results.items()
            },
        }

        json_path = os.path.join(self.OUTPUT_DIR, "convergence_data.json")
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        assert os.path.isfile(json_path)

        png_path = plot_convergence(mms_results, self.OUTPUT_DIR)
        assert os.path.isfile(png_path)


# ---------------------------------------------------------------------------
# Graded production mesh single-solve recovery
# ---------------------------------------------------------------------------

@skip_without_firedrake
@pytest.mark.slow
class TestMMSProductionGradedMesh:
    """Single-mesh recovery on the demo's graded rectangle (Nx=8, Ny=80,
    beta=3).  Thresholds pinned ~6x above pilot baseline:

      Pilot (2026-05-14):
        u_O2: L2 = 1.40e-3,  H1 = 5.45e-2
        u_H2O2: L2 = 2.41e-3, H1 = 5.46e-2
        mu_H: L2 = 2.93e-3,   H1 = 1.57e-1
        phi:  L2 = 1.55e-3,   H1 = 1.19e-1
        Newton iters: 3
    """

    L2_THRESHOLDS = {
        "u_O2": 1e-2,    # ~6x above 1.40e-3
        "u_H2O2": 1.5e-2,  # ~6x above 2.41e-3
        "mu_H": 2e-2,    # ~6x above 2.93e-3
        "phi": 1e-2,     # ~6x above 1.55e-3
    }
    H1_THRESHOLDS = {
        "u_O2": 3e-1,    # ~6x above 5.45e-2
        "u_H2O2": 3e-1,  # ~6x above 5.46e-2
        "mu_H": 1.0,     # ~6x above 1.57e-1
        "phi": 7e-1,     # ~6x above 1.19e-1
    }
    NEWTON_ITER_CAP = 30

    @pytest.fixture(scope="class")
    def graded_mesh_results(self):
        verify = _import_graded_verifier()
        return verify(verbose=True)

    def test_newton_converges_within_iteration_cap(self, graded_mesh_results):
        assert graded_mesh_results["newton_converged"] is True, (
            f"Newton failed on graded mesh: "
            f"{graded_mesh_results.get('snes_reason', '<no reason>')}"
        )
        iters = int(graded_mesh_results["newton_iterations"])
        assert 0 <= iters < self.NEWTON_ITER_CAP, (
            f"Newton iterations {iters} exceed cap {self.NEWTON_ITER_CAP}"
        )

    def test_l2_recovery(self, graded_mesh_results):
        for field, thr in self.L2_THRESHOLDS.items():
            err = float(graded_mesh_results[f"{field}_L2"])
            assert math.isfinite(err), f"{field}: L2 not finite ({err})"
            assert err < thr, (
                f"{field}: L2 = {err:.3e} >= threshold {thr:.3e}"
            )

    def test_h1_recovery(self, graded_mesh_results):
        for field, thr in self.H1_THRESHOLDS.items():
            err = float(graded_mesh_results[f"{field}_H1"])
            assert math.isfinite(err), f"{field}: H1 not finite ({err})"
            assert err < thr, (
                f"{field}: H1 = {err:.3e} >= threshold {thr:.3e}"
            )


# ---------------------------------------------------------------------------
# 12 broken-config tests (TestMMSAsserts)
# ---------------------------------------------------------------------------

def _force_formulation(formulation_value: str):
    def mutate(sp):
        opts = dict(sp.solver_options)
        bv = dict(opts["bv_convergence"])
        bv["formulation"] = formulation_value
        opts["bv_convergence"] = bv
        return sp.with_solver_options(opts)
    return mutate


def _force_log_rate(value: bool):
    def mutate(sp):
        opts = dict(sp.solver_options)
        bv = dict(opts["bv_convergence"])
        bv["bv_log_rate"] = bool(value)
        opts["bv_convergence"] = bv
        return sp.with_solver_options(opts)
    return mutate


def _force_water_ionization(value: bool):
    def mutate(sp):
        opts = dict(sp.solver_options)
        bv = dict(opts["bv_convergence"])
        bv["enable_water_ionization"] = bool(value)
        opts["bv_convergence"] = bv
        return sp.with_solver_options(opts)
    return mutate


def _force_cation_hydrol(value: bool):
    def mutate(sp):
        opts = dict(sp.solver_options)
        bv = dict(opts["bv_convergence"])
        bv["enable_cation_hydrolysis"] = bool(value)
        opts["bv_convergence"] = bv
        return sp.with_solver_options(opts)
    return mutate


def _force_dt(dt_value: float):
    def mutate(sp):
        from Forward.params import SolverParams
        params = sp.to_list()
        params[2] = float(dt_value)
        return SolverParams.from_list(params)
    return mutate


def _force_snes_atol(value: float):
    def mutate(sp):
        opts = dict(sp.solver_options)
        opts["snes_atol"] = float(value)
        return sp.with_solver_options(opts)
    return mutate


def _force_suppress_poisson(value: bool):
    def mutate(sp):
        opts = dict(sp.solver_options)
        nondim = dict(opts.get("nondim", {}))
        nondim["suppress_poisson_source"] = bool(value)
        opts["nondim"] = nondim
        return sp.with_solver_options(opts)
    return mutate


def _drop_reaction(label: str):
    """Drop the R4e reaction (the second entry in the parallel list)."""
    def mutate(sp):
        opts = dict(sp.solver_options)
        bv_bc = dict(opts["bv_bc"])
        rxns = list(bv_bc.get("reactions", []))
        # Identify reaction by label-via-n_electrons (R4e is 4-electron).
        keep = []
        for rxn in rxns:
            if int(rxn.get("n_electrons", 0)) == 4 and label.upper() == "R4E":
                continue
            if int(rxn.get("n_electrons", 0)) == 2 and label.upper() == "R2E":
                continue
            keep.append(dict(rxn))
        bv_bc["reactions"] = keep
        opts["bv_bc"] = bv_bc
        return sp.with_solver_options(opts)
    return mutate


def _drop_counterion(label: str):
    """Drop the counterion matching ``label`` (Cs+ or SO4)."""
    def mutate(sp):
        opts = dict(sp.solver_options)
        bv_bc = dict(opts["bv_bc"])
        cs = list(bv_bc.get("boltzmann_counterions", []))
        keep = []
        for entry in cs:
            entry_label = str(entry.get("label", "")).upper()
            if label.upper() in entry_label:
                continue
            if label.upper() == "SO4" and int(entry.get("z", 0)) == -2:
                continue
            if label.upper() == "CS" and int(entry.get("z", 0)) == +1 and "K" not in entry_label:
                continue
            keep.append(dict(entry))
        bv_bc["boltzmann_counterions"] = keep
        opts["bv_bc"] = bv_bc
        return sp.with_solver_options(opts)
    return mutate


def _replace_counterion(old_label: str, new_label: str):
    """Replace Cs+ with a different counterion entry."""
    def mutate(sp):
        opts = dict(sp.solver_options)
        bv_bc = dict(opts["bv_bc"])
        cs = list(bv_bc.get("boltzmann_counterions", []))
        new_cs = []
        for entry in cs:
            if str(entry.get("label", "")) == old_label:
                # Replace with a ClO4- entry (z=-1, smaller bulk concentration).
                new_cs.append({
                    "z": -1,
                    "c_bulk_nondim": 0.05,
                    "phi_clamp": 50.0,
                    "steric_mode": "bikerman",
                    "a_nondim": 1e-5,
                    "label": new_label,
                })
            else:
                new_cs.append(dict(entry))
        bv_bc["boltzmann_counterions"] = new_cs
        opts["bv_bc"] = bv_bc
        return sp.with_solver_options(opts)
    return mutate


def _force_stern(value):
    """Set Stern capacitance to ``None`` (no Stern) or a different value."""
    def mutate(sp):
        opts = dict(sp.solver_options)
        bv_bc = dict(opts["bv_bc"])
        if value is None:
            bv_bc.pop("stern_capacitance_f_m2", None)
        else:
            bv_bc["stern_capacitance_f_m2"] = float(value)
        opts["bv_bc"] = bv_bc
        return sp.with_solver_options(opts)
    return mutate


def _force_k0_r4e_factor(factor: float):
    """Replace R4e k0 with K0_HAT_R4E * ``factor``."""
    def mutate(sp):
        from scripts._bv_common import K0_HAT_R4E
        opts = dict(sp.solver_options)
        bv_bc = dict(opts["bv_bc"])
        rxns = list(bv_bc.get("reactions", []))
        new_rxns = []
        for rxn in rxns:
            new_rxn = dict(rxn)
            if int(new_rxn.get("n_electrons", 0)) == 4:
                new_rxn["k0"] = float(K0_HAT_R4E) * float(factor)
            new_rxns.append(new_rxn)
        bv_bc["reactions"] = new_rxns
        opts["bv_bc"] = bv_bc
        return sp.with_solver_options(opts)
    return mutate


BROKEN_CONFIGS = [
    ("formulation_logc",   _force_formulation("logc"),       "prebuild",
     r"formulation"),
    ("log_rate_off",        _force_log_rate(False),           "prebuild",
     r"bv_log_rate"),
    ("water_on",            _force_water_ionization(True),    "prebuild",
     r"water_ionization"),
    ("cation_hydrol_on",    _force_cation_hydrol(True),       "prebuild",
     r"cation hydrolysis"),
    ("dt_small",            _force_dt(0.1),                   "prebuild",
     r"dt"),
    ("snes_loose",          _force_snes_atol(1e-2),           "prebuild",
     r"snes_atol"),
    ("poisson_suppressed",  _force_suppress_poisson(True),    "prebuild",
     r"suppress_poisson_source"),
    ("missing_reaction_R4e", _drop_reaction("R4E"),            "prebuild",
     r"rxns"),
    ("one_counterion",      _drop_counterion("SO4"),          "prebuild",
     r"bikerman"),
    ("wrong_counterion",    _replace_counterion("Cs+", "ClO4-"), "prebuild",
     r"Cs|a_Cs|c_b_Cs"),
    ("no_stern",             _force_stern(None),               "postbuild",
     r"bv_stern_capacitance_model"),
    ("k0_r4e_wrong",         _force_k0_r4e_factor(1.0),        "postbuild",
     r"k0_R4e"),
]


def _prepare_mms_context_for_asserts(sp, mesh, phase: str):
    """Run the MMS pipeline up to ``phase`` and stop, raising any
    AssertionError that fires.
    """
    helpers = _import_solve_helpers()
    helpers["prebuild"](sp)
    if phase == "prebuild":
        return
    from Forward.bv_solver import build_context, build_forms
    ctx = build_context(sp, mesh=mesh)
    ctx = build_forms(ctx, sp)
    helpers["postbuild"](ctx, sp)
    if phase == "postbuild":
        return
    raise ValueError(f"Unknown phase {phase!r}")


@skip_without_firedrake
@pytest.mark.slow
class TestMMSAsserts:
    """Each broken-config must raise ``AssertionError`` at the documented
    invariant layer (prebuild vs postbuild) with a recognizable message.
    """

    @pytest.mark.parametrize("label,mutate,layer,match",
                             [(c[0], c[1], c[2], c[3]) for c in BROKEN_CONFIGS])
    def test_broken_config_caught(self, label, mutate, layer, match):
        import firedrake as fd

        make_sp = _import_make_sp()
        sp = mutate(make_sp())
        mesh = fd.UnitSquareMesh(8, 8)
        with pytest.raises(AssertionError, match=match):
            _prepare_mms_context_for_asserts(sp, mesh, phase=layer)
