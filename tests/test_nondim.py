"""Unit tests for the Nondim package (constants, scales, transform).

These tests exercise pure-Python / NumPy logic and do NOT require Firedrake.
"""

from __future__ import annotations

import math

import numpy as np
import pytest


# ===================================================================
# Nondim.constants
# ===================================================================

class TestConstants:
    """Verify that physical constants have the expected values."""

    def test_faraday_constant(self):
        from Nondim.constants import FARADAY_CONSTANT
        assert FARADAY_CONSTANT == pytest.approx(96485.3329, rel=1e-8)

    def test_gas_constant(self):
        from Nondim.constants import GAS_CONSTANT
        assert GAS_CONSTANT == pytest.approx(8.314462618, rel=1e-8)

    def test_default_temperature(self):
        from Nondim.constants import DEFAULT_TEMPERATURE_K
        assert DEFAULT_TEMPERATURE_K == pytest.approx(298.15, rel=1e-6)

    def test_vacuum_permittivity(self):
        from Nondim.constants import VACUUM_PERMITTIVITY_F_PER_M
        assert VACUUM_PERMITTIVITY_F_PER_M == pytest.approx(8.8541878128e-12, rel=1e-8)

    def test_relative_permittivity_water(self):
        from Nondim.constants import DEFAULT_RELATIVE_PERMITTIVITY_WATER
        assert DEFAULT_RELATIVE_PERMITTIVITY_WATER == pytest.approx(78.5, rel=1e-6)

    def test_molar_to_mol_per_m3(self):
        from Nondim.constants import MOLAR_TO_MOL_PER_M3
        assert MOLAR_TO_MOL_PER_M3 == 1000.0


# ===================================================================
# Nondim.scales  —  NondimScales dataclass + build_physical_scales
# ===================================================================

class TestNondimScales:
    """Test NondimScales construction and derived quantities."""

    def test_default_construction(self, default_nondim_kwargs):
        from Nondim.scales import build_physical_scales
        scales = build_physical_scales(**default_nondim_kwargs)

        # Check types
        assert isinstance(scales.d_species_m2_s, list)
        assert isinstance(scales.d_ref_m2_s, float)
        assert isinstance(scales.thermal_voltage_v, float)

    def test_d_ref_is_geometric_mean(self, default_nondim_kwargs):
        from Nondim.scales import build_physical_scales
        scales = build_physical_scales(**default_nondim_kwargs)

        d1, d2 = default_nondim_kwargs["d_species_m2_s"]
        expected_d_ref = math.sqrt(d1 * d2)
        assert scales.d_ref_m2_s == pytest.approx(expected_d_ref, rel=1e-12)

    def test_bulk_concentration_conversion(self, default_nondim_kwargs):
        from Nondim.scales import build_physical_scales
        scales = build_physical_scales(**default_nondim_kwargs)

        # 0.1 M = 100 mol/m^3
        expected = default_nondim_kwargs["c_bulk_m"] * 1000.0
        assert scales.bulk_concentration_mol_m3 == pytest.approx(expected, rel=1e-12)

    def test_c_inf_conversion(self, default_nondim_kwargs):
        from Nondim.scales import build_physical_scales
        scales = build_physical_scales(**default_nondim_kwargs)

        expected = default_nondim_kwargs["c_inf_m"] * 1000.0
        assert scales.c_inf_mol_m3 == pytest.approx(expected, rel=1e-12)

    def test_thermal_voltage(self, default_nondim_kwargs):
        from Nondim.scales import build_physical_scales
        from Nondim.constants import GAS_CONSTANT, FARADAY_CONSTANT
        scales = build_physical_scales(**default_nondim_kwargs)

        expected_vt = GAS_CONSTANT * 298.15 / FARADAY_CONSTANT
        assert scales.thermal_voltage_v == pytest.approx(expected_vt, rel=1e-10)
        # Sanity: roughly 25.7 mV
        assert 0.0255 < scales.thermal_voltage_v < 0.0260

    def test_time_scale(self, default_nondim_kwargs):
        from Nondim.scales import build_physical_scales
        scales = build_physical_scales(**default_nondim_kwargs)

        L = default_nondim_kwargs["length_scale_m"]
        expected_time = L * L / scales.d_ref_m2_s
        assert scales.time_s == pytest.approx(expected_time, rel=1e-12)

    def test_kappa_scale(self, default_nondim_kwargs):
        from Nondim.scales import build_physical_scales
        scales = build_physical_scales(**default_nondim_kwargs)

        expected_kappa = scales.d_ref_m2_s / default_nondim_kwargs["length_scale_m"]
        assert scales.kappa_m_s == pytest.approx(expected_kappa, rel=1e-12)

    def test_molar_flux_scale(self, default_nondim_kwargs):
        from Nondim.scales import build_physical_scales
        scales = build_physical_scales(**default_nondim_kwargs)

        L = default_nondim_kwargs["length_scale_m"]
        expected = scales.d_ref_m2_s * scales.bulk_concentration_mol_m3 / L
        assert scales.molar_flux_mol_m2_s == pytest.approx(expected, rel=1e-12)

    def test_current_density_scale(self, default_nondim_kwargs):
        from Nondim.scales import build_physical_scales
        from Nondim.constants import FARADAY_CONSTANT
        scales = build_physical_scales(**default_nondim_kwargs)

        expected = FARADAY_CONSTANT * scales.molar_flux_mol_m2_s
        assert scales.current_density_a_m2 == pytest.approx(expected, rel=1e-12)

    def test_debye_length(self, default_nondim_kwargs):
        from Nondim.scales import build_physical_scales
        from Nondim.constants import FARADAY_CONSTANT
        scales = build_physical_scales(**default_nondim_kwargs)

        expected_debye = math.sqrt(
            scales.permittivity_f_m * scales.thermal_voltage_v
            / (FARADAY_CONSTANT * scales.bulk_concentration_mol_m3)
        )
        assert scales.debye_length_m == pytest.approx(expected_debye, rel=1e-10)
        # Sanity: for 0.1 M, Debye length ~ 1 nm
        assert 1e-10 < scales.debye_length_m < 1e-8

    def test_debye_ratio(self, default_nondim_kwargs):
        from Nondim.scales import build_physical_scales
        scales = build_physical_scales(**default_nondim_kwargs)

        expected_ratio = scales.debye_length_m / default_nondim_kwargs["length_scale_m"]
        assert scales.debye_ratio == pytest.approx(expected_ratio, rel=1e-12)

    def test_diffusivity_scale_equals_d_ref(self, default_nondim_kwargs):
        from Nondim.scales import build_physical_scales
        scales = build_physical_scales(**default_nondim_kwargs)
        assert scales.diffusivity_scale_m2_s == scales.d_ref_m2_s

    def test_concentration_scale_equals_bulk(self, default_nondim_kwargs):
        from Nondim.scales import build_physical_scales
        scales = build_physical_scales(**default_nondim_kwargs)
        assert scales.concentration_scale_mol_m3 == scales.bulk_concentration_mol_m3


class TestNondimScalesToDict:
    """Test the to_dict() method of NondimScales."""

    def test_to_dict_keys(self, default_nondim_kwargs):
        from Nondim.scales import build_physical_scales
        scales = build_physical_scales(**default_nondim_kwargs)
        d = scales.to_dict()

        expected_keys = {
            "d_species_m2_s",
            "d_ref_m2_s",
            "bulk_concentration_mol_m3",
            "c_inf_mol_m3",
            "temperature_k",
            "thermal_voltage_v",
            "permittivity_f_m",
            "length_scale_m",
            "diffusivity_scale_m2_s",
            "concentration_scale_mol_m3",
            "time_scale_s",
            "kappa_scale_m_s",
            "molar_flux_scale_mol_m2_s",
            "current_density_scale_a_m2",
            "current_density_scale_a_cm2",
            "debye_length_m",
            "debye_to_length_ratio",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values_match_attributes(self, default_nondim_kwargs):
        from Nondim.scales import build_physical_scales
        scales = build_physical_scales(**default_nondim_kwargs)
        d = scales.to_dict()

        assert d["d_ref_m2_s"] == scales.d_ref_m2_s
        assert d["temperature_k"] == scales.temperature_k
        assert d["thermal_voltage_v"] == scales.thermal_voltage_v
        assert d["length_scale_m"] == scales.length_scale_m
        assert d["time_scale_s"] == scales.time_s
        # Note: dict key is "kappa_scale_m_s", attribute is "kappa_m_s"
        assert d["kappa_scale_m_s"] == scales.kappa_m_s

    def test_current_density_a_cm2_conversion(self, default_nondim_kwargs):
        from Nondim.scales import build_physical_scales
        scales = build_physical_scales(**default_nondim_kwargs)
        d = scales.to_dict()

        assert d["current_density_scale_a_cm2"] == pytest.approx(
            d["current_density_scale_a_m2"] / 1e4, rel=1e-12
        )


class TestBuildPhysicalScalesValidation:
    """Input validation for build_physical_scales."""

    def test_negative_diffusivity_raises(self):
        from Nondim.scales import build_physical_scales
        with pytest.raises(ValueError, match="strictly positive diffusivities"):
            build_physical_scales(d_species_m2_s=(-1e-9, 1e-9))

    def test_zero_diffusivity_raises(self):
        from Nondim.scales import build_physical_scales
        with pytest.raises(ValueError, match="strictly positive diffusivities"):
            build_physical_scales(d_species_m2_s=(0.0, 1e-9))

    def test_zero_concentration_raises(self):
        from Nondim.scales import build_physical_scales
        with pytest.raises(ValueError, match="c_bulk_m must be > 0"):
            build_physical_scales(d_species_m2_s=(1e-9, 2e-9), c_bulk_m=0.0)

    def test_negative_concentration_raises(self):
        from Nondim.scales import build_physical_scales
        with pytest.raises(ValueError, match="c_bulk_m must be > 0"):
            build_physical_scales(d_species_m2_s=(1e-9, 2e-9), c_bulk_m=-0.5)

    def test_negative_c_inf_raises(self):
        from Nondim.scales import build_physical_scales
        with pytest.raises(ValueError, match="c_inf_m must be >= 0"):
            build_physical_scales(d_species_m2_s=(1e-9, 2e-9), c_inf_m=-1.0)

    def test_zero_temperature_raises(self):
        from Nondim.scales import build_physical_scales
        with pytest.raises(ValueError, match="temperature_k must be > 0"):
            build_physical_scales(d_species_m2_s=(1e-9, 2e-9), temperature_k=0.0)

    def test_zero_length_raises(self):
        from Nondim.scales import build_physical_scales
        with pytest.raises(ValueError, match="length_scale_m must be > 0"):
            build_physical_scales(d_species_m2_s=(1e-9, 2e-9), length_scale_m=0.0)

    def test_zero_permittivity_raises(self):
        from Nondim.scales import build_physical_scales
        with pytest.raises(ValueError, match="relative_permittivity must be > 0"):
            build_physical_scales(d_species_m2_s=(1e-9, 2e-9), relative_permittivity=0.0)

    def test_single_species(self):
        """build_physical_scales should work for a single species."""
        from Nondim.scales import build_physical_scales
        scales = build_physical_scales(d_species_m2_s=(5e-9,))
        assert scales.d_ref_m2_s == pytest.approx(5e-9, rel=1e-12)
        assert len(scales.d_species_m2_s) == 1

    def test_three_species(self):
        """build_physical_scales should work for three species."""
        from Nondim.scales import build_physical_scales
        ds = (1e-9, 2e-9, 4e-9)
        scales = build_physical_scales(d_species_m2_s=ds)
        expected = float(np.exp(np.mean(np.log(ds))))
        assert scales.d_ref_m2_s == pytest.approx(expected, rel=1e-12)
        assert len(scales.d_species_m2_s) == 3


# ===================================================================
# Nondim.transform  —  build_model_scaling
# ===================================================================

class TestBuildModelScalingDisabled:
    """Test build_model_scaling with nondimensionalization disabled."""

    def _make_params_disabled(self):
        return {"nondim": {"enabled": False}}

    def test_disabled_returns_identity_scales(self):
        from Nondim.transform import build_model_scaling
        params = self._make_params_disabled()
        scaling = build_model_scaling(
            params=params,
            n_species=2,
            dt=0.1,
            t_end=10.0,
            D_vals=[1e-9, 2e-9],
            c0_vals=[100.0, 100.0],
            phi_applied=0.05,
            phi0=0.0,
        )
        assert scaling["enabled"] is False
        assert scaling["potential_scale_v"] == 1.0
        assert scaling["length_scale_m"] == 1.0
        assert scaling["concentration_scale_mol_m3"] == 1.0
        assert scaling["diffusivity_scale_m2_s"] == 1.0
        assert scaling["time_scale_s"] == 1.0
        assert scaling["kappa_scale_m_s"] == 1.0

    def test_disabled_passthrough_values(self):
        from Nondim.transform import build_model_scaling
        params = self._make_params_disabled()
        scaling = build_model_scaling(
            params=params,
            n_species=2,
            dt=0.1,
            t_end=10.0,
            D_vals=[1e-9, 2e-9],
            c0_vals=[100.0, 50.0],
            phi_applied=0.05,
            phi0=0.01,
        )
        assert scaling["dt_model"] == pytest.approx(0.1)
        assert scaling["t_end_model"] == pytest.approx(10.0)
        assert scaling["D_model_vals"] == [1e-9, 2e-9]
        assert scaling["c0_model_vals"] == [100.0, 50.0]
        assert scaling["phi_applied_model"] == pytest.approx(0.05)
        assert scaling["phi0_model"] == pytest.approx(0.01)

    def test_disabled_poisson_uses_permittivity(self):
        from Nondim.transform import build_model_scaling
        from Nondim.constants import (
            VACUUM_PERMITTIVITY_F_PER_M,
            DEFAULT_RELATIVE_PERMITTIVITY_WATER,
            FARADAY_CONSTANT,
        )
        params = {"nondim": {"enabled": False}}
        scaling = build_model_scaling(
            params=params, n_species=2, dt=0.1, t_end=1.0,
            D_vals=[1e-9, 1e-9], c0_vals=[1.0, 1.0],
            phi_applied=0.0, phi0=0.0,
        )
        expected_eps = DEFAULT_RELATIVE_PERMITTIVITY_WATER * VACUUM_PERMITTIVITY_F_PER_M
        assert scaling["poisson_coefficient"] == pytest.approx(expected_eps, rel=1e-8)
        assert scaling["charge_rhs_prefactor"] == pytest.approx(FARADAY_CONSTANT, rel=1e-8)


class TestBuildModelScalingEnabled:
    """Test build_model_scaling with nondimensionalization enabled."""

    def _make_params_enabled(self, **overrides):
        """Build a nondim config dict with enabled=True."""
        cfg = {
            "enabled": True,
            "diffusivity_scale_m2_s": 1e-9,
            "concentration_scale_mol_m3": 100.0,
            "length_scale_m": 1e-4,
            "potential_scale_v": 0.02569,  # ~V_T
        }
        cfg.update(overrides)
        return {"nondim": cfg}

    def test_enabled_scales_diffusivities(self):
        from Nondim.transform import build_model_scaling
        params = self._make_params_enabled()
        D_phys = [2e-9, 3e-9]
        scaling = build_model_scaling(
            params=params, n_species=2, dt=0.1, t_end=1.0,
            D_vals=D_phys, c0_vals=[100.0, 50.0],
            phi_applied=0.05, phi0=0.0,
        )
        d_scale = 1e-9
        assert scaling["D_model_vals"][0] == pytest.approx(2e-9 / d_scale, rel=1e-10)
        assert scaling["D_model_vals"][1] == pytest.approx(3e-9 / d_scale, rel=1e-10)

    def test_enabled_scales_concentrations(self):
        from Nondim.transform import build_model_scaling
        params = self._make_params_enabled()
        scaling = build_model_scaling(
            params=params, n_species=2, dt=0.1, t_end=1.0,
            D_vals=[1e-9, 1e-9], c0_vals=[100.0, 50.0],
            phi_applied=0.0, phi0=0.0,
        )
        c_scale = 100.0
        assert scaling["c0_model_vals"][0] == pytest.approx(100.0 / c_scale, rel=1e-10)
        assert scaling["c0_model_vals"][1] == pytest.approx(50.0 / c_scale, rel=1e-10)

    def test_enabled_scales_time(self):
        from Nondim.transform import build_model_scaling
        params = self._make_params_enabled()
        scaling = build_model_scaling(
            params=params, n_species=2, dt=0.5, t_end=10.0,
            D_vals=[1e-9, 1e-9], c0_vals=[100.0, 100.0],
            phi_applied=0.0, phi0=0.0,
        )
        L = 1e-4
        D = 1e-9
        t_scale = L * L / D  # = 10.0 s
        assert scaling["dt_model"] == pytest.approx(0.5 / t_scale, rel=1e-10)
        assert scaling["t_end_model"] == pytest.approx(10.0 / t_scale, rel=1e-10)

    def test_enabled_scales_potential(self):
        from Nondim.transform import build_model_scaling
        V_T = 0.02569
        params = self._make_params_enabled(potential_scale_v=V_T)
        scaling = build_model_scaling(
            params=params, n_species=2, dt=0.1, t_end=1.0,
            D_vals=[1e-9, 1e-9], c0_vals=[100.0, 100.0],
            phi_applied=0.1, phi0=0.02,
        )
        assert scaling["phi_applied_model"] == pytest.approx(0.1 / V_T, rel=1e-6)
        assert scaling["phi0_model"] == pytest.approx(0.02 / V_T, rel=1e-6)

    def test_dimless_inputs_passthrough(self):
        """When all inputs are flagged as dimensionless, values pass through unchanged."""
        from Nondim.transform import build_model_scaling
        params = {
            "nondim": {
                "enabled": True,
                "diffusivity_scale_m2_s": 1e-9,
                "concentration_scale_mol_m3": 100.0,
                "length_scale_m": 1e-4,
                "potential_scale_v": 0.02569,
                "diffusivity_inputs_are_dimensionless": True,
                "concentration_inputs_are_dimensionless": True,
                "potential_inputs_are_dimensionless": True,
                "time_inputs_are_dimensionless": True,
                "kappa_inputs_are_dimensionless": True,
            }
        }
        scaling = build_model_scaling(
            params=params, n_species=2, dt=0.01, t_end=1.0,
            D_vals=[1.5, 1.6], c0_vals=[1.0, 0.0],
            phi_applied=-5.0, phi0=0.0,
        )
        assert scaling["D_model_vals"] == [1.5, 1.6]
        assert scaling["c0_model_vals"] == [1.0, 0.0]
        assert scaling["dt_model"] == pytest.approx(0.01)
        assert scaling["t_end_model"] == pytest.approx(1.0)
        assert scaling["phi_applied_model"] == pytest.approx(-5.0)

    def test_electromigration_prefactor_unity_at_vt(self):
        """When potential_scale = V_T, electromigration prefactor should be ~1."""
        from Nondim.transform import build_model_scaling
        from Nondim.constants import GAS_CONSTANT, FARADAY_CONSTANT
        T = 298.15
        V_T = GAS_CONSTANT * T / FARADAY_CONSTANT
        params = {
            "nondim": {
                "enabled": True,
                "diffusivity_scale_m2_s": 1e-9,
                "concentration_scale_mol_m3": 100.0,
                "length_scale_m": 1e-4,
                "potential_scale_v": V_T,
            }
        }
        scaling = build_model_scaling(
            params=params, n_species=2, dt=0.1, t_end=1.0,
            D_vals=[1e-9, 1e-9], c0_vals=[100.0, 100.0],
            phi_applied=0.0, phi0=0.0,
        )
        assert scaling["electromigration_prefactor"] == pytest.approx(1.0, rel=1e-10)

    def test_flux_and_current_density_scales(self):
        from Nondim.transform import build_model_scaling
        from Nondim.constants import FARADAY_CONSTANT
        D_scale = 1e-9
        c_scale = 100.0
        L = 1e-4
        params = {
            "nondim": {
                "enabled": True,
                "diffusivity_scale_m2_s": D_scale,
                "concentration_scale_mol_m3": c_scale,
                "length_scale_m": L,
            }
        }
        scaling = build_model_scaling(
            params=params, n_species=2, dt=0.1, t_end=1.0,
            D_vals=[1e-9, 1e-9], c0_vals=[100.0, 100.0],
            phi_applied=0.0, phi0=0.0,
        )
        expected_flux = D_scale * c_scale / L
        expected_j = FARADAY_CONSTANT * expected_flux
        assert scaling["flux_scale_mol_m2_s"] == pytest.approx(expected_flux, rel=1e-10)
        assert scaling["current_density_scale_a_m2"] == pytest.approx(expected_j, rel=1e-10)

    def test_zero_diffusivity_raises(self):
        from Nondim.transform import build_model_scaling
        params = {"nondim": {"enabled": True, "diffusivity_scale_m2_s": 1e-9,
                             "concentration_scale_mol_m3": 100.0, "length_scale_m": 1e-4}}
        with pytest.raises(ValueError, match="strictly positive"):
            build_model_scaling(
                params=params, n_species=2, dt=0.1, t_end=1.0,
                D_vals=[0.0, 1e-9], c0_vals=[100.0, 100.0],
                phi_applied=0.0, phi0=0.0,
            )


class TestTransformHelpers:
    """Test internal helper functions from Nondim.transform."""

    def test_as_list_scalar_broadcast(self):
        from Nondim.transform import _as_list
        result = _as_list(3.14, 3, "test")
        assert result == [3.14, 3.14, 3.14]

    def test_as_list_sequence(self):
        from Nondim.transform import _as_list
        result = _as_list([1.0, 2.0], 2, "test")
        assert result == [1.0, 2.0]

    def test_as_list_wrong_length_raises(self):
        from Nondim.transform import _as_list
        with pytest.raises(ValueError, match="must have length 3"):
            _as_list([1.0, 2.0], 3, "test")

    def test_pos_positive(self):
        from Nondim.transform import _pos
        assert _pos(3.14, "x") == 3.14

    def test_pos_zero_raises(self):
        from Nondim.transform import _pos
        with pytest.raises(ValueError, match="must be > 0"):
            _pos(0.0, "x")

    def test_pos_negative_raises(self):
        from Nondim.transform import _pos
        with pytest.raises(ValueError, match="must be > 0"):
            _pos(-1.0, "x")

    def test_bool_true_variants(self):
        from Nondim.transform import _bool
        assert _bool(True) is True
        assert _bool("true") is True
        assert _bool("1") is True
        assert _bool("yes") is True
        assert _bool("on") is True

    def test_bool_false_variants(self):
        from Nondim.transform import _bool
        assert _bool(False) is False
        assert _bool("false") is False
        assert _bool("0") is False
        assert _bool("no") is False
        assert _bool("off") is False


# ===================================================================
# Nondim.compat  —  dict-wrapper functions
# ===================================================================

class TestNondimCompat:
    """Test the backward-compatible dict wrappers in Nondim.compat."""

    def test_build_physical_scales_dict_returns_dict(self, default_nondim_kwargs):
        from Nondim.compat import build_physical_scales_dict
        result = build_physical_scales_dict(**default_nondim_kwargs)
        assert isinstance(result, dict)

    def test_build_physical_scales_dict_matches_to_dict(self, default_nondim_kwargs):
        from Nondim.compat import build_physical_scales_dict
        from Nondim.scales import build_physical_scales
        d = build_physical_scales_dict(**default_nondim_kwargs)
        scales = build_physical_scales(**default_nondim_kwargs)
        expected = scales.to_dict()
        assert set(d.keys()) == set(expected.keys())
        for k in d:
            if isinstance(d[k], float):
                assert d[k] == pytest.approx(expected[k], rel=1e-12), f"Mismatch on key '{k}'"

    def test_build_solver_options_structure(self, default_nondim_kwargs):
        from Nondim.compat import build_physical_scales_dict, build_solver_options
        scales = build_physical_scales_dict(**default_nondim_kwargs)
        opts = build_solver_options(scales)
        assert isinstance(opts, dict)
        assert "snes_type" in opts
        assert "robin_bc" in opts
        assert "nondim" in opts
        assert opts["nondim"]["enabled"] is True
        assert isinstance(opts["robin_bc"]["kappa"], list)
