"""Textbook verification audit for Nondim package (scales.py, constants.py).

This file independently validates the correctness of nondimensionalization
computations by comparing library outputs against values computed from
first-principles textbook formulas using NIST CODATA physical constants.

Audit of existing tests in test_nondim.py
-----------------------------------------
The following existing test classes were reviewed and confirmed correct:

- TestConstants: All 6 tests verified against NIST CODATA 2018 values.
  Each assertion matches the values defined in Nondim/constants.py exactly.
  Constants: F=96485.3329 C/mol, R=8.314462618 J/(mol*K), T=298.15 K,
  eps_0=8.8541878128e-12 F/m, eps_r=78.5, MOLAR_TO_MOL_PER_M3=1000.

- TestNondimScales: All 14 tests confirmed correct.
  - test_default_construction: type checks only, correct.
  - test_d_ref_is_geometric_mean: uses math.sqrt(d1*d2) for 2-species, correct
    (equivalent to exp(mean(log(D))) for n=2).
  - test_bulk_concentration_conversion: c_bulk_m * 1000 = mol/m3, correct.
  - test_c_inf_conversion: same pattern, correct.
  - test_thermal_voltage: R*T/F with sanity check 25.5-26.0 mV, correct.
  - test_time_scale: L^2/D_ref, correct.
  - test_kappa_scale: D_ref/L, correct.
  - test_molar_flux_scale: D_ref*c_ref/L, correct.
  - test_current_density_scale: F*J_ref, correct.
  - test_debye_length: sqrt(eps*V_T/(F*c_ref)), correct.
  - test_debye_ratio: lambda_D/L_ref, correct.
  - test_diffusivity_scale_equals_d_ref: identity check, correct.
  - test_concentration_scale_equals_bulk: identity check, correct.

- TestNondimScalesToDict: All 3 tests confirmed correct.
  Key mapping and cm2 conversion both verified.

- TestBuildPhysicalScalesValidation: All 9 tests confirmed correct.
  Boundary conditions (zero, negative) properly tested.

- TestBuildModelScalingDisabled: All 3 tests confirmed correct.
  Identity scales and permittivity passthrough verified.

- TestBuildModelScalingEnabled: All 8 tests confirmed correct.
  Scaling division by reference values verified for D, c, t, phi.
  Electromigration prefactor = 1 at V_T confirmed.
  Flux and current density scale formulas confirmed.

- TestTransformHelpers: All 8 tests confirmed correct.
  _as_list, _pos, _bool all tested for expected behavior.

- TestNondimCompat: All 3 tests confirmed correct.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from Nondim.constants import (
    FARADAY_CONSTANT,
    GAS_CONSTANT,
    DEFAULT_TEMPERATURE_K,
    VACUUM_PERMITTIVITY_F_PER_M,
    DEFAULT_RELATIVE_PERMITTIVITY_WATER,
    MOLAR_TO_MOL_PER_M3,
)
from Nondim.scales import build_physical_scales


# NIST CODATA 2018 reference values (used for independent verification)
_R = 8.314462618        # J / (mol * K)
_F = 96485.3329         # C / mol
_T = 298.15             # K (25 degC)
_EPS_0 = 8.8541878128e-12  # F / m (vacuum permittivity)
_EPS_R = 78.5           # dimensionless (water at ~25 degC)


class TestTextbookVerification:
    """Independent textbook formula verification for Nondim/scales.py.

    Every test computes the expected value from first principles using
    NIST constants defined above, then compares against the library output.
    """

    def test_thermal_voltage_textbook(self):
        """V_T = R*T/F, should be ~25.69 mV at 298.15 K."""
        V_T_expected = _R * _T / _F

        # Sanity: roughly 25.69 mV
        assert 0.02565 < V_T_expected < 0.02575

        scales = build_physical_scales(
            d_species_m2_s=(1e-9,),
            c_bulk_m=0.1,
            temperature_k=_T,
        )
        assert scales.thermal_voltage_v == pytest.approx(V_T_expected, rel=1e-10)

    def test_debye_length_textbook(self):
        """lambda_D = sqrt(eps_r * eps_0 * V_T / (F * c_ref)) for 0.1 M 1:1 electrolyte.

        Expected range: ~0.97 nm (between 0.5e-9 and 2e-9 m).
        """
        c_bulk_m = 0.1  # mol/L
        c_ref = c_bulk_m * 1000.0  # mol/m3
        eps = _EPS_R * _EPS_0
        V_T = _R * _T / _F

        lambda_D_expected = math.sqrt(eps * V_T / (_F * c_ref))

        # Sanity: for 0.1 M, Debye length should be sub-nanometer to few nm
        assert 0.5e-9 < lambda_D_expected < 2e-9

        scales = build_physical_scales(
            d_species_m2_s=(1e-9,),
            c_bulk_m=c_bulk_m,
            temperature_k=_T,
            relative_permittivity=_EPS_R,
        )
        assert scales.debye_length_m == pytest.approx(lambda_D_expected, rel=1e-10)

    def test_diffusivity_scale_is_geometric_mean(self):
        """For D = [1e-9, 4e-9], d_ref = exp(mean(log(D))) = 2e-9."""
        D = [1e-9, 4e-9]
        d_ref_expected = float(np.exp(np.mean(np.log(D))))
        # Geometric mean of 1e-9 and 4e-9 = 2e-9
        assert d_ref_expected == pytest.approx(2e-9, rel=1e-10)

        scales = build_physical_scales(
            d_species_m2_s=D,
            c_bulk_m=0.1,
        )
        assert scales.d_ref_m2_s == pytest.approx(d_ref_expected, rel=1e-10)

    def test_concentration_scale_is_bulk_converted(self):
        """c_ref = c_bulk_m * 1000 mol/m3 (build_physical_scales uses bulk concentration).

        Note: build_physical_scales uses c_bulk as the concentration scale,
        while build_model_scaling auto-computes max(abs(c_all)) when no explicit
        scale is provided. This test verifies the scales.py behavior.
        """
        c_bulk_m = 0.5  # mol/L
        c_ref_expected = c_bulk_m * 1000.0  # 500 mol/m3

        scales = build_physical_scales(
            d_species_m2_s=(1e-9,),
            c_bulk_m=c_bulk_m,
        )
        assert scales.concentration_scale_mol_m3 == pytest.approx(c_ref_expected, rel=1e-10)

    def test_time_scale_formula(self):
        """t_ref = L^2 / D_ref."""
        D = [2e-9, 8e-9]
        L = 1e-4
        d_ref = float(np.exp(np.mean(np.log(D))))
        t_ref_expected = L * L / d_ref

        scales = build_physical_scales(
            d_species_m2_s=D,
            c_bulk_m=0.1,
            length_scale_m=L,
        )
        assert scales.time_s == pytest.approx(t_ref_expected, rel=1e-10)

    def test_kappa_scale_formula(self):
        """kappa_ref = D_ref / L_ref."""
        D = [3e-9, 5e-9]
        L = 2e-4
        d_ref = float(np.exp(np.mean(np.log(D))))
        kappa_expected = d_ref / L

        scales = build_physical_scales(
            d_species_m2_s=D,
            c_bulk_m=0.1,
            length_scale_m=L,
        )
        assert scales.kappa_m_s == pytest.approx(kappa_expected, rel=1e-10)

    def test_molar_flux_scale_formula(self):
        """J_ref = D_ref * c_ref / L_ref."""
        D = [1e-9, 4e-9]
        c_bulk_m = 0.2
        L = 1e-4
        d_ref = float(np.exp(np.mean(np.log(D))))
        c_ref = c_bulk_m * 1000.0
        J_ref_expected = d_ref * c_ref / L

        scales = build_physical_scales(
            d_species_m2_s=D,
            c_bulk_m=c_bulk_m,
            length_scale_m=L,
        )
        assert scales.molar_flux_mol_m2_s == pytest.approx(J_ref_expected, rel=1e-10)

    def test_current_density_scale_formula(self):
        """j_ref = F * J_ref = F * D_ref * c_ref / L_ref."""
        D = [1e-9, 4e-9]
        c_bulk_m = 0.2
        L = 1e-4
        d_ref = float(np.exp(np.mean(np.log(D))))
        c_ref = c_bulk_m * 1000.0
        J_ref = d_ref * c_ref / L
        j_ref_expected = _F * J_ref

        scales = build_physical_scales(
            d_species_m2_s=D,
            c_bulk_m=c_bulk_m,
            length_scale_m=L,
        )
        assert scales.current_density_a_m2 == pytest.approx(j_ref_expected, rel=1e-10)

    def test_debye_ratio_formula(self):
        """debye_ratio = lambda_D / L_ref (NOT squared).

        Note: The Poisson coefficient (lambda_D/L)^2 is computed separately
        in transform.py. The NondimScales.debye_ratio is the unsquared ratio.
        """
        c_bulk_m = 0.1
        L = 1e-4
        c_ref = c_bulk_m * 1000.0
        eps = _EPS_R * _EPS_0
        V_T = _R * _T / _F
        lambda_D = math.sqrt(eps * V_T / (_F * c_ref))
        ratio_expected = lambda_D / L

        scales = build_physical_scales(
            d_species_m2_s=(1e-9,),
            c_bulk_m=c_bulk_m,
            length_scale_m=L,
            relative_permittivity=_EPS_R,
            temperature_k=_T,
        )
        assert scales.debye_ratio == pytest.approx(ratio_expected, rel=1e-10)
