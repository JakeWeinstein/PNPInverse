"""Unit tests for Forward.steady_state.common utilities.

These tests exercise pure-Python / NumPy logic and do NOT require Firedrake.
They cover:
- SteadyStateConfig and SteadyStateResult dataclass construction
- add_percent_noise  (pure numpy)
- CSV I/O round-trip  (write_phi_applied_flux_csv + read_phi_applied_flux_csv)
- results_to_flux_array and all_results_converged
- observed_flux_from_species_flux
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from Forward.steady_state.common import (
    SteadyStateConfig,
    SteadyStateResult,
    add_percent_noise,
    write_phi_applied_flux_csv,
    read_phi_applied_flux_csv,
    results_to_flux_array,
    all_results_converged,
    observed_flux_from_species_flux,
)


# ===================================================================
# SteadyStateConfig
# ===================================================================

class TestSteadyStateConfig:
    """Test SteadyStateConfig defaults and custom construction."""

    def test_defaults(self):
        cfg = SteadyStateConfig()
        assert cfg.relative_tolerance == pytest.approx(1e-3)
        assert cfg.absolute_tolerance == pytest.approx(1e-8)
        assert cfg.consecutive_steps == 5
        assert cfg.max_steps == 200
        assert cfg.flux_observable == "total_species"
        assert cfg.species_index is None
        assert cfg.verbose is False
        assert cfg.print_every == 25

    def test_custom_values(self):
        cfg = SteadyStateConfig(
            relative_tolerance=1e-5,
            absolute_tolerance=1e-10,
            consecutive_steps=10,
            max_steps=500,
            flux_observable="species",
            species_index=0,
            verbose=True,
            print_every=50,
        )
        assert cfg.relative_tolerance == pytest.approx(1e-5)
        assert cfg.absolute_tolerance == pytest.approx(1e-10)
        assert cfg.consecutive_steps == 10
        assert cfg.max_steps == 500
        assert cfg.flux_observable == "species"
        assert cfg.species_index == 0
        assert cfg.verbose is True
        assert cfg.print_every == 50


# ===================================================================
# SteadyStateResult
# ===================================================================

class TestSteadyStateResult:
    """Test SteadyStateResult construction and properties."""

    def test_construction(self):
        r = SteadyStateResult(
            phi_applied=0.05,
            converged=True,
            steps_taken=100,
            final_time=10.0,
            species_flux=[0.1, -0.05],
            observed_flux=0.05,
            final_relative_change=1e-5,
            final_absolute_change=1e-10,
        )
        assert r.phi_applied == pytest.approx(0.05)
        assert r.converged is True
        assert r.steps_taken == 100
        assert r.final_time == pytest.approx(10.0)
        assert r.species_flux == [0.1, -0.05]
        assert r.observed_flux == pytest.approx(0.05)
        assert r.failure_reason == ""

    def test_phi0_backward_compat_alias(self):
        r = SteadyStateResult(
            phi_applied=0.123,
            converged=True,
            steps_taken=10,
            final_time=1.0,
            species_flux=[0.0],
            observed_flux=0.0,
            final_relative_change=None,
            final_absolute_change=None,
        )
        assert r.phi0 == pytest.approx(0.123)

    def test_failure_reason(self):
        r = SteadyStateResult(
            phi_applied=0.05,
            converged=False,
            steps_taken=200,
            final_time=20.0,
            species_flux=[0.0, 0.0],
            observed_flux=0.0,
            final_relative_change=0.1,
            final_absolute_change=0.01,
            failure_reason="max_steps exceeded",
        )
        assert r.converged is False
        assert r.failure_reason == "max_steps exceeded"

    def test_none_changes(self):
        """final_relative_change and final_absolute_change can be None."""
        r = SteadyStateResult(
            phi_applied=0.0,
            converged=False,
            steps_taken=1,
            final_time=0.1,
            species_flux=[0.0],
            observed_flux=0.0,
            final_relative_change=None,
            final_absolute_change=None,
            failure_reason="solver exception",
        )
        assert r.final_relative_change is None
        assert r.final_absolute_change is None


# ===================================================================
# add_percent_noise
# ===================================================================

class TestAddPercentNoise:
    """Test the add_percent_noise utility."""

    def test_zero_noise_returns_copy(self):
        values = [1.0, 2.0, 3.0]
        result = add_percent_noise(values, 0.0)
        np.testing.assert_array_equal(result, values)
        # Should be a copy, not the same object
        result[0] = 999.0
        assert values[0] == 1.0

    def test_noise_changes_values(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = add_percent_noise(values, 10.0, seed=42)
        # With 10% noise and seed=42, values should differ
        assert not np.allclose(result, values)

    def test_reproducibility_with_seed(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        r1 = add_percent_noise(values, 10.0, seed=42)
        r2 = add_percent_noise(values, 10.0, seed=42)
        np.testing.assert_array_equal(r1, r2)

    def test_different_seeds_give_different_results(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        r1 = add_percent_noise(values, 10.0, seed=42)
        r2 = add_percent_noise(values, 10.0, seed=99)
        assert not np.array_equal(r1, r2)

    def test_noise_magnitude_scales_with_percent(self):
        """Higher noise percent should produce larger deviations on average."""
        values = np.linspace(1.0, 10.0, 100)
        low_noise = add_percent_noise(values, 1.0, seed=0)
        high_noise = add_percent_noise(values, 50.0, seed=0)
        low_dev = np.std(low_noise - values)
        high_dev = np.std(high_noise - values)
        assert high_dev > low_dev

    def test_negative_noise_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            add_percent_noise([1.0, 2.0], -5.0)

    def test_nan_values_preserved(self):
        """NaN entries should remain NaN after noise addition."""
        values = [1.0, float("nan"), 3.0, float("nan"), 5.0]
        result = add_percent_noise(values, 10.0, seed=42)
        assert np.isnan(result[1])
        assert np.isnan(result[3])
        assert np.isfinite(result[0])
        assert np.isfinite(result[2])
        assert np.isfinite(result[4])

    def test_all_nan_returns_copy(self):
        """If all values are NaN, return a copy without crashing."""
        values = [float("nan"), float("nan")]
        result = add_percent_noise(values, 10.0, seed=42)
        assert np.isnan(result[0])
        assert np.isnan(result[1])

    def test_noise_is_zero_mean_approximately(self):
        """Over many samples, noise should average near zero."""
        values = np.ones(10_000) * 5.0
        noisy = add_percent_noise(values, 10.0, seed=123)
        mean_noise = np.mean(noisy - values)
        # With 10k samples, mean should be well within 0.1
        assert abs(mean_noise) < 0.1


# ===================================================================
# CSV I/O round-trip
# ===================================================================

class TestCSVRoundTrip:
    """Test write_phi_applied_flux_csv + read_phi_applied_flux_csv."""

    def test_write_and_read_clean_only(self, tmp_path, sample_steady_state_results):
        csv_path = str(tmp_path / "subdir" / "test.csv")
        write_phi_applied_flux_csv(csv_path, sample_steady_state_results)
        assert os.path.isfile(csv_path)

        data = read_phi_applied_flux_csv(csv_path, flux_column="flux_clean")
        assert "phi_applied" in data
        assert "flux" in data
        assert len(data["phi_applied"]) == 3
        np.testing.assert_allclose(
            data["phi_applied"], [0.01, 0.02, 0.03], rtol=1e-10
        )
        np.testing.assert_allclose(
            data["flux"], [0.078, 0.156, 0.200], rtol=1e-10
        )

    def test_write_and_read_with_noisy_flux(self, tmp_path, sample_steady_state_results):
        csv_path = str(tmp_path / "noisy.csv")
        noisy_flux = [0.08, 0.16, 0.21]
        write_phi_applied_flux_csv(csv_path, sample_steady_state_results, noisy_flux=noisy_flux)

        data = read_phi_applied_flux_csv(csv_path, flux_column="flux_noisy")
        np.testing.assert_allclose(data["flux"], [0.08, 0.16, 0.21], rtol=1e-10)

    def test_read_falls_back_to_clean_when_noisy_missing(self, tmp_path, sample_steady_state_results):
        """When no noisy column exists, reading flux_noisy should fall back to flux_clean."""
        csv_path = str(tmp_path / "clean_only.csv")
        write_phi_applied_flux_csv(csv_path, sample_steady_state_results)  # no noisy_flux

        data = read_phi_applied_flux_csv(csv_path, flux_column="flux_noisy")
        # Should fall back to flux_clean
        np.testing.assert_allclose(
            data["flux"], [0.078, 0.156, 0.200], rtol=1e-10
        )

    def test_creates_parent_directories(self, tmp_path, sample_steady_state_results):
        deep_path = str(tmp_path / "a" / "b" / "c" / "out.csv")
        write_phi_applied_flux_csv(deep_path, sample_steady_state_results)
        assert os.path.isfile(deep_path)

    def test_round_trip_preserves_precision(self, tmp_path):
        """Values with many decimal places survive the round trip."""
        results = [
            SteadyStateResult(
                phi_applied=0.0123456789012345,
                converged=True,
                steps_taken=10,
                final_time=1.0,
                species_flux=[1.23456789e-10],
                observed_flux=1.23456789e-10,
                final_relative_change=1e-12,
                final_absolute_change=1e-15,
            )
        ]
        csv_path = str(tmp_path / "precision.csv")
        write_phi_applied_flux_csv(csv_path, results)
        data = read_phi_applied_flux_csv(csv_path, flux_column="flux_clean")
        assert data["phi_applied"][0] == pytest.approx(0.0123456789012345, rel=1e-12)
        assert data["flux"][0] == pytest.approx(1.23456789e-10, rel=1e-10)


# ===================================================================
# results_to_flux_array
# ===================================================================

class TestResultsToFluxArray:
    """Test results_to_flux_array helper."""

    def test_basic(self, sample_steady_state_results):
        arr = results_to_flux_array(sample_steady_state_results)
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == float
        assert len(arr) == 3
        np.testing.assert_allclose(arr, [0.078, 0.156, 0.200], rtol=1e-10)

    def test_empty_list(self):
        arr = results_to_flux_array([])
        assert isinstance(arr, np.ndarray)
        assert len(arr) == 0

    def test_single_result(self):
        result = SteadyStateResult(
            phi_applied=0.0, converged=True, steps_taken=1,
            final_time=0.1, species_flux=[1.0], observed_flux=1.0,
            final_relative_change=0.0, final_absolute_change=0.0,
        )
        arr = results_to_flux_array([result])
        assert arr.shape == (1,)
        assert arr[0] == pytest.approx(1.0)


# ===================================================================
# all_results_converged
# ===================================================================

class TestAllResultsConverged:
    """Test all_results_converged helper."""

    def test_all_converged(self):
        results = [
            SteadyStateResult(
                phi_applied=0.0, converged=True, steps_taken=10,
                final_time=1.0, species_flux=[0.0], observed_flux=0.0,
                final_relative_change=None, final_absolute_change=None,
            ),
            SteadyStateResult(
                phi_applied=0.1, converged=True, steps_taken=20,
                final_time=2.0, species_flux=[0.0], observed_flux=0.0,
                final_relative_change=None, final_absolute_change=None,
            ),
        ]
        assert all_results_converged(results) is True

    def test_one_failed(self, sample_steady_state_results):
        # The fixture has one non-converged result
        assert all_results_converged(sample_steady_state_results) is False

    def test_empty_list_is_true(self):
        assert all_results_converged([]) is True


# ===================================================================
# observed_flux_from_species_flux
# ===================================================================

class TestObservedFluxFromSpeciesFlux:
    """Test the flux observable mapping function."""

    def test_total_species(self):
        flux = observed_flux_from_species_flux(
            [0.5, -0.2, 0.1],
            z_vals=[1, -1, 0],
            flux_observable="total_species",
            species_index=None,
        )
        assert flux == pytest.approx(0.4)

    def test_total_charge(self):
        from Nondim.constants import FARADAY_CONSTANT
        species_flux = [0.5, -0.2]
        z_vals = [1, -1]
        flux = observed_flux_from_species_flux(
            species_flux,
            z_vals=z_vals,
            flux_observable="total_charge",
            species_index=None,
        )
        expected = FARADAY_CONSTANT * (1 * 0.5 + (-1) * (-0.2))
        assert flux == pytest.approx(expected, rel=1e-10)

    def test_charge_proxy_no_f(self):
        species_flux = [0.5, -0.2]
        z_vals = [1, -1]
        flux = observed_flux_from_species_flux(
            species_flux,
            z_vals=z_vals,
            flux_observable="charge_proxy_no_f",
            species_index=None,
        )
        expected = 1 * 0.5 + (-1) * (-0.2)
        assert flux == pytest.approx(expected)

    def test_species_mode(self):
        flux = observed_flux_from_species_flux(
            [0.5, -0.2, 0.3],
            z_vals=[1, -1, 0],
            flux_observable="species",
            species_index=1,
        )
        assert flux == pytest.approx(-0.2)

    def test_species_mode_requires_index(self):
        with pytest.raises(ValueError, match="species_index must be set"):
            observed_flux_from_species_flux(
                [0.5, -0.2],
                z_vals=[1, -1],
                flux_observable="species",
                species_index=None,
            )

    def test_species_mode_index_out_of_bounds(self):
        with pytest.raises(ValueError, match="out of bounds"):
            observed_flux_from_species_flux(
                [0.5, -0.2],
                z_vals=[1, -1],
                flux_observable="species",
                species_index=5,
            )

    def test_unknown_observable_raises(self):
        with pytest.raises(ValueError, match="Unknown flux_observable"):
            observed_flux_from_species_flux(
                [0.5, -0.2],
                z_vals=[1, -1],
                flux_observable="invalid_mode",
                species_index=None,
            )

    def test_z_vals_size_mismatch_raises(self):
        with pytest.raises(ValueError, match="does not match"):
            observed_flux_from_species_flux(
                [0.5, -0.2],
                z_vals=[1, -1, 0],  # 3 vs 2
                flux_observable="total_charge",
                species_index=None,
            )
