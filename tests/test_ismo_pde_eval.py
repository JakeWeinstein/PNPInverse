"""Tests for Surrogate.ismo_pde_eval -- data integration round-trip tests.

These tests use synthetic data and do NOT require Firedrake.
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from Surrogate.ismo_pde_eval import (
    AugmentedDataset,
    PDEEvalResult,
    PDESolverBundle,
    QualityReport,
    SurrogatePDEComparison,
    check_pde_quality,
    compare_surrogate_vs_pde,
    integrate_new_data,
)
from Surrogate.sampling import ParameterBounds


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_ETA = 22  # matches real voltage grid


def _make_pde_result(
    n_candidates: int = 10,
    n_valid: int = 8,
    n_eta: int = N_ETA,
    seed: int = 42,
) -> PDEEvalResult:
    """Create a synthetic PDEEvalResult for testing."""
    rng = np.random.default_rng(seed)
    candidate_params = rng.random((n_candidates, 4))
    # Scale to plausible ranges
    candidate_params[:, 0] *= 1e-3  # k0_1
    candidate_params[:, 1] *= 1e-4  # k0_2
    candidate_params[:, 2] = 0.1 + candidate_params[:, 2] * 0.8  # alpha_1
    candidate_params[:, 3] = 0.1 + candidate_params[:, 3] * 0.8  # alpha_2

    valid_mask = np.zeros(n_candidates, dtype=bool)
    valid_mask[:n_valid] = True

    valid_params = candidate_params[:n_valid]
    cd = rng.standard_normal((n_valid, n_eta)) * 5.0
    pc = rng.standard_normal((n_valid, n_eta)) * 2.0
    timings = rng.random(n_candidates) * 10.0

    return PDEEvalResult(
        candidate_params=candidate_params,
        current_density=cd,
        peroxide_current=pc,
        valid_mask=valid_mask,
        timings=timings,
        n_valid=n_valid,
        n_failed=n_candidates - n_valid,
        valid_params=valid_params,
    )


def _make_existing_npz(path: str, n_samples: int = 50, n_eta: int = N_ETA, seed: int = 0):
    """Create a synthetic training data .npz file."""
    rng = np.random.default_rng(seed)
    params = rng.random((n_samples, 4))
    params[:, 0] *= 1e-3
    params[:, 1] *= 1e-4
    params[:, 2] = 0.1 + params[:, 2] * 0.8
    params[:, 3] = 0.1 + params[:, 3] * 0.8

    cd = rng.standard_normal((n_samples, n_eta)) * 5.0
    pc = rng.standard_normal((n_samples, n_eta)) * 2.0
    phi = np.linspace(-0.5, 0.5, n_eta)

    np.savez_compressed(
        path,
        parameters=params,
        current_density=cd,
        peroxide_current=pc,
        phi_applied=phi,
        n_existing=np.int64(40),
        n_gapfill=np.int64(10),
    )
    return params, cd, pc, phi


class _MockSurrogate:
    """Duck-typed surrogate with predict_batch() for testing."""

    def __init__(self, offset: float = 0.0):
        self._offset = offset

    def predict_batch(self, params: np.ndarray) -> dict:
        """Return predictions with a controllable offset from 'truth'."""
        rng = np.random.default_rng(hash(params.tobytes()) % (2**31))
        n = params.shape[0]
        return {
            "current_density": rng.standard_normal((n, N_ETA)) * 5.0 + self._offset,
            "peroxide_current": rng.standard_normal((n, N_ETA)) * 2.0 + self._offset,
        }


# ---------------------------------------------------------------------------
# Tests: dataclass construction
# ---------------------------------------------------------------------------


class TestDataclasses:
    """Verify all dataclasses are frozen."""

    def test_pde_eval_result_is_frozen(self):
        result = _make_pde_result()
        with pytest.raises(AttributeError):
            result.n_valid = 999  # type: ignore[misc]

    def test_quality_report_is_frozen(self):
        report = QualityReport(
            n_candidates=10,
            n_converged=8,
            n_nan_detected=0,
            n_extreme_values=0,
            n_bounds_violations=0,
            n_passed_all_checks=8,
            flagged_indices=np.array([], dtype=int),
            flags=[],
            passed=True,
        )
        with pytest.raises(AttributeError):
            report.passed = False  # type: ignore[misc]

    def test_augmented_dataset_is_frozen(self):
        ds = AugmentedDataset(
            output_path="/tmp/test.npz",
            n_original=50,
            n_new=10,
            n_total=60,
            provenance=np.array(["original"] * 50 + ["ismo_iter_1"] * 10),
        )
        with pytest.raises(AttributeError):
            ds.n_total = 999  # type: ignore[misc]

    def test_surrogate_pde_comparison_is_frozen(self):
        comp = SurrogatePDEComparison(
            candidate_params=np.empty((0, 4)),
            cd_nrmse_per_candidate=np.array([]),
            pc_nrmse_per_candidate=np.array([]),
            cd_rmse_per_candidate=np.array([]),
            pc_rmse_per_candidate=np.array([]),
            cd_max_error=0.0,
            pc_max_error=0.0,
            cd_mean_nrmse=0.0,
            pc_mean_nrmse=0.0,
            is_converged=True,
        )
        with pytest.raises(AttributeError):
            comp.is_converged = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Tests: integrate_new_data round-trip
# ---------------------------------------------------------------------------


class TestIntegrateNewData:
    """Data integration round-trip tests."""

    def test_basic_integration(self, tmp_path):
        """Merge new data into existing dataset and verify shapes."""
        existing_path = str(tmp_path / "existing.npz")
        output_path = str(tmp_path / "augmented.npz")
        old_params, old_cd, old_pc, phi = _make_existing_npz(existing_path, n_samples=50)
        pde_result = _make_pde_result(n_candidates=10, n_valid=8)

        result = integrate_new_data(
            pde_result,
            existing_data_path=existing_path,
            output_path=output_path,
            iteration_tag="ismo_iter_1",
        )

        assert isinstance(result, AugmentedDataset)
        assert result.n_original == 50
        assert result.n_new == 8
        assert result.n_total == 58
        assert result.output_path == output_path
        assert len(result.provenance) == 58

        # Verify the saved file is loadable and has correct shapes
        saved = np.load(output_path, allow_pickle=True)
        assert saved["parameters"].shape == (58, 4)
        assert saved["current_density"].shape == (58, N_ETA)
        assert saved["peroxide_current"].shape == (58, N_ETA)
        assert saved["phi_applied"].shape == (N_ETA,)

        # Original data is preserved exactly
        np.testing.assert_array_equal(saved["parameters"][:50], old_params)
        np.testing.assert_array_equal(saved["current_density"][:50], old_cd)
        np.testing.assert_array_equal(saved["peroxide_current"][:50], old_pc)

    def test_provenance_arrays_present(self, tmp_path):
        """Verify provenance arrays are saved in the augmented file."""
        existing_path = str(tmp_path / "existing.npz")
        output_path = str(tmp_path / "augmented.npz")
        _make_existing_npz(existing_path, n_samples=20)
        pde_result = _make_pde_result(n_candidates=5, n_valid=4)

        integrate_new_data(
            pde_result,
            existing_data_path=existing_path,
            output_path=output_path,
            iteration_tag="ismo_iter_3",
        )

        saved = np.load(output_path, allow_pickle=True)
        source = saved["provenance_source"]
        assert len(source) == 24
        assert all(s == "original" for s in source[:20])
        assert all(s == "ismo_iter_3" for s in source[20:])

    def test_extra_keys_carried_forward(self, tmp_path):
        """Extra metadata keys from the existing file are preserved."""
        existing_path = str(tmp_path / "existing.npz")
        output_path = str(tmp_path / "augmented.npz")
        _make_existing_npz(existing_path, n_samples=20)
        pde_result = _make_pde_result(n_candidates=5, n_valid=3)

        integrate_new_data(
            pde_result,
            existing_data_path=existing_path,
            output_path=output_path,
        )

        saved = np.load(output_path, allow_pickle=True)
        # n_existing and n_gapfill should be carried forward
        assert "n_existing" in saved
        assert "n_gapfill" in saved
        assert int(saved["n_existing"]) == 40
        assert int(saved["n_gapfill"]) == 10

    def test_ismo_metadata_saved(self, tmp_path):
        """ISMO metadata JSON is saved in the augmented file."""
        existing_path = str(tmp_path / "existing.npz")
        output_path = str(tmp_path / "augmented.npz")
        _make_existing_npz(existing_path, n_samples=20)
        pde_result = _make_pde_result(n_candidates=5, n_valid=4)

        comparison = SurrogatePDEComparison(
            candidate_params=pde_result.valid_params,
            cd_nrmse_per_candidate=np.array([0.01, 0.02, 0.015, 0.03]),
            pc_nrmse_per_candidate=np.array([0.005, 0.01, 0.008, 0.02]),
            cd_rmse_per_candidate=np.array([0.1, 0.2, 0.15, 0.3]),
            pc_rmse_per_candidate=np.array([0.05, 0.1, 0.08, 0.2]),
            cd_max_error=0.03,
            pc_max_error=0.02,
            cd_mean_nrmse=0.01875,
            pc_mean_nrmse=0.01075,
            is_converged=True,
        )

        integrate_new_data(
            pde_result,
            existing_data_path=existing_path,
            output_path=output_path,
            iteration_tag="ismo_iter_2",
            comparison=comparison,
        )

        saved = np.load(output_path, allow_pickle=True)
        meta = json.loads(str(saved["ismo_metadata"]))
        assert meta["iteration_tag"] == "ismo_iter_2"
        assert meta["n_original"] == 20
        assert meta["n_new"] == 4
        assert meta["is_converged"] is True

    def test_original_file_not_modified(self, tmp_path):
        """The existing data file must not be overwritten."""
        existing_path = str(tmp_path / "existing.npz")
        output_path = str(tmp_path / "augmented.npz")
        _make_existing_npz(existing_path, n_samples=20)

        # Record modification time
        mtime_before = os.path.getmtime(existing_path)

        pde_result = _make_pde_result(n_candidates=5, n_valid=4)
        integrate_new_data(
            pde_result,
            existing_data_path=existing_path,
            output_path=output_path,
        )

        mtime_after = os.path.getmtime(existing_path)
        assert mtime_before == mtime_after

    def test_all_failed_is_noop(self, tmp_path):
        """When n_valid=0, integration is a no-op returning original path."""
        existing_path = str(tmp_path / "existing.npz")
        output_path = str(tmp_path / "augmented.npz")
        _make_existing_npz(existing_path, n_samples=20)

        pde_result = _make_pde_result(n_candidates=5, n_valid=0)

        result = integrate_new_data(
            pde_result,
            existing_data_path=existing_path,
            output_path=output_path,
        )

        assert result.n_new == 0
        assert result.n_total == 20
        assert result.output_path == existing_path
        assert not os.path.exists(output_path)

    def test_provenance_nrmse_with_comparison(self, tmp_path):
        """Per-candidate NRMSE values are stored when comparison is provided."""
        existing_path = str(tmp_path / "existing.npz")
        output_path = str(tmp_path / "augmented.npz")
        _make_existing_npz(existing_path, n_samples=10)
        pde_result = _make_pde_result(n_candidates=5, n_valid=3)

        nrmse_cd = np.array([0.01, 0.02, 0.03])
        nrmse_pc = np.array([0.005, 0.01, 0.015])
        comparison = SurrogatePDEComparison(
            candidate_params=pde_result.valid_params,
            cd_nrmse_per_candidate=nrmse_cd,
            pc_nrmse_per_candidate=nrmse_pc,
            cd_rmse_per_candidate=np.zeros(3),
            pc_rmse_per_candidate=np.zeros(3),
            cd_max_error=0.03,
            pc_max_error=0.015,
            cd_mean_nrmse=0.02,
            pc_mean_nrmse=0.01,
            is_converged=True,
        )

        integrate_new_data(
            pde_result,
            existing_data_path=existing_path,
            output_path=output_path,
            comparison=comparison,
        )

        saved = np.load(output_path, allow_pickle=True)
        nrmse_cd_saved = saved["provenance_nrmse_cd"]
        assert np.all(np.isnan(nrmse_cd_saved[:10]))  # original samples
        np.testing.assert_allclose(nrmse_cd_saved[10:], nrmse_cd)


# ---------------------------------------------------------------------------
# Tests: compare_surrogate_vs_pde
# ---------------------------------------------------------------------------


class TestCompareSurrogateVsPDE:
    """Test surrogate-PDE comparison logic."""

    def test_basic_comparison(self):
        """Comparison returns valid structure with correct sizes."""
        pde_result = _make_pde_result(n_candidates=10, n_valid=8)
        surrogate = _MockSurrogate(offset=0.0)

        comp = compare_surrogate_vs_pde(
            surrogate,
            pde_result,
            convergence_threshold=0.02,
            cd_reference_range=10.0,
            pc_reference_range=5.0,
        )

        assert isinstance(comp, SurrogatePDEComparison)
        assert comp.cd_nrmse_per_candidate.shape == (8,)
        assert comp.pc_nrmse_per_candidate.shape == (8,)
        assert isinstance(comp.is_converged, bool)

    def test_convergence_detected(self):
        """When surrogate matches PDE exactly, is_converged should be True."""
        # Create a mock surrogate that returns exactly the PDE truth
        pde_result = _make_pde_result(n_candidates=5, n_valid=5)

        class PerfectSurrogate:
            def __init__(self, cd, pc):
                self._cd = cd
                self._pc = pc

            def predict_batch(self, params):
                return {
                    "current_density": self._cd.copy(),
                    "peroxide_current": self._pc.copy(),
                }

        surrogate = PerfectSurrogate(
            pde_result.current_density, pde_result.peroxide_current
        )
        comp = compare_surrogate_vs_pde(
            surrogate, pde_result, convergence_threshold=0.01
        )

        assert comp.is_converged is True
        assert comp.cd_mean_nrmse == 0.0
        assert comp.pc_mean_nrmse == 0.0

    def test_all_failed_sentinel(self):
        """When n_valid=0, returns sentinel with is_converged=False."""
        pde_result = _make_pde_result(n_candidates=5, n_valid=0)
        surrogate = _MockSurrogate()

        comp = compare_surrogate_vs_pde(surrogate, pde_result)

        assert comp.is_converged is False
        assert comp.cd_mean_nrmse == float("inf")
        assert comp.pc_mean_nrmse == float("inf")
        assert comp.candidate_params.shape == (0, 4)

    def test_reference_range_affects_nrmse(self):
        """Using a fixed reference range should produce different NRMSE.

        The reference range sets the floor for per-sample normalization.
        We use flat truth curves (ptp ~ 0) so the floor dominates, making
        the effect of the reference range visible.
        """
        rng = np.random.default_rng(99)
        n_valid = 5
        # Flat truth curves so per-sample ptp ~ 0 -> floor dominates
        flat_cd = np.ones((n_valid, N_ETA)) * 3.0
        flat_pc = np.ones((n_valid, N_ETA)) * 1.0
        params = rng.random((n_valid, 4))

        pde_result = PDEEvalResult(
            candidate_params=params,
            current_density=flat_cd,
            peroxide_current=flat_pc,
            valid_mask=np.ones(n_valid, dtype=bool),
            timings=np.zeros(n_valid),
            n_valid=n_valid,
            n_failed=0,
            valid_params=params,
        )

        # Surrogate that adds a constant offset -> nonzero RMSE
        class OffsetSurrogate:
            def predict_batch(self, p):
                return {
                    "current_density": np.ones((len(p), N_ETA)) * 3.5,
                    "peroxide_current": np.ones((len(p), N_ETA)) * 1.5,
                }

        surrogate = OffsetSurrogate()

        comp_small_ref = compare_surrogate_vs_pde(
            surrogate, pde_result, cd_reference_range=1.0, pc_reference_range=1.0
        )
        comp_large_ref = compare_surrogate_vs_pde(
            surrogate, pde_result, cd_reference_range=100.0, pc_reference_range=100.0
        )

        # Larger reference range -> larger floor -> smaller NRMSE
        assert comp_large_ref.cd_mean_nrmse < comp_small_ref.cd_mean_nrmse


# ---------------------------------------------------------------------------
# Tests: check_pde_quality
# ---------------------------------------------------------------------------


class TestCheckPDEQuality:
    """Test quality check logic."""

    def test_clean_data_passes(self):
        """Normal data with no issues should pass all checks."""
        pde_result = _make_pde_result(n_candidates=10, n_valid=8)
        bounds = ParameterBounds(
            k0_1_range=(0.0, 1.0),
            k0_2_range=(0.0, 1.0),
            alpha_1_range=(0.0, 1.0),
            alpha_2_range=(0.0, 1.0),
        )

        report = check_pde_quality(pde_result, bounds)

        assert isinstance(report, QualityReport)
        assert report.n_converged == 8
        assert report.n_nan_detected == 0
        assert report.passed is True

    def test_nan_detection(self):
        """NaN in CD/PC curves should be flagged."""
        pde_result = _make_pde_result(n_candidates=5, n_valid=5)
        # Inject NaN into first sample
        cd = pde_result.current_density.copy()
        cd[0, 5] = np.nan
        pde_result_with_nan = PDEEvalResult(
            candidate_params=pde_result.candidate_params,
            current_density=cd,
            peroxide_current=pde_result.peroxide_current,
            valid_mask=pde_result.valid_mask,
            timings=pde_result.timings,
            n_valid=pde_result.n_valid,
            n_failed=pde_result.n_failed,
            valid_params=pde_result.valid_params,
        )
        bounds = ParameterBounds(
            k0_1_range=(0.0, 1.0),
            k0_2_range=(0.0, 1.0),
            alpha_1_range=(0.0, 1.0),
            alpha_2_range=(0.0, 1.0),
        )

        report = check_pde_quality(pde_result_with_nan, bounds)

        assert report.n_nan_detected == 1
        assert report.passed is False
        assert 0 in report.flagged_indices

    def test_extreme_value_detection(self):
        """Extreme CD/PC values should be flagged."""
        pde_result = _make_pde_result(n_candidates=5, n_valid=5)
        cd = pde_result.current_density.copy()
        cd[2, 10] = 100.0  # extreme value
        pde_result_extreme = PDEEvalResult(
            candidate_params=pde_result.candidate_params,
            current_density=cd,
            peroxide_current=pde_result.peroxide_current,
            valid_mask=pde_result.valid_mask,
            timings=pde_result.timings,
            n_valid=pde_result.n_valid,
            n_failed=pde_result.n_failed,
            valid_params=pde_result.valid_params,
        )
        bounds = ParameterBounds(
            k0_1_range=(0.0, 1.0),
            k0_2_range=(0.0, 1.0),
            alpha_1_range=(0.0, 1.0),
            alpha_2_range=(0.0, 1.0),
        )

        report = check_pde_quality(pde_result_extreme, bounds)

        assert report.n_extreme_values >= 1
        assert report.passed is False
        assert 2 in report.flagged_indices

    def test_bounds_violation_detection(self):
        """Out-of-bounds parameters should be flagged."""
        pde_result = _make_pde_result(n_candidates=5, n_valid=5)
        tight_bounds = ParameterBounds(
            k0_1_range=(1e-10, 1e-10),  # impossibly tight
            k0_2_range=(0.0, 1.0),
            alpha_1_range=(0.0, 1.0),
            alpha_2_range=(0.0, 1.0),
        )

        report = check_pde_quality(pde_result, tight_bounds)

        assert report.n_bounds_violations >= 1
        assert report.passed is False

    def test_all_failed_report(self):
        """When n_valid=0, report should indicate failure."""
        pde_result = _make_pde_result(n_candidates=5, n_valid=0)
        bounds = ParameterBounds()

        report = check_pde_quality(pde_result, bounds)

        assert report.n_converged == 0
        assert report.passed is False
        assert len(report.flags) > 0


# ---------------------------------------------------------------------------
# Tests: module-level imports (no Firedrake at module level)
# ---------------------------------------------------------------------------


class TestNoFiredrakeImport:
    """Verify that importing ismo_pde_eval does not trigger Firedrake."""

    def test_import_succeeds_without_firedrake(self):
        """The module should be importable without Firedrake."""
        # If we got this far, the import already succeeded.
        import Surrogate.ismo_pde_eval as mod

        assert hasattr(mod, "PDESolverBundle")
        assert hasattr(mod, "PDEEvalResult")
        assert hasattr(mod, "evaluate_candidates_with_pde")
        assert hasattr(mod, "compare_surrogate_vs_pde")
        assert hasattr(mod, "integrate_new_data")
        assert hasattr(mod, "check_pde_quality")
