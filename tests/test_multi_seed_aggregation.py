"""Tests for multi-seed v13 aggregation logic.

Validates parse_v13_csv and aggregate_seed_results with mock data so that
expected median/IQR/max can be computed by hand.
"""

from __future__ import annotations

import csv
import os
import sys
import textwrap

import numpy as np
import pytest

# Ensure project root is importable
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.studies.run_multi_seed_v13 import (
    aggregate_seed_results,
    parse_v13_csv,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CSV_HEADER = [
    "phase", "k0_1", "k0_2", "alpha_1", "alpha_2",
    "k0_1_err_pct", "k0_2_err_pct", "alpha_1_err_pct", "alpha_2_err_pct",
    "loss", "time_s",
]

# Seven rows mimicking real v13 output (S1-S4, P1, P2)
MOCK_ROWS = [
    ["S1_alpha", "1.0e-03", "5.0e-05", "0.620", "0.490", "10.0", "20.0", "5.0", "8.0", "1.0e-03", "0.1"],
    ["S2_joint", "1.1e-03", "4.8e-05", "0.625", "0.495", "8.0", "15.0", "3.0", "6.0", "5.0e-04", "0.5"],
    ["S3_cascade_p1", "1.15e-03", "4.9e-05", "0.626", "0.498", "6.0", "12.0", "2.0", "4.0", "3.0e-04", "3.0"],
    ["S3_cascade_p2", "1.16e-03", "4.95e-05", "0.626", "0.499", "5.5", "11.0", "1.5", "3.5", "2.5e-04", "3.0"],
    ["S4_multistart", "1.18e-03", "5.0e-05", "0.627", "0.500", "4.0", "9.0", "1.0", "2.0", "1.0e-04", "5.0"],
    ["P1_pde_shallow", "1.20e-03", "5.1e-05", "0.627", "0.500", "2.0", "5.0", "0.5", "1.0", "5.0e-05", "80.0"],
    ["P2_pde_full_cathodic", "1.25e-03", "5.25e-05", "0.627", "0.500", "1.0", "3.0", "0.2", "0.5", "1.0e-05", "200.0"],
]


@pytest.fixture()
def mock_csv(tmp_path):
    """Write a mock master_comparison_v13.csv and return its path."""
    csv_path = tmp_path / "master_comparison_v13.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for row in MOCK_ROWS:
            writer.writerow(row)
    return str(csv_path)


@pytest.fixture()
def mock_csv_no_p2(tmp_path):
    """CSV with no P2 row."""
    csv_path = tmp_path / "master_comparison_v13.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        # Only surrogate rows, no P2
        for row in MOCK_ROWS[:5]:
            writer.writerow(row)
    return str(csv_path)


# ---------------------------------------------------------------------------
# Tests: parse_v13_csv
# ---------------------------------------------------------------------------


class TestParseV13CSV:
    """Verify P2 row extraction from v13 output CSV."""

    def test_extracts_p2_row(self, mock_csv):
        result = parse_v13_csv(mock_csv)
        assert result is not None
        assert float(result["k0_1_err_pct"]) == pytest.approx(1.0)
        assert float(result["k0_2_err_pct"]) == pytest.approx(3.0)
        assert float(result["alpha_1_err_pct"]) == pytest.approx(0.2)
        assert float(result["alpha_2_err_pct"]) == pytest.approx(0.5)

    def test_returns_none_when_no_p2(self, mock_csv_no_p2):
        result = parse_v13_csv(mock_csv_no_p2)
        assert result is None


# ---------------------------------------------------------------------------
# Tests: aggregate_seed_results
# ---------------------------------------------------------------------------


class TestAggregateSeedResults:
    """Verify median/IQR/max computation across 5 mock seeds."""

    @pytest.fixture()
    def five_seed_results(self):
        """5 seeds with known error percentages for hand-verifiable stats.

        k0_1_err_pct: [1, 2, 3, 4, 5]  -> median=3, p25=2, p75=4, max=5
        k0_2_err_pct: [2, 4, 6, 8, 10] -> median=6, p25=4, p75=8, max=10
        alpha_1_err_pct: [0.1, 0.2, 0.3, 0.4, 0.5] -> median=0.3, p25=0.2, p75=0.4, max=0.5
        alpha_2_err_pct: [0.5, 1.0, 1.5, 2.0, 2.5] -> median=1.5, p25=1.0, p75=2.0, max=2.5
        """
        return [
            {"k0_1_err_pct": 1.0, "k0_2_err_pct": 2.0, "alpha_1_err_pct": 0.1, "alpha_2_err_pct": 0.5},
            {"k0_1_err_pct": 2.0, "k0_2_err_pct": 4.0, "alpha_1_err_pct": 0.2, "alpha_2_err_pct": 1.0},
            {"k0_1_err_pct": 3.0, "k0_2_err_pct": 6.0, "alpha_1_err_pct": 0.3, "alpha_2_err_pct": 1.5},
            {"k0_1_err_pct": 4.0, "k0_2_err_pct": 8.0, "alpha_1_err_pct": 0.4, "alpha_2_err_pct": 2.0},
            {"k0_1_err_pct": 5.0, "k0_2_err_pct": 10.0, "alpha_1_err_pct": 0.5, "alpha_2_err_pct": 2.5},
        ]

    def test_median(self, five_seed_results):
        stats = aggregate_seed_results(five_seed_results)
        assert stats["k0_1_err_pct"]["median"] == pytest.approx(3.0)
        assert stats["k0_2_err_pct"]["median"] == pytest.approx(6.0)
        assert stats["alpha_1_err_pct"]["median"] == pytest.approx(0.3)
        assert stats["alpha_2_err_pct"]["median"] == pytest.approx(1.5)

    def test_iqr(self, five_seed_results):
        stats = aggregate_seed_results(five_seed_results)
        assert stats["k0_1_err_pct"]["p25"] == pytest.approx(2.0)
        assert stats["k0_1_err_pct"]["p75"] == pytest.approx(4.0)
        assert stats["k0_2_err_pct"]["p25"] == pytest.approx(4.0)
        assert stats["k0_2_err_pct"]["p75"] == pytest.approx(8.0)

    def test_max(self, five_seed_results):
        stats = aggregate_seed_results(five_seed_results)
        assert stats["k0_1_err_pct"]["max"] == pytest.approx(5.0)
        assert stats["k0_2_err_pct"]["max"] == pytest.approx(10.0)
        assert stats["alpha_1_err_pct"]["max"] == pytest.approx(0.5)
        assert stats["alpha_2_err_pct"]["max"] == pytest.approx(2.5)
