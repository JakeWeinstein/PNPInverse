"""AUDT-04 diagnostic metadata schema validation tests.

Validates that every diagnostic tool produces a JSON metadata sidecar with
required fields documenting the tool's justification per AUDT-04.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Metadata schema constants (shared across all diagnostic tests)
# ---------------------------------------------------------------------------

REQUIRED_METADATA_KEYS = {
    "tool_name",
    "phase",
    "requirement",
    "justification_type",
    "reference",
    "rationale",
    "parameters",
    "generated",
}

VALID_JUSTIFICATION_TYPES = {"literature", "empirical", "simplest"}


def validate_metadata(metadata: dict) -> bool:
    """Check that *metadata* contains all required AUDT-04 keys.

    Returns True if valid, False otherwise.
    """
    if not isinstance(metadata, dict):
        return False
    if not REQUIRED_METADATA_KEYS.issubset(metadata.keys()):
        return False
    if metadata.get("justification_type") not in VALID_JUSTIFICATION_TYPES:
        return False
    return True


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMetadataValidation:
    """Verify validate_metadata helper logic."""

    def test_valid_metadata(self):
        meta = {
            "tool_name": "Multi-Seed Pipeline Robustness Assessment",
            "phase": "07-baseline-diagnostics",
            "requirement": "DIAG-01",
            "justification_type": "empirical",
            "reference": "Standard practice in inverse problems",
            "rationale": "Quantifies pipeline sensitivity to noise realization",
            "parameters": {"n_seeds": 20, "noise_percent": 2.0},
            "generated": "2026-03-10T12:00:00Z",
        }
        assert validate_metadata(meta) is True

    def test_missing_key(self):
        meta = {
            "tool_name": "Test Tool",
            "phase": "07",
            # missing requirement, justification_type, reference, rationale, parameters, generated
        }
        assert validate_metadata(meta) is False

    def test_invalid_justification_type(self):
        meta = {
            "tool_name": "Test Tool",
            "phase": "07",
            "requirement": "DIAG-01",
            "justification_type": "invalid_type",
            "reference": "ref",
            "rationale": "reason",
            "parameters": {},
            "generated": "2026-03-10T12:00:00Z",
        }
        assert validate_metadata(meta) is False


class TestMetadataFromScript:
    """Verify that the multi-seed script's write_metadata produces valid output."""

    def test_write_metadata_produces_valid_json(self, tmp_path):
        from scripts.studies.run_multi_seed_v13 import write_metadata

        write_metadata(str(tmp_path))

        import json

        meta_path = tmp_path / "metadata.json"
        assert meta_path.exists(), "metadata.json was not created"
        meta = json.loads(meta_path.read_text())
        assert validate_metadata(meta) is True
