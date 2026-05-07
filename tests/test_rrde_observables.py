"""Unit tests for the M1 RRDE post-processing module.

Pure-Python tests on ``Forward.bv_solver.rrde_observables`` and the
``ExperimentMetadata`` factory at ``scripts._bv_common``.  No Firedrake
dependency: these run as part of ``pytest -m "not slow"``.

Coverage:
- Sign conventions (cathodic disk → positive ring; abs() in selectivity).
- Canonical 2e/4e pathways: 100%/0% selectivity and ``n_e=2``/``n_e=4``.
- Edge cases: zero current, NaN inputs, invalid ``N_collection``.
- Surface-pH proxy at known pH 4 / pH 7.
- Metadata schema and honest placeholder defaults.
"""

from __future__ import annotations

import dataclasses
import math

import pytest


# ---------------------------------------------------------------------------
# Surface-pH proxy
# ---------------------------------------------------------------------------

def test_surface_pH_proxy_at_pH4():
    """C_scale = 0.5 mol/m^3, c_H_nondim = 0.2 → c_H = 1e-4 mol/L → pH 4.0.

    These are the production scaling constants (``C_SCALE = 0.5``,
    ``C_HP_HAT = 0.2``) so the bulk H+ surface mean rounds to pH 4.
    """
    from Forward.bv_solver.rrde_observables import compute_surface_pH_proxy

    pH = compute_surface_pH_proxy(0.2, 0.5)
    assert pH == pytest.approx(4.0, abs=1e-12)


def test_surface_pH_proxy_at_pH7():
    """C_scale = 1.0 mol/m^3, c_H_nondim = 1e-4 → c_H = 1e-7 mol/L → pH 7.0."""
    from Forward.bv_solver.rrde_observables import compute_surface_pH_proxy

    pH = compute_surface_pH_proxy(1e-4, 1.0)
    assert pH == pytest.approx(7.0, abs=1e-12)


def test_surface_pH_proxy_zero_or_negative_returns_nan():
    """Numerical noise pushing c_H to <= 0 should produce NaN, not raise."""
    from Forward.bv_solver.rrde_observables import compute_surface_pH_proxy

    assert math.isnan(compute_surface_pH_proxy(0.0, 0.5))
    assert math.isnan(compute_surface_pH_proxy(-1e-12, 0.5))


def test_surface_pH_proxy_nan_input_returns_nan():
    from Forward.bv_solver.rrde_observables import compute_surface_pH_proxy

    assert math.isnan(compute_surface_pH_proxy(float("nan"), 0.5))


def test_surface_pH_proxy_invalid_C_scale_raises():
    from Forward.bv_solver.rrde_observables import compute_surface_pH_proxy

    with pytest.raises(ValueError):
        compute_surface_pH_proxy(0.2, 0.0)
    with pytest.raises(ValueError):
        compute_surface_pH_proxy(0.2, -0.5)


# ---------------------------------------------------------------------------
# Ring current
# ---------------------------------------------------------------------------

def test_ring_current_sign_flip():
    """Cathodic peroxide current (negative) → positive ring current."""
    from Forward.bv_solver.rrde_observables import compute_ring_current

    j_ring = compute_ring_current(-1.0, 0.224)
    assert j_ring == pytest.approx(0.224, abs=1e-12)
    assert j_ring > 0


def test_ring_current_takes_magnitude():
    """Anodic peroxide (positive) also produces positive ring current."""
    from Forward.bv_solver.rrde_observables import compute_ring_current

    assert compute_ring_current(+1.0, 0.224) == pytest.approx(0.224, abs=1e-12)


def test_ring_current_zero_disk():
    from Forward.bv_solver.rrde_observables import compute_ring_current

    assert compute_ring_current(0.0, 0.224) == 0.0


def test_ring_current_invalid_N_raises():
    from Forward.bv_solver.rrde_observables import compute_ring_current

    with pytest.raises(ValueError):
        compute_ring_current(-1.0, 0.0)
    with pytest.raises(ValueError):
        compute_ring_current(-1.0, -0.1)
    with pytest.raises(ValueError):
        compute_ring_current(-1.0, 1.1)
    with pytest.raises(ValueError):
        compute_ring_current(-1.0, float("nan"))


# ---------------------------------------------------------------------------
# Selectivity (S_H2O2_percent)
# ---------------------------------------------------------------------------

def test_perfect_2e_pathway_gives_100pct_selectivity():
    """When all disk current is peroxide-producing, S = 100%."""
    from Forward.bv_solver.rrde_observables import (
        compute_ring_current,
        compute_selectivity_percent,
    )

    j_disk = -1.0
    j_h2o2_disk = -1.0
    N = 0.224
    j_ring = compute_ring_current(j_h2o2_disk, N)
    s = compute_selectivity_percent(j_disk, j_ring, N)
    assert s == pytest.approx(100.0, abs=1e-12)


def test_perfect_4e_pathway_gives_0pct_selectivity():
    """When net peroxide is zero (full reduction to water), S = 0%."""
    from Forward.bv_solver.rrde_observables import (
        compute_ring_current,
        compute_selectivity_percent,
    )

    j_disk = -1.0
    j_h2o2_disk = 0.0
    N = 0.224
    j_ring = compute_ring_current(j_h2o2_disk, N)
    s = compute_selectivity_percent(j_disk, j_ring, N)
    assert s == pytest.approx(0.0, abs=1e-12)


def test_zero_disk_current_handled():
    """When |I_disk| = 0 and I_ring = 0, S and n_e should be NaN."""
    from Forward.bv_solver.rrde_observables import (
        compute_n_e_rrde,
        compute_ring_current,
        compute_selectivity_percent,
    )

    j_ring = compute_ring_current(0.0, 0.224)
    assert math.isnan(compute_selectivity_percent(0.0, j_ring, 0.224))
    assert math.isnan(compute_n_e_rrde(0.0, j_ring, 0.224))


def test_selectivity_invalid_N_raises():
    from Forward.bv_solver.rrde_observables import compute_selectivity_percent

    with pytest.raises(ValueError):
        compute_selectivity_percent(-1.0, 0.5, 0.0)
    with pytest.raises(ValueError):
        compute_selectivity_percent(-1.0, 0.5, 1.5)


# ---------------------------------------------------------------------------
# Apparent electron count (n_e)
# ---------------------------------------------------------------------------

def test_perfect_2e_pathway_gives_n_e_2():
    from Forward.bv_solver.rrde_observables import (
        compute_n_e_rrde,
        compute_ring_current,
    )

    j_disk = -1.0
    j_h2o2_disk = -1.0
    N = 0.224
    j_ring = compute_ring_current(j_h2o2_disk, N)
    n_e = compute_n_e_rrde(j_disk, j_ring, N)
    assert n_e == pytest.approx(2.0, abs=1e-12)


def test_perfect_4e_pathway_gives_n_e_4():
    from Forward.bv_solver.rrde_observables import (
        compute_n_e_rrde,
        compute_ring_current,
    )

    j_disk = -1.0
    j_h2o2_disk = 0.0
    N = 0.224
    j_ring = compute_ring_current(j_h2o2_disk, N)
    n_e = compute_n_e_rrde(j_disk, j_ring, N)
    assert n_e == pytest.approx(4.0, abs=1e-12)


def test_n_e_in_intermediate_range():
    """50/50 mixed pathway should give n_e between 2 and 4.

    With ``j_disk=-1.0`` and ``j_h2o2_disk=-0.5``:
    ``|disk|=1.0``, ``ring/N=0.5``, ``denom=1.5``, ``n_e=4*1/1.5``.
    """
    from Forward.bv_solver.rrde_observables import (
        compute_n_e_rrde,
        compute_ring_current,
    )

    j_disk = -1.0
    j_h2o2_disk = -0.5
    N = 0.224
    j_ring = compute_ring_current(j_h2o2_disk, N)
    n_e = compute_n_e_rrde(j_disk, j_ring, N)
    assert 2.0 < n_e < 4.0
    assert n_e == pytest.approx(4.0 / 1.5, abs=1e-12)


# ---------------------------------------------------------------------------
# Assembly bundle
# ---------------------------------------------------------------------------

def test_assemble_rrde_observables_returns_frozen_dataclass():
    from Forward.bv_solver.rrde_observables import (
        RRDEObservables,
        assemble_rrde_observables,
    )

    obs = assemble_rrde_observables(
        j_disk=-1.0,
        j_h2o2_disk=-1.0,
        c_H_surface_nondim=0.2,
        C_scale_mol_m3=0.5,
        N_collection=0.224,
    )
    assert isinstance(obs, RRDEObservables)
    with pytest.raises(dataclasses.FrozenInstanceError):
        obs.j_disk_model = 0.0  # type: ignore[misc]


def test_assemble_perfect_2e_pathway():
    """Sign-convention regression test for the production pH-4 scaling."""
    from Forward.bv_solver.rrde_observables import assemble_rrde_observables

    obs = assemble_rrde_observables(
        j_disk=-1.0,
        j_h2o2_disk=-1.0,
        c_H_surface_nondim=0.2,
        C_scale_mol_m3=0.5,
        N_collection=0.224,
    )
    assert obs.j_disk_model == pytest.approx(-1.0)
    assert obs.j_h2o2_disk_model == pytest.approx(-1.0)
    assert obs.j_ring_model == pytest.approx(0.224)
    assert obs.surface_pH_proxy == pytest.approx(4.0)
    assert obs.S_H2O2_percent == pytest.approx(100.0)
    assert obs.n_e_rrde == pytest.approx(2.0)


def test_assemble_sign_convention_regression():
    """Cathodic disk inputs produce positive ring and physical S/n_e ranges."""
    from Forward.bv_solver.rrde_observables import assemble_rrde_observables

    obs = assemble_rrde_observables(
        j_disk=-1.0,
        j_h2o2_disk=-0.7,
        c_H_surface_nondim=0.2,
        C_scale_mol_m3=0.5,
        N_collection=0.224,
    )
    assert obs.j_disk_model < 0
    assert obs.j_h2o2_disk_model < 0
    assert obs.j_ring_model > 0
    assert 0.0 <= obs.S_H2O2_percent <= 100.0
    assert 2.0 <= obs.n_e_rrde <= 4.0


def test_assemble_propagates_nan_on_failed_solve():
    """NaN cd/pc inputs (failed solve) propagate to NaN observables."""
    from Forward.bv_solver.rrde_observables import assemble_rrde_observables

    obs = assemble_rrde_observables(
        j_disk=float("nan"),
        j_h2o2_disk=float("nan"),
        c_H_surface_nondim=0.2,
        C_scale_mol_m3=0.5,
        N_collection=0.224,
    )
    assert math.isnan(obs.j_disk_model)
    assert math.isnan(obs.j_h2o2_disk_model)
    assert math.isnan(obs.j_ring_model)
    # surface_pH_proxy is still computable from c_H alone:
    assert obs.surface_pH_proxy == pytest.approx(4.0)
    assert math.isnan(obs.S_H2O2_percent)
    assert math.isnan(obs.n_e_rrde)


# ---------------------------------------------------------------------------
# ExperimentMetadata schema + honest defaults
# ---------------------------------------------------------------------------

def test_metadata_schema_required_fields():
    """ExperimentMetadata exposes every M1 plan field."""
    from scripts._bv_common import ExperimentMetadata, make_experiment_metadata

    md = make_experiment_metadata()
    expected = {
        "catalyst",
        "geometry",
        "pH_bulk",
        "cation",
        "anion_model",
        "rotation_rate_rpm",
        "L_eff_m",
        "N_collection",
        "electrolyte_model",
        "comparison_status",
        "source_authority",
        "target_curve",
        "acceptance_tier",
    }
    actual = {f.name for f in dataclasses.fields(md)}
    assert expected == actual
    assert isinstance(md, ExperimentMetadata)


def test_metadata_default_placeholders_are_honest():
    """Defaults flag a run as not-yet-deck-comparable until M0 lands."""
    from scripts._bv_common import make_experiment_metadata

    md = make_experiment_metadata()
    assert md.source_authority == "memory"
    assert md.comparison_status == "internal_baseline_only"
    assert md.electrolyte_model == "pH_countercharge_surrogate"
    assert md.anion_model == "ClO4_protonic_surrogate"
    assert md.catalyst == "generic_carbon"
    assert md.geometry == "stagnant_film"
    assert md.pH_bulk == 4.0
    assert md.cation is None
    assert md.rotation_rate_rpm is None
    assert md.L_eff_m is None
    assert md.N_collection is None
    assert md.target_curve is None
    assert md.acceptance_tier == "trend"


def test_metadata_is_frozen_dataclass():
    from scripts._bv_common import make_experiment_metadata

    md = make_experiment_metadata()
    with pytest.raises(dataclasses.FrozenInstanceError):
        md.catalyst = "other"  # type: ignore[misc]


def test_metadata_invalid_N_raises():
    from scripts._bv_common import make_experiment_metadata

    with pytest.raises(ValueError):
        make_experiment_metadata(N_collection=0.0)
    with pytest.raises(ValueError):
        make_experiment_metadata(N_collection=1.5)
    with pytest.raises(ValueError):
        make_experiment_metadata(N_collection=-0.1)


def test_metadata_invalid_pH_raises():
    from scripts._bv_common import make_experiment_metadata

    with pytest.raises(ValueError):
        make_experiment_metadata(pH_bulk=-1.0)


def test_metadata_invalid_L_eff_raises():
    from scripts._bv_common import make_experiment_metadata

    with pytest.raises(ValueError):
        make_experiment_metadata(L_eff_m=0.0)
    with pytest.raises(ValueError):
        make_experiment_metadata(L_eff_m=-1e-5)


def test_metadata_invalid_rotation_rate_raises():
    from scripts._bv_common import make_experiment_metadata

    with pytest.raises(ValueError):
        make_experiment_metadata(rotation_rate_rpm=-100.0)


def test_metadata_serialises_via_asdict():
    """JSON round-trip via dataclasses.asdict — used by the study script."""
    from scripts._bv_common import make_experiment_metadata

    md = make_experiment_metadata(N_collection=0.224, target_curve="mangan_fig3a")
    d = dataclasses.asdict(md)
    assert d["N_collection"] == 0.224
    assert d["target_curve"] == "mangan_fig3a"
    assert d["source_authority"] == "memory"
    # All values must be JSON-serialisable primitives or None.
    import json
    json.dumps(d)


def test_metadata_promote_status_works():
    """M0 completion can override defaults to mark a run deck-comparable."""
    from scripts._bv_common import make_experiment_metadata

    md = make_experiment_metadata(
        source_authority="Mangan2025_deck",
        comparison_status="deck_proxy",
        target_curve="fig3a_pH4_KClO4",
        N_collection=0.224,
        acceptance_tier="semi_quant",
    )
    assert md.source_authority == "Mangan2025_deck"
    assert md.comparison_status == "deck_proxy"
    assert md.target_curve == "fig3a_pH4_KClO4"
    assert md.N_collection == 0.224
    assert md.acceptance_tier == "semi_quant"
