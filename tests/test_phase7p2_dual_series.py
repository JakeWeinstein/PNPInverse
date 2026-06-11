"""Fast tests for the Phase 7.2 dual-series objective (Firedrake-free).

Covers the session-43 conventions: frozen-mask immutability, model-side
N mapping, raise-don't-penalize convergence discipline, bin-center
alignment, sigma handling.
"""
import math
from pathlib import Path

import pytest

from calibration.phase7_wls import (
    A_DISK_CM2,
    A_RING_CM2,
    DualSeriesTarget,
    SolveFailureError,
    load_dual_target,
    score_dual_series,
)

BINNED = Path("data/k2so4_ph6p39_rrde_binned.csv")


def _toy_target(n=5):
    v = tuple(0.1 + 0.1 * k for k in range(n))
    return DualSeriesTarget(
        v=v,
        j_disk=tuple(-1.0 - 0.2 * k for k in range(n)),
        sigma_disk=(0.05,) * n,
        j_ring=tuple(0.10 + 0.01 * k for k in range(n)),
        sigma_ring=(0.005,) * n,
    )


class TestLoadDualTarget:
    def test_loads_real_binned_target(self):
        t = load_dual_target(BINNED)
        assert t.n == 30
        assert all(s > 0 for s in t.sigma_disk)
        assert all(s > 0 for s in t.sigma_ring)
        assert all(j <= 0 for j in t.j_disk)      # cathodic-negative
        assert all(j >= -1e-6 for j in t.j_ring)  # anodic-positive
        assert t.v == tuple(sorted(t.v))

    def test_mask_freezes_bins(self):
        full = load_dual_target(BINNED)
        mask = [True] * full.n
        mask[3] = False
        t = load_dual_target(BINNED, mask=mask)
        assert t.n == full.n - 1
        assert full.v[3] not in t.v

    def test_mask_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="mask length"):
            load_dual_target(BINNED, mask=[True] * 3)


class TestScoreDualSeries:
    def test_exact_match_scores_zero(self):
        t = _toy_target()
        pc = tuple(-j * A_RING_CM2 / (0.224 * A_DISK_CM2)
                   for j in t.j_ring)
        r = score_dual_series(t.v, t.j_disk, pc, t, interp=False)
        assert r.j_opt == pytest.approx(0.0, abs=1e-18)
        assert r.chi2_raw == pytest.approx(0.0, abs=1e-18)

    def test_model_side_n_mapping(self):
        """Changing N changes the objective with the SAME data."""
        t = _toy_target()
        pc = tuple(-j * A_RING_CM2 / (0.224 * A_DISK_CM2)
                   for j in t.j_ring)
        r1 = score_dual_series(t.v, t.j_disk, pc, t, interp=False,
                               n_coll=0.224)
        r2 = score_dual_series(t.v, t.j_disk, pc, t, interp=False,
                               n_coll=0.20)
        assert r1.j_opt == pytest.approx(0.0, abs=1e-18)
        assert r2.score_ring > 0.0
        assert r2.score_disk == pytest.approx(0.0, abs=1e-18)

    def test_nan_raises_not_penalizes(self):
        t = _toy_target()
        pc = [-0.1] * t.n
        cd = list(t.j_disk)
        cd[2] = math.nan
        with pytest.raises(SolveFailureError):
            score_dual_series(t.v, cd, pc, t, interp=False)
        with pytest.raises(SolveFailureError):
            score_dual_series(t.v, cd, pc, t, interp=True)

    def test_bin_center_alignment_enforced(self):
        t = _toy_target()
        v_off = tuple(v + 0.01 for v in t.v)
        with pytest.raises(ValueError, match="bin centers"):
            score_dual_series(v_off, t.j_disk, [-0.1] * t.n, t,
                              interp=False)

    def test_interp_requires_coverage(self):
        t = _toy_target()
        v_short = t.v[1:]
        with pytest.raises(SolveFailureError, match="cover"):
            score_dual_series(v_short, t.j_disk[1:], [-0.1] * (t.n - 1),
                              t, interp=True)

    def test_ring_weight_scale(self):
        t = _toy_target()
        pc = [-0.2] * t.n
        cd = list(t.j_disk)
        r1 = score_dual_series(t.v, cd, pc, t, interp=False)
        r2 = score_dual_series(t.v, cd, pc, t, interp=False,
                               w_ring_scale=2.0)
        assert r2.j_opt == pytest.approx(
            r1.score_disk + 2.0 * r1.score_ring)
        # raw chi2 is weight-independent (fixed-vector statistic)
        assert r2.chi2_raw == pytest.approx(r1.chi2_raw)

    def test_raw_chi2_is_unnormalized_sum(self):
        t = _toy_target()
        pc = [-0.2] * t.n
        r = score_dual_series(t.v, t.j_disk, pc, t, interp=False)
        assert r.chi2_raw == pytest.approx(
            t.n * (r.score_disk + r.score_ring))

    def test_ceiling_validity_flag(self):
        t = _toy_target()
        cd = [-10.0] * t.n
        r = score_dual_series(t.v, cd, [-0.1] * t.n, t, interp=False)
        assert any("O2-4e" in f for f in r.validity_failures)
