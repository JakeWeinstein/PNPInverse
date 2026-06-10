"""Phase 7 WLS fit-harness tests (Firedrake-free, fast tier)."""
from __future__ import annotations

import math
import os
import sys

import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from calibration.phase7_wls import (
    O2_2E_CEILING_MA_CM2,
    O2_4E_CEILING_MA_CM2,
    Slide15Target,
    WLSResult,
    load_target,
    score_curve,
)

V2_CSV = os.path.join(_ROOT, "data", "mangan_deck_p15_h2o2_current_v2.csv")


def _toy_target(**overrides):
    base = dict(
        v=(-0.3, -0.1, 0.1, 0.3, 0.5),
        j=(-0.18, -0.30, -0.37, -0.20, 0.0),
        sigma=(0.01, 0.01, 0.02, 0.05, 0.005),
        thresholded_zero=(False, False, False, False, True),
    )
    base.update(overrides)
    return Slide15Target(**base)


class TestLoadTarget:
    def test_loads_v2_csv(self):
        t = load_target(V2_CSV)
        assert t.n >= 20
        assert min(t.v) == pytest.approx(-0.312, abs=0.02)
        assert max(t.v) == pytest.approx(0.68, abs=0.03)
        assert min(t.j) == pytest.approx(-0.357, abs=0.02)
        assert all(s > 0 for s in t.sigma)
        # the zeroed tail must be flagged
        assert any(t.thresholded_zero)
        for flag, j in zip(t.thresholded_zero, t.j):
            if flag:
                assert j == 0.0


class TestScoreCurve:
    def test_perfect_model_scores_zero(self):
        t = _toy_target()
        res = score_curve(list(t.v), list(t.j), t)
        assert res.chi2 == pytest.approx(0.0, abs=1e-20)
        assert res.n_scored == t.n

    def test_off_by_one_sigma_scores_one_per_point(self):
        t = _toy_target(thresholded_zero=(False,) * 5)
        model = [j + s for j, s in zip(t.j, t.sigma)]
        res = score_curve(list(t.v), model, t)
        assert res.chi2_per_point == pytest.approx(1.0, rel=1e-6)

    def test_nonconverged_points_dropped_not_zero_filled(self):
        t = _toy_target()
        v = list(t.v)
        model = list(t.j)
        model[2] = None        # non-converged point
        res = score_curve(v, model, t)
        # interpolation through remaining points — finite chi2, and the
        # dropped point must NOT have been treated as 0.0 (which would
        # give a huge residual at v=0.1 where target is -0.37)
        r_at_peak = dict((round(vv, 3), rr) for vv, rr in res.residuals)[0.1]
        assert abs(r_at_peak) < 10.0

    def test_too_few_points_raises(self):
        t = _toy_target()
        with pytest.raises(ValueError, match="not scoreable"):
            score_curve([-0.3, 0.1], [-0.2, None], t)

    def test_target_bins_outside_model_range_dropped(self):
        t = _toy_target()
        # model only spans [-0.1, 0.5]
        res = score_curve(list(t.v)[1:], list(t.j)[1:], t)
        assert res.n_scored == t.n - 1

    def test_thresholded_zero_one_sided(self):
        t = _toy_target()
        # model exactly zero in the tail: no penalty
        res0 = score_curve(list(t.v), list(t.j), t)
        # model slightly ANODIC of zero in tail (positive): no penalty
        model_pos = list(t.j)[:-1] + [+0.003]
        res_pos = score_curve(list(t.v), model_pos, t)
        # model cathodic in the zeroed tail: penalized
        model_neg = list(t.j)[:-1] + [-0.05]
        res_neg = score_curve(list(t.v), model_neg, t)
        tail_r = {round(v, 3): r for v, r in res_neg.residuals}[0.5]
        assert res0.chi2 == pytest.approx(0.0, abs=1e-20)
        assert res_pos.chi2 == pytest.approx(0.0, abs=1e-12)
        assert abs(tail_r) > 1.0

    def test_route_aware_validity_gates(self):
        t = _toy_target()
        # pc exceeding the O2-2e ceiling flags validity
        model = [-(O2_2E_CEILING_MA_CM2 * 1.5)] * 4 + [0.0]
        res = score_curve(list(t.v), model, t)
        assert any("O2-2e" in f for f in res.validity_failures)
        # cd exceeding the O2-4e ceiling flags validity
        res2 = score_curve(
            list(t.v), list(t.j), t,
            cd_mA_cm2=[-(O2_4E_CEILING_MA_CM2 * 1.2)] * 5,
        )
        assert any("O2-4e" in f for f in res2.validity_failures)

    def test_hinges_are_soft_not_fatal(self):
        t = _toy_target()
        # tiny cd (below the 1.5 hinge) adds penalty but still scores
        res = score_curve(
            list(t.v), list(t.j), t,
            cd_mA_cm2=[-0.5] * 5,
            surface_ph=[5.0] * 5,   # below the [7,10] band at max |j|
        )
        assert res.chi2 == pytest.approx(0.0, abs=1e-20)
        assert res.hinge_penalty > 0.0
        assert res.total == pytest.approx(res.hinge_penalty, rel=1e-9)
