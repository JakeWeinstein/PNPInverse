"""Phase 7.3 P0.2 — unit tests for the disk-onset extractor logic.

Locks the non-trivial pieces: plateau-connected onset (isolated high-V blips
past a sub-threshold gap are ignored), background estimation, and the small
linear-slope helper.  Pure numpy; no Firedrake.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import scripts.studies.phase7p3_p0_2_onset_extractor as ox


def test_onset_clean_monotone_curve():
    """A clean cathodic wave: −y_corr rises from 0 at high V to large at Vmin.
    Onset at thr=0.1 is the interpolated crossing."""
    v = np.linspace(0.0, 0.8, 81)            # V increasing
    c = np.clip((0.5 - v) * 4.0, 0.0, None)  # cathodic-positive, 0 above V=0.5
    y_corr = -c
    onset = ox._onset_at(v, y_corr, 0.1)
    # crossing where c = 0.1 -> 0.5 - v = 0.025 -> v = 0.475
    assert onset == pytest.approx(0.475, abs=2e-3)


def test_onset_ignores_isolated_high_v_blip():
    """An isolated above-threshold blip at high V, separated from the main
    wave by a sub-threshold gap, must NOT be reported as the onset (the
    plateau-connected wave turn-on is)."""
    v = np.linspace(0.0, 0.8, 81)
    c = np.clip((0.5 - v) * 4.0, 0.0, None)
    c[75] = 0.3                              # spurious blip near V≈0.75
    onset = ox._onset_at(v, -c, 0.1)
    assert onset == pytest.approx(0.475, abs=2e-3)  # not ~0.75


def test_onset_none_when_plateau_below_threshold():
    """If even the cathodic plateau (Vmin) is below thr, the wave is
    unreached -> None."""
    v = np.linspace(0.0, 0.8, 81)
    c = np.full_like(v, 0.02)                # everywhere below thr=0.1
    assert ox._onset_at(v, -c, 0.1) is None


def test_baseline_from_anodic_window():
    v = np.linspace(0.0, 0.8, 81)
    # flat -0.05 over the whole most-anodic window (v >= v.max()-BG_WINDOW_V).
    y = np.where(v >= v.max() - ox.BG_WINDOW_V - 1e-9, -0.05, -2.0)
    base, std, n = ox._baseline(v, y)
    assert base == pytest.approx(-0.05, abs=1e-9)
    assert std == pytest.approx(0.0, abs=1e-9)
    assert n >= ox.BG_MIN_PTS


def test_ols_slope_recovers_known_line():
    ph = [2.0, 4.0, 6.0]
    on = [0.10 + 0.03 * p for p in ph]       # +30 mV/pH exactly
    fit = ox._ols_slope(ph, on)
    assert fit["slope_v_per_ph"] == pytest.approx(0.03, abs=1e-9)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-9)
