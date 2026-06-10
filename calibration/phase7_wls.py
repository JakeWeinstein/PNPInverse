"""Phase 7 WLS objective vs the v2 slide-15 target (Firedrake-free).

Scores a model H2O2-current curve against the precision-extracted
slide-15 target ``data/mangan_deck_p15_h2o2_current_v2.csv``.

Loss semantics (locked by critique session 41): this is a
RENDERED-CURVE REPRODUCTION metric, not a likelihood against
experimental replicates.  Per-bin sigma is digitization/measurement
scatter; chi2 values are comparable RELATIVELY (across parameter
corners / ablations), never quoted as absolute goodness-of-fit.

Conventions
-----------
* Model points come on the deck-V axis (the dual-pathway driver emits
  ``v_rhe_deck``); non-converged points must arrive as None/NaN and are
  DROPPED, never zero-filled.
* Model pc is interpolated onto the target-bin V's with PCHIP
  (monotone-preserving; no overshoot at the cliff).
* Bins flagged ``thresholded_zero`` (the j_ring<0.001 zeroed tail) are
  scored as one-sided onset constraints: penalty only if the model
  produces MORE cathodic current than sigma allows there.
* Route-aware diffusion-limit guards (critique R1#7): O2-based, never
  H+-based.

Soft sanity hinges (critique R1#12/R2#6 — sanity bounds, NOT
identifiability fixes): local-pH band [7, 10] at max |j|; far-cathodic
|cd| within [1.5, 5.99] mA/cm2.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

# Route-aware diffusion-limit constants at the deck-comparison film
# (L_eff = 15.4 um, 1600-rpm O2 Levich equivalent).  F2-style validity
# gates must use these — never an H+-based limit (the whole point of
# the water route is to exceed the H+ cap).
O2_4E_CEILING_MA_CM2 = 5.71   # total cd ceiling
O2_2E_CEILING_MA_CM2 = 2.86   # pc ceiling
I_LIM_TOLERANCE = 1.05

PH_HINGE_LO, PH_HINGE_HI = 7.0, 10.0
CD_HINGE_LO, CD_HINGE_HI = 1.5, 5.99
HINGE_WEIGHT = 10.0


@dataclass(frozen=True)
class Slide15Target:
    v: tuple
    j: tuple
    sigma: tuple
    thresholded_zero: tuple

    @property
    def n(self) -> int:
        return len(self.v)


def load_target(csv_path: str | Path) -> Slide15Target:
    """Load the v2 binned fit target (columns per the v2 extraction)."""
    rows = []
    with open(csv_path) as fh:
        reader = csv.reader(
            line for line in fh if line.strip() and not line.startswith("#")
        )
        header = next(reader)
        idx = {name: k for k, name in enumerate(header)}
        for rec in reader:
            rows.append((
                float(rec[idx["V_RHE_V"]]),
                float(rec[idx["j_h2o2_mA_cm2"]]),
                float(rec[idx["sigma_mA_cm2"]]),
                int(rec[idx["thresholded_zero"]]),
            ))
    if not rows:
        raise ValueError(f"no data rows in {csv_path}")
    rows.sort(key=lambda r: r[0])
    return Slide15Target(
        v=tuple(r[0] for r in rows),
        j=tuple(r[1] for r in rows),
        sigma=tuple(r[2] for r in rows),
        thresholded_zero=tuple(bool(r[3]) for r in rows),
    )


def _pchip(xs: Sequence[float], ys: Sequence[float]):
    from scipy.interpolate import PchipInterpolator
    return PchipInterpolator(xs, ys, extrapolate=False)


def _clean_series(v_deck, pc):
    """Drop non-converged (None/NaN) points; require >= 4 survivors."""
    pairs = [
        (float(v), float(p))
        for v, p in zip(v_deck, pc)
        if p is not None and v is not None and math.isfinite(float(p))
    ]
    if len(pairs) < 4:
        raise ValueError(
            f"only {len(pairs)} converged model points — not scoreable"
        )
    pairs.sort(key=lambda t: t[0])
    return [p[0] for p in pairs], [p[1] for p in pairs]


@dataclass
class WLSResult:
    chi2: float
    n_scored: int
    chi2_per_point: float
    residuals: list = field(default_factory=list)
    hinge_penalty: float = 0.0
    validity_failures: list = field(default_factory=list)
    total: float = 0.0


def score_curve(
    v_deck: Sequence[float],
    pc_mA_cm2: Sequence[Optional[float]],
    target: Slide15Target,
    *,
    cd_mA_cm2: Optional[Sequence[Optional[float]]] = None,
    surface_ph: Optional[Sequence[Optional[float]]] = None,
) -> WLSResult:
    """WLS chi2 of a model pc curve against the v2 target.

    Target bins outside the converged model V-range are dropped (no
    extrapolation).  Thresholded-zero bins use a one-sided penalty.
    Optional cd/surface_ph series add the sanity hinges and the
    route-aware I_lim validity checks.
    """
    vm, pm = _clean_series(v_deck, pc_mA_cm2)
    interp = _pchip(vm, pm)

    chi2 = 0.0
    n_scored = 0
    residuals = []
    for vt, jt, sig, thr in zip(
        target.v, target.j, target.sigma, target.thresholded_zero
    ):
        pj = interp(vt)
        if pj is None or not math.isfinite(float(pj)):
            continue  # outside converged model range
        pj = float(pj)
        if thr:
            # one-sided: penalize only model current MORE cathodic than
            # the zeroed tail allows
            r = min(0.0, pj - jt) / sig   # pj < jt(=0) -> negative -> penalty
        else:
            r = (pj - jt) / sig
        chi2 += r * r
        n_scored += 1
        residuals.append((vt, r))
    if n_scored == 0:
        raise ValueError("no target bins inside the converged model range")

    result = WLSResult(
        chi2=chi2, n_scored=n_scored, chi2_per_point=chi2 / n_scored,
        residuals=residuals,
    )

    # --- validity gates (route-aware I_lim; F2-style) ---
    max_abs_pc = max(abs(p) for p in pm)
    if max_abs_pc > I_LIM_TOLERANCE * O2_2E_CEILING_MA_CM2:
        result.validity_failures.append(
            f"pc {max_abs_pc:.3f} exceeds O2-2e ceiling"
        )
    cd_clean = None
    if cd_mA_cm2 is not None:
        try:
            _, cd_clean = _clean_series(v_deck, cd_mA_cm2)
        except ValueError:
            cd_clean = None
    if cd_clean is not None:
        max_abs_cd = max(abs(c) for c in cd_clean)
        if max_abs_cd > I_LIM_TOLERANCE * O2_4E_CEILING_MA_CM2:
            result.validity_failures.append(
                f"cd {max_abs_cd:.3f} exceeds O2-4e ceiling"
            )
        # far-cathodic |cd| hinge
        if max_abs_cd < CD_HINGE_LO:
            result.hinge_penalty += HINGE_WEIGHT * (CD_HINGE_LO - max_abs_cd) ** 2
        elif max_abs_cd > CD_HINGE_HI:
            result.hinge_penalty += HINGE_WEIGHT * (max_abs_cd - CD_HINGE_HI) ** 2

    # --- local-pH hinge at max |j| ---
    if surface_ph is not None and cd_clean is not None:
        ph_pairs = [
            (abs(float(c)), float(p))
            for c, p in zip(cd_mA_cm2, surface_ph)
            if c is not None and p is not None
            and math.isfinite(float(c)) and math.isfinite(float(p))
        ]
        if ph_pairs:
            ph_at_max_j = max(ph_pairs, key=lambda t: t[0])[1]
            if ph_at_max_j < PH_HINGE_LO:
                result.hinge_penalty += (
                    HINGE_WEIGHT * (PH_HINGE_LO - ph_at_max_j) ** 2
                )
            elif ph_at_max_j > PH_HINGE_HI:
                result.hinge_penalty += (
                    HINGE_WEIGHT * (ph_at_max_j - PH_HINGE_HI) ** 2
                )

    result.total = result.chi2_per_point + result.hinge_penalty
    return result


def score_iv_json(report: dict, target: Slide15Target) -> WLSResult:
    """Score a dual-pathway driver iv_curve.json report dict."""
    return score_curve(
        report["v_rhe_deck"],
        report["pc_mA_cm2"],
        target,
        cd_mA_cm2=report.get("cd_mA_cm2"),
        surface_ph=report.get("surface_pH"),
    )
