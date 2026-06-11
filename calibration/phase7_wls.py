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


# ===========================================================================
# Phase 7.2 — dual-series (disk + raw ring) objective, K2SO4 pH 6.39
# ===========================================================================
#
# Session-43 conventions:
# * Disk series: cd model vs binned j_disk (mA/cm2_disk,
#   cathodic-negative, LSV background subtracted in Stage 0).
# * Ring series: RAW baseline-corrected j_ring (mA/cm2_ring,
#   anodic-positive); the collection model lives on the MODEL side:
#       j_ring_model = -pc_model * N * A_d / A_r
#   so N variants are pure model-side refits (R2#4).
# * sigma is a conservative single-observation predictive scale;
#   "standardized residual score" per series — NOT reduced chi2.
# * Convergence discipline: a non-finite model value at any masked-in
#   bin raises SolveFailureError — the optimizer never sees penalty
#   values (R2#1).  Raw chi2 on the FIXED masked vector is logged at
#   every call.

A_DISK_CM2 = 0.19635
A_RING_CM2 = 0.109956
N_COLL_DEFAULT = 0.224


class SolveFailureError(RuntimeError):
    """A masked-in objective voltage has no converged model value."""


@dataclass(frozen=True)
class DualSeriesTarget:
    v: tuple                 # V_RHE (physical iR axis), bin centers
    j_disk: tuple            # mA/cm2_disk, cathodic-negative
    sigma_disk: tuple
    j_ring: tuple            # mA/cm2_ring, anodic-positive
    sigma_ring: tuple

    @property
    def n(self) -> int:
        return len(self.v)


def load_dual_target(csv_path: str | Path,
                     mask: Optional[Sequence[bool]] = None
                     ) -> DualSeriesTarget:
    """Load the Stage-0 binned target; optional FROZEN bin mask."""
    rows = []
    with open(csv_path) as fh:
        reader = csv.reader(
            line for line in fh
            if line.strip() and not line.startswith("#"))
        header = next(reader)
        idx = {name: k for k, name in enumerate(header)}
        for rec in reader:
            rows.append(tuple(float(rec[idx[c]]) for c in (
                "v_phys", "j_disk", "sigma_j_disk",
                "j_ring", "sigma_j_ring")))
    if not rows:
        raise ValueError(f"no data rows in {csv_path}")
    rows.sort(key=lambda r: r[0])
    if mask is not None:
        if len(mask) != len(rows):
            raise ValueError("mask length != bin count")
        rows = [r for r, m in zip(rows, mask) if m]
    if any(r[2] <= 0 or r[4] <= 0 for r in rows):
        raise ValueError("non-positive sigma in target")
    return DualSeriesTarget(
        v=tuple(r[0] for r in rows),
        j_disk=tuple(r[1] for r in rows),
        sigma_disk=tuple(r[2] for r in rows),
        j_ring=tuple(r[3] for r in rows),
        sigma_ring=tuple(r[4] for r in rows),
    )


@dataclass
class DualSeriesResult:
    j_opt: float                 # optimization objective (normalized)
    chi2_raw: float              # raw chi2 on the fixed masked vector
    n_disk: int
    n_ring: int
    score_disk: float            # per-series standardized residual score
    score_ring: float            # (mean squared standardized residual)
    residuals_disk: list = field(default_factory=list)
    residuals_ring: list = field(default_factory=list)
    validity_failures: list = field(default_factory=list)


def score_dual_series(
    v_rhe: Sequence[float],
    cd_mA_cm2: Sequence[Optional[float]],
    pc_mA_cm2: Sequence[Optional[float]],
    target: DualSeriesTarget,
    *,
    n_coll: float = N_COLL_DEFAULT,
    w_ring_scale: float = 1.0,
    interp: bool = True,
    o2_4e_ceiling: float = O2_4E_CEILING_MA_CM2,
) -> DualSeriesResult:
    """Dual-series objective.

    interp=True: model on its own (iteration) grid, PCHIP onto bin
    centers — model must COVER the target range.  interp=False: model
    solved AT bin centers (final scoring); V alignment asserted to
    1e-9.  Any non-finite model value inside the mask raises
    SolveFailureError (never a penalty value).
    """
    def _series(vals):
        out = []
        for v in vals:
            out.append(float(v) if v is not None
                       and math.isfinite(float(v)) else math.nan)
        return out

    cd = _series(cd_mA_cm2)
    pc = _series(pc_mA_cm2)
    vm = [float(v) for v in v_rhe]

    if interp:
        ok = [k for k in range(len(vm))
              if math.isfinite(cd[k]) and math.isfinite(pc[k])]
        if len(ok) != len(vm):
            raise SolveFailureError(
                f"{len(vm) - len(ok)} non-converged model points")
        order = sorted(ok, key=lambda k: vm[k])
        vs = [vm[k] for k in order]
        if vs[0] > target.v[0] + 1e-9 or vs[-1] < target.v[-1] - 1e-9:
            raise SolveFailureError(
                "model grid does not cover the target window")
        cd_t = _pchip(vs, [cd[k] for k in order])(target.v)
        pc_t = _pchip(vs, [pc[k] for k in order])(target.v)
        cd_t, pc_t = list(map(float, cd_t)), list(map(float, pc_t))
    else:
        if len(vm) != target.n or any(
                abs(a - b) > 1e-9 for a, b in zip(sorted(vm), target.v)):
            raise ValueError("interp=False requires model solved AT "
                             "the target bin centers")
        order = sorted(range(len(vm)), key=lambda k: vm[k])
        cd_t = [cd[k] for k in order]
        pc_t = [pc[k] for k in order]
        bad = [k for k in range(target.n)
               if not (math.isfinite(cd_t[k]) and math.isfinite(pc_t[k]))]
        if bad:
            raise SolveFailureError(
                f"non-converged at bin centers {bad}")

    res = DualSeriesResult(0.0, 0.0, target.n, target.n, 0.0, 0.0)
    chi_d = chi_r = 0.0
    for k in range(target.n):
        rd = (cd_t[k] - target.j_disk[k]) / target.sigma_disk[k]
        jr_model = -pc_t[k] * n_coll * A_DISK_CM2 / A_RING_CM2
        rr = (jr_model - target.j_ring[k]) / target.sigma_ring[k]
        chi_d += rd * rd
        chi_r += rr * rr
        res.residuals_disk.append((target.v[k], rd))
        res.residuals_ring.append((target.v[k], rr))
    res.score_disk = chi_d / target.n
    res.score_ring = chi_r / target.n
    res.chi2_raw = chi_d + chi_r
    res.j_opt = res.score_disk + w_ring_scale * res.score_ring

    max_abs_cd = max(abs(c) for c in cd_t)
    if max_abs_cd > I_LIM_TOLERANCE * o2_4e_ceiling:
        res.validity_failures.append(
            f"cd {max_abs_cd:.3f} exceeds O2-4e ceiling "
            f"{o2_4e_ceiling}")
    max_abs_pc = max(abs(p) for p in pc_t)
    if max_abs_pc > I_LIM_TOLERANCE * o2_4e_ceiling / 2.0:
        res.validity_failures.append(
            f"pc {max_abs_pc:.3f} exceeds O2-2e ceiling "
            f"{o2_4e_ceiling / 2.0}")
    return res
