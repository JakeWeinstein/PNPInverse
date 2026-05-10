"""Extract K⁺ Tafel slopes from Brianna 2019 0.1 M K₂SO₄ RRDE LSV data.

Phase 6β v9 post-Gate-4 plan §F (parallel-safe).

The source xlsx ``data/EChem Reactor Modeling-Seitz-Mangan/Brianna/
0,1M K2SO4 data 8-15-19.xlsx`` documents 6 disks at pH ∈
{6.39, 5.21, 4.21, 3.42, 2.35, 1.65} in the ``Exp Info`` sheet, but
only Disk 1 (pH 6.39) has its raw E-vs-j traces in the workbook.
Disks 2–6 have summary statistics only (ring onset potential, max
ring current, peroxide selectivity) — no LSV scan tables.  The
plan's "6 pH values" assumption is therefore narrowed to "1 pH at
3 cycles" for this delivery; the gap is documented in
``docs/missing_data.md`` M1.

Per cycle, the workbook stores the iR-corrected disk current
density on a forward + reverse RHE-referenced voltage sweep.  This
script:

1. Parses the three cycle sheets (``cycle 1/2/3``) at the documented
   block offsets.
2. Isolates the cathodic forward sweep (``dE/dt < 0`` segment).
3. Selects the kinetic region — log|j| vs E should be monotonically
   linear there.  Defaults to ``|j| ∈ [10%, 60%] |j_limiting|`` which
   strips the onset noise and the mass-transport plateau.
4. Linear-fits ``log10|j| = a · E + b`` and reports the slope as
   ``b_Tafel = −1 / a`` in mV/decade (sign convention: cathodic
   ORR Tafel slope is positive).
5. Writes ``data/derived/k_plus_tafel_slopes_from_brianna_2019.xlsx``
   with one row per cycle: cycle, pH, slope_mV_per_decade,
   intercept_logj_at_E0, R², N_points, V_min, V_max, j_min, j_max.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))


SRC_XLSX = os.path.join(
    _ROOT,
    "data",
    "EChem Reactor Modeling-Seitz-Mangan",
    "Brianna",
    "0,1M K2SO4 data 8-15-19.xlsx",
)
OUT_XLSX = os.path.join(
    _ROOT, "data", "derived", "k_plus_tafel_slopes_from_brianna_2019.xlsx"
)
OUT_JSON = os.path.join(
    _ROOT, "data", "derived", "k_plus_tafel_slopes_from_brianna_2019.json"
)

# Documented sheet/column layout (verified 2026-05-10).
CYCLE_SHEETS = ("cycle 1", "cycle 2", "cycle 3")
DATA_FIRST_ROW = 8                # row index where numeric data begins
COL_E_VS_RHE_IR = 6               # 'Edisk/V vs RHE (iR New)'
COL_J_DISK_MA_CM2 = 7             # 'Idisk/mA/cm^2'

# Tafel kinetic-region defaults.  See module docstring §3.
J_FRAC_LO_DEFAULT = 0.10          # drop |j| < 10% |j_lim| (onset noise)
J_FRAC_HI_DEFAULT = 0.60          # drop |j| > 60% |j_lim| (plateau)
PH_LSV_DEFAULT = 6.39             # documented in 'Exp Info' Disk 1


@dataclass(frozen=True)
class TafelFit:
    cycle_name: str
    pH: float
    slope_mV_per_decade: float
    intercept_log_j_at_e0_v: float   # log10|j| extrapolated to E=0 vs RHE
    r_squared: float
    n_points: int
    e_min_v: float
    e_max_v: float
    j_min_ma_cm2: float
    j_max_ma_cm2: float
    j_lim_ma_cm2: float              # estimated mass-transport plateau
    j_frac_window_lo: float
    j_frac_window_hi: float
    notes: Optional[str] = None


def _load_cycle_lsv(xlsx_path: str, sheet_name: str) -> pd.DataFrame:
    """Parse a 'cycle N' sheet into a DataFrame with columns (E_V, j_mA_cm2).

    Assumes the documented Brianna 2019 layout: row index 7 is the
    in-sheet header, row 8+ is numeric data; col 6 is iR-corrected
    voltage vs RHE, col 7 is disk current density.  Drops rows where
    either column is non-numeric.
    """
    raw = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)
    if raw.shape[0] < DATA_FIRST_ROW + 1:
        raise ValueError(
            f"{sheet_name!r}: expected ≥{DATA_FIRST_ROW + 1} rows, "
            f"got {raw.shape[0]}"
        )
    if raw.shape[1] <= COL_J_DISK_MA_CM2:
        raise ValueError(
            f"{sheet_name!r}: expected ≥{COL_J_DISK_MA_CM2 + 1} columns, "
            f"got {raw.shape[1]}"
        )
    data = raw.iloc[DATA_FIRST_ROW:, [COL_E_VS_RHE_IR, COL_J_DISK_MA_CM2]]
    data.columns = ["E_V", "j_mA_cm2"]
    data = data.apply(pd.to_numeric, errors="coerce").dropna()
    return data.reset_index(drop=True)


def _split_forward_cathodic(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, str]:
    """Return only the forward (E decreasing) sweep slice.

    LSV traces ramp E from a starting potential down to a turning
    point (forward sweep) and back up (reverse sweep).  Tafel analysis
    uses the forward sweep to avoid hysteresis.
    """
    # Find the index of the minimum E — that's the turning point.
    e = df["E_V"].to_numpy()
    i_min = int(np.argmin(e))
    forward = df.iloc[: i_min + 1].copy()
    note = (
        f"forward sweep [0:{i_min + 1}] of {len(df)} rows; turning point "
        f"at E={e[i_min]:.4f} V"
    )
    return forward, note


def _tafel_window(
    df_cathodic: pd.DataFrame,
    *,
    j_frac_lo: float,
    j_frac_hi: float,
) -> Tuple[pd.DataFrame, float, str]:
    """Slice the kinetic region between j_frac_lo·|j_lim| and j_frac_hi·|j_lim|.

    j_lim is estimated as the most-negative current density observed
    in the cathodic forward sweep (typical RRDE convention: the
    plateau current at the rotation rate).  Returns the filtered
    DataFrame plus the j_lim estimate and a one-line note.
    """
    j = df_cathodic["j_mA_cm2"].to_numpy()
    cathodic = df_cathodic[j < 0.0].copy()
    if len(cathodic) == 0:
        raise ValueError(
            "no cathodic (j<0) rows in forward sweep; LSV may not have "
            "crossed the open-circuit potential"
        )
    j_lim = float(cathodic["j_mA_cm2"].min())
    abs_jlim = abs(j_lim)
    abs_j = cathodic["j_mA_cm2"].abs()
    keep = (abs_j > j_frac_lo * abs_jlim) & (abs_j < j_frac_hi * abs_jlim)
    sliced = cathodic[keep].copy().reset_index(drop=True)
    note = (
        f"|j_lim|={abs_jlim:.4f} mA/cm²; window |j| ∈ "
        f"({j_frac_lo:.2f}, {j_frac_hi:.2f}) · |j_lim| → "
        f"{len(sliced)} rows"
    )
    return sliced, j_lim, note


def _fit_tafel(
    df_window: pd.DataFrame,
) -> Tuple[float, float, float]:
    """Fit log10|j| = a · E + b on the kinetic-region slice.

    Returns (a, b, r²).  The Tafel slope in mV/decade is -1000 / a
    (cathodic Tafel slopes are conventionally reported as positive
    even though dE/d(log|j|) is negative for cathodic reactions).
    """
    if len(df_window) < 3:
        raise ValueError(
            f"insufficient points for fit ({len(df_window)} < 3)"
        )
    e = df_window["E_V"].to_numpy()
    log_abs_j = np.log10(df_window["j_mA_cm2"].abs().to_numpy())
    a, b = np.polyfit(e, log_abs_j, deg=1)
    log_abs_j_pred = a * e + b
    ss_res = float(np.sum((log_abs_j - log_abs_j_pred) ** 2))
    ss_tot = float(np.sum((log_abs_j - log_abs_j.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(a), float(b), float(r_squared)


def fit_one_cycle(
    xlsx_path: str,
    sheet_name: str,
    *,
    pH: float = PH_LSV_DEFAULT,
    j_frac_lo: float = J_FRAC_LO_DEFAULT,
    j_frac_hi: float = J_FRAC_HI_DEFAULT,
) -> TafelFit:
    full = _load_cycle_lsv(xlsx_path, sheet_name)
    forward, fwd_note = _split_forward_cathodic(full)
    window, j_lim, win_note = _tafel_window(
        forward, j_frac_lo=j_frac_lo, j_frac_hi=j_frac_hi,
    )
    a, b, r2 = _fit_tafel(window)
    slope_mV_per_decade = -1000.0 / a if a != 0.0 else float("nan")
    return TafelFit(
        cycle_name=sheet_name,
        pH=float(pH),
        slope_mV_per_decade=float(slope_mV_per_decade),
        intercept_log_j_at_e0_v=float(b),
        r_squared=float(r2),
        n_points=int(len(window)),
        e_min_v=float(window["E_V"].min()),
        e_max_v=float(window["E_V"].max()),
        j_min_ma_cm2=float(window["j_mA_cm2"].min()),
        j_max_ma_cm2=float(window["j_mA_cm2"].max()),
        j_lim_ma_cm2=float(j_lim),
        j_frac_window_lo=float(j_frac_lo),
        j_frac_window_hi=float(j_frac_hi),
        notes=f"{fwd_note}; {win_note}",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract K⁺ Tafel slopes from Brianna 2019 0.1 M K₂SO₄ "
            "LSV data (Phase 6β v9 post-Gate-4 plan §F)."
        )
    )
    parser.add_argument(
        "--src-xlsx", default=SRC_XLSX,
        help="Source Brianna xlsx (default: %(default)s)",
    )
    parser.add_argument(
        "--out-xlsx", default=OUT_XLSX,
        help="Output xlsx (default: %(default)s)",
    )
    parser.add_argument(
        "--out-json", default=OUT_JSON,
        help="Output JSON (default: %(default)s)",
    )
    parser.add_argument(
        "--j-frac-lo", type=float, default=J_FRAC_LO_DEFAULT,
        help=(
            "Lower bound of Tafel kinetic-region window as fraction of "
            "|j_lim| (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--j-frac-hi", type=float, default=J_FRAC_HI_DEFAULT,
        help=(
            "Upper bound of Tafel kinetic-region window as fraction of "
            "|j_lim| (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--pH", type=float, default=PH_LSV_DEFAULT,
        help="LSV pH (Brianna disk 1 = 6.39, default: %(default)s)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.src_xlsx):
        print(
            f"[tafel] source xlsx missing: {args.src_xlsx}",
            file=sys.stderr,
        )
        return 1

    fits = []
    for sheet in CYCLE_SHEETS:
        try:
            fit = fit_one_cycle(
                args.src_xlsx, sheet,
                pH=args.pH,
                j_frac_lo=args.j_frac_lo,
                j_frac_hi=args.j_frac_hi,
            )
            print(
                f"[tafel] {sheet}: slope = {fit.slope_mV_per_decade:.2f} "
                f"mV/decade (R² = {fit.r_squared:.4f}, "
                f"N = {fit.n_points}, "
                f"E ∈ [{fit.e_min_v:.3f}, {fit.e_max_v:.3f}] V)",
                flush=True,
            )
            fits.append(fit)
        except Exception as exc:
            print(
                f"[tafel] {sheet}: FAILED ({type(exc).__name__}: {exc})",
                file=sys.stderr,
            )

    if not fits:
        print("[tafel] no successful fits; aborting", file=sys.stderr)
        return 1

    # Persist as DataFrame → xlsx + JSON.
    rows = [
        {
            "cycle_name": f.cycle_name,
            "pH": f.pH,
            "slope_mV_per_decade": f.slope_mV_per_decade,
            "intercept_log10j_at_E0_V": f.intercept_log_j_at_e0_v,
            "R_squared": f.r_squared,
            "N_points": f.n_points,
            "E_min_V_vs_RHE": f.e_min_v,
            "E_max_V_vs_RHE": f.e_max_v,
            "j_min_mA_cm2": f.j_min_ma_cm2,
            "j_max_mA_cm2": f.j_max_ma_cm2,
            "j_lim_mA_cm2": f.j_lim_ma_cm2,
            "window_frac_lo": f.j_frac_window_lo,
            "window_frac_hi": f.j_frac_window_hi,
            "notes": f.notes,
        }
        for f in fits
    ]
    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(args.out_xlsx), exist_ok=True)
    df.to_excel(args.out_xlsx, index=False, sheet_name="K_plus_Tafel")
    with open(args.out_json, "w") as fh:
        json.dump({
            "source_xlsx": os.path.relpath(args.src_xlsx, _ROOT),
            "extraction_script": os.path.relpath(__file__, _ROOT),
            "cation": "K+",
            "electrolyte": "0.1 M K2SO4",
            "rotation_rate": "documented in source xlsx; see Exp Info",
            "j_frac_window_lo": args.j_frac_lo,
            "j_frac_window_hi": args.j_frac_hi,
            "fits": rows,
            "scope_caveat": (
                "Source xlsx documents 6 pH values (6.39, 5.21, 4.21, "
                "3.42, 2.35, 1.65) in 'Exp Info' but only the pH 6.39 "
                "Disk 1 raw LSV traces are in the workbook; pH 5.21–"
                "1.65 raw traces are missing.  See docs/missing_data.md "
                "M1 for the gap log."
            ),
        }, fh, indent=2)

    print(f"\n[tafel] DONE. {len(fits)} cycles fit.", flush=True)
    print(f"[tafel]      out_xlsx = {args.out_xlsx}", flush=True)
    print(f"[tafel]      out_json = {args.out_json}", flush=True)
    print(f"[tafel]      mean slope = "
          f"{np.mean([f.slope_mV_per_decade for f in fits]):.2f} mV/decade",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
