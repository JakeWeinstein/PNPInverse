"""L_eff transport-domain sweep plotter — visual companion to
``score_l_eff_sweep.py``.

Reads ``summary.json`` and per-combo ``iv_curve.json`` from
``StudyResults/fast_realignment_2026-05-08/l_eff_transport_sweep/`` and
emits a 4-panel figure:

  (top-left)     cd vs V_RHE for all 8 (L_eff, ratio) combos overlaid.
                 L_eff distinguishes color (viridis, dark→bright =
                 thick→thin); ratio distinguishes linestyle
                 (``-`` for 1e-18, ``--`` for 1e-30).  The deck
                 left-plateau (~-0.18 mA/cm²) and peak (~-0.40 mA/cm²
                 at +0.10 V) are drawn as guide lines so the visual gap
                 is obvious at a glance.
  (top-right)    |cd_plateau| vs 1/L_eff in log-log.  Levich linearity
                 prediction = a straight line of slope 1; a fitted line
                 is overlaid with the slope annotated.  This is the
                 quantitative falsifiable check for prediction 1.
  (bottom-left)  pc vs V_RHE (gross R_2e peroxide current).
  (bottom-right) Surface pH proxy vs V_RHE.  pH < 9 at the smallest
                 L_eff and deepest cathodic V_RHE = falsifiable
                 prediction 3 PASS.

No Firedrake imports — pure JSON / numpy / matplotlib.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))

DEFAULT_SWEEP_DIR = (
    Path(_ROOT) / "StudyResults" / "fast_realignment_2026-05-08"
    / "l_eff_transport_sweep"
)
DEFAULT_PNG = DEFAULT_SWEEP_DIR / "l_eff_transport_sweep_overlay.png"

# Mangan page-15 deck features (memory:
# project_mangan_full_grid_curves.md, project_pass_a_outcome.md).
DECK_LEFT_PLATEAU_MA_CM2 = -0.18
DECK_PEAK_MA_CM2 = -0.40
DECK_PEAK_V_RHE = +0.10

LINESTYLE_BY_RATIO = {
    1e-18: "-",
    1e-30: "--",
    1e-24: "--",   # historic alias used in the prior ratio sweep
}


def _combo_label(l_eff_m: float, ratio: float) -> str:
    return f"L{round(l_eff_m * 1e6)}um_ratio_{ratio:g}"


def _arr(data: dict, key: str) -> np.ndarray:
    return np.array(
        [x if x is not None else np.nan for x in data.get(key, [])],
        dtype=float,
    )


def _load_combo(sweep_dir: Path, l_eff_m: float, ratio: float) -> dict:
    path = sweep_dir / _combo_label(l_eff_m, ratio) / "iv_curve.json"
    with open(path) as f:
        return json.load(f)


def _cd_plateau(data: dict, *, v_rhe_min: float = -0.40) -> float | None:
    """Return cd at the deepest cathodic V_RHE that converged.

    Falls back through the V_RHE grid until it finds a converged point;
    returns ``None`` only if nothing converged on the cathodic side.
    """
    v_rhe = np.array(data["v_rhe"], dtype=float)
    cd = _arr(data, "cd_mA_cm2")
    converged = np.array(data["converged"], dtype=bool)
    mask = np.isfinite(cd) & converged
    if not mask.any():
        return None
    # Pick the most cathodic converged point (smallest V_RHE).
    idx = np.where(mask)[0]
    j = idx[np.argmin(v_rhe[idx])]
    if v_rhe[j] > v_rhe_min + 0.05:
        return None
    return float(cd[j])


def plot_overlay(sweep_dir: Path, png_path: Path) -> None:
    with open(sweep_dir / "summary.json") as f:
        summary = json.load(f)

    l_eff_values = list(summary["l_eff_values_m"])
    ratios = list(summary["ratios"])
    anchor_v = float(summary.get("anchor_v_rhe", 0.55))

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    ax_cd, ax_levich = axes[0]
    ax_pc, ax_pH = axes[1]

    cmap = matplotlib.colormaps["viridis"]

    # Panel 1: cd vs V_RHE overlay.
    plateau_by_ratio: dict[float, list[tuple[float, float]]] = {
        r: [] for r in ratios
    }
    for ratio_idx, ratio in enumerate(ratios):
        ls = LINESTYLE_BY_RATIO.get(ratio, "-.")
        for l_idx, l_eff_m in enumerate(l_eff_values):
            try:
                data = _load_combo(sweep_dir, l_eff_m, ratio)
            except FileNotFoundError:
                continue
            v_rhe = np.array(data["v_rhe"], dtype=float)
            converged = np.array(data["converged"], dtype=bool)
            cd = _arr(data, "cd_mA_cm2")
            color = cmap(l_idx / max(len(l_eff_values) - 1, 1))
            mask = converged & np.isfinite(cd)
            label = (
                f"L={l_eff_m * 1e6:.0f} µm, ratio={ratio:g}"
            )
            if mask.any():
                ax_cd.plot(
                    v_rhe[mask], cd[mask],
                    marker="o", linestyle=ls, color=color,
                    markersize=3.5, alpha=0.9, label=label,
                )

                pc = _arr(data, "pc_mA_cm2")
                pc_mask = converged & np.isfinite(pc)
                if pc_mask.any():
                    ax_pc.plot(
                        v_rhe[pc_mask], pc[pc_mask],
                        marker="s", linestyle=ls, color=color,
                        markersize=3.5, alpha=0.9, label=label,
                    )

                pH = _arr(data, "surface_pH_proxy")
                pH_mask = converged & np.isfinite(pH)
                if pH_mask.any():
                    ax_pH.plot(
                        v_rhe[pH_mask], pH[pH_mask],
                        marker="d", linestyle=ls, color=color,
                        markersize=3.5, alpha=0.9, label=label,
                    )

            plateau = _cd_plateau(data)
            if plateau is not None:
                plateau_by_ratio[ratio].append((float(l_eff_m), float(plateau)))

    # Deck guide lines.
    ax_cd.axhline(
        DECK_LEFT_PLATEAU_MA_CM2, color="red", linestyle=":", alpha=0.6,
        label=f"deck left plateau ({DECK_LEFT_PLATEAU_MA_CM2:+.2f} mA/cm²)",
    )
    ax_cd.axhline(
        DECK_PEAK_MA_CM2, color="darkred", linestyle=":", alpha=0.6,
        label=f"deck peak ({DECK_PEAK_MA_CM2:+.2f} mA/cm²)",
    )
    ax_cd.axvline(
        DECK_PEAK_V_RHE, color="darkred", linestyle=":", alpha=0.4,
    )
    ax_cd.axvline(anchor_v, color="gray", linestyle=":", alpha=0.6)
    ax_cd.set_xlabel("V_RHE (V)")
    ax_cd.set_ylabel("cd total (mA/cm²)")
    ax_cd.set_title("Disk-side: total current density")
    ax_cd.grid(alpha=0.3)
    ax_cd.legend(fontsize=7, loc="best", framealpha=0.85, ncol=2)

    # Panel 2: |cd_plateau| vs 1/L_eff log-log + Levich slope fit.
    for ratio in ratios:
        pts = plateau_by_ratio[ratio]
        if not pts:
            continue
        L = np.array([p[0] for p in pts], dtype=float)
        plateau = np.array([p[1] for p in pts], dtype=float)
        inv_L = 1.0 / L
        abs_plateau = np.abs(plateau)

        ls = LINESTYLE_BY_RATIO.get(ratio, "-.")
        ax_levich.loglog(
            inv_L, abs_plateau, marker="o", linestyle=ls,
            label=f"ratio={ratio:g}",
        )

        # Slope-1 Levich reference line (anchored at the largest L_eff).
        mask = np.isfinite(inv_L) & np.isfinite(abs_plateau) & (abs_plateau > 0)
        if mask.sum() >= 2:
            log_invL = np.log(inv_L[mask])
            log_p = np.log(abs_plateau[mask])
            slope, intercept = np.polyfit(log_invL, log_p, 1)
            invL_dense = np.linspace(log_invL.min(), log_invL.max(), 50)
            ax_levich.plot(
                np.exp(invL_dense),
                np.exp(intercept + slope * invL_dense),
                linestyle=":", alpha=0.6,
                label=f"  fit: slope = {slope:+.2f}",
            )

    # Reference unit-slope line for visual sanity.
    if len(l_eff_values) >= 2:
        invL_ref = 1.0 / np.array(sorted(l_eff_values))
        ref_y = invL_ref * abs(DECK_LEFT_PLATEAU_MA_CM2) / invL_ref[-1]
        ax_levich.loglog(
            invL_ref, ref_y, color="black", linestyle="--", alpha=0.4,
            label="slope-1 (Levich) reference",
        )

    ax_levich.set_xlabel("1 / L_eff  (1/m)")
    ax_levich.set_ylabel("|cd_plateau|  (mA/cm²)")
    ax_levich.set_title(
        "Levich linearity check: slope ≈ 1 confirms 1/L_eff scaling"
    )
    ax_levich.grid(True, which="both", alpha=0.3)
    ax_levich.legend(fontsize=8, loc="best", framealpha=0.85)

    # Panel 3: pc decoration.
    ax_pc.axvline(anchor_v, color="gray", linestyle=":", alpha=0.6)
    ax_pc.set_xlabel("V_RHE (V)")
    ax_pc.set_ylabel("pc gross R_2e (mA/cm²)")
    ax_pc.set_title("Disk-side: gross peroxide (R_2e) current")
    ax_pc.grid(alpha=0.3)
    ax_pc.legend(fontsize=7, loc="best", framealpha=0.85, ncol=2)

    # Panel 4: surface pH guide line at pH = 9 (prediction 3 threshold).
    ax_pH.axhline(
        9.0, color="red", linestyle=":", alpha=0.6,
        label="pH < 9 threshold (prediction 3)",
    )
    ax_pH.axvline(anchor_v, color="gray", linestyle=":", alpha=0.6)
    ax_pH.set_xlabel("V_RHE (V)")
    ax_pH.set_ylabel("surface pH proxy")
    ax_pH.set_title("Surface pH proxy — does the H+ floor open up at small L_eff?")
    ax_pH.grid(alpha=0.3)
    ax_pH.legend(fontsize=7, loc="best", framealpha=0.85, ncol=2)

    fig.suptitle(
        "L_eff transport-domain sweep — H+ Levich-limit hypothesis "
        f"(anchor @ {anchor_v:+.2f} V, dotted)"
    )
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=160)
    plt.close(fig)


def main(argv: list[str]) -> int:
    sweep_dir = Path(argv[1]) if len(argv) > 1 else DEFAULT_SWEEP_DIR
    png_path = Path(argv[2]) if len(argv) > 2 else DEFAULT_PNG
    if not (sweep_dir / "summary.json").exists():
        print(f"ERROR: summary.json not found at {sweep_dir}", file=sys.stderr)
        return 1
    plot_overlay(sweep_dir, png_path)
    size = png_path.stat().st_size
    print(f"  output -> {png_path} ({size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
