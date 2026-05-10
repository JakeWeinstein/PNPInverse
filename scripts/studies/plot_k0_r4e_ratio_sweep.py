"""K0_R4e ratio-sweep overlay plotter.

Reads ``summary.json`` from
``StudyResults/fast_realignment_2026-05-08/k0_r4e_ratio_sweep/`` and
the per-ratio iv_curve JSONs, then emits a 2-panel overlay PNG: cd vs
V_RHE on the left, pc (gross R_2e) vs V_RHE on the right. One curve
per ratio, color-coded by log10(ratio).

No Firedrake imports — pure JSON / numpy / matplotlib.
"""
from __future__ import annotations

import json
import math
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
    / "k0_r4e_ratio_sweep"
)
DEFAULT_PNG = DEFAULT_SWEEP_DIR / "k0_ratio_sweep_overlay.png"


def _load_summary(sweep_dir: Path) -> dict:
    with open(sweep_dir / "summary.json") as f:
        return json.load(f)


def _load_per_ratio(sweep_dir: Path, label: str) -> dict:
    with open(sweep_dir / label / "pass_a_iv_curve.json") as f:
        return json.load(f)


def _ratio_label(ratio: float) -> str:
    if ratio == 1.0:
        return "ratio_1e+00"
    return f"ratio_{ratio:g}"


def _color_for_ratio(ratio: float, n_total: int, idx: int):
    """Map ratios to a perceptually-ordered colormap (viridis)."""
    cmap = matplotlib.colormaps["viridis"]
    # Spread index across [0, 1).
    t = idx / max(n_total - 1, 1)
    return cmap(t)


def plot_overlay(sweep_dir: Path, png_path: Path) -> None:
    summary = _load_summary(sweep_dir)
    ratios = list(summary["ratios"])
    n_ratios = len(ratios)

    fig, (ax_cd, ax_pc) = plt.subplots(
        1, 2, figsize=(13, 5), sharex=True,
    )

    anchor_v = float(summary.get("anchor_v_rhe", 0.55))

    for idx, ratio in enumerate(ratios):
        label = _ratio_label(ratio)
        try:
            data = _load_per_ratio(sweep_dir, label)
        except FileNotFoundError:
            continue

        v_rhe = np.array(data["v_rhe"], dtype=float)
        converged = np.array(data["converged"], dtype=bool)
        cd = np.array(
            [x if x is not None else np.nan for x in data["cd_mA_cm2"]],
            dtype=float,
        )
        pc = np.array(
            [x if x is not None else np.nan for x in data["pc_mA_cm2"]],
            dtype=float,
        )
        anchor_ok = bool(data.get("anchor", {}).get("converged", False))
        mask = converged & np.isfinite(cd)
        mask_pc = converged & np.isfinite(pc)

        color = _color_for_ratio(ratio, n_ratios, idx)
        leg = f"{ratio:g}"
        if not anchor_ok:
            leg += " [anchor failed]"

        if mask.any():
            ax_cd.plot(
                v_rhe[mask], cd[mask],
                marker="o", linestyle="-", color=color,
                label=leg, alpha=0.9,
            )
        else:
            # No converged points; emit a placeholder so legend still shows.
            ax_cd.plot([], [], color=color, label=leg)

        if mask_pc.any():
            ax_pc.plot(
                v_rhe[mask_pc], pc[mask_pc],
                marker="s", linestyle="--", color=color,
                label=leg, alpha=0.9,
            )
        else:
            ax_pc.plot([], [], color=color, label=leg)

    for ax in (ax_cd, ax_pc):
        ax.axvline(
            anchor_v, color="gray", linestyle=":", alpha=0.6,
            label=f"anchor @ {anchor_v:+.2f} V",
        )
        ax.set_xlabel("V_RHE (V)")
        ax.grid(alpha=0.3)

    ax_cd.set_ylabel("cd total (mA/cm²)")
    ax_cd.set_title("Total current density")
    ax_pc.set_ylabel("pc gross R_2e (mA/cm²)")
    ax_pc.set_title("Gross peroxide current (deck-aligned)")

    # pc spans many decades across ratios — symlog handles sign + range.
    ax_pc.set_yscale("symlog", linthresh=1e-12)

    ax_cd.legend(
        title="K0_R4e/K0_R2e", fontsize=8, loc="best", framealpha=0.85,
    )
    ax_pc.legend(
        title="K0_R4e/K0_R2e", fontsize=8, loc="best", framealpha=0.85,
    )
    fig.suptitle(
        "K0_R4e/K0_R2e ratio sweep — anchor + warm-walk grid driver"
    )
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=160)
    plt.close(fig)


def main(argv: list[str]) -> int:
    sweep_dir = Path(argv[1]) if len(argv) > 1 else DEFAULT_SWEEP_DIR
    png_path = Path(argv[2]) if len(argv) > 2 else DEFAULT_PNG
    summary_path = sweep_dir / "summary.json"
    if not summary_path.exists():
        print(f"ERROR: summary.json not found at {summary_path}",
              file=sys.stderr)
        return 1
    plot_overlay(sweep_dir, png_path)
    size = png_path.stat().st_size
    print(f"  output -> {png_path} ({size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
