"""Mangan full-grid plotter — overlay cd & pc over page-15 V_RHE band.

Reads ``summary.json`` and per-ratio ``iv_curve.json`` from
``StudyResults/fast_realignment_2026-05-08/mangan_full_grid/`` and
emits a 2-panel PNG: cd vs V_RHE on the left, pc (gross R_2e) on the
right. Linear y-scale (vs the symlog used for the wider ratio sweep)
since the two ratios here span similar magnitudes; that lets the
Butler shape read clearly at a glance against the deck band.

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
    / "mangan_full_grid"
)
DEFAULT_PNG = DEFAULT_SWEEP_DIR / "mangan_full_grid_overlay.png"


def _ratio_label(ratio: float) -> str:
    return f"ratio_{ratio:g}"


def _arr(data, key):
    return np.array(
        [x if x is not None else np.nan for x in data.get(key, [])],
        dtype=float,
    )


def plot_overlay(sweep_dir: Path, png_path: Path) -> None:
    with open(sweep_dir / "summary.json") as f:
        summary = json.load(f)
    ratios = list(summary["ratios"])
    anchor_v = float(summary.get("anchor_v_rhe", 0.55))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    ax_cd, ax_pc = axes[0]
    ax_ring, ax_sel = axes[1]

    cmap = matplotlib.colormaps["viridis"]

    for idx, ratio in enumerate(ratios):
        label = _ratio_label(ratio)
        with open(sweep_dir / label / "iv_curve.json") as f:
            data = json.load(f)
        v_rhe = np.array(data["v_rhe"], dtype=float)
        converged = np.array(data["converged"], dtype=bool)
        cd = _arr(data, "cd_mA_cm2")
        pc = _arr(data, "pc_mA_cm2")
        j_ring = _arr(data, "j_ring_mA_cm2")
        sel = _arr(data, "S_H2O2_percent")

        color = cmap(idx / max(len(ratios) - 1, 1))
        leg = f"K0_R4e/K0_R2e = {ratio:g}"
        if not bool(data.get("anchor", {}).get("converged", False)):
            leg += " [anchor failed]"

        def _plot(ax, y, marker, ls):
            mask = converged & np.isfinite(y)
            if mask.any():
                ax.plot(
                    v_rhe[mask], y[mask],
                    marker=marker, linestyle=ls, color=color,
                    label=leg, alpha=0.9,
                )
            else:
                ax.plot([], [], color=color, label=leg)

        _plot(ax_cd, cd, "o", "-")
        _plot(ax_pc, pc, "s", "--")
        _plot(ax_ring, j_ring, "^", "-.")
        _plot(ax_sel, sel, "D", ":")

    for ax in (ax_cd, ax_pc, ax_ring, ax_sel):
        ax.axvline(
            anchor_v, color="gray", linestyle=":", alpha=0.6,
        )
        ax.set_xlabel("V_RHE (V)")
        ax.grid(alpha=0.3)

    ax_cd.set_ylabel("cd total (mA/cm²)")
    ax_cd.set_title("Disk-side: total current density")
    ax_pc.set_ylabel("pc gross R_2e (mA/cm²)")
    ax_pc.set_title("Disk-side: gross peroxide current")
    ax_ring.set_ylabel("j_ring (mA/cm²)")
    ax_ring.set_title(f"Ring-side (deck-aligned, N={summary.get('n_collection_used', 0.224) if False else 0.224})")
    ax_sel.set_ylabel("S_H2O2 (%)")
    ax_sel.set_title("Peroxide selectivity (RRDE)")
    ax_sel.set_ylim(-5, 105)

    for ax in (ax_cd, ax_pc, ax_ring, ax_sel):
        ax.legend(fontsize=9, loc="best", framealpha=0.85)

    fig.suptitle(
        "Promising K0_R4e/K0_R2e ratios — Mangan page-15 V_RHE band"
        f"  (anchor @ {anchor_v:+.2f} V, dotted)"
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
