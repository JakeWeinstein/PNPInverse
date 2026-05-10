"""Pass A grid plotter — read iv_curve JSON, emit cd & pc PNG.

Reads
``StudyResults/fast_realignment_2026-05-08/pass_a_grid/pass_a_iv_curve.json``
and writes ``pass_a_cd_pc.png`` next to it. Two y-axes (cd left, pc
right), V_RHE on x. Failed points are skipped from the curves and
annotated as red Xs at the bottom rail. The anchor V_RHE is marked
with a dotted vertical line.

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

DEFAULT_JSON = (
    Path(_ROOT) / "StudyResults" / "fast_realignment_2026-05-08"
    / "pass_a_grid" / "pass_a_iv_curve.json"
)
DEFAULT_PNG = DEFAULT_JSON.with_name("pass_a_cd_pc.png")

REQUIRED_KEYS = ("v_rhe", "cd_mA_cm2", "pc_mA_cm2", "converged",
                 "n_converged", "n_total")


def _load_iv_curve(json_path: Path) -> dict:
    with open(json_path) as f:
        data = json.load(f)
    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        raise ValueError(
            f"iv_curve JSON missing keys: {missing!r} "
            f"(expected {list(REQUIRED_KEYS)!r})"
        )
    return data


def plot_pass_a(json_path: Path, png_path: Path) -> None:
    data = _load_iv_curve(json_path)

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

    n_conv = int(data["n_converged"])
    n_tot = int(data["n_total"])
    anchor_v = (data.get("anchor") or {}).get("v_rhe")

    fig, ax_cd = plt.subplots(figsize=(7.5, 4.8))
    ax_pc = ax_cd.twinx()

    mask = converged & np.isfinite(cd) & np.isfinite(pc)

    line_cd, = ax_cd.plot(
        v_rhe[mask], cd[mask],
        marker="o", linestyle="-", color="C0", label="cd (total)",
    )
    line_pc, = ax_pc.plot(
        v_rhe[mask], pc[mask],
        marker="s", linestyle="--", color="C3",
        label="pc (gross R_2e)",
    )

    # Mark failed voltages as red Xs near the bottom of the cd-axis range.
    failed = ~converged
    if failed.any():
        if mask.any():
            y_floor = float(np.nanmin(cd[mask]))
        else:
            y_floor = -1.0
        ax_cd.scatter(
            v_rhe[failed],
            np.full(int(failed.sum()), y_floor),
            marker="x", color="red", s=80, zorder=5, label="failed",
        )

    if anchor_v is not None:
        ax_cd.axvline(
            float(anchor_v), color="gray", linestyle=":", alpha=0.7,
            label=f"anchor @ {float(anchor_v):+.2f} V",
        )

    ax_cd.set_xlabel("V_RHE (V)")
    ax_cd.set_ylabel("cd (mA/cm²)", color="C0")
    ax_pc.set_ylabel("pc (mA/cm²)", color="C3")
    ax_cd.tick_params(axis="y", labelcolor="C0")
    ax_pc.tick_params(axis="y", labelcolor="C3")
    ax_cd.grid(alpha=0.3)

    title = f"Pass A IV curve — {n_conv}/{n_tot} converged"
    if failed.any():
        bad_vs = ", ".join(f"{v:+.2f}" for v in v_rhe[failed])
        title += f"  (failed: {bad_vs})"
    ax_cd.set_title(title)

    handles = [line_cd, line_pc]
    if failed.any():
        handles.append(ax_cd.collections[0])
    if anchor_v is not None:
        handles.append(ax_cd.lines[-1])
    labels = [h.get_label() for h in handles]
    ax_cd.legend(handles, labels, loc="best", framealpha=0.85)

    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=160)
    plt.close(fig)


def main(argv: list[str]) -> int:
    json_path = Path(argv[1]) if len(argv) > 1 else DEFAULT_JSON
    png_path = Path(argv[2]) if len(argv) > 2 else DEFAULT_PNG
    if not json_path.exists():
        print(f"ERROR: input JSON not found: {json_path}", file=sys.stderr)
        return 1
    plot_pass_a(json_path, png_path)
    size = png_path.stat().st_size
    print(f"  output -> {png_path} ({size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
