"""Plot Jithin Fig 4.31/4.32 emulations from our volume-scale sweep.

Reads ``StudyResults/jithin_fig_4_31_4_32/iv_curve.json`` and produces:

    fig_4_31_ratio.png  — c*_O2 / c^b_O2 vs V, one curve per volume scale
    fig_4_32_jv.png     — simulated jV, one curve per volume scale
    fig_combined.png    — 1×2 side-by-side panel

Axis conventions and colors match Jithin's thesis Fig 4.31/4.32 (blue =
a_k³, orange = a_k³/4, green = a_k³/10, red = a_k³/100) so the reader can
flip between pages.  We do not overlay his digitized data — this is a
simulator-vs-simulator shape/axes comparison.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]
JSON_PATH = _ROOT / "StudyResults" / "jithin_fig_4_31_4_32" / "iv_curve.json"
OUT_DIR = JSON_PATH.parent

# Jithin Fig 4.31/4.32 color order: full volume → smallest volume.
SCALE_COLORS = ["C0", "C1", "C2", "C3"]


def _nan_array(values: List[Any]) -> np.ndarray:
    return np.array(
        [np.nan if v is None else float(v) for v in values], dtype=float
    )


def _ok_scales(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [s for s in data.get("scales", []) if s.get("status") == "ok"]


def _v_axis(scale: Dict[str, Any]) -> np.ndarray:
    return np.array(
        scale.get("v_rhe_jithin") or scale["v_rhe"], dtype=float
    )


def plot_fig_4_31(data: Dict[str, Any]) -> Path:
    """c_O2(OHP)/c_O2_bulk vs V — Jithin Fig 4.31 axes."""
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for i, s in enumerate(_ok_scales(data)):
        v = _v_axis(s)
        ratio = _nan_array(s["ratio_O2_OHP_over_bulk"])
        finite = np.isfinite(ratio)
        ax.plot(
            v[finite], ratio[finite], lw=2.0,
            color=SCALE_COLORS[i % len(SCALE_COLORS)],
            label=s.get("label", s.get("tag")),
        )
    ax.set_xlabel("Applied Voltage (vs RHE)")
    ax.set_ylabel(r"$c^*_{O_2} / c^b_{O_2}$")
    ax.set_xlim(-0.45, 0.75)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Fig 4.31 reproduction — O$_2$(OHP)/bulk vs volume scale")
    out = OUT_DIR / "fig_4_31_ratio.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_fig_4_32(data: Dict[str, Any]) -> Path:
    """Simulated jV curves — Jithin Fig 4.32 axes."""
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for i, s in enumerate(_ok_scales(data)):
        v = _v_axis(s)
        cd = _nan_array(s["cd_mA_cm2"])
        finite = np.isfinite(cd)
        ax.plot(
            v[finite], cd[finite], lw=2.0,
            color=SCALE_COLORS[i % len(SCALE_COLORS)],
            label=s.get("label", s.get("tag")),
        )
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_xlabel("Applied Voltage (vs RHE)")
    ax.set_ylabel("Current Density (mA/cm$^2$)")
    ax.set_xlim(-0.45, 0.75)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Fig 4.32 reproduction — jV vs volume scale")
    out = OUT_DIR / "fig_4_32_jv.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_combined(data: Dict[str, Any]) -> Path:
    """1×2 panel: Fig 4.31 (ratio) + Fig 4.32 (jV)."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    scales = _ok_scales(data)

    for i, s in enumerate(scales):
        v = _v_axis(s)
        ratio = _nan_array(s["ratio_O2_OHP_over_bulk"])
        f1 = np.isfinite(ratio)
        axes[0].plot(
            v[f1], ratio[f1], lw=2.0,
            color=SCALE_COLORS[i % len(SCALE_COLORS)],
            label=s.get("label", s.get("tag")),
        )
        cd = _nan_array(s["cd_mA_cm2"])
        f2 = np.isfinite(cd)
        axes[1].plot(
            v[f2], cd[f2], lw=2.0,
            color=SCALE_COLORS[i % len(SCALE_COLORS)],
            label=s.get("label", s.get("tag")),
        )

    axes[0].set_xlabel("Applied Voltage (vs RHE)")
    axes[0].set_ylabel(r"$c^*_{O_2} / c^b_{O_2}$")
    axes[0].set_xlim(-0.45, 0.75)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].set_title("Fig 4.31: O$_2$(OHP)/bulk vs V")

    axes[1].axhline(0.0, color="k", lw=0.5)
    axes[1].set_xlabel("Applied Voltage (vs RHE)")
    axes[1].set_ylabel("Current Density (mA/cm$^2$)")
    axes[1].set_xlim(-0.45, 0.75)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="lower right", fontsize=9)
    axes[1].set_title("Fig 4.32: simulated jV")

    cfg = data.get("config", {})
    ocp = cfg.get("ocp_offset_v", 0.785)
    fig.suptitle(
        "Jithin Thesis §4.8 volume sweep — our solver reproduction "
        f"(L_eff={cfg.get('l_eff_m', 5e-5) * 1e6:.0f} µm, "
        f"C_S={cfg.get('stern_target', 1.738):.2f} F/m², "
        f"A_Tafel={cfg.get('A_tafel_jithin_target_mV_dec', 142):.0f} mV/dec, "
        f"Cs⁺/SO₄²⁻ pH 2, steric via PDE; "
        f"x-axis = Jithin V_RHE = ours + {ocp:.2f} V OCP)",
        fontsize=9.5, y=1.02,
    )
    out = OUT_DIR / "fig_combined.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    if not JSON_PATH.exists():
        print(f"ERROR: {JSON_PATH} not found; run the study first.",
              file=sys.stderr)
        return 1
    with open(JSON_PATH) as f:
        data = json.load(f)

    cfg = data.get("config", {})
    scales = _ok_scales(data)
    print(f"Loaded {JSON_PATH}")
    print(f"  config: L_eff={cfg.get('l_eff_m'):.1e} m, "
          f"C_S={cfg.get('stern_target'):.3f} F/m² (steric via PDE)")
    print(f"  scales with data: {len(scales)}/{len(data.get('scales', []))}")
    for s in scales:
        print(f"    {s['tag']:>12}: {s['n_converged']}/{s['n_total']} converged")

    paths = [
        plot_fig_4_31(data),
        plot_fig_4_32(data),
        plot_combined(data),
    ]
    for p in paths:
        if p:
            print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
