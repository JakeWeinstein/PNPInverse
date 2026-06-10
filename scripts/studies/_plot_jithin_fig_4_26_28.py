"""Plot Jithin Fig 4.26/4.27/4.28 emulations from our solver's V-sweep.

Reads ``StudyResults/jithin_fig_4_26_4_27_4_28/iv_curve.json`` and produces:

    fig_4_26_ratio.png   — c*_O2 / c^b_O2 vs V_RHE       (Jithin Fig 4.26)
    fig_4_27_O2.png      — c_O2(y) at 2 voltages         (Jithin Fig 4.27)
    fig_4_28_H.png       — c_H+(y) at 2 voltages         (Jithin Fig 4.28)
    fig_combined.png     — 1×3 side-by-side panel

Axis conventions match Jithin's thesis figures so the reader can flip
between pages. We do not overlay Jithin's curves (we don't have his data)
— this is simulator-vs-simulator shape/axes comparison.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]
JSON_PATH = _ROOT / "StudyResults" / "jithin_fig_4_26_4_27_4_28" / "iv_curve.json"
OUT_DIR = JSON_PATH.parent


def _nan_array(values: list[Any]) -> np.ndarray:
    return np.array(
        [np.nan if v is None else float(v) for v in values], dtype=float
    )


def plot_fig_4_26(data: Dict[str, Any]) -> Path:
    """c_O2(OHP) / c_O2_bulk vs V_RHE — Jithin Fig 4.26 axes."""
    v = np.array(data.get("v_rhe_jithin") or data["v_rhe"], dtype=float)
    ratio = _nan_array(data["ratio_O2_OHP_over_bulk"])
    finite = np.isfinite(ratio)

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(v[finite], ratio[finite], lw=2.0, color="C0", label="Our solver")
    ax.set_xlabel("Applied Voltage (vs RHE)")
    ax.set_ylabel(r"$c^*_{O_2} / c^b_{O_2}$")
    ax.set_xlim(0.0, 0.75)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_title(
        "Fig 4.26 reproduction — Bikerman occlusion vs V\n"
        f"({finite.sum()}/{len(v)} grid points converged)"
    )
    out = OUT_DIR / "fig_4_26_ratio.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def _plot_profile_panel(ax, profiles: list, species_key: str,
                        y_label: str, x_max_nm: float = 14.0):
    """Render c_species(x_nm) for each captured voltage."""
    colors = ["C0", "C1", "C2", "C3"]
    linestyles = ["-", "--", "-.", ":"]
    for i, prof in enumerate(profiles):
        y_nm = np.array(prof["y_nm"], dtype=float)
        # Jithin plots in mol/L (M), we have mol/m³ → divide by 1000.
        c_mol_L = np.array(prof[species_key], dtype=float) / 1000.0
        # Prefer Jithin-convention V label if present.
        v_label = prof.get("v_rhe_jithin", prof.get("v_rhe"))
        ax.plot(
            y_nm, c_mol_L,
            color=colors[i % len(colors)],
            ls=linestyles[i % len(linestyles)],
            lw=2.0,
            label=f"V = {v_label:+.2f} V vs RHE",
        )
    ax.set_xlabel("x (nm)")
    ax.set_ylabel(y_label)
    ax.set_xlim(0, x_max_nm)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)


def plot_fig_4_27(data: Dict[str, Any]) -> Path:
    """c_O2(y) at 2 voltages — Jithin Fig 4.27."""
    profiles = data.get("profiles", [])
    if not profiles:
        return Path()

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    _plot_profile_panel(
        ax, profiles, species_key="c_O2_mol_m3",
        y_label="Conc of O2 (M)", x_max_nm=14.0,
    )
    ax.set_title("Fig 4.27 reproduction — O$_2$ profiles at 2 voltages")
    # Match Jithin's y-axis: 0 to 0.00025 M (c_O2_bulk = 0.25 mol/m³ = 2.5e-4 M)
    ax.set_ylim(-1e-5, 3.0e-4)
    out = OUT_DIR / "fig_4_27_O2.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_fig_4_28(data: Dict[str, Any]) -> Path:
    """c_H+(y) at 2 voltages — Jithin Fig 4.28."""
    profiles = data.get("profiles", [])
    if not profiles:
        return Path()

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    _plot_profile_panel(
        ax, profiles, species_key="c_H_mol_m3",
        y_label=r"Conc of H$^+$ (M)", x_max_nm=14.0,
    )
    ax.set_title(r"Fig 4.28 reproduction — H$^+$ profiles at 2 voltages")
    # Match Jithin's y-axis (range varies, give us some headroom)
    out = OUT_DIR / "fig_4_28_H.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_combined(data: Dict[str, Any]) -> Path:
    """1×3 panel of all three Jithin figures side-by-side."""
    v = np.array(data.get("v_rhe_jithin") or data["v_rhe"], dtype=float)
    ratio = _nan_array(data["ratio_O2_OHP_over_bulk"])
    finite = np.isfinite(ratio)
    profiles = data.get("profiles", [])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # Panel 1: Fig 4.26
    axes[0].plot(v[finite], ratio[finite], lw=2.0, color="C0")
    axes[0].set_xlabel("Applied Voltage (vs RHE)")
    axes[0].set_ylabel(r"$c^*_{O_2} / c^b_{O_2}$")
    axes[0].set_xlim(0.0, 0.75)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Fig 4.26: O$_2$(OHP)/bulk vs V")

    # Panel 2: Fig 4.27
    _plot_profile_panel(
        axes[1], profiles, species_key="c_O2_mol_m3",
        y_label="Conc of O2 (M)", x_max_nm=14.0,
    )
    axes[1].set_title("Fig 4.27: O$_2$(x) at 2 voltages")
    axes[1].set_ylim(-1e-5, 3.0e-4)

    # Panel 3: Fig 4.28
    _plot_profile_panel(
        axes[2], profiles, species_key="c_H_mol_m3",
        y_label=r"Conc of H$^+$ (M)", x_max_nm=14.0,
    )
    axes[2].set_title(r"Fig 4.28: H$^+$(x) at 2 voltages")

    cfg = data.get("config", {})
    ocp = cfg.get("ocp_offset_v", 0.785)
    fig.suptitle(
        f"Jithin Thesis §4.8 default settings — our solver reproduction "
        f"(L_eff={cfg.get('l_eff_m', 5e-5) * 1e6:.0f} µm, "
        f"C_S={cfg.get('stern_target', 1.738):.2f} F/m², "
        f"A_Tafel={cfg.get('A_tafel_mV_dec', 142):.0f} mV/dec, "
        f"Cs⁺/SO₄²⁻ pH 2; x-axis is Jithin V_RHE = ours + {ocp:.2f} V OCP)",
        fontsize=10, y=1.02,
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
    print(f"Loaded {JSON_PATH}")
    print(f"  config: L_eff={cfg.get('l_eff_m'):.1e} m, "
          f"C_S={cfg.get('stern_target'):.3f} F/m², "
          f"α={cfg.get('alpha_tafel'):.3f}, "
          f"A_Tafel={cfg.get('A_tafel_mV_dec'):.1f} mV/dec")
    print(f"  converged: {data['n_converged']}/{data['n_total']}")
    print(f"  profiles captured: {len(data.get('profiles', []))}")

    paths = [
        plot_fig_4_26(data),
        plot_fig_4_27(data),
        plot_fig_4_28(data),
        plot_combined(data),
    ]
    for p in paths:
        if p:
            print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
