"""Clean multi-panel overlay of the K0_R4e factor sweep at λ=1, Δβ=0.

Design: highlight 3-4 key K0 values prominently (v10b production,
the corrected deck-fit, and informative extremes); fade other points
to thin gray background for context.  Compares vs the deck K@pH4
4-point ensemble (high variance) and the slide-15 Cs⁺ pH 4 envelope.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
N_COLLECTION = 0.224
DECK_MEAN_H2O2_PCT = 50.95
DECK_STD_H2O2_PCT = 27.47


# K0 sweep manifest.  ``style`` controls highlighting:
#   "hero": thick colored curve in legend (key K0 values)
#   "bg":   faded thin gray curve, no legend entry (context only)
RUNS: Tuple[Tuple[str, str, float, str, Optional[str]], ...] = (
    # (label, path, k0, style, color)
    ("1e-18", "StudyResults/phase6b_k0_r4e_sweep_18/eval_k0r4e_1p00eN18_stern.json",     1e-18,    "hero", "#1f78b4"),  # blue
    ("2.52e-18", "StudyResults/phase6b_k0_r4e_sweep_2p52e18/eval_k0r4e_2p52eN18_stern.json", 2.52e-18, "bg",   None),
    ("3e-18", "StudyResults/phase6b_k0_r4e_sweep_3e18/eval_k0r4e_3p00eN18_stern.json",  3e-18,    "bg",   None),
    ("1e-17", "StudyResults/phase6b_k0_r4e_sweep_1e17/eval_k0r4e_1p00eN17_stern.json",  1e-17,    "bg",   None),
    ("3e-17", "StudyResults/phase6b_k0_r4e_sweep_3e17/eval_k0r4e_3p00eN17_stern.json",  3e-17,    "bg",   None),
    ("1e-16", "StudyResults/phase6b_k0_r4e_sweep_16/eval_k0r4e_1p00eN16_stern.json",     1e-16,    "bg",   None),
    ("1e-15", "StudyResults/phase6b_k0_r4e_sweep_1e15/eval_k0r4e_1p00eN15_stern.json",  1e-15,    "hero", "#33a02c"),  # green (deck fit)
    ("1e-14", "StudyResults/phase6b_lambda_sweep_diag_lam1/eval_lambda_1p0000_stern.json", 1e-14, "hero", "#ff7f00"),  # orange (v10b)
    ("1e-12", "StudyResults/phase6b_k0_r4e_sweep_12/eval_k0r4e_1p00eN12_stern.json",     1e-12,    "bg",   None),
    ("1e-10", "StudyResults/phase6b_k0_r4e_sweep_10/eval_k0r4e_1p00eN10_stern.json",     1e-10,    "hero", "#e31a1c"),  # red (4e-dominant)
)


def gross_h2o2_pct(R_2e: float, R_4e: float) -> Optional[float]:
    if R_2e is None or R_4e is None:
        return None
    if (isinstance(R_2e, float) and math.isnan(R_2e)) or (
        isinstance(R_4e, float) and math.isnan(R_4e)
    ):
        return None
    r2, r4 = float(R_2e), float(R_4e)
    if r2 < 0 or r4 < 0:
        return None
    if abs(r2) + abs(r4) < 1e-6:
        return None
    return 100.0 * r2 / (r2 + r4)


def extract_curves(eval_json: Dict[str, Any]) -> Dict[str, np.ndarray]:
    recs = eval_json.get("per_v_records", [])
    recs = [r for r in recs if r.get("v_rhe") is not None]
    recs.sort(key=lambda r: float(r["v_rhe"]))
    v = np.array([float(r["v_rhe"]) for r in recs])
    cd = np.array([
        float(r["cd_mA_cm2"]) if r.get("cd_mA_cm2") is not None else np.nan
        for r in recs
    ])
    pc = np.array([
        float(r["pc_mA_cm2"]) if r.get("pc_mA_cm2") is not None else np.nan
        for r in recs
    ])
    sel_raw = [
        gross_h2o2_pct(r.get("R_2e_current_nondim"), r.get("R_4e_current_nondim"))
        for r in recs
    ]
    sel = np.array([np.nan if x is None else x for x in sel_raw], dtype=float)
    theta = np.array([
        float(r["theta"]) if r.get("theta") is not None else np.nan
        for r in recs
    ])
    return {"v": v, "cd": cd, "pc": pc, "sel": sel, "theta": theta}


def load_eval(path: str) -> Optional[Dict[str, Any]]:
    full = path if os.path.isabs(path) else os.path.join(_ROOT, path)
    if not os.path.exists(full):
        return None
    try:
        with open(full, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def main() -> int:
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    runs: List[Tuple[str, float, str, Optional[str], Dict[str, np.ndarray]]] = []
    for label, path, k0, style, color in RUNS:
        j = load_eval(path)
        if j is None:
            print(f"[skip] {path}")
            continue
        runs.append((label, k0, style, color, extract_curves(j)))

    if not runs:
        print("no runs loaded")
        return 1

    # === FIGURE 1: 2x2 panel — peroxide current, total current, gross H2O2%, theta ===
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(
        "K0_R4e factor sweep at λ=1, Δβ=0, K₂SO₄ pH 4 v10b stack "
        "(Phase 6β step-10 follow-up, 2026-05-21)",
        fontsize=13, fontweight="bold",
    )
    ax_pc, ax_cd = axes[0]
    ax_sel, ax_th = axes[1]

    # Background runs first (faded gray, no legend)
    for label, k0, style, color, c in runs:
        if style == "bg":
            ax_pc.plot(c["v"], c["pc"], color="lightgray", lw=1.0, alpha=0.7, zorder=1)
            ax_cd.plot(c["v"], c["cd"], color="lightgray", lw=1.0, alpha=0.7, zorder=1)
            ax_sel.plot(c["v"], c["sel"], color="lightgray", lw=1.0, alpha=0.7, zorder=1)
            ax_th.plot(c["v"], c["theta"], color="lightgray", lw=1.0, alpha=0.7, zorder=1)

    # Hero runs on top
    for label, k0, style, color, c in runs:
        if style != "hero":
            continue
        # Friendly label with role
        if abs(k0 - 1e-14) < 1e-30:
            role = "v10b production"
        elif abs(k0 - 1e-15) < 1e-30:
            role = "deck-fit (corrected)"
        elif abs(k0 - 1e-18) < 1e-30:
            role = "low extreme (~100% H₂O₂)"
        elif abs(k0 - 1e-10) < 1e-30:
            role = "high extreme (~0% H₂O₂)"
        else:
            role = ""
        lab = f"K0={label}  ({role})"
        ax_pc.plot(c["v"], c["pc"], "-o", color=color, lw=2.5, ms=5, label=lab, zorder=3)
        ax_cd.plot(c["v"], c["cd"], "-o", color=color, lw=2.5, ms=5, label=lab, zorder=3)
        ax_sel.plot(c["v"], c["sel"], "-o", color=color, lw=2.5, ms=5, label=lab, zorder=3)
        ax_th.plot(c["v"], c["theta"], "-o", color=color, lw=2.5, ms=5, label=lab, zorder=3)

    # Slide-15 deck envelope on pc panel
    v_slide = np.array([-0.45, -0.30, -0.15, 0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 1.0])
    pc_slide = np.array([-0.18, -0.20, -0.30, -0.45, -0.55, -0.50, -0.30, -0.10, -0.02, 0.0, 0.0])
    ax_pc.plot(
        v_slide, pc_slide, "--", color="black", lw=2.0, alpha=0.8,
        label="deck slide-15 envelope (Cs⁺ pH 4, eyeballed)",
        zorder=4,
    )

    # Deck mean + std band on selectivity panel
    ax_sel.axhspan(
        DECK_MEAN_H2O2_PCT - DECK_STD_H2O2_PCT,
        DECK_MEAN_H2O2_PCT + DECK_STD_H2O2_PCT,
        color="red", alpha=0.10, zorder=0,
    )
    ax_sel.axhline(
        DECK_MEAN_H2O2_PCT, color="red", lw=2.0, ls="--", zorder=2,
        label=f"deck K@pH4 mean ({DECK_MEAN_H2O2_PCT:.1f}±{DECK_STD_H2O2_PCT:.0f} pp, n=4)",
    )
    # Deck individual measurements
    deck_pts = [
        (0.346, 54.8),  # 2020-03-19 pH 3.99
        (0.413, 88.0),  # 2019-08-15 pH 4.21
        (0.370, 35.0),  # 2020-06-15 pH 4.03
        (0.314, 26.0),  # 2020-09-29 pH 4.02
    ]
    for i, (v, h) in enumerate(deck_pts):
        ax_sel.scatter(
            [v], [h], s=200, marker="X", color="red",
            edgecolor="black", zorder=6,
            linewidths=1.5,
            label="deck K@pH4 individual measurements (n=4)" if i == 0 else None,
        )

    # Axis configuration
    ax_pc.set_xlabel(r"$V_{\mathrm{RHE}}$ (V)")
    ax_pc.set_ylabel("pc — peroxide partial current (mA/cm²)")
    ax_pc.set_title("(a) Peroxide partial current  vs  deck slide-15 envelope")
    ax_pc.grid(True, alpha=0.3)
    ax_pc.set_xlim(-0.15, 1.0)
    ax_pc.axhline(0, color="black", lw=0.5)
    ax_pc.legend(loc="lower right", framealpha=0.95)

    ax_cd.set_xlabel(r"$V_{\mathrm{RHE}}$ (V)")
    ax_cd.set_ylabel("cd — total disk current (mA/cm²)")
    ax_cd.set_title("(b) Total disk current (cathodic = negative)")
    ax_cd.grid(True, alpha=0.3)
    ax_cd.set_xlim(-0.15, 1.0)
    ax_cd.axhline(0, color="black", lw=0.5)
    ax_cd.legend(loc="lower right", framealpha=0.95)

    ax_sel.set_xlabel(r"$V_{\mathrm{RHE}}$ (V)")
    ax_sel.set_ylabel(r"gross H$_2$O$_2$% = 100·R$_{2e}$/(R$_{2e}$+R$_{4e}$)")
    ax_sel.set_title("(c) Per-V H₂O₂ selectivity (deck convention)  vs  deck K@pH4 points")
    ax_sel.grid(True, alpha=0.3)
    ax_sel.set_xlim(-0.15, 1.0)
    ax_sel.set_ylim(0, 100)
    ax_sel.legend(loc="upper right", framealpha=0.95)

    ax_th.set_xlabel(r"$V_{\mathrm{RHE}}$ (V)")
    ax_th.set_ylabel(r"$\theta = \Gamma / \Gamma_{\max}$")
    ax_th.set_title("(d) MOH coverage at OHP (hydrolysis saturation)")
    ax_th.grid(True, alpha=0.3)
    ax_th.set_xlim(-0.15, 1.0)
    ax_th.set_ylim(0, 1)
    ax_th.legend(loc="upper right", framealpha=0.95)

    # Annotation explaining background lines once
    fig.text(
        0.5, 0.01,
        "gray background curves: K0 ∈ {2.52e-18, 3e-18, 1e-17, 3e-17, 1e-16, 1e-12} "
        "(intermediate sweep points, shown for context)",
        ha="center", fontsize=9, style="italic", color="dimgray",
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    out_dir = Path(_ROOT) / "writeups" / "HydrolysisMeetingPrep"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "k0_r4e_sweep_curves.png"
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"saved {out_path}")
    plt.close(fig)

    # === FIGURE 2: the loss curve — max gross H2O2% vs K0 ===
    fig2, ax2 = plt.subplots(figsize=(11, 6.5))
    log_k0 = []
    max_sel = []
    for label, k0, style, color, c in runs:
        log_k0.append(math.log10(k0))
        # Restrict to deck mask [-0.06, +1.0]
        mask = (c["v"] >= -0.06) & (c["v"] <= 1.0)
        sel_in_mask = c["sel"][mask]
        sel_in_mask = sel_in_mask[~np.isnan(sel_in_mask)]
        max_sel.append(float(sel_in_mask.max()) if sel_in_mask.size else np.nan)
    log_k0 = np.array(log_k0)
    max_sel = np.array(max_sel)
    sort_idx = np.argsort(log_k0)

    ax2.plot(
        10**log_k0[sort_idx], max_sel[sort_idx],
        "o-", color="#1f4e79", lw=2.5, ms=10,
        label="model: max gross H₂O₂%  in deck mask",
        zorder=3,
    )

    # Mark and annotate key K0 values
    for label, k0, style, color, c in runs:
        if style != "hero":
            continue
        mask = (c["v"] >= -0.06) & (c["v"] <= 1.0)
        sel_in_mask = c["sel"][mask]
        sel_in_mask = sel_in_mask[~np.isnan(sel_in_mask)]
        if sel_in_mask.size == 0:
            continue
        mh = float(sel_in_mask.max())
        if abs(k0 - 1e-14) < 1e-30:
            ax2.scatter(
                [k0], [mh], s=400, marker="*", color=color,
                edgecolor="black", linewidth=1.5, zorder=5,
                label=f"v10b production (K0=10⁻¹⁴ → {mh:.1f}%)",
            )
        elif abs(k0 - 1e-15) < 1e-30:
            ax2.scatter(
                [k0], [mh], s=350, marker="D", color=color,
                edgecolor="black", linewidth=1.5, zorder=5,
                label=f"corrected deck-fit (K0=10⁻¹⁵ → {mh:.1f}%)",
            )

    # Deck mean + std band
    ax2.axhspan(
        DECK_MEAN_H2O2_PCT - DECK_STD_H2O2_PCT,
        DECK_MEAN_H2O2_PCT + DECK_STD_H2O2_PCT,
        color="red", alpha=0.10, zorder=0,
        label=f"deck K@pH4 mean ± std ({DECK_MEAN_H2O2_PCT:.1f}±{DECK_STD_H2O2_PCT:.0f} pp)",
    )
    ax2.axhline(
        DECK_MEAN_H2O2_PCT, color="red", lw=2.0, ls="--", zorder=2,
    )
    ax2.axhline(
        38.9, color="gray", lw=1.0, ls=":", alpha=0.7, zorder=1,
        label="hydrolysis-OFF baseline (38.9 pp)",
    )

    # Annotate the previous misleading "Phase D fit"
    ax2.annotate(
        "Phase D 'fit' K0=2.5×10⁻¹⁸\n"
        "(used DEPRECATED signed-pc;\n"
        "  actually gives 99.8% gross H₂O₂% —\n"
        "  NOT a deck match)",
        xy=(2.5e-18, 99.6), xytext=(7e-18, 75),
        fontsize=9, color="darkred",
        arrowprops=dict(arrowstyle="->", color="darkred", alpha=0.7),
        bbox=dict(facecolor="white", edgecolor="darkred", alpha=0.85),
    )

    ax2.set_xscale("log")
    ax2.set_xlabel(r"$K_0^{R_{4e}}$ factor (multiplier on $K_0^{R_{4e}}$ baseline)")
    ax2.set_ylabel("max gross H₂O₂% in deck mask [-0.06, +1.0] V")
    ax2.set_title(
        r"Max gross H₂O₂% = 100·R$_{2e}$/(R$_{2e}$+R$_{4e}$) vs K0_R4e factor"
        "\n"
        r"(λ=1, Δβ=0, physical $a_{\mathrm{nondim}}$, K₂SO₄ pH 4, v10b stack)",
        fontsize=12,
    )
    ax2.legend(loc="upper right", framealpha=0.95, fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(3e-19, 3e-10)
    ax2.set_ylim(-5, 110)
    plt.tight_layout()
    out_path2 = out_dir / "k0_r4e_sweep_loss.png"
    plt.savefig(out_path2, dpi=140, bbox_inches="tight")
    print(f"saved {out_path2}")
    plt.close(fig2)

    # CSV dump
    csv_path = out_dir / "k0_r4e_sweep_curves.csv"
    with open(csv_path, "w") as f:
        header = ["v_rhe"]
        for label, _, _, _, _ in runs:
            for field in ("pc", "cd", "sel", "theta"):
                header.append(f"K0_{label}_{field}")
        f.write(",".join(header) + "\n")
        ref_v = runs[0][4]["v"]
        for i in range(len(ref_v)):
            row = [f"{ref_v[i]:+.4f}"]
            for _, _, _, _, c in runs:
                for field in ("pc", "cd", "sel", "theta"):
                    val = c[field][i] if i < len(c[field]) else float("nan")
                    row.append(f"{val:.6g}")
            f.write(",".join(row) + "\n")
    print(f"saved {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
