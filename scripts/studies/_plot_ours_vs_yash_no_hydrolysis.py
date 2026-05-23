"""Overlay our forward solver (Yash-matched params, no hydrolysis) vs Yash.

Reads:
  - StudyResults/yash_match_no_hydrolysis_25pt/factor_*/iv_curve.json
      (our model, 25 V_RHE points, L_eff=6 um, C_S=1.38 F/m^2,
       Cs+/SO4(2-), pH 4, water+cation hydrolysis OFF)
  - data/derived/yash_best_fit_base_lsv.csv
      (Yash's 200-point LSV from his sim_*.npy snapshots)

Writes:
  - StudyResults/yash_match_no_hydrolysis_25pt/overlay_ours_vs_yash.{png,pdf}

No experimental data is plotted (intentional - model-vs-model only).
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent.parent
RESULTS_DIR = _ROOT / "StudyResults" / "yash_match_no_hydrolysis_25pt"
YASH_CSV = _ROOT / "data" / "derived" / "yash_best_fit_base_lsv.csv"


def _discover_factors(results_dir: Path) -> list[float]:
    pat = re.compile(r"^factor_(?P<f>[^/]+)$")
    factors: list[float] = []
    for child in sorted(results_dir.glob("factor_*")):
        if not child.is_dir():
            continue
        if not (child / "iv_curve.json").exists():
            continue
        m = pat.match(child.name)
        if m is None:
            continue
        try:
            factors.append(float(m.group("f")))
        except ValueError:
            continue
    factors.sort(reverse=True)
    return factors


def _load_ours(factor: float) -> dict | None:
    p = RESULTS_DIR / f"factor_{factor:g}" / "iv_curve.json"
    if not p.exists():
        print(f"[plot] WARN missing {p}", file=sys.stderr)
        return None
    with open(p) as f:
        return json.load(f)


def _to_arr(values: list) -> np.ndarray:
    return np.array(
        [np.nan if v is None else float(v) for v in values], dtype=float,
    )


def _load_yash(csv_path: Path) -> dict:
    v: list[float] = []
    j_tot: list[float] = []
    j_h2o2: list[float] = []
    sel: list[float] = []
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("V_RHE"):
                continue
            cols = line.split(",")
            try:
                v.append(float(cols[0]))
                j_tot.append(float(cols[2]))
                j_h2o2.append(float(cols[3]))
                sel.append(float(cols[4]))
            except (ValueError, IndexError):
                continue
    return {
        "V_RHE": np.array(v),
        "j_total": np.array(j_tot),
        "j_H2O2": np.array(j_h2o2),
        "selectivity_pct": np.array(sel),
    }


def main() -> int:
    if not RESULTS_DIR.exists():
        print(f"[plot] missing {RESULTS_DIR}", file=sys.stderr)
        return 1

    factors = _discover_factors(RESULTS_DIR)
    ours: dict[float, dict] = {}
    for f in factors:
        rep = _load_ours(f)
        if rep is not None:
            ours[f] = rep
    if not ours:
        print("[plot] no factor json files loaded", file=sys.stderr)
        return 1
    print(f"[plot] found {len(ours)} factor curves: "
          f"{', '.join(f'{f:g}' for f in factors)}")

    yash = _load_yash(YASH_CSV) if YASH_CSV.exists() else None
    if yash is None:
        print(f"[plot] WARN missing {YASH_CSV} (Yash overlay disabled)",
              file=sys.stderr)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=False)
    ax_cd, ax_pc = axes
    cmap = plt.get_cmap("viridis")
    n = len(ours)
    colors = [cmap(t) for t in np.linspace(0.05, 0.88, n)]

    for color, factor in zip(colors, sorted(ours.keys(), reverse=True)):
        rep = ours[factor]
        v = np.array(rep["v_rhe"], dtype=float)
        cd = _to_arr(rep["cd_mA_cm2"])
        pc = _to_arr(rep["pc_mA_cm2"])
        n_ok = int(rep.get("n_converged", 0))
        n_tot = int(rep.get("n_total", len(v)))
        label = f"ours $k_0^{{R4e}}/k_0^{{R2e}}={factor:g}$  ({n_ok}/{n_tot})"
        ax_cd.plot(v, cd, marker="o", color=color, lw=1.6, ms=4, label=label)
        ax_pc.plot(v, pc, marker="o", color=color, lw=1.6, ms=4, label=label)

    if yash is not None:
        # Filter Yash selectivity noise: keep only rows where |j_total| > 0.005 mA/cm^2
        # (selectivity divides by j_total; near-zero j makes selectivity nonsense).
        mask = np.abs(yash["j_total"]) > 0.005
        ax_cd.plot(
            yash["V_RHE"], yash["j_total"],
            color="black", lw=2.0, ls="-", marker="x", ms=4, mew=1.0,
            label="Yash best_fit_base (j_disk)", zorder=10,
        )
        ax_pc.plot(
            yash["V_RHE"][mask], yash["j_H2O2"][mask],
            color="black", lw=2.0, ls="-", marker="x", ms=4, mew=1.0,
            label="Yash best_fit_base (j_H2O2 via selectivity)", zorder=10,
        )

    for ax in axes:
        ax.axhline(0.0, color="black", lw=0.6, ls="-", alpha=0.4)
        ax.set_xlabel(r"$V_{\mathrm{RHE}}$ (V)")
        ax.grid(True, alpha=0.3)
    ax_cd.set_ylabel(r"$j_{\mathrm{disk}}$ (mA/cm$^2$)")
    ax_cd.set_title(r"Total disk current vs $V_{\mathrm{RHE}}$")
    ax_pc.set_ylabel(r"$j_{\mathrm{H_2O_2}}$ (mA/cm$^2$)")
    ax_pc.set_title(r"Gross H$_2$O$_2$ current vs $V_{\mathrm{RHE}}$")
    ax_cd.legend(loc="lower right", fontsize=8, framealpha=0.85)
    ax_pc.legend(loc="lower right", fontsize=8, framealpha=0.85)

    fig.suptitle(
        "Our solver (Yash-matched params) vs Yash best_fit_base  |  "
        "Cs$^+$/SO$_4^{2-}$, pH 4, $L_{\\mathrm{eff}}$=6 $\\mu$m, "
        "$C_S$=1.38 F/m$^2$, no hydrolysis",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    out_png = RESULTS_DIR / "overlay_ours_vs_yash.png"
    out_pdf = RESULTS_DIR / "overlay_ours_vs_yash.pdf"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")

    # Side-table: cd/pc at min/max V_RHE for each model
    print("\n--- our-model summary ---")
    print(f"{'factor':>10s}  {'V_min':>7s}  {'cd@Vmin':>10s}  "
          f"{'pc@Vmin':>10s}  {'V_max':>7s}  {'cd@Vmax':>10s}  {'pc@Vmax':>10s}")
    for factor in sorted(ours.keys(), reverse=True):
        rep = ours[factor]
        v = np.array(rep["v_rhe"], dtype=float)
        cd = _to_arr(rep["cd_mA_cm2"])
        pc = _to_arr(rep["pc_mA_cm2"])
        i_lo, i_hi = int(np.argmin(v)), int(np.argmax(v))
        print(
            f"{factor:10.0e}  {v[i_lo]:+7.3f}  {cd[i_lo]:+10.4f}  "
            f"{pc[i_lo]:+10.4f}  {v[i_hi]:+7.3f}  {cd[i_hi]:+10.4f}  "
            f"{pc[i_hi]:+10.4f}"
        )

    if yash is not None:
        print("\n--- Yash best_fit_base summary (V_RHE band of our grid) ---")
        m = (yash["V_RHE"] >= -0.40) & (yash["V_RHE"] <= +0.55)
        if m.any():
            v_in, j_in, jh_in = (
                yash["V_RHE"][m], yash["j_total"][m], yash["j_H2O2"][m],
            )
            print(f"  n={m.sum()} pts in [-0.40, +0.55] V")
            print(f"  V_RHE  range: [{v_in.min():+.3f}, {v_in.max():+.3f}] V")
            print(f"  j_disk range: [{j_in.min():+.4f}, {j_in.max():+.4f}] mA/cm^2")
            print(f"  j_H2O2 range: [{jh_in.min():+.4f}, {jh_in.max():+.4f}] mA/cm^2")

    return 0


if __name__ == "__main__":
    sys.exit(main())
