"""Overlay hydrolysis-on (best K0 candidates) + no-hydrolysis + Yash.

Reads:
  - StudyResults/yash_match_hydrolysis_on/factor_*/iv_curve.json
      (our model, hydrolysis ON, lambda=1, Cs+/SO4, L_eff=6 um,
       C_S baseline=0.20 F/m^2 - PRODUCTION not Yash-matched 1.38)
  - StudyResults/yash_match_no_hydrolysis_25pt/factor_*/iv_curve.json
      (our model, no hydrolysis, Cs+/SO4, L_eff=6 um, C_S=1.38 F/m^2)
  - data/derived/yash_best_fit_base_lsv.csv

Writes:
  - StudyResults/yash_match_hydrolysis_on/overlay_ours_vs_yash.{png,pdf}

No experimental data on this plot (model-vs-model only).
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
HYDROL_DIR = _ROOT / "StudyResults" / "yash_match_hydrolysis_on"
NO_HYDROL_DIR = _ROOT / "StudyResults" / "yash_match_no_hydrolysis_25pt"
YASH_CSV = _ROOT / "data" / "derived" / "yash_best_fit_base_lsv.csv"


def _parse_factor_label(name: str) -> float | None:
    # Two forms: "factor_1e-17" (no_hydrol naming)
    #            "factor_1.00en17" (hydrol_on naming)
    m = re.match(r"^factor_(.+)$", name)
    if m is None:
        return None
    raw = m.group(1).replace("n", "-").replace("p", "+")
    try:
        return float(raw)
    except ValueError:
        return None


def _discover_factors(results_dir: Path) -> list[tuple[float, Path]]:
    if not results_dir.exists():
        return []
    out: list[tuple[float, Path]] = []
    for child in sorted(results_dir.glob("factor_*")):
        if not child.is_dir():
            continue
        ivp = child / "iv_curve.json"
        if not ivp.exists():
            continue
        f = _parse_factor_label(child.name)
        if f is None:
            continue
        out.append((f, ivp))
    out.sort(key=lambda x: x[0], reverse=True)
    return out


def _load_no_hydrol(path: Path) -> dict:
    d = json.load(open(path))
    return {
        "v": np.array(d["v_rhe"], dtype=float),
        "cd": np.array([np.nan if x is None else float(x) for x in d["cd_mA_cm2"]]),
        "pc": np.array([np.nan if x is None else float(x) for x in d["pc_mA_cm2"]]),
        "n_converged": int(d.get("n_converged", 0)),
        "n_total": int(d.get("n_total", 0)),
    }


def _load_hydrol_on(path: Path) -> dict:
    d = json.load(open(path))
    recs = d.get("per_v_records", [])
    v = np.array([r["v_rhe"] for r in recs], dtype=float)
    cd = np.array([
        np.nan if r.get("cd_mA_cm2") is None else float(r["cd_mA_cm2"])
        for r in recs
    ])
    # pc_gross_mA_cm2 is GROSS H2O2 current (slide-15 convention, -I_SCALE * R_2e)
    pc = np.array([
        np.nan if r.get("pc_gross_mA_cm2") is None else float(r["pc_gross_mA_cm2"])
        for r in recs
    ])
    n_ok = sum(1 for r in recs if r.get("snes_converged"))
    return {
        "v": v, "cd": cd, "pc": pc,
        "n_converged": n_ok, "n_total": len(recs),
    }


def _load_yash(csv_path: Path) -> dict | None:
    if not csv_path.exists():
        return None
    v, jt, jh, sel = [], [], [], []
    for line in open(csv_path):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("V_RHE"):
            continue
        cols = line.split(",")
        try:
            v.append(float(cols[0]))
            jt.append(float(cols[2]))
            jh.append(float(cols[3]))
            sel.append(float(cols[4]))
        except (ValueError, IndexError):
            continue
    return {
        "V_RHE": np.array(v),
        "j_total": np.array(jt),
        "j_H2O2": np.array(jh),
        "selectivity_pct": np.array(sel),
    }


def main() -> int:
    hyd_factors = _discover_factors(HYDROL_DIR)
    noh_factors = _discover_factors(NO_HYDROL_DIR)
    yash = _load_yash(YASH_CSV)

    if not hyd_factors:
        print(f"[plot] no hydrolysis-on factor curves found in {HYDROL_DIR}",
              file=sys.stderr)
        # still emit plot if we have at least no-hydrolysis curves

    hyd_curves: list[tuple[float, dict]] = [
        (f, _load_hydrol_on(p)) for f, p in hyd_factors
    ]
    noh_curves: list[tuple[float, dict]] = [
        (f, _load_no_hydrol(p)) for f, p in noh_factors
    ]
    print(f"[plot] hydrol-on   factors: "
          f"{', '.join(f'{f:g}' for f, _ in hyd_curves) or '(none)'}")
    print(f"[plot] no-hydrol  factors: "
          f"{', '.join(f'{f:g}' for f, _ in noh_curves) or '(none)'}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False)
    ax_cd_full, ax_pc_full = axes[0]
    ax_cd_zoom, ax_pc_zoom = axes[1]

    # Hydrolysis-ON (vivid) palette
    hyd_cmap = plt.get_cmap("plasma")
    n_h = max(1, len(hyd_curves))
    hyd_colors = [hyd_cmap(t) for t in np.linspace(0.10, 0.85, n_h)]

    for color, (factor, c) in zip(hyd_colors, hyd_curves):
        lbl = f"hydrol-ON $k_0^{{R4e}}/k_0^{{R2e}}={factor:g}$  ({c['n_converged']}/{c['n_total']})"
        for ax in (ax_cd_full, ax_cd_zoom):
            ax.plot(c["v"], c["cd"], marker="o", color=color, lw=2.0, ms=5, label=lbl, zorder=5)
        for ax in (ax_pc_full, ax_pc_zoom):
            ax.plot(c["v"], c["pc"], marker="o", color=color, lw=2.0, ms=5, label=lbl, zorder=5)

    # No-hydrolysis (light) palette - only show 4 representative for clarity
    interesting_noh = [1.0, 1e-12, 1e-15, 1e-18]
    noh_filtered = [(f, c) for f, c in noh_curves if any(abs(f-x) < 1e-6*abs(x)+1e-30 for x in interesting_noh)]
    if not noh_filtered:
        noh_filtered = noh_curves  # fall back to all
    noh_cmap = plt.get_cmap("viridis")
    n_n = max(1, len(noh_filtered))
    noh_colors = [noh_cmap(t) for t in np.linspace(0.10, 0.85, n_n)]

    for color, (factor, c) in zip(noh_colors, noh_filtered):
        lbl = f"no-hydrol $k_0^{{R4e}}/k_0^{{R2e}}={factor:g}$"
        for ax in (ax_cd_full, ax_cd_zoom):
            ax.plot(c["v"], c["cd"], color=color, lw=1.2, ls="--", alpha=0.55, label=lbl, zorder=2)
        for ax in (ax_pc_full, ax_pc_zoom):
            ax.plot(c["v"], c["pc"], color=color, lw=1.2, ls="--", alpha=0.55, label=lbl, zorder=2)

    if yash is not None:
        mask = np.abs(yash["j_total"]) > 0.005
        for ax in (ax_cd_full, ax_cd_zoom):
            ax.plot(yash["V_RHE"], yash["j_total"], color="black", lw=2.5, marker="x",
                  ms=4, mew=1.0, label="Yash best_fit_base (j_disk)", zorder=10)
        for ax in (ax_pc_full, ax_pc_zoom):
            ax.plot(yash["V_RHE"][mask], yash["j_H2O2"][mask], color="black", lw=2.5,
                  marker="x", ms=4, mew=1.0,
                  label="Yash best_fit_base (j_H2O2)", zorder=10)

    for ax in axes.ravel():
        ax.axhline(0.0, color="black", lw=0.6, ls="-", alpha=0.4)
        ax.set_xlabel(r"$V_{\mathrm{RHE}}$ (V)")
        ax.grid(True, alpha=0.3)

    # Full-range top row
    ax_cd_full.set_ylabel(r"$j_{\mathrm{disk}}$ (mA/cm$^2$)")
    ax_cd_full.set_title(r"$j_{\mathrm{disk}}$ — full range")
    ax_pc_full.set_ylabel(r"$j_{\mathrm{H_2O_2}}$ (mA/cm$^2$)")
    ax_pc_full.set_title(r"$j_{\mathrm{H_2O_2}}$ — full range")
    ax_cd_full.legend(loc="lower right", fontsize=7.0, framealpha=0.85)
    ax_pc_full.legend(loc="lower right", fontsize=7.0, framealpha=0.85)

    # Zoom to Yash's range
    ax_cd_zoom.set_ylim(-2.0, 0.10)
    ax_pc_zoom.set_ylim(-2.0, 0.10)
    ax_cd_zoom.set_ylabel(r"$j_{\mathrm{disk}}$ (mA/cm$^2$)")
    ax_cd_zoom.set_title(r"$j_{\mathrm{disk}}$ — zoomed to Yash range ($\pm$ 2 mA/cm$^2$)")
    ax_pc_zoom.set_ylabel(r"$j_{\mathrm{H_2O_2}}$ (mA/cm$^2$)")
    ax_pc_zoom.set_title(r"$j_{\mathrm{H_2O_2}}$ — zoomed to Yash range")

    fig.suptitle(
        "Our solver (hydrolysis ON + OFF) vs Yash best_fit_base  |  "
        "Cs$^+$/SO$_4^{2-}$, pH 4, $L_{\\mathrm{eff}}$=6 $\\mu$m\n"
        "no-hydrol at $C_S$=1.38 F/m$^2$; hydrol-ON at $C_S$=0.20 F/m$^2$",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out_dir = HYDROL_DIR if HYDROL_DIR.exists() else NO_HYDROL_DIR
    out_png = out_dir / "overlay_ours_vs_yash.png"
    out_pdf = out_dir / "overlay_ours_vs_yash.pdf"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")

    # Summary table
    print("\n--- hydrolysis-ON summary ---")
    print(f"{'factor':>10s}  {'n_ok':>6s}  {'cd@Vmin':>9s}  {'cd@Vmax':>9s}  "
          f"{'pc@Vmin':>9s}  {'pc@Vmax':>9s}  {'max_pc':>9s}")
    for factor, c in hyd_curves:
        i_lo, i_hi = int(np.argmin(c["v"])), int(np.argmax(c["v"]))
        pc_finite = c["pc"][np.isfinite(c["pc"])]
        max_pc = float(np.min(pc_finite)) if len(pc_finite) else float("nan")
        print(f"{factor:10.0e}  {c['n_converged']:>3d}/{c['n_total']:<3d}  "
              f"{c['cd'][i_lo]:+9.4f}  {c['cd'][i_hi]:+9.4f}  "
              f"{c['pc'][i_lo]:+9.4f}  {c['pc'][i_hi]:+9.4f}  {max_pc:+9.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
