"""Plotter for solver_demo_slide15_no_speculative_cs.

Reads each ``factor_*/iv_curve.json`` in the chosen results directory
and emits a 2-panel figure: total disk current density (cd) and gross
H2O2 current (pc) vs V_RHE.  One curve per K0_R4e factor.

Usage::

    cd PNPInverse
    source ../venv-firedrake/bin/activate

    # default: Stern results dir, auto-discover factors
    python -u scripts/studies/plot_solver_demo_slide15_no_speculative_cs.py

    # no-Stern parallel plot:
    python -u scripts/studies/plot_solver_demo_slide15_no_speculative_cs.py \\
        --results-dir StudyResults/solver_demo_slide15_no_speculative_cs_noStern \\
        --title-suffix " (C_S -> infinity, no Stern)"

    # explicit factor list:
    python -u scripts/studies/plot_solver_demo_slide15_no_speculative_cs.py \\
        --factors=1,1e-6,1e-12,1e-13,1e-14,1e-15,1e-16,1e-17,1e-18
"""
from __future__ import annotations

import argparse
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

DEFAULT_RESULTS_DIR = (
    _ROOT / "StudyResults" / "solver_demo_slide15_no_speculative_cs"
)

EXPERIMENTAL_CSV = _ROOT / "data" / "mangan_deck_p15_h2o2_current.csv"


def _load_experimental_csv(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    v_list: list[float] = []
    j_list: list[float] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("V_RHE"):
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            try:
                v_list.append(float(parts[0]))
                j_list.append(float(parts[1]))
            except ValueError:
                continue
    if not v_list:
        return None
    return np.array(v_list), np.array(j_list)


def _factor_label(factor: float) -> str:
    return f"factor_{factor:g}"


def _parse_factor_list(arg: str) -> list[float]:
    out: list[float] = []
    for tok in arg.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    if not out:
        raise ValueError(f"--factors must be non-empty (got {arg!r})")
    return out


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
    # Sort descending so factor=1 (highest) plots first.
    factors.sort(reverse=True)
    return factors


def _load_one(results_dir: Path, factor: float) -> dict | None:
    p = results_dir / _factor_label(factor) / "iv_curve.json"
    if not p.exists():
        print(f"[plot] WARN missing {p}", file=sys.stderr)
        return None
    with open(p) as f:
        return json.load(f)


def _to_arr(values: list) -> np.ndarray:
    return np.array(
        [np.nan if v is None else float(v) for v in values], dtype=float,
    )


def _factor_legend_label(factor: float) -> str:
    return f"$k_0^{{R4e}}/k_0^{{R2e}} = {factor:g}$"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot the solver demo I-V curves across K0_R4e factors.",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
        help=(
            "Directory containing factor_*/iv_curve.json. "
            f"Default: {DEFAULT_RESULTS_DIR}."
        ),
    )
    parser.add_argument(
        "--factors", type=_parse_factor_list, default=None,
        help=(
            "Comma-separated factor list to plot. "
            "When omitted, auto-discovers factor_*/iv_curve.json in results-dir."
        ),
    )
    parser.add_argument(
        "--title-suffix", type=str, default="",
        help=(
            "Optional string appended to the figure title (e.g. for the "
            "no-Stern variant)."
        ),
    )
    parser.add_argument(
        "--out-stem", type=str, default="iv_curves",
        help="Filename stem for the output PNG/PDF (without extension).",
    )
    parser.add_argument(
        "--overlay-experimental", action="store_true",
        help=(
            "Overlay the Mangan deck page-15 H2O2 current density "
            "(Cs+/pH 4, 37-pt digitisation) on the gross-H2O2 panel."
        ),
    )
    args = parser.parse_args()

    results_dir: Path = args.results_dir
    if not results_dir.exists():
        print(f"[plot] no results dir at {results_dir}", file=sys.stderr)
        return 1

    if args.factors is None:
        factors = _discover_factors(results_dir)
    else:
        factors = sorted(args.factors, reverse=True)
    if not factors:
        print(f"[plot] no factors found / supplied at {results_dir}",
              file=sys.stderr)
        return 1

    reports: dict[float, dict] = {}
    for f in factors:
        rep = _load_one(results_dir, f)
        if rep is not None:
            reports[f] = rep

    if not reports:
        print("[plot] no reports loaded; nothing to plot", file=sys.stderr)
        return 1

    # H+ diffusion Levich limit at L_eff=100 um, pH 4 (c_H = 0.1 mol/m^3):
    # i_lim_H = F * D_H * c_H / L_eff  (A/m^2);  /10 -> mA/cm^2.
    F_CONST = 96485.33212
    D_H_M2_S = 9.311e-9
    C_H_PH4_MOL_M3 = 0.1
    L_EFF_M = 100e-6
    LEVICH_H_MA_CM2 = -(F_CONST * D_H_M2_S * C_H_PH4_MOL_M3 / L_EFF_M) / 10.0

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True)
    ax_cd, ax_pc = axes

    n_curves = len(reports)
    cmap = plt.get_cmap("viridis")
    colors = [cmap(t) for t in np.linspace(0.05, 0.88, n_curves)]

    all_v: list[float] = []
    all_cd: list[float] = []
    all_pc: list[float] = []
    for color, factor in zip(colors, factors):
        if factor not in reports:
            continue
        rep = reports[factor]
        v = np.array(rep["v_rhe"], dtype=float)
        cd = _to_arr(rep["cd_mA_cm2"])
        pc = _to_arr(rep["pc_mA_cm2"])
        n_ok = int(rep.get("n_converged", 0))
        n_tot = int(rep.get("n_total", len(v)))
        label = f"{_factor_legend_label(factor)}  ({n_ok}/{n_tot})"

        lw = 1.6 if n_curves <= 6 else 1.3
        ms = 4 if n_curves <= 6 else 3
        ax_cd.plot(
            v, cd, marker="o", color=color, lw=lw, ms=ms, label=label,
        )
        ax_pc.plot(
            v, pc, marker="o", color=color, lw=lw, ms=ms, label=label,
        )
        all_v.extend(v.tolist())
        all_cd.extend(cd[np.isfinite(cd)].tolist())
        all_pc.extend(pc[np.isfinite(pc)].tolist())

    exp_v: np.ndarray | None = None
    exp_j: np.ndarray | None = None
    if args.overlay_experimental:
        exp_pair = _load_experimental_csv(EXPERIMENTAL_CSV)
        if exp_pair is not None:
            exp_v, exp_j = exp_pair

    v_min = min(all_v)
    v_max = max(all_v)
    v_pad = 0.02 * (v_max - v_min)
    x_lo = v_min - v_pad
    x_hi = v_max + v_pad

    data_y_min = min(min(all_cd, default=0.0), min(all_pc, default=0.0))
    y_lo = min(data_y_min, LEVICH_H_MA_CM2)
    if exp_j is not None:
        y_lo = min(y_lo, float(np.min(exp_j)))
    y_pad_lo = 0.10 * abs(y_lo)
    y_pad_hi = 0.10 * abs(y_lo)
    y_min = y_lo - y_pad_lo
    y_max = 0.0 + y_pad_hi

    for ax in axes:
        ax.axhline(0.0, color="black", lw=0.7, ls="-", alpha=0.4)
        ax.axhline(
            LEVICH_H_MA_CM2, color="C3", lw=0.9, ls="--", alpha=0.6,
            zorder=1,
        )
        ax.set_xlabel(r"$V_{\mathrm{RHE}}$ (V)")
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3)
        ax.text(
            0.99, 0.02,
            f"H$^+$ diffusion Levich limit  ({LEVICH_H_MA_CM2:+.4f} mA/cm$^2$, pH 4)",
            transform=ax.transAxes,
            fontsize=8, color="C3", alpha=0.85,
            va="bottom", ha="right",
        )

    ax_cd.set_ylabel(r"$j_{\mathrm{disk}}$ (mA/cm$^2$)")
    ax_cd.set_title(r"Total disk current density vs $V_{\mathrm{RHE}}$")
    ax_pc.set_ylabel(r"$j_{\mathrm{H_2O_2,gross}}$ (mA/cm$^2$)")
    ax_pc.set_title(r"Gross H$_2$O$_2$ current (R2e channel) vs $V_{\mathrm{RHE}}$")

    if exp_v is not None and exp_j is not None:
        ax_pc.plot(
            exp_v, exp_j, marker="x", color="black", lw=2.0, ms=6,
            mew=1.5, ls="-", zorder=10,
            label="Mangan deck p15 (exp, Cs$^+$/pH 4)",
        )
        ax_pc.legend(loc="lower left", fontsize=8, frameon=True, framealpha=0.9)

    fig.suptitle(
        "PNP+BV forward solver demo — Cs$^+$/SO$_4^{2-}$, slide-15 V band"
        + str(args.title_suffix) + "\n"
        "no water hydrolysis, no cation hydrolysis; physical $a_{\\mathrm{nondim}}$ "
        "for dynamic species; $L_{\\mathrm{eff}}=100$ μm",
        fontsize=11,
    )

    handles, labels = ax_cd.get_legend_handles_labels()
    ncol = min(len(labels), 5)
    fig.legend(
        handles, labels,
        loc="lower center", ncol=ncol,
        bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=8.5,
    )

    leg_rows = (len(labels) + ncol - 1) // ncol
    bottom_pad = 0.04 + 0.025 * leg_rows
    fig.tight_layout(rect=(0, bottom_pad, 1, 0.93))

    out_png = results_dir / f"{args.out_stem}.png"
    out_pdf = results_dir / f"{args.out_stem}.pdf"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")

    print(
        "\nFactor       n_ok/n_tot    cd@-0.40V   cd@+0.55V   "
        "pc@-0.40V   pc@+0.55V"
    )
    for factor in factors:
        if factor not in reports:
            continue
        rep = reports[factor]
        v = np.array(rep["v_rhe"], dtype=float)
        cd = _to_arr(rep["cd_mA_cm2"])
        pc = _to_arr(rep["pc_mA_cm2"])
        i_lo = int(np.argmin(v))
        i_hi = int(np.argmax(v))
        n_ok = int(rep.get("n_converged", 0))
        n_tot = int(rep.get("n_total", len(v)))
        cd_lo = cd[i_lo] if np.isfinite(cd[i_lo]) else float("nan")
        cd_hi = cd[i_hi] if np.isfinite(cd[i_hi]) else float("nan")
        pc_lo = pc[i_lo] if np.isfinite(pc[i_lo]) else float("nan")
        pc_hi = pc[i_hi] if np.isfinite(pc[i_hi]) else float("nan")
        print(
            f"{factor:9.0e}    {n_ok:>3d}/{n_tot:<3d}     "
            f"{cd_lo:+8.4f}    {cd_hi:+8.4f}    "
            f"{pc_lo:+8.4f}    {pc_hi:+8.4f}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
