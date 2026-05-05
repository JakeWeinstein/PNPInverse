"""MMS voltage sweep: probe convergence under eta-clip activation.

Runs the production-faithful MMS test from
``mms_bv_3sp_logc_boltzmann.py`` at multiple V_RHE values to quantify
the effect of the eta-clip (active for R2 below V_RHE = +0.495 V; see
``docs/clipping_conventions.md``).

The MMS source builder mirrors the production weak form term-by-term.
Two modes are exposed via ``--clip-mode``:

  unclipped (default)
      Source uses the unclipped continuum ``eta = (V - E_eq)/V_T``.
      The discrete operator clips at +/-50. Errors thus include both
      h^p truncation AND the clip-induced model error. Above the R2
      unclip threshold the clip is inactive and rates stay at h^p;
      below threshold rates degrade and (deeper still) Newton may
      stop converging entirely. This is the "what does the clip
      cost?" test.

  consistent
      Source applies the same eta-clip as the discrete operator. Tests
      whether the discrete operator self-consistently recovers its
      (clipped) continuum solution at h^p. Should pass at h^p for all
      V; failure here would indicate a discrete-operator bug
      independent of the clip's physical fidelity.

  both
      Run both modes back-to-back so the two effects can be separated.

Default voltages span the R2 cathodic unclip threshold (+0.495 V):

  +0.55 V  - both reactions unclipped (TestMMSConvergence baseline)
  +0.495 V - R2 right at clip threshold
  +0.40 V  - R2 just clipped
  +0.20 V  - R2 moderately clipped
  -0.10 V  - R2 well clipped
  -0.30 V  - R2 deeply clipped (production saturation regime)
  -0.50 V  - R2 heavily clipped (edge of production grid)

Mesh sweep per voltage: UnitSquareMesh(N) for N in [8, 16, 32, 64].

Outputs (in --out-dir, default ``StudyResults/mms_voltage_sweep/``):

  results_<mode>.json          per-voltage rates + raw errors
  rate_vs_voltage_<mode>.png   summary plot (rate vs V_RHE)
  convergence_<mode>.png       per-voltage L2/H1 curves grid

Usage::

    python scripts/verification/mms_voltage_sweep.py
    python scripts/verification/mms_voltage_sweep.py --clip-mode both
    python scripts/verification/mms_voltage_sweep.py \
        --voltages 0.55 0.40 -0.30 --Nvals 8 16 32

Run from the PNPInverse/ directory with the ``../venv-firedrake/bin/activate``
environment.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Path / Firedrake cache setup (mirror the underlying MMS script)
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("FIREDRAKE_TSFC_KERNEL_CACHE_DIR", "/tmp/firedrake-tsfc")
os.environ.setdefault("PYOP2_CACHE_DIR", "/tmp/pyop2")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts._bv_common import V_T
from scripts.verification.mms_bv_3sp_logc_boltzmann import (
    SPECIES_NAMES,
    run_mms,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_VOLTAGES: list[float] = [0.55, 0.495, 0.40, 0.20, -0.10, -0.30, -0.50]
DEFAULT_NVALS: list[int] = [8, 16, 32, 64]

# Reactions (must match scripts/_bv_common.py)
_E_EQ_R1 = 0.68
_E_EQ_R2 = 1.78
_CLIP_HALF_WIDTH = 50.0  # |eta_scaled| clip threshold (nondim)

# Field bookkeeping (mirrors the underlying MMS module)
_N_SPECIES = 3  # O2, H2O2, H+


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clip_status(v_rhe: float, e_eq: float) -> tuple[float, str]:
    """Return ``(eta_unclipped, status)`` where status is 'unclipped',
    'clipped (cathodic)', or 'clipped (anodic)' for the given reaction's
    cathodic exponent at V_RHE."""
    eta = (v_rhe - e_eq) / V_T
    if abs(eta) <= _CLIP_HALF_WIDTH:
        return eta, "unclipped"
    return eta, "clipped (cathodic)" if eta < 0 else "clipped (anodic)"


def _compute_rate(h_vals: list[float], err_vals: list[float]) -> tuple[float, float]:
    """Log-log linear regression. Returns ``(slope, r_squared)``.

    Returns ``(nan, nan)`` if any error is non-finite or if there are
    fewer than 2 valid points.
    """
    if len(h_vals) < 2 or len(err_vals) < 2 or len(h_vals) != len(err_vals):
        return float("nan"), float("nan")
    h_arr = np.array(h_vals, dtype=float)
    err_arr = np.array(err_vals, dtype=float)
    if not np.all(np.isfinite(err_arr)) or np.any(err_arr <= 0):
        return float("nan"), float("nan")
    from scipy.stats import linregress
    log_h = np.log(h_arr)
    log_err = np.log(err_arr)
    slope, _intercept, r_value, _p, _se = linregress(log_h, log_err)
    return float(slope), float(r_value ** 2)


def _field_keys() -> list[tuple[str, str, str]]:
    """Return list of ``(json_key, plot_label, color)`` for all 4 fields."""
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    keys: list[tuple[str, str, str]] = []
    for i in range(_N_SPECIES):
        keys.append((f"u{i}", SPECIES_NAMES[i], colors[i]))
    keys.append(("phi", "phi", colors[3]))
    return keys


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------
def run_voltage_sweep(
    voltages: list[float],
    n_vals: list[int],
    *,
    clip_source: bool,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run MMS convergence at each voltage and aggregate results."""
    sweep: dict[str, Any] = {
        "voltages": [],
        "n_vals": list(n_vals),
        "clip_source": bool(clip_source),
        "per_voltage": [],
    }

    for v_rhe in voltages:
        eta_hat = v_rhe / V_T
        eta_R1, R1_status = _clip_status(v_rhe, _E_EQ_R1)
        eta_R2, R2_status = _clip_status(v_rhe, _E_EQ_R2)
        if verbose:
            print()
            print("#" * 80)
            print(f"  V_RHE = {v_rhe:+.3f} V    eta_hat = {eta_hat:+.3f}")
            print(f"  R1 (E_eq={_E_EQ_R1}): eta={eta_R1:+.2f} -> {R1_status}")
            print(f"  R2 (E_eq={_E_EQ_R2}): eta={eta_R2:+.2f} -> {R2_status}")
            print(f"  clip_source   = {clip_source}")
            print("#" * 80)

        t0 = time.time()
        try:
            mms_res = run_mms(
                n_vals,
                verbose=verbose,
                eta_hat=eta_hat,
                v_rhe=v_rhe,
                clip_source=clip_source,
            )
            run_error: str | None = None
        except Exception as exc:
            print(f"  [ERROR] V={v_rhe} raised: {type(exc).__name__}: {exc}")
            mms_res = {"N": [], "h": []}
            run_error = f"{type(exc).__name__}: {exc}"
        elapsed = time.time() - t0

        # Per-field rates + raw errors
        h_list = list(mms_res.get("h", []))
        fields: dict[str, dict[str, Any]] = {}
        for key, _label, _color in _field_keys():
            for norm in ("L2", "H1"):
                err_list = list(mms_res.get(f"{key}_{norm}", []))
                rate, r2 = _compute_rate(h_list, err_list)
                fields[f"{key}_{norm}"] = {
                    "rate": rate,
                    "r_squared": r2,
                    "errors": err_list,
                }

        sweep["per_voltage"].append({
            "v_rhe": v_rhe,
            "eta_hat": eta_hat,
            "eta_R1": eta_R1,
            "eta_R2": eta_R2,
            "r1_clip_status": R1_status,
            "r2_clip_status": R2_status,
            "newton_converged_count": len(mms_res.get("N", [])),
            "n_attempts": len(n_vals),
            "h_values": h_list,
            "fields": fields,
            "elapsed_seconds": float(elapsed),
            "run_error": run_error,
        })
        sweep["voltages"].append(v_rhe)

    return sweep


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_rate_vs_voltage(sweep: dict[str, Any], out_path: str) -> str:
    """Summary plot: convergence rate vs voltage for each field, L2 + H1."""
    voltages = [pv["v_rhe"] for pv in sweep["per_voltage"]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, norm, expected, ylim in (
        (axes[0], "L2", 2.0, (-0.5, 3.0)),
        (axes[1], "H1", 1.0, (-0.5, 2.0)),
    ):
        for key, label, color in _field_keys():
            rates = [
                pv["fields"].get(f"{key}_{norm}", {}).get("rate", float("nan"))
                for pv in sweep["per_voltage"]
            ]
            ax.plot(voltages, rates, "o-", color=color, label=label, linewidth=1.5)
        ax.axhline(expected, color="k", linestyle=":", linewidth=1,
                   label=f"theoretical {expected:.1f}")
        ax.axhline(expected - 0.2, color="gray", linestyle="--", linewidth=0.8,
                   label="PASS band (-0.2)")
        ax.axvline(+0.495, color="red", linestyle="--", linewidth=0.8,
                   label="R2 unclip threshold")
        ax.set_xlabel("$V_{RHE}$ [V]")
        ax.set_ylabel(f"{norm} convergence rate")
        ax.set_title(f"{norm} rate vs voltage  (clip_source={sweep['clip_source']})")
        ax.set_ylim(*ylim)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "MMS voltage sweep: clip-induced convergence degradation",
        fontsize=12,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_convergence_grid(sweep: dict[str, Any], out_path: str) -> str:
    """Per-voltage convergence curves: rows=voltages, cols=(L2, H1)."""
    pv_list = sweep["per_voltage"]
    n_v = len(pv_list)
    fig, axes = plt.subplots(n_v, 2, figsize=(13, 3.0 * n_v), squeeze=False)

    for row, pv in enumerate(pv_list):
        h_list = pv["h_values"]
        for col, norm, expected in ((0, "L2", 2.0), (1, "H1", 1.0)):
            ax = axes[row, col]
            for key, label, color in _field_keys():
                err = pv["fields"].get(f"{key}_{norm}", {}).get("errors", [])
                if len(err) == len(h_list) and len(h_list) >= 2:
                    h_arr = np.array(h_list)
                    e_arr = np.array(err, dtype=float)
                    mask = np.isfinite(e_arr) & (e_arr > 0)
                    if mask.sum() >= 2:
                        ax.loglog(
                            h_arr[mask], e_arr[mask],
                            "o-", color=color, label=label,
                            linewidth=1.3, markersize=4,
                        )
            if h_list:
                h_ref = np.array([h_list[0], h_list[-1]])
                # Reference line at the first finite L2/H1 anchor
                ax.loglog(
                    h_ref, h_ref ** expected, "k:",
                    linewidth=0.8, label=f"$O(h^{{{int(expected)}}})$",
                )
            rate_line = (
                f"{norm}  V={pv['v_rhe']:+.3f}V  "
                f"R2: {pv['r2_clip_status']}  "
                f"Newton: {pv['newton_converged_count']}/{pv['n_attempts']}"
            )
            ax.set_title(rate_line, fontsize=9)
            ax.set_xlabel("$h$")
            ax.set_ylabel(f"{norm} error")
            ax.grid(True, which="both", alpha=0.3)
            if row == 0 and col == 1:
                ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        f"MMS voltage sweep convergence curves (clip_source={sweep['clip_source']})",
        fontsize=12,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Pretty-print summary table
# ---------------------------------------------------------------------------
def print_summary(sweep: dict[str, Any]) -> None:
    print()
    print(f"=== clip_source = {sweep['clip_source']} ===")
    header = (
        f"  {'V_RHE':>7s}  {'R2 status':>20s}  "
        f"{'O2 L2':>8s}  {'H2O2 L2':>9s}  {'H+ L2':>8s}  {'phi L2':>8s}  "
        f"{'O2 H1':>8s}  {'H2O2 H1':>9s}  {'H+ H1':>8s}  {'phi H1':>8s}  "
        f"{'Newton':>9s}"
    )
    print(header)
    print("-" * len(header))
    for pv in sweep["per_voltage"]:
        f = pv["fields"]
        def _r(key: str) -> str:
            r = f.get(key, {}).get("rate", float("nan"))
            return f"{r:+.3f}" if np.isfinite(r) else "  nan "
        nconv = pv["newton_converged_count"]
        natt = pv["n_attempts"]
        print(
            f"  {pv['v_rhe']:>+7.3f}  {pv['r2_clip_status']:>20s}  "
            f"{_r('u0_L2'):>8s}  {_r('u1_L2'):>9s}  {_r('u2_L2'):>8s}  {_r('phi_L2'):>8s}  "
            f"{_r('u0_H1'):>8s}  {_r('u1_H1'):>9s}  {_r('u2_H1'):>8s}  {_r('phi_H1'):>8s}  "
            f"{nconv:>4d}/{natt:<4d}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--voltages",
        type=float,
        nargs="+",
        default=DEFAULT_VOLTAGES,
        help=f"V_RHE values (V vs RHE). Default: {DEFAULT_VOLTAGES}",
    )
    parser.add_argument(
        "--Nvals",
        type=int,
        nargs="+",
        default=DEFAULT_NVALS,
        help=f"UnitSquareMesh(N) sizes per voltage. Default: {DEFAULT_NVALS}",
    )
    parser.add_argument(
        "--clip-mode",
        choices=["unclipped", "consistent", "both"],
        default="unclipped",
        help=(
            "unclipped: source uses true eta (default; shows clip-induced "
            "model error). consistent: source uses clipped eta (verifies "
            "discrete operator). both: run both modes."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(_ROOT, "StudyResults", "mms_voltage_sweep"),
        help="Output directory for JSON + plots.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    verbose = not args.quiet
    modes = ["unclipped", "consistent"] if args.clip_mode == "both" else [args.clip_mode]

    all_sweeps: dict[str, dict[str, Any]] = {}
    for mode in modes:
        clip_source = (mode == "consistent")
        if verbose:
            print()
            print("=" * 80)
            print(f"  MODE: {mode}  (clip_source={clip_source})")
            print("=" * 80)

        sweep = run_voltage_sweep(
            args.voltages, args.Nvals,
            clip_source=clip_source, verbose=verbose,
        )
        sweep["metadata"] = {
            "date": datetime.now(timezone.utc).isoformat(),
            "voltages": list(args.voltages),
            "n_vals": list(args.Nvals),
            "mode": mode,
            "description": (
                "MMS convergence rate at multiple voltages spanning the R2 "
                "eta-clip activation threshold (+0.495 V). Source mode = "
                f"{mode}."
            ),
        }
        all_sweeps[mode] = sweep

        # JSON
        json_path = os.path.join(args.out_dir, f"results_{mode}.json")
        with open(json_path, "w") as f:
            json.dump(sweep, f, indent=2)
        print(f"\n[OK] JSON saved -> {json_path}")

        # Plots
        rate_plot = os.path.join(args.out_dir, f"rate_vs_voltage_{mode}.png")
        plot_rate_vs_voltage(sweep, rate_plot)
        print(f"[OK] Plot saved -> {rate_plot}")

        conv_plot = os.path.join(args.out_dir, f"convergence_{mode}.png")
        plot_convergence_grid(sweep, conv_plot)
        print(f"[OK] Plot saved -> {conv_plot}")

    # Final summary
    for sweep in all_sweeps.values():
        print_summary(sweep)


if __name__ == "__main__":
    main()
