"""Plot Jithin Fig 4.36 emulation jV curve + OHP-concentration diagnostics.

Reads ``StudyResults/jithin_emulation_fig436/iv_curve.json`` produced by
``_run_jithin_emulation_fig436.py`` and produces:

* ``iv_curve.png``   — total current density vs V_RHE, overlaid against
  Jithin's reported plateaus.
* ``ohp_diag.png``   — c_O2(OHP), c_H(OHP), and phi(OHP) vs V_RHE.
  Used to investigate Gap 2 (Jithin's far-cathodic cliff vs our flat
  plateau): if c_O2(OHP) ≈ 0 across the cathodic plateau, additional
  steric occlusion of O₂ by cation pile-up cannot create the cliff in
  our solver — it would need a smaller starting c_O2(OHP) to be
  further reduced.

Usage::

    python -u scripts/studies/_plot_jithin_emulation_fig436.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve()
_ROOT = str(_HERE.parents[2])

DEFAULT_DIR = Path(_ROOT) / "StudyResults" / "jithin_emulation_fig436"


def _load_array(report: dict, key: str) -> np.ndarray:
    if key not in report:
        return np.array([])
    return np.array(
        [x if x is not None else np.nan for x in report[key]], dtype=float
    )


def _plot_iv(report: dict, run_dir: Path) -> None:
    v = np.array(report["v_rhe"], dtype=float)
    cd = _load_array(report, "cd_mA_cm2")
    pc = _load_array(report, "pc_mA_cm2")

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    finite_cd = np.isfinite(cd)
    ax.plot(
        v[finite_cd], cd[finite_cd],
        "o-", color="C0", linewidth=1.5,
        label="Ours (Jithin-emulation total cd)",
    )
    finite_pc = np.isfinite(pc)
    ax.plot(
        v[finite_pc], pc[finite_pc],
        "x--", color="C2", linewidth=1.0, alpha=0.6,
        label="Ours (peroxide-pathway current alone)",
    )

    ref = report.get("jithin_reference", {})
    if (val := ref.get("fig436_simulated_plateau_mA_cm2")) is not None:
        ax.axhline(
            val, color="C1", linestyle=":", linewidth=1.5,
            label=f"Jithin Fig 4.36 simulated plateau ≈ {val:+.3f} mA/cm²",
        )
    if (val := ref.get("exp_plateau_mA_cm2")) is not None:
        ax.axhline(
            val, color="C3", linestyle=":", linewidth=1.5,
            label=f"Experimental plateau ≈ {val:+.3f} mA/cm²",
        )
    if (val := ref.get("diffusion_limit_calc_L10um_mA_cm2")) is not None:
        ax.axhline(
            val, color="grey", linestyle="-.", linewidth=1.0,
            label=f"Levich limit at L=10μm ≈ {val:+.3f} mA/cm²",
        )
    ax.axhline(0.0, color="k", linewidth=0.5)

    cfg = report["config"]
    alpha_ne = cfg.get(
        "alpha_times_n_e",
        cfg["alpha_tafel"] * cfg["n_electrons_tafel"],
    )
    stern = cfg.get("stern_target", cfg.get("stern_baseline", 0.20))
    ax.set_xlabel("V_RHE (V)")
    ax.set_ylabel("Current density (mA/cm²)")
    ax.set_title(
        f"Jithin Fig 4.36 emulation: Tafel-only R2e, "
        f"α·n_e = {alpha_ne:.2f} (A_Tafel = {cfg['A_tafel_mV_dec']:.1f} mV/dec)\n"
        f"Cs⁺/SO₄²⁻ pH 2; L_eff = {cfg['l_eff_m'] * 1e6:.0f} μm; "
        f"C_S = {stern:.2f} F/m² (Jithin L_Stern=0.6 nm → 1.16)"
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    out_png = run_dir / "iv_curve.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"wrote {out_png}")


def _plot_ohp_diag(report: dict, run_dir: Path) -> None:
    """OHP-surface concentrations vs V_RHE — Gap 2 investigation."""
    v = np.array(report["v_rhe"], dtype=float)
    c_o2 = _load_array(report, "c_O2_OHP_nondim")
    c_h = _load_array(report, "c_H_OHP_nondim")
    phi = _load_array(report, "phi_OHP_nondim")

    if c_o2.size == 0 or not np.any(np.isfinite(c_o2)):
        print("no OHP-concentration data in JSON; skipping ohp_diag.png")
        return

    cfg = report["config"]
    c_scale = float(cfg.get("C_SCALE_mol_m3", 1.2))
    c_o2_bulk = float(cfg.get("c_o2_mol_m3", 0.25))
    c_h_bulk = float(cfg.get("c_hp_mol_m3", 10.0))

    fig, (axa, axb, axc) = plt.subplots(3, 1, figsize=(8.0, 9.0), sharex=True)

    # Panel A — O₂(OHP) in mol/m³
    finite = np.isfinite(c_o2)
    axa.plot(
        v[finite], c_o2[finite] * c_scale, "o-", color="C0",
        label="c_O₂(OHP)",
    )
    axa.axhline(c_o2_bulk, color="C0", linestyle=":", linewidth=1.0,
                label=f"bulk c_O₂ = {c_o2_bulk:.3f} mol/m³")
    axa.set_ylabel("c_O₂ at OHP (mol/m³)")
    axa.set_yscale("symlog", linthresh=1e-4)
    axa.legend(loc="best", fontsize=8)
    axa.grid(True, alpha=0.3)
    axa.set_title(
        "OHP-surface diagnostics — Gap 2 investigation\n"
        "Does cation pile-up at OHP sterically occlude O₂?"
    )

    # Panel B — H⁺(OHP) in mol/m³ (log)
    finite_h = np.isfinite(c_h)
    axb.plot(
        v[finite_h], c_h[finite_h] * c_scale, "o-", color="C3",
        label="c_H⁺(OHP)",
    )
    axb.axhline(c_h_bulk, color="C3", linestyle=":", linewidth=1.0,
                label=f"bulk c_H⁺ = {c_h_bulk:.1f} mol/m³ (pH 2)")
    axb.set_ylabel("c_H⁺ at OHP (mol/m³)")
    axb.set_yscale("log")
    axb.legend(loc="best", fontsize=8)
    axb.grid(True, alpha=0.3, which="both")

    # Panel C — φ(OHP) in nondim units (β·ψ)
    finite_phi = np.isfinite(phi)
    axc.plot(
        v[finite_phi], phi[finite_phi], "o-", color="C2",
        label="φ(OHP) [nondim, β·ψ]",
    )
    axc.axhline(0.0, color="k", linewidth=0.5)
    axc.set_xlabel("V_RHE (V)")
    axc.set_ylabel("φ at OHP (nondim)")
    axc.legend(loc="best", fontsize=8)
    axc.grid(True, alpha=0.3)

    out_png = run_dir / "ohp_diag.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"wrote {out_png}")


def main() -> int:
    run_dir = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    )
    json_path = run_dir / "iv_curve.json"
    if not json_path.exists():
        print(f"NOT FOUND: {json_path}", file=sys.stderr)
        return 1
    with open(json_path) as f:
        report = json.load(f)
    _plot_iv(report, run_dir)
    _plot_ohp_diag(report, run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
