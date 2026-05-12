"""Plot peroxide-current vs V_RHE from a Phase D eval_db snapshot.

Reads `StudyResults/phase6b_step10_phase_D/eval_db_*.json` (default:
Δ_β=0 Stern baseline) and writes a PNG with three curves:

  - cathodic disk total (cd_mA_cm2, sign-flipped to plot as positive
    reduction magnitude)
  - peroxide partial current at disk (pc_mA_cm2 == gross_h2o2)
  - ring-detected peroxide current (ring_current_ring_basis_mA_cm2)

The grey shaded band marks the V_kin point (excluded from the
selectivity observable). Vertical dashed line = anchor V.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-db",
        default="StudyResults/phase6b_step10_phase_D/eval_db_0p0_stern_baseline.json",
    )
    parser.add_argument(
        "--outfile",
        default="StudyResults/phase6b_step10_phase_D/phase_D_peroxide_IV_baseline.png",
    )
    parser.add_argument(
        "--title",
        default="Phase D Δ_β=0 Stern baseline — peroxide IV (V_RHE [-0.1, +1.0])",
    )
    args = parser.parse_args()

    with open(args.eval_db, "r", encoding="utf-8") as f:
        db = json.load(f)

    records = [r for r in db["per_v_records"] if r["snes_converged"]]
    v_rhe = np.array([r["v_rhe"] for r in records])
    cd = np.array([r["cd_mA_cm2"] for r in records])
    pc = np.array([r["pc_mA_cm2"] for r in records])
    ring = np.array([r["ring_current_ring_basis_mA_cm2"] for r in records])

    cfg = db["config"]
    v_anchor = cfg["v_anchor"]
    mask_lo, mask_hi = cfg["observable_mask"]
    v_kin = cfg["v_kin_excluded_from_observables"]
    agg = db["aggregated_observables"]
    max_sel = agg["max_H2O2_selectivity_in_window_pct"]
    argmax_v = agg["argmax_V_for_selectivity"]

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.0), sharex=True)

    # Top: disk currents (cathodic ORR total + peroxide partial)
    ax = axes[0]
    ax.plot(v_rhe, -cd, "o-", color="#2c3e50", lw=1.6, ms=4, label="−j_disk (cathodic, total ORR)")
    ax.plot(v_rhe, pc, "s-", color="#c0392b", lw=1.6, ms=4, label="j_peroxide (disk, gross 2e⁻)")
    ax.axvline(v_anchor, ls="--", color="#7f8c8d", lw=0.8, alpha=0.7, label=f"anchor V = {v_anchor:+.2f}")
    ax.axvspan(v_kin - 0.02, v_kin + 0.02, color="#bdc3c7", alpha=0.4, label=f"V_kin = {v_kin:+.2f} (excluded)")
    ax.axvspan(mask_lo, mask_hi, color="#ecf0f1", alpha=0.3, zorder=-1)
    ax.set_ylabel("current density (mA cm$^{-2}$)")
    ax.set_title(args.title)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8.5, frameon=True)

    # Bottom: ring-detected peroxide + selectivity-style readout
    ax = axes[1]
    ax.plot(v_rhe, ring, "^-", color="#8e44ad", lw=1.6, ms=4, label="j_ring (peroxide, N=0.224)")
    ax.axvline(v_anchor, ls="--", color="#7f8c8d", lw=0.8, alpha=0.7)
    ax.axvspan(v_kin - 0.02, v_kin + 0.02, color="#bdc3c7", alpha=0.4)
    ax.axvspan(mask_lo, mask_hi, color="#ecf0f1", alpha=0.3, zorder=-1)
    # Annotate the selectivity argmax
    ax.scatter(
        [argmax_v],
        [ring[np.argmin(np.abs(v_rhe - argmax_v))]],
        s=110,
        marker="*",
        color="gold",
        edgecolor="black",
        zorder=5,
        label=f"argmax sel = {max_sel:.2f}% @ V = {argmax_v:+.2f}",
    )
    ax.set_xlabel("V vs RHE (V)")
    ax.set_ylabel("ring current density (mA cm$^{-2}$)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8.5, frameon=True)

    plt.tight_layout()
    out = Path(args.outfile)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=160)
    print(f"wrote {out}")
    print(f"  V_RHE range: [{v_rhe.min():+.3f}, {v_rhe.max():+.3f}] V")
    print(f"  max |j_disk|   = {np.abs(cd).max():7.4f} mA/cm² at V={v_rhe[np.argmax(np.abs(cd))]:+.3f}")
    print(f"  max j_peroxide = {pc.max():7.4f} mA/cm² at V={v_rhe[np.argmax(pc)]:+.3f}")
    print(f"  max j_ring     = {ring.max():7.4f} mA/cm² at V={v_rhe[np.argmax(ring)]:+.3f}")
    print(f"  argmax selectivity = {max_sel:.2f}% at V={argmax_v:+.3f}")


if __name__ == "__main__":
    main()
