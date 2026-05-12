"""Plot model j_peroxide in slide-15 convention (cathodic = negative).

Side-by-side: model curve on slide-15 axes (V from -0.5 to +1.25,
y from -0.6 to 0.0 mA/cm²) and the 4 K@pH4 scalar deck points
(ring_onset, max_ring_current → disk-basis peroxide magnitude).

This is for qualitative shape comparison vs Mangan-deck slide 15
(Cs+ pH 4, volcano shape with peak near V ≈ +0.1 V).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    eval_db_path = "StudyResults/phase6b_step10_phase_D/eval_db_0p0_stern_baseline.json"
    deck_audit_path = "StudyResults/phase6b_step10_phase_D/data_audit_K_at_pH4.json"
    out_path = "StudyResults/phase6b_step10_phase_D/phase_D_vs_slide15_convention.png"

    with open(eval_db_path, "r", encoding="utf-8") as f:
        db = json.load(f)
    with open(deck_audit_path, "r", encoding="utf-8") as f:
        deck = json.load(f)

    records = [r for r in db["per_v_records"] if r["snes_converged"]]
    v_rhe = np.array([r["v_rhe"] for r in records])
    pc = np.array([r["pc_mA_cm2"] for r in records])  # disk peroxide partial, positive

    # Slide-15 convention: peroxide current density plotted as negative (cathodic)
    pc_slide = -pc

    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    # Model curve (slide convention)
    ax.plot(
        v_rhe,
        pc_slide,
        "o-",
        color="#c0392b",
        lw=1.8,
        ms=5,
        label="Model j_peroxide (Δ_β=0 Stern baseline, K₂SO₄ pH 4)",
    )

    # Deck K@pH4 scalar points: each row has (ring_onset_V, max_ring_current)
    # Convert max_ring_current to disk-basis: divide by N_collection=0.224
    N_collection = 0.2237
    for row in deck["rows"]:
        v_onset = row["ring_onset_pot_V_at_0.01_mA_cm2"]
        max_ring = row["max_ring_current_mA_cm2"]
        max_disk_peroxide = max_ring / N_collection
        # Plot a horizontal error-bar-like span from onset to overpotential
        # to indicate where this experiment's max sits (rough)
        v_peak_est = v_onset - row["overpotential_V"]
        ax.scatter(
            [v_onset],
            [0.0],
            marker="v",
            s=70,
            color="#2980b9",
            edgecolor="black",
            zorder=4,
            label=("deck K@pH4 ring onset" if row is deck["rows"][0] else None),
        )
        ax.scatter(
            [v_peak_est],
            [-max_disk_peroxide],
            marker="*",
            s=130,
            color="#27ae60",
            edgecolor="black",
            zorder=4,
            label=("deck K@pH4 max peroxide (disk basis)" if row is deck["rows"][0] else None),
        )

    # Slide-15 approximate volcano envelope (digitized from figure inspection)
    # Onset ~+0.4V, peak ~-0.55 at V≈+0.1, then climbing back to ~-0.2 at V=-0.3
    v_slide = np.array([-0.45, -0.30, -0.15, 0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 1.0, 1.25])
    j_slide = np.array([-0.18, -0.20, -0.30, -0.45, -0.55, -0.50, -0.30, -0.10, -0.02, 0.00, 0.00, 0.00])
    ax.plot(
        v_slide,
        j_slide,
        "--",
        color="#8e44ad",
        lw=2.0,
        alpha=0.7,
        label="Slide 15 envelope (Cs⁺ pH 4, digitized eyeball)",
    )

    # Annotations
    ax.axhline(0.0, color="black", lw=0.5)
    ax.set_xlim(-0.5, 1.25)
    ax.set_ylim(-1.1, 0.05)
    ax.set_xlabel("Applied Voltage (V vs RHE)")
    ax.set_ylabel("Peroxide Current Density (mA/cm²)  [cathodic = negative]")
    ax.set_title(
        "Model peroxide IV vs Mangan slide-15 (V10B kinetics, Δ_β=0, K₂SO₄ pH 4)"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8.5, frameon=True)

    # Onset comparison text box
    deck_onset_mean = deck["statistics"]["ring_onset_pot_V_at_0.01_mA_cm2"]["mean"]
    # Find model onset where pc crosses 0.01/N (ring threshold on disk basis)
    onset_thresh = 0.01 / N_collection
    onset_idx = np.where(pc >= onset_thresh)[0]
    model_onset = v_rhe[onset_idx[-1]] if len(onset_idx) else float("nan")
    text = (
        f"Deck K@pH4 onset (mean): V = +{deck_onset_mean:.3f} V\n"
        f"Model onset (j_per > {onset_thresh:.3f}): V = {model_onset:+.3f} V\n"
        f"Shift: model is {(model_onset - deck_onset_mean) * 1000:+.0f} mV anodic\n"
        f"Model has NO peak — monotonic plateau→cliff\n"
        f"Slide 15 has a peak at V ≈ +0.1 V then declines"
    )
    ax.text(
        0.65,
        -0.80,
        text,
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="#999"),
        family="monospace",
    )

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    print(f"wrote {out_path}")
    print(f"  model onset: V = {model_onset:+.3f} V   deck mean onset: V = +{deck_onset_mean:.3f} V")
    print(f"  shift: {(model_onset - deck_onset_mean) * 1000:+.0f} mV")


if __name__ == "__main__":
    main()
