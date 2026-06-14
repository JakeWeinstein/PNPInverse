"""Phase 7.3 M3 — ring SHAPE: C1 (faradaic) vs a uniform-suppression null.

Post-hoc (no solver). Tests which reproduces the data RING SHAPE at each pH:
  * C1 fit       — the exploratory C1 net ring (faradaic, surface-c_H gated).
  * uniform null — N0's ring scaled by a single per-pH factor s(pH) =
                   data_peak / N0_peak (a pH-dependent ring-collection /
                   escape efficiency; preserves N0's V-shape, disk = N0).
Both are matched to the data ring PEAK by construction, so the discriminator
is the SHAPE (V-location of the peroxide) — scored as RMS(model − data) on
the data's V grid.

Output: StudyResults/phase7p3_m3_c1_exploratory_fit/ring_shape_compare.{json,png}
Run from PNPInverse/ (no Firedrake).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

A_DISK_CM2, A_RING_CM2, N_COLL = 0.19635, 0.109956, 0.224
STOICH_H2O2 = {"R2e_acid": +1, "R2e_water": +1, "R4e_acid": 0,
               "R4e_water": 0, "C1_h2o2_reduction": -1}
PH_SER = Path(_ROOT) / "StudyResults" / "phase7p2_ph_series_generalization"
FIT = Path(_ROOT) / "StudyResults" / "phase7p3_m3_c1_exploratory_fit"
DIGITIZED = PH_SER / "digitized_experimental_3panel.json"
PHS = [2.0, 4.0, 6.0]


def _n0_ring(ph):
    d = json.load(open(PH_SER / f"iv_curve_pH{ph}.json"))
    v = np.array(d["v_rhe_deck"], float)
    pc = np.array([p if p is not None else np.nan for p in d["pc_mA_cm2"]], float)
    o = np.argsort(v)
    return v[o], -pc[o] * N_COLL * A_DISK_CM2 / A_RING_CM2


def _c1_ring(ph):
    d = json.load(open(FIT / f"iv_C1fit_pH{ph}.json"))
    v = np.array(d["v_rhe_deck"], float)
    per = d.get("per_reaction", [None] * len(v))
    netp = np.full(len(v), np.nan)
    for i, pr in enumerate(per):
        if pr:
            netp[i] = sum(STOICH_H2O2.get(p["label"], 0)
                          * (p["rate_2e_units_mA_cm2"] or 0.0) for p in pr)
    o = np.argsort(v)
    return v[o], -netp[o] * N_COLL * A_DISK_CM2 / A_RING_CM2


def _data_ring(dig, ph):
    k = str(int(ph))
    v = np.array(dig["ring"][k]["v_rhe"]); y = np.array(dig["ring"][k]["value"])
    o = np.argsort(v)
    return v[o], y[o]


def _rms_on_data(mv, my, dv, dy):
    lo, hi = max(mv.min(), dv.min()), min(mv.max(), dv.max())
    m = (dv >= lo) & (dv <= hi)
    if m.sum() < 3:
        return None
    return float(np.sqrt(np.mean((np.interp(dv[m], mv, my) - dy[m]) ** 2)))


def main() -> int:
    dig = json.load(open(DIGITIZED))
    out = {"test": "phase7p3_M3_ring_shape_C1_vs_uniform_null", "by_pH": {}}
    for ph in PHS:
        vn, rn = _n0_ring(ph)
        vc, rc = _c1_ring(ph)
        dv, dy = _data_ring(dig, ph)
        n0_peak, data_peak = float(np.nanmax(rn)), float(np.nanmax(dy))
        s = data_peak / n0_peak                       # uniform null scale
        ru = rn * s
        rms_c1 = _rms_on_data(vc, rc, dv, dy)
        rms_null = _rms_on_data(vn, ru, dv, dy)
        out["by_pH"][str(ph)] = {
            "null_scale": s, "n0_peak": n0_peak, "data_peak": data_peak,
            "ring_shape_rms_C1": rms_c1,
            "ring_shape_rms_uniform_null": rms_null,
            "shape_winner": ("uniform_null" if (rms_c1 and rms_null
                             and rms_null < rms_c1) else "C1"),
            "data_peak_V": float(dv[np.argmax(dy)]),
            "n0_peak_V": float(vn[np.argmax(rn)]),
        }
    with open(FIT / "ring_shape_compare.json", "w") as f:
        json.dump(out, f, indent=2)

    print("=" * 74)
    print("  Ring SHAPE: C1 (faradaic) vs uniform-suppression null")
    print("=" * 74)
    print("  pH | data_peak@V | N0_peak@V | RMS(C1) | RMS(null) | shape winner")
    for ph in PHS:
        r = out["by_pH"][str(ph)]
        print(f"  {ph:>3} | {r['data_peak']:.3f}@{r['data_peak_V']:.2f} | "
              f"{r['n0_peak']:.3f}@{r['n0_peak_V']:.2f} | "
              f"{r['ring_shape_rms_C1']:.4f} | "
              f"{r['ring_shape_rms_uniform_null']:.4f} | {r['shape_winner']}")
    _plot(dig, out)
    print(f"\n  wrote {FIT / 'ring_shape_compare.json'}")
    return 0


def _plot(dig, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=False)
    for j, ph in enumerate(PHS):
        vn, rn = _n0_ring(ph)
        vc, rc = _c1_ring(ph)
        dv, dy = _data_ring(dig, ph)
        s = out["by_pH"][str(ph)]["null_scale"]
        ax = axes[j]
        ax.plot(dv, dy, "k-", lw=2, label="data")
        ax.plot(vn, rn, color="0.6", ls="--", label="N0 (unscaled)")
        ax.plot(vn, rn * s, "g-", label=f"uniform null (×{s:.2f})")
        ax.plot(vc, rc, "r-", label="C1 fit")
        ax.set_title(f"pH {ph} ring — shape compare")
        ax.set_xlabel("V_RHE (V)")
        ax.grid(alpha=0.3)
        if j == 0:
            ax.set_ylabel("ring mA/cm²")
            ax.legend(fontsize=8)
    fig.suptitle("Ring shape: does faradaic C1 or a uniform suppression match "
                 "the data? (both peak-matched)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIT / "ring_shape_compare.png", dpi=125)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
