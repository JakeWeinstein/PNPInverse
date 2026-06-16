"""Phase 7.3 P0.2 — disk-onset extractor with thresholds + bootstrap CI.

Extracts the ORR DISK onset (the first-electron-transfer onset feature, S1)
from the digitized 3-panel disk curves (pH 2/4/6) at MULTIPLE small absolute
current thresholds {0.05, 0.10, 0.20} mA/cm² AFTER per-pH background
subtraction, with a feature-in-window guard and a digitization-uncertainty
bootstrap → onset-vs-pH ± CI on both the RHE and SHE axes.

Why this design (plan §2 P0.2 + brainstorm §7):
  * The digitized disk curves carry a pH-DEPENDENT high-V baseline
    (−0.099 / −0.076 / −0.057 mA/cm² at pH 2/4/6); raw small-threshold
    onsets are contaminated by it, so we subtract a robust per-pH baseline
    estimated from the most-anodic flat window first.
  * Digitized V is on a resampled near-uniform grid (point scatter ≈ 0), so
    the real onset uncertainty is the figure AXIS-READING error, not point
    jitter — bootstrapped via σ_V (V-axis) and σ_y (current) injection.
  * Onset DEFINITION uncertainty (threshold spread 0.05↔0.20) is reported
    separately; it is the dominant, most honest band here.
  * Only threshold-robustness is available (digitized = RHE axis only; no
    iR / +I·Rs / −I·Rs axis variants — those need the raw sheet).

The low-fidelity `metrics.json:exp_info` RING-onset scalars (6 pH) are
reported alongside as the broad qualitative trend ONLY (area-mixed, sheet).

Output: StudyResults/phase7p3_p0_2_onset/onset_metrics.json + onset_vs_pH.png

Run from PNPInverse/ (no Firedrake needed):
    python -u scripts/studies/phase7p3_p0_2_onset_extractor.py
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

# One Nernstian slope shared with the model E_eq shift (plan P0.1 / P0.2).
from scripts.studies.drivers.solver_demo_slide15_dual_pathway_cs import (
    _nernst_slope_v_per_ph,
)

DIGITIZED = (Path(_ROOT) / "StudyResults" / "phase7p2_ph_series_generalization"
             / "digitized_experimental_3panel.json")
METRICS = (Path(_ROOT) / "StudyResults" / "phase7p2_ph_series_generalization"
           / "metrics.json")
OUT = Path(_ROOT) / "StudyResults" / "phase7p3_p0_2_onset"

PH_DIGITIZED = {"2": 2.0, "4": 4.0, "6": 6.0}
THRESHOLDS = (0.05, 0.10, 0.20)   # mA/cm², post-background

BG_WINDOW_V = 0.04     # most-anodic ΔV used to estimate the disk baseline
BG_MIN_PTS = 5         # minimum points required in the baseline window
EDGE_GUARD_V = 0.010   # onset within this of Vmax is flagged (window edge)
SIGMA_V = 0.005        # V-axis reading uncertainty (line thickness / cal)
SIGMA_Y_ABS = 0.015    # absolute current-reading σ near onset (mA/cm²);
#                        NOT a fraction of full-scale — the onset thresholds
#                        are O(0.05–0.2), so a full-scale fraction (~0.08)
#                        would swamp them and pin the bootstrap at the edge.
N_BOOT = 3000
SEED = 20260614


def _sorted_curve(panel):
    v = np.asarray(panel["v_rhe"], float)
    y = np.asarray(panel["value"], float)
    o = np.argsort(v)
    return v[o], y[o]


def _baseline(v, y):
    """Robust disk baseline from the most-anodic flat window."""
    hi = v >= (v.max() - BG_WINDOW_V)
    if hi.sum() < BG_MIN_PTS:
        return None, None, int(hi.sum())
    return float(np.median(y[hi])), float(np.std(y[hi])), int(hi.sum())


def _onset_at(v, y_corr, thr):
    """Onset = turn-on of the main cathodic wave CONNECTED to the cathodic
    plateau (Vmin), so isolated high-V digitization blips past a sub-threshold
    gap are ignored.  Walk anodically from Vmin while −y_corr stays ≥ thr; the
    break point's interpolated crossing is the onset.  None if the plateau
    itself is below thr (wave never reaches it)."""
    c = -y_corr                       # cathodic-positive, large at low V (Vmin)
    n = len(c)
    if n == 0 or c[0] < thr:          # cathodic plateau below thr -> unreached
        return None
    i = 0
    while i + 1 < n and c[i + 1] >= thr:
        i += 1
    if i == n - 1:
        return float(v[i])            # wave above thr across the whole window
    v0, v1 = v[i], v[i + 1]           # V increasing; c0>=thr>c1
    c0, c1 = c[i], c[i + 1]
    if c0 == c1:
        return float(v0)
    return float(v0 + (thr - c0) * (v1 - v0) / (c1 - c0))


def _extract_point(v, y, thr, *, baseline):
    y_corr = y - baseline
    onset = _onset_at(v, y_corr, thr)
    edge_flag = onset is not None and onset >= v.max() - EDGE_GUARD_V
    reached = onset is not None
    return {"onset_rhe": onset, "edge_flag": bool(edge_flag),
            "reached": bool(reached)}


def _bootstrap_onset(v, y, thr, *, sigma_v, sigma_y, rng, n=N_BOOT):
    """Inject V-axis + current digitization noise, re-estimate baseline and
    onset each draw → median + 16/84 percentile CI (np.nan if often unreached)."""
    out = np.full(n, np.nan)
    for b in range(n):
        vb = v + rng.normal(0.0, sigma_v, size=v.shape)
        yb = y + rng.normal(0.0, sigma_y, size=y.shape)
        ob = np.argsort(vb)
        vb, yb = vb[ob], yb[ob]
        base_b, _, npb = _baseline(vb, yb)
        if base_b is None:
            continue
        o = _onset_at(vb, yb - base_b, thr)
        if o is not None:
            out[b] = o
    good = out[np.isfinite(out)]
    if good.size < 0.5 * n:
        return None
    return {
        "median": float(np.median(good)),
        "ci16": float(np.percentile(good, 16)),
        "ci84": float(np.percentile(good, 84)),
        "frac_reached": float(good.size / n),
    }


def _ols_slope(ph, onset):
    """Slope d(onset)/dpH with a residual-based standard error (n is small —
    interpret as descriptive, not inferential; threshold-robustness is the
    real evidence)."""
    ph = np.asarray(ph, float)
    on = np.asarray(onset, float)
    m = np.isfinite(on)
    ph, on = ph[m], on[m]
    if ph.size < 2:
        return None
    A = np.vstack([ph, np.ones_like(ph)]).T
    coef, *_ = np.linalg.lstsq(A, on, rcond=None)
    slope, intercept = float(coef[0]), float(coef[1])
    resid = on - (slope * ph + intercept)
    if ph.size > 2:
        s2 = float(resid @ resid) / (ph.size - 2)
        se = float(np.sqrt(s2 / np.sum((ph - ph.mean()) ** 2)))
    else:
        se = None
    ss_tot = float(np.sum((on - on.mean()) ** 2))
    r2 = float(1.0 - (resid @ resid) / ss_tot) if ss_tot > 0 else None
    return {"slope_v_per_ph": slope, "intercept_v": intercept,
            "se_v_per_ph": se, "r2": r2, "n": int(ph.size)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    S = _nernst_slope_v_per_ph()
    dig = json.load(open(DIGITIZED))

    per_pH = {}
    for key, ph in PH_DIGITIZED.items():
        v, y = _sorted_curve(dig["disk"][key])
        base, base_std, n_bg = _baseline(v, y)
        sigma_y = max(SIGMA_Y_ABS, base_std or 0.0)
        rec = {"pH": ph, "n_pts": int(v.size),
               "v_window": [float(v.min()), float(v.max())],
               "baseline_mA_cm2": base, "baseline_std": base_std,
               "n_baseline_pts": n_bg, "sigma_v": SIGMA_V, "sigma_y": sigma_y,
               "by_threshold": {}}
        for thr in THRESHOLDS:
            pt = _extract_point(v, y, thr, baseline=base)
            boot = _bootstrap_onset(v, y, thr, sigma_v=SIGMA_V,
                                    sigma_y=sigma_y, rng=rng)
            onset_rhe = pt["onset_rhe"]
            rec["by_threshold"][f"{thr:.2f}"] = {
                "onset_rhe": onset_rhe,
                "onset_she": (None if onset_rhe is None
                              else onset_rhe - S * ph),
                "edge_flag": pt["edge_flag"],
                "reached": pt["reached"],
                "bootstrap": boot,
            }
        per_pH[key] = rec

    # Onset-vs-pH slope per threshold (RHE + SHE), digitized pH 2/4/6 only.
    slopes = {}
    for thr in THRESHOLDS:
        tk = f"{thr:.2f}"
        phs, on_rhe, on_she = [], [], []
        for key, ph in PH_DIGITIZED.items():
            t = per_pH[key]["by_threshold"][tk]
            # drop edge-flagged points from the slope (window-edge artifact)
            o = None if t["edge_flag"] else t["onset_rhe"]
            phs.append(ph)
            on_rhe.append(o)
            on_she.append(None if o is None else o - S * ph)
        # pH-2-excluded robustness slope (pH 2 is the noisiest digitized
        # curve and dominates the 3-pt fit; pH 4 & 6 are clean + sustained).
        phs_x = [p for p in phs if p != 2.0]
        on_rhe_x = [o for p, o in zip(phs, on_rhe) if p != 2.0]
        on_she_x = [o for p, o in zip(phs, on_she) if p != 2.0]
        slopes[tk] = {
            "rhe": _ols_slope(phs, on_rhe),
            "she": _ols_slope(phs, on_she),
            "rhe_excl_pH2": _ols_slope(phs_x, on_rhe_x),
            "used_pH": [p for p, o in zip(phs, on_rhe) if o is not None],
        }

    # Low-fidelity Exp Info RING-onset trend (6 pH) — qualitative context only.
    exp_info_ring = None
    try:
        mj = json.load(open(METRICS))
        rows = {float(k): val.get("exp_info", {})
                for k, val in mj.get("by_pH", {}).items()
                if isinstance(val, dict) and val.get("exp_info")}
        phs = sorted(rows)
        on = [rows[p].get("onset") for p in phs]
        exp_info_ring = {
            "definition": "ring onset @ j_ring=0.01 mA/cm² (sheet, area-mixed)",
            "pH": phs, "onset_rhe": on,
            "slope_fit": _ols_slope(phs, on),
            "note": "LOW-CONFIDENCE: derived ring scalar, not disk; "
                    "broad-trend context only (plan §1).",
        }
    except Exception as exc:  # pragma: no cover
        exp_info_ring = {"error": str(exc)}

    # Threshold-robustness verdict: does the disk-onset RHE slope keep a
    # consistent (positive) SIGN across all thresholds with reached points?
    rhe_slopes = [slopes[f"{t:.2f}"]["rhe"]["slope_v_per_ph"]
                  for t in THRESHOLDS
                  if slopes[f"{t:.2f}"]["rhe"] is not None]
    sign_consistent = (len(rhe_slopes) >= 2
                       and (all(s > 0 for s in rhe_slopes)
                            or all(s < 0 for s in rhe_slopes)))
    # Is the slope sign STABLE to dropping pH 2 (the noisy point)? And does
    # the disk-onset sign AGREE with the ring-onset sign? If either fails,
    # onset is not a robust mechanism discriminator (plan: onset = a gauge
    # question; the frame-invariant RING MAGNITUDE is load-bearing -> M3/C).
    rhe_slopes_x = [slopes[f"{t:.2f}"]["rhe_excl_pH2"]["slope_v_per_ph"]
                    for t in THRESHOLDS
                    if slopes[f"{t:.2f}"]["rhe_excl_pH2"] is not None]
    sign_stable_drop_pH2 = (len(rhe_slopes) and len(rhe_slopes_x)
                            and all((a > 0) == (b > 0)
                                    for a, b in zip(rhe_slopes, rhe_slopes_x)))
    ring_slope = None
    if isinstance(exp_info_ring, dict) and exp_info_ring.get("slope_fit"):
        ring_slope = exp_info_ring["slope_fit"]["slope_v_per_ph"]
    disk_ring_sign_agree = (ring_slope is not None and len(rhe_slopes)
                            and all((s > 0) == (ring_slope > 0)
                                    for s in rhe_slopes))
    onset_is_robust_discriminator = bool(
        sign_consistent and sign_stable_drop_pH2 and disk_ring_sign_agree)

    result = {
        "test": "phase7p3_P0.2_disk_onset_extractor",
        "nernst_slope_v_per_ph": S,
        "thresholds_mA_cm2": list(THRESHOLDS),
        "config": {"bg_window_v": BG_WINDOW_V, "sigma_v": SIGMA_V,
                   "sigma_y_abs": SIGMA_Y_ABS, "n_boot": N_BOOT,
                   "edge_guard_v": EDGE_GUARD_V, "seed": SEED},
        "per_pH": per_pH,
        "onset_vs_pH_slope": slopes,
        "rhe_slope_sign_consistent_across_thresholds": bool(sign_consistent),
        "rhe_slopes_v_per_ph": rhe_slopes,
        "rhe_slopes_excl_pH2_v_per_ph": rhe_slopes_x,
        "sign_stable_to_dropping_pH2": bool(sign_stable_drop_pH2),
        "disk_ring_onset_sign_agree": bool(disk_ring_sign_agree),
        "onset_is_robust_discriminator": onset_is_robust_discriminator,
        "exp_info_ring_trend_lowconf": exp_info_ring,
    }
    with open(OUT / "onset_metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    _plot(per_pH, slopes, S)

    # Console summary.
    print("=" * 78)
    print("  Phase 7.3 P0.2 — disk onset vs pH (digitized pH 2/4/6)")
    print(f"  Nernst slope S = {S:.6f} V/pH")
    print("=" * 78)
    for key, ph in PH_DIGITIZED.items():
        r = per_pH[key]
        print(f"  pH {ph}: baseline={r['baseline_mA_cm2']:.4f} "
              f"(±{r['baseline_std']:.4f}, n={r['n_baseline_pts']}), "
              f"window V=[{r['v_window'][0]:.3f},{r['v_window'][1]:.3f}]")
        for thr in THRESHOLDS:
            t = r["by_threshold"][f"{thr:.2f}"]
            o = t["onset_rhe"]
            b = t["bootstrap"]
            ci = (f"[{b['ci16']:.3f},{b['ci84']:.3f}]" if b else "n/a")
            flag = " EDGE!" if t["edge_flag"] else ""
            print(f"      thr {thr:.2f}: onset_RHE="
                  f"{'none' if o is None else f'{o:.3f}'} "
                  f"onset_SHE={'none' if o is None else f'{o - S*ph:.3f}'} "
                  f"CI(RHE)={ci}{flag}")
    print("  --- onset-vs-pH slope (edge-flagged pts dropped) ---")
    for thr in THRESHOLDS:
        tk = f"{thr:.2f}"
        sr, ss = slopes[tk]["rhe"], slopes[tk]["she"]
        if sr is None:
            print(f"      thr {tk}: <2 usable pts")
            continue
        se = "" if sr["se_v_per_ph"] is None else f"±{sr['se_v_per_ph']*1000:.0f}"
        print(f"      thr {tk}: RHE {sr['slope_v_per_ph']*1000:+.0f}{se} mV/pH "
              f"(R²={sr['r2']}) | SHE {ss['slope_v_per_ph']*1000:+.0f} mV/pH "
              f"| pts pH {slopes[tk]['used_pH']}")
    print(f"  RHE-slope sign consistent across thresholds: {sign_consistent}")
    print(f"  RHE slopes excl. pH2 (mV/pH): "
          f"{[round(s*1000) for s in rhe_slopes_x]}  "
          f"-> sign stable to dropping pH2: {sign_stable_drop_pH2}")
    if isinstance(exp_info_ring, dict) and exp_info_ring.get("slope_fit"):
        e = exp_info_ring["slope_fit"]
        print(f"  [low-conf] Exp Info RING onset (6 pH): "
              f"{e['slope_v_per_ph']*1000:+.0f} mV/pH RHE (R²={e['r2']:.2f})")
        print(f"  disk-onset vs ring-onset sign agree: {disk_ring_sign_agree}")
    print(f"  ==> ONSET IS A ROBUST MECHANISM DISCRIMINATOR: "
          f"{onset_is_robust_discriminator}")
    if not onset_is_robust_discriminator:
        print("      (expected per plan §3 — onset is a gauge question; the "
              "frame-invariant RING MAGNITUDE is load-bearing -> M3/C)")
    print(f"\n  wrote {OUT / 'onset_metrics.json'}")
    return 0


def _plot(per_pH, slopes, S):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {0.05: "tab:blue", 0.10: "tab:green", 0.20: "tab:red"}
    for ax, axis in zip(axes, ("rhe", "she")):
        for thr in THRESHOLDS:
            tk = f"{thr:.2f}"
            phs, on = [], []
            for key, ph in PH_DIGITIZED.items():
                t = per_pH[key]["by_threshold"][tk]
                o = t["onset_rhe"]
                if o is None or t["edge_flag"]:
                    continue
                phs.append(ph)
                on.append(o if axis == "rhe" else o - S * ph)
            if phs:
                ax.plot(phs, on, "o-", color=colors[thr],
                        label=f"thr {thr:.2f} mA/cm²")
            sl = slopes[tk][axis]
            if sl is not None:
                xx = np.array([1.5, 6.5])
                ax.plot(xx, sl["slope_v_per_ph"] * xx + sl["intercept_v"],
                        "--", color=colors[thr], alpha=0.4)
        ax.set_xlabel("bulk pH")
        ax.set_ylabel(f"disk onset V_{axis.upper()} (V)")
        ax.set_title(f"Disk onset vs pH — {axis.upper()} axis")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "onset_vs_pH.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
