"""Overlay frozen-θ_L water-route MODEL on the DIGITIZED experimental
3-panel ACS figure (Brianna Ruggiero, 0.1 M K₂SO₄ RRDE).

Two halves:
  1. Digitize the raster `ACS_experimental_3panel_reference.png` by
     color segmentation (each bulk-pH curve is a distinct MATLAB-default
     colour). Axes were calibrated from the frame/tick pixels:
       * x (shared): V_RHE 0.0 (px 247) → 0.8 (px 1045)
       * selectivity panel y∈[6,306]:  0 %  (px 306) → 60 % (px 34)
       * ring panel        y∈[415,730]: 0.0 (px 730) → 0.30 (px 415) mA/cm²
       * disk panel        y∈[730,1045]:0.0 (px 730) → −6.0 (px 1044) mA/cm²
  2. Load the model iv_curve_pH{2,4,6}.json and compute the same three
     observables on the v_rhe_deck (RHE) axis, using the EXACT formulas
     from phase7p2_ph_series_generalization.py:
       ring jr = -pc·N·A_disk/A_ring ;  disk = cd ;
       selectivity = 100·|pc| / (|pc| + max(0,(|cd|-|pc|)/2)).

Output: StudyResults/phase7p2_ph_series_generalization/
  model_vs_data_3panel_overlay.png  (experimental solid + markers,
  model dashed, matched per-pH colours) and
  digitized_experimental_3panel.json (the pulled data, persisted).

Run from PNPInverse/ (no Firedrake needed; pure post-processing).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[2]
OUT = _ROOT / "StudyResults" / "phase7p2_ph_series_generalization"
IMG = OUT / "ACS_experimental_3panel_reference.png"

# ── ring/selectivity geometry (matches the generalization script) ──
A_DISK_CM2, A_RING_CM2, N_COLL = 0.19635, 0.109956, 0.224

# ── digitization calibration (pixels → data) ──────────────────────
X0_PX, X1_PX, V0, V1 = 247, 1045, 0.0, 0.8            # shared x-axis
PANELS = {
    "sel":  dict(y0=6,   y1=306,  cal=((306, 0.0), (34, 60.0))),
    "ring": dict(y0=415, y1=730,  cal=((730, 0.0), (415, 0.30))),
    "disk": dict(y0=730, y1=1045, cal=((730, 0.0), (1044, -6.0))),
}
# MATLAB-default curve colours read off the legend swatches.
REFS = {
    2:  (163, 20, 46),    # dark red
    4:  (255, 105, 41),   # orange
    6:  (237, 177, 32),   # gold
    10: (119, 172, 48),   # green   (not overlaid — model failed alkaline)
    12: (0, 114, 189),    # blue    (   "" )
}
PH_OVERLAY = [2, 4, 6]
COLOR = {2: "#A3142E", 4: "#FF6929", 6: "#E0A800"}     # plotting colours
# disk-panel legend box to exclude from segmentation (px)
LEGEND_BOX = dict(x0=690, x1=1015, y0=898, y1=1040)

SAT_MIN = 45          # min (max-min) channel spread to count as "coloured"
TOL = 78              # max RGB distance to a reference colour


def _xpix_to_v(x: np.ndarray) -> np.ndarray:
    return V0 + (x - X0_PX) * (V1 - V0) / (X1_PX - X0_PX)


def _ypix_to_val(y: np.ndarray, panel: str) -> np.ndarray:
    (ya, va), (yb, vb) = PANELS[panel]["cal"]
    return va + (y - ya) * (vb - va) / (yb - ya)


def _classify(arr: np.ndarray) -> np.ndarray:
    """Per-pixel nearest-reference label (-1 = none). arr: (H,W,3) int."""
    sat = arr.max(2) - arr.min(2)
    coloured = sat >= SAT_MIN
    labels = np.full(arr.shape[:2], -1, dtype=int)
    best = np.full(arr.shape[:2], 1e9)
    for ph, c in REFS.items():
        d = np.sqrt(((arr - np.array(c)) ** 2).sum(2))
        take = coloured & (d < TOL) & (d < best)
        labels[take] = ph
        best[take] = d[take]
    return labels


def digitize(arr: np.ndarray) -> dict:
    """Return {panel: {ph: (V[], val[])}} by column-median segmentation."""
    labels = _classify(arr)
    out: dict = {}
    for panel, cfg in PANELS.items():
        y0, y1 = cfg["y0"] + 2, cfg["y1"] - 2
        out[panel] = {}
        for ph in PH_OVERLAY:
            vs, vals = [], []
            for x in range(X0_PX + 1, X1_PX):
                col = labels[y0:y1, x]
                rows = np.where(col == ph)[0]
                if rows.size == 0:
                    continue
                if panel == "disk":  # drop legend-box hits
                    yabs = y0 + rows
                    keep = ~((x >= LEGEND_BOX["x0"]) & (x <= LEGEND_BOX["x1"])
                             & (yabs >= LEGEND_BOX["y0"])
                             & (yabs <= LEGEND_BOX["y1"]))
                    rows = rows[keep]
                    if rows.size == 0:
                        continue
                vs.append(x)
                vals.append(y0 + np.median(rows))
            if not vs:
                continue
            V = _xpix_to_v(np.array(vs))
            val = _ypix_to_val(np.array(vals), panel)
            out[panel][ph] = (V, val)
    return out


def model_curves(ph_float: float) -> dict:
    d = json.load(open(OUT / f"iv_curve_pH{ph_float}.json"))
    v = np.array(d["v_rhe_deck"], float)
    cd = np.array([c if c is not None else np.nan for c in d["cd_mA_cm2"]], float)
    pc = np.array([p if p is not None else np.nan for p in d["pc_mA_cm2"]], float)
    o = np.argsort(v)
    v, cd, pc = v[o], cd[o], pc[o]
    jr = -pc * N_COLL * A_DISK_CM2 / A_RING_CM2
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.abs(pc)
        r4 = np.maximum(0.0, (np.abs(cd) - np.abs(pc)) / 2.0)
        sel = 100.0 * r2 / (r2 + r4)
    # selectivity is meaningless once the ORR current vanishes (0/0 →
    # climbs to ~73%); restrict to measurable current, as the metrics
    # script does (|cd| > 0.02 mA/cm²) — this also matches the V-domain
    # over which the experimental selectivity curves actually exist.
    sel = np.where(np.abs(cd) > 0.02, sel, np.nan)
    return {"v": v, "sel": sel, "ring": jr, "disk": cd}


def _v_to_xpix(v: np.ndarray) -> np.ndarray:
    return X0_PX + (np.asarray(v) - V0) * (X1_PX - X0_PX) / (V1 - V0)


def _val_to_ypix(val: np.ndarray, panel: str) -> np.ndarray:
    (ya, va), (yb, vb) = PANELS[panel]["cal"]
    return ya + (np.asarray(val) - va) * (yb - ya) / (vb - va)


def write_verification(arr: np.ndarray, data: dict, dest: Path) -> None:
    """Re-project digitized points onto the source raster to prove the
    color segmentation + axis calibration are correct."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, a = plt.subplots(figsize=(8, 9))
    a.imshow(arr.astype("uint8"))
    for panel, d in data.items():
        for ph, (V, val) in d.items():
            a.plot(_v_to_xpix(V), _val_to_ypix(val, panel), ".",
                   ms=1.6, color="black")
    a.set_title("digitized points (black) re-projected onto source image",
                fontsize=10)
    a.axis("off")
    fig.tight_layout()
    fig.savefig(dest, dpi=150)
    print("wrote", dest)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = np.asarray(Image.open(IMG).convert("RGB")).astype(int)
    data = digitize(arr)

    # persist the pulled experimental data
    dump = {p: {str(ph): {"v_rhe": V.tolist(), "value": val.tolist()}
                for ph, (V, val) in d.items()} for p, d in data.items()}
    (OUT / "digitized_experimental_3panel.json").write_text(json.dumps(dump, indent=1))

    write_verification(arr, data, OUT / "digitization_verification.png")

    models = {ph: model_curves(float(ph)) for ph in PH_OVERLAY}

    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(3, 1, figsize=(7.2, 10.5), sharex=True)
    panels = [("sel", "H$_2$O$_2$ Selectivity (%)", (0, 80)),
              ("ring", "Ring Current\nDensity (mA/cm$^2$)", (0, 0.40)),
              ("disk", "Disk Current\nDensity (mA/cm$^2$)", (-5.0, 0.3))]
    for a, (key, ylab, ylim) in zip(ax, panels):
        for ph in PH_OVERLAY:
            c = COLOR[ph]
            if ph in data[key]:
                V, val = data[key][ph]
                a.plot(V, val, "-", color=c, lw=1.2, alpha=0.6)
            m = models[ph]
            a.plot(m["v"], m[key], "--", color=c, lw=2.2, marker="o", ms=3.5)
        a.set_ylabel(ylab)
        a.set_ylim(*ylim)
        a.grid(alpha=0.25)
    ax[0].set_title("Frozen-θ$_L$ water-route MODEL vs experimental ACS data\n"
                    "K$_2$SO$_4$ RRDE — bulk pH 2 / 4 / 6", fontsize=10)

    ph_keys = [Line2D([], [], color=COLOR[ph], lw=3, label=f"pH {ph}")
               for ph in PH_OVERLAY]
    style_keys = [
        Line2D([], [], color="0.35", lw=1.4, alpha=0.7,
               label="experimental (digitized)"),
        Line2D([], [], color="0.35", lw=2.2, ls="--", marker="o", ms=4,
               label="frozen-θ$_L$ model"),
    ]
    leg1 = ax[0].legend(handles=ph_keys, fontsize=8, ncol=3,
                        loc="upper left", framealpha=0.9, title="colour = pH")
    ax[0].add_artist(leg1)
    ax[0].legend(handles=style_keys, fontsize=8, loc="lower right",
                 framealpha=0.9)
    ax[-1].set_xlabel("Potential (V vs. RHE)")
    ax[-1].set_xlim(0.0, 0.8)
    fig.tight_layout()
    dest = OUT / "model_vs_data_3panel_overlay.png"
    fig.savefig(dest, dpi=160)
    print("wrote", dest)
    print("wrote", OUT / "digitized_experimental_3panel.json")
    # quick coverage report
    for key, _, _ in panels:
        for ph in PH_OVERLAY:
            n = len(data[key].get(ph, ([],))[0]) if ph in data[key] else 0
            print(f"  {key:4s} pH{ph}: {n} digitized pts")


if __name__ == "__main__":
    main()
