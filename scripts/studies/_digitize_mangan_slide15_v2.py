#!/usr/bin/env python3
"""Precision re-extraction of the slide-15 Cs+ pH4 H2O2 current curve (v2).

Supersedes the 2026-05-07 hand-eyeballed 37-point digitization
(``data/mangan_deck_p15_h2o2_current.csv``).  Instead of reading pixels
off a screenshot, this parses the ORIGINAL VECTOR SVG of the figure:

    data/EChem Reactor Modeling-Seitz-Mangan/Yash-Trends/Results/
        experimental only.svg

which is the matplotlib-generated source of the deck "slide 15" plot
("Current density of H2O2 production", Cs2SO4, pH 4).  The magenta
dashed trace is stored as an absolute-coordinate path (id=path136,
stroke #bf00bf, no transforms), so the plotted experimental LSV samples
are recovered at SVG float precision (~4e-6 V equivalent) — exact for
all practical purposes, up to matplotlib's vertex path-simplification
(754 of the 1200 plotted vertices survive; dropped vertices deviate by
<= ~0.1 px from the retained polyline).

Provenance of the underlying data (from Yash's plotting.ipynb, cells
1-4, in "Data and Plotting.zip" alongside the SVG):

    Cs_df  = read_excel("Tafel slope analysis cation-pH-Li-K-Cs.xlsx",
                        sheet_name='Cs+')        # xlsx NOT on disk here
    v      = iR-corrected disk potential, 'V_RHE_iRdisk_LSV (V vs RHE)'
    j_ring = 'J_ring_LSV_bl' (baseline-corrected ring current density)
    j_ring[j_ring < 0.001] = 0                   # threshold -> exact 0s
    j_H2O2 = j_ring * 0.11 / (0.224 * 0.196)     # ring area / (N * disk area)
    one argmin outlier deleted; sorted by V; first 1200 points plotted
    plotted as -j_H2O2 (cathodic = negative)

Calibration: x ticks -0.50..+1.25 step 0.25 (markers at uniform 50.806
SVG units), y ticks 0.0..-0.6 step -0.1 (uniform 40.999 units).  Tick
uniformity is asserted; the affine is fit by least squares over all
ticks and residuals are reported (gate: < 1% of axis span).

Outputs
-------
- data/mangan_deck_p15_h2o2_current_v2_full.csv   all extracted vertices
- data/mangan_deck_p15_h2o2_current_v2.csv        binned fit target
    (<= 40 bins, per-bin median + scatter sigma, thresholded-zero flag)
- data/derived/slide15_digitization_qa.png        pixel-space overlay on
    the native-resolution PNG export (614x461; = SVG coords * 4/3)
- data/derived/slide15_v1_vs_v2.png               old eyeball vs v2

Run from PNPInverse/:  python scripts/studies/_digitize_mangan_slide15_v2.py
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SVG_PATH = (
    REPO
    / "data"
    / "EChem Reactor Modeling-Seitz-Mangan"
    / "Yash-Trends"
    / "Results"
    / "experimental only.svg"
)
PNG_PATH = SVG_PATH.with_name("experimental_only.png")
V1_CSV = REPO / "data" / "mangan_deck_p15_h2o2_current.csv"
OUT_FULL = REPO / "data" / "mangan_deck_p15_h2o2_current_v2_full.csv"
OUT_FIT = REPO / "data" / "mangan_deck_p15_h2o2_current_v2.csv"
QA_DIR = REPO / "data" / "derived"
QA_OVERLAY = QA_DIR / "slide15_digitization_qa.png"
QA_V1V2 = QA_DIR / "slide15_v1_vs_v2.png"

# Tick values confirmed visually against the rendered figure
# (writeups/WeekOfFeb25/assets/mangan_slide15.png and the PNG export).
X_TICK_VALUES = [-0.50, -0.25, 0.00, 0.25, 0.50, 0.75, 1.00, 1.25]
Y_TICK_VALUES = [0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6]

DATA_STROKE = "#bf00bf"  # matplotlib 'm' (0.75, 0, 0.75)
N_BINS_MAX = 40
SIGMA_FLOOR_MA_CM2 = 0.005  # floor for low-scatter bins (zero tail etc.)
SVG_TO_PNG = 96.0 / 72.0  # PNG exported at 96 dpi from a 72 dpi-pt canvas


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_ticks(svg: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_tick_positions, y_tick_positions) in SVG coords."""
    uses = re.findall(
        r'<use[^>]*xlink:href="#(m[0-9a-f]+)"[^>]*x="([\d.]+)"[^>]*y="([\d.]+)"',
        svg,
    )
    groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for href, x, y in uses:
        groups[href].append((float(x), float(y)))

    x_ticks = y_ticks = None
    for pts in groups.values():
        xs = sorted({p[0] for p in pts})
        ys = sorted({p[1] for p in pts})
        if len(ys) == 1 and len(xs) >= 2:  # bottom axis: shared y
            x_ticks = np.array(xs)
        elif len(xs) == 1 and len(ys) >= 2:  # left axis: shared x
            y_ticks = np.array(ys)
    if x_ticks is None or y_ticks is None:
        raise RuntimeError("Could not identify tick marker groups in SVG")
    return x_ticks, y_ticks


def assert_uniform(positions: np.ndarray, label: str, tol_frac: float = 1e-3) -> None:
    gaps = np.diff(positions)
    spread = (gaps.max() - gaps.min()) / gaps.mean()
    if spread > tol_frac:
        raise RuntimeError(f"{label} tick spacing non-uniform: {spread:.2e}")


def fit_affine(positions: np.ndarray, values: list[float]) -> tuple[float, float, float]:
    """Least-squares value = a + b*pos; returns (a, b, max_resid_in_value_units)."""
    A = np.vstack([np.ones_like(positions), positions]).T
    coef, *_ = np.linalg.lstsq(A, np.array(values), rcond=None)
    resid = np.abs(A @ coef - np.array(values)).max()
    return float(coef[0]), float(coef[1]), float(resid)


def parse_data_path(svg: str) -> np.ndarray:
    """Extract the magenta data path vertices (absolute M/L only)."""
    candidates = []
    for m in re.finditer(r"<path\b[^>]*?>", svg, flags=re.S):
        elem = m.group(0)
        style = re.search(r'style="([^"]*)"', elem)
        if not style or DATA_STROKE not in style.group(1).lower():
            continue
        d_attr = re.search(r'\bd="([^"]*)"', elem, flags=re.S)
        if d_attr:
            candidates.append(d_attr.group(1))
    if not candidates:
        raise RuntimeError(f"No path with stroke {DATA_STROKE} found")
    d = max(candidates, key=len)
    cmds = set(re.findall(r"[A-Za-z]", d))
    if not cmds <= {"M", "L"}:
        raise RuntimeError(f"Unexpected path commands {cmds}; only absolute M/L supported")
    pairs = re.findall(r"(-?\d+\.?\d*)\s+(-?\d+\.?\d*)", d)
    pts = np.array([[float(x), float(y)] for x, y in pairs])
    if len(pts) < 100:
        raise RuntimeError(f"Data path suspiciously short: {len(pts)} vertices")
    return pts


def main() -> None:
    svg = SVG_PATH.read_text()
    x_ticks, y_ticks = parse_ticks(svg)

    if len(x_ticks) != len(X_TICK_VALUES):
        raise RuntimeError(f"Expected {len(X_TICK_VALUES)} x ticks, got {len(x_ticks)}")
    if len(y_ticks) != len(Y_TICK_VALUES):
        raise RuntimeError(f"Expected {len(Y_TICK_VALUES)} y ticks, got {len(y_ticks)}")
    assert_uniform(x_ticks, "x")
    assert_uniform(y_ticks, "y")

    # SVG y increases downward; y_ticks sorted ascending = top (j=0) first.
    ax, bx, rx = fit_affine(x_ticks, X_TICK_VALUES)
    ay, by, ry = fit_affine(y_ticks, Y_TICK_VALUES)
    x_span = max(X_TICK_VALUES) - min(X_TICK_VALUES)
    y_span = max(Y_TICK_VALUES) - min(Y_TICK_VALUES)
    print(f"x affine: V = {ax:.6f} + {bx:.8f}*x   max resid {rx:.2e} V "
          f"({100 * rx / x_span:.4f}% of span)")
    print(f"y affine: j = {ay:.6f} + {by:.8f}*y   max resid {ry:.2e} mA/cm2 "
          f"({100 * ry / y_span:.4f}% of span)")
    if rx / x_span > 0.01 or ry / y_span > 0.01:
        raise RuntimeError("Calibration residual exceeds 1% of axis span")

    pts_svg = parse_data_path(svg)
    v = ax + bx * pts_svg[:, 0]
    j = ay + by * pts_svg[:, 1]
    order = np.argsort(v)
    v, j = v[order], j[order]
    pts_svg = pts_svg[order]
    print(f"extracted {len(v)} vertices; V in [{v.min():.4f}, {v.max():.4f}], "
          f"j in [{j.min():.4f}, {j.max():.4f}]")

    # --- full-resolution CSV ---
    svg_sha = _sha256(SVG_PATH)
    header_common = "\n".join([
        "# Slide-15 'Current density of H2O2 production' (Cs2SO4, pH 4, RRDE)",
        "# v2 precision extraction from ORIGINAL VECTOR SVG (not pixel eyeball)",
        f"# source_svg: {SVG_PATH.relative_to(REPO)}",
        f"# source_svg_sha256: {svg_sha}",
        "# data_path: id=path136, stroke #bf00bf, absolute coords, no transforms",
        f"# extraction_date: {date.today().isoformat()}",
        "# calibration: x ticks -0.50..1.25 step 0.25; y ticks 0.0..-0.6 step -0.1",
        f"# calibration_max_residual: x {rx:.2e} V, y {ry:.2e} mA/cm2",
        "# provenance (Yash plotting.ipynb): j_H2O2 = J_ring_LSV_bl*0.11/(0.224*0.196);",
        "#   j_ring<0.001 zeroed (exact-0 tail is THRESHOLDED, not measured 0);",
        "#   V = iR-corrected disk potential vs RHE; 1 outlier deleted; first 1200",
        "#   sorted points plotted; matplotlib path-simplification retains 754.",
        "# sign convention: cathodic negative (as plotted)",
    ])
    with OUT_FULL.open("w") as fh:
        fh.write(header_common + "\nV_RHE_V,j_h2o2_mA_cm2\n")
        for vi, ji in zip(v, j):
            fh.write(f"{vi:.5f},{ji:.5f}\n")
    print(f"wrote {OUT_FULL.relative_to(REPO)} ({len(v)} rows)")

    # --- binned fit target ---
    v_lo, v_hi = v.min(), v.max()
    n_bins = min(N_BINS_MAX, max(10, int(np.ceil((v_hi - v_lo) / 0.03))))
    edges = np.linspace(v_lo, v_hi, n_bins + 1)
    rows = []
    for k in range(n_bins):
        sel = (v >= edges[k]) & (v <= edges[k + 1] if k == n_bins - 1 else v < edges[k + 1])
        if sel.sum() < 2:
            continue
        vb = float(np.median(v[sel]))
        jb = float(np.median(j[sel]))
        # robust scatter: MAD -> sigma, then floor
        mad = float(np.median(np.abs(j[sel] - jb)))
        sigma = max(1.4826 * mad, SIGMA_FLOOR_MA_CM2)
        # SVG path values in the zeroed tail are -2e-5-ish (sub-pixel
        # wiggle), not exact 0.0; flag bins entirely within half a line
        # width of zero.
        thresholded = bool(np.all(np.abs(j[sel]) < 0.003))
        rows.append((vb, jb, sigma, int(sel.sum()), int(thresholded)))
    with OUT_FIT.open("w") as fh:
        fh.write(header_common + "\n")
        fh.write("# binned fit target: <=40 bins, per-bin median; sigma = max(1.4826*MAD,"
                 f" {SIGMA_FLOOR_MA_CM2}) [digitization/noise scatter, NOT experimental"
                 " replicate uncertainty — WLS chi2 is a relative metric only]\n")
        fh.write("# thresholded_zero=1: bin is in the j_ring<0.001 zeroed tail; treat as"
                 " 'onset constraint', not an exact measured zero\n")
        fh.write("V_RHE_V,j_h2o2_mA_cm2,sigma_mA_cm2,n_vertices,thresholded_zero\n")
        for vb, jb, sigma, n, flag in rows:
            fh.write(f"{vb:.5f},{jb:.5f},{sigma:.5f},{n},{flag}\n")
    print(f"wrote {OUT_FIT.relative_to(REPO)} ({len(rows)} bins)")

    # --- QA overlay in pixel space on the native PNG export ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    QA_DIR.mkdir(parents=True, exist_ok=True)
    img = np.asarray(Image.open(PNG_PATH))
    fig, axp = plt.subplots(figsize=(10, 7.5), dpi=130)
    axp.imshow(img)
    axp.plot(pts_svg[:, 0] * SVG_TO_PNG, pts_svg[:, 1] * SVG_TO_PNG,
             "c-", lw=0.6, alpha=0.9, label="v2 extracted path (overlay)")
    # binned fit target back-projected to pixel space
    vb_arr = np.array([r[0] for r in rows]); jb_arr = np.array([r[1] for r in rows])
    axp.plot((vb_arr - ax) / bx * SVG_TO_PNG, (jb_arr - ay) / by * SVG_TO_PNG,
             "ko", ms=3, mfc="yellow", label="binned fit target")
    axp.set_title("QA: v2 extraction overlaid on native PNG export (pixel space)")
    axp.legend(loc="lower right")
    axp.set_axis_off()
    fig.savefig(QA_OVERLAY, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {QA_OVERLAY.relative_to(REPO)}")

    # --- v1 vs v2 comparison ---
    v1_rows = [
        tuple(float(tok) for tok in line.split(","))
        for line in V1_CSV.read_text().splitlines()
        if line and not line.startswith("#") and not line[0].isalpha()
    ]
    v1 = {"V_RHE_V": np.array([r[0] for r in v1_rows]),
          "j_h2o2_mA_cm2": np.array([r[1] for r in v1_rows])}
    fig, axc = plt.subplots(figsize=(9, 6), dpi=130)
    axc.plot(v, j, "-", color="0.6", lw=0.8, label="v2 full (754 vertices, exact)")
    axc.errorbar(vb_arr, jb_arr, yerr=[r[2] for r in rows], fmt="o", ms=4,
                 color="crimson", label="v2 binned fit target")
    axc.plot(v1["V_RHE_V"], v1["j_h2o2_mA_cm2"], "s--", ms=4, color="royalblue",
             alpha=0.7, label="v1 eyeball (37 pts, 2026-05-07)")
    axc.set_xlabel("Applied Voltage (V vs RHE)")
    axc.set_ylabel("Peroxide Current Density (mA/cm$^2$)")
    axc.set_title("Slide-15 target: v1 eyeball vs v2 vector extraction")
    axc.legend()
    axc.grid(alpha=0.3)
    fig.savefig(QA_V1V2, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {QA_V1V2.relative_to(REPO)}")

    # v1 error stats on overlapping range
    v1v = np.asarray(v1["V_RHE_V"], dtype=float)
    v1j = np.asarray(v1["j_h2o2_mA_cm2"], dtype=float)
    sel = (v1v >= v_lo) & (v1v <= v_hi)
    j_interp = np.interp(v1v[sel], v, j)
    err = v1j[sel] - j_interp
    print(f"v1 eyeball error vs v2 (n={sel.sum()}): "
          f"mean {err.mean():+.4f}, max |err| {np.abs(err).max():.4f} mA/cm2")
    # headline features
    imin = int(np.argmin(j))
    print(f"v2 trough: j = {j[imin]:.4f} mA/cm2 at V = {v[imin]:.4f}")
    nz = np.where(j < -0.01)[0]
    print(f"v2 last |j|>0.01 at V = {v[nz[-1]]:.4f} (zero-tail starts after)")


if __name__ == "__main__":
    main()
