"""Generate two slide-ready PNGs for the May 2026 group meeting.

Outputs (all saved next to this script):

    voltage_window.png       IV curve with annotated convergence regions.
    repo_dashboard.png       4-panel stats: commits, files, lines, AI lines.

Run from the repo root:

    python writeups/WeekOfMay4/make_slide_plots.py
"""
from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent

V13_COMMIT = "27b8223"  # "v13 pipeline and paper v1"
SINCE = "2026-03-04"

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})


# ---------------------------------------------------------------------------
# Plot 1: voltage-window expansion
# ---------------------------------------------------------------------------

def _read_iv_csv(path: Path) -> dict[str, np.ndarray]:
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    def _col(name: str) -> np.ndarray:
        out = []
        for r in rows:
            raw = r.get(name, "")
            try:
                out.append(float(raw)) if raw not in ("", None) else out.append(np.nan)
            except ValueError:
                out.append(np.nan)
        return np.asarray(out, dtype=float)

    return {
        "v": _col("V_RHE"),
        "cd": _col("cd_mA_cm2"),
        "pc": _col("pc_mA_cm2"),
        "method": np.asarray([r.get("method", "") for r in rows]),
    }


def _read_2b_stern_pass() -> dict[str, np.ndarray]:
    """Load the 3sp + Bikerman + muh + Stern 0.10 F/m^2 pass from May 4."""
    import json
    path = REPO / "StudyResults/peroxide_window_3sp_bikerman_muh_2b/iv_curve.json"
    with path.open() as f:
        bundle = json.load(f)
    stern = next(r for r in bundle["reports"]
                 if r["label"].endswith("stern_0p10_clip50"))
    v = np.asarray(stern["v_rhe"], dtype=float)
    cd = np.asarray([
        np.nan if x is None else float(x) for x in stern["cd_mA_cm2"]
    ])
    pc = np.asarray([
        np.nan if x is None else float(x) for x in stern["pc_mA_cm2"]
    ])
    converged = np.asarray(stern["converged"], dtype=bool)
    return {"v": v, "cd": cd, "pc": pc, "converged": converged}


def make_voltage_window_plot() -> Path:
    """Reach-ladder showing which solver variant reaches which voltage."""
    out = OUT_DIR / "voltage_window.png"

    # (label, v_min, v_max, color, annotation)
    # Ordered top-to-bottom in chronological order so reading top-down
    # tells the story.
    rows = [
        (
            "Pre-rebuild\n(4sp concentration)",
            -0.50, 0.10, "#7d7d7d",
            "cathodic edge only",
        ),
        (
            "Apr 27 rebuild\n(3sp+Boltzmann+log-c+log-rate)",
            -0.50, 0.60, "#2ca02c",
            "+0.60 V",
        ),
        (
            "May 4 production, no Stern\n(+ Bikerman-IC)",
            -0.50, 1.00, "#1f77b4",
            "14/15 to +1.00 V",
        ),
        (
            "May 4 production + Stern $0.10$ F/m$^2$\n(+ Bikerman-IC)",
            -0.50, 1.00, "#9467bd",
            "15/15 to +1.00 V",
        ),
        (
            "Log-c + steric + Debye-Boltz.\n(warm-walk extension)",
            0.20, 2.00, "#ff7f0e",
            "demonstrated +2.00 V",
        ),
    ]
    n = len(rows)

    fig, ax = plt.subplots(figsize=(12.0, 4.6), dpi=200)

    # R2 unclip threshold
    ax.axvline(0.495, color="#c8651b", linestyle=":", linewidth=1.4,
               zorder=1)

    bar_h = 0.50
    for i, (label, vmin, vmax, color, note) in enumerate(rows):
        y = n - 1 - i  # top row = newest? No — keep chronological top-down
        y = i  # row 0 at top
        # invert so row 0 (oldest) is at the bottom and the latest at the top
    # Re-iterate with the flipped y: latest at top, earliest at bottom.
    for i, (label, vmin, vmax, color, note) in enumerate(rows):
        y = i
        ax.barh(y, vmax - vmin, left=vmin, height=bar_h,
                color=color, edgecolor="black", linewidth=0.8,
                alpha=0.85, zorder=3)
        # Reach marker on the right end (where the solver stopped).
        ax.plot([vmax], [y], marker="o", markersize=10,
                markerfacecolor=color, markeredgecolor="black",
                markeredgewidth=0.9, zorder=4)
        # Annotation text just to the right of the marker.
        ax.text(vmax + 0.04, y, note, va="center", ha="left",
                fontsize=10, color="#222")

    # R2 unclip label up near the top (above all bars)
    ax.text(0.495, n - 0.4,
            r"R$_2$ unclips" "\n" r"at $+0.495$ V",
            color="#c8651b", fontsize=9, ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                      edgecolor="#c8651b", linewidth=0.6),
            zorder=5)

    # Kinetic dead-zone shading
    ax.axvspan(0.5, 2.10, color="#fff4c2", alpha=0.30, zorder=0)
    ax.text(1.55, -0.85,
            r"Kinetic dead zone above $\sim +0.5$ V" "\n"
            r"$|I|\sim 10^{-12}$--$10^{-16}$ mA/cm$^2$",
            color="#7a5d00", fontsize=9, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#fffae0",
                      edgecolor="#b8860b", linewidth=0.6))

    # Y-axis: row labels
    ax.set_yticks(range(n))
    ax.set_yticklabels([row[0] for row in rows], fontsize=10)
    ax.invert_yaxis()  # row 0 (oldest) at top, latest at bottom

    # X-axis
    ax.set_xlabel(r"Converged voltage range $V_{\mathrm{RHE}}$ (V)",
                  fontsize=11)
    ax.set_xlim(-0.7, 2.30)
    ax.set_xticks(np.arange(-0.5, 2.01, 0.25))
    ax.tick_params(axis="x", labelsize=9)

    # Reference voltage tick labels (V=0)
    ax.axvline(0, color="0.5", linewidth=0.6, zorder=1)

    # Title
    ax.set_title("Forward-solver reach: which version converges where",
                 fontsize=12, pad=10)

    # Tidy
    ax.set_ylim(n - 0.5 + 0.3, -0.5 - 0.7)  # extra room for annotations
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)
    for spine in ("right", "top"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO)}")
    return out


# ---------------------------------------------------------------------------
# Plot 2: repo dashboard
# ---------------------------------------------------------------------------

def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO).decode().strip()


def _commit_count(since: str) -> int:
    return int(_git("rev-list", "--count", f"--since={since}", "HEAD"))


def _shortstat_totals(since: str) -> tuple[int, int, int]:
    log = _git("log", f"--since={since}", "--shortstat",
               "--no-merges", "--pretty=format:")
    files = ins = dels = 0
    for line in log.splitlines():
        line = line.strip()
        if not line or "|" in line:
            continue
        if "file" not in line:
            continue
        parts = [p.strip() for p in line.split(",")]
        for p in parts:
            tokens = p.split()
            if not tokens:
                continue
            n = int(tokens[0])
            if "file" in p:
                files += n
            elif "insertion" in p:
                ins += n
            elif "deletion" in p:
                dels += n
    return files, ins, dels


def _wc(paths: Iterable[Path]) -> int:
    total = 0
    for p in paths:
        try:
            with p.open(encoding="utf-8", errors="ignore") as f:
                total += sum(1 for _ in f)
        except OSError:
            pass
    return total


def make_repo_dashboard() -> Path:
    out = OUT_DIR / "repo_dashboard.png"

    commits = _commit_count(SINCE)
    files_changed, ins, dels = _shortstat_totals(SINCE)

    py_files = sorted((REPO).rglob("*.py"))
    py_files = [p for p in py_files
                if not any(part in {"venv", "venv-firedrake",
                                    "__pycache__", ".git", "archive"}
                           for part in p.parts)]
    n_py_total = len(py_files)
    py_lines_total = _wc(py_files)

    bvsolver_files = sorted((REPO / "Forward/bv_solver").glob("*.py"))
    bvsolver_lines = _wc(bvsolver_files)

    studies_files = sorted((REPO / "scripts/studies").glob("*.py"))
    studies_lines = _wc(studies_files)

    test_files = sorted((REPO / "tests").rglob("test_*.py"))
    test_lines = _wc(test_files)

    docs_files = sorted((REPO / "docs").glob("*.md"))
    docs_lines = _wc(docs_files)

    handoff_files = sorted((REPO / "docs").glob("CHATGPT_HANDOFF*.md"))
    n_handoffs = len(handoff_files)

    studyresults_dirs = [p for p in (REPO / "StudyResults").iterdir()
                         if p.is_dir()]
    n_studyresult_dirs = len(studyresults_dirs)

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.0), dpi=200)
    fig.suptitle(f"Repo activity since the v13 paper "
                 f"(2026-03-04 → today, {commits} commits)",
                 fontsize=14, fontweight="bold")

    # Panel A: top-line counts
    ax = axes[0, 0]
    items = [
        ("Commits", commits),
        ("Files changed", files_changed),
        ("Study scripts", len(studies_files)),
        ("Study result\ndirectories", n_studyresult_dirs),
        ("Test files", len(test_files)),
        ("ChatGPT\nhandoff docs", n_handoffs),
    ]
    labels = [a for a, _ in items]
    values = [b for _, b in items]
    colors = ["#1f77b4", "#1f77b4", "#2ca02c", "#2ca02c",
              "#ff7f0e", "#9467bd"]
    bars = ax.barh(labels, values, color=colors, alpha=0.85)
    for b, v in zip(bars, values):
        ax.text(b.get_width() + max(values) * 0.01,
                b.get_y() + b.get_height() / 2,
                f"{v:,}", va="center", fontsize=10, fontweight="bold")
    ax.set_title("(a) High-level activity", fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.18)
    ax.set_xscale("log")
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)

    # Panel B: lines added vs deleted
    ax = axes[0, 1]
    ax.bar(["Insertions", "Deletions", "Net"],
           [ins, dels, ins - dels],
           color=["#2ca02c", "#d62728", "#1f77b4"], alpha=0.85)
    yvals = [ins, dels, ins - dels]
    yspan = max(yvals) - min(yvals)
    for x, y in zip(range(3), yvals):
        ax.text(x, y + yspan * 0.01,
                f"{y:,}", ha="center", va="bottom", fontsize=10,
                fontweight="bold")
    ax.set_title("(b) Lines changed since v13 presentation", fontsize=12)
    ax.set_ylabel("lines")
    ax.set_ylim(0, max(yvals) * 1.10)

    # Panel C: lines per area
    ax = axes[1, 0]
    items = [
        (f"Forward/bv_solver/\n({len(bvsolver_files)} files)",
         bvsolver_lines, "#1f77b4"),
        (f"scripts/studies/\n({len(studies_files)} files)",
         studies_lines, "#2ca02c"),
        (f"tests/\n({len(test_files)} files)",
         test_lines, "#ff7f0e"),
        (f"docs/\n({len(docs_files)} markdown)",
         docs_lines, "#9467bd"),
    ]
    labs = [a for a, _, _ in items]
    vals = [b for _, b, _ in items]
    cols = [c for _, _, c in items]
    bars = ax.bar(labs, vals, color=cols, alpha=0.85)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(vals) * 0.01,
                f"{v:,}", ha="center", va="bottom", fontsize=10,
                fontweight="bold")
    ax.set_title("(c) Current code volume by area", fontsize=12)
    ax.set_ylabel("lines")
    ax.set_ylim(0, max(vals) * 1.12)
    ax.tick_params(axis="x", labelsize=9)

    # Panel D: claim panel — build with explicit newlines so the
    # final string contains real \n characters (raw f-strings would
    # leave the backslash-n literal).
    ax = axes[1, 1]
    ax.axis("off")
    bullet_lines = [
        "Every line in this period was AI-written.",
        "",
        f"•  {commits} commits in ~2 months (~{commits / 60:.1f}/day).",
        f"•  {n_py_total:,} Python files in the tree; "
        f"{py_lines_total:,} lines total.",
        f"•  {n_handoffs} ChatGPT handoff docs (#2 → #10).",
        f"•  {len(test_files)} test files alongside production code.",
        f"•  {n_studyresult_dirs} StudyResults/ directories"
        " (one per numerical experiment).",
        "",
        "The bottleneck is now scientific judgment",
        "and verification, not implementation throughput.",
    ]
    text = "\n".join(bullet_lines)
    ax.text(0.02, 0.98, text, ha="left", va="top", fontsize=11.5,
            family="serif",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#fff8d6",
                      edgecolor="#b8860b", linewidth=0.8))

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO)}")
    return out


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    make_voltage_window_plot()
    make_repo_dashboard()
