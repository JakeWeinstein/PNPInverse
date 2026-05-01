"""V23 Phase 0.2 — Basin-of-attraction map around TRUE.

Per HANDOFF_11 §3.0.2. Samples initial guesses at controlled radii from
TRUE in (log_k0, alpha) space, runs TRF clean-data inverse, and labels
the converged basin. Answers: how close to TRUE does the init have to be
before clean TRF recovers all 4 params?

Reduced default design (24 starts ~ 1.2 h) vs handoff's full (120 starts):
  log_k0 radii: [0.10, 0.30, 0.50]   (3 values; handoff suggests 6)
  alpha radii:  [0.05, 0.10]          (2 values; handoff suggests 4)
  4 LHS samples per (r_logk0, r_alpha) pair
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import qmc

# TRUE values from scripts/_bv_common.py constants
THETA_TRUE = np.array([
    -6.673770,  # log(K0_HAT_R1) ≈ log(1.2632e-3)
    -9.851186,  # log(K0_HAT_R2) ≈ log(5.2632e-5)
    0.6270,     # ALPHA_R1
    0.5000,     # ALPHA_R2
])

V_GRID = ["-0.10", "0.10", "0.20", "0.30", "0.40", "0.50", "0.60"]
OUT_BASE = "v23_single_experiment_basin_map"


def _here() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _gen_lhs(r_logk0: float, r_alpha: float, n: int, seed: int) -> np.ndarray:
    """Latin hypercube samples in [-1, 1]^4, scaled by per-coord radius.
    Returns (n, 4) array of theta perturbations to add to TRUE."""
    sampler = qmc.LatinHypercube(d=4, seed=seed)
    u = sampler.random(n)  # shape (n, 4) in [0, 1]
    u = 2.0 * u - 1.0      # rescale to [-1, 1]
    radii = np.array([r_logk0, r_logk0, r_alpha, r_alpha])
    return u * radii


def _spawn_one(theta: np.ndarray, sub: str, master_log: Path) -> int:
    cmd = [
        sys.executable,
        "scripts/studies/v18_logc_lsq_inverse.py",
        "--method", "trf",
        "--log-rate",
        "--v-grid", *V_GRID,
        "--init", "true",  # not used; overridden by --start-theta
        "--start-theta", *[f"{x:.10f}" for x in theta],
        "--out-base", OUT_BASE,
        "--out_subdir", sub,
    ]
    with open(master_log, "a") as f:
        f.write(f"\n=== START {sub} {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        f.write(f"theta_init: {theta.tolist()}\n")
        f.flush()
        res = subprocess.run(cmd, cwd=_here(), stdout=f, stderr=subprocess.STDOUT)
        f.write(f"=== END   {sub} exit={res.returncode} "
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    return res.returncode


def _classify(r: dict) -> str:
    k02 = r["k0_2_err_pct"]
    others_abs = [abs(r["k0_1_err_pct"]), abs(r["alpha_1_err_pct"]),
                  abs(r["alpha_2_err_pct"])]
    if abs(k02) < 10 and all(e < 10 for e in others_abs):
        return "TRUE_BASIN"
    if k02 > 30:
        return "WRONG_HIGH"
    if k02 < -30:
        return "WRONG_LOW"
    return "OTHER"


def _aggregate(out_root: Path, design: list) -> dict:
    rows = []
    for entry in design:
        sub = entry["sub"]
        path = out_root / sub / "result.json"
        if not path.exists():
            rows.append({**entry, "status": "MISSING"})
            continue
        with open(path) as f:
            d = json.load(f)
        r = d["result"]
        row = {**entry,
               "k0_1_err_pct": r["k0_1_err_pct"],
               "k0_2_err_pct": r["k0_2_err_pct"],
               "alpha_1_err_pct": r["alpha_1_err_pct"],
               "alpha_2_err_pct": r["alpha_2_err_pct"],
               "cost_final": r["cost_final"],
               "n_evals": r["n_evals"],
               "wall_min": r["wall_minutes"],
               "status": "OK"}
        row["lt5_count"] = (
            int(abs(r["k0_1_err_pct"]) < 5) + int(abs(r["k0_2_err_pct"]) < 5)
            + int(abs(r["alpha_1_err_pct"]) < 5) + int(abs(r["alpha_2_err_pct"]) < 5)
        )
        row["basin"] = _classify(r)
        rows.append(row)
    return {"theta_true": THETA_TRUE.tolist(), "design": design, "rows": rows}


def _write_summary(out_root: Path, agg: dict) -> None:
    rows = agg["rows"]
    import csv
    with open(out_root / "basin_map.csv", "w", newline="") as f:
        keys = ["sub", "r_logk0", "r_alpha", "sample_idx",
                "init_log_k0_1", "init_log_k0_2", "init_alpha_1", "init_alpha_2",
                "k0_1_err_pct", "k0_2_err_pct", "alpha_1_err_pct", "alpha_2_err_pct",
                "cost_final", "n_evals", "wall_min", "lt5_count", "basin", "status"]
        w = csv.writer(f); w.writerow(keys)
        for r in rows:
            w.writerow([r.get(k, "") for k in keys])

    # group by (r_logk0, r_alpha) and report basin distribution
    by_radius = {}
    for r in rows:
        key = (r["r_logk0"], r["r_alpha"])
        by_radius.setdefault(key, []).append(r)
    lines = [
        "# V23 Phase 0.2 — Basin-of-attraction map around TRUE\n",
        "Per HANDOFF_11 §3.0.2. LHS starts at controlled radii from TRUE.",
        "TRF (bounded), G0, log-rate, no prior.\n",
        "## Setup\n",
        "TRUE = [log_k0_1, log_k0_2, alpha_1, alpha_2] = "
        f"{['%.4f' % v for v in THETA_TRUE.tolist()]}",
        "Each row in the table is a single TRF run starting at "
        "TRUE + LHS_offset where the offset is bounded per-coord by",
        "(±r_logk0, ±r_logk0, ±r_alpha, ±r_alpha).\n",
        "## Basin distribution per (r_logk0, r_alpha) pair\n",
        "| r_logk0 | r_alpha | n | TRUE_BASIN | WRONG_HIGH | WRONG_LOW | OTHER | MISSING |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (r_lk, r_a), bucket in sorted(by_radius.items()):
        cts = {"TRUE_BASIN": 0, "WRONG_HIGH": 0, "WRONG_LOW": 0,
               "OTHER": 0, "MISSING": 0}
        for x in bucket:
            cts[x.get("basin", x.get("status", "MISSING"))] += 1
        lines.append(f"| {r_lk:.2f} | {r_a:.2f} | {len(bucket)} "
                     f"| {cts['TRUE_BASIN']} | {cts['WRONG_HIGH']} "
                     f"| {cts['WRONG_LOW']} | {cts['OTHER']} | {cts['MISSING']} |")
    lines.append("\n## Per-run details\n")
    lines.append("| sub | r_logk0 | r_alpha | k0_1 err | k0_2 err | α_1 err | α_2 err | cost | <5% | basin |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        if r.get("status") != "OK":
            lines.append(f"| {r['sub']} | {r['r_logk0']:.2f} | {r['r_alpha']:.2f} "
                         f"| — | — | — | — | — | — | {r.get('status','MISSING')} |")
            continue
        lines.append(
            f"| {r['sub']} | {r['r_logk0']:.2f} | {r['r_alpha']:.2f} "
            f"| {r['k0_1_err_pct']:+.2f}% | {r['k0_2_err_pct']:+.2f}% "
            f"| {r['alpha_1_err_pct']:+.2f}% | {r['alpha_2_err_pct']:+.2f}% "
            f"| {r['cost_final']:.2e} | {r['lt5_count']} | {r['basin']} |"
        )
    # Verdict
    lines.append("\n## Verdict\n")
    smallest_break_radius = None
    for (r_lk, r_a), bucket in sorted(by_radius.items()):
        true_count = sum(1 for x in bucket if x.get("basin") == "TRUE_BASIN")
        if true_count < len(bucket):
            if smallest_break_radius is None or (r_lk + r_a) < smallest_break_radius[0] + smallest_break_radius[1]:
                smallest_break_radius = (r_lk, r_a)
    if smallest_break_radius is None:
        lines.append("All sampled radii recover TRUE basin. The basin of attraction "
                     "extends at least to the largest probed radius.")
    else:
        lines.append(f"Basin breaks at (r_logk0={smallest_break_radius[0]:.2f}, "
                     f"r_alpha={smallest_break_radius[1]:.2f}). Inits beyond this "
                     "radius can fail to recover TRUE.")
    with open(out_root / "summary.md", "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r-logk0", nargs="+", type=float,
                        default=[0.10, 0.30, 0.50])
    parser.add_argument("--r-alpha", nargs="+", type=float,
                        default=[0.05, 0.10])
    parser.add_argument("--samples-per-pair", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    out_root = _here() / "StudyResults" / OUT_BASE
    out_root.mkdir(parents=True, exist_ok=True)
    master_log = out_root / "_master_run.log"

    # Build design list
    design = []
    for r_lk in args.r_logk0:
        for r_a in args.r_alpha:
            offsets = _gen_lhs(r_lk, r_a, args.samples_per_pair,
                               seed=args.seed + int(1e3 * r_lk + 1e6 * r_a))
            for i, off in enumerate(offsets):
                theta = THETA_TRUE + off
                sub = f"r_lk{r_lk:.2f}_a{r_a:.2f}_s{i}".replace(".", "p")
                design.append({"sub": sub, "r_logk0": r_lk, "r_alpha": r_a,
                               "sample_idx": i,
                               "init_log_k0_1": float(theta[0]),
                               "init_log_k0_2": float(theta[1]),
                               "init_alpha_1": float(theta[2]),
                               "init_alpha_2": float(theta[3])})

    print(f"[v23-basinmap] Total runs: {len(design)}")
    if not args.aggregate_only:
        with open(master_log, "a") as f:
            f.write(f"\n=== START v23_basin_map "
                    f"({len(design)} runs) "
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        for entry in design:
            theta = np.array([entry["init_log_k0_1"], entry["init_log_k0_2"],
                              entry["init_alpha_1"], entry["init_alpha_2"]])
            print(f"[v23-basinmap] {entry['sub']}", flush=True)
            rc = _spawn_one(theta, entry["sub"], master_log)
            print(f"[v23-basinmap] {entry['sub']} exit={rc}", flush=True)
        with open(master_log, "a") as f:
            f.write(f"=== END v23_basin_map "
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    agg = _aggregate(out_root, design)
    _write_summary(out_root, agg)
    print(f"[v23-basinmap] Aggregated to {out_root}/{{summary.md,basin_map.csv}}")


if __name__ == "__main__":
    main()
