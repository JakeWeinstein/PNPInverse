"""V23 Phase 0.1 — Restart-with-perturbation from best wrong basin.

Per HANDOFF_11. Tests whether perturbing log_k0_2 from the V19 G0 minus20
wrong-basin endpoint causes TRF to escape to the TRUE basin or return to
the same basin (confirming barrier).

Perturbations applied to log_k0_2 only; other params held at the wrong-basin
endpoint values. TRF on G0 (bounded), no prior, log-rate ON.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Wrong-basin endpoint from
# StudyResults/v19_lograte_extended_trf_clean/minus20_v2_initcache/result.json
THETA_WRONG_BASIN = {
    "log_k0_1": -6.712083,
    "log_k0_2": -9.387864,
    "alpha_1": 0.627423,
    "alpha_2": 0.495219,
}

# TRUE values (for reference and bookkeeping)
THETA_TRUE = {
    "log_k0_1": -6.673770,  # ≈ log(1.2632e-3)
    "log_k0_2": -9.851186,  # ≈ log(5.2632e-5)
    "alpha_1": 0.6270,
    "alpha_2": 0.5000,
}

PERTURBATIONS = [-0.75, -0.50, -0.25, +0.25, +0.50, +0.75]
V_GRID = ["-0.10", "0.10", "0.20", "0.30", "0.40", "0.50", "0.60"]
OUT_BASE = "v23_restart_perturbation"


def _here() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _spawn_one(perturb: float, out_dir: Path, master_log: Path) -> int:
    """Run v18 inverse with custom start-theta = wrong basin + perturbation."""
    log_k0_2_perturbed = THETA_WRONG_BASIN["log_k0_2"] + perturb
    start_theta = [
        THETA_WRONG_BASIN["log_k0_1"],
        log_k0_2_perturbed,
        THETA_WRONG_BASIN["alpha_1"],
        THETA_WRONG_BASIN["alpha_2"],
    ]
    sign = "p" if perturb >= 0 else "m"
    sub = f"perturb_{sign}{abs(perturb):.2f}".replace(".", "")
    cmd = [
        sys.executable,
        "scripts/studies/v18_logc_lsq_inverse.py",
        "--method", "trf",
        "--log-rate",
        "--v-grid", *V_GRID,
        "--init", "true",  # not used, overridden by --start-theta
        "--start-theta", *[f"{x:.10f}" for x in start_theta],
        "--out-base", OUT_BASE,
        "--out_subdir", sub,
    ]
    with open(master_log, "a") as f:
        f.write(f"\n=== START perturb={perturb:+.2f} (log_k0_2={log_k0_2_perturbed:.4f}) "
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        f.write(f"cmd: {' '.join(cmd)}\n")
        f.flush()
        res = subprocess.run(cmd, cwd=_here(), stdout=f, stderr=subprocess.STDOUT)
        f.write(f"=== END   perturb={perturb:+.2f} exit={res.returncode} "
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    return res.returncode


def _aggregate(out_root: Path) -> dict:
    """Walk per-perturbation result.json files and build a summary."""
    rows = []
    for perturb in PERTURBATIONS:
        sign = "p" if perturb >= 0 else "m"
        sub = f"perturb_{sign}{abs(perturb):.2f}".replace(".", "")
        result_path = out_root / sub / "result.json"
        if not result_path.exists():
            rows.append({"perturb": perturb, "status": "MISSING"})
            continue
        with open(result_path) as f:
            d = json.load(f)
        r = d["result"]
        rows.append({
            "perturb": perturb,
            "log_k0_2_start": THETA_WRONG_BASIN["log_k0_2"] + perturb,
            "k0_1_err_pct": r["k0_1_err_pct"],
            "k0_2_err_pct": r["k0_2_err_pct"],
            "alpha_1_err_pct": r["alpha_1_err_pct"],
            "alpha_2_err_pct": r["alpha_2_err_pct"],
            "cost_final": r["cost_final"],
            "n_evals": r["n_evals"],
            "wall_min": r["wall_minutes"],
            "status": "OK",
            "lt5_count": int(abs(r["k0_1_err_pct"]) < 5)
                       + int(abs(r["k0_2_err_pct"]) < 5)
                       + int(abs(r["alpha_1_err_pct"]) < 5)
                       + int(abs(r["alpha_2_err_pct"]) < 5),
        })
    return {"theta_wrong_basin": THETA_WRONG_BASIN,
            "theta_true": THETA_TRUE,
            "rows": rows}


def _classify_basin(row: dict) -> str:
    """Label endpoint as 'TRUE_BASIN' (all 4 <10%) vs 'WRONG_BASIN_HIGH'
    (k0_2 > +30%) vs 'WRONG_BASIN_LOW' (k0_2 < -30%) vs 'OTHER'."""
    if row.get("status") != "OK":
        return row.get("status", "MISSING")
    k02 = row["k0_2_err_pct"]
    others = [abs(row["k0_1_err_pct"]), abs(row["alpha_1_err_pct"]),
              abs(row["alpha_2_err_pct"])]
    if abs(k02) < 10 and all(e < 10 for e in others):
        return "TRUE_BASIN"
    if k02 > 30:
        return "WRONG_HIGH"
    if k02 < -30:
        return "WRONG_LOW"
    return "OTHER"


def _write_summary(out_root: Path, agg: dict) -> None:
    rows = agg["rows"]
    # CSV
    import csv
    with open(out_root / "restart_results.csv", "w", newline="") as f:
        if rows and rows[0].get("status") == "OK":
            keys = ["perturb", "log_k0_2_start", "cost_final", "n_evals",
                    "wall_min", "k0_1_err_pct", "k0_2_err_pct",
                    "alpha_1_err_pct", "alpha_2_err_pct", "lt5_count", "basin"]
        else:
            keys = list(rows[0].keys()) if rows else []
        w = csv.writer(f)
        w.writerow(keys)
        for r in rows:
            r["basin"] = _classify_basin(r)
            w.writerow([r.get(k, "") for k in keys])
    # JSON
    with open(out_root / "restart_results.json", "w") as f:
        json.dump(agg, f, indent=2)
    # Markdown summary
    lines = [
        "# V23 Phase 0.1 — Restart-with-perturbation from minus20 wrong basin\n",
        "Per HANDOFF_11 §3.0.1. Tests whether perturbing log_k0_2 only causes",
        "TRF to escape to TRUE basin or return to same wrong basin.\n",
        "## Setup\n",
        "Wrong-basin endpoint (V19 G0 minus20):",
        "```",
        f"log_k0_1 = {THETA_WRONG_BASIN['log_k0_1']:+.6f}  (TRUE {THETA_TRUE['log_k0_1']:+.6f})",
        f"log_k0_2 = {THETA_WRONG_BASIN['log_k0_2']:+.6f}  (TRUE {THETA_TRUE['log_k0_2']:+.6f}, +0.464 above)",
        f"alpha_1  = {THETA_WRONG_BASIN['alpha_1']:.6f}  (TRUE {THETA_TRUE['alpha_1']:.6f})",
        f"alpha_2  = {THETA_WRONG_BASIN['alpha_2']:.6f}  (TRUE {THETA_TRUE['alpha_2']:.6f})",
        "```\n",
        "Perturbations applied to log_k0_2 only; other params held.",
        "TRF (bounded), G0, log-rate, no prior, σ_data=2%×max|target|.\n",
        "## Results\n",
        "| perturb | log_k0_2 start | cost | k0_1 err | k0_2 err | α_1 err | α_2 err | <5% | basin |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        if r.get("status") != "OK":
            lines.append(f"| {r['perturb']:+.2f} | — | — | — | — | — | — | — | {r.get('status','MISSING')} |")
            continue
        lines.append(
            f"| {r['perturb']:+.2f} | {r['log_k0_2_start']:.3f} | {r['cost_final']:.3e} "
            f"| {r['k0_1_err_pct']:+.2f}% | {r['k0_2_err_pct']:+.2f}% "
            f"| {r['alpha_1_err_pct']:+.2f}% | {r['alpha_2_err_pct']:+.2f}% "
            f"| {r['lt5_count']} | {r['basin']} |"
        )
    # Verdict
    basins = {r.get("basin") for r in rows if r.get("status") == "OK"}
    lines.append("\n## Verdict\n")
    if "TRUE_BASIN" in basins and "WRONG_HIGH" in basins:
        lines.append("PARTIAL — some perturbations escaped to TRUE basin; "
                     "restart-with-perturbation IS a viable diagnostic strategy.")
    elif basins == {"WRONG_HIGH"} or basins == {"WRONG_LOW"} or basins == {"WRONG_HIGH", "WRONG_LOW"}:
        lines.append("BARRIER CONFIRMED — every perturbation returned to a wrong basin. "
                     "log_k0_2 perturbation alone cannot escape; basin barrier is real.")
    elif "TRUE_BASIN" in basins and len(basins) == 1:
        lines.append("ESCAPE — every perturbation escaped to TRUE basin. "
                     "Original minus20 endpoint was a saddle, not a basin.")
    else:
        lines.append(f"MIXED — basins observed: {sorted(basins)}.")
    with open(out_root / "summary.md", "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Skip running; only re-aggregate existing results.")
    args = parser.parse_args()

    out_root = _here() / "StudyResults" / OUT_BASE
    out_root.mkdir(parents=True, exist_ok=True)
    master_log = out_root / "_master_run.log"

    if not args.aggregate_only:
        with open(master_log, "a") as f:
            f.write(f"\n=== START v23_restart_perturbation "
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        for perturb in PERTURBATIONS:
            print(f"[v23-restart] perturb={perturb:+.2f}", flush=True)
            rc = _spawn_one(perturb, out_root, master_log)
            print(f"[v23-restart] perturb={perturb:+.2f} exit={rc}", flush=True)
        with open(master_log, "a") as f:
            f.write(f"=== END v23_restart_perturbation "
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    agg = _aggregate(out_root)
    _write_summary(out_root, agg)
    print(f"[v23-restart] Aggregated to {out_root}/{{summary.md,restart_results.csv,restart_results.json}}")


if __name__ == "__main__":
    main()
