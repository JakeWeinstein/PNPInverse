"""V23 Phase 2A — Multi-experiment FIM with bulk O2 variation.

Per HANDOFF_11 §5-§6.  Computes whitened sensitivity matrices at TRUE for
each experiment (different c_O2_HAT), then evaluates joint FIM diagnostics
for several experimental designs.

Goal:
  Test whether the weak Fisher direction stops being almost-pure log_k0_1
  when the experiment is repeated at multiple bulk O2 concentrations.

Per-experiment forward + adjoint at TRUE is delegated to
v18_logc_lsq_inverse.py with --save-fim-at-true and --c-o2-hat.

Output:
  StudyResults/v23_multiexperiment_fim/
      summary.md
      design_table.csv
      fim_by_design.json
      weak_vectors.csv
      experiments/<exp_name>/fim_at_true.json
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PARAM_NAMES = ["log_k0_1", "log_k0_2", "alpha_1", "alpha_2"]

# Per HANDOFF_11 §6.2: c_O2_bulk in mol/m^3 of [0.25, 0.50, 1.00].
# Solver units are HAT (normalized to baseline = 1.0).  If baseline c_O2 is
# 0.50 mol/m^3, then HAT values are [0.5, 1.0, 2.0] (low/base/high).
EXPERIMENTS = [
    {"name": "ORR_O2_low",  "c_o2_hat": 0.50},
    {"name": "ORR_O2_base", "c_o2_hat": 1.00},
    {"name": "ORR_O2_high", "c_o2_hat": 2.00},
]

# Designs to compare per HANDOFF_11 §6.3.
DESIGNS = [
    {"name": "A_baseline_only",         "experiments": ["ORR_O2_base"]},
    {"name": "B_low_base",              "experiments": ["ORR_O2_low", "ORR_O2_base"]},
    {"name": "C_base_high",             "experiments": ["ORR_O2_base", "ORR_O2_high"]},
    {"name": "D_low_base_high",         "experiments": ["ORR_O2_low", "ORR_O2_base", "ORR_O2_high"]},
]

V_GRID = ["-0.10", "0.10", "0.20", "0.30", "0.40", "0.50", "0.60"]
OUT_BASE = "v23_multiexperiment_fim"


def _here() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _run_one_experiment(exp: dict, out_root: Path, master_log: Path) -> int:
    """Run v18 with --save-fim-at-true and --c-o2-hat for a single experiment."""
    sub = exp["name"]
    cmd = [
        sys.executable,
        "scripts/studies/v18_logc_lsq_inverse.py",
        "--method", "trf",
        "--log-rate",
        "--v-grid", *V_GRID,
        "--init", "true",
        "--c-o2-hat", f"{exp['c_o2_hat']:.4f}",
        "--save-fim-at-true",
        "--out-base", OUT_BASE,
        "--out_subdir", f"experiments/{sub}",
    ]
    with open(master_log, "a") as f:
        f.write(f"\n=== START {sub} c_O2_hat={exp['c_o2_hat']:.4f} "
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        f.write(f"cmd: {' '.join(cmd)}\n")
        f.flush()
        res = subprocess.run(cmd, cwd=_here(), stdout=f, stderr=subprocess.STDOUT)
        f.write(f"=== END   {sub} exit={res.returncode} "
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    return res.returncode


def _load_experiment_S(out_root: Path, exp_name: str) -> dict:
    """Load fim_at_true.json for a single experiment."""
    path = out_root / "experiments" / exp_name / "fim_at_true.json"
    with open(path) as f:
        return json.load(f)


def _build_design_S(exp_data_by_name: dict, design: dict,
                    noise_model: str) -> tuple[np.ndarray, np.ndarray]:
    """Stack whitened sensitivities across experiments for a given design.

    Returns (S_white_stacked, sigma_stacked).
    For 'global_max' noise: σ = 2% × max|y| computed PER EXPERIMENT, applied
    to that experiment's rows only.  Different experiments have different
    σ scales, so they cannot share a global σ.
    """
    S_blocks = []
    sigma_blocks = []
    for exp_name in design["experiments"]:
        d = exp_data_by_name[exp_name]
        cd = np.array(d["observables_at_true"]["cd"])
        pc = np.array(d["observables_at_true"]["pc"])
        S_cd_raw = np.array(d["S_cd_raw"])
        S_pc_raw = np.array(d["S_pc_raw"])
        if noise_model == "global_max":
            sig_cd = 0.02 * np.max(np.abs(cd)) * np.ones_like(cd)
            sig_pc = 0.02 * np.max(np.abs(pc)) * np.ones_like(pc)
        elif noise_model == "local_rel":
            sig_cd = np.maximum(0.02 * np.abs(cd), 1e-10)
            sig_pc = np.maximum(0.02 * np.abs(pc), 1e-10)
        else:
            raise ValueError(f"unknown noise_model {noise_model}")
        S_blocks.append(S_cd_raw / sig_cd[:, None])
        S_blocks.append(S_pc_raw / sig_pc[:, None])
        sigma_blocks.append(sig_cd)
        sigma_blocks.append(sig_pc)
    return np.vstack(S_blocks), np.concatenate(sigma_blocks)


def _fim_metrics(S_white: np.ndarray) -> dict:
    if not np.all(np.isfinite(S_white)):
        return {"error": "non-finite sensitivity"}
    _, sv, _ = np.linalg.svd(S_white, full_matrices=False)
    F = S_white.T @ S_white
    evals, evecs = np.linalg.eigh(F)
    weak_v = evecs[:, 0]
    diag = np.diag(F)
    corr = F / np.sqrt(np.outer(np.maximum(diag, 1e-300),
                                np.maximum(diag, 1e-300)))
    cond_F = float(evals[-1] / max(evals[0], 1e-300))
    # Predicted parameter std devs (diagonal of F^-1, where F is well-cond).
    try:
        F_inv = np.linalg.inv(F)
        pred_std = np.sqrt(np.maximum(np.diag(F_inv), 0.0))
    except np.linalg.LinAlgError:
        pred_std = np.full(S_white.shape[1], np.nan)
    return {
        "n_residuals": int(S_white.shape[0]),
        "n_params": int(S_white.shape[1]),
        "sigma_min": float(sv[-1]),
        "sigma_max": float(sv[0]),
        "condition_S": float(sv[0] / max(sv[-1], 1e-300)),
        "condition_F": cond_F,
        "fim_eigenvalues": evals.tolist(),
        "fim_diagonal": diag.tolist(),
        "weak_eigvec": dict(zip(PARAM_NAMES, weak_v.tolist())),
        "weak_eigvec_log_k0_1_component": float(abs(weak_v[0])),
        "correlation_matrix": corr.tolist(),
        "predicted_std": dict(zip(PARAM_NAMES, pred_std.tolist())),
    }


def _aggregate(out_root: Path) -> dict:
    exp_data = {}
    for exp in EXPERIMENTS:
        try:
            exp_data[exp["name"]] = _load_experiment_S(out_root, exp["name"])
        except FileNotFoundError:
            print(f"  WARNING: missing fim_at_true.json for {exp['name']}")
    fim_by_design = {}
    for design in DESIGNS:
        for noise in ("global_max", "local_rel"):
            try:
                S_w, sigma_stack = _build_design_S(exp_data, design, noise)
                metrics = _fim_metrics(S_w)
                metrics["sigma_size"] = int(len(sigma_stack))
                fim_by_design[f"{design['name']}__{noise}"] = {
                    "design": design["name"],
                    "noise_model": noise,
                    "experiments": design["experiments"],
                    "metrics": metrics,
                }
            except KeyError as e:
                fim_by_design[f"{design['name']}__{noise}"] = {
                    "design": design["name"],
                    "noise_model": noise,
                    "error": f"missing experiment: {e}",
                }
    return {"experiments": list(exp_data.keys()),
            "fim_by_design": fim_by_design}


def _write_outputs(out_root: Path, agg: dict) -> None:
    with open(out_root / "fim_by_design.json", "w") as f:
        json.dump(agg, f, indent=2)
    # design_table.csv
    rows = []
    for key, entry in agg["fim_by_design"].items():
        m = entry.get("metrics", {})
        if "error" in entry:
            rows.append({"design": entry["design"], "noise": entry["noise_model"],
                         "error": entry["error"]})
            continue
        rows.append({
            "design": entry["design"],
            "noise": entry["noise_model"],
            "n_experiments": len(entry["experiments"]),
            "n_residuals": m.get("n_residuals"),
            "sigma_min": f"{m.get('sigma_min', 0):.3e}",
            "sigma_max": f"{m.get('sigma_max', 0):.3e}",
            "cond_F": f"{m.get('condition_F', 0):.3e}",
            "weak_lk0_1": f"{m.get('weak_eigvec_log_k0_1_component', 0):.3f}",
            "weak_eigvec": m.get("weak_eigvec", {}),
            "predicted_std": m.get("predicted_std", {}),
        })
    keys = ["design", "noise", "n_experiments", "n_residuals",
            "sigma_min", "sigma_max", "cond_F", "weak_lk0_1"]
    with open(out_root / "design_table.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(keys)
        for r in rows:
            w.writerow([r.get(k, "") for k in keys])

    # weak_vectors.csv
    with open(out_root / "weak_vectors.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["design", "noise"] + PARAM_NAMES)
        for key, entry in agg["fim_by_design"].items():
            m = entry.get("metrics", {})
            wv = m.get("weak_eigvec", {})
            if not wv:
                continue
            w.writerow([entry["design"], entry["noise_model"]]
                       + [f"{wv.get(p, 0):+.4f}" for p in PARAM_NAMES])

    # summary.md
    lines = [
        "# V23 Phase 2A — Multi-experiment FIM with bulk O2 variation\n",
        "Per HANDOFF_11 §5-§6.  Tests whether the weak Fisher direction "
        "stops being almost-pure log_k0_1 when ORR is repeated at "
        "different bulk c_O2.\n",
        "## Experiments\n",
        "Each experiment forward-solves at G0 (V_RHE = " + str(V_GRID) +
        ") with the indicated bulk c_O2_hat:\n",
    ]
    for exp in EXPERIMENTS:
        c = exp["c_o2_hat"]
        tag = "baseline" if c == 1.0 else f"{c:.2f}x baseline"
        lines.append(f"- **{exp['name']}**: c_O2_hat = {c:.2f} ({tag})")
    lines.append("\n## Designs\n")
    for d in DESIGNS:
        lines.append(f"- **{d['name']}**: stacks " + ", ".join(d["experiments"]))

    lines.append("\n## FIM diagnostics — global_max noise (σ = 2% × max|y| per experiment)\n")
    lines.append("| design | n_residuals | σ_min | σ_max | cond(F) | |log_k0_1| in weak | weak_eigvec |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for d in DESIGNS:
        key = f"{d['name']}__global_max"
        if key not in agg["fim_by_design"]:
            continue
        m = agg["fim_by_design"][key].get("metrics", {})
        wv = m.get("weak_eigvec", {})
        if "error" in agg["fim_by_design"][key]:
            lines.append(f"| {d['name']} | — | — | — | — | — | {agg['fim_by_design'][key]['error']} |")
            continue
        wv_str = "[" + ", ".join(f"{wv.get(p, 0):+.3f}" for p in PARAM_NAMES) + "]"
        lines.append(
            f"| {d['name']} | {m['n_residuals']} "
            f"| {m['sigma_min']:.3e} | {m['sigma_max']:.3e} "
            f"| {m['condition_F']:.3e} | {m['weak_eigvec_log_k0_1_component']:.3f} "
            f"| {wv_str} |"
        )

    lines.append("\n## FIM diagnostics — local_rel noise (σ = 2% × |y_i| per row)\n")
    lines.append("| design | n_residuals | σ_min | σ_max | cond(F) | |log_k0_1| in weak | weak_eigvec |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for d in DESIGNS:
        key = f"{d['name']}__local_rel"
        if key not in agg["fim_by_design"]:
            continue
        m = agg["fim_by_design"][key].get("metrics", {})
        wv = m.get("weak_eigvec", {})
        if "error" in agg["fim_by_design"][key]:
            lines.append(f"| {d['name']} | — | — | — | — | — | {agg['fim_by_design'][key]['error']} |")
            continue
        wv_str = "[" + ", ".join(f"{wv.get(p, 0):+.3f}" for p in PARAM_NAMES) + "]"
        lines.append(
            f"| {d['name']} | {m['n_residuals']} "
            f"| {m['sigma_min']:.3e} | {m['sigma_max']:.3e} "
            f"| {m['condition_F']:.3e} | {m['weak_eigvec_log_k0_1_component']:.3f} "
            f"| {wv_str} |"
        )

    lines.append("\n## Pass criteria (per HANDOFF_11 §5.3)\n")
    lines.append("- weak eigvec |log_k0_1| should drop below 0.95 (was ~0.999 in single-experiment baseline)")
    lines.append("- cond(F) should improve materially")
    lines.append("- σ_min should improve")
    lines.append("\n## Verdict\n")
    base_w = None
    for d in DESIGNS:
        key = f"{d['name']}__global_max"
        if key in agg["fim_by_design"] and "metrics" in agg["fim_by_design"][key]:
            m = agg["fim_by_design"][key]["metrics"]
            if d["name"] == "A_baseline_only":
                base_w = m["weak_eigvec_log_k0_1_component"]
                base_cond = m["condition_F"]
            else:
                pass
    if base_w is not None:
        lines.append(f"Single-experiment baseline (A): |log_k0_1| in weak = {base_w:.3f}.")
    multi_designs = [d["name"] for d in DESIGNS if d["name"] != "A_baseline_only"]
    for dname in multi_designs:
        key = f"{dname}__global_max"
        if key in agg["fim_by_design"] and "metrics" in agg["fim_by_design"][key]:
            m = agg["fim_by_design"][key]["metrics"]
            verdict = ("PASS" if m["weak_eigvec_log_k0_1_component"] < 0.95
                       else "MARGINAL" if m["weak_eigvec_log_k0_1_component"] < 0.99
                       else "FAIL")
            lines.append(f"- {dname}: |log_k0_1| = {m['weak_eigvec_log_k0_1_component']:.3f} → {verdict}")

    with open(out_root / "summary.md", "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Skip running experiments; only re-aggregate.")
    args = parser.parse_args()

    out_root = _here() / "StudyResults" / OUT_BASE
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "experiments").mkdir(parents=True, exist_ok=True)
    master_log = out_root / "_master_run.log"

    if not args.aggregate_only:
        with open(master_log, "a") as f:
            f.write(f"\n=== START v23_multiexperiment_fim "
                    f"({len(EXPERIMENTS)} experiments) "
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        for exp in EXPERIMENTS:
            print(f"[v23-multifim] {exp['name']} c_O2_hat={exp['c_o2_hat']:.2f}",
                  flush=True)
            rc = _run_one_experiment(exp, out_root, master_log)
            print(f"[v23-multifim] {exp['name']} exit={rc}", flush=True)
        with open(master_log, "a") as f:
            f.write(f"=== END v23_multiexperiment_fim "
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    print("[v23-multifim] Aggregating FIM diagnostics...")
    agg = _aggregate(out_root)
    _write_outputs(out_root, agg)
    print(f"[v23-multifim] Done.  See {out_root}/summary.md")


if __name__ == "__main__":
    main()
