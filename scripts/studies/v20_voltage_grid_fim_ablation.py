"""V20 — Voltage-grid FIM ablation.

Per `docs/PNP Log Rate Multi Init Handoff.md` Task B: compare FIM
diagnostics across voltage grids to test whether adding low/onset
voltages (0.00, 0.15, 0.25) and mild negatives improves the weak
direction (currently log_k0_1) at TRUE.

Reads pre-computed S_cd_raw, S_pc_raw at the unified 13-voltage grid
from ``v19_bv_clip_audit.py`` output and subsets rows to construct
each grid's whitened FIM.

Output:
    StudyResults/v20_voltage_grid_fim_ablation/
        summary.md
        fim_by_grid.json
        weak_eigvec_by_grid.csv
        leverage_by_voltage.csv
"""
from __future__ import annotations

import csv
import json
import os
import sys
from typing import Any

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np


PARAM_NAMES = ("log_k0_1", "log_k0_2", "alpha_1", "alpha_2")


def canonical_ridge() -> np.ndarray:
    v = np.array([0.0, -47.0, 0.0, +1.0])
    return v / np.linalg.norm(v)


def fim_metrics(S_white: np.ndarray) -> dict[str, Any]:
    if not np.all(np.isfinite(S_white)):
        return {"error": "non-finite sensitivity"}
    _, sv, _ = np.linalg.svd(S_white, full_matrices=False)
    F = S_white.T @ S_white
    evals, evecs = np.linalg.eigh(F)
    cond_F = float(evals[-1] / max(evals[0], 1e-300))
    weak_v = evecs[:, 0]
    cos_sim = float(abs(np.dot(weak_v, canonical_ridge())))
    diag = np.diag(F)
    corr = F / np.sqrt(np.outer(diag, diag))
    return {
        "n_residuals": int(S_white.shape[0]),
        "n_params": int(S_white.shape[1]),
        "singular_values": sv.tolist(),
        "fim_eigenvalues": evals.tolist(),
        "fim_eigenvectors": evecs.tolist(),
        "fim_diagonal": diag.tolist(),
        "correlation_matrix": corr.tolist(),
        "condition_number": cond_F,
        "weak_eigvec": dict(zip(PARAM_NAMES, weak_v.tolist())),
        "weak_eigvec_canonical_ridge_cos": cos_sim,
        "weak_eigvec_log_k0_1_component": float(abs(weak_v[0])),
    }


def per_voltage_leverage(S_white: np.ndarray, n_obs: int) -> list[float]:
    """Row leverage = ||row||^2 / ||S||_F^2 for each pair (cd, pc) at V_i."""
    NV = S_white.shape[0] // n_obs
    row_norms_sq = (S_white ** 2).sum(axis=1)
    total = row_norms_sq.sum()
    leverage = []
    for i in range(NV):
        row_indices = [i + k * NV for k in range(n_obs)]
        leverage.append(float(row_norms_sq[row_indices].sum() / total))
    return leverage


def per_voltage_weak_dir_contribution(S_white: np.ndarray, n_obs: int) -> list[float]:
    """How much each voltage contributes to the smallest singular direction."""
    U, sv, Vt = np.linalg.svd(S_white, full_matrices=False)
    # smallest singular direction: U[:, -1] (left sv) tells row contributions
    weak_left = U[:, -1]
    NV = S_white.shape[0] // n_obs
    contribs = []
    for i in range(NV):
        row_indices = [i + k * NV for k in range(n_obs)]
        contribs.append(float((weak_left[row_indices] ** 2).sum()))
    return contribs


def main() -> None:
    in_dir = os.path.join(
        _ROOT, "StudyResults", "v20_voltage_grid_fim_ablation",
        "unified_13V_cap50_lograte",
    )
    fim_path = os.path.join(in_dir, "fim_by_cap.json")
    obs_path = os.path.join(in_dir, "observables_by_cap.csv")

    if not os.path.exists(fim_path):
        print(f"FATAL: {fim_path} not found. Run v19_bv_clip_audit.py with "
              f"--caps 50 --log-rate --v-grid <13V> --out-dir <unified_dir> first.")
        sys.exit(1)

    with open(fim_path) as f:
        fim_data = json.load(f)
    cap_data = fim_data["50"]
    S_cd_full = np.array(cap_data["S_cd_raw"])
    S_pc_full = np.array(cap_data["S_pc_raw"])

    cd_vals = []; pc_vals = []; voltages = []
    with open(obs_path) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            if row["cap"] != "50":
                continue
            if row["converged"] != "True":
                continue
            voltages.append(float(row["V_RHE"]))
            cd_vals.append(float(row["cd"]))
            pc_vals.append(float(row["pc"]))
    V_full = np.array(voltages)
    cd_full = np.array(cd_vals)
    pc_full = np.array(pc_vals)
    NV_full = len(V_full)
    assert S_cd_full.shape == (NV_full, 4), \
        f"S_cd shape {S_cd_full.shape} != ({NV_full}, 4)"

    print(f"Unified V_GRID ({NV_full} voltages):")
    for i, v in enumerate(V_full):
        print(f"  {i:2d}  V={v:+.2f}  cd={cd_full[i]:+.3e}  pc={pc_full[i]:+.3e}")
    print()

    # Define grids per Task B in handoff
    grids = {
        "G0_current":          [-0.10, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
        "G1_add_zero":         [-0.10, 0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
        "G2_densify_R1_onset": [-0.10, 0.00, 0.10, 0.15, 0.20, 0.25,
                                0.30, 0.40, 0.50, 0.60],
        "G3_add_mild_negative": [-0.30, -0.20, -0.10, 0.00, 0.10, 0.15,
                                 0.20, 0.25, 0.30, 0.40, 0.50, 0.60],
        "G4_add_strong_negative": [-0.50, -0.30, -0.10, 0.00, 0.10, 0.15,
                                   0.20, 0.25, 0.30, 0.40, 0.50, 0.60],
        "G_best_candidate": [-0.20, -0.10, 0.00, 0.10, 0.15, 0.20, 0.25,
                             0.30, 0.40, 0.50, 0.60],
    }

    # Map voltages to row indices in the unified grid
    def find_index(v: float) -> int | None:
        idx = np.where(np.isclose(V_full, v, atol=1e-3))[0]
        if len(idx) == 0:
            return None
        return int(idx[0])

    rel = 0.02
    OUT_DIR = os.path.join(_ROOT, "StudyResults", "v20_voltage_grid_fim_ablation")
    os.makedirs(OUT_DIR, exist_ok=True)

    # Two noise models per Task 2 in handoff:
    #   global_max: sigma = 2% * max_V(|y|), one sigma per observable type
    #   local_rel:  sigma_y(V) = 2% * |y(V)|, per-V sigma
    NOISE_MODELS = ("global_max", "local_rel")
    grid_results = {nm: [] for nm in NOISE_MODELS}

    for grid_name, V_subset in grids.items():
        row_idx = [find_index(v) for v in V_subset]
        missing = [v for v, idx in zip(V_subset, row_idx) if idx is None]
        V_kept = [v for v, idx in zip(V_subset, row_idx) if idx is not None]
        kept_idx = [idx for idx in row_idx if idx is not None]
        if missing:
            print(f"  {grid_name}: dropped non-converged voltages "
                  f"{missing}; using {len(V_kept)} voltages")
        if len(kept_idx) < 2:
            print(f"  {grid_name}: SKIP — too few voltages converged")
            continue

        V_subset = V_kept
        S_cd = S_cd_full[kept_idx]
        S_pc = S_pc_full[kept_idx]
        cd_sub = cd_full[kept_idx]
        pc_sub = pc_full[kept_idx]

        for noise_model in NOISE_MODELS:
            if noise_model == "global_max":
                sigma_cd = rel * np.max(np.abs(cd_sub))
                sigma_pc = rel * np.max(np.abs(pc_sub))
                sigma_cd_arr = np.full(len(V_subset), sigma_cd)
                sigma_pc_arr = np.full(len(V_subset), sigma_pc)
            else:  # local_rel
                sigma_cd_arr = rel * np.maximum(np.abs(cd_sub), 1e-12)
                sigma_pc_arr = rel * np.maximum(np.abs(pc_sub), 1e-12)

            S_cd_white = S_cd / sigma_cd_arr[:, None]
            S_pc_white = S_pc / sigma_pc_arr[:, None]
            S_both = np.vstack([S_cd_white, S_pc_white])

            m = fim_metrics(S_both)
            leverage = per_voltage_leverage(S_both, n_obs=2)
            weak_contrib = per_voltage_weak_dir_contribution(S_both, n_obs=2)

            result = {
                "grid_name": grid_name,
                "noise_model": noise_model,
                "V_grid": V_subset,
                "n_voltages": len(V_subset),
                "sv_min": m["singular_values"][-1],
                "sv_max": m["singular_values"][0],
                "cond_F": m["condition_number"],
                "ridge_cos": m["weak_eigvec_canonical_ridge_cos"],
                "weak_eigvec_log_k0_1_component": m["weak_eigvec_log_k0_1_component"],
                "weak_eigvec": m["weak_eigvec"],
                "fim_diagonal": m["fim_diagonal"],
                "correlation_matrix": m["correlation_matrix"],
                "leverage_by_voltage": dict(zip([f"{v:+.2f}" for v in V_subset],
                                                 leverage)),
                "weak_dir_contribution_by_voltage": dict(zip(
                    [f"{v:+.2f}" for v in V_subset], weak_contrib)),
            }
            grid_results[noise_model].append(result)

            we = m["weak_eigvec"]
            we_str = (f"[{we['log_k0_1']:+.3f},{we['log_k0_2']:+.3f},"
                      f"{we['alpha_1']:+.3f},{we['alpha_2']:+.3f}]")
            print(f"  [{noise_model}] {grid_name:<25}  NV={len(V_subset):2d}  "
                  f"sv_min={m['singular_values'][-1]:>8.3e}  "
                  f"cond={m['condition_number']:>10.2e}  "
                  f"ridge_cos={m['weak_eigvec_canonical_ridge_cos']:>5.3f}  "
                  f"|k0_1|={m['weak_eigvec_log_k0_1_component']:>5.3f}  "
                  f"weak={we_str}")

    # Save outputs
    with open(os.path.join(OUT_DIR, "fim_by_grid.json"), "w") as f:
        json.dump({
            "config": {
                "input_dir": in_dir,
                "unified_V_grid": V_full.tolist(),
                "rel_noise": rel,
                "noise_models": list(NOISE_MODELS),
            },
            "grids_by_noise_model": grid_results,
        }, f, indent=2)

    with open(os.path.join(OUT_DIR, "weak_eigvec_by_grid.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["noise_model", "grid_name", "n_voltages", "sv_min",
                    "cond_F", "ridge_cos",
                    "weak_log_k0_1", "weak_log_k0_2", "weak_alpha_1",
                    "weak_alpha_2", "weak_log_k0_1_abs"])
        for nm in NOISE_MODELS:
            for r in grid_results[nm]:
                we = r["weak_eigvec"]
                w.writerow([nm, r["grid_name"], r["n_voltages"], r["sv_min"],
                            r["cond_F"], r["ridge_cos"],
                            we["log_k0_1"], we["log_k0_2"], we["alpha_1"],
                            we["alpha_2"], r["weak_eigvec_log_k0_1_component"]])

    with open(os.path.join(OUT_DIR, "leverage_by_voltage.csv"), "w", newline="") as f:
        w = csv.writer(f)
        all_voltages = sorted(V_full.tolist())
        header = ["noise_model", "grid_name"] + [f"V={v:+.2f}" for v in all_voltages]
        w.writerow(header)
        for nm in NOISE_MODELS:
            for r in grid_results[nm]:
                row = [nm, r["grid_name"]]
                for v in all_voltages:
                    key = f"{v:+.2f}"
                    row.append(f"{r['leverage_by_voltage'].get(key, 0.0):.4f}")
                w.writerow(row)

    with open(os.path.join(OUT_DIR, "summary.md"), "w") as fout:
        fout.write("# V20 — Voltage-grid FIM ablation\n\n")
        fout.write("Per Task B in `CHATGPT_HANDOFF_8_LOGRATE_MULTIINIT.md`: test "
                   "whether adding low/onset voltages (0.00, 0.15, 0.25) and mild "
                   "negatives reduces the new weak direction (log_k0_1).\n\n")
        fout.write("Setup: cap=50, log-rate ON, observables=CD+PC. Computed under "
                   "two noise models (global 2% max and local 2% rel).\n\n")
        fout.write("Note: V=-0.50 and V=-0.30 failed cold-ramp; G3 and G4 are "
                   "evaluated on the converged subset.\n\n")
        for nm in NOISE_MODELS:
            fout.write(f"\n## Results — noise model: {nm}\n\n")
            fout.write("| grid | NV | sv_min | cond(F) | ridge_cos | |k0_1| weak | weak_eigvec |\n")
            fout.write("|---|---:|---:|---:|---:|---:|---|\n")
            for r in grid_results[nm]:
                we = r["weak_eigvec"]
                we_str = (f"[{we['log_k0_1']:+.3f}, {we['log_k0_2']:+.3f}, "
                          f"{we['alpha_1']:+.3f}, {we['alpha_2']:+.3f}]")
                fout.write(f"| {r['grid_name']} | {r['n_voltages']} | "
                           f"{r['sv_min']:.3e} | {r['cond_F']:.2e} | "
                           f"{r['ridge_cos']:.3f} | "
                           f"{r['weak_eigvec_log_k0_1_component']:.3f} | "
                           f"{we_str} |\n")

        fout.write("\n## Per-voltage leverage (fraction of ||S||², global_max noise)\n\n")
        all_voltages = sorted(V_full.tolist())
        fout.write("| grid | " + " | ".join(f"V={v:+.2f}" for v in all_voltages)
                   + " |\n")
        fout.write("|---|" + "|".join(["---:"] * len(all_voltages)) + "|\n")
        for r in grid_results["global_max"]:
            row = [r["grid_name"]]
            for v in all_voltages:
                key = f"{v:+.2f}"
                lev = r['leverage_by_voltage'].get(key, 0.0)
                row.append(f"{lev:.3f}" if lev > 0 else "—")
            fout.write("| " + " | ".join(row) + " |\n")

        fout.write("\n## Per-voltage contribution to smallest singular direction (global_max)\n\n")
        fout.write("| grid | " + " | ".join(f"V={v:+.2f}" for v in all_voltages)
                   + " |\n")
        fout.write("|---|" + "|".join(["---:"] * len(all_voltages)) + "|\n")
        for r in grid_results["global_max"]:
            row = [r["grid_name"]]
            for v in all_voltages:
                key = f"{v:+.2f}"
                c = r['weak_dir_contribution_by_voltage'].get(key, 0.0)
                row.append(f"{c:.3f}" if c > 0 else "—")
            fout.write("| " + " | ".join(row) + " |\n")

        # Find best grid per noise model
        fout.write("\n## Decision\n\n")
        for nm in NOISE_MODELS:
            best = min(grid_results[nm], key=lambda r: r["cond_F"])
            worst = max(grid_results[nm], key=lambda r: r["cond_F"])
            fout.write(f"### Under {nm}\n")
            fout.write(f"- Best (lowest cond): **{best['grid_name']}** "
                       f"(cond={best['cond_F']:.2e}, sv_min={best['sv_min']:.2e})\n")
            fout.write(f"- Worst: {worst['grid_name']} "
                       f"(cond={worst['cond_F']:.2e})\n\n")
        fout.write("Recommendation: under both noise models, evaluate which grid "
                   "gives lowest cond(F). The right noise model for the "
                   "publishable result depends on actual measurement-instrument "
                   "characteristics (constant-floor vs proportional).\n")

    print()
    print(f"Saved: {OUT_DIR}/fim_by_grid.json")
    print(f"Saved: {OUT_DIR}/weak_eigvec_by_grid.csv")
    print(f"Saved: {OUT_DIR}/leverage_by_voltage.csv")
    print(f"Saved: {OUT_DIR}/summary.md")


if __name__ == "__main__":
    main()
