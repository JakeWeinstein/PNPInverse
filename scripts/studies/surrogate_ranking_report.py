#!/usr/bin/env python3
"""Synthesize all Phase 3 benchmark results into a comprehensive surrogate ranking.

Reads:
  - StudyResults/surrogate_fidelity/fidelity_summary.json   (prediction accuracy)
  - StudyResults/surrogate_fidelity/per_sample_errors.csv    (per-sample for k0_2 bins)
  - StudyResults/surrogate_fidelity/k02_stratified_errors.json (k0_2 stratified)
  - StudyResults/gradient_benchmark/gradient_accuracy.json   (gradient quality)
  - StudyResults/gradient_benchmark/gradient_speed.json      (gradient timing)
  - StudyResults/inverse_benchmark/recovery_summary.json     (inverse recovery)
  - StudyResults/inverse_benchmark/timing_table.csv          (inference speed)

Writes:
  - StudyResults/surrogate_fidelity/ranking_report.json
  - Updates StudyResults/surrogate_fidelity/fidelity_summary.json (ensures gp + pce)

Ranking weights:
  40% inverse recovery, 25% prediction accuracy, 20% k0_2 performance,
  10% speed, 5% gradient quality
"""
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ROOT = Path(__file__).resolve().parent.parent.parent
STUDY = ROOT / "StudyResults"
FIDELITY_DIR = STUDY / "surrogate_fidelity"
GRADIENT_DIR = STUDY / "gradient_benchmark"
INVERSE_DIR = STUDY / "inverse_benchmark"

# Canonical 6-model keys used throughout the report.
# Maps canonical name -> display name
MODEL_NAMES = {
    "nn_ensemble": "NN Ensemble (D1)",
    "rbf_baseline": "RBF Baseline",
    "pod_rbf_log": "POD-RBF (log)",
    "pod_rbf_nolog": "POD-RBF (nolog)",
    "gp_fixed": "GP (GPyTorch)",
    "pce": "PCE (ChaosPy)",
}

# Maps the recovery_summary.json model names to our canonical keys
RECOVERY_NAME_MAP = {
    "NN D1-default": "nn_ensemble",
    "RBF baseline": "rbf_baseline",
    "POD-RBF log": "pod_rbf_log",
    "POD-RBF nolog": "pod_rbf_nolog",
    "GP": "gp_fixed",
}

WEIGHTS = {
    "inverse_recovery": 0.40,
    "prediction_accuracy": 0.25,
    "k02_performance": 0.20,
    "speed": 0.10,
    "gradient_quality": 0.05,
}

GO_NOGO_THRESHOLD = 0.107  # 10.7% worst-case NRMSE


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_csv_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# 1. Prediction accuracy
# ---------------------------------------------------------------------------
def get_prediction_accuracy(fidelity):
    """Return {model: {cd_median, pc_median, combined}} from fidelity_summary."""
    result = {}
    for mkey in MODEL_NAMES:
        m = fidelity["models"][mkey]
        cd_med = m["cd_median_nrmse"]
        pc_med = m["pc_median_nrmse"]
        result[mkey] = {
            "cd_median_nrmse": cd_med,
            "pc_median_nrmse": pc_med,
            "combined": (cd_med + pc_med) / 2.0,
        }
    return result


# ---------------------------------------------------------------------------
# 2. Gradient quality
# ---------------------------------------------------------------------------
def get_gradient_quality(grad_acc):
    """Return {model: {method, mean_rel_diff}}."""
    result = {}
    # NN ensemble: autograd vs FD at h=1e-5 -> relative diff
    nn_auto = next(e for e in grad_acc["nn_ensemble"] if e["method"] == "autograd")
    nn_fd5 = next(e for e in grad_acc["nn_ensemble"] if e["method"] == "fd_h=1e-05")
    # Compute mean pairwise relative difference between autograd and finest FD
    auto_errs = np.array(nn_auto["point_errors"])
    fd5_errs = np.array(nn_fd5["point_errors"])
    nn_rel_diff = float(np.mean(np.abs(auto_errs - fd5_errs) / (auto_errs + 1e-12)))
    result["nn_ensemble"] = {"method": "autograd", "mean_rel_diff": nn_rel_diff}

    # GP: autograd is broken (1e12 errors), effectively unusable
    result["gp_fixed"] = {"method": "fd_only_broken_autograd", "mean_rel_diff": 1.0}

    # PCE: analytic gradient with near-zero error
    pce_analytic = next(e for e in grad_acc["pce"] if e["method"] == "analytic")
    result["pce"] = {
        "method": "analytic",
        "mean_rel_diff": float(pce_analytic["relative_error"]),
    }

    # RBF baseline, POD-RBF: FD only
    rbf_best = next(e for e in grad_acc["rbf_baseline"] if e["method"] == "fd_h=1e-05")
    result["rbf_baseline"] = {
        "method": "fd_only",
        "mean_rel_diff": float(rbf_best["relative_error"]),
    }

    # pod_rbf appears as "pod_rbf" in gradient_accuracy
    pod_best = next(e for e in grad_acc["pod_rbf"] if e["method"] == "fd_h=1e-05")
    result["pod_rbf_log"] = {
        "method": "fd_only",
        "mean_rel_diff": float(pod_best["relative_error"]),
    }
    # pod_rbf_nolog shares same gradient engine as pod_rbf_log
    result["pod_rbf_nolog"] = {
        "method": "fd_only",
        "mean_rel_diff": float(pod_best["relative_error"]),
    }

    return result


# ---------------------------------------------------------------------------
# 3. k0_2 stratified performance
# ---------------------------------------------------------------------------
def get_k02_performance(k02_data, per_sample_path):
    """Return {model: {worst_bin_nrmse, per_bin}}."""
    result = {}

    # Models that have k0_2 stratified data directly
    for mkey in ["nn_ensemble", "rbf_baseline", "pod_rbf_log", "pod_rbf_nolog"]:
        mdata = k02_data["models"][mkey]
        # Worst bin = highest mean CD NRMSE across bins
        worst_cd = max(mdata["cd"][b]["mean"] for b in k02_data["k02_bins"])
        worst_pc = max(mdata["pc"][b]["mean"] for b in k02_data["k02_bins"])
        worst_combined = (worst_cd + worst_pc) / 2.0
        per_bin = {}
        for b in k02_data["k02_bins"]:
            per_bin[b] = {
                "cd_mean_nrmse": mdata["cd"][b]["mean"],
                "pc_mean_nrmse": mdata["pc"][b]["mean"],
            }
        result[mkey] = {"worst_bin_nrmse": worst_combined, "per_bin": per_bin}

    # For gp_fixed and pce, compute from per_sample_errors.csv
    rows = load_csv_rows(per_sample_path)
    for canon_key, col_prefix in [("gp_fixed", "gp_fixed"), ("pce", "pce")]:
        bins = {}
        for row in rows:
            k02 = float(row["k0_2"])
            log_k02 = np.log10(k02) if k02 > 0 else -10
            # Determine bin
            if log_k02 < -6:
                bname = "[1e-7,1e-6)"
            elif log_k02 < -5:
                bname = "[1e-6,1e-5)"
            elif log_k02 < -4:
                bname = "[1e-5,1e-4)"
            elif log_k02 < -3:
                bname = "[1e-4,1e-3)"
            elif log_k02 < -2:
                bname = "[1e-3,1e-2)"
            else:
                bname = "[1e-2,1e-1)"

            cd_nrmse = float(row[f"{col_prefix}_cd_nrmse"])
            pc_nrmse = float(row[f"{col_prefix}_pc_nrmse"])

            if bname not in bins:
                bins[bname] = {"cd": [], "pc": []}
            bins[bname]["cd"].append(cd_nrmse)
            bins[bname]["pc"].append(pc_nrmse)

        per_bin = {}
        worst_combined = 0.0
        for bname in k02_data["k02_bins"]:
            if bname in bins:
                cd_mean = float(np.mean(bins[bname]["cd"]))
                pc_mean = float(np.mean(bins[bname]["pc"]))
            else:
                cd_mean = 0.0
                pc_mean = 0.0
            per_bin[bname] = {"cd_mean_nrmse": cd_mean, "pc_mean_nrmse": pc_mean}
            worst_combined = max(worst_combined, (cd_mean + pc_mean) / 2.0)
        result[canon_key] = {"worst_bin_nrmse": worst_combined, "per_bin": per_bin}

    return result


# ---------------------------------------------------------------------------
# 4. Inverse recovery
# ---------------------------------------------------------------------------
def get_inverse_recovery(recovery):
    """Return {model: {max_rel_error_pct, k02_median_error_pct, per_param}}."""
    result = {}
    for rec_name, canon_key in RECOVERY_NAME_MAP.items():
        if rec_name in recovery["models"]:
            m = recovery["models"][rec_name]
            result[canon_key] = {
                "max_rel_error_pct": m["max_error_pct"]["median"],
                "k02_median_error_pct": m["k0_2_error_pct"]["median"],
                "k01_median_error_pct": m["k0_1_error_pct"]["median"],
                "n_runs": m["n_runs"],
                "mean_time_s": m["mean_time_s"],
            }

    # PCE was not tested for inverse recovery -- assign worst penalty
    # Use the worst median max error among tested models as the penalty value
    worst_median = max(r["max_rel_error_pct"] for r in result.values())
    result["pce"] = {
        "max_rel_error_pct": worst_median * 1.5,  # 50% penalty above worst
        "k02_median_error_pct": worst_median * 1.5,
        "k01_median_error_pct": worst_median * 1.5,
        "n_runs": 0,
        "mean_time_s": 0.0,
        "note": "Not tested -- penalty assigned (1.5x worst tested model)",
    }

    return result


# ---------------------------------------------------------------------------
# 5. Speed
# ---------------------------------------------------------------------------
def get_speed(grad_speed, timing_rows):
    """Return {model: {predict_ms, gradient_ms, total_ms}}."""
    # Gradient speed from gradient_speed.json (ms per eval)
    # Predict speed: approximate from timing_table (inference time / n_candidates)
    # We use gradient speed as the main metric since prediction speed
    # dominates in multistart optimization

    result = {}

    # From gradient_speed.json, get fastest gradient method per model
    speed_map = {}
    for key, entries in grad_speed.items():
        fastest = min(entries, key=lambda e: e["ms_per_eval"])
        speed_map[key] = fastest["ms_per_eval"]

    # Map to canonical names
    result["nn_ensemble"] = {
        "gradient_ms": speed_map.get("nn_ensemble", 999),
        "predict_ms": speed_map.get("nn_ensemble", 999) * 0.3,  # predict < gradient
    }
    result["gp_fixed"] = {
        "gradient_ms": speed_map.get("gp", 999),
        "predict_ms": speed_map.get("gp", 999) * 0.3,
    }
    result["pce"] = {
        "gradient_ms": speed_map.get("pce", 999),
        "predict_ms": speed_map.get("pce", 999) * 0.3,
    }
    result["rbf_baseline"] = {
        "gradient_ms": speed_map.get("rbf_baseline", 999),
        "predict_ms": speed_map.get("rbf_baseline", 999) * 0.3,
    }
    result["pod_rbf_log"] = {
        "gradient_ms": speed_map.get("pod_rbf", 999),
        "predict_ms": speed_map.get("pod_rbf", 999) * 0.3,
    }
    result["pod_rbf_nolog"] = {
        "gradient_ms": speed_map.get("pod_rbf", 999),
        "predict_ms": speed_map.get("pod_rbf", 999) * 0.3,
    }

    # Also pull mean_time_s from timing_table for context
    for row in timing_rows:
        name = row["model_name"]
        method = row["method"]
        if method == "cascade" and name in RECOVERY_NAME_MAP:
            canon = RECOVERY_NAME_MAP[name]
            result[canon]["cascade_time_s"] = float(row["mean_time_s"])

    for mkey in MODEL_NAMES:
        r = result[mkey]
        r["total_ms"] = r["gradient_ms"] + r["predict_ms"]

    return result


# ---------------------------------------------------------------------------
# Normalization and composite score
# ---------------------------------------------------------------------------
def normalize_scores(raw_scores, models):
    """Normalize to [0,1] where 0=best, 1=worst."""
    values = [raw_scores[m] for m in models]
    vmin, vmax = min(values), max(values)
    if vmax - vmin < 1e-15:
        return {m: 0.0 for m in models}
    return {m: (raw_scores[m] - vmin) / (vmax - vmin) for m in models}


def compute_composite(models, pred_acc, grad_qual, k02_perf, inv_rec, speed):
    """Compute weighted composite score for each model."""
    raw = {}
    norm = {}

    # Prediction accuracy (lower = better)
    raw["prediction_accuracy"] = {m: pred_acc[m]["combined"] for m in models}
    norm["prediction_accuracy"] = normalize_scores(
        raw["prediction_accuracy"], models
    )

    # Gradient quality (lower = better)
    raw["gradient_quality"] = {m: grad_qual[m]["mean_rel_diff"] for m in models}
    norm["gradient_quality"] = normalize_scores(raw["gradient_quality"], models)

    # k0_2 performance (lower = better)
    raw["k02_performance"] = {m: k02_perf[m]["worst_bin_nrmse"] for m in models}
    norm["k02_performance"] = normalize_scores(raw["k02_performance"], models)

    # Inverse recovery (lower = better)
    raw["inverse_recovery"] = {
        m: inv_rec[m]["max_rel_error_pct"] for m in models
    }
    norm["inverse_recovery"] = normalize_scores(raw["inverse_recovery"], models)

    # Speed (lower total_ms = better)
    raw["speed"] = {m: speed[m]["total_ms"] for m in models}
    norm["speed"] = normalize_scores(raw["speed"], models)

    composites = {}
    for m in models:
        score = sum(
            WEIGHTS[dim] * norm[dim][m]
            for dim in WEIGHTS
        )
        composites[m] = float(score)

    return composites, raw, norm


# ---------------------------------------------------------------------------
# Go/no-go check
# ---------------------------------------------------------------------------
def check_go_nogo(fidelity):
    """Check if at least one model has 99th percentile worst-case < threshold."""
    # Use per_sample_errors.csv to compute 99th percentile
    per_sample = load_csv_rows(FIDELITY_DIR / "per_sample_errors.csv")

    best_worst = float("inf")
    best_model = None

    for mkey in MODEL_NAMES:
        cd_col = f"{mkey}_cd_nrmse"
        pc_col = f"{mkey}_pc_nrmse"
        cd_vals = sorted([float(r[cd_col]) for r in per_sample])
        pc_vals = sorted([float(r[pc_col]) for r in per_sample])

        # 99th percentile (ignoring the very worst outliers from near-zero PC ranges)
        n = len(cd_vals)
        idx_99 = min(int(0.99 * n), n - 1)
        cd_99 = cd_vals[idx_99]
        pc_99 = pc_vals[idx_99]
        worst_99 = max(cd_99, pc_99)

        if worst_99 < best_worst:
            best_worst = worst_99
            best_model = mkey

    return {
        "passed": best_worst < GO_NOGO_THRESHOLD,
        "best_worst_case_99th": float(best_worst),
        "best_model": best_model,
        "threshold": GO_NOGO_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# Selection recommendation
# ---------------------------------------------------------------------------
def make_recommendation(ranked_models, composites, pred_acc, inv_rec, speed):
    """Make selection recommendation for Phases 4-5."""
    primary = ranked_models[0]
    secondary = ranked_models[1] if len(ranked_models) > 1 else None

    # Check if secondary has meaningfully different strengths
    use_secondary = False
    secondary_rationale = ""
    if secondary:
        # Check if secondary is much better in some dimension
        pri_inv = inv_rec[primary]["max_rel_error_pct"]
        sec_inv = inv_rec[secondary]["max_rel_error_pct"]
        pri_speed = speed[primary]["total_ms"]
        sec_speed = speed[secondary]["total_ms"]
        pri_pred = pred_acc[primary]["combined"]
        sec_pred = pred_acc[secondary]["combined"]

        # If secondary has >30% better prediction or >50% better speed
        if sec_pred < pri_pred * 0.7 or sec_speed < pri_speed * 0.5:
            use_secondary = True
            secondary_rationale = (
                f"{MODEL_NAMES[secondary]} has "
                f"{'better prediction accuracy' if sec_pred < pri_pred * 0.7 else ''}"
                f"{'better speed' if sec_speed < pri_speed * 0.5 else ''}"
                f" vs {MODEL_NAMES[primary]}"
            )

    strengths = []
    weaknesses = []
    m = primary
    if inv_rec[m]["max_rel_error_pct"] < 15:
        strengths.append("excellent inverse recovery")
    if pred_acc[m]["combined"] < 0.02:
        strengths.append("strong prediction accuracy")
    if speed[m]["total_ms"] < 5:
        strengths.append("fast inference")
    if inv_rec[m]["max_rel_error_pct"] > 30:
        weaknesses.append("poor inverse recovery")
    if speed[m]["total_ms"] > 100:
        weaknesses.append("slow for multistart")

    rationale = (
        f"{MODEL_NAMES[primary]} selected as primary surrogate "
        f"(composite score: {composites[primary]:.4f}). "
        f"Strengths: {', '.join(strengths) if strengths else 'balanced across dimensions'}. "
        f"Weaknesses: {', '.join(weaknesses) if weaknesses else 'none critical'}."
    )

    rec = {
        "primary_surrogate": primary,
        "primary_display_name": MODEL_NAMES[primary],
        "secondary_surrogate": secondary if use_secondary else None,
        "secondary_display_name": MODEL_NAMES.get(secondary) if use_secondary else None,
        "rationale": rationale,
        "secondary_rationale": secondary_rationale if use_secondary else None,
        "phase4_config": (
            f"Use {MODEL_NAMES[primary]} with cascade inference for ISMO. "
            f"Mean cascade time: {inv_rec[primary].get('mean_time_s', 'N/A')}s."
        ),
        "phase5_config": (
            f"Use {MODEL_NAMES[primary]} for PDE refinement with space-mapping bias correction. "
            f"k0_2 median error: {inv_rec[primary]['k02_median_error_pct']:.1f}%."
        ),
    }
    return rec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("SURROGATE RANKING REPORT")
    print("=" * 70)

    # Load all data
    fidelity = load_json(FIDELITY_DIR / "fidelity_summary.json")
    k02_data = load_json(FIDELITY_DIR / "k02_stratified_errors.json")
    grad_acc = load_json(GRADIENT_DIR / "gradient_accuracy.json")
    grad_speed = load_json(GRADIENT_DIR / "gradient_speed.json")
    recovery = load_json(INVERSE_DIR / "recovery_summary.json")
    timing_rows = load_csv_rows(INVERSE_DIR / "timing_table.csv")

    models = list(MODEL_NAMES.keys())

    # Compute all dimension scores
    print("\n[1/6] Computing prediction accuracy scores...")
    pred_acc = get_prediction_accuracy(fidelity)

    print("[2/6] Computing gradient quality scores...")
    grad_qual = get_gradient_quality(grad_acc)

    print("[3/6] Computing k0_2 stratified performance...")
    k02_perf = get_k02_performance(
        k02_data, FIDELITY_DIR / "per_sample_errors.csv"
    )

    print("[4/6] Computing inverse recovery scores...")
    inv_rec = get_inverse_recovery(recovery)

    print("[5/6] Computing speed scores...")
    speed = get_speed(grad_speed, timing_rows)

    print("[6/6] Computing composite scores and ranking...")
    composites, raw, norm = compute_composite(
        models, pred_acc, grad_qual, k02_perf, inv_rec, speed
    )

    # Rank (lower composite = better)
    ranked = sorted(models, key=lambda m: composites[m])
    ranks = {m: i + 1 for i, m in enumerate(ranked)}

    # Go/no-go
    go_nogo = check_go_nogo(fidelity)

    # Recommendation
    recommendation = make_recommendation(ranked, composites, pred_acc, inv_rec, speed)

    # Build output
    report = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "weights": WEIGHTS,
            "go_no_go_threshold": GO_NOGO_THRESHOLD,
            "n_models": len(models),
            "model_keys": models,
        },
        "models": {},
        "go_no_go": go_nogo,
        "recommendation": recommendation,
    }

    for m in models:
        report["models"][m] = {
            "display_name": MODEL_NAMES[m],
            "prediction_accuracy": {
                "cd_median_nrmse": pred_acc[m]["cd_median_nrmse"],
                "pc_median_nrmse": pred_acc[m]["pc_median_nrmse"],
                "combined": pred_acc[m]["combined"],
                "score_norm": float(norm["prediction_accuracy"][m]),
            },
            "gradient_quality": {
                "method": grad_qual[m]["method"],
                "mean_rel_diff": grad_qual[m]["mean_rel_diff"],
                "score_norm": float(norm["gradient_quality"][m]),
            },
            "k02_performance": {
                "worst_bin_nrmse": k02_perf[m]["worst_bin_nrmse"],
                "per_bin": k02_perf[m]["per_bin"],
                "score_norm": float(norm["k02_performance"][m]),
            },
            "inverse_recovery": {
                "max_rel_error_pct": inv_rec[m]["max_rel_error_pct"],
                "k02_median_error_pct": inv_rec[m]["k02_median_error_pct"],
                "n_runs": inv_rec[m]["n_runs"],
                "score_norm": float(norm["inverse_recovery"][m]),
            },
            "speed": {
                "gradient_ms": speed[m]["gradient_ms"],
                "total_ms": speed[m]["total_ms"],
                "score_norm": float(norm["speed"][m]),
            },
            "composite_score": float(composites[m]),
            "rank": ranks[m],
        }

    # Write ranking report
    out_path = FIDELITY_DIR / "ranking_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote: {out_path}")

    # Update fidelity_summary.json -- ensure metadata reflects all 6 models
    fidelity["metadata"]["model_names"] = models
    with open(FIDELITY_DIR / "fidelity_summary.json", "w") as f:
        json.dump(fidelity, f, indent=2)
    print(f"Updated: {FIDELITY_DIR / 'fidelity_summary.json'}")

    # Print formatted ranking table
    print("\n" + "=" * 70)
    print("RANKING TABLE")
    print("=" * 70)
    header = (
        f"{'Rank':<5} {'Model':<22} {'Composite':<10} "
        f"{'InvRec':<8} {'PredAcc':<8} {'k02':<8} {'Speed':<8} {'Grad':<8}"
    )
    print(header)
    print("-" * len(header))

    for m in ranked:
        r = report["models"][m]
        print(
            f"{r['rank']:<5} {MODEL_NAMES[m]:<22} {r['composite_score']:.4f}    "
            f"{r['inverse_recovery']['score_norm']:.3f}   "
            f"{r['prediction_accuracy']['score_norm']:.3f}   "
            f"{r['k02_performance']['score_norm']:.3f}   "
            f"{r['speed']['score_norm']:.3f}   "
            f"{r['gradient_quality']['score_norm']:.3f}"
        )

    print("\n" + "=" * 70)
    print("GO/NO-GO DECISION")
    print("=" * 70)
    status = "PASS" if go_nogo["passed"] else "NO-GO"
    print(f"  Status: {status}")
    print(f"  Best 99th-percentile worst-case: {go_nogo['best_worst_case_99th']:.4f}")
    print(f"  Best model: {MODEL_NAMES[go_nogo['best_model']]}")
    print(f"  Threshold: {go_nogo['threshold']}")

    print("\n" + "=" * 70)
    print("SELECTION RECOMMENDATION")
    print("=" * 70)
    print(f"  Primary:   {recommendation['primary_display_name']}")
    if recommendation["secondary_surrogate"]:
        print(f"  Secondary: {recommendation['secondary_display_name']}")
    print(f"  Rationale: {recommendation['rationale']}")
    print(f"  Phase 4:   {recommendation['phase4_config']}")
    print(f"  Phase 5:   {recommendation['phase5_config']}")

    # Sanity check: no NaN values
    for m in models:
        r = report["models"][m]
        assert np.isfinite(r["composite_score"]), f"NaN in composite for {m}"
        for dim in WEIGHTS:
            assert np.isfinite(
                r.get(dim, r.get(f"{dim}", {})).get("score_norm", 0.0)
                if isinstance(r.get(dim), dict)
                else 0.0
            ), f"NaN in {dim} norm for {m}"

    print(f"\nAll {len(models)} models ranked. No NaN values detected.")
    print("Done.")


if __name__ == "__main__":
    main()
