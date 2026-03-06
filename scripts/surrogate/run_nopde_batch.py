#!/usr/bin/env python
"""Batch runner: surrogate-only (--no-pde) across all models and noise seeds."""

import subprocess
import sys
import re
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

PYTHON = "/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/venv-firedrake/bin/python"
SCRIPT = "scripts/surrogate/Infer_BVMaster_charged_v13_ultimate.py"
CWD = "/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse"

MODELS = ["nn-ensemble", "rbf", "pod-rbf-log", "pod-rbf-nolog", "nn-single"]

# 0% noise + 5 seeds at 2%
NOISE_CONFIGS = [
    {"label": "0%", "args": ["--noise-percent", "0.0"]},
    {"label": "seed=20260226", "args": ["--noise-seed", "20260226"]},
    {"label": "seed=42", "args": ["--noise-seed", "42"]},
    {"label": "seed=123", "args": ["--noise-seed", "123"]},
    {"label": "seed=7777", "args": ["--noise-seed", "7777"]},
    {"label": "seed=99999", "args": ["--noise-seed", "99999"]},
]

def run_one(model, noise_cfg):
    """Run a single surrogate-only evaluation and parse the result."""
    label = noise_cfg["label"]
    cmd = [
        PYTHON, SCRIPT,
        "--model-type", model,
        "--no-pde",
        "--surr-strategy", "all",
    ] + noise_cfg["args"]

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, cwd=CWD, capture_output=True, text=True, timeout=300
        )
        elapsed = time.time() - t0
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return model, label, {"error": "TIMEOUT", "elapsed": 300}
    except Exception as e:
        return model, label, {"error": str(e), "elapsed": time.time() - t0}

    # Parse "Best result:" block
    parsed = {"elapsed": elapsed}
    best_match = re.search(r"Best result:.*?\(max err = ([\d.]+)%\)", output)
    if best_match:
        parsed["max_err"] = float(best_match.group(1))

    for param in ["k0_1", "k0_2", "alpha_1", "alpha_2"]:
        pat = re.compile(rf"{param}\s*=\s*[\d.eE+-]+\s*\(err\s+([\d.]+)%\)")
        m = pat.search(output)
        if m:
            parsed[param] = float(m.group(1))

    if "max_err" not in parsed:
        parsed["error"] = "PARSE_FAIL"
        # Save last 30 lines for debugging
        parsed["tail"] = "\n".join(output.strip().split("\n")[-30:])

    return model, label, parsed


def main():
    jobs = []
    for model in MODELS:
        for nc in NOISE_CONFIGS:
            jobs.append((model, nc))

    total = len(jobs)
    print(f"Running {total} surrogate-only evaluations...")
    print(f"Models: {MODELS}")
    print(f"Noise configs: {[nc['label'] for nc in NOISE_CONFIGS]}")
    print()

    results = {}
    # Run up to 5 in parallel (one per model type to avoid contention)
    with ProcessPoolExecutor(max_workers=5) as pool:
        futures = {}
        for model, nc in jobs:
            f = pool.submit(run_one, model, nc)
            futures[f] = (model, nc["label"])

        done_count = 0
        for f in as_completed(futures):
            done_count += 1
            model, label, parsed = f.result()
            results[(model, label)] = parsed
            status = f"max_err={parsed.get('max_err', 'N/A')}%" if "error" not in parsed else parsed["error"]
            print(f"  [{done_count}/{total}] {model:15s} {label:15s} → {status}  ({parsed['elapsed']:.0f}s)")

    # Print summary table
    print("\n" + "=" * 120)
    print("SURROGATE-ONLY RESULTS (--no-pde)")
    print("=" * 120)

    # Header
    noise_labels = [nc["label"] for nc in NOISE_CONFIGS]
    header = f"{'Model':<15s}"
    for nl in noise_labels:
        header += f" | {nl:>14s}"
    print(header)
    print("-" * len(header))

    # Max error table
    print("\n--- Max Error (%) ---")
    print(header)
    print("-" * len(header))
    for model in MODELS:
        row = f"{model:<15s}"
        for nl in noise_labels:
            p = results.get((model, nl), {})
            val = p.get("max_err")
            row += f" | {val:>13.2f}%" if val is not None else " |           N/A"
        print(row)

    # Per-parameter tables
    for param_name in ["k0_1", "k0_2", "alpha_1", "alpha_2"]:
        print(f"\n--- {param_name} Error (%) ---")
        print(header)
        print("-" * len(header))
        for model in MODELS:
            row = f"{model:<15s}"
            for nl in noise_labels:
                p = results.get((model, nl), {})
                val = p.get(param_name)
                row += f" | {val:>13.2f}%" if val is not None else " |           N/A"
            print(row)

    # Summary statistics
    print(f"\n--- Summary: Mean Max Error (%) across 2% noise seeds ---")
    noise_seed_labels = [nc["label"] for nc in NOISE_CONFIGS[1:]]  # skip 0%
    for model in MODELS:
        errs = []
        for nl in noise_seed_labels:
            p = results.get((model, nl), {})
            v = p.get("max_err")
            if v is not None:
                errs.append(v)
        if errs:
            mean_e = sum(errs) / len(errs)
            med_e = sorted(errs)[len(errs) // 2]
            min_e = min(errs)
            max_e = max(errs)
            print(f"  {model:<15s}: mean={mean_e:.2f}%, median={med_e:.2f}%, min={min_e:.2f}%, max={max_e:.2f}%")

    # Print any failures
    failures = [(k, v) for k, v in results.items() if "error" in v]
    if failures:
        print(f"\n--- FAILURES ({len(failures)}) ---")
        for (model, label), parsed in failures:
            print(f"  {model} / {label}: {parsed['error']}")
            if "tail" in parsed:
                print(f"    Last output:\n{parsed['tail'][:500]}")


if __name__ == "__main__":
    main()
