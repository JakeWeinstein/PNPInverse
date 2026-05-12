"""Throwaway validation script for Phase D Opt F (post Opt-A revert).

Runs a 6-V anodic production-mode benchmark to confirm:
    (a) per-V wall is back to ~10 s (matching the prior
        ``benchmark_6v_production.json`` baseline before the Opt-A
        prototype),
    (b) the Optimization F anchor cache HITs on the second eval and
        skips the ~70-sec anchor solve.

Run from PNPInverse/ with the venv-firedrake activated::

    source ../venv-firedrake/bin/activate
    MPLCONFIGDIR=/tmp XDG_CACHE_HOME=/tmp PYOP2_CACHE_DIR=/tmp/pyop2 \\
        FIREDRAKE_TSFC_KERNEL_CACHE_DIR=/tmp/firedrake-tsfc OMP_NUM_THREADS=1 \\
        python -u scripts/studies/_phase_D_validate_optimized_path.py
"""
import glob
import os
import sys
import time

# Ensure the repo root is on sys.path so the scripts.* imports resolve
# regardless of where the script is invoked from.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main() -> int:
    from scripts.studies.phase6b_step10_phase_D_fit_eval import (
        V_ANCHOR, evaluate_delta_beta,
    )

    # 6-V anodic production-mode benchmark (matches the user's spec).
    # Anchor at +0.55; visit anodic-side neighbours that the per-V λ=1
    # ramp can warm-start through cleanly.  No V_KIN here -- this is a
    # wall-time + cache-HIT smoke test only.
    v_grid = (0.55, 0.50, 0.45, 0.40, 0.35, 0.30)

    cache_dir = os.path.join(
        _ROOT, "StudyResults", "phase6b_step10_phase_D"
    )
    cache_pattern = os.path.join(cache_dir, "anchor_cache_*.pkl")

    # Fresh cache: delete all anchor_cache_*.pkl files
    n_pre = 0
    for p in glob.glob(cache_pattern):
        os.remove(p)
        n_pre += 1
    print(f"[setup] deleted {n_pre} pre-existing anchor cache files",
          flush=True)

    # Run 1: cold cache
    print(
        f"\n=== Run 1: COLD CACHE, Δ_β=0, σ=stern, "
        f"{len(v_grid)}-V grid (production) ===",
        flush=True,
    )
    t0 = time.time()
    result1 = evaluate_delta_beta(
        delta_beta_pm2=0.0, sigma_mapping="stern",
        v_grid=v_grid, v_anchor=V_ANCHOR, mode="production",
    )
    t1 = time.time() - t0
    print(f"\nRun 1 wall: {t1:.1f}s", flush=True)
    cache1 = result1["config"]["anchor_cache"]
    print(
        f"Run 1 anchor_cache: status={cache1['status']}, "
        f"hash={cache1['config_hash']}",
        flush=True,
    )
    n_converged1 = sum(
        1 for r in result1["per_v_records"] if r.get("snes_converged")
    )
    print(
        f"Run 1 converged: {n_converged1}/{len(result1['per_v_records'])}",
        flush=True,
    )

    # Run 2: same σ-mapping, different Δ_β — should HIT the cache.
    print(
        "\n=== Run 2: WARM CACHE, Δ_β=1e3, σ=stern "
        "(anchor cache HIT expected) ===",
        flush=True,
    )
    t0 = time.time()
    result2 = evaluate_delta_beta(
        delta_beta_pm2=1e3, sigma_mapping="stern",
        v_grid=v_grid, v_anchor=V_ANCHOR, mode="production",
    )
    t2 = time.time() - t0
    print(f"\nRun 2 wall: {t2:.1f}s", flush=True)
    cache2 = result2["config"]["anchor_cache"]
    print(
        f"Run 2 anchor_cache: status={cache2['status']}, "
        f"hash={cache2['config_hash']}",
        flush=True,
    )
    n_converged2 = sum(
        1 for r in result2["per_v_records"] if r.get("snes_converged")
    )
    print(
        f"Run 2 converged: {n_converged2}/{len(result2['per_v_records'])}",
        flush=True,
    )

    saved = t1 - t2
    print(
        f"\n=== Summary ===\n"
        f"  Run 1 (cold): {t1:.1f}s, anchor_cache={cache1['status']}\n"
        f"  Run 2 (warm): {t2:.1f}s, anchor_cache={cache2['status']}\n"
        f"  wall savings (cold - warm): {saved:.1f}s",
        flush=True,
    )

    cache_hit_ok = cache2["status"] == "hit"
    saved_ok = saved >= 30.0
    print(
        f"\nVerdict: cache_HIT={'PASS' if cache_hit_ok else 'FAIL'}, "
        f"wall_savings_>=30s={'PASS' if saved_ok else 'FAIL'}",
        flush=True,
    )
    return 0 if (cache_hit_ok and saved_ok) else 2


if __name__ == "__main__":
    sys.exit(main())
