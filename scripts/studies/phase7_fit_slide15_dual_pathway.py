"""Phase 7 stage-5b fit: water-route kinetics vs the v2 slide-15 target.

Nelder-Mead over theta = (log10 f_k0_w2e, log10 f_k0_w4e, alpha_w2e,
alpha_w4e), water routes only (acid k0=0), coarse 13-pt grid per
evaluation, anchor per-theta (no cross-theta warm starts).  Objective =
WLS total (chi2/pt + sanity hinges) from ``calibration.phase7_wls``
plus validity/bound penalties.

Every evaluation is checkpointed to
``StudyResults/phase7_fit/evals/eval_<n>.json`` (theta, score, full
iv report) so the run is resumable/auditable and 1D profile slices can
be cut afterward (identifiability artifact per critique R1#12).

x0 = sweepB_center (analytic Tafel calibration, chi2/pt = 118.8):
f_2w=1e-3, f_4w=4e-15, a2=0.627, a4=0.325.

Usage (background, ~6-7 min/eval, default cap 60 evals ~ 6.5 h):
    python -u scripts/studies/phase7_fit_slide15_dual_pathway.py \\
        [--maxfev 60] [--out-name phase7_fit]
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(line_buffering=True)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

X0 = (np.log10(1e-3), np.log10(4e-15), 0.627, 0.325)
SIMPLEX_STEPS = (0.5, 0.5, 0.06, 0.02)   # initial simplex spread per param

LOG10_K2_BOUNDS = (-8.0, 1.0)
LOG10_K4_BOUNDS = (-22.0, -8.0)
ALPHA_BOUNDS = (0.05, 0.95)
BOUND_PENALTY = 1e4
VALIDITY_PENALTY = 1e3
FAILED_RUN_SCORE = 1e6


def _bound_penalty(x, lo, hi):
    if x < lo:
        return BOUND_PENALTY * (lo - x) ** 2
    if x > hi:
        return BOUND_PENALTY * (x - hi) ** 2
    return 0.0


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--maxfev", type=int, default=60)
    parser.add_argument("--out-name", default="phase7_fit")
    parser.add_argument("--fine-grid-final", action="store_true", default=True)
    args = parser.parse_args()

    out_dir = Path(_ROOT) / "StudyResults" / args.out_name
    evals_dir = out_dir / "evals"
    evals_dir.mkdir(parents=True, exist_ok=True)

    from calibration.phase7_wls import load_target, score_iv_json
    import scripts.studies.solver_demo_slide15_dual_pathway_cs as dp

    target = load_target(
        os.path.join(_ROOT, "data", "mangan_deck_p15_h2o2_current_v2.csv")
    )

    eval_count = [0]
    best = {"score": float("inf"), "theta": None, "n": None}

    def _opts(theta, coarse=True):
        lg_k2, lg_k4, a2, a4 = theta
        return SimpleNamespace(
            routes="water",
            k0_water_2e_factor=float(10.0 ** lg_k2),
            k0_water_4e_factor=float(10.0 ** lg_k4),
            k0_acid_4e_factor=1e-15,        # inert (acid routes k0=0)
            alpha_water_2e=float(a2),
            alpha_water_4e=float(a4),
            l_eff_um=15.4,
            bulk_h_mol_m3=0.1,
            enable_water_ionization=True,
            coarse_grid=bool(coarse),
        )

    def objective(theta):
        n = eval_count[0]
        eval_count[0] += 1
        t0 = time.time()
        pen = (
            _bound_penalty(theta[0], *LOG10_K2_BOUNDS)
            + _bound_penalty(theta[1], *LOG10_K4_BOUNDS)
            + _bound_penalty(theta[2], *ALPHA_BOUNDS)
            + _bound_penalty(theta[3], *ALPHA_BOUNDS)
        )
        if pen >= BOUND_PENALTY:    # far out of bounds: skip the solve
            print(f"[eval {n:03d}] OUT OF BOUNDS theta={np.round(theta, 4)} "
                  f"pen={pen:.1f}", flush=True)
            return FAILED_RUN_SCORE + pen

        try:
            report = dp._run(_opts(theta))
        except Exception as exc:
            print(f"[eval {n:03d}] RUN ERROR {type(exc).__name__}: {exc}",
                  flush=True)
            report = {"anchor_converged": False, "error": str(exc)}

        if not report.get("anchor_converged"):
            score = FAILED_RUN_SCORE
            detail = "anchor failed"
        else:
            try:
                res = score_iv_json(report, target)
                score = (res.total + pen
                         + VALIDITY_PENALTY * len(res.validity_failures))
                detail = (f"chi2/pt={res.chi2_per_point:.2f} "
                          f"hinge={res.hinge_penalty:.2f} "
                          f"valid_fail={res.validity_failures}")
            except Exception as exc:
                score = FAILED_RUN_SCORE
                detail = f"unscoreable: {exc}"

        wall = time.time() - t0
        rec = {
            "n": n,
            "theta": [float(t) for t in theta],
            "params": {
                "k0_water_2e_factor": float(10.0 ** theta[0]),
                "k0_water_4e_factor": float(10.0 ** theta[1]),
                "alpha_water_2e": float(theta[2]),
                "alpha_water_4e": float(theta[3]),
            },
            "score": float(score),
            "detail": detail,
            "wall_seconds": wall,
            "report": report,
        }
        with open(evals_dir / f"eval_{n:03d}.json", "w") as fh:
            json.dump(rec, fh, indent=1)
        marker = ""
        if score < best["score"]:
            best.update(score=score, theta=list(map(float, theta)), n=n)
            marker = "  <-- BEST"
        print(f"[eval {n:03d}] score={score:10.3f}  "
              f"theta=({theta[0]:+.3f},{theta[1]:+.3f},"
              f"{theta[2]:.3f},{theta[3]:.3f})  {detail}  "
              f"{wall:.0f}s{marker}", flush=True)
        return score

    from scipy.optimize import minimize

    # Initial simplex: x0 plus per-param steps (Nelder-Mead default
    # simplex is too tiny for log-space kinetics).
    sim = [np.array(X0, dtype=float)]
    for k, step in enumerate(SIMPLEX_STEPS):
        v = np.array(X0, dtype=float)
        v[k] += step
        sim.append(v)

    t0 = time.time()
    result = minimize(
        objective, np.array(X0, dtype=float), method="Nelder-Mead",
        options={
            "maxfev": int(args.maxfev),
            "initial_simplex": np.array(sim),
            "xatol": 1e-3, "fatol": 0.5,
        },
    )
    print(f"\nNM done: {result.message}  nfev={result.nfev}  "
          f"best score={best['score']:.3f} @ eval {best['n']}", flush=True)

    summary = {
        "x0": list(X0),
        "best_theta": best["theta"],
        "best_score_coarse": best["score"],
        "best_eval_n": best["n"],
        "nfev": int(result.nfev),
        "message": str(result.message),
        "wall_seconds": time.time() - t0,
    }

    # Final: best theta on the fine 25-pt grid.
    if args.fine_grid_final and best["theta"] is not None:
        print("\nfinal re-run of best theta on fine grid...", flush=True)
        report = dp._run(_opts(best["theta"], coarse=False))
        with open(out_dir / "best_theta_fine_grid.json", "w") as fh:
            json.dump({"theta": best["theta"], "report": report}, fh, indent=1)
        if report.get("anchor_converged"):
            res = score_iv_json(report, target)
            summary["best_score_fine"] = float(res.total)
            summary["best_fine_chi2_per_point"] = float(res.chi2_per_point)
            summary["best_fine_n_converged"] = report.get("n_converged")
            print(f"fine-grid: chi2/pt={res.chi2_per_point:.2f} "
                  f"({report.get('n_converged')}/{report.get('n_total')} pts)",
                  flush=True)

    with open(out_dir / "fit_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {out_dir / 'fit_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
