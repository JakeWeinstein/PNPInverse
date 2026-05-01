"""V19 Stage 2 — noise-model FIM audit.

Per GPT's `PNP Log Rate Next Steps Handoff.md` Task 2: re-whiten the existing
S_cd, S_pc sensitivities from `StudyResults/v19_bv_lograte_audit/extended_v_to_60/`
under multiple noise models to test whether the FIM ridge-breaking survives
realistic measurement noise.

Noise models:
  A. global 2% max:        sigma_y = 0.02 * max_V(|y(V)|)
  B. local 2% relative:    sigma_y(V) = 0.02 * |y(V)|
  C. local 2% + abs floor: sigma_y(V) = sqrt((0.02*|y(V)|)^2 + sigma_abs^2)

For C, sweep sigma_abs over a range of absolute floors per observable type.

This is post-processing only — reuses S_cd_raw, S_pc_raw matrices already
saved at TRUE parameters.  No new forward solves.
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
    names = ("log_k0_1", "log_k0_2", "alpha_1", "alpha_2")
    return {
        "n_residuals": int(S_white.shape[0]),
        "n_params": int(S_white.shape[1]),
        "singular_values": sv.tolist(),
        "fim_eigenvalues": evals.tolist(),
        "condition_number": cond_F,
        "weak_eigvec": dict(zip(names, weak_v.tolist())),
        "weak_eigvec_canonical_ridge_cos": cos_sim,
    }


def main() -> None:
    in_dir = os.path.join(_ROOT, "StudyResults", "v19_bv_lograte_audit", "extended_v_to_60")
    fim_path = os.path.join(in_dir, "fim_by_cap.json")
    obs_path = os.path.join(in_dir, "observables_by_cap.csv")

    with open(fim_path) as f:
        fim_data = json.load(f)
    cap_data = fim_data["50"]
    S_cd = np.array(cap_data["S_cd_raw"])
    S_pc = np.array(cap_data["S_pc_raw"])

    # Load targets per V
    cds = []; pcs = []; voltages = []
    with open(obs_path) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            if row["cap"] != "50" or row["converged"] != "True":
                continue
            voltages.append(float(row["V_RHE"]))
            cds.append(float(row["cd"]))
            pcs.append(float(row["pc"]))
    cd_arr = np.array(cds); pc_arr = np.array(pcs)
    V = np.array(voltages)
    NV = len(V)
    assert S_cd.shape == (NV, 4), f"S_cd shape {S_cd.shape} != ({NV}, 4)"
    assert S_pc.shape == (NV, 4), f"S_pc shape {S_pc.shape} != ({NV}, 4)"

    print(f"V_GRID:    {V.tolist()}")
    print(f"NV:        {NV}")
    print(f"|cd| range: [{abs(cd_arr).min():.3e}, {abs(cd_arr).max():.3e}]")
    print(f"|pc| range: [{abs(pc_arr).min():.3e}, {abs(pc_arr).max():.3e}]")
    print()

    OUT_DIR = os.path.join(_ROOT, "StudyResults", "v19_lograte_noise_model_fim")
    os.makedirs(OUT_DIR, exist_ok=True)

    # Noise model definitions
    cd_max = float(np.max(np.abs(cd_arr)))
    pc_max = float(np.max(np.abs(pc_arr)))
    rel = 0.02

    def sigma_global_max():
        return rel * cd_max * np.ones(NV), rel * pc_max * np.ones(NV)

    def sigma_local_rel():
        return rel * np.abs(cd_arr), rel * np.abs(pc_arr)

    def sigma_local_floor(sigma_abs_cd, sigma_abs_pc):
        s_cd = np.sqrt((rel * np.abs(cd_arr))**2 + sigma_abs_cd**2)
        s_pc = np.sqrt((rel * np.abs(pc_arr))**2 + sigma_abs_pc**2)
        return s_cd, s_pc

    # We use the same absolute-floor magnitude for both observables, since they
    # are both currents (mA/cm^2 in the I_SCALE convention).  The user can
    # adjust per-observable in a future audit.
    floors = [1e-6, 1e-7, 1e-8, 1e-9]

    models = {}
    s_cd, s_pc = sigma_global_max()
    models["A_global_2pct_max"] = (s_cd, s_pc)
    s_cd, s_pc = sigma_local_rel()
    models["B_local_2pct"] = (s_cd, s_pc)
    for f in floors:
        s_cd, s_pc = sigma_local_floor(f, f)
        models[f"C_local_2pct_floor_{f:.0e}"] = (s_cd, s_pc)

    # Compute FIM per model
    rows = []
    for name, (s_cd, s_pc) in models.items():
        S_cd_white = S_cd / s_cd[:, None]
        S_pc_white = S_pc / s_pc[:, None]
        S_both = np.vstack([S_cd_white, S_pc_white])
        m = fim_metrics(S_both)
        sv_min = m["singular_values"][-1]
        sv_max = m["singular_values"][0]
        cond = m["condition_number"]
        ridge = m["weak_eigvec_canonical_ridge_cos"]
        we = m["weak_eigvec"]
        we_str = (f"[{we['log_k0_1']:+.3f},{we['log_k0_2']:+.3f},"
                  f"{we['alpha_1']:+.3f},{we['alpha_2']:+.3f}]")
        rows.append({
            "noise_model": name,
            "sigma_cd_min": float(s_cd.min()),
            "sigma_cd_max": float(s_cd.max()),
            "sigma_pc_min": float(s_pc.min()),
            "sigma_pc_max": float(s_pc.max()),
            "sv_min": float(sv_min),
            "sv_max": float(sv_max),
            "cond_F": cond,
            "ridge_cos": ridge,
            "weak_eigvec": we,
            "weak_eigvec_str": we_str,
        })
        print(f"  {name:<32}  sv_min={sv_min:>10.3e}  cond={cond:>10.2e}  "
              f"ridge_cos={ridge:>6.3f}  weak={we_str}")

    # Save outputs
    with open(os.path.join(OUT_DIR, "fim_by_noise_model.json"), "w") as f:
        json.dump({
            "config": {
                "input_dir": in_dir,
                "V_GRID": V.tolist(),
                "n_residuals": 2 * NV,
                "n_params": 4,
                "rel_noise": rel,
                "abs_floors": floors,
            },
            "results": rows,
        }, f, indent=2)

    with open(os.path.join(OUT_DIR, "fim_by_noise_model.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["noise_model", "sigma_cd_min", "sigma_cd_max",
                    "sigma_pc_min", "sigma_pc_max",
                    "sv_min", "sv_max", "cond_F", "ridge_cos",
                    "weak_eigvec_str"])
        for r in rows:
            w.writerow([r["noise_model"],
                        r["sigma_cd_min"], r["sigma_cd_max"],
                        r["sigma_pc_min"], r["sigma_pc_max"],
                        r["sv_min"], r["sv_max"], r["cond_F"], r["ridge_cos"],
                        r["weak_eigvec_str"]])

    with open(os.path.join(OUT_DIR, "weak_eigvecs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["noise_model", "log_k0_1", "log_k0_2", "alpha_1", "alpha_2",
                    "ridge_cos"])
        for r in rows:
            we = r["weak_eigvec"]
            w.writerow([r["noise_model"], we["log_k0_1"], we["log_k0_2"],
                        we["alpha_1"], we["alpha_2"], r["ridge_cos"]])

    # Summary markdown
    summary_path = os.path.join(OUT_DIR, "summary.md")
    with open(summary_path, "w") as fout:
        fout.write("# V19 — noise-model FIM audit\n\n")
        fout.write("Re-whitening of existing S_cd, S_pc sensitivities at TRUE params under\n")
        fout.write("multiple noise models.  Forward data: `StudyResults/v19_bv_lograte_audit/extended_v_to_60/`\n")
        fout.write(f"(cap=50, log-rate ON, V_GRID={V.tolist()}).\n\n")
        fout.write(f"|cd| range: [{abs(cd_arr).min():.3e}, {abs(cd_arr).max():.3e}]  (mA/cm^2 in I_SCALE units)\n\n")
        fout.write(f"|pc| range: [{abs(pc_arr).min():.3e}, {abs(pc_arr).max():.3e}]\n\n")
        fout.write("## Results\n\n")
        fout.write("| noise_model | sv_min | cond(F) | ridge_cos | weak_eigvec |\n")
        fout.write("|---|---:|---:|---:|---|\n")
        for r in rows:
            fout.write(f"| {r['noise_model']} | {r['sv_min']:.3e} | "
                       f"{r['cond_F']:.2e} | {r['ridge_cos']:.3f} | "
                       f"{r['weak_eigvec_str']} |\n")
        fout.write("\n")
        fout.write("## Interpretation rule (per GPT plan)\n\n")
        fout.write("Pass: under realistic σ_abs the FIM still has cond ≤ ~1e8-1e9,\n")
        fout.write("ridge_cos not near 1, weak eigvec not pure log_k0_2.\n\n")
        fout.write("Note: σ_abs values are in the same units as cd, pc (mA/cm^2 in I_SCALE\n")
        fout.write("convention).  At V=+0.60 V, |cd| ≈ 1.8e-9 — sub-1e-6 floors essentially\n")
        fout.write("zero out the high-V signal and we expect the FIM to revert toward the\n")
        fout.write("baseline single-experiment ridge.\n")

    print()
    print(f"Saved: {os.path.join(OUT_DIR, 'fim_by_noise_model.json')}")
    print(f"Saved: {os.path.join(OUT_DIR, 'fim_by_noise_model.csv')}")
    print(f"Saved: {os.path.join(OUT_DIR, 'weak_eigvecs.csv')}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
