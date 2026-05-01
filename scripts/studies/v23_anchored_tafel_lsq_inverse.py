"""V23 Phase 1 — Anchored Tafel-coordinate TRF orchestrator.

Per HANDOFF_11 §4. Runs v18_logc_lsq_inverse.py with --anchor-tafel for
3 anchor pairs × 4 standard inits = 12 runs.

Anchored coords:
    beta_j = log_k0_j - alpha_j * n_e * (V_anchor_j - E_eq_j) / V_T

The forward solver still takes physical (log_k0_j, alpha_j); v18's
optimizer interface transforms x = [beta_1, beta_2, alpha_1, alpha_2]
to physical before each forward call and chain-rules the Jacobian.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ANCHOR_PAIRS = [
    (0.25, 0.50),
    (0.30, 0.50),
    (0.30, 0.55),
]
INITS = ["plus20", "minus20", "k0high_alow", "k0low_ahigh"]
V_GRID = ["-0.10", "0.10", "0.20", "0.30", "0.40", "0.50", "0.60"]
OUT_BASE = "v23_anchored_tafel_parameterization"


def _here() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _spawn_one(v_anc_1: float, v_anc_2: float, init: str,
               sub: str, master_log: Path) -> int:
    cmd = [
        sys.executable,
        "scripts/studies/v18_logc_lsq_inverse.py",
        "--method", "trf",
        "--log-rate",
        "--v-grid", *V_GRID,
        "--init", init,
        "--anchor-tafel",
        "--v-anchor-1", f"{v_anc_1:.3f}",
        "--v-anchor-2", f"{v_anc_2:.3f}",
        "--out-base", OUT_BASE,
        "--out_subdir", sub,
    ]
    with open(master_log, "a") as f:
        f.write(f"\n=== START {sub} {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        f.write(f"cmd: {' '.join(cmd)}\n")
        f.flush()
        res = subprocess.run(cmd, cwd=_here(), stdout=f, stderr=subprocess.STDOUT)
        f.write(f"=== END   {sub} exit={res.returncode} "
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    return res.returncode


def _classify(r: dict) -> str:
    k02 = r["k0_2_err_pct"]
    others = [abs(r["k0_1_err_pct"]), abs(r["alpha_1_err_pct"]),
              abs(r["alpha_2_err_pct"])]
    if abs(k02) < 10 and all(e < 10 for e in others):
        return "TRUE_BASIN"
    if k02 > 30:
        return "WRONG_HIGH"
    if k02 < -30:
        return "WRONG_LOW"
    return "OTHER"


def _aggregate(out_root: Path) -> dict:
    rows = []
    for v_anc_1, v_anc_2 in ANCHOR_PAIRS:
        for init in INITS:
            sub = f"a{v_anc_1:.2f}_{v_anc_2:.2f}_{init}".replace(".", "p")
            path = out_root / sub / "result.json"
            if not path.exists():
                rows.append({"sub": sub, "v_anc_1": v_anc_1, "v_anc_2": v_anc_2,
                             "init": init, "status": "MISSING"})
                continue
            with open(path) as f:
                d = json.load(f)
            r = d["result"]
            rows.append({
                "sub": sub, "v_anc_1": v_anc_1, "v_anc_2": v_anc_2,
                "init": init,
                "beta_1": r.get("beta_1"), "beta_2": r.get("beta_2"),
                "k0_1_err_pct": r["k0_1_err_pct"],
                "k0_2_err_pct": r["k0_2_err_pct"],
                "alpha_1_err_pct": r["alpha_1_err_pct"],
                "alpha_2_err_pct": r["alpha_2_err_pct"],
                "cost_final": r["cost_final"],
                "n_evals": r["n_evals"],
                "wall_min": r["wall_minutes"],
                "lt5_count": int(abs(r["k0_1_err_pct"]) < 5)
                            + int(abs(r["k0_2_err_pct"]) < 5)
                            + int(abs(r["alpha_1_err_pct"]) < 5)
                            + int(abs(r["alpha_2_err_pct"]) < 5),
                "basin": _classify(r),
                "status": "OK",
            })
    return {"rows": rows}


def _write_summary(out_root: Path, agg: dict) -> None:
    rows = agg["rows"]
    import csv
    with open(out_root / "by_anchor_pair.csv", "w", newline="") as f:
        keys = ["sub", "v_anc_1", "v_anc_2", "init",
                "beta_1", "beta_2",
                "k0_1_err_pct", "k0_2_err_pct",
                "alpha_1_err_pct", "alpha_2_err_pct",
                "cost_final", "n_evals", "wall_min",
                "lt5_count", "basin", "status"]
        w = csv.writer(f); w.writerow(keys)
        for r in rows:
            w.writerow([r.get(k, "") for k in keys])

    # Per-anchor-pair summary
    lines = [
        "# V23 Phase 1 — Anchored Tafel coordinate TRF\n",
        "Per HANDOFF_11 §4. Optimizer in anchored coords",
        "(beta_1, beta_2, alpha_1, alpha_2) where",
        "beta_j = log_k0_j - alpha_j * n_e * (V_anchor_j - E_eq_j) / V_T.",
        "Forward solver and result reporting in physical coords.\n",
        "## Setup\n",
        "- Anchor pairs: " + str(ANCHOR_PAIRS),
        "- Inits: " + str(INITS),
        "- TRF (bounded), G0, log-rate, no prior.\n",
        "## Per-anchor-pair, per-init result\n",
        "| anchor (V_a1, V_a2) | init | k0_1 err | k0_2 err | α_1 err | α_2 err | cost | <5% | basin |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        if r.get("status") != "OK":
            lines.append(f"| ({r['v_anc_1']:.2f}, {r['v_anc_2']:.2f}) | {r['init']} "
                         f"| — | — | — | — | — | — | {r.get('status','MISSING')} |")
            continue
        lines.append(
            f"| ({r['v_anc_1']:.2f}, {r['v_anc_2']:.2f}) | {r['init']} "
            f"| {r['k0_1_err_pct']:+.2f}% | {r['k0_2_err_pct']:+.2f}% "
            f"| {r['alpha_1_err_pct']:+.2f}% | {r['alpha_2_err_pct']:+.2f}% "
            f"| {r['cost_final']:.2e} | {r['lt5_count']} | {r['basin']} |"
        )
    # Verdict
    lines.append("\n## Verdict\n")
    by_anchor = {}
    for r in rows:
        if r.get("status") != "OK":
            continue
        key = (r["v_anc_1"], r["v_anc_2"])
        by_anchor.setdefault(key, []).append(r)
    for key, bucket in sorted(by_anchor.items()):
        true_count = sum(1 for r in bucket if r["basin"] == "TRUE_BASIN")
        avg_lt5 = sum(r["lt5_count"] for r in bucket) / max(1, len(bucket))
        lines.append(f"- Anchor ({key[0]:.2f}, {key[1]:.2f}): "
                     f"TRUE_BASIN in {true_count}/{len(bucket)} inits, "
                     f"avg <5% count = {avg_lt5:.1f}.")
    # Compare to baseline TRF G0 lt5 = 6 across 4 inits = avg 1.5
    lines.append("\nBaseline (no anchor, TRF G0): avg <5% count = 1.5 (6 of 16 params across 4 inits).")
    lines.append("If avg <5% count > 1.5, anchored coords help.  If equal, anchor is "
                 "scientifically interesting (effective rate identifiable) but doesn't fix "
                 "joint k0 recovery.  If TRUE_BASIN appears in more inits, the anchor "
                 "rotates the basin geometry favorably.")
    with open(out_root / "summary.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(out_root / "endpoints.json", "w") as f:
        json.dump(agg, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    out_root = _here() / "StudyResults" / OUT_BASE
    out_root.mkdir(parents=True, exist_ok=True)
    master_log = out_root / "_master_run.log"

    if not args.aggregate_only:
        with open(master_log, "a") as f:
            f.write(f"\n=== START v23_anchored_tafel "
                    f"({len(ANCHOR_PAIRS)} anchors × {len(INITS)} inits) "
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        for v_anc_1, v_anc_2 in ANCHOR_PAIRS:
            for init in INITS:
                sub = f"a{v_anc_1:.2f}_{v_anc_2:.2f}_{init}".replace(".", "p")
                print(f"[v23-anchored] {sub}", flush=True)
                rc = _spawn_one(v_anc_1, v_anc_2, init, sub, master_log)
                print(f"[v23-anchored] {sub} exit={rc}", flush=True)
        with open(master_log, "a") as f:
            f.write(f"=== END v23_anchored_tafel "
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    agg = _aggregate(out_root)
    _write_summary(out_root, agg)
    print(f"[v23-anchored] Aggregated to {out_root}/{{summary.md,by_anchor_pair.csv,endpoints.json}}")


if __name__ == "__main__":
    main()
