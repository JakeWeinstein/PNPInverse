"""V20 — Compare TRF results between G0 (current grid) and G2 (densified grid).

Reads result.json from:
  - StudyResults/v19_lograte_extended_trf_clean/<init>_v2_initcache/  (G0)
  - StudyResults/v20_best_grid_trf_clean/trf_init_<init>/  (G2)

For each init: tabulate parameter errors, cost, status. Identify which
inits (if any) improved with G2.
"""
from __future__ import annotations

import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


INITS = ["plus20", "minus20", "k0high_alow", "k0low_ahigh"]


def load_result(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)["result"]


def main() -> None:
    G0_DIR = os.path.join(_ROOT, "StudyResults",
                          "v19_lograte_extended_trf_clean")
    G2_DIR = os.path.join(_ROOT, "StudyResults", "v20_best_grid_trf_clean")
    OUT = os.path.join(G2_DIR, "summary.md")

    rows = []
    print("=" * 110)
    print(f"  {'init':<14} {'grid':<3} {'k0_1':>8} {'k0_2':>8} "
          f"{'a_1':>8} {'a_2':>8} {'<5%count':>9} {'cost':>10} {'wall':>5} {'status'}")
    print("=" * 110)
    for init in INITS:
        g0 = load_result(os.path.join(G0_DIR, f"{init}_v2_initcache",
                                       "result.json"))
        g2 = load_result(os.path.join(G2_DIR, f"trf_init_{init}",
                                       "result.json"))
        for tag, r in [("G0", g0), ("G2", g2)]:
            if r is None:
                print(f"  {init:<14} {tag:<3} N/A")
                continue
            errs = [
                r["k0_1_err_pct"], r["k0_2_err_pct"],
                r["alpha_1_err_pct"], r["alpha_2_err_pct"],
            ]
            n_under_5 = sum(1 for e in errs if abs(e) < 5)
            print(f"  {init:<14} {tag:<3} "
                  f"{errs[0]:+8.2f} {errs[1]:+8.2f} "
                  f"{errs[2]:+8.2f} {errs[3]:+8.2f} "
                  f"{n_under_5:>9d} {r['cost_final']:>10.3g} "
                  f"{r['wall_minutes']:>5.1f} {r['status']}")
            rows.append({
                "init": init, "grid": tag,
                "k0_1": errs[0], "k0_2": errs[1],
                "alpha_1": errs[2], "alpha_2": errs[3],
                "n_under_5pct": n_under_5,
                "cost": r["cost_final"],
                "wall_minutes": r["wall_minutes"],
                "status": r["status"],
                "message": r.get("message", ""),
            })

    print()
    # Summary deltas
    print("Per-init G2 vs G0 deltas:")
    for init in INITS:
        g0 = next((r for r in rows if r["init"] == init and r["grid"] == "G0"),
                  None)
        g2 = next((r for r in rows if r["init"] == init and r["grid"] == "G2"),
                  None)
        if g0 is None or g2 is None:
            continue
        d_cost = g2["cost"] - g0["cost"]
        d_under5 = g2["n_under_5pct"] - g0["n_under_5pct"]
        verdict = "BETTER" if (d_cost < 0 or d_under5 > 0) else (
            "WORSE" if (d_cost > 0 or d_under5 < 0) else "SAME")
        print(f"  {init:<14}  d_cost={d_cost:+.3g}  d_under5={d_under5:+d}  → {verdict}")

    # Write summary.md
    with open(OUT, "w") as f:
        f.write("# V20 Task C — TRF on G2 grid (clean data)\n\n")
        f.write("Compares TRF inverse on G0 (7-V grid) vs G2 (10-V grid with "
                "+0.00, +0.15, +0.25 added).\n\n")
        f.write("Setup: bv_log_rate=True, observables=CD+PC, regularization=none, "
                "init_cache=cold-solve at INIT, σ=2%×max|target|.\n\n")
        f.write("## Results\n\n")
        f.write("| init | grid | k0_1 err | k0_2 err | α_1 err | α_2 err | <5% count | cost | wall (min) | status |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for r in rows:
            f.write(f"| {r['init']} | {r['grid']} | "
                    f"{r['k0_1']:+.2f}% | {r['k0_2']:+.2f}% | "
                    f"{r['alpha_1']:+.2f}% | {r['alpha_2']:+.2f}% | "
                    f"{r['n_under_5pct']} | {r['cost']:.3g} | "
                    f"{r['wall_minutes']:.1f} | {r['status']} |\n")

        f.write("\n## Per-init verdict (G2 vs G0)\n\n")
        f.write("| init | Δcost | Δ(<5% count) | verdict |\n")
        f.write("|---|---:|---:|---|\n")
        for init in INITS:
            g0 = next((r for r in rows if r["init"] == init and r["grid"] == "G0"),
                      None)
            g2 = next((r for r in rows if r["init"] == init and r["grid"] == "G2"),
                      None)
            if g0 is None or g2 is None:
                f.write(f"| {init} | — | — | INCOMPLETE |\n")
                continue
            d_cost = g2["cost"] - g0["cost"]
            d_under5 = g2["n_under_5pct"] - g0["n_under_5pct"]
            verdict = "BETTER" if (d_cost < 0 or d_under5 > 0) else (
                "WORSE" if (d_cost > 0 or d_under5 < 0) else "SAME")
            f.write(f"| {init} | {d_cost:+.3g} | {d_under5:+d} | {verdict} |\n")

        # Pass criterion from handoff Task C
        f.write("\n## Handoff Task C pass criterion\n\n")
        f.write("- ≥1 init recovers all 4 params to <10%: ")
        any_under_10 = any(r["grid"] == "G2" and
                          all(abs(r[p]) < 10 for p in
                              ("k0_1", "k0_2", "alpha_1", "alpha_2"))
                          for r in rows)
        f.write("**PASS**\n" if any_under_10 else "FAIL\n")
        f.write("- α stays <5% in most inits: ")
        n_alpha_under5 = sum(1 for r in rows if r["grid"] == "G2" and
                            abs(r["alpha_1"]) < 5 and abs(r["alpha_2"]) < 5)
        f.write(f"{n_alpha_under5}/4 inits\n")
        f.write("- Stalled high-cost endpoints reduced: ")
        n_high_g0 = sum(1 for r in rows if r["grid"] == "G0" and r["cost"] > 100)
        n_high_g2 = sum(1 for r in rows if r["grid"] == "G2" and r["cost"] > 100)
        f.write(f"G0: {n_high_g0}/4, G2: {n_high_g2}/4\n")

    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
