#!/usr/bin/env python3
"""Rank Phase 7 dual-pathway sweep corners against the v2 slide-15 target.

Usage:  python scripts/studies/_score_phase7_sweeps.py [glob ...]
Default glob: StudyResults/phase7_dual_pathway/*/iv_curve.json
"""
from __future__ import annotations

import glob
import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from calibration.phase7_wls import load_target, score_iv_json

TARGET = os.path.join(_ROOT, "data", "mangan_deck_p15_h2o2_current_v2.csv")


def main() -> int:
    patterns = sys.argv[1:] or [
        os.path.join(_ROOT, "StudyResults", "phase7_dual_pathway",
                     "*", "iv_curve.json"),
    ]
    target = load_target(TARGET)
    rows = []
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            name = os.path.basename(os.path.dirname(path))
            rep = json.load(open(path))
            if not rep.get("anchor_converged"):
                rows.append((float("inf"), name, "anchor FAILED", "", ""))
                continue
            try:
                res = score_iv_json(rep, target)
            except Exception as exc:
                rows.append((float("inf"), name,
                             f"unscoreable: {exc}", "", ""))
                continue
            pcs = [p for p in rep["pc_mA_cm2"] if p is not None]
            vds = [v for v, p in zip(rep["v_rhe_deck"], rep["pc_mA_cm2"])
                   if p is not None]
            if pcs:
                ipk = pcs.index(min(pcs))
                peak = f"pc_min={min(pcs):+.3f}@{vds[ipk]:+.2f}"
            else:
                peak = "no pc"
            conv = f"{rep['n_converged']}/{rep['n_total']}"
            extra = (f"hinge={res.hinge_penalty:.2f}"
                     + (f" INVALID:{res.validity_failures}"
                        if res.validity_failures else ""))
            rows.append((res.total, name,
                         f"chi2/pt={res.chi2_per_point:8.2f}", peak,
                         f"{conv} {extra}"))
    rows.sort(key=lambda r: r[0])
    print(f"{'total':>10}  {'corner':<28} {'fit':<18} {'peak':<24} notes")
    for total, name, fit, peak, notes in rows:
        t = f"{total:10.2f}" if total != float("inf") else "       inf"
        print(f"{t}  {name:<28} {fit:<18} {peak:<24} {notes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
