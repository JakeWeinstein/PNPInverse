"""Profile the production 3sp + bikerman + muh + Stern sweep.

Wraps ``scripts.studies.peroxide_window_3sp_bikerman_muh._run_one_pass``
with cProfile + PETSc ``-log_view``. Runs only the Stern @ 0.10 F/m² pass
on the full 15-V production grid with ``exponent_clip=100`` (the
PC-trustworthy default per CLAUDE.md Hard Rule 2).

Outputs:
    StudyResults/profile_3sp_bikerman_muh_stern/
        profile.prof     # cProfile dump (snakeviz/pstats)
        petsc_log.txt    # PETSc -log_view ascii_info_detail
        summary.txt      # Top-50 cumulative + tottime stats
        iv_curve.json    # Sweep result (so we can spot-check correctness)

Run from PNPInverse/ with venv-firedrake activated:

    source ../venv-firedrake/bin/activate
    python -u scripts/profile/profile_production_sweep.py
"""
from __future__ import annotations

import cProfile
import json
import os
import pstats
import sys
import time
from io import StringIO

# Resolve repo root and prepare output directory BEFORE importing
# firedrake/petsc, so PETSC_OPTIONS routes -log_view to the right file.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))

OUT_DIR = os.path.join(_ROOT, "StudyResults", "profile_3sp_bikerman_muh_stern")
os.makedirs(OUT_DIR, exist_ok=True)

PROFILE_PATH = os.path.join(OUT_DIR, "profile.prof")
PETSC_LOG_PATH = os.path.join(OUT_DIR, "petsc_log.txt")
SUMMARY_PATH = os.path.join(OUT_DIR, "summary.txt")
IV_PATH = os.path.join(OUT_DIR, "iv_curve.json")

# PETSc -log_view writes at PETSc finalisation (program exit).  ascii_info_detail
# expands per-event MFlops/Msgs columns we care about for solver internals.
os.environ.setdefault(
    "PETSC_OPTIONS", f"-log_view :{PETSC_LOG_PATH}:ascii_info_detail"
)

if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Import the production study module (does NOT trigger firedrake import yet —
# the firedrake imports live inside _run_one_pass).
from scripts.studies import peroxide_window_3sp_bikerman_muh as study  # noqa: E402

# Override exponent_clip to 100 (the PC-trustworthy default per CLAUDE.md).
# Module global is read inside _run_one_pass, so monkey-patching here works.
study.EXPONENT_CLIP = 100.0
LABEL = "stern_0p10_clip100_profile"
CS = 0.10


def _print_and_log(text: str, summary_buf: StringIO) -> None:
    print(text)
    summary_buf.write(text + "\n")


def main() -> None:
    summary_buf = StringIO()

    header = (
        "=" * 78 + "\n"
        "  PROFILING: 3sp + bikerman + muh + Stern (cs=0.10), clip=100\n"
        "=" * 78 + "\n"
        f"  output dir: {OUT_DIR}\n"
        f"  profile:    {PROFILE_PATH}\n"
        f"  petsc log:  {PETSC_LOG_PATH}\n"
        f"  V grid:     {list(study.V_TEST)}  (n={len(study.V_TEST)})\n"
        f"  mesh Ny:    {study.MESH_NY}\n"
        f"  formulation:{study.FORMULATION}\n"
        f"  initializer:{study.INITIALIZER}\n"
        f"  clip:       {study.EXPONENT_CLIP} (overridden from script default)\n"
    )
    print(header)
    summary_buf.write(header + "\n")

    prof = cProfile.Profile()
    t0 = time.time()
    prof.enable()
    try:
        report = study._run_one_pass(
            LABEL, CS, v_rhe_grid=list(study.V_TEST)
        )
    finally:
        prof.disable()
    elapsed = time.time() - t0

    prof.dump_stats(PROFILE_PATH)

    # Persist the IV result so we can sanity-check that the profiled run
    # actually converged (vs. profiling a degenerate code path).
    with open(IV_PATH, "w") as f:
        json.dump(
            {
                "label": LABEL,
                "cs_f_m2": CS,
                "exponent_clip": study.EXPONENT_CLIP,
                "v_rhe": list(study.V_TEST),
                "report": {
                    k: v for k, v in report.items() if k != "diagnostics"
                },
            },
            f,
            indent=2,
            default=str,
        )

    summary_buf.write(
        "=" * 78 + "\n"
        f"  Total wall time: {elapsed:.1f}s\n"
        f"  Reported wall:   {report['wall_seconds']:.1f}s\n"
        f"  Converged:       {report['n_converged']}/{report['n_total']}\n"
        + "=" * 78 + "\n\n"
    )

    summary_buf.write("# Per-voltage method × convergence\n")
    for v, ok, m in zip(study.V_TEST, report["converged"], report["method"]):
        summary_buf.write(f"    V={v:+.3f}  ok={ok}  method={m}\n")
    summary_buf.write("\n")

    # Top 50 by cumulative time — best for finding "what called what slowly".
    summary_buf.write("# Top 50 by cumulative time (cumtime)\n")
    stats = pstats.Stats(prof, stream=summary_buf)
    stats.strip_dirs().sort_stats("cumulative").print_stats(50)

    # Top 50 by tottime — best for finding self-time hot spots
    # (excludes time spent in subcalls).
    summary_buf.write("\n# Top 50 by total time excluding subcalls (tottime)\n")
    stats2 = pstats.Stats(prof, stream=summary_buf)
    stats2.strip_dirs().sort_stats("tottime").print_stats(50)

    summary_text = summary_buf.getvalue()
    with open(SUMMARY_PATH, "w") as f:
        f.write(summary_text)

    print(summary_text)
    print(f"  wrote profile -> {PROFILE_PATH}")
    print(f"  wrote summary -> {SUMMARY_PATH}")
    print(f"  wrote iv     -> {IV_PATH}")
    print(f"  petsc log dumped at exit -> {PETSC_LOG_PATH}")


if __name__ == "__main__":
    main()
