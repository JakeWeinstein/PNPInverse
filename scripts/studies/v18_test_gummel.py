"""V18: Test Gummel solver across the onset region."""
import sys, os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts._bv_common import (
    setup_firedrake_env, V_T,
    FOUR_SPECIES_CHARGED, make_bv_solver_params,
    SNES_OPTS_CHARGED,
)
setup_firedrake_env()

import numpy as np
import time
from Forward.bv_solver.gummel_solver import solve_gummel_steady
from Forward.bv_solver import make_graded_rectangle_mesh

E_EQ_R1 = 0.68
E_EQ_R2 = 1.78


def test_gummel_point(V_RHE):
    phi_hat = V_RHE / V_T
    sp = make_bv_solver_params(
        eta_hat=phi_hat, dt=0.5, t_end=80.0,
        species=FOUR_SPECIES_CHARGED,
        snes_opts=SNES_OPTS_CHARGED,
        E_eq_r1=E_EQ_R1, E_eq_r2=E_EQ_R2,
    )
    mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)

    print(f"\n{'='*60}")
    print(f"V_RHE = {V_RHE:.2f}V (phi_hat = {phi_hat:.2f})")
    print(f"{'='*60}")

    t0 = time.time()
    result = solve_gummel_steady(
        list(sp), mesh=mesh,
        max_gummel_iter=100,
        gummel_rtol=1e-5,
        omega=0.3,
        np_substeps=10,
        verbose=True,
    )
    elapsed = time.time() - t0

    print(f"\nResult: converged={result['converged']}, "
          f"iter={result['n_iter']}, cd={result['cd']:.6f}, "
          f"pc={result['pc']:.6f}, c_min={result['c_min']}, "
          f"time={elapsed:.1f}s")
    return result


def main():
    # Test first in the known-working cathodic regime
    v_points = [-0.3, 0.0, 0.1, 0.15, 0.2, 0.3]

    results = []
    for V in v_points:
        try:
            r = test_gummel_point(V)
            results.append({"V_RHE": V, **{k: v for k, v in r.items()
                                            if k not in ("C", "PHI", "ctx", "mesh")}})
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({"V_RHE": V, "error": str(e)})

    print("\n" + "=" * 70)
    print("GUMMEL SOLVER SUMMARY")
    print("=" * 70)
    for r in results:
        V = r["V_RHE"]
        conv = r.get("converged", False)
        cd = r.get("cd", float("nan"))
        iters = r.get("n_iter", 0)
        print(f"  V={V:6.2f}V: conv={conv}, cd={cd:.6f}, iters={iters}")


if __name__ == "__main__":
    main()
