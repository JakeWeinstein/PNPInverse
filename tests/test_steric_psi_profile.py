"""Identity tests for the Bikerman composite psi profile (Option 2b).

The composite profile used inside ``_try_debye_boltzmann_ic`` and
``_try_debye_boltzmann_ic_muh`` to seed phi when the analytic counterion
is steric-aware (4sp dynamic ClO4-, or a bikerman-mode counterion entry):

    nu          = 2 * a * c_bulk_charged                  (charged-pair frac)
    psi_sat     = ln(2 / nu)                              (saturation threshold)
    alpha(psi_D)= sqrt((2/(nu*lam_D^2)) * ln[1 + nu*(cosh psi_D - 1)])
    y_match     = (|psi_D| - psi_sat) / alpha(psi_D)

    psi(y) = s * (|psi_D| - alpha * y)            for y in [0, y_match]
    psi(y) = s * psi_sat * exp(-(y-y_match)/lam_D) for y >= y_match

where ``s = sign(psi_D)``.  The tests in this module are formula-level
identities and do not require Firedrake.

P_2b_1  --  psi(y_match) = s*psi_sat exactly (continuity).
P_2b_2  --  |psi(5*lam_D)| < 0.01 * psi_sat (outer decay).
P_2b_3  --  composite vs scipy.integrate.solve_ivp on the Bikerman first
            integral, slow + scipy required.
"""
from __future__ import annotations

import math

import numpy as np
import pytest


# Parameters matching the production setup (a = A_DEFAULT, c_b = C_CLO4_HAT).
from scripts._bv_common import A_DEFAULT as A_NONDIM, C_CLO4_HAT as C_BULK_NONDIM

LAMBDA_D = 0.05  # nondim Debye length; representative production value
NU = 2.0 * A_NONDIM * C_BULK_NONDIM       # 2 * 0.01 * C_CLO4_HAT
PSI_SAT = math.log(2.0 / NU)              # ln(2/NU) -- saturation threshold


def composite_psi(y: float, psi_D: float, *, lam_D: float = LAMBDA_D, nu: float = NU) -> float:
    """Reference Python implementation of the composite psi profile.

    Mirrors the UFL block planned for ``forms_logc.py`` / ``forms_logc_muh.py``
    so the formula is unit-tested independent of Firedrake.
    """
    psi_d_abs = abs(psi_D)
    if nu <= 0.0 or psi_d_abs < 1e-6:
        # Linear-Debye limit -- pure exponential.
        return psi_D * math.exp(-y / lam_D)

    psi_sat_local = math.log(2.0 / nu)

    # Below saturation: pure exponential.
    if psi_d_abs <= psi_sat_local * (1.0 - 1e-3):
        return psi_D * math.exp(-y / lam_D)

    arg_cosh = math.cosh(psi_d_abs)
    alpha = math.sqrt(
        (2.0 / (nu * lam_D ** 2)) * math.log(1.0 + nu * (arg_cosh - 1.0))
    )
    y_match = (psi_d_abs - psi_sat_local) / alpha
    sign_psi_D = 1.0 if psi_D >= 0.0 else -1.0

    if y < y_match:
        return sign_psi_D * (psi_d_abs - alpha * y)
    return sign_psi_D * psi_sat_local * math.exp(-(y - y_match) / lam_D)


def _y_match_alpha(psi_D: float, *, lam_D: float = LAMBDA_D, nu: float = NU) -> tuple[float, float]:
    psi_d_abs = abs(psi_D)
    arg_cosh = math.cosh(psi_d_abs)
    alpha = math.sqrt(
        (2.0 / (nu * lam_D ** 2)) * math.log(1.0 + nu * (arg_cosh - 1.0))
    )
    psi_sat_local = math.log(2.0 / nu)
    y_match = (psi_d_abs - psi_sat_local) / alpha
    return y_match, alpha


@pytest.mark.parametrize("psi_D", [11.7, 19.4, 25.7, 38.9])
def test_psi_continuity_at_y_match(psi_D):
    """P_2b_1: composite psi is continuous at y_match and equals s*psi_sat there.

    Tested at psi_D values corresponding to V_RHE = +0.3, +0.5, +0.66, +1.0 V
    (psi_D ~ V_RHE / V_T with V_T = 0.02569 V).
    """
    y_match, _ = _y_match_alpha(psi_D)
    psi_at_match = composite_psi(y_match, psi_D)
    sign = 1.0 if psi_D >= 0 else -1.0
    expected = sign * PSI_SAT

    # Two different evaluations of the same identity: from inside zone 1
    # (at y just below y_match) and from inside zone 2 (at y just above).
    eps = 1e-12 * y_match
    psi_z1 = composite_psi(y_match - eps, psi_D)
    psi_z2 = composite_psi(y_match + eps, psi_D)

    assert abs(psi_at_match - expected) < 1e-10, (
        f"psi(y_match)={psi_at_match} should equal s*psi_sat={expected}"
    )
    assert abs(psi_z1 - expected) < 1e-9
    assert abs(psi_z2 - expected) < 1e-9


@pytest.mark.parametrize("psi_D", [11.7, 19.4, 25.7, 38.9])
def test_psi_decays_far_from_electrode(psi_D):
    """P_2b_2: composite |psi| < 0.01 * psi_sat at y = 5 * lam_D.

    For our setup psi_sat ~ 6.21 and y_match/lam_D < 0.3 over the tested
    psi_D range, so the outer exponential decay leaves <1% of psi_sat at
    five Debye lengths.
    """
    psi_far = composite_psi(5.0 * LAMBDA_D, psi_D)
    assert abs(psi_far) < 0.01 * PSI_SAT, (
        f"At y=5*lam_D, |psi|={abs(psi_far):.6f} must be < 1% of "
        f"psi_sat={PSI_SAT:.4f}; got ratio={abs(psi_far)/PSI_SAT:.4f}"
    )


def test_psi_falls_through_to_pure_exponential_below_saturation():
    """If |psi_D| <= psi_sat * (1 - 1e-3), composite returns the linear-Debye
    exponential (Gouy-Chapman in the unsaturated regime).
    """
    psi_D = 0.99 * PSI_SAT  # below saturation threshold
    y = 0.5 * LAMBDA_D
    expected = psi_D * math.exp(-y / LAMBDA_D)
    got = composite_psi(y, psi_D)
    assert abs(got - expected) < 1e-12


def test_psi_falls_through_to_pure_exponential_for_tiny_psi_D():
    """|psi_D| < 1e-6 returns linear-Debye exponential."""
    psi_D = 5e-7
    y = 0.3 * LAMBDA_D
    expected = psi_D * math.exp(-y / LAMBDA_D)
    got = composite_psi(y, psi_D)
    assert abs(got - expected) < 1e-18


def test_psi_negative_branch_is_signed_mirror_of_positive():
    """Composite is sign-aware; psi(y; -psi_D) = -psi(y; +psi_D)."""
    psi_D = 19.4
    y_grid = np.linspace(0.0, 0.5, 11)
    for y in y_grid:
        pos = composite_psi(float(y), psi_D)
        neg = composite_psi(float(y), -psi_D)
        assert abs(pos + neg) < 1e-12


@pytest.mark.slow
@pytest.mark.parametrize("psi_D", [11.7, 19.4, 25.7, 38.9])
def test_psi_qualitatively_tracks_first_integral_ode(psi_D):
    """P_2b_3: composite has the qualitative shape of the Bikerman ODE.

    The first integral

        (dpsi/dy)^2 = (2/(nu*lam_D^2)) * ln[1 + nu*(cosh psi - 1)]

    integrated from psi(0) = psi_D with dpsi/dy = -sqrt(...) (decay).

    The composite uses a constant slope alpha(psi_D) in zone 1, which is
    the BKSA matched-asymptotic outer slope.  The true ODE has slope
    alpha(psi) that decreases as psi decays toward psi_sat, so the
    composite reaches saturation faster than the true profile.  This
    pointwise mismatch is well documented (Bazant-Kilic-Storey-Ajdari
    2009 fig 4); the composite is still a useful Newton-friendly seed
    because it captures all the gross features:

      * monotonic decay,
      * crossover near a y_match O(lam_D),
      * outer exponential approach to zero,
      * boundary at psi_D, large-y limit at 0.

    This test asserts those gross properties on the scipy solution and
    bounds the area-under-the-curve mismatch at 60% of the integrated
    composite (matched-asymptotic deficit is concentrated in zone 1, so
    a tight pointwise bound is not informative).
    """
    pytest.importorskip("scipy")
    from scipy.integrate import solve_ivp

    def rhs(_y, psi):
        psi_val = float(psi[0])
        if psi_val <= 0.0:
            return [0.0]
        log_arg = max(NU * (math.cosh(psi_val) - 1.0), 0.0)
        slope_sq = (2.0 / (NU * LAMBDA_D ** 2)) * math.log1p(log_arg)
        return [-math.sqrt(max(slope_sq, 0.0))]

    y_max = 5.0 * LAMBDA_D
    sol = solve_ivp(
        rhs,
        t_span=(0.0, y_max),
        y0=[psi_D],
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
        dense_output=True,
    )
    assert sol.success, f"scipy solve_ivp failed: {sol.message}"

    y_grid = np.linspace(0.0, y_max, 201)
    psi_ode = sol.sol(y_grid).reshape(-1)
    psi_comp = np.array([composite_psi(float(y), psi_D) for y in y_grid])

    # Both monotone non-increasing in y (within numerical noise).
    assert np.all(np.diff(psi_ode) <= 1e-9), "ODE solution must be monotone"
    assert np.all(np.diff(psi_comp) <= 1e-9), "Composite must be monotone"

    # Boundary value matches at y=0.
    assert abs(psi_comp[0] - psi_D) < 1e-12
    assert abs(psi_ode[0] - psi_D) < 1e-6

    # Far-field decay: both reach < psi_sat at y >> y_match.
    assert psi_ode[-1] < PSI_SAT, f"ODE psi at y_max={y_max} = {psi_ode[-1]}"
    assert psi_comp[-1] < PSI_SAT

    # Area under the curve agreement (better metric than pointwise for
    # matched-asymptotic profiles).  Both should be O(psi_D * lam_D).
    area_comp = float(np.trapezoid(psi_comp, y_grid))
    area_ode = float(np.trapezoid(psi_ode, y_grid))
    rel_area_diff = abs(area_comp - area_ode) / max(abs(area_ode), 1e-30)
    assert rel_area_diff < 0.60, (
        f"composite area={area_comp:.4f} vs ODE area={area_ode:.4f}; "
        f"rel diff={rel_area_diff:.3f} at psi_D={psi_D}"
    )
