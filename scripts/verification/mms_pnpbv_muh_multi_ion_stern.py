"""Production-faithful MMS for `logc_muh` + Cs+/SO4(2-) multi-ion + Stern stack.

Backs the derivation in
``docs/solver/mms_pnpbv_muh_multi_ion_stern_derivation.md``
and the implementation plan in ``.plans/mms-muh-multi-ion-stern/PLAN.md``.

Verifies the *exact* production stack used by
``scripts/studies/solver_demo_slide15_no_speculative_cs.py``:

  - 3 dynamic species (O2, H2O2, H+) with PHYSICAL hard-sphere ``a_nondim``
    (Marcus / Stokes radii: 1.70 / 2.00 / 2.80 Å)
  - log-concentration primary variables for O2/H2O2; ``mu_H`` for H+
    (electrochemical-potential primary in the ``logc_muh`` formulation)
  - log-rate Butler-Volmer (Ruggiero parallel-2e/4e topology)
  - Cs+ and SO4(2-) Bikerman counterions under the multi-ion shared-theta
    closure
  - Stern Robin BC at the electrode with C_S = 0.20 F/m^2 (literature)
  - ``exponent_clip = 100``, ``u_clamp = 100``

Manufactured solution (in nondim units)::

    c_i^ex(x, y)  = c0_HAT[i] * (1 + delta_i * cos(pi x) * (1 - y)^2)
    phi^ex(x, y)  = (1 - y) * (alpha0 + alpha1 * cos(pi x))
                    + gamma * y * (1 - y) * cos(pi x)
    mu_H^ex(x, y) = ln(c_H^ex) + em * z_H * phi^ex

Boundary conditions satisfied by the manufactured shape::

    bulk    (y=1): u_i(x,1) = ln(c0_HAT[i]),  mu_H(x,1) = ln(c0_H),  phi(x,1) = 0
    electrode (y=0): Stern Robin (natural, with g_S source); BV flux (natural,
                     with g_i^elec source)
    side walls (x=0,1): natural zero-flux (auto-satisfied by sin(pi x) factors)

Source-term construction follows §4 of the derivation:

  S_c_i  = -div(J_i^ex)
  g_i^elec = J_i^ex . n - sum_j s_{ij} * R_j^ex
  S_phi  = -eps_coeff * div(grad(phi^ex))
           - charge_rhs * (z_H * c_H^ex + z_scale * sum_k z_k * c_k^ster,ex)
  g_S    = eps_coeff * (grad(phi^ex) . n) - C_S^model * (phi_app^model - phi^ex)

These are subtracted from F_res so the manufactured solution becomes a
continuum residual zero; the discrete CG1 solution u_h then approximates
u_exact at L2 ~ O(h^2), H1 ~ O(h^1) per CG1 theory.

Mesh: UnitSquareMesh(N, N) for the asymptotic convergence study; the
production graded rectangle (Nx=8, Ny=80, beta=3) is exercised by
``verify_on_graded_production_mesh`` for single-mesh recovery.

Expected: L2 rate >= 1.8, H1 rate >= 0.8 per primary unknown, R^2 > 0.99.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from math import log, pi
from typing import Any, Callable, List, Optional, Sequence

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("FIREDRAKE_TSFC_KERNEL_CACHE_DIR", "/tmp/firedrake-tsfc")
os.environ.setdefault("PYOP2_CACHE_DIR", "/tmp/pyop2")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import firedrake as fd  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from Forward.bv_solver import (  # noqa: E402
    build_context,
    build_forms,
    make_graded_rectangle_mesh,
)
from Forward.bv_solver.cation_hydrolysis import is_cation_hydrolysis_enabled  # noqa: E402
from Forward.bv_solver.config import (  # noqa: E402
    _get_bv_boltzmann_counterions_cfg,
    _get_bv_convergence_cfg,
)
from Forward.bv_solver.water_ionization import is_water_ionization_enabled  # noqa: E402
from Nondim.transform import _get_nondim_cfg  # noqa: E402

from scripts._bv_common import (  # noqa: E402
    A_CSPLUS_HAT,
    A_SO4_HAT,
    ALPHA_R1,
    ALPHA_R2E,
    ALPHA_R4E,
    C_CSPLUS_HAT,
    C_HP_HAT,
    C_O2_HAT,
    C_SO4_HAT,
    D_H2O2_HAT,
    D_HP_HAT,
    D_O2_HAT,
    DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,
    DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
    E_EQ_R2E_V,
    E_EQ_R4E_V,
    H2O2_SEED_NONDIM,
    K0_HAT_R1,
    K0_HAT_R2E,
    K0_HAT_R4E,
    SNES_OPTS_CHARGED,
    SpeciesConfig,
    V_T,
    make_bv_solver_params,
    setup_firedrake_env,
)


# ---------------------------------------------------------------------------
# Top-level constants (plan §2)
# ---------------------------------------------------------------------------

V_RHE_TEST = 0.55                       # V vs RHE — demo anchor
DT_LARGE = 1.0e15
T_END_LARGE = 1.0e15

K0_R4E_FACTOR_MMS = 1.0e-18             # derivation §5.5; R4e/R2e bounded
STERN_C_S_F_M2 = 0.20                   # production target (Hard Rule #6)

DELTA_PERTURB = (0.30, 0.30, 0.30)      # (O2, H2O2, H+) perturbation amplitudes
ALPHA0, ALPHA1, GAMMA = 0.5, 0.5, 0.5   # phi^ex envelope params

SRC_QUAD_DEGREE_INITIAL = 8             # candidate; pinned by pilot 10.1
MESH_SIZES = (8, 16, 32, 64)

L_EFF_M_MMS = 1.0e-4                    # 100 um (matches L_REF) — unit square
EXPONENT_CLIP = 100.0
U_CLAMP = 100.0

# H2O2 bulk c0 for the MMS test problem.
#
# Production uses ``H2O2_SEED_NONDIM = 1e-4`` (a "seed" to avoid log(0) at
# start-up; the actual c_H2O2 grows away from this seed during continuation).
# The MMS test starts Newton directly at U=U_manuf with no continuation, so
# c_H2O2(manuf) ≈ 1e-4 lands H2O2 far from Newton's basin: the parallel-2e/4e
# topology has weak coupling on H2O2 (only +1·R_R2e ≈ 2.5 at electrode; R_R4e
# stoichiometry on H2O2 is 0) and the H2O2 sub-equation becomes
# Newton-ill-conditioned (Jacobian diagonal scales like c_H2O2 ≈ 1e-4).  At
# c0_H2O2 = 1.0 the H2O2 column of the Jacobian carries similar weight as
# the other species, and Newton finds the discrete solution cleanly.  The
# operator under test (Bikerman closure, BV log-rate, Stern Robin, muh
# transform) is unchanged — only the bulk Dirichlet BC value differs from
# production.
H2O2_C0_MMS = 1.0

# Per-species physical hard-sphere a_nondim (matches solver_demo_slide15).
_N_A = 6.02214076e23
_C_SCALE = 1.2


def _a_nondim_from_radius_m(r_m: float) -> float:
    a_phys = (4.0 / 3.0) * math.pi * r_m ** 3 * _N_A
    return a_phys * _C_SCALE


A_O2_PHYSICAL = _a_nondim_from_radius_m(1.70e-10)
A_H2O2_PHYSICAL = _a_nondim_from_radius_m(2.00e-10)
A_HP_PHYSICAL = _a_nondim_from_radius_m(2.80e-10)


SPECIES_LABELS = ["O2 (u)", "H2O2 (u)", "H+ (mu)", "phi"]
FIELD_NAMES = ["u_O2", "u_H2O2", "mu_H", "phi"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_close(name: str, got: float, expected: float,
                  *, rel: float = 1e-9, abs_tol: float = 0.0) -> None:
    """Equality assert via ``math.isclose``.  Used inside scripts/verification
    where ``pytest.approx`` is unavailable (script must also run standalone).
    """
    if not math.isclose(got, expected, rel_tol=rel, abs_tol=abs_tol):
        raise AssertionError(
            f"_assert_close({name}): got {got!r}, expected {expected!r} "
            f"(rel_tol={rel}, abs_tol={abs_tol})"
        )


@dataclass
class OwnedCoeffTracker:
    """Per-mesh ledger of ``fd.Constant`` objects created by MMS builders.

    Threaded through every expression-construction helper so the source-
    independence check (phase 3 invariant) has an identity-based whitelist.
    """

    _owned: set = field(default_factory=set)

    def constant(self, value: float, *, label: str = "") -> Any:
        c = fd.Constant(float(value))
        self._owned.add(c)
        return c

    @property
    def coeffs(self) -> frozenset:
        return frozenset(self._owned)


def _expr_min(expr, mesh, *, degree: int = 4) -> float:
    """Pointwise min via DG-k interpolation.  Uses ``fd.interpolate`` to
    avoid the L2-projection smoothing of extrema you'd get with
    ``fd.project``.
    """
    P = fd.Function(fd.FunctionSpace(mesh, "DG", degree)).interpolate(expr)
    return float(P.dat.data_ro.min())


def _expr_max(expr, mesh, *, degree: int = 4) -> float:
    P = fd.Function(fd.FunctionSpace(mesh, "DG", degree)).interpolate(expr)
    return float(P.dat.data_ro.max())


def _expr_abs_max(expr, mesh, *, degree: int = 4) -> float:
    P = fd.Function(fd.FunctionSpace(mesh, "DG", degree)).interpolate(expr)
    return float(np.abs(P.dat.data_ro).max())


def _expr_indicator_measure(expr, threshold: float, *, mesh, degree: int,
                            comparison: str = "lt") -> float:
    """Measure of ``{x : expr {<,>} threshold}`` via quadrature.  Uses an
    explicit ``fd.Constant`` threshold to avoid UFL coercion of a Python
    float into a ``ScalarValue`` whose mesh association would be missing.
    """
    thr = fd.Constant(float(threshold))
    cond = fd.lt(expr, thr) if comparison == "lt" else fd.gt(expr, thr)
    dx_q = fd.dx(domain=mesh, degree=degree)
    return float(fd.assemble(
        fd.conditional(cond, fd.Constant(1.0), fd.Constant(0.0)) * dx_q
    ))


def _domain_volume(mesh, *, degree: int) -> float:
    return float(fd.assemble(fd.Constant(1.0) * fd.dx(domain=mesh, degree=degree)))


def _phi_sub_index(ctx) -> int:
    """Resolve the (possibly negative) ``phi_index`` to an absolute index
    for ``Function.sub(i)`` calls.
    """
    indices = ctx["mixed_space_indices"]
    raw = indices.phi_index
    n_subs = ctx["W"].num_sub_spaces()
    return raw if raw >= 0 else n_subs + raw


def _snapshot_live_coeffs(ctx) -> dict:
    """Capture floating-point values of live continuation coefficients so we
    can verify they didn't change between source build and solve.
    """
    snap = {
        "phi_applied": float(ctx["phi_applied_func"].dat.data_ro[0]),
        "stern_coeff": None,
        "k0_funcs": [float(f.dat.data_ro[0]) for f in ctx.get("bv_k0_funcs", [])],
        "alpha_funcs": [float(f.dat.data_ro[0]) for f in ctx.get("bv_alpha_funcs", [])],
        "z_scale": None,
    }
    sc = ctx.get("stern_coeff_const")
    if sc is not None:
        snap["stern_coeff"] = float(sc)
    zs = ctx.get("boltzmann_z_scale")
    if zs is not None:
        snap["z_scale"] = float(zs.dat.data_ro[0])
    return snap


def _assert_live_coeffs_unchanged(ctx, snapshots: dict) -> None:
    assert float(ctx["phi_applied_func"].dat.data_ro[0]) == snapshots["phi_applied"], (
        "phi_applied_func changed between source build and solve"
    )
    if snapshots["stern_coeff"] is not None:
        assert float(ctx["stern_coeff_const"]) == snapshots["stern_coeff"], (
            "stern_coeff_const changed between source build and solve"
        )
    for k, (f, s) in enumerate(zip(ctx.get("bv_k0_funcs", []), snapshots["k0_funcs"])):
        assert float(f.dat.data_ro[0]) == s, f"bv_k0_funcs[{k}] mutated"
    for k, (f, s) in enumerate(zip(ctx.get("bv_alpha_funcs", []), snapshots["alpha_funcs"])):
        assert float(f.dat.data_ro[0]) == s, f"bv_alpha_funcs[{k}] mutated"
    if snapshots["z_scale"] is not None:
        assert float(ctx["boltzmann_z_scale"].dat.data_ro[0]) == snapshots["z_scale"], (
            "boltzmann_z_scale mutated"
        )


# ---------------------------------------------------------------------------
# Phase-0 prebuild invariants (operate on ``sp`` BEFORE build_forms)
# ---------------------------------------------------------------------------

def _assert_prebuild_config_invariants(sp) -> None:
    """Phase 0 — operates on ``sp`` only, before ``build_forms``.

    Catches feature-flag drift that would otherwise crash ``build_forms``
    (e.g. ``cation_hydrol_on`` would raise inside ``resolve_counterion_index``
    before our invariants fire).
    """
    params = sp[10]
    conv_cfg = params.get("bv_convergence", {})
    bv_bc = params.get("bv_bc", {})

    # Feature flags.
    assert not is_water_ionization_enabled(conv_cfg), (
        "MMS requires enable_water_ionization=False"
    )
    assert not is_cation_hydrolysis_enabled(conv_cfg), (
        "MMS requires cation hydrolysis disabled"
    )

    # Formulation, log_rate.
    assert str(conv_cfg.get("formulation", "")).lower() == "logc_muh", (
        "MMS requires formulation=logc_muh"
    )
    assert bool(conv_cfg.get("bv_log_rate", False)) is True, (
        "MMS requires bv_log_rate=True"
    )

    # Clip.
    assert bool(conv_cfg.get("clip_exponent", True)) is True, (
        "MMS requires clip_exponent=True (production default)"
    )
    _assert_close("exponent_clip", float(conv_cfg.get("exponent_clip", 100.0)),
                  EXPONENT_CLIP, rel=1e-9)

    # Reactions list shape — exactly two parallel reactions.
    rxns = bv_bc.get("reactions", [])
    assert isinstance(rxns, (list, tuple)) and len(rxns) == 2, (
        f"MMS requires len(rxns) == 2 (R2e + R4e); got {len(rxns)}"
    )

    # Counterions: exactly two bikerman with exact (Cs+, SO4(2-)) identities.
    counterions = bv_bc.get("boltzmann_counterions", [])
    bikerman = [e for e in counterions if e.get("steric_mode") == "bikerman"]
    assert len(bikerman) == 2, (
        f"MMS requires exactly 2 bikerman counterions (Cs+, SO4); "
        f"got {len(bikerman)}"
    )
    cs = next((e for e in bikerman if int(e["z"]) == +1), None)
    assert cs is not None, "MMS requires a Cs+ counterion (z=+1)"
    _assert_close("a_Cs+", float(cs["a_nondim"]), A_CSPLUS_HAT, rel=1e-9)
    _assert_close("c_b_Cs+", float(cs["c_bulk_nondim"]), C_CSPLUS_HAT, rel=1e-9)
    so4 = next((e for e in bikerman if int(e["z"]) == -2), None)
    assert so4 is not None, "MMS requires a SO4(2-) counterion (z=-2)"
    _assert_close("a_SO4", float(so4["a_nondim"]), A_SO4_HAT, rel=1e-9)
    _assert_close("c_b_SO4", float(so4["c_bulk_nondim"]), C_SO4_HAT, rel=1e-9)

    # suppress_poisson_source lives on params['nondim'], NOT on conv_cfg.
    nondim_cfg = params.get("nondim", {})
    assert not bool(nondim_cfg.get("suppress_poisson_source", False)), (
        "MMS requires suppress_poisson_source=False"
    )

    # dt, SNES tolerances — keep time term negligible + Newton tight enough
    # to drive residual to discretization-error floor.
    assert float(sp[2]) >= 1e12, f"MMS requires dt >= 1e12; got {float(sp[2])}"
    assert float(params.get("snes_atol", 1e-7)) <= 1e-5, (
        f"MMS requires snes_atol <= 1e-5; got {params.get('snes_atol')}"
    )
    assert float(params.get("snes_rtol", 1e-8)) <= 1e-7, (
        f"MMS requires snes_rtol <= 1e-7; got {params.get('snes_rtol')}"
    )

    # Species identity from sp.
    assert int(sp[0]) == 3, f"MMS requires 3 dynamic species; got {sp[0]}"
    assert list(sp[4]) == [0, 0, 1], f"MMS requires z_vals=[0,0,1]; got {list(sp[4])}"
    roles = list(bv_bc.get("species_roles", []))
    assert roles == ["neutral", "neutral", "proton"], (
        f"MMS requires species_roles=['neutral','neutral','proton']; got {roles!r}"
    )
    a_vals = list(sp[6])
    _assert_close("a[O2]", a_vals[0], A_O2_PHYSICAL, rel=1e-9)
    _assert_close("a[H2O2]", a_vals[1], A_H2O2_PHYSICAL, rel=1e-9)
    _assert_close("a[H]", a_vals[2], A_HP_PHYSICAL, rel=1e-9)


# ---------------------------------------------------------------------------
# Phase-1 postbuild static invariants (after ``build_forms``)
# ---------------------------------------------------------------------------

def _assert_postbuild_static_ctx_invariants(ctx, sp) -> None:
    scaling = ctx["nondim"]
    conv_cfg = ctx["bv_convergence"]

    csm = scaling.get("bv_stern_capacitance_model")
    assert csm is not None and float(csm) > 0, (
        f"MMS requires bv_stern_capacitance_model > 0; got {csm}"
    )

    assert ctx.get("logc_muh_transform") is True, (
        "MMS requires logc_muh_transform=True on ctx"
    )

    indices = ctx.get("mixed_space_indices")
    if indices is not None and getattr(indices, "gamma_index", None) is not None:
        raise AssertionError(
            "MMS does not support a Γ slot in the mixed space "
            f"(gamma_index={indices.gamma_index})"
        )

    rxns = scaling["bv_reactions"]
    assert len(rxns) == 2, f"MMS post-build expects len(bv_reactions)=2; got {len(rxns)}"
    r2e, r4e = rxns[0], rxns[1]

    # R2e identity (reversible, 2-electron, c_ref-anchored anodic species idx=1).
    _assert_close("E_eq_R2e", float(r2e["E_eq_model"]), E_EQ_R2E_V / V_T, rel=1e-9)
    _assert_close("alpha_R2e", float(r2e["alpha"]), ALPHA_R2E, rel=1e-9)
    assert int(r2e["n_electrons"]) == 2
    assert bool(r2e["reversible"]) is True
    assert int(r2e["cathodic_species"]) == 0
    assert int(r2e["anodic_species"]) == 1
    assert tuple(r2e["stoichiometry"]) == (-1, +1, -2)
    cf2e = r2e["cathodic_conc_factors"][0]
    assert int(cf2e["species"]) == 2 and float(cf2e["power"]) == 2.0
    _assert_close("c_ref_R2e_factor", float(cf2e["c_ref_nondim"]), C_HP_HAT, rel=1e-9)
    _assert_close("k0_R2e", float(r2e["k0_model"]), K0_HAT_R2E, rel=1e-9)
    assert float(r2e["k0_model"]) > 0.0

    # R4e identity (irreversible, 4-electron, no anodic branch).
    _assert_close("E_eq_R4e", float(r4e["E_eq_model"]), E_EQ_R4E_V / V_T, rel=1e-9)
    _assert_close("alpha_R4e", float(r4e["alpha"]), ALPHA_R4E, rel=1e-9)
    assert int(r4e["n_electrons"]) == 4
    assert bool(r4e["reversible"]) is False
    assert int(r4e["cathodic_species"]) == 0
    assert r4e.get("anodic_species") is None
    assert tuple(r4e["stoichiometry"]) == (-1, 0, -4)
    assert float(r4e["c_ref_model"]) == 0.0
    cf4e = r4e["cathodic_conc_factors"][0]
    assert int(cf4e["species"]) == 2 and float(cf4e["power"]) == 4.0
    _assert_close(
        "k0_R4e", float(r4e["k0_model"]),
        K0_HAT_R4E * K0_R4E_FACTOR_MMS, rel=1e-9,
    )
    assert float(r4e["k0_model"]) > 0.0

    # c0_model_vals (nondim) lives on ctx['nondim'] after build_forms.
    c0_vals = list(scaling["c0_model_vals"])
    _assert_close("c0[O2]", c0_vals[0], C_O2_HAT, rel=1e-9)
    _assert_close("c0[H2O2]", c0_vals[1], H2O2_C0_MMS, rel=1e-9)
    _assert_close("c0[H]", c0_vals[2], C_HP_HAT, rel=1e-9)


# ---------------------------------------------------------------------------
# Manufactured fields and the shared-theta closure (plan §6)
# ---------------------------------------------------------------------------

@dataclass
class ManufacturedFields:
    c_ex: list                # per dynamic species — UFL c_i^ex (concentration)
    u_ex: list                # per dynamic species — UFL log(c_i^ex) (proton entry = reconstruction)
    mu_H_ex: Any              # UFL mu_H^ex = ln(c_H^ex) + em*z_H*phi^ex
    phi_ex: Any               # UFL phi^ex
    phi_app_model: float
    em_z_H: float
    h_idx: int


@dataclass
class ClosureFields:
    A_dyn_ex: Any
    theta_b: float
    D_ex: Any
    c_steric_ex: list         # per counterion
    P_k_ex: list              # per counterion (a_k * c_k^ster)
    rho_k_ex: list            # per counterion (z_k * c_k^ster)
    sum_P_k_ex: Any
    theta_inner_ex: Any
    mu_steric_ex: Any
    counterion_metadata: list


def _make_manufactured_fields(mesh, ctx, sp, *,
                              owned: OwnedCoeffTracker) -> ManufacturedFields:
    """Build the manufactured fields as UFL expressions in (x, y)."""
    scaling = ctx["nondim"]
    em = float(scaling["electromigration_prefactor"])
    z_vals = list(sp[4])
    n = int(sp[0])

    # Resolve which species index carries the proton (mu_H) — production
    # uses the role-aware index if roles are provided, else falls back to
    # z=+1 search.  Both paths converge on idx=2 for our species layout.
    h_idx = int(ctx["mu_species"][0]) if ctx.get("mu_species") else 2
    em_z_H = em * float(z_vals[h_idx])

    c0 = list(scaling["c0_model_vals"])
    phi_app_model = float(scaling["phi_applied_model"])

    x, y = fd.SpatialCoordinate(mesh)

    delta = [owned.constant(float(d), label=f"delta_{i}")
             for i, d in enumerate(DELTA_PERTURB)]
    a0 = owned.constant(float(ALPHA0), label="alpha0")
    a1 = owned.constant(float(ALPHA1), label="alpha1")
    g = owned.constant(float(GAMMA), label="gamma")

    one = owned.constant(1.0, label="one")
    pi_c = owned.constant(math.pi, label="pi")

    # Manufactured concentrations c_i^ex = c0_i * (1 + delta_i * cos(pi x) * (1-y)^2)
    c0_consts = [owned.constant(float(c0[i]), label=f"c0_{i}") for i in range(n)]
    cos_pix = fd.cos(pi_c * x)
    one_minus_y = one - y
    one_minus_y_sq = one_minus_y * one_minus_y
    c_ex = [
        c0_consts[i] * (one + delta[i] * cos_pix * one_minus_y_sq)
        for i in range(n)
    ]

    # phi^ex = (1-y) * (a0 + a1 cos(pi x)) + gamma * y(1-y) * cos(pi x)
    phi_ex = one_minus_y * (a0 + a1 * cos_pix) + g * y * one_minus_y * cos_pix

    # Reconstructed log-concentrations and mu_H.
    u_ex = [None] * n
    for i in range(n):
        u_ex[i] = fd.ln(c_ex[i])
    em_z_H_const = owned.constant(em_z_H, label="em_z_H")
    mu_H_ex = u_ex[h_idx] + em_z_H_const * phi_ex

    return ManufacturedFields(
        c_ex=c_ex, u_ex=u_ex, mu_H_ex=mu_H_ex, phi_ex=phi_ex,
        phi_app_model=phi_app_model, em_z_H=em_z_H, h_idx=h_idx,
    )


def _build_shared_theta_closure_ex(*,
                                   counterions_cfg: list,
                                   a_dyn: list,
                                   c0_dyn: list,
                                   z_dyn: list,
                                   manuf: ManufacturedFields,
                                   owned: OwnedCoeffTracker) -> ClosureFields:
    """Independent UFL composition for the multi-ion shared-theta Bikerman
    closure.  Mirrors ``boltzmann.py:91-268`` algebraically but does NOT
    consume the production-side ``ctx['steric_boltzmann']`` bundle so a
    wiring bug symmetric across both sides cannot cancel and pass the test.

    The source-builder closure uses UNCLIPPED, UNFLOORED expressions.
    Validity relies on the pre-rates margin invariants (phase 2a) having
    passed.  The closure-algebra smoke test (pilot 10.9) uses a separate
    helper that mirrors production clamps/floors exactly, for the 1e-9
    comparison.
    """
    bikerman = [e for e in counterions_cfg if e.get("steric_mode") == "bikerman"]
    if len(bikerman) != 2:
        raise ValueError(
            f"_build_shared_theta_closure_ex: expected 2 bikerman entries; "
            f"got {len(bikerman)}"
        )

    a_dyn_const = [owned.constant(float(a), label=f"a_dyn_{i}")
                   for i, a in enumerate(a_dyn)]

    # A_dyn(c_dyn) = sum_i a_i * c_i  (no clamp, no floor)
    A_dyn_ex = sum(a_dyn_const[i] * manuf.c_ex[i] for i in range(len(manuf.c_ex)))

    # Bulk packing constant theta_b = 1 - A_dyn_bulk - sum_k a_k * c_b_k.
    A_dyn_bulk = sum(float(a_dyn[i]) * float(c0_dyn[i]) for i in range(len(c0_dyn)))
    A_an_bulk = sum(
        float(e["a_nondim"]) * float(e["c_bulk_nondim"]) for e in bikerman
    )
    theta_b = 1.0 - A_dyn_bulk - A_an_bulk
    if theta_b <= 0.0:
        raise ValueError(
            f"_build_shared_theta_closure_ex: theta_b = {theta_b:.6g} <= 0"
        )
    theta_b_const = owned.constant(theta_b, label="theta_b")

    # Per-ion exponentials (unclamped — clamp identity at u_exact, see §5).
    per_ion = []
    for entry in bikerman:
        z_k = int(entry["z"])
        c_k = float(entry["c_bulk_nondim"])
        a_k = float(entry["a_nondim"])
        z_const = owned.constant(float(z_k), label=f"z_{entry.get('label', z_k)}")
        c_const = owned.constant(c_k, label=f"c_b_{entry.get('label', z_k)}")
        a_const = owned.constant(a_k, label=f"a_{entry.get('label', z_k)}")
        q_k = fd.exp(-z_const * manuf.phi_ex)
        per_ion.append({
            "z": z_k, "c_bulk": c_k, "a_nondim": a_k,
            "z_const": z_const, "c_const": c_const, "a_const": a_const,
            "q": q_k,
        })

    # Shared denominator D(phi) = theta_b + sum_k a_k * c_b_k * q_k.
    D_ex = theta_b_const + sum(p["a_const"] * p["c_const"] * p["q"] for p in per_ion)

    one_const = owned.constant(1.0, label="closure_one")
    free_dyn = one_const - A_dyn_ex   # unfloored (floor identity at u_exact)

    c_steric_ex = []
    P_k_ex = []
    rho_k_ex = []
    for p in per_ion:
        c_k_ster = p["c_const"] * p["q"] * free_dyn / D_ex
        c_steric_ex.append(c_k_ster)
        P_k_ex.append(p["a_const"] * c_k_ster)
        rho_k_ex.append(p["z_const"] * c_k_ster)

    sum_P_k_ex = sum(P_k_ex)
    # z_scale = 1.0 at MMS runtime; multiply explicitly so the algebra
    # mirrors production's theta_inner shape.
    z_scale_one = owned.constant(1.0, label="z_scale_runtime")
    theta_inner_ex = one_const - A_dyn_ex - z_scale_one * sum_P_k_ex
    mu_steric_ex = -fd.ln(theta_inner_ex)   # unfloored (packing_floor identity at u_exact)

    return ClosureFields(
        A_dyn_ex=A_dyn_ex,
        theta_b=theta_b,
        D_ex=D_ex,
        c_steric_ex=c_steric_ex,
        P_k_ex=P_k_ex,
        rho_k_ex=rho_k_ex,
        sum_P_k_ex=sum_P_k_ex,
        theta_inner_ex=theta_inner_ex,
        mu_steric_ex=mu_steric_ex,
        counterion_metadata=[
            {"z": p["z"], "c_bulk": p["c_bulk"], "a_nondim": p["a_nondim"]}
            for p in per_ion
        ],
    )


# ---------------------------------------------------------------------------
# Phase-2a pre-rates margin invariants
# ---------------------------------------------------------------------------

def _assert_pre_rates_margin_invariants(ctx, sp, *, manuf: ManufacturedFields,
                                        closure: ClosureFields,
                                        quad_degree: int) -> None:
    mesh = ctx["mesh"]
    scaling = ctx["nondim"]
    conv_cfg = ctx["bv_convergence"]
    phi_ex = manuf.phi_ex

    bv_exp_scale = float(scaling["bv_exponent_scale"])
    phi_app_model = float(scaling["phi_applied_model"])
    exp_clip = float(conv_cfg["exponent_clip"])
    u_clamp = float(conv_cfg.get("u_clamp", 30.0))

    rxns = scaling["bv_reactions"]
    for j, lbl in enumerate(("R2e", "R4e")):
        eta_expr = fd.Constant(bv_exp_scale) * (
            fd.Constant(phi_app_model) - phi_ex - fd.Constant(float(rxns[j]["E_eq_model"]))
        )
        eta_amax = _expr_abs_max(eta_expr, mesh, degree=4)
        assert eta_amax < 0.9 * exp_clip, (
            f"eta_{lbl} abs max {eta_amax:.3g} too close to clip {exp_clip}"
        )

    # u_clamp margin (per primary unknown).  For the proton, the production
    # clamp is on the reconstruction u_H = mu_H - em*z_H*phi.
    for label, field_expr in (("O2", manuf.u_ex[0]),
                              ("H2O2", manuf.u_ex[1]),
                              ("H_recon", manuf.u_ex[2])):
        u_amax = _expr_abs_max(field_expr, mesh, degree=4)
        assert u_amax < 0.9 * u_clamp, (
            f"|u_{label}| max {u_amax:.3g} too close to clamp {u_clamp}"
        )

    # Ion phi_clamp margin (per counterion).
    phi_amax = _expr_abs_max(phi_ex, mesh, degree=4)
    counterions = _get_bv_boltzmann_counterions_cfg(sp[10])
    for entry in counterions:
        if entry.get("steric_mode") != "bikerman":
            continue
        phi_clamp_k = float(entry["phi_clamp"])
        label = entry.get("label", entry.get("z"))
        assert phi_amax < 0.9 * phi_clamp_k, (
            f"|phi| max {phi_amax:.3g} too close to clamp_{label} {phi_clamp_k}"
        )

    # free_dyn floor: (1 - A_dyn) > floor.
    A_dyn_max = _expr_abs_max(closure.A_dyn_ex, mesh, degree=4)
    assert A_dyn_max < 0.99, f"A_dyn max {A_dyn_max:.3g} too close to 1"

    # packing_floor margin via DG-interp min AND quadrature indicator.
    packing_floor = float(conv_cfg.get("packing_floor", 1e-8))
    theta_min = _expr_min(closure.theta_inner_ex, mesh, degree=4)
    assert theta_min > 10 * packing_floor, (
        f"min(theta_inner) {theta_min:.3g} too close to floor {packing_floor}"
    )
    bad_measure = _expr_indicator_measure(
        closure.theta_inner_ex, 10 * packing_floor,
        mesh=mesh, degree=quad_degree, comparison="lt",
    )
    vol = _domain_volume(mesh, degree=quad_degree)
    assert bad_measure < 1e-12 * vol, (
        f"theta_inner measure-below-10*floor = {bad_measure:.3e} > 1e-12*vol"
    )


# ---------------------------------------------------------------------------
# BV rate builder (post-shared-theta; consumes manuf + scaling)
# ---------------------------------------------------------------------------

def _build_bv_rates_ex(*, ctx, manuf: ManufacturedFields,
                      owned: OwnedCoeffTracker) -> list:
    """Per-reaction R_j UFL with the muh substitution
    ``u_H = mu_H - em*z_H*phi`` baked in.

    Mirrors ``forms_logc_muh.py`` log-rate path verbatim.  At u_exact the
    `eta_clipped` identity holds (verified by the margin asserts), so this
    builder uses the unclipped expression.
    """
    scaling = ctx["nondim"]
    rxns = scaling["bv_reactions"]
    bv_exp_scale_c = owned.constant(float(scaling["bv_exponent_scale"]),
                                    label="bv_exp_scale")
    phi_app_c = owned.constant(float(scaling["phi_applied_model"]),
                               label="phi_applied_model")

    # u_exprs[i] at u_exact — the muh entry uses the reconstruction
    # ln(c_H^ex), which mathematically equals (mu_H^ex - em*z_H*phi^ex).
    # Using the ln(c_H^ex) form is the simpler and identical expression.
    u_exprs_ex = list(manuf.u_ex)

    rates: list = []
    for j, rxn in enumerate(rxns):
        k0 = float(rxn["k0_model"])
        alpha = float(rxn["alpha"])
        n_e = float(rxn["n_electrons"])
        cat_idx = int(rxn["cathodic_species"])
        E_eq_j = float(rxn.get("E_eq_model", 0.0))

        k0_c = owned.constant(k0, label=f"k0_{j}")
        alpha_c = owned.constant(alpha, label=f"alpha_{j}")
        n_e_c = owned.constant(n_e, label=f"n_e_{j}")
        E_eq_c = owned.constant(E_eq_j, label=f"E_eq_{j}")

        eta_j = bv_exp_scale_c * (phi_app_c - manuf.phi_ex - E_eq_c)

        log_cath = fd.ln(k0_c) + u_exprs_ex[cat_idx] - alpha_c * n_e_c * eta_j
        for factor in rxn.get("cathodic_conc_factors", []):
            sp_idx = int(factor["species"])
            power_c = owned.constant(float(factor["power"]),
                                     label=f"power_{j}_{sp_idx}")
            c_ref_log = fd.ln(owned.constant(
                max(float(factor["c_ref_nondim"]), 1e-12),
                label=f"c_ref_{j}_{sp_idx}",
            ))
            log_cath = log_cath + power_c * (u_exprs_ex[sp_idx] - c_ref_log)
        cathodic = fd.exp(log_cath)

        if bool(rxn["reversible"]) and rxn.get("anodic_species") is not None:
            anod_idx = int(rxn["anodic_species"])
            log_anod = (
                fd.ln(k0_c) + u_exprs_ex[anod_idx]
                + (owned.constant(1.0, label=f"one_anod_{j}") - alpha_c)
                * n_e_c * eta_j
            )
            anodic = fd.exp(log_anod)
        elif bool(rxn["reversible"]) and float(rxn.get("c_ref_model", 0.0)) > 1e-30:
            c_ref_j = owned.constant(float(rxn["c_ref_model"]),
                                     label=f"c_ref_model_{j}")
            log_anod = (
                fd.ln(k0_c) + fd.ln(c_ref_j)
                + (owned.constant(1.0, label=f"one_anod_{j}") - alpha_c)
                * n_e_c * eta_j
            )
            anodic = fd.exp(log_anod)
        else:
            anodic = owned.constant(0.0, label=f"zero_anod_{j}")

        rates.append(cathodic - anodic)

    return rates


# ---------------------------------------------------------------------------
# Phase-2b post-rates invariants
# ---------------------------------------------------------------------------

def _assert_post_rates_invariants(ctx, sp, *, manuf: ManufacturedFields,
                                  rxn_rates: list, quad_degree: int) -> None:
    mesh = ctx["mesh"]
    ds_e = fd.ds(int(ctx["bv_settings"]["electrode_marker"]),
                 domain=mesh, degree=quad_degree)
    R2e_norm = float(fd.assemble(rxn_rates[0] ** 2 * ds_e)) ** 0.5
    R4e_norm = float(fd.assemble(rxn_rates[1] ** 2 * ds_e)) ** 0.5
    R_ratio = R4e_norm / max(R2e_norm, 1e-300)
    assert 10 < R_ratio < 1e5, (
        f"R4e/R2e = {R_ratio:.3e} outside finite window — "
        f"K0_R4e_factor likely mis-set or V_RHE wrong"
    )


# ---------------------------------------------------------------------------
# Source terms (plan §6; derivation §4)
# ---------------------------------------------------------------------------

@dataclass
class MMSSourceTerms:
    S_c: list                 # per dynamic species — UFL on dx
    g_elec: list              # per dynamic species — UFL on ds_elec
    S_phi: Any                # UFL on dx
    g_S: Any                  # UFL on ds_elec
    z_scale_const: Any


def _build_source_terms(ctx, sp, *, manuf: ManufacturedFields,
                        closure: ClosureFields, rxn_rates: list,
                        owned: OwnedCoeffTracker,
                        quad_degree: int) -> MMSSourceTerms:
    """Compose all four MMS sources (interior NP, electrode BV, interior
    Poisson, electrode Stern Robin) at u_exact.  Pure construction — does
    NOT mutate ``ctx['F_res']``.
    """
    scaling = ctx["nondim"]
    n = int(ctx["n_species"])
    mesh = ctx["mesh"]

    em = float(scaling["electromigration_prefactor"])
    eps_coeff = float(scaling["poisson_coefficient"])
    charge_rhs = float(scaling["charge_rhs_prefactor"])
    D_model = [float(v) for v in scaling["D_model_vals"]]
    z_vals = list(sp[4])

    n_vec = fd.FacetNormal(mesh)
    em_z = [owned.constant(em * float(z_vals[i]), label=f"em_z_{i}")
            for i in range(n)]
    eps_c = owned.constant(eps_coeff, label="eps_coeff")
    charge_rhs_c = owned.constant(charge_rhs, label="charge_rhs")

    # NP fluxes at u_exact.
    # For non-mu species (z_i = 0 for O2/H2O2): J_i = D_i * c_i * (grad u_i + grad mu_steric)
    # For mu species (H+): J_H = D_H * c_H * (grad mu_H + grad mu_steric)
    h_idx = manuf.h_idx
    J_ex = []
    grad_mu_steric = fd.grad(closure.mu_steric_ex)
    for i in range(n):
        D_c = owned.constant(D_model[i], label=f"D_{i}")
        if i == h_idx:
            ideal_grad = fd.grad(manuf.mu_H_ex)
        else:
            # Equivalent to grad(u_i) + em*z_i*grad(phi) because z_i = 0 for
            # both O2 and H2O2; we still write it as grad(u_i) so the side-
            # wall identity ∂_x u_i|_{x=0,1} = 0 is transparent.
            ideal_grad = fd.grad(manuf.u_ex[i]) + em_z[i] * fd.grad(manuf.phi_ex)
        J_ex.append(D_c * manuf.c_ex[i] * (ideal_grad + grad_mu_steric))

    # Interior NP sources: S_c_i = -div(J_i^ex)
    S_c = [-fd.div(J_ex[i]) for i in range(n)]

    # Electrode BV electrode boundary sources:
    # g_i^elec = J_i^ex . n - sum_j s_{ij} * R_j^ex
    rxns = scaling["bv_reactions"]
    g_elec = []
    for i in range(n):
        flux_outward = fd.dot(J_ex[i], n_vec)
        bv_sum_terms = []
        for j, rxn in enumerate(rxns):
            stoi_i = int(rxn["stoichiometry"][i])
            if stoi_i != 0:
                s_const = owned.constant(float(stoi_i), label=f"stoi_{j}_{i}")
                bv_sum_terms.append(s_const * rxn_rates[j])
        if bv_sum_terms:
            bv_sum = sum(bv_sum_terms)
        else:
            bv_sum = owned.constant(0.0, label=f"zero_bv_{i}")
        g_elec.append(flux_outward - bv_sum)

    # Poisson interior source:
    # S_phi = -eps_coeff * div(grad(phi^ex))
    #         - charge_rhs * (z_H * c_H^ex + z_scale * sum_k z_k * c_k^ster,ex)
    z_consts = [owned.constant(float(z_vals[i]), label=f"z_dyn_{i}")
                for i in range(n)]
    # z_scale on the Poisson side mirrors `boltzmann_z_scale` in production
    # (default 1.0 at MMS runtime).
    z_scale_c = owned.constant(1.0, label="z_scale_poisson")
    rho_dyn = sum(z_consts[i] * manuf.c_ex[i] for i in range(n))
    rho_steric = sum(closure.rho_k_ex)
    S_phi = (
        -eps_c * fd.div(fd.grad(manuf.phi_ex))
        - charge_rhs_c * rho_dyn
        - z_scale_c * charge_rhs_c * rho_steric
    )

    # Stern Robin electrode boundary source:
    # g_S = eps_coeff * (grad(phi^ex) . n) - C_S^model * (phi_app^model - phi^ex)
    cs_model = float(scaling["bv_stern_capacitance_model"])
    cs_c = owned.constant(cs_model, label="C_S_model")
    phi_app_c = owned.constant(float(scaling["phi_applied_model"]),
                               label="phi_app_for_stern")
    g_S = eps_c * fd.dot(fd.grad(manuf.phi_ex), n_vec) - cs_c * (phi_app_c - manuf.phi_ex)

    return MMSSourceTerms(S_c=S_c, g_elec=g_elec, S_phi=S_phi, g_S=g_S,
                          z_scale_const=z_scale_c)


def _allowed_geometry_for_mesh(mesh, *, quad_degree: int) -> set:
    """Probe ``extract_coefficients`` on a representative geometry expression
    using the same extraction path as ``_assert_source_independence``.
    """
    from ufl.algorithms.analysis import extract_coefficients

    x, y = fd.SpatialCoordinate(mesh)
    n_vec = fd.FacetNormal(mesh)
    test_expr = fd.cos(fd.pi * x) * (1.0 - y) ** 2
    test_form = fd.dot(fd.grad(test_expr), n_vec) * fd.ds(
        domain=mesh, degree=quad_degree,
    )
    return (set(extract_coefficients(test_expr))
            | set(extract_coefficients(test_form)))


def _assert_source_independence(sources: MMSSourceTerms, ctx: dict,
                                owned: OwnedCoeffTracker,
                                *, quad_degree: int) -> None:
    """Phase 3 invariant — every Function appearing in any MMS source is
    either (a) a live ctx coefficient we are *allowed* to reference (so
    runtime ramps stay in lockstep) or (b) an owned ``fd.Constant`` we
    created (tracked via OwnedCoeffTracker), or (c) a geometric terminal.
    """
    from ufl.algorithms.analysis import extract_coefficients

    mesh = ctx["mesh"]
    FORBIDDEN = {ctx["U"], ctx["U_prev"]}
    ALLOWED_LIVE = {
        ctx.get("phi_applied_func"),
        ctx.get("stern_coeff_const"),
        ctx.get("boltzmann_z_scale"),
    }
    for f in ctx.get("bv_k0_funcs", []):
        ALLOWED_LIVE.add(f)
    for f in ctx.get("bv_alpha_funcs", []):
        ALLOWED_LIVE.add(f)
    for f in ctx.get("steric_a_funcs", []):
        ALLOWED_LIVE.add(f)
    ALLOWED_LIVE.discard(None)
    ALLOWED_GEOMETRY = _allowed_geometry_for_mesh(mesh, quad_degree=quad_degree)
    ALLOWED = ALLOWED_LIVE | owned.coeffs | ALLOWED_GEOMETRY

    def _iter(s):
        for i, x in enumerate(s.S_c):
            yield (f"S_c_{i}", x)
        for i, x in enumerate(s.g_elec):
            yield (f"g_elec_{i}", x)
        yield ("S_phi", s.S_phi)
        yield ("g_S", s.g_S)

    for label, S in _iter(sources):
        coeffs = set(extract_coefficients(S))
        bad = coeffs & FORBIDDEN
        assert not bad, (
            f"Source {label} references {bad} — independence violation"
        )
        unknown = coeffs - ALLOWED
        # Filter out NOT-Coefficient leaves (Constants without owned tracking
        # but inside the source builder are conservatively allowed because
        # extract_coefficients returns only ufl.Coefficient instances; any
        # leak through FORBIDDEN already raised above).  Constants don't
        # appear in extract_coefficients output, so `unknown` here is
        # strictly Function or non-Constant Coefficient subclasses.
        if unknown:
            raise AssertionError(
                f"Source {label} references unrecognized coefficients "
                f"{unknown}. Either route through OwnedCoeffTracker, add "
                f"to ALLOWED_LIVE, or expand ALLOWED_GEOMETRY probe."
            )


def _inject_source_terms(ctx, sources: MMSSourceTerms, *,
                         quad_degree: int) -> dict:
    """Subtract source UFL from ``F_res`` and rederive ``J_form``."""
    mesh = ctx["mesh"]
    dx_q = fd.dx(domain=mesh, degree=quad_degree)
    ds_q = fd.ds(int(ctx["bv_settings"]["electrode_marker"]),
                 domain=mesh, degree=quad_degree)

    F = ctx["F_res"]
    v_tests = fd.TestFunctions(ctx["W"])
    indices = ctx["mixed_space_indices"]
    v_list = v_tests[indices.species_slice]
    w_test = v_tests[indices.phi_index]

    for i, S in enumerate(sources.S_c):
        F = F - S * v_list[i] * dx_q
    for i, g in enumerate(sources.g_elec):
        F = F - g * v_list[i] * ds_q
    F = F - sources.S_phi * w_test * dx_q
    F = F - sources.g_S * w_test * ds_q

    ctx["F_res"] = F
    ctx["J_form"] = fd.derivative(F, ctx["U"])
    return ctx


def _build_manufactured_source(ctx, sp, *, manuf: ManufacturedFields,
                               closure: ClosureFields, rxn_rates: list,
                               owned: OwnedCoeffTracker,
                               quad_degree: int) -> dict:
    """Orchestrate build → independence-check → inject."""
    sources = _build_source_terms(
        ctx, sp, manuf=manuf, closure=closure, rxn_rates=rxn_rates,
        owned=owned, quad_degree=quad_degree,
    )
    _assert_source_independence(sources, ctx, owned, quad_degree=quad_degree)
    return _inject_source_terms(ctx, sources, quad_degree=quad_degree)


# ---------------------------------------------------------------------------
# U_manuf interpolation (muh-aware: H+ entry uses mu_H, NOT u_H)
# ---------------------------------------------------------------------------

def _interpolate_U_manuf(ctx, manuf: ManufacturedFields):
    W = ctx["W"]
    indices = ctx["mixed_space_indices"]
    U_manuf = fd.Function(W)

    species_indices = list(range(W.num_sub_spaces()))[indices.species_slice]
    n = len(species_indices)
    h_idx = manuf.h_idx

    for i in range(n):
        sub_idx = species_indices[i]
        sub_fn = U_manuf.sub(sub_idx)
        if i == h_idx:
            sub_fn.interpolate(manuf.mu_H_ex)
        else:
            sub_fn.interpolate(manuf.u_ex[i])

    phi_idx_abs = _phi_sub_index(ctx)
    U_manuf.sub(phi_idx_abs).interpolate(manuf.phi_ex)
    return U_manuf


# ---------------------------------------------------------------------------
# SolverParams factory (plan §7)
# ---------------------------------------------------------------------------

def make_sp_production_muh():
    """Build SolverParams for the production logc_muh + Cs+/SO4(2-) +
    Stern stack used by ``solver_demo_slide15_no_speculative_cs.py``,
    with three adaptations for MMS:

      (1) phi_applied at the test voltage (V_RHE_TEST/V_T).
      (2) dt = T_END_LARGE makes the time term O(h^2/dt) ≈ negligible.
      (3) K0_R4e_factor = 1e-18 keeps R4e/R2e bounded inside (10, 1e5)
          (see derivation §5.5).
    """
    setup_firedrake_env()

    species = SpeciesConfig(
        n_species=3,
        z_vals=[0, 0, 1],
        d_vals_hat=[D_O2_HAT, D_H2O2_HAT, D_HP_HAT],
        a_vals_hat=[A_O2_PHYSICAL, A_H2O2_PHYSICAL, A_HP_PHYSICAL],
        c0_vals_hat=[C_O2_HAT, H2O2_C0_MMS, C_HP_HAT],
        stoichiometry_r1=[-1, +1, -2],
        stoichiometry_r2=[0, -1, -2],
        k0_legacy=[K0_HAT_R1] * 3,
        alpha_legacy=[ALPHA_R1] * 3,
        stoichiometry_legacy=[-1, -1, -1],
        c_ref_legacy=[1.0, 0.0, 1.0],
        roles=["neutral", "neutral", "proton"],
    )

    snes_opts = {
        **SNES_OPTS_CHARGED,
        "snes_max_it":               100,
        "snes_atol":                 1e-8,
        "snes_rtol":                 1e-8,
        "snes_stol":                 1e-12,
        "snes_linesearch_type":      "bt",
        "snes_linesearch_maxlambda": 0.5,
        "snes_divergence_tolerance": 1e20,
    }

    k0_r4e_target = float(K0_HAT_R4E) * float(K0_R4E_FACTOR_MMS)
    rxns = [
        {
            "k0": float(K0_HAT_R2E),
            "alpha": float(ALPHA_R2E),
            "cathodic_species": 0,
            "anodic_species": 1,
            "c_ref": 1.0,
            "stoichiometry": [-1, +1, -2],
            "n_electrons": 2,
            "reversible": True,
            "E_eq_v": float(E_EQ_R2E_V),
            "cathodic_conc_factors": [
                {"species": 2, "power": 2, "c_ref_nondim": float(C_HP_HAT)},
            ],
        },
        {
            "k0": k0_r4e_target,
            "alpha": float(ALPHA_R4E),
            "cathodic_species": 0,
            "anodic_species": None,
            "c_ref": 0.0,
            "stoichiometry": [-1, 0, -4],
            "n_electrons": 4,
            "reversible": False,
            "E_eq_v": float(E_EQ_R4E_V),
            "cathodic_conc_factors": [
                {"species": 2, "power": 4, "c_ref_nondim": float(C_HP_HAT)},
            ],
        },
    ]

    sp = make_bv_solver_params(
        eta_hat=float(V_RHE_TEST) / float(V_T),
        dt=DT_LARGE, t_end=T_END_LARGE,
        species=species,
        snes_opts=snes_opts,
        formulation="logc_muh",
        log_rate=True,
        u_clamp=U_CLAMP,
        bv_reactions=rxns,
        boltzmann_counterions=[
            DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,
            DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
        ],
        multi_ion_enabled=True,
        initializer="debye_boltzmann",
        l_eff_m=L_EFF_M_MMS,
        stern_capacitance_f_m2=STERN_C_S_F_M2,
    )

    new_opts = dict(sp.solver_options)
    new_bv = dict(new_opts["bv_convergence"])
    new_bv["exponent_clip"] = float(EXPONENT_CLIP)
    new_opts["bv_convergence"] = new_bv
    sp = sp.with_solver_options(new_opts)
    return sp


def _extract_solver_parameters(sp) -> dict:
    """Pull SNES/KSP/PC options from the SolverParams params dict."""
    params = sp[10]
    keep_prefixes = ("snes_", "ksp_", "pc_", "mat_")
    return {k: v for k, v in params.items() if k.startswith(keep_prefixes)}


# ---------------------------------------------------------------------------
# Error norms (UFL-based, high-degree quadrature)
# ---------------------------------------------------------------------------

def _ufl_l2_error(u_ufl, u_h, mesh, *, degree: int) -> float:
    dx_q = fd.dx(domain=mesh, degree=degree)
    return float(fd.sqrt(fd.assemble((u_ufl - u_h) ** 2 * dx_q)))


def _ufl_h1_error(u_ufl, u_h, mesh, *, degree: int) -> float:
    dx_q = fd.dx(domain=mesh, degree=degree)
    diff = u_ufl - u_h
    grad_diff = fd.grad(u_ufl) - fd.grad(u_h)
    integrand = diff ** 2 + fd.inner(grad_diff, grad_diff)
    return float(fd.sqrt(fd.assemble(integrand * dx_q)))


# ---------------------------------------------------------------------------
# Solve on a single mesh
# ---------------------------------------------------------------------------

def _solve_mms_on_mesh(mesh, sp, snes_params, *, quad_degree: int) -> dict:
    """End-to-end MMS setup + solve + error norms on a single mesh."""
    _assert_prebuild_config_invariants(sp)
    ctx = build_context(sp, mesh=mesh)
    ctx = build_forms(ctx, sp)
    _assert_postbuild_static_ctx_invariants(ctx, sp)

    owned = OwnedCoeffTracker()
    manuf = _make_manufactured_fields(mesh, ctx, sp, owned=owned)

    counterions_cfg = _get_bv_boltzmann_counterions_cfg(sp[10])
    closure = _build_shared_theta_closure_ex(
        counterions_cfg=counterions_cfg,
        a_dyn=list(sp[6]),
        c0_dyn=list(ctx["nondim"]["c0_model_vals"])[:int(sp[0])],
        z_dyn=list(sp[4]),
        manuf=manuf,
        owned=owned,
    )

    _assert_pre_rates_margin_invariants(
        ctx, sp, manuf=manuf, closure=closure, quad_degree=quad_degree,
    )

    rxn_rates = _build_bv_rates_ex(ctx=ctx, manuf=manuf, owned=owned)
    _assert_post_rates_invariants(
        ctx, sp, manuf=manuf, rxn_rates=rxn_rates, quad_degree=quad_degree,
    )

    _build_manufactured_source(
        ctx, sp, manuf=manuf, closure=closure, rxn_rates=rxn_rates,
        owned=owned, quad_degree=quad_degree,
    )

    U_manuf = _interpolate_U_manuf(ctx, manuf)
    ctx["U"].assign(U_manuf)
    ctx["U_prev"].assign(U_manuf)
    snaps = _snapshot_live_coeffs(ctx)

    F_initial = float(fd.norm(fd.assemble(ctx["F_res"]), norm_type="L2")) if False else None
    # ``fd.norm`` doesn't accept assembled cofunctions directly; instead we
    # report the assembled L2 norm of the residual coefficient vector below.
    res_initial_vec = fd.assemble(ctx["F_res"])
    F_initial = float(np.sqrt(res_initial_vec.dat.norm ** 2)) if hasattr(
        res_initial_vec.dat, "norm"
    ) else None

    problem = fd.NonlinearVariationalProblem(
        ctx["F_res"], ctx["U"], bcs=ctx["bcs"], J=ctx["J_form"],
    )
    solver = fd.NonlinearVariationalSolver(
        problem, solver_parameters=snes_params,
    )

    out: dict = {
        "newton_converged": False,
        "newton_iterations": -1,
        "snes_reason": "",
        "F_res_l2_initial": F_initial,
        "F_res_l2_final": None,
    }
    try:
        solver.solve()
        out["newton_converged"] = True
        out["newton_iterations"] = int(solver.snes.getIterationNumber())
        out["snes_reason"] = str(solver.snes.getConvergedReason())
    except fd.ConvergenceError as exc:
        out["newton_converged"] = False
        out["snes_reason"] = f"ConvergenceError: {exc}"

    _assert_live_coeffs_unchanged(ctx, snaps)

    res_final_vec = fd.assemble(ctx["F_res"])
    if hasattr(res_final_vec.dat, "norm"):
        out["F_res_l2_final"] = float(res_final_vec.dat.norm)

    if not out["newton_converged"]:
        for f in FIELD_NAMES:
            out[f"{f}_L2"] = float("nan")
            out[f"{f}_H1"] = float("nan")
        out["c_H_L2"] = float("nan")
        return out

    phi_idx_abs = _phi_sub_index(ctx)
    h_idx = manuf.h_idx

    species_indices = list(range(ctx["W"].num_sub_spaces()))[
        ctx["mixed_space_indices"].species_slice
    ]
    sub_O2 = ctx["U"].sub(species_indices[0])
    sub_H2O2 = ctx["U"].sub(species_indices[1])
    sub_muH = ctx["U"].sub(species_indices[h_idx])
    sub_phi = ctx["U"].sub(phi_idx_abs)

    out["u_O2_L2"] = _ufl_l2_error(manuf.u_ex[0], sub_O2, mesh, degree=quad_degree)
    out["u_O2_H1"] = _ufl_h1_error(manuf.u_ex[0], sub_O2, mesh, degree=quad_degree)
    out["u_H2O2_L2"] = _ufl_l2_error(manuf.u_ex[1], sub_H2O2, mesh, degree=quad_degree)
    out["u_H2O2_H1"] = _ufl_h1_error(manuf.u_ex[1], sub_H2O2, mesh, degree=quad_degree)
    out["mu_H_L2"] = _ufl_l2_error(manuf.mu_H_ex, sub_muH, mesh, degree=quad_degree)
    out["mu_H_H1"] = _ufl_h1_error(manuf.mu_H_ex, sub_muH, mesh, degree=quad_degree)
    out["phi_L2"] = _ufl_l2_error(manuf.phi_ex, sub_phi, mesh, degree=quad_degree)
    out["phi_H1"] = _ufl_h1_error(manuf.phi_ex, sub_phi, mesh, degree=quad_degree)

    em_z_H = manuf.em_z_H
    c_H_h = fd.exp(sub_muH - fd.Constant(em_z_H) * sub_phi)
    c_H_ex = manuf.c_ex[h_idx]
    out["c_H_L2"] = _ufl_l2_error(c_H_ex, c_H_h, mesh, degree=quad_degree)

    return out


# ---------------------------------------------------------------------------
# Outer ``run_mms`` loop
# ---------------------------------------------------------------------------

def run_mms(N_list: Sequence[int] = MESH_SIZES, *,
            quad_degree: int = SRC_QUAD_DEGREE_INITIAL,
            verbose: bool = True) -> dict:
    """Run the MMS convergence study on ``UnitSquareMesh(N, N)`` for
    ``N in N_list``.  Returns a dict suitable for ``compute_rates`` and
    convergence-rate assertions.
    """
    sp = make_sp_production_muh()
    snes_params = _extract_solver_parameters(sp)

    out: dict = {
        "N": [], "h": [],
        "newton_converged": [], "newton_iterations": [], "snes_reason": [],
        "F_res_l2_initial": [], "F_res_l2_final": [],
        "u_O2_L2": [], "u_O2_H1": [],
        "u_H2O2_L2": [], "u_H2O2_H1": [],
        "mu_H_L2": [], "mu_H_H1": [],
        "phi_L2": [], "phi_H1": [],
        "c_H_L2": [],
    }

    if verbose:
        _print_banner(N_list, quad_degree=quad_degree)

    for N in N_list:
        t0 = time.time()
        mesh = fd.UnitSquareMesh(int(N), int(N))
        errs = _solve_mms_on_mesh(mesh, sp, snes_params, quad_degree=quad_degree)
        elapsed = time.time() - t0

        out["N"].append(int(N))
        out["h"].append(1.0 / float(N))
        out["newton_converged"].append(bool(errs["newton_converged"]))
        out["newton_iterations"].append(int(errs.get("newton_iterations", -1)))
        out["snes_reason"].append(str(errs.get("snes_reason", "")))
        out["F_res_l2_initial"].append(errs.get("F_res_l2_initial"))
        out["F_res_l2_final"].append(errs.get("F_res_l2_final"))
        for f in ("u_O2_L2", "u_O2_H1", "u_H2O2_L2", "u_H2O2_H1",
                  "mu_H_L2", "mu_H_H1", "phi_L2", "phi_H1", "c_H_L2"):
            out[f].append(float(errs.get(f, float("nan"))))

        if verbose:
            status = "ok" if errs["newton_converged"] else "FAIL"
            print(
                f"  N={N:4d}  h={1.0/float(N):.5f}  [{status}, "
                f"iters={errs.get('newton_iterations', -1)}]  "
                f"u_O2_L2={errs.get('u_O2_L2'):.3e}  "
                f"mu_H_L2={errs.get('mu_H_L2'):.3e}  "
                f"phi_L2={errs.get('phi_L2'):.3e}  "
                f"({elapsed:.1f}s)",
                flush=True,
            )

    return out


def verify_on_graded_production_mesh(verbose: bool = True,
                                     *, Nx: int = 8, Ny: int = 80,
                                     beta: float = 3.0,
                                     quad_degree: int = SRC_QUAD_DEGREE_INITIAL,
                                     ) -> dict:
    """Single-mesh MMS recovery on the production graded rectangle.

    Mirrors ``solver_demo_slide15_no_speculative_cs.py`` (Nx=8, Ny=80,
    beta=3.0).  Returns ``_solve_mms_on_mesh`` output plus ``mesh_label``
    + ``elapsed_seconds``.
    """
    sp = make_sp_production_muh()
    snes_params = _extract_solver_parameters(sp)
    if verbose:
        print(
            f"  graded mesh Nx={Nx} Ny={Ny} beta={beta} "
            f"L_eff_m={L_EFF_M_MMS:.2e}", flush=True,
        )

    t0 = time.time()
    mesh = make_graded_rectangle_mesh(
        Nx=Nx, Ny=Ny, beta=beta,
        domain_height_hat=L_EFF_M_MMS / 1.0e-4,
    )
    errs = _solve_mms_on_mesh(mesh, sp, snes_params, quad_degree=quad_degree)
    elapsed = time.time() - t0
    errs["mesh_label"] = f"graded Nx={Nx}, Ny={Ny}, beta={beta}"
    errs["elapsed_seconds"] = float(elapsed)
    if verbose:
        status = "ok" if errs["newton_converged"] else "FAIL"
        print(
            f"  [graded] [{status}] iters={errs.get('newton_iterations', -1)} "
            f"u_O2_L2={errs.get('u_O2_L2'):.3e} "
            f"phi_L2={errs.get('phi_L2'):.3e} "
            f"({elapsed:.1f}s)", flush=True,
        )
    return errs


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_banner(N_list, *, quad_degree: int) -> None:
    print("=" * 80)
    print("  MMS — logc_muh + Cs+/SO4(2-) multi-ion + Stern Robin BV")
    print("=" * 80)
    print(f"  V_RHE                 = {V_RHE_TEST} V (phi_app_model = {V_RHE_TEST / V_T:.4f})")
    print(f"  K0_R4e_factor         = {K0_R4E_FACTOR_MMS:g}")
    print(f"  C_S                   = {STERN_C_S_F_M2} F/m^2 (direct, no two-stage anchor)")
    print(f"  delta_(O2,H2O2,H)     = {DELTA_PERTURB}")
    print(f"  (alpha0,alpha1,gamma) = ({ALPHA0}, {ALPHA1}, {GAMMA})")
    print(f"  exponent_clip         = {EXPONENT_CLIP}")
    print(f"  u_clamp               = {U_CLAMP}")
    print(f"  src quad degree       = {quad_degree}")
    print(f"  mesh sweep            = {list(N_list)}")
    print("=" * 80)


def compute_rates(h_list, err_list):
    rates = [None]
    for k in range(1, len(h_list)):
        if err_list[k] > 0 and err_list[k - 1] > 0:
            rates.append(log(err_list[k - 1] / err_list[k]) / log(h_list[k - 1] / h_list[k]))
        else:
            rates.append(None)
    return rates


def format_table(results: dict) -> str:
    lines = ["", "=" * 110]
    lines.append("  Full Error Table — logc_muh + Cs+/SO4 + Stern")
    lines.append("=" * 110)
    h_list = results["h"]

    fields: List[str] = []
    for f in FIELD_NAMES:
        fields.append(f"{f}_L2")
        fields.append(f"{f}_H1")

    header = f"  {'N':>4} {'h':>8}  "
    for fn in fields:
        header += f"{fn:>10}  {'rate':>5}  "
    lines.append(header)
    lines.append("-" * 110)

    rates = {fn: compute_rates(h_list, results[fn]) for fn in fields}
    for k in range(len(results["N"])):
        row = f"  {results['N'][k]:>4} {results['h'][k]:>8.4f}  "
        for fn in fields:
            r = rates[fn][k]
            r_str = f"{r:.2f}" if r is not None else "---"
            row += f"{results[fn][k]:>10.3e}  {r_str:>5}  "
        lines.append(row)
    lines.append("=" * 110)
    return "\n".join(lines)


def format_summary(results: dict) -> str:
    lines = ["", "=" * 80]
    lines.append("  MMS — Convergence Rate Summary")
    lines.append("=" * 80)
    h_list = results["h"]
    all_pass = True
    for f, label in zip(FIELD_NAMES, SPECIES_LABELS):
        for norm in ("L2", "H1"):
            key = f"{f}_{norm}"
            rates = compute_rates(h_list, results[key])
            final = rates[-1] if rates[-1] is not None else 0.0
            lo = 1.8 if norm == "L2" else 0.8
            status = "PASS" if final >= lo else "FAIL"
            if status == "FAIL":
                all_pass = False
            lines.append(
                f"  {label:>10s} {norm}: rate = {final:.4f}  (>= {lo:.1f})  [{status}]"
            )
    lines.append("-" * 80)
    lines.append(f"  Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    lines.append("=" * 80)
    return "\n".join(lines)


def plot_convergence(results: dict, out_dir: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    h = np.array(results["h"])
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]

    ax = axes[0]
    for f, lbl, c in zip(FIELD_NAMES, SPECIES_LABELS, colors):
        ax.loglog(h, results[f"{f}_L2"], "o-", color=c, linewidth=1.5,
                  markersize=5, label=f"{lbl} $L^2$")
    h_ref = np.array([h[0], h[-1]])
    scale = results["u_O2_L2"][0] / h[0] ** 2
    ax.loglog(h_ref, scale * h_ref ** 2, "k:", linewidth=0.8, label=r"$O(h^2)$")
    ax.set_xlabel("$h$"); ax.set_ylabel("$L^2$ error")
    ax.set_title("$L^2$ Convergence (logc_muh + multi-ion + Stern)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which="both")

    ax = axes[1]
    for f, lbl, c in zip(FIELD_NAMES, SPECIES_LABELS, colors):
        ax.loglog(h, results[f"{f}_H1"], "o-", color=c, linewidth=1.5,
                  markersize=5, label=f"{lbl} $H^1$")
    scale = results["u_O2_H1"][0] / h[0] ** 1
    ax.loglog(h_ref, scale * h_ref ** 1, "k-.", linewidth=0.8, label=r"$O(h^1)$")
    ax.set_xlabel("$h$"); ax.set_ylabel("$H^1$ error")
    ax.set_title("$H^1$ Convergence (logc_muh + multi-ion + Stern)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which="both")

    fig.suptitle(
        f"MMS: logc_muh + Cs+/SO4 + Stern  "
        f"(V_RHE = {V_RHE_TEST} V, K0_R4e_factor = {K0_R4E_FACTOR_MMS:g})",
        fontsize=11,
    )
    plt.tight_layout()
    png = os.path.join(out_dir, "mms_logc_muh_multi_ion_stern.png")
    fig.savefig(png, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return png


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--Nvals", type=int, nargs="+", default=list(MESH_SIZES))
    parser.add_argument("--quad-degree", type=int, default=SRC_QUAD_DEGREE_INITIAL)
    parser.add_argument(
        "--out-dir", default="StudyResults/mms_logc_muh_multi_ion_stern",
    )
    args = parser.parse_args()

    out_dir = os.path.join(_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    results = run_mms(args.Nvals, quad_degree=args.quad_degree)
    if sum(results["newton_converged"]) < 2:
        print("\n[ERROR] Not enough converged solves for convergence analysis.")
        return 1

    print(format_table(results))
    print(format_summary(results))

    summary_path = os.path.join(out_dir, "mms_logc_muh_summary.txt")
    with open(summary_path, "w") as f:
        f.write("MMS — logc_muh + Cs+/SO4 + Stern\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(format_table(results) + "\n\n")
        f.write(format_summary(results) + "\n")
    print(f"\n[MMS] Summary saved -> {summary_path}", flush=True)

    json_path = os.path.join(out_dir, "convergence_data.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[MMS] JSON saved -> {json_path}", flush=True)

    png = plot_convergence(results, out_dir)
    print(f"[MMS] Plot saved -> {png}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
