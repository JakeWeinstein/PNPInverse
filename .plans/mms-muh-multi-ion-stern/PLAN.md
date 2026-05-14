# Implementation Plan — MMS for `logc_muh` + multi-ion + Stern stack

**Status:** plan only; not yet executed.
**Backs:** `docs/solver/mms_pnpbv_muh_multi_ion_stern_derivation.md` (post-GPT-critique session 38, verdict APPROVED).
**Hardened by:** GPT-critique session 39, 7 rounds, 62 accepted issues.
**Owner:** PNPInverse / forward-solver verification.
**Date:** 2026-05-14.

## 0. Scope & non-goals

Build a convergence-rate MMS test that verifies the production PNP–BV stack
used by `scripts/studies/solver_demo_slide15_no_speculative_cs.py`:
`logc_muh` formulation, 3 dynamic species (O₂, H₂O₂, H⁺) with physical
hard-sphere `a_nondim`, Cs⁺/SO₄²⁻ multi-ion shared-θ Bikerman closure,
Stern Robin BC at the electrode, parallel R2e + R4e log-rate BV. Three test
classes:

- `TestMMSConvergence`: UnitSquareMesh N ∈ {8, 16, 32, 64}; assert L2 slope
  ≥ 1.8, H1 slope ≥ 0.8, R² > 0.99 per primary unknown
  (u_O2, u_H2O2, μ_H, φ). Newton-convergence is a precondition fixture.
- `TestMMSProductionGradedMesh`: single-solve recovery on (Nx=8, Ny=80, β=3);
  per-field thresholds + Newton-iteration cap (< 30).
- `TestMMSAsserts`: 12 parametrized broken-config tests, all caught by
  the MMS invariant layers (prebuild / postbuild / runtime).

Non-goals (deferred): Stern-off cross-check, K0_R4e_factor=1 secondary
sweep, clip-activation MMS, saturation-active MMS, time-dependent MMS,
adjoint Taylor-test, c_ref-anchored anodic branch coverage. See derivation
§8.

## 1. Files

```
scripts/verification/mms_pnpbv_muh_multi_ion_stern.py   NEW  (~900 LOC)
tests/test_mms_logc_muh_multi_ion_stern.py              NEW  (~350 LOC)
```

No edits to production code. Existing `conftest.py::skip_without_firedrake`
is reused.

## 2. Top-level constants and module guards

`tests/test_mms_logc_muh_multi_ion_stern.py` top:

```python
import os
import pytest

if os.environ.get("PYTEST_XDIST_WORKER") is not None:
    pytest.skip(
        "MMS multi-ion + Stern tests are not xdist-safe (shared Firedrake "
        "TSFC cache and StudyResults/ outputs). Run as "
        "`pytest -p no:xdist tests/test_mms_logc_muh_multi_ion_stern.py`.",
        allow_module_level=True,
    )
```

`scripts/verification/mms_pnpbv_muh_multi_ion_stern.py` constants:

```python
V_RHE_TEST = 0.55                       # V vs RHE — demo anchor
DT_LARGE   = 1.0e15
T_END_LARGE = 1.0e15
K0_R4E_FACTOR_MMS = 1.0e-18             # derivation §5.5; bounded R4e/R2e
STERN_C_S_F_M2  = 0.20                  # production target; NO two-stage anchor
DELTA_PERTURB   = (0.30, 0.30, 0.30)    # (O2, H2O2, H+) — matches existing MMS
ALPHA0, ALPHA1, GAMMA = 0.5, 0.5, 0.5
SRC_QUAD_DEGREE_INITIAL = 8             # candidate; final pinned by pilot 10.1
MESH_SIZES   = (8, 16, 32, 64)
```

## 3. Build-time pipeline (the load-bearing ordering invariant)

```
prebuild_config_asserts(sp)                          ← phase 0: sp only
  ↓
build_context(sp) → build_forms(ctx, sp)
  ↓
postbuild_static_ctx_asserts(ctx, sp)                ← phase 1: built ctx
  ↓
OwnedCoeffTracker()                                  (per-mesh, per-run)
  ↓
manuf    = _make_manufactured_fields(mesh, ctx, sp, owned=owned)
closure  = _build_shared_theta_closure_ex(..., owned=owned)
  ↓
pre_rates_margin_asserts(ctx, sp, manuf, closure)    ← phase 2a: clamps & floors
  ↓                                                     (MUST pass before rates;
                                                         rate builder uses
                                                         unclipped η whose
                                                         validity depends on
                                                         η-margin)
rxn_rates = _build_bv_rates_ex(..., owned=owned)
  ↓
post_rates_invariants(ctx, sp, manuf, rxn_rates)     ← phase 2b: R_ratio finite window
  ↓
sources   = _build_source_terms(ctx, sp, manuf, closure, rxn_rates,
                                  owned=owned, quad_degree=quad_degree)
  ↓
source_independence_asserts(sources, ctx, owned)     ← phase 3: extract_coefficients
  ↓                                                     vs FORBIDDEN ∪ ALLOWED
inject_source_terms(ctx, sources, quad_degree=quad_degree)
  ↓                                                     (mutates F_res)
U_manuf = interpolate_U_manuf(ctx, manuf)
U.assign(U_manuf); U_prev.assign(U_manuf)
  ↓
snapshots = snapshot_live_coeffs(ctx)
  ↓
solver.solve()
  ↓
assert_live_coeffs_unchanged(ctx, snapshots)
  ↓
compute per-field L2/H1 errors  (μ_H, not u_H, for proton)
```

## 4. Helpers and primitives

```python
import math
import fd  # firedrake

def _assert_close(name: str, got: float, expected: float,
                  *, rel: float = 1e-9, abs_tol: float = 0.0) -> None:
    """math.isclose-based equality assert. Replaces pytest.approx
    inside scripts/verification (which must run standalone)."""
    if not math.isclose(got, expected, rel_tol=rel, abs_tol=abs_tol):
        raise AssertionError(
            f"_assert_close({name}): got {got!r}, expected {expected!r} "
            f"(rel_tol={rel}, abs_tol={abs_tol})"
        )


class OwnedCoeffTracker:
    """Per-mesh ledger of fd.Constant objects created by MMS builders.
    Threaded into every expression-construction helper so the source-
    independence check has an identity-based whitelist."""
    def __init__(self):
        self._owned: set = set()
    def constant(self, value: float, *, label: str = "") -> fd.Constant:
        c = fd.Constant(float(value))
        self._owned.add(c)
        return c
    @property
    def coeffs(self) -> frozenset:
        return frozenset(self._owned)


def _expr_min(expr, mesh, *, degree: int) -> float:
    """Pointwise min via DG-k interpolation. fd.interpolate (not project)
    to avoid L2-projection smoothing of extrema."""
    P = fd.interpolate(expr, fd.FunctionSpace(mesh, 'DG', degree))
    return float(P.dat.data_ro.min())

def _expr_max(expr, mesh, *, degree: int) -> float:
    P = fd.interpolate(expr, fd.FunctionSpace(mesh, 'DG', degree))
    return float(P.dat.data_ro.max())

def _expr_abs_max(expr, mesh, *, degree: int) -> float:
    P = fd.interpolate(expr, fd.FunctionSpace(mesh, 'DG', degree))
    import numpy as np
    return float(np.abs(P.dat.data_ro).max())

def _expr_indicator_measure(expr, threshold: float, *, mesh, degree: int,
                            comparison: str = "lt") -> float:
    """Measure of {x: expr {<,>} threshold} via quadrature, explicit
    degree, fd.Constant threshold (avoids UFL coercion)."""
    thr = fd.Constant(float(threshold))
    cond = fd.lt(expr, thr) if comparison == "lt" else fd.gt(expr, thr)
    dx_q = fd.dx(domain=mesh, degree=degree)
    return float(fd.assemble(
        fd.conditional(cond, fd.Constant(1.0), fd.Constant(0.0)) * dx_q
    ))

def _domain_volume(mesh, *, degree: int) -> float:
    """Domain volume at matching quadrature degree."""
    return float(fd.assemble(
        fd.Constant(1.0) * fd.dx(domain=mesh, degree=degree)
    ))

def _phi_sub_index(ctx) -> int:
    """Resolve relative phi_index (which may be -1) to absolute index
    for Function.sub(i) calls."""
    indices = ctx['mixed_space_indices']
    raw = indices.phi_index
    n_subs = ctx['W'].num_sub_spaces()
    return raw if raw >= 0 else n_subs + raw

def _snapshot_live_coeffs(ctx) -> dict:
    """Capture floating-point values of live continuation coefficients
    so we can verify they didn't change between source build and solve."""
    return {
        "phi_applied": float(ctx['phi_applied_func'].dat.data_ro[0]),
        "stern_coeff": float(ctx['stern_coeff_const'])
                       if ctx.get('stern_coeff_const') is not None else None,
        "k0_funcs":    [float(f.dat.data_ro[0]) for f in ctx.get('bv_k0_funcs', [])],
        "alpha_funcs": [float(f.dat.data_ro[0]) for f in ctx.get('bv_alpha_funcs', [])],
        "z_scale":     float(ctx['boltzmann_z_scale'].dat.data_ro[0])
                       if ctx.get('boltzmann_z_scale') is not None else None,
    }

def _assert_live_coeffs_unchanged(ctx, snapshots: dict) -> None:
    """Verify no continuation setter fired between source build and solve."""
    assert float(ctx['phi_applied_func'].dat.data_ro[0]) == snapshots['phi_applied']
    if snapshots['stern_coeff'] is not None:
        assert float(ctx['stern_coeff_const']) == snapshots['stern_coeff']
    for f, s in zip(ctx.get('bv_k0_funcs', []), snapshots['k0_funcs']):
        assert float(f.dat.data_ro[0]) == s
    # ... similarly for alpha_funcs, z_scale
```

## 5. Invariant phases

### 5.1 Phase 0 — `_assert_prebuild_config_invariants(sp)`

Runs on `sp` only, before `build_forms`. Catches feature-flag drift
that would otherwise crash `build_forms` (e.g. `cation_hydrol_on` →
`resolve_counterion_index` ValueError before our invariants fire).

```python
def _assert_prebuild_config_invariants(sp) -> None:
    params   = sp[10]
    conv_cfg = params.get('bv_convergence', {})
    bv_bc    = params.get('bv_bc', {})

    # Feature flags
    assert not is_water_ionization_enabled(conv_cfg), \
        "MMS requires enable_water_ionization=False"
    assert not is_cation_hydrolysis_enabled(conv_cfg), \
        "MMS requires cation hydrolysis disabled"

    # Formulation, log_rate
    assert str(conv_cfg.get('formulation', '')).lower() == 'logc_muh'
    assert bool(conv_cfg.get('bv_log_rate', False)) is True

    # Clip
    assert bool(conv_cfg.get('clip_exponent', True)) is True
    _assert_close('exponent_clip', float(conv_cfg.get('exponent_clip', 100.0)),
                  100.0, rel=1e-9)

    # Reactions list shape
    rxns = bv_bc.get('reactions', [])
    assert isinstance(rxns, (list, tuple)) and len(rxns) == 2

    # Counterions: exactly two bikerman with EXACT identities (not just count)
    counterions = bv_bc.get('boltzmann_counterions', [])
    bikerman = [e for e in counterions if e.get('steric_mode') == 'bikerman']
    assert len(bikerman) == 2
    # Cs+ identity
    cs = next((e for e in bikerman if int(e['z']) == +1), None)
    assert cs is not None, "Missing Cs+ counterion"
    _assert_close('a_Cs+',    float(cs['a_nondim']),     A_CSPLUS_HAT, rel=1e-9)
    _assert_close('c_b_Cs+',  float(cs['c_bulk_nondim']), C_CSPLUS_HAT, rel=1e-9)
    # SO4 identity
    so4 = next((e for e in bikerman if int(e['z']) == -2), None)
    assert so4 is not None, "Missing SO4(2-) counterion"
    _assert_close('a_SO4',    float(so4['a_nondim']),     A_SO4_HAT, rel=1e-9)
    _assert_close('c_b_SO4',  float(so4['c_bulk_nondim']), C_SO4_HAT, rel=1e-9)

    # suppress_poisson_source lives on params['nondim'], NOT on conv_cfg
    nondim_cfg = params.get('nondim', {})
    assert not bool(nondim_cfg.get('suppress_poisson_source', False))

    # dt, SNES tolerances
    assert float(sp[2]) >= 1e12
    assert float(params.get('snes_atol', 1e-7)) <= 1e-5
    assert float(params.get('snes_rtol', 1e-8)) <= 1e-7

    # Species identity from sp (sp[8]/sp[4]/sp[6] and bv_bc['species_roles'])
    assert int(sp[0]) == 3
    assert list(sp[4]) == [0, 0, 1]
    roles = list(bv_bc.get('species_roles', []))
    assert roles == ['neutral', 'neutral', 'proton']
    a_vals = list(sp[6])
    _assert_close('a[O2]',   a_vals[0], A_O2_PHYSICAL,   rel=1e-9)
    _assert_close('a[H2O2]', a_vals[1], A_H2O2_PHYSICAL, rel=1e-9)
    _assert_close('a[H]',    a_vals[2], A_HP_PHYSICAL,   rel=1e-9)
```

### 5.2 Phase 1 — `_assert_postbuild_static_ctx_invariants(ctx, sp)`

Runs after `build_forms`. Inspects derived state (Stern conversion,
mixed_space_indices, reaction model values).

```python
def _assert_postbuild_static_ctx_invariants(ctx, sp) -> None:
    scaling  = ctx['nondim']
    conv_cfg = ctx['bv_convergence']      # NOT scaling

    # Stern coefficient nondim conversion produced a positive value
    csm = scaling.get('bv_stern_capacitance_model')
    assert csm is not None and float(csm) > 0

    # muh transform plumbed
    assert ctx.get('logc_muh_transform') is True

    # No Γ slot
    indices = ctx.get('mixed_space_indices')
    assert indices is None or indices.gamma_index is None

    # Reaction identity (E_eq_model, k0_model, etc. populated by build_forms)
    rxns = scaling['bv_reactions']
    r2e, r4e = rxns[0], rxns[1]
    # R2e
    _assert_close('E_eq_R2e', float(r2e['E_eq_model']), E_EQ_R2E_V / V_T, rel=1e-9)
    _assert_close('alpha_R2e', float(r2e['alpha']),     ALPHA_R2E,        rel=1e-9)
    assert int(r2e['n_electrons']) == 2
    assert bool(r2e['reversible']) is True
    assert int(r2e['cathodic_species']) == 0
    assert int(r2e['anodic_species'])   == 1
    assert tuple(r2e['stoichiometry']) == (-1, +1, -2)
    cf2e = r2e['cathodic_conc_factors'][0]
    assert int(cf2e['species']) == 2 and float(cf2e['power']) == 2.0
    _assert_close('c_ref_R2e', float(cf2e['c_ref_nondim']), C_HP_HAT, rel=1e-9)
    _assert_close('k0_R2e', float(r2e['k0_model']), K0_HAT_R2E, rel=1e-9)
    assert float(r2e['k0_model']) > 0.0
    assert bool(r2e.get('enabled', True)) is True
    # R4e
    _assert_close('E_eq_R4e', float(r4e['E_eq_model']), E_EQ_R4E_V / V_T, rel=1e-9)
    _assert_close('alpha_R4e', float(r4e['alpha']),     ALPHA_R4E,        rel=1e-9)
    assert int(r4e['n_electrons']) == 4
    assert bool(r4e['reversible']) is False
    assert int(r4e['cathodic_species']) == 0
    assert r4e.get('anodic_species') is None
    assert tuple(r4e['stoichiometry']) == (-1, 0, -4)
    assert float(r4e['c_ref_model']) == 0.0
    cf4e = r4e['cathodic_conc_factors'][0]
    assert int(cf4e['species']) == 2 and float(cf4e['power']) == 4.0
    _assert_close('k0_R4e', float(r4e['k0_model']),
                  K0_HAT_R4E * K0_R4E_FACTOR_MMS, rel=1e-9)
    assert float(r4e['k0_model']) > 0.0
    assert bool(r4e.get('enabled', True)) is True

    # c0_model_vals (nondim) lives on ctx['nondim'] after build_forms
    c0_vals = list(scaling['c0_model_vals'])
    _assert_close('c0[O2]',   c0_vals[0], C_O2_HAT,         rel=1e-9)
    _assert_close('c0[H2O2]', c0_vals[1], H2O2_SEED_NONDIM, rel=1e-9)
    _assert_close('c0[H]',    c0_vals[2], C_HP_HAT,         rel=1e-9)
```

### 5.3 Phase 2a — `_assert_pre_rates_margin_invariants(ctx, sp, manuf, closure)`

Clamp/floor margin checks. Must pass BEFORE `_build_bv_rates_ex` runs,
because the rate-builder uses unclipped η expressions whose validity
depends on the η margin (otherwise the source side encodes a model
that diverges from the production-clipped residual).

```python
def _assert_pre_rates_margin_invariants(ctx, sp, *, manuf, closure,
                                         quad_degree: int) -> None:
    mesh    = ctx['mesh']
    scaling = ctx['nondim']
    conv_cfg = ctx['bv_convergence']
    phi_ex  = manuf['phi_ex']
    bv_exp_scale  = float(scaling['bv_exponent_scale'])
    phi_app_model = float(scaling['phi_applied_model'])
    exp_clip = float(conv_cfg['exponent_clip'])    # FROM bv_convergence, NOT nondim
    u_clamp  = float(conv_cfg.get('u_clamp', 30.0))

    # η-margins (one per reaction)
    rxns = scaling['bv_reactions']
    for j, lbl in enumerate(['R2e', 'R4e']):
        eta_expr = bv_exp_scale * (phi_app_model - phi_ex - float(rxns[j]['E_eq_model']))
        eta_amax = _expr_abs_max(eta_expr, mesh, degree=4)
        assert eta_amax < 0.9 * exp_clip, \
            f"η_{lbl} abs max {eta_amax:.3g} too close to clip {exp_clip}"

    # u_clamp margin (per primary unknown; for proton use reconstruction)
    for u_field, name in [(manuf['u_ex'][0], 'O2'),
                          (manuf['u_ex'][1], 'H2O2'),
                          (manuf['u_ex'][2], 'H_recon')]:  # μ_H - em·z_H·φ
        u_amax = _expr_abs_max(u_field, mesh, degree=4)
        assert u_amax < 0.9 * u_clamp, \
            f"|u_{name}| max {u_amax:.3g} too close to clamp {u_clamp}"

    # Ion phi_clamp margin (per counterion)
    phi_amax = _expr_abs_max(phi_ex, mesh, degree=4)
    for ion_label, phi_clamp_k in [('Cs+', PHI_CLAMP_CSPLUS),
                                    ('SO4',  PHI_CLAMP_SO4)]:
        assert phi_amax < 0.9 * phi_clamp_k, \
            f"|φ| max {phi_amax:.3g} too close to clamp_{ion_label} {phi_clamp_k}"

    # free_dyn_floor margin (1 - A_dyn > floor)
    A_dyn_max = _expr_abs_max(closure['A_dyn_ex'], mesh, degree=4)
    assert A_dyn_max < 0.99, f"A_dyn max {A_dyn_max:.3g} too close to 1"

    # packing_floor margin via DG-interp min AND quadrature indicator
    packing_floor = float(conv_cfg.get('packing_floor', 1e-8))
    theta_min = _expr_min(closure['theta_inner_ex'], mesh, degree=4)
    assert theta_min > 10 * packing_floor, \
        f"min(θ_inner) {theta_min:.3g} too close to floor {packing_floor}"
    bad_measure = _expr_indicator_measure(
        closure['theta_inner_ex'], 10 * packing_floor,
        mesh=mesh, degree=quad_degree, comparison='lt',
    )
    vol = _domain_volume(mesh, degree=quad_degree)
    assert bad_measure < 1e-12 * vol
```

### 5.4 Phase 2b — `_assert_post_rates_invariants(ctx, sp, manuf, rxn_rates)`

R_ratio finite window check. Needs rxn_rates so it runs after rate
construction.

```python
def _assert_post_rates_invariants(ctx, sp, *, manuf, rxn_rates,
                                    quad_degree: int) -> None:
    mesh = ctx['mesh']
    ds_e = fd.ds(ctx['bv_settings']['electrode_marker'],
                 domain=mesh, degree=quad_degree)   # explicit degree
    R2e_norm = float(fd.assemble(rxn_rates[0]**2 * ds_e))**0.5
    R4e_norm = float(fd.assemble(rxn_rates[1]**2 * ds_e))**0.5
    R_ratio = R4e_norm / max(R2e_norm, 1e-300)
    assert 10 < R_ratio < 1e5, (
        f"R4e/R2e = {R_ratio:.3e} outside finite window — "
        f"K0_R4e_factor likely mis-set or V_RHE wrong"
    )
```

### 5.5 Phase 3 — `_assert_source_independence(sources, ctx, owned)`

Per-mesh allowed-geometry set, identity-only whitelist.

```python
@dataclass
class MMSSourceTerms:
    S_c: list                # per dynamic species (UFL on dx)
    g_elec: list             # per dynamic species (UFL on ds_elec)
    S_phi: object            # UFL on dx
    g_S: object              # UFL on ds_elec

def _allowed_geometry_for_mesh(mesh, *, quad_degree: int) -> set:
    """Probe extract_coefficients on a representative geometry expression
    using the SAME extraction path as _assert_source_independence."""
    from ufl.algorithms.analysis import extract_coefficients
    x, y = fd.SpatialCoordinate(mesh)
    n_vec = fd.FacetNormal(mesh)
    test_expr = fd.cos(fd.pi * x) * (1.0 - y)**2
    test_form = fd.dot(fd.grad(test_expr), n_vec) * fd.ds(
        domain=mesh, degree=quad_degree
    )
    return (set(extract_coefficients(test_expr))
            | set(extract_coefficients(test_form)))

def _assert_source_independence(sources: MMSSourceTerms, ctx: dict,
                                 owned: OwnedCoeffTracker,
                                 *, quad_degree: int) -> None:
    from ufl.algorithms.analysis import extract_coefficients
    mesh = ctx['mesh']
    FORBIDDEN = {ctx['U'], ctx['U_prev']}
    ALLOWED_LIVE = {
        ctx['phi_applied_func'],
        ctx.get('stern_coeff_const'),
        *ctx.get('bv_k0_funcs', []),
        *ctx.get('bv_alpha_funcs', []),
        ctx.get('boltzmann_z_scale'),
    }
    ALLOWED_LIVE.discard(None)
    ALLOWED_GEOMETRY = _allowed_geometry_for_mesh(mesh, quad_degree=quad_degree)
    ALLOWED = ALLOWED_LIVE | owned.coeffs | ALLOWED_GEOMETRY

    def _iter(sources):
        for i, s in enumerate(sources.S_c):      yield (f"S_c_{i}", s)
        for i, s in enumerate(sources.g_elec):  yield (f"g_elec_{i}", s)
        yield ("S_phi", sources.S_phi)
        yield ("g_S",   sources.g_S)

    for label, S in _iter(sources):
        coeffs = set(extract_coefficients(S))
        bad = coeffs & FORBIDDEN
        assert not bad, f"Source {label} references {bad} — independence violation"
        unknown = coeffs - ALLOWED
        assert not unknown, (
            f"Source {label} references unrecognized coefficients {unknown}. "
            f"Either route through OwnedCoeffTracker, add to ALLOWED_LIVE, "
            f"or expand ALLOWED_GEOMETRY probe."
        )
```

## 6. Builders

(All threaded with `owned: OwnedCoeffTracker`; every `fd.Constant(...)`
inside MMS construction goes through `owned.constant(...)`.)

```python
def _make_manufactured_fields(mesh, ctx, sp, *, owned: OwnedCoeffTracker) -> dict:
    """Returns dict with c_ex, u_ex (proton entry is recon = ln c_H_ex),
    mu_H_ex, phi_ex, phi_app_model."""
    ...

def _build_shared_theta_closure_ex(*, counterions_cfg, a_dyn, c0_dyn, z_dyn,
                                    phi_ex, c_dyn_ex,
                                    owned: OwnedCoeffTracker) -> dict:
    """Independent UFL for q_k_ex, D_ex, c_steric_ex, P_k_ex, rho_k_ex,
    theta_b, A_dyn_ex, theta_inner_ex, mu_steric_ex.

    Mirrors boltzmann.py:91–268 algebraically. Does NOT consume the
    production ctx['steric_boltzmann'] bundle (independence policy).

    The source-builder closure uses UNCLIPPED, UNFLOORED expressions.
    Validity relies on _assert_pre_rates_margin_invariants having
    passed. The closure-algebra smoke test (pilot 10.9) uses a SEPARATE
    helper that mirrors production clamps/floors exactly, for the
    1e-9 comparison."""
    ...

def _build_bv_rates_ex(*, reactions_cfg, scaling, u_ex, phi_ex, phi_app_model,
                       owned: OwnedCoeffTracker) -> list:
    """Per-reaction R_j UFL with muh substitution `u_H = μ_H - em·z_H·φ`."""
    ...

def _build_source_terms(ctx, sp, *, manuf, closure, rxn_rates,
                         owned: OwnedCoeffTracker,
                         quad_degree: int) -> MMSSourceTerms:
    """Pure construction; does NOT mutate ctx['F_res']. Returns
    MMSSourceTerms container so _assert_source_independence can
    inspect each term before injection."""
    mesh = ctx['mesh']
    n_vec = fd.FacetNormal(mesh)
    # Per derivation §4:
    # S_c_i  = -div(J_i_ex)
    # g_i^elec = J_i_ex·n - Σ_j s_{ij} R_j_ex
    # S_phi  = -ε∇²φ_ex - ρ_c(z_H c_H_ex + Σ_k z_k c_k^ster,ex)
    # g_S    = ε(∇φ_ex·n) - C_S^model(φ_app^model - φ_ex)
    ...
    return MMSSourceTerms(S_c=S_c, g_elec=g_elec, S_phi=S_phi, g_S=g_S)

def _inject_source_terms(ctx, sources: MMSSourceTerms, *,
                          quad_degree: int) -> dict:
    """Mutate F_res by subtracting source UFL; rederive J_form."""
    mesh = ctx['mesh']
    dx_q = fd.dx(domain=mesh, degree=quad_degree)
    ds_q = fd.ds(ctx['bv_settings']['electrode_marker'],
                 domain=mesh, degree=quad_degree)
    F = ctx['F_res']
    v_tests = fd.TestFunctions(ctx['W'])
    indices = ctx['mixed_space_indices']
    v_list = v_tests[indices.species_slice]
    w_test = v_tests[_phi_sub_index(ctx)]
    for i, S in enumerate(sources.S_c):
        F = F - S * v_list[i] * dx_q
    for i, g in enumerate(sources.g_elec):
        F = F - g * v_list[i] * ds_q
    F = F - sources.S_phi * w_test * dx_q
    F = F - sources.g_S    * w_test * ds_q
    ctx['F_res'] = F
    ctx['J_form'] = fd.derivative(F, ctx['U'])
    return ctx

def _build_manufactured_source(ctx, sp, *, manuf, closure, rxn_rates,
                                 quad_degree: int,
                                 owned: OwnedCoeffTracker) -> dict:
    """Orchestrates: build → independence-check → inject."""
    sources = _build_source_terms(ctx, sp, manuf=manuf, closure=closure,
                                    rxn_rates=rxn_rates, owned=owned,
                                    quad_degree=quad_degree)
    _assert_source_independence(sources, ctx, owned, quad_degree=quad_degree)
    return _inject_source_terms(ctx, sources, quad_degree=quad_degree)
```

## 7. Factory: `make_sp_production_muh()`

Mirrors `solver_demo_slide15_no_speculative_cs.py:_make_sp` with:

- `eta_hat = V_RHE_TEST / V_T` (fixed).
- `dt = DT_LARGE, t_end = T_END_LARGE`.
- `K0_R4e_factor = K0_R4E_FACTOR_MMS = 1e-18`.
- `stern_capacitance_f_m2 = STERN_C_S_F_M2 = 0.20` (direct; no two-stage anchor).
- SNES: `atol=1e-5, rtol=1e-8, stol=1e-12, max_it=80`, l2 line search.
- `formulation="logc_muh"`, `log_rate=True`.
- `multi_ion_enabled=True`, `boltzmann_counterions=[Cs+, SO4²⁻]`.
- Physical hard-sphere a_nondim for dynamic species.

**Verification** (NOT directly inspectable on returned tuple):

```python
sp = make_sp_production_muh()
mesh = fd.UnitSquareMesh(8, 8)
ctx = build_context(sp, mesh=mesh); ctx = build_forms(ctx, sp)
assert ctx['nondim']['bv_stern_capacitance_model'] > 0
```

## 8. Outer `run_mms` loop

```python
def run_mms(N_list=MESH_SIZES, *, quad_degree=SRC_QUAD_DEGREE_INITIAL,
            verbose=True) -> dict:
    out = {
        "N": [], "h": [],
        "newton_converged": [], "newton_iterations": [], "snes_reason": [],
        "F_res_l2_initial": [], "F_res_l2_final": [],
        "u_O2_L2": [], "u_O2_H1": [],
        "u_H2O2_L2": [], "u_H2O2_H1": [],
        "mu_H_L2": [], "mu_H_H1": [],
        "phi_L2": [], "phi_H1": [],
        "c_H_L2": [],   # diagnostic only
    }
    for N in N_list:
        mesh = fd.UnitSquareMesh(N, N)
        sp = make_sp_production_muh()
        _assert_prebuild_config_invariants(sp)
        ctx = build_context(sp, mesh=mesh)
        ctx = build_forms(ctx, sp)
        _assert_postbuild_static_ctx_invariants(ctx, sp)

        owned = OwnedCoeffTracker()
        manuf   = _make_manufactured_fields(mesh, ctx, sp, owned=owned)
        closure = _build_shared_theta_closure_ex(..., owned=owned)
        _assert_pre_rates_margin_invariants(ctx, sp, manuf=manuf,
                                              closure=closure,
                                              quad_degree=quad_degree)
        rxn_rates = _build_bv_rates_ex(..., owned=owned)
        _assert_post_rates_invariants(ctx, sp, manuf=manuf,
                                        rxn_rates=rxn_rates,
                                        quad_degree=quad_degree)
        _build_manufactured_source(ctx, sp, manuf=manuf, closure=closure,
                                     rxn_rates=rxn_rates,
                                     quad_degree=quad_degree, owned=owned)

        U_manuf = _interpolate_U_manuf(ctx, manuf)
        ctx['U'].assign(U_manuf); ctx['U_prev'].assign(U_manuf)
        snaps = _snapshot_live_coeffs(ctx)

        F_initial = float(fd.norm(ctx['F_res'], norm_type='L2'))
        problem = fd.NonlinearVariationalProblem(ctx['F_res'], ctx['U'],
                                                 bcs=ctx['bcs'], J=ctx['J_form'])
        solver  = fd.NonlinearVariationalSolver(problem, solver_parameters=...)
        try:
            solver.solve()
            converged = True
            iters     = int(solver.snes.getIterationNumber())
            reason    = str(solver.snes.getConvergedReason())
        except fd.ConvergenceError as exc:
            converged = False; iters = -1; reason = str(exc)

        _assert_live_coeffs_unchanged(ctx, snaps)
        F_final = float(fd.norm(ctx['F_res'], norm_type='L2'))

        out["N"].append(N); out["h"].append(1.0/N)
        out["newton_converged"].append(converged)
        out["newton_iterations"].append(iters)
        out["snes_reason"].append(reason)
        out["F_res_l2_initial"].append(F_initial)
        out["F_res_l2_final"].append(F_final)

        if not converged:
            for k in ("u_O2_L2", "u_O2_H1", "u_H2O2_L2", "u_H2O2_H1",
                      "mu_H_L2", "mu_H_H1", "phi_L2", "phi_H1", "c_H_L2"):
                out[k].append(float('nan'))
            continue

        # Per-field error norms (proton on μ_H, NOT u_H)
        phi_idx_abs = _phi_sub_index(ctx)
        out["u_O2_L2"].append(_ufl_l2_error(manuf['u_ex'][0],   ctx['U'].sub(0), mesh, degree=quad_degree))
        out["u_O2_H1"].append(_ufl_h1_error(manuf['u_ex'][0],   ctx['U'].sub(0), mesh, degree=quad_degree))
        out["u_H2O2_L2"].append(_ufl_l2_error(manuf['u_ex'][1], ctx['U'].sub(1), mesh, degree=quad_degree))
        out["u_H2O2_H1"].append(_ufl_h1_error(manuf['u_ex'][1], ctx['U'].sub(1), mesh, degree=quad_degree))
        out["mu_H_L2"].append(_ufl_l2_error(manuf['mu_H_ex'],   ctx['U'].sub(2), mesh, degree=quad_degree))
        out["mu_H_H1"].append(_ufl_h1_error(manuf['mu_H_ex'],   ctx['U'].sub(2), mesh, degree=quad_degree))
        out["phi_L2"].append(_ufl_l2_error(manuf['phi_ex'],     ctx['U'].sub(phi_idx_abs), mesh, degree=quad_degree))
        out["phi_H1"].append(_ufl_h1_error(manuf['phi_ex'],     ctx['U'].sub(phi_idx_abs), mesh, degree=quad_degree))
        # Diagnostic: c_H = exp(μ_H - em·z_H·φ)
        em_z_H = float(ctx['nondim']['electromigration_prefactor']) * 1.0
        c_H_h = fd.exp(ctx['U'].sub(2) - fd.Constant(em_z_H) * ctx['U'].sub(phi_idx_abs))
        c_H_ex_func = fd.Constant(C_HP_HAT) * (1 + fd.Constant(0.3) * fd.cos(fd.pi*fd.SpatialCoordinate(mesh)[0])*(1-fd.SpatialCoordinate(mesh)[1])**2)
        out["c_H_L2"].append(_ufl_l2_error(c_H_ex_func, c_H_h, mesh, degree=quad_degree))
    return out
```

## 9. Pilot validation (10 sub-steps, run BEFORE pytest assertions land)

Output to `StudyResults/mms_logc_muh_multi_ion_stern/pilot/`:

| # | Step | Purpose |
|---|---|---|
| 10.0 | Geometry coefficient pre-flight | Document `extract_coefficients(geom_expr)` behavior; verify `_allowed_geometry_for_mesh` returns a sane set |
| 10.1 | Quadrature sweep `(N, degree) ∈ {32,64} × {6,8,10,12,16}` | Pin SRC_QUAD_DEGREE to smallest degree where errors plateau on BOTH meshes |
| 10.2 | Newton-convergence sanity at N=8 | Source builder works at smallest mesh |
| 10.3 | R_ratio check at N=32 | Confirm R4e/R2e ∈ (10, 1e5) at K0_R4e_factor=1e-18 |
| 10.4 | θ_inner discrete-min check | DG-interp .min() AND quadrature indicator agree |
| 10.5 | Perturbed-IC pilot | U_init = U_manuf + ε·sin(πx)sin(πy); BCs reapplied; Newton must converge back to within 1% of unperturbed L2 |
| 10.7 | Residual reduction (perturbed only) | ||F_res(U_init)|| / ||F_res(U_final)|| > 100 (skip if initial < 1e3·atol) |
| 10.8 | SNES tolerance sensitivity | (atol, rtol) ∈ {(1e-5,1e-8), (1e-7,1e-10)} at N=32 and N=64 |
| 10.9 | Closure algebra smoke | Independent closure-mirror (with full clamps/floors/z_scale) vs production bundle at U.assign(U_manuf); agreement to 1e-9 on c_steric, packing_contribution, charge_density (per-ion, UNSCALED) |

(No 10.6 — was iteration-count floor, removed in R3 as overcorrection.)

## 10. Tests

### 10.1 `TestMMSConvergence`

```python
class TestMMSConvergence:
    MESH_SIZES = (8, 16, 32, 64)
    EXPECTED_L2_RATE = 2.0
    EXPECTED_H1_RATE = 1.0
    RATE_TOL = 0.2
    MIN_R_SQUARED = 0.99
    FIELDS = ["u_O2", "u_H2O2", "mu_H", "phi"]

    @pytest.fixture(scope="class")
    def mms_results(self):
        run_mms = _import_run_mms()
        res = run_mms(self.MESH_SIZES, verbose=True)
        if not all(res["newton_converged"]):
            failed = [N for N, c in zip(res["N"], res["newton_converged"]) if not c]
            pytest.fail(f"Newton failed on meshes: {failed} "
                        f"(reasons: {res['snes_reason']})")
        return res

    def test_l2_convergence_rates(self, mms_results): ...
    def test_h1_convergence_rates(self, mms_results): ...
    def test_gci_output(self, mms_results): ...
    def test_save_convergence_artifacts(self, mms_results): ...
```

(Newton-convergence is a fixture precondition — no separate
`test_newton_converges` needed.)

### 10.2 `TestMMSProductionGradedMesh`

```python
class TestMMSProductionGradedMesh:
    # Per-field thresholds pinned from pilot — populated post-pilot
    L2_THRESHOLDS = {"u_O2": ..., "u_H2O2": ..., "mu_H": ..., "phi": ...}
    H1_THRESHOLDS = {"u_O2": ..., "u_H2O2": ..., "mu_H": ..., "phi": ...}
    NEWTON_ITER_CAP = 30

    @pytest.fixture(scope="class")
    def graded_mesh_results(self):
        verify = _import_graded_verifier()
        return verify(verbose=True)

    def test_newton_converges_within_iteration_cap(self, graded_mesh_results):
        assert graded_mesh_results["newton_converged"] is True
        assert graded_mesh_results["newton_iterations"] < self.NEWTON_ITER_CAP

    def test_l2_recovery(self, graded_mesh_results):
        for field, thr in self.L2_THRESHOLDS.items():
            err = graded_mesh_results[f"{field}_L2"]
            assert math.isfinite(err) and err < thr

    def test_h1_recovery(self, graded_mesh_results):
        for field, thr in self.H1_THRESHOLDS.items():
            err = graded_mesh_results[f"{field}_H1"]
            assert math.isfinite(err) and err < thr
```

### 10.3 `TestMMSAsserts` (12 parametrized broken-config tests)

```python
BROKEN_CONFIGS = [
    # (label, mutate, layer, regex)
    ("formulation_logc",  _force_formulation("logc"),       "prebuild",  r"formulation.*logc_muh"),
    ("log_rate_off",       _force_log_rate(False),           "prebuild",  r"bv_log_rate"),
    ("water_on",           _force_water_ionization(True),    "prebuild",  r"enable_water_ionization"),
    ("cation_hydrol_on",   _force_cation_hydrol(True),       "prebuild",  r"cation hydrolysis"),
    ("dt_small",           _force_dt(0.1),                   "prebuild",  r"dt"),
    ("snes_loose",         _force_snes_tol(atol=1e-2),       "prebuild",  r"snes_atol"),
    ("poisson_suppressed", _force_suppress_poisson(True),    "prebuild",  r"suppress_poisson_source"),
    ("missing_reaction",   _drop_reaction("R4e"),            "prebuild",  r"len.*rxns|n_reactions"),
    ("one_counterion",     _drop_counterion("SO4"),          "prebuild",  r"bikerman.*counterion|len.*bikerman"),
    ("wrong_counterion",   _replace_counterion("Cs","ClO4"), "prebuild",  r"Cs\+|ClO4|a_Cs|c_b_Cs"),
    ("no_stern",            _force_stern(None),               "postbuild", r"bv_stern_capacitance_model"),
    ("k0_r4e_wrong",        _force_k0_r4e_factor(1.0),        "postbuild", r"k0_R4e|k0_model.*R4e"),
]


class TestMMSAsserts:
    @pytest.mark.parametrize("label,mutate,layer,match", BROKEN_CONFIGS)
    def test_broken_config_caught(self, label, mutate, layer, match):
        sp = mutate(make_sp_production_muh())
        mesh = fd.UnitSquareMesh(8, 8)
        with pytest.raises(AssertionError, match=match):
            _prepare_mms_context_for_asserts(sp, mesh, phase=layer)
```

`_prepare_mms_context_for_asserts(sp, mesh, phase)` stops at the
expected layer per `phase ∈ {prebuild, postbuild, pre_rates, runtime,
source, both}`.

## 11. Dependency graph

```
Step 1 scaffolding
  └─→ Step 2 factory
        └─→ Step 4 helpers (OwnedCoeffTracker, _expr_*, _phi_sub_index, ...)
              └─→ Step 5.1 prebuild asserts
                    └─→ Step 6 manuf field builder
                          └─→ Step 6 closure helper
                                ├─→ Step 5.2 postbuild static asserts
                                ├─→ Step 5.3 pre-rates margin asserts
                                └─→ Step 6 BV rates helper
                                      └─→ Step 5.4 post-rates invariants
                                            └─→ Step 6 source-term builder + independence
                                                  └─→ Step 8 outer run_mms
                                                        ├─→ Step 9 graded mesh
                                                        └─→ Step 10 pilot validation
                                                              ├─→ Step 11.1 TestMMSConvergence
                                                              ├─→ Step 11.2 TestMMSProductionGradedMesh
                                                              └─→ Step 11.3 TestMMSAsserts
```

## 12. Risks and unknowns

| Risk | Mitigation | Evidence required (pilot) |
|---|---|---|
| Newton fails to converge at N=8 | 10.2 sanity test | Newton converged + reason `CONVERGED_*` |
| Convergence rate < 1.8 due to quadrature error | 10.1 two-mesh sweep | Rate stable across degrees 8 → 12 → 16 on both N=32 and N=64 |
| R4e/R2e outside (10, 1e5) | 10.3; retune K0_R4e_factor | Single L2-norm assertion on ds_e |
| θ_inner_ex hits packing_floor | 10.4 dual-method check | DG-interp min > 10·floor AND indicator measure < 1e-12·vol |
| Newton-from-U_manuf degenerates to interpolation test | 10.5 perturbed IC pilot | ≥ 3 iterations on perturbed start; L2 within 1% of unperturbed |
| Initial-residual too small for ratio check | 10.7 guards reduction check with absolute floor | Skip ratio if initial < 1e3·atol |
| Geometry coefficients fail independence check | 10.0 pre-flight + per-mesh whitelist | Pilot logs geometry coeff set |
| Continuation setter fires mid-test | _snapshot/_assert_live_coeffs_unchanged | Post-solve value match |
| Closure transcription bug vs production bundle | 10.9 smoke | 1e-9 agreement on c_steric, P, charge_density |
| SNES tolerance dominates discretization error | 10.8 sensitivity sweep | L2 errors stable across atol/rtol tighter by 10² |
| pytest-xdist cache collision | Module-level skip on PYTEST_XDIST_WORKER | Document in test docstring |
| Test wall time blows up | Mark slow; cap N=64 | Empirical timing in pilot |

## 13. Acceptance criteria

1. `pytest tests/test_mms_logc_muh_multi_ion_stern.py -m slow -p no:xdist` passes end to end.
2. All four primary unknowns achieve L2 slope ≥ 1.8 and H1 slope ≥ 0.8 with R² > 0.99.
3. `verify_on_graded_production_mesh` converges Newton in < 30 iterations and meets per-field thresholds.
4. JSON + PNG artifacts land in `StudyResults/mms_logc_muh_multi_ion_stern/`.
5. **`TestMMSAsserts` passes all 12 parametrized broken-config cases at the expected MMS invariant layer** (mandatory; was "optional" pre-R2).
6. Pilot artifacts (10.0 – 10.9) saved and referenced from the test docstrings.
7. The derivation doc and this plan are linked from both new files' docstrings.

## 14. Estimated effort

- Steps 1–2 (scaffolding + factory): 1 h
- Step 4 (helpers): 2 h
- Step 5 (3-phase invariant harness + asserts): 4 h
- Step 6 (builders + source-term construction + independence check): 5 h
- Step 8–9 (run_mms outer + graded): 3 h
- Step 9 (pilot, includes debugging): 6–10 h
- Step 10 (3 test classes including 12 broken-config parametrization): 4 h

Total: 25–29 hours. The pilot half-budget is deliberate; that's where
unknown unknowns surface.

## 15. Open questions to resolve during implementation

1. Does `ctx['stern_coeff_const']` exist by that exact name on the
   built ctx, or under a different key? Verify at code time and
   adjust the snapshot helper.
2. What does `extract_coefficients` actually return for geometric
   terminals on the Firedrake version we have? Pilot 10.0 documents
   this; whitelist updated accordingly.
3. For the graded mesh in step 9, does the manufactured solution need
   re-design? Expected answer: same shape works (rectangle maps 1-1 to
   unit square), but verify in pilot.
4. Does `boltzmann.py:91–268` change between MMS authoring and merge?
   Pin the test to a specific commit via git-log-reference in the test
   docstring; revisit on any production-side closure refactor.
