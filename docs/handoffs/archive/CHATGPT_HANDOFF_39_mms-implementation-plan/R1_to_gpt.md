# Critique session 39 — Round 1

## Section 1: Context bundle

### Background

This is the implementation plan for the MMS convergence test whose math
derivation went through critique session 38 (verdict APPROVED). The
derivation is at
`docs/solver/mms_pnpbv_muh_multi_ion_stern_derivation.md`; final
revision and 31-issue ledger are at
`docs/handoffs/CHATGPT_HANDOFF_38_mms-logc-muh-multi-ion-stern/FINAL_REVISION.md`.

Now we need to turn that derivation into runnable code. The plan below
sequences the implementation in dependency order, identifies pre-flight
checks (quadrature sweep, Newton-convergence sanity, R_ratio check,
θ_inner discrete-min check) that must happen BEFORE pytest assertions
land, and lists the risks I'm tracking.

### Stack reminder

- `formulation='logc_muh'`, primary unknowns `(u_O2, u_H2O2, μ_H, φ)`,
  `em·z_H = +1`.
- 3 dynamic species O₂, H₂O₂, H⁺ with physical hard-sphere `a_nondim`
  (1.49e−5, 2.42e−5, 6.65e−5).
- 2 analytic Bikerman counterions Cs⁺ + SO₄²⁻ under shared-θ closure.
- Stern Robin BC at electrode, C_S = 0.20 F/m² (production target).
- Parallel R2e (E°=0.695 V, α=0.627, n_e=2, reversible) + R4e
  (E°=1.23 V, α=0.5, n_e=4, irreversible); log-rate; cathodic_conc
  factors c_H power 2 / 4.
- `exponent_clip = 100`, `u_clamp = 100`.
- `enable_water_ionization=False`, `lambda_hydrolysis=0`.

### Key derivation outputs being implemented

1. Manufactured fields:
   - `c_i^ex = c0_i · [1 + δ_i cos(πx)(1−y)²]`, δ_i = 0.30.
   - `φ^ex = (1−y)(α₀ + α₁ cos πx) + γ·y(1−y) cos(πx)`, α₀=α₁=γ=0.5.
   - `μ_H^ex = ln c_H^ex + em·z_H·φ^ex`.
   - All bulk Dirichlets met automatically; side walls auto-zero-flux.
2. Four source terms to inject into F_res:
   - NP interior: `S_c_i = −∇·J_i^ex`.
   - NP electrode: `g_i^elec = J_i^ex·n − Σ_j s_{ij}·R_j^ex`.
   - Poisson interior: `S_φ = −ε_c ∇²φ^ex − ρ_c(z_H c_H^ex + Σ_k z_k c_k^ster,ex)`.
   - Stern Robin: `g_S = ε_c (∇φ^ex·n) − C_S^model(φ_app^model − φ^ex)`.
3. Pinned K0_R4e_factor = **1e−18** (not 1; full strength gives R4e/R2e ≈ e^46 ≈ 1e20).
4. Pinned V_RHE = +0.55 V (demo anchor).
5. Source builder composes its own UFL for the multi-ion shared-θ closure
   **independently** from `boltzmann.py:91–268`; does NOT consume
   production-side `steric_boltz` bundles.

### Critical things GPT should pressure-test

1. **Dependency ordering** — does step N truly require step N−1 to be
   finished? Are there opportunities for parallel execution / earlier
   shake-out?
2. **Factory misconfiguration** — does `make_sp_production_muh` correctly
   omit the demo's two-stage Stern anchor while still passing the
   stack-invariant asserts at step 6?
3. **Source-builder logic** — every detail of step 6: does the boundary
   normal sign convention match, does the proton case use `μ_H^ex` (not
   the reconstruction `u_H_recon_ex`) for the muh primary, does
   `fd.derivative` after F_res mutation give the right Jacobian?
4. **Pilot validation completeness** — step 10's four checks. Are any
   missing? (I'm explicitly thinking about: does the quadrature sweep
   need a separate side-by-side comparison against the existing logc
   MMS to confirm we're not regressing? Should the pilot include a
   "deliberately broken config" run to verify the stack-asserts catch
   them?)
5. **Threshold choices** — L2 slope ≥ 1.8, H1 slope ≥ 0.8, R² > 0.99.
   Existing MMS uses identical tolerances (rate_tol = 0.2 on expected
   2.0 / 1.0); is that appropriate for the muh+multi-ion+Stern stack or
   should I tighten/loosen?
6. **Error norm conventions** — for the proton, comparing
   `U.sub(h_idx)` against `μ_H^ex` (not `u_H^ex`). Have I got the
   semantics right? In particular: the L2/H1 error of μ_H tells us
   about the error in the *primary unknown*; should I additionally
   report the error in `c_H = exp(μ_H − em·z_H·φ)` so we can compare
   against the existing logc MMS's c_i_L2 reporting?
7. **Risk mitigation** — am I missing any risks in §5? What about:
   adjoint-mode work that depends on F_res being functional-of-U
   (the source terms are constants in U, so this should be fine — but
   call it out); test running in `pytest-xdist`-style parallel mode
   stomping on the shared cache dirs (`/tmp/firedrake-tsfc`).
8. **Acceptance criteria sufficiency** — is criterion 5 ("source builder
   asserts catch all §7 invariants on a deliberately broken config")
   robust enough? Should we write a separate `TestMMSAsserts` class
   that parametrizes broken configs?

Be aggressive about missing pre-flight checks, latent ordering bugs,
and anything that would only surface during the pilot and waste time.

## Section 2: Artifact under review

# Implementation Plan — MMS for `logc_muh` + multi-ion + Stern stack

**Status:** plan only; not yet executed.
**Backs:** `docs/solver/mms_pnpbv_muh_multi_ion_stern_derivation.md` (post-GPT-critique session 38, verdict APPROVED).
**Owner:** PNPInverse / forward-solver verification.
**Date:** 2026-05-14.

## 0. Scope & non-goals

Build a convergence-rate MMS test that verifies the production PNP–BV stack
used by `scripts/studies/solver_demo_slide15_no_speculative_cs.py`:
`logc_muh` formulation, 3 dynamic species (O₂, H₂O₂, H⁺) with physical
hard-sphere `a_nondim`, Cs⁺/SO₄²⁻ multi-ion shared-θ Bikerman closure,
Stern Robin BC at the electrode, parallel R2e + R4e log-rate BV. Two test
classes per the derivation §7:

- `TestMMSConvergence`: UnitSquareMesh N ∈ {8, 16, 32, 64}; assert L2 slope
  ≥ 1.8, H1 slope ≥ 0.8, R² > 0.99 per primary unknown
  (u_O2, u_H2O2, μ_H, φ).
- `TestMMSProductionGradedMesh`: single-solve recovery on (Nx=8, Ny=80, β=3).

Non-goals (deferred): Stern-off cross-check, K0_R4e_factor=1 secondary
sweep, clip-activation MMS, saturation-active MMS, time-dependent MMS,
adjoint Taylor-test, c_ref-anchored anodic branch coverage. See derivation
§8.

## 1. Files

```
scripts/verification/mms_pnpbv_muh_multi_ion_stern.py   NEW  (~700 LOC)
tests/test_mms_logc_muh_multi_ion_stern.py              NEW  (~250 LOC)
```

No edits to production code. Existing `conftest.py::skip_without_firedrake`
is reused.

## 2. Top-level constants and config

In `scripts/verification/mms_pnpbv_muh_multi_ion_stern.py`:

```python
V_RHE_TEST = 0.55                       # V vs RHE — demo anchor
DT_LARGE   = 1.0e15
T_END_LARGE = 1.0e15
K0_R4E_FACTOR_MMS = 1.0e-18             # derivation §5.5; bounded R4e/R2e
STERN_C_S_F_M2  = 0.20                  # production target; NO two-stage anchor
DELTA_PERTURB   = (0.30, 0.30, 0.30)    # (O2, H2O2, H+) — matches existing MMS
ALPHA0          = 0.5
ALPHA1          = 0.5
GAMMA           = 0.5
SRC_QUAD_DEGREE_INITIAL = 8             # pinned post-sweep
MESH_SIZES   = (8, 16, 32, 64)
```

Plus reaction configs (R2e + R4e) reused from the demo's `_make_sp`.

## 3. Implementation steps (dependency order)

### Step 1 — Scaffolding

Create the two new files with module-level imports, header docstrings,
and `setup_firedrake_env()` boilerplate matching the existing MMS pattern.

`tests/test_mms_logc_muh_multi_ion_stern.py` skeleton:

```python
@skip_without_firedrake
@pytest.mark.slow
class TestMMSConvergence: ...

@skip_without_firedrake
@pytest.mark.slow
class TestMMSProductionGradedMesh: ...
```

**Verification:** `pytest --collect-only tests/test_mms_logc_muh_multi_ion_stern.py`
discovers two classes, no collection errors.

### Step 2 — Factory `make_sp_production_muh`

Mirror `scripts/studies/solver_demo_slide15_no_speculative_cs.py:_make_sp`
with these MMS-specific differences:
- `eta_hat = V_RHE_TEST / V_T` (fixed; not sweep).
- `dt = DT_LARGE, t_end = T_END_LARGE` (time term negligible).
- `K0_R4e_factor = K0_R4E_FACTOR_MMS` (1e−18).
- `stern_capacitance_f_m2 = STERN_C_S_F_M2` direct (no two-stage anchor;
  source builder will provide the manufactured initial guess so anchor
  numerics aren't needed).
- SNES: `atol=1e-5, rtol=1e-8, stol=1e-12, max_it=80`,
  `snes_linesearch_type="l2"`, `snes_linesearch_maxlambda=0.3`. (Mirrors
  existing logc MMS at `mms_bv_3sp_logc_boltzmann.py:139–148`.)
- `formulation="logc_muh"`, `log_rate=True`, `initializer="debye_boltzmann"`
  (initializer choice irrelevant for MMS since we overwrite with U_manuf
  before solve, but the factory requires a valid value).
- `multi_ion_enabled=True`, `boltzmann_counterions=[Cs⁺, SO₄²⁻]` from
  `DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,
  DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC`.
- `species` = SpeciesConfig with physical hard-sphere a_nondim from the
  demo (`A_O2_PHYSICAL`, `A_H2O2_PHYSICAL`, `A_HP_PHYSICAL`).

Signature:
```python
def make_sp_production_muh(*, mesh_Nx: int | None = None) -> SolverParams:
    ...
```

**Verification:** smoke-call `make_sp_production_muh()`, inspect the
returned tuple — formulation key reads `"logc_muh"`, `bv_reactions` has
2 entries with the right E_eq and α, `bv_stern_capacitance_model > 0`
in the converted scaling.

### Step 3 — Manufactured field UFL builders

`_make_manufactured_fields(mesh: fd.Mesh, ctx: dict, sp) -> dict`

Returns:
```
{
  "c_ex":           [c_O2_ex,  c_H2O2_ex,  c_H_ex],         # UFL
  "u_ex":           [u_O2_ex,  u_H2O2_ex,  u_H_recon_ex],   # UFL; proton is recon = ln c_H
  "mu_H_ex":        mu_H_ex,                                # UFL
  "phi_ex":         phi_ex,                                 # UFL
  "phi_app_model":  float,                                  # for hand checks
}
```

Field shapes per derivation §3.1:

```python
x, y = fd.SpatialCoordinate(mesh)
c_ex = [c0_i * (1 + δ_i * fd.cos(π·x) * (1-y)²) for i in {O2, H2O2, H}]
phi_ex = (1-y)·[α0 + α1·cos(π·x)] + γ·y·(1-y)·cos(π·x)
mu_H_ex = fd.ln(c_H_ex) + Constant(em·z_H) · phi_ex
u_ex[h_idx] = mu_H_ex − Constant(em·z_H) · phi_ex     # = ln c_H_ex by construction
```

`c0_i, em, z_H` read off `ctx["nondim"]`.

### Step 4 — Independent shared-θ closure helper

`_build_shared_theta_closure_ex(*, counterions_cfg, a_dyn, c0_dyn,
phi_ex, c_dyn_ex) -> dict`

Composes its **own** UFL for the multi-ion closure, mirroring
`Forward/bv_solver/boltzmann.py:91–268` algebraically but **without
consuming the production-side `steric_boltz` bundle objects** (derivation
§7 Independence policy).

Returns:
```
{
  "q_k_ex":        [q_k_ex per ion],
  "D_ex":          D_ex,                       # shared denominator
  "c_steric_ex":   [c_k_steric_ex per ion],
  "P_k_ex":        [a_k · c_k_steric_ex per ion],
  "rho_k_ex":      [z_k · c_k_steric_ex per ion],
  "theta_b":       float (bulk packing constant),
  "A_dyn_ex":      Σ a_i · c_i_ex,
  "theta_inner_ex": 1 − A_dyn_ex − Σ P_k_ex,
  "mu_steric_ex":  -ln(theta_inner_ex),
}
```

Reads counterions from `_get_bv_boltzmann_counterions_cfg(sp[10])`.

**Verification:** sanity-print θ_b ≈ 0.991 against the derivation §5.4
hand-computed value; sanity-print c_Cs_steric_ex evaluated at a few mesh
points and confirm magnitude ~ c_Cs^bulk · q ~ O(100).

### Step 5 — muh-aware BV rate helper

`_build_bv_rates_ex(*, reactions_cfg, scaling, u_ex, phi_ex) ->
list[ufl.Expr]`

Builds one UFL R_j_ex per reaction following derivation §3.3:
- η_j_ex = bv_exp_scale · (φ_app_model − φ_ex − E_eq_j_model)
- log_cath: `ln(k0_j) + u_ex[cat_j] + Σ_f power·(u_ex[sp_f] − ln c_ref_f) − α·n_e·η`
- log_anod: by branch (reversible+anodic / reversible+c_ref / irreversible)
- R_j = exp(log_cath) − exp(log_anod)

`u_ex[h_idx]` is **already** the muh-reconstructed `μ_H − em·z_H·φ` per
step 3, so no extra substitution here.

**Verification:** evaluate R_R4e_ex and R_R2e_ex at (x=0.5, y=0)
assembled-norm in (10, 1e5) ratio (the §5.5 invariant). If outside,
flag and stop.

### Step 6 — Source-injection routine `_build_manufactured_source`

```python
def _build_manufactured_source(
    ctx: dict, sp,
    *,
    manuf: dict,           # output of _make_manufactured_fields
    closure: dict,         # output of _build_shared_theta_closure_ex
    rxn_rates: list,       # output of _build_bv_rates_ex
    quad_degree: int = SRC_QUAD_DEGREE_INITIAL,
) -> dict:
```

Mutates `ctx['F_res']` in place by appending the four sources per
derivation §4:

1. **NP interior** (per species i): `S_c_i = −∇·J_i_ex`;
   `F_res -= S_c_i · v_i · dx_q`.
   - For non-mu species: `J_i_ex = D_i · c_i_ex · ∇(u_i_ex + μ_steric_ex)`.
   - For proton: `J_H_ex = D_H · c_H_ex · ∇(μ_H_ex + μ_steric_ex)`.
2. **NP electrode boundary** (per species i):
   `g_i_elec = J_i_ex·n − Σ_j s_{ij}·R_j_ex`;
   `F_res -= g_i_elec · v_i · ds_q(elec)`.
   Use `fd.dot(J_i_ex, fd.FacetNormal(mesh))` for the normal projection.
3. **Poisson interior**:
   `S_phi = −ε_c·∇²φ_ex − ρ_c·z_H·c_H_ex − ρ_c·Σ_k z_k·c_k_steric_ex`;
   `F_res -= S_phi · w · dx_q`.
4. **Stern Robin boundary**:
   `g_S = ε_c·(∇φ_ex·n) − C_S_model·(φ_app_model − φ_ex)`;
   `F_res -= g_S · w · ds_q(elec)`.

All on `dx_q = fd.dx(domain=mesh, degree=quad_degree)` and
`ds_q = fd.ds(domain=mesh, degree=quad_degree)`.

Re-derive Jacobian: `ctx["J_form"] = fd.derivative(ctx["F_res"], U)`.

### Step 7 — Stack-invariant asserts + runtime invariants

Embed the §7 derivation asserts as a private `_assert_stack_invariants(
ctx, sp, closure, rxn_rates)` called at the top of step 6. Implements:
- `len(scaling['bv_reactions']) == 2`
- `conv_cfg['bv_log_rate'] is True`
- `scaling['bv_stern_capacitance_model'] > 0`
- `not nondim_cfg['suppress_poisson_source']`
- `not is_water_ionization_enabled(conv_cfg)`
- `not is_cation_hydrolysis_enabled(conv_cfg)`
- `ctx['n_species'] == 3`
- exactly 2 bikerman counterions
- `ctx['mixed_space_indices'].gamma_index is None`
- `scaling['dt_model'] >= 1e12`
- `snes_atol <= 1e-5, snes_rtol <= 1e-7`

Plus the **field-dependent runtime invariants** computed against
`manuf, closure, rxn_rates`:
- θ_inner_ex discrete-min indicator: `assemble(conditional(θ < 1e-7, 1, 0)*dx) < 1e-12 · vol`
- R_ratio in finite window: `10 < ||R_R4e_ex||_L2(ds_e) / ||R_R2e_ex||_L2(ds_e) < 1e5`

Failure raises `AssertionError` with the offending value in the message
(so a green-light run never needs human inspection, but a failure points
straight at the cause).

### Step 8 — Outer `run_mms` loop

```python
def run_mms(
    N_list: Sequence[int] = MESH_SIZES,
    *,
    quad_degree: int = SRC_QUAD_DEGREE_INITIAL,
    verbose: bool = True,
) -> dict:
```

Per mesh:
1. `mesh = fd.UnitSquareMesh(N, N)`.
2. `sp = make_sp_production_muh()`; ensure `sp.solver_options["bv_convergence"]["exponent_clip"]`
   is the canonical 100.0 (Hard Rule #2).
3. `ctx = build_context(sp, mesh=mesh); ctx = build_forms(ctx, sp)`.
4. `manuf = _make_manufactured_fields(mesh, ctx, sp)`.
5. `closure = _build_shared_theta_closure_ex(...)` reading counterions from sp.
6. `rxn_rates = _build_bv_rates_ex(...)`.
7. `ctx = _build_manufactured_source(ctx, sp, manuf=manuf, closure=closure,
   rxn_rates=rxn_rates, quad_degree=quad_degree)`.
8. Build `U_manuf`: interpolate `u_O2_ex, u_H2O2_ex, μ_H_ex, φ_ex` onto
   each `W.sub(...)`. **For the proton, interpolate μ_H_ex, NOT u_H_recon_ex.**
9. `U.assign(U_manuf); U_prev.assign(U_manuf)`.
10. Solve via `fd.NonlinearVariationalSolver` with `snes_params` from sp.
11. Compute L2 / H1 errors per primary unknown:
    - O2: `_ufl_l2_error(u_O2_ex, U.sub(O2_idx), mesh, degree=quad_degree)`
    - H2O2: same with u_H2O2_ex
    - **H+: `_ufl_l2_error(μ_H_ex, U.sub(h_idx), mesh, degree=quad_degree)`**
    - φ: same with phi_ex

Returns `{"N": [...], "h": [...], "u_O2_L2": [...], "u_O2_H1": [...], ...,
"mu_H_L2": [...], "mu_H_H1": [...], "phi_L2": [...], "phi_H1": [...]}`.

### Step 9 — `verify_on_graded_production_mesh` (single solve)

Mirrors the existing logc MMS's `verify_on_graded_production_mesh` but
for the new stack. Uses the graded rectangle Nx=8, Ny=80, β=3 from the
demo. Returns the same dict shape as `run_mms` for a single mesh.

**Newton-converged check** is the load-bearing one here; L2/H1
thresholds are 6× above the empirically-measured baseline.

### Step 10 — Pilot validation (BEFORE adding pytest assertions)

Run-once shake-out before the convergence tests can land:

1. **Quadrature sweep at N=32**: run `run_mms([32], quad_degree=d)` for
   `d ∈ {6, 8, 10, 12, 16}`. Tabulate per-field L2 errors. Pin
   `SRC_QUAD_DEGREE` to smallest `d` where each field's L2 is within
   1% of the `d=16` value. Document the table in the test file's
   class docstring.
2. **Newton-convergence sanity at N=8**: `run_mms([8])` must report
   `newton_converged=True`. If not, debug source-builder bug before
   widening.
3. **R_ratio check at N=32**: confirm `||R_R4e_ex||_L2 / ||R_R2e_ex||_L2`
   lies in `(10, 1e5)` at the chosen K0_R4e_factor; if not, retune
   K0_R4e_factor.
4. **θ_inner_ex discrete-min**: confirm
   `min over mesh nodes of theta_inner_ex > 1e-7` at the chosen envelope;
   if not, tighten (α0, α1, γ) — should pass at 0.5/0.5/0.5.

Save pilot artifacts to
`StudyResults/mms_logc_muh_multi_ion_stern/pilot/` (JSON + plots) and
reference them in the test file docstring.

### Step 11 — `TestMMSConvergence`

```python
class TestMMSConvergence:
    MESH_SIZES = (8, 16, 32, 64)
    EXPECTED_L2_RATE = 2.0
    EXPECTED_H1_RATE = 1.0
    RATE_TOL = 0.2          # accept slope >= 1.8 (L2) and >= 0.8 (H1)
    MIN_R_SQUARED = 0.99

    @pytest.fixture(scope="class")
    def mms_results(self):
        return run_mms(self.MESH_SIZES, verbose=True)

    def test_newton_converges_on_all_meshes(self, mms_results): ...
    def test_l2_convergence_rates(self, mms_results): ...   # 4 fields × scipy.linregress
    def test_h1_convergence_rates(self, mms_results): ...
    def test_gci_output(self, mms_results): ...             # Roache, Fs=1.25
    def test_save_convergence_artifacts(self, mms_results): ...  # JSON + PNG
```

`test_l2_convergence_rates` iterates over fields
`["u_O2", "u_H2O2", "mu_H", "phi"]` using `assert_convergence_rate`
imported from the existing `tests/test_mms_convergence.py` (already
written; reuse).

Artifacts land at
`StudyResults/mms_logc_muh_multi_ion_stern/convergence_data.json` and
`mms_convergence.png`.

### Step 12 — `TestMMSProductionGradedMesh`

```python
class TestMMSProductionGradedMesh:
    L2_THRESHOLD = ?       # set after pilot; 6× empirically-observed
    H1_THRESHOLD = ?       # set after pilot

    @pytest.fixture(scope="class")
    def graded_mesh_results(self):
        return verify_on_graded_production_mesh(verbose=True)

    def test_newton_converges(self, graded_mesh_results): ...
    def test_l2_recovery(self, graded_mesh_results): ...
    def test_h1_recovery(self, graded_mesh_results): ...
```

Thresholds pinned post-pilot (step 10 output).

### Step 13 — CI integration

- Mark both classes `@pytest.mark.slow`.
- Document running invocation in the test file docstring:
  ```
  source ../venv-firedrake/bin/activate
  pytest tests/test_mms_logc_muh_multi_ion_stern.py -m slow -s -vv --log-cli-level=INFO
  ```
- No CI config change needed (slow tests already excluded by default).
- Plumb output artifacts under `StudyResults/mms_logc_muh_multi_ion_stern/`.

## 4. Dependency graph

```
Step 1 (scaffolding)
  └─→ Step 2 (factory)
        └─→ Step 3 (manuf fields)
              ├─→ Step 4 (closure helper)
              │     └─→ Step 6 (source injection)
              └─→ Step 5 (BV rates helper)
                    └─→ Step 6 (source injection)
                          └─→ Step 7 (asserts/invariants — embedded in step 6)
                                └─→ Step 8 (run_mms loop)
                                      ├─→ Step 9 (graded mesh)
                                      └─→ Step 10 (pilot validation)
                                            ├─→ Step 11 (TestMMSConvergence)
                                            └─→ Step 12 (TestMMSProductionGradedMesh)
                                                  └─→ Step 13 (CI integration)
```

## 5. Risks and unknowns

| Risk | Mitigation | Pre-pilot evidence required |
|---|---|---|
| Newton fails to converge at N=8 because manufactured source overwhelms RHS | Pilot step 10.2 must pass before proceeding | Newton convergence + `||F_res||_final < snes_atol` |
| Convergence rate falls below 1.8 (L2) because quadrature error dominates | Quadrature sweep (step 10.1) pins degree where error plateaus | Rate stable across degree 8 → 12 → 16 |
| R4e/R2e outside (10, 1e5) at chosen K0_R4e_factor | Step 10.3 retunes factor; alternatives 1e-16 or 1e-20 | Single L2-norm computation per ds_e |
| θ_inner_ex hits packing_floor on some mesh | Step 10.4; tighten envelope if needed | Discrete-min via indicator |
| Stern setter or runtime mutation accidentally fires between source build and solve | `_assert_stack_invariants` checks; no setters called in `run_mms` body | Code review of the new module |
| Existing `_ufl_l2_error` / `_ufl_h1_error` quadrature degree is too low for μ_H_ex (because of em·z_H·φ_ex coupling) | Pass `degree=quad_degree` explicitly per call site | Step 10.1 catches if quadrature is the bottleneck |
| Production residual changes between MMS authoring and merge (Phase 6β churn) | Tie MMS to specific Forward/bv_solver commit; rerun pilot on update | git pin in test file docstring |
| Test wall time blows up | Mark slow; cap N≤64 for convergence sweep | Empirical timing in pilot |

## 6. Empirical pilot artifacts to produce

Output to `StudyResults/mms_logc_muh_multi_ion_stern/pilot/`:

- `quadrature_sweep.json`: per-degree L2 / H1 / Newton-iteration counts.
- `quadrature_sweep.png`: log-log L2 vs degree per field.
- `n8_newton_convergence.json`: Newton history, residual norm trace.
- `r_ratio_check.json`: R_R4e_norm, R_R2e_norm, ratio.
- `theta_min_check.json`: discrete min, indicator-measure, sample plot
  of θ_inner_ex on N=32 mesh.

These set the per-field thresholds in step 12 and feed into the
convergence-test docstring.

## 7. Acceptance criteria

The implementation is complete when:

1. `pytest tests/test_mms_logc_muh_multi_ion_stern.py -m slow` passes end
   to end on the venv-firedrake environment.
2. All four primary unknowns achieve L2 slope ≥ 1.8 and H1 slope ≥ 0.8
   with R² > 0.99 in the N ∈ {8, 16, 32, 64} sweep.
3. `verify_on_graded_production_mesh` converges Newton in fewer than 30
   iterations on the production graded rectangle and meets the
   empirically-pinned thresholds.
4. JSON + PNG artifacts land in
   `StudyResults/mms_logc_muh_multi_ion_stern/`.
5. Source builder asserts catch all the §7 invariants on a deliberately
   broken config (sanity test, optional but recommended).
6. Test class docstrings reference the quadrature sweep and pilot results.
7. The derivation doc
   `docs/solver/mms_pnpbv_muh_multi_ion_stern_derivation.md` and the
   handoff `docs/handoffs/CHATGPT_HANDOFF_38_.../FINAL_REVISION.md` are
   linked from the test file.

## 8. Estimated effort

- Steps 1–2: 1 h.
- Steps 3–5: 4 h (manufactured fields + closure helper + BV helper).
- Step 6: 4 h (term-by-term source injection, careful with `fd.dot` /
  `fd.FacetNormal` / quadrature).
- Step 7: 2 h (asserts + runtime invariants).
- Steps 8–9: 3 h (run_mms loop + graded-mesh single solve).
- Step 10 (pilot, includes debugging): 4–8 h.
- Steps 11–13: 3 h.

Total: 21–25 hours of focused work. Half-budget for the pilot is
deliberate — that's where unknown unknowns surface.

## 9. Open questions to resolve during implementation

1. Should the source builder accept `quad_degree` as a per-mesh argument
   (to support the quadrature sweep) or as a module-level constant?
   Recommendation: kwarg with sensible default, so the sweep is a clean
   call.
2. For the graded mesh in step 9, does the manufactured solution need
   re-design (the existing logc MMS reuses the `(x,y) ∈ [0,1]²` shape
   on a graded rectangle)? Expected answer: same shape works because
   the rectangle is mapped 1-to-1 to the unit square. Verify in pilot.
3. Should we emit per-mesh Newton-iteration plots to catch
   non-monotonic convergence? Yes if pilot shows N=64 hits the iteration
   cap.
4. Is `boltzmann.py:91-268` stable enough that mirroring its algebra in
   step 4 is safe? Pin the test to a specific commit via
   git-log-reference in the test docstring; revisit if `boltzmann.py`
   undergoes a structural change.

---

This plan is ready for adversarial GPT review. Loop will pressure-test
dependency ordering, factory misconfiguration risk, source-builder
correctness, pilot design, threshold choices, and any missing setup
steps.


---

## Section 3: Critique prompt

You are an adversarial reviewer. Be critical. Be argumentative.
Find every hole: missing steps, wrong algebra, untested assumptions,
edge cases not addressed, implicit dependencies, claims without
evidence, off-by-one errors, sign errors, dimensional errors. Don't
be polite — if something is wrong, say so. Concision over hedging.

For each issue, state:
  - WHAT is wrong (specific, not vague — name the line or symbol)
  - WHY it matters (what breaks downstream if uncorrected)
  - WHAT to do (concrete fix, or what evidence would close the gap)

Number your issues. After all issues, end your response with exactly
one of these lines, no other text after it:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN

Use APPROVED only when there are no issues you would block on.
Minor nitpicks alone do not justify ISSUES_REMAIN — call them out
but still verdict APPROVED. Use ISSUES_REMAIN whenever any of your
issues are genuinely blocking.
