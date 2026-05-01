# Codebase Evidence: Three Numerical Changes to PNP-BV Forward Solver

This is a verification report. Every assertion below is grounded in `file:line`
references to the live tree at
`/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse`.

---

## 1. Change #1 — 4-species PNP → 3-species PNP + analytic Boltzmann counterion

### 1.1 Where the Boltzmann counterion lives

The writeup says: "Operationally this is implemented through the Boltzmann
contribution to the Poisson residual; see `add_boltzmann()` in the current
study scripts."

**Verdict:** Correct that it lives in study scripts, **not** in
`Forward/bv_solver/forms_logc.py`. The function `add_boltzmann()` is duplicated
across multiple study scripts (a "monkey-patch" pattern), defined locally each
time and applied right after `build_forms_logc()`. Confirmed call sites:

| File | `add_boltzmann` def | Applied at |
|---|---|---|
| `scripts/studies/v18_logc_lsq_inverse.py` | line 234 | line 259 |
| `scripts/studies/v24_3sp_logc_vs_4sp_validation.py` | line 250 | line 274 |
| `scripts/studies/v18_logc_diagnostics.py` | line 117 | line 142 |
| `scripts/studies/v18_logc_noise_sensitivity.py` | line 85 | line 107 |
| `scripts/studies/v19_lograte_extended_adjoint_check.py` | line 166 | line 192 |
| `scripts/plot_iv_curves_3sp_true.py` | line 108 | line 134 |
| (4-sp variant) `scripts/studies/v18_test_3species_boltzmann.py` | line 109 | line 158 |

`Forward/bv_solver/forms_logc.py` itself contains no Boltzmann term — its
Poisson residual is the bare three-explicit-species sum at line 418:

> `F_res -= charge_rhs * sum(z[i] * ci[i] * w for i in range(n)) * dx`

The Boltzmann ClO4⁻ contribution is *added on top of* this residual by the
study scripts after `build_forms_logc` returns.

### 1.2 Exact formula and sign convention

Reference implementation in `scripts/studies/v24_3sp_logc_vs_4sp_validation.py:250-262`:

```python
def add_boltzmann(ctx):
    scaling = ctx["nondim"]
    W = ctx["W"]; U = ctx["U"]; mesh_ = ctx["mesh"]
    phi = fd.split(U)[-1]
    w = fd.TestFunctions(W)[-1]
    dx = fd.Measure("dx", domain=mesh_)
    charge_rhs = fd.Constant(float(scaling["charge_rhs_prefactor"]))
    c_bulk = fd.Constant(C_CLO4_HAT)
    phi_cl = fd.min_value(fd.max_value(phi, fd.Constant(-50.0)),
                          fd.Constant(50.0))
    ctx["F_res"] -= charge_rhs * fd.Constant(-1.0) * c_bulk * fd.exp(phi_cl) * w * dx
    ctx["J_form"] = fd.derivative(ctx["F_res"], U)
    return ctx
```

The 4-species reference (`scripts/studies/v18_test_3species_boltzmann.py:109-147`)
has the sign convention spelled out as a comment block:
- z_ClO4 = -1 (line 132 comment: `c = c_bulk * exp(-z*phi) = c_bulk * exp(phi) for z=-1`)
- Term added: `−charge_rhs · z_ClO4 · c_bulk · exp(φ) · w dx`
- With z_ClO4 = -1, the literal subtracted term is
  `charge_rhs · (−1) · c_bulk · exp(φ) · w dx`, i.e. the term added to the
  residual is `+charge_rhs · c_bulk · exp(φ) · w dx`.

This matches the writeup's claim: c_ClO4 = c_bulk · exp(−z_ClO4 · φ / V_T)
with z_ClO4 = -1 (so the sign in the exponent is positive in φ).

The φ clamp at ±50 (in dimensionless units, i.e. ±50·V_T ≈ ±1.28 V) is
described in the comments themselves as "Clip phi to prevent exp overflow"
(`v18_test_3species_boltzmann.py:133-134`) — supports the writeup's "overflow
protection rather than concentration floors" framing.

### 1.3 Why this is a *reduction*, not just a refactor

`scripts/studies/v24_3sp_logc_vs_4sp_validation.py:23-26` annotates the change:

> "drop ClO4- as a dynamic species, add an analytic Boltzmann factor in
> Poisson"

and `THREE_SPECIES_Z = [0, 0, 1]` at line 207 — i.e. only H⁺ is charged in the
3-species model; O₂ and H₂O₂ are neutral; ClO4⁻ is gone from the
Nernst–Planck system entirely.

---

## 2. Change #2 — concentrations → log-concentrations (u_i = ln c_i)

### 2.1 Function space and time scheme

`Forward/bv_solver/forms_logc.py:55-56`:

```python
V_scalar = fd.FunctionSpace(mesh, "CG", order)
W = fd.MixedFunctionSpace([V_scalar for _ in range(n_species)] + [V_scalar])
```

Same `V_scalar` for all `u_i` and for `phi`. CG `order` is whatever
`solver_params[1]` carries (production runs use order=1; e.g.
`v24_3sp_logc_vs_4sp_validation.py:246` passes `1`). Confirms the writeup's
"Firedrake CG monolithic" claim.

**Time scheme.** `forms_logc.py:283-285`:

```python
# Time-stepping residual: (c - c_old)/dt
F_res += ((c_i - c_old) / dt_const) * v * dx
F_res += fd.dot(Jflux, fd.grad(v)) * dx
```

This is **backward Euler** (single previous step, evaluated implicitly: the
flux uses current `c_i = exp(u_i)` and current `phi`). No BDF2 history is
maintained anywhere — `U_prev` is the only past state (see `build_context_logc`
at lines 57-58). The writeup's "backward Euler" statement holds.

### 2.2 The log-c residual

Header comment at `forms_logc.py:9-13`:

> The weak form (Nernst-Planck):
>   ∫ (exp(u) - exp(u_old))/dt · v dx + ∫ D·exp(u)·(∇u + z·∇φ)·∇v dx = BV terms
> Poisson:
>   ∫ ε·∇φ·∇w dx - ∫ charge_rhs·Σ z_i·exp(u_i)·w dx = 0

Implementation at `forms_logc.py:268-285`:
- `c_i = exp(u_i)` (with symmetric clamp, see §2.3)
- Drift: `drift = em * z[i] * phi` (line 275)
- Flux: `Jflux = D[i] * c_i * (fd.grad(u_i) + fd.grad(drift))` (line 281)
- Time term `(c_i - c_old) / dt_const * v * dx` (line 284)

The key log-transform identity ∇c = c·∇u is used on line 281 — this is the
"cleaner Jacobian structure" the docstring mentions at line 264-265.

### 2.3 The `_U_CLAMP` clamps — overflow protection, not floor

`forms_logc.py:185-198`:

```python
_U_CLAMP = float(conv_cfg.get("u_clamp", 30.0))
ci = [fd.exp(fd.min_value(fd.max_value(ui[i], fd.Constant(-_U_CLAMP)), fd.Constant(_U_CLAMP)))
      for i in range(n)]
```

Comment (lines 188-192) says explicitly:

> Default 30: exp(±30) covers [9.4e-14, 1.07e+13], adequate for typical EDL
> profiles. ... Note that the log-rate BV path uses ui[i] directly and bypasses
> this clamp on the boundary residual; the clamp here only affects bulk PDE
> terms (time derivative, diffusion).

This **supports** the writeup's claim that the clamps are "overflow protection
rather than concentration floors" — they are symmetric in `u`, applied to both
underflow (-30) and overflow (+30), and the log-rate boundary path bypasses
them entirely (see §3 below).

The sole asymmetric "floor" is the bulk Dirichlet BC for product species:
`forms_logc.py:431` sets `_C_FLOOR = 1e-20`, used at line 435-437 to avoid
`ln(0)` for H₂O₂ initial concentration. This is also overflow protection
(of `ln`), not a kinetic floor.

### 2.4 Bulk c_i never appears as a primary unknown

`forms_logc.py:177-178`:

```python
ui = fd.split(U)[:-1]      # log-concentrations
phi = fd.split(U)[-1]
```

Trial space stores `(u_1, ..., u_n, φ)`. There is no `c_i` DOF. Concentrations
are derived expressions (line 194-195, line 292) used only to *evaluate* the
nonlinear residual; the Newton solver iterates on `u_i`.

---

## 3. Change #3 — standard BV → log-rate BV

### 3.1 The toggle and line range

`bv_log_rate` toggle is parsed in `Forward/bv_solver/config.py:112` (default
False; true values produce the log-rate branch). The toggle is read in
`forms_logc.py:299`:

```python
bv_log_rate = bool(conv_cfg.get("bv_log_rate", False))
```

Writeup claim: "see `forms_logc.py:294-360`". **Verified in spirit, slightly
off in line range.** The branch starts at line 294 (with the docstring
comment) and the if-block runs **lines 299-389** (containing both the
log-rate true branch at 324-356 and the standard-BV false branch at 357-380,
followed by the species-loop residual subtraction at 385-388). Line 360 lands
midway in the standard-BV `else` branch, not at the end of the log-rate
section. The 294-360 window does cover the log-rate branch and the start of
the contrast standard-BV branch — adequate for the writeup's purpose, but the
exact endpoint is line 388 if "the log-rate machinery + its standard-BV
counterpart" is the intended scope.

### 3.2 The cathodic log-rate algebra

`forms_logc.py:324-339`:

```python
if bv_log_rate:
    # Cathodic: log_r = ln(k0) + u_cat + sum power*(u_sp - ln c_ref)
    #                   - alpha * n_e * eta_clipped
    log_cathodic = (
        fd.ln(k0_j) + ui[cat_idx]
        - alpha_j * n_e_j * eta_j
    )
    for factor in rxn.get("cathodic_conc_factors", []):
        sp_idx = factor["species"]
        power = fd.Constant(float(factor["power"]))
        c_ref_log = fd.ln(fd.Constant(
            max(float(factor["c_ref_nondim"]), 1e-12)
        ))
        log_cathodic = log_cathodic + power * (ui[sp_idx] - c_ref_log)
    cathodic = fd.exp(log_cathodic)
```

This **literally evaluates** ln r_cat = ln k_0 + u_cat + Σ_f ν_f (u_sp(f) − ln c_ref,f) − α n_e η/V_T, then exponentiates. Confirms the writeup's algebra exactly.

### 3.3 Anodic branch

`forms_logc.py:340-356`:

```python
if rxn["reversible"] and rxn["anodic_species"] is not None:
    anod_idx = rxn["anodic_species"]
    log_anodic = (
        fd.ln(k0_j) + ui[anod_idx]
        + (fd.Constant(1.0) - alpha_j) * n_e_j * eta_j
    )
    anodic = fd.exp(log_anodic)
elif rxn["reversible"] and float(rxn["c_ref_model"]) > 1e-30:
    c_ref_j_log = fd.ln(fd.Constant(float(rxn["c_ref_model"])))
    log_anodic = (
        fd.ln(k0_j) + c_ref_j_log
        + (fd.Constant(1.0) - alpha_j) * n_e_j * eta_j
    )
    anodic = fd.exp(log_anodic)
else:
    anodic = fd.Constant(0.0)
```

Two important asymmetries vs the cathodic branch:
1. The anodic log-rate has only the bare anodic-species term `u_anod` — there
   is **no** sum-over-stoichiometric-power loop on the anodic side. The
   writeup's claim of "both branches subtracted with sum_f ν_f (u_sp(f) − ln c_ref,f)"
   is partially supported: it holds for the cathodic branch but the anodic
   branch is single-species (matches one-electron Nernst-style anodic limit).
2. R2 in the production setup is irreversible (`v24_…validation.py:233`:
   `"reversible": False`), so its anodic branch falls into the `else` at
   line 355 → `anodic = 0`.

Net rate at line 382: `R_j = cathodic - anodic`, and the residual subtracts
this scaled by stoichiometry on the boundary at line 388:

```python
F_res -= fd.Constant(float(stoi[i])) * R_j * v_list[i] * ds(electrode_marker)
```

### 3.4 Why this fixes the "phantom R2 cathodic sink"

Documented in `docs/CHATGPT_HANDOFF_6_LOGRATE_BREAKTHROUGH.md:107-118`:

> Without log-rate, V=+0.40 cold-fails... The mechanism is the `_U_CLAMP=30`
> clamp on `u`: c_surf in BV uses `exp(clamp(u, ±30))`, so when Newton needs
> c_H2O2 below `exp(-30) ≈ 9.4e-14` during iteration, the clamp pins it.
> Combined with the saturated `exp(50)` in R2's BV exponent, the floor times
> the huge exp gives a spurious R2 sink that nothing else can balance — Newton
> stalls.
>
> Log-rate evaluates `exp(ln k0 + u_H2O2 + 2(u_H − ln c_ref) − α·n_e·η)`. The
> `u_H2O2` enters additively, so it can be arbitrarily negative and the whole
> expression decays smoothly to zero. No floor, no phantom sink.

The clamp-bypass is real and documented in the code at `forms_logc.py:294-298`:

> "This uses ui[i] (unclamped) instead of c_surf[i] (= exp(clamp(ui))),
> which removes the artificial R2 sink that the lower _U_CLAMP creates
> when c_H2O2 underflows during Newton iteration."

### 3.5 Two-reaction setup with E_eq and n_e

Production reaction config in `v24_3sp_logc_vs_4sp_validation.py:222-237` and
`v18_logc_lsq_inverse.py:198-228`:

- **R1 (ORR step 1):** O₂ + 2H⁺ + 2e⁻ → H₂O₂; `n_electrons=2`,
  `cathodic_species=O2`, `anodic_species=H2O2`, `stoichiometry=[-1, +1, -2]`
  (consumes O₂ and 2 H⁺, produces H₂O₂), `cathodic_conc_factors` adds
  power-2 in H⁺. `E_eq_v=0.68 V`.
- **R2 (peroxide reduction):** H₂O₂ + 2H⁺ + 2e⁻ → 2H₂O;
  `n_electrons=2`, `cathodic_species=H2O2`, `anodic_species=None`,
  `stoichiometry=[0, -1, -2]`, irreversible. `E_eq_v=1.78 V`.

Both reactions match the writeup's E_eq^(1)=0.68 V, E_eq^(2)=1.78 V, n_e=2,
plus the H⁺ stoichiometric power. Per-reaction E_eq is built into a separate
`eta_j` at `forms_logc.py:317-322`.

---

## 4. Continuation pipeline — `grid_charge_continuation.py`

### 4.1 Two phases: voltage sweep + charge ramp

`Forward/bv_solver/grid_charge_continuation.py:9-17`:

> **Phase 1** (neutral voltage sweep, z=0): full voltage grid, two-branch
> sweep, Lagrange predictor, bridge points, SER adaptive dt.
>
> **Phase 2** (per-point charge ramp, z: 0 → 1): aggressive-first adaptive
> z-ramp.

So the file does **both** voltage continuation (Phase 1) and charge
continuation (Phase 2). The homotopy parameter is `z` (a scalar applied
multiplicatively to each species' nominal valence: `zc[i].assign(z_nom[i] * z_val)`
at line 442). Phase 1 marches the voltage grid at z=0 (neutral); Phase 2 ramps
z 0→1 *per voltage point*.

### 4.2 Bisection + voltage substeps

The writeup's "voltage continuation with delta-V substeps" + "bisection
fallback" lives in **two places**:

1. **In `grid_charge_continuation.py`** for the cold neutral sweep — bridge
   points at `lines 351-381` insert sub-voltages when `gap > max_eta_gap`
   (default 3.0 dimensionless = ~0.077 V), and the `_bisect_eta` helper
   (lines 300-323) halves the step on SNES failure.

2. **In `v24_3sp_logc_vs_4sp_validation.py`** for the *3sp* warm-start
   continuation — `solve_warm_3sp_step` at lines 350-402 implements the
   `n_substeps` linear ramp from anchor V to target V, with recursive bisection
   `_march(...)` on failure (line 368-388). The substep count is the
   `--warm-substeps` flag (default 4, line 73-75), and bisection depth is
   `--bisect-depth` (default 3, line 76-78). Same idiom inside
   `v18_logc_lsq_inverse.py` driving the inverse solver.

So the writeup's "warm-start with bisection fallback" is correct and lives in
two contexts: the production grid sweep (`grid_charge_continuation.py`) and
the validation/inverse scripts that need to step *outward* from a converged
3sp anchor.

### 4.3 Charge ramp adaptive logic

`grid_charge_continuation.py:436-545` — `_adaptive_z_ramp`:
1. Try z=1 directly (line 466-469). If usable, done.
2. Else binary search for an initial foothold (lines 472-490) up to 4 levels.
3. Then geometric acceleration toward z=1 (lines 495-543) with midpoint
   bisection on failure.

This is explicitly **charge** continuation only at this layer. The voltage
piece is in Phase 1, which uses the same `_try_timestep` + `_bisect_eta`
machinery on `phi_applied_func` instead of `z_consts`.

---

## 5. Validation pipeline — eight-voltage table

### 5.1 Script and outputs

`scripts/studies/v24_3sp_logc_vs_4sp_validation.py` regenerates the writeup's
overlap-window comparison from scratch. Header comment at lines 5-9
explicitly references the writeup claim: "matched closely within the
validation tolerances used for the inverse work".

Outputs land at `StudyResults/v24_3sp_logc_vs_4sp_validation/`:
- `comparison.csv` (8 rows)
- `comparison.png`
- `summary.md`
- `raw_values.json`

### 5.2 Default voltage grid

`v24_…validation.py:62-67`:

```python
default=[-0.50, -0.40, -0.30, -0.20, -0.10, 0.00, 0.05, 0.10],
```

Eight V_RHE points spanning the documented overlap window. Confirms the
writeup's "eight-voltage validation table".

### 5.3 5% F2 threshold

`v24_…validation.py:80-82`:

```python
p.add_argument("--rel-tol-pct", type=float, default=5.0,
               help="Per-voltage PASS threshold as %% of max|observable|.")
```

And `v24_…validation.py:31`: "matches the F2 diffusion-limit tolerance used
elsewhere in the validation framework". The comparison.csv shows all 8 rows
PASS the 5% threshold.

### 5.4 Warm-start continuation strategy

`v24_…validation.py:411-461`:
- Phase 1: cold-start z-ramp at each V independently (line 412-426).
- Phase 2: warm-start *outward* from cold successes — cathodic walk from the
  lowest-success index marching toward index 0 (lines 437-461), then anodic
  walk (lines 464-488). Each warm step uses `solve_warm_3sp_step` with
  paf-substepping + recursive bisection.

In the actual `comparison.csv` the warm-start kicks in at V_RHE = -0.30,
-0.40, -0.50 (column `method_3sp = warm←-0.20/-0.30/-0.40`). The cold-start
range was V_RHE ∈ [-0.20, +0.10].

### 5.5 Numbers from the live comparison.csv

From `StudyResults/v24_3sp_logc_vs_4sp_validation/comparison.csv`:

| V_RHE | cd_4sp | cd_3sp | |Δcd|/cd_max % | pc_3sp | method | verdict |
|---:|---:|---:|---:|---:|---|---|
| -0.50 | -1.839e-1 | -1.841e-1 | 0.108 | -1.813e-1 | warm←-0.40 | PASS |
| -0.40 | -1.828e-1 | -1.831e-1 | 0.122 | -1.639e-1 | warm←-0.30 | PASS |
| -0.30 | -1.813e-1 | -1.817e-1 | 0.212 | -6.504e-2 | warm←-0.20 | PASS |
| -0.20 | -1.797e-1 | -1.802e-1 | 0.268 | -1.568e-3 | cold | PASS |
| -0.10 | -1.775e-1 | -1.780e-1 | 0.266 | 2.91e-6 | cold | PASS |
|  0.00 | -1.733e-1 | -1.738e-1 | 0.266 | 1.53e-5 | cold | PASS |
| +0.05 | -1.694e-1 | -1.699e-1 | 0.269 | 1.54e-5 | cold | PASS |
| +0.10 | -1.629e-1 | -1.631e-1 | 0.100 | 1.54e-5 | cold | PASS |

All 8 PASS at ≤0.27% error (well below the 5% F2 threshold). Confirms the
writeup's quantitative overlap claim with concrete persisted artefacts.

---

## 6. Pressure-tested writeup claims

| Claim | Verdict | Evidence |
|---|---|---|
| `add_boltzmann()` lives in current study scripts | TRUE | 6 active call sites listed in §1.1; not in `forms_logc.py` |
| `bv_log_rate=True` toggle in BV convergence config | TRUE | `config.py:112`, default False |
| Toggle implementation at `forms_logc.py:294-360` | MOSTLY TRUE | Block actually spans 294-389; 294-360 covers the log-rate branch but ends mid-`else` |
| Backward Euler time scheme | TRUE | Single `c_old` from `U_prev`; `(c_i - c_old) / dt` at `forms_logc.py:284` |
| BDF / multistep | FALSE | Only one history slot (`U_prev`), see `forms_logc.py:57-58, 284` |
| Firedrake CG monolithic, same order for u_i and φ | TRUE | `forms_logc.py:55-56`; `MixedFunctionSpace([V_scalar]*(n+1))` |
| Boltzmann uses c_ClO4 = c_bulk·exp(−z·φ/V_T), z=−1 | TRUE | `v18_test_3species_boltzmann.py:113-141`, `v24_…:260` |
| Clamps in log-c PDE are overflow protection, not floors | TRUE | Symmetric ±30 clamp at `forms_logc.py:193-198`; bypassed in log-rate boundary path |
| Cathodic algebra: ln r = ln k₀ + u_cat + Σν(u_sp − ln c_ref) − α n_e η/V_T | TRUE | `forms_logc.py:324-339` |
| Both anodic and cathodic carry full Σν power loop | FALSE | Cathodic has the loop (line 331-337); anodic is single-species (line 340-356) |
| 8-voltage validation table at `StudyResults/v24_…/comparison.csv` | TRUE | File exists, 8 rows, all PASS; confirmed inline in §5.5 |
| 5% F2 threshold | TRUE | `v24_…validation.py:80-82, 31` |
| Voltage continuation with δV substeps + bisection fallback | TRUE | `grid_charge_continuation.py:300-323, 351-381` (cold sweep); `v24_…validation.py:350-402` (3sp warm-step) |
| R2 unclips at V_RHE > +0.495 V (not +1.14 V) | TRUE | `forms_logc.py:216-235` arithmetic; `CHATGPT_HANDOFF_6:120-142` derivation |
| n_e=2 for both reactions, R1 E_eq=0.68 V, R2 E_eq=1.78 V | TRUE | `v24_…validation.py:100, 222-237`; `v18_logc_lsq_inverse.py:198-228` |
| R2 is irreversible in production | TRUE | `v24_…validation.py:233` — `"reversible": False` |
| Canonical PNP+BV doc has log-c / Boltzmann / log-rate | FALSE | `docs/PNP Equation Formulations.tex` uses c_i directly (lines 24-32), classical BV (lines 41-43, 73-81), no Boltzmann/spectator/ClO4 anywhere; all three changes are NEW vs canonical |

---

## 7. Key Takeaways

- **The three changes are layered and orthogonal.** Change 1 (Boltzmann) is a
  *modeling* reduction implemented as a script-level monkey-patch onto the
  Poisson residual. Change 2 (log-c) is a *primary-variable transform* that
  rewrites the entire weak form in `Forward/bv_solver/forms_logc.py`. Change 3
  (log-rate BV) is a *boundary-flux algebra* toggle inside the same file
  selected by `bv_log_rate`.

- **The canonical group doc (`docs/PNP Equation Formulations.tex`) contains
  none of the three changes.** It uses (c_i, φ) primary variables, classical
  exp(±α·n·F·η/RT) BV at `tex:41-43, 73-81`, and never mentions Boltzmann
  spectator counterions. All three changes are genuine numerical novelty
  relative to the group's own canonical derivation.

- **The "phantom R2 sink" pathology is real and well-described in the code +
  handoff documents.** It comes from the asymmetric interaction between the
  symmetric u-clamp (overflow protection) and the saturated R2 BV exponent.
  Log-rate BV cures it by evaluating the rate as `exp(linear-in-u)` so that
  underflowing surface H₂O₂ produces a smooth zero rate instead of a clamp×exp(50)
  spurious sink.

- **The writeup line-range pointer (`forms_logc.py:294-360`) is approximate.**
  The log-rate true branch is lines 324-356 and the contrast standard-BV branch
  is 357-380; 360 lands inside the standard-BV `else`. A more precise pointer
  would be `forms_logc.py:294-388`.

- **The Boltzmann counterion implementation is NOT in `Forward/bv_solver/`.**
  It is duplicated as a monkey-patch in every study script. If the writeup is
  to be rebuilt as a paper, it would be worth migrating `add_boltzmann` into
  a proper module (e.g. as an option in `forms_logc.build_forms_logc`) so
  there is one canonical implementation. The current pattern works but
  diffuses the literature claim across ~7 script copies.

- **The 8-voltage validation passes the 5% F2 threshold by an order of
  magnitude.** Worst case is 0.27% at V=+0.05 V (the cold-start regime).
  Cathodic warm-start points (-0.30 to -0.50 V) are even tighter at
  ~0.1-0.2%. This is a strong empirical defense of the Boltzmann + log-c
  reduction in the overlap window.

### Files referenced

- `Forward/bv_solver/forms_logc.py` (log-c weak forms + log-rate BV)
- `Forward/bv_solver/config.py` (`bv_log_rate` toggle parsing)
- `Forward/bv_solver/grid_charge_continuation.py` (Phase 1 voltage sweep + Phase 2 z-ramp)
- `scripts/studies/v18_logc_lsq_inverse.py` (production inverse, source of `add_boltzmann`)
- `scripts/studies/v24_3sp_logc_vs_4sp_validation.py` (regenerates the 8-voltage table)
- `scripts/studies/v18_test_3species_boltzmann.py` (reference 4sp variant of Boltzmann patch with full sign-convention comments)
- `docs/CHATGPT_HANDOFF_6_LOGRATE_BREAKTHROUGH.md` (phantom R2 sink narrative)
- `docs/CHATGPT_HANDOFF_7_LOGRATE_VALIDATION.md` (log-rate validation handoff)
- `docs/PNP Equation Formulations.tex` (canonical group derivation; none of the three changes appear)
- `StudyResults/v24_3sp_logc_vs_4sp_validation/comparison.csv` (8-voltage table data)
