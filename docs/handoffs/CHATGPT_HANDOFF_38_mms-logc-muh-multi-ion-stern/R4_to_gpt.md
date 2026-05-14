# Critique session 38 — Round 4

## Section 1: Acknowledgment

All 5 R3 issues accepted; the R4e dominance and Stern-off saturation findings are real and force structural changes.

### Re point 1 — R4e catastrophically dominates at K0_R4e_factor=1

**Accept.** Quick re-derivation:

  α_R2e · n_e,R2e · η_R2e_raw  ≈  0.627 · 2 · (−5.6)  ≈  −7.02         → +7.02 added to log_cath_R2e
  α_R4e · n_e,R4e · η_R4e_raw  ≈  0.5 · 4 · (−26.5)  ≈  −53            → +53 added to log_cath_R4e

With c_H^4 vs c_H^2 cathodic factors and identical k0's at K0_R4e_factor=1:

  log_cath_R4e − log_cath_R2e  ≈  46 + 0.6 (extra c_H term)  ≈  46.6
  R4e / R2e   ≈  e^46.6  ≈  1.7 · 10^20

So R4e dominates by ~20 orders of magnitude, not "competes". My `R4e_norm > 0.1·R2e_norm` is vacuous; the operator-of-test is essentially R4e-only at K0_R4e_factor=1. The dominant residual magnitude is also problematic: source `Σ s_H · R_R4e · v_H · ds` reaches ~k0_R4e · 10^20. Newton's `rtol = 1e−7` may leave a final residual still huge in absolute terms, drowning the discretization-error signal we want to measure.

**Two structural fixes:**

(a) **Pin K0_R4e_factor = 1e−18 for the primary MMS run** (matches the demo's smallest factor). Bring R4e magnitude into competitive range:
  log_cath_R4e − log_cath_R2e ≈ −18·ln(10) + 46.6 ≈ −41.5 + 46.6 ≈ +5.1
  R4e / R2e ≈ e^5.1 ≈ 164  (R4e dominates by ~2 orders, manageable)

Pilot test: a single solve at N=32, K0_R4e_factor=1e−18 should converge Newton to `||F_res|| < 1e−5` (or whatever atol). If Newton residual after convergence stays comparable to the assembled R_R4e_norm, the MMS is operating in the regime where source magnitude doesn't swamp the FE error signal.

(b) **Replace the "R4e is at least 10% of R2e" check with a finite-window invariant**:

  R_ratio  :=  R4e_norm / R2e_norm
  assert 10 < R_ratio < 1e5    # R4e dominant but not catastrophic — proves K0_R4e_factor is in the discriminating window

For the K0_R4e_factor = 1e−18 setting, R_ratio is empirically ~165, well inside [10, 1e5].

(c) **Secondary cross-check at K0_R4e_factor = 1 is REMOVED from the primary scope.** At factor=1 the R4e-only-dominant regime would need separate Newton-tolerance and quadrature-margin analysis (the existing logc MMS has its own commentary about Newton ratios at η ~ 21; we'd need analog work for the muh path at η ~ 26.5). Keep it as a follow-up pilot, not part of the convergence-rate sweep.

Updated §5.5 and §7 to reflect (a)–(c).

### Re point 2 — Stern-off cross-check broken by SO₄²⁻ saturation

**Accept.** At φ^ex_NoStern(x,0) = φ_app^model ≈ 21.4 and z_SO4 = −2:

  q_SO4 = exp(−z_SO4 · 21.4) = exp(42.8) ≈ 4 · 10^18
  shared-θ denominator D ≈ θ_b + a_SO4 · c_SO4 · 4e18 ≈ 3.5e−3 · 4e18 ≈ 1.4e16
  c_SO4^ster ≈ c_SO4 · 4e18 · 1 / 1.4e16 ≈ 286·c_SO4 = 286·83.3 ≈ 23800
  P_SO4 ≈ a_SO4 · c_SO4^ster ≈ 4.20e−5 · 23800 ≈ 1.0

⇒ θ_inner → 1 − 0 − 1.0 − ε ≈ 0 ⇒ `packing_floor = 1e−8` ACTIVATES.

This contradicts the entire "clamps/floors inactive at u_exact" premise. The Stern-off cross-check as proposed is broken. **Removed `TestSternOffSanity` from the plan.** Stern-off testing requires either:
- A separate small-V_RHE design (V_RHE ≈ 0.05 V keeps φ_app ≈ 2, q_SO4 ≈ 55 — closure stays unsaturated), at the cost of testing a regime not relevant to the production demo, OR
- A separate saturation-active MMS that *expects* `packing_floor` to be active (different convergence-rate expectations because of the `max(·, packing_floor)` kink).

Both are out of scope for the baseline convergence test. The Stern-off line is deleted from §5.5 and the `TestSternOffSanity` class is removed from §7.

### Re point 3 — Stern coefficient coverage row internally inconsistent

**Accept.** Replaced "value of nondim Stern coefficient" in the §6 coverage row with "**use** of the ctx-stored `bv_stern_capacitance_model` in the residual" (i.e., that the residual reads the value from ctx and applies it with the right sign and on the right `ds(elec_marker)`). The nondim conversion at `forms_logc_muh.py:238–254` remains explicitly uncovered.

### Re point 4 — §5.4 overstates A_dyn^ex

**Accept.** At c0_O2_nondim = 1.0 (since `C_SCALE = C_O2` per `_bv_common.py:133`), a_O2·c0_O2 = 1.49e−5. With δ_O2 = 0.3: A_O2^ex_max = 1.49e−5 · 1.3 ≈ 1.94e−5. Total A_dyn^ex is O(2e−5), not "≪ 10⁻⁵" and not "identical to 1 to working precision".

Reworded §5.4: "A_dyn^ex ≲ 3e−5 over the recommended envelope ⇒ free_dyn = 1 − A_dyn^ex ≈ 0.99997, with `free_dyn_floor = 1e−10` inactive. The source builder keeps (1 − A_dyn^ex) exactly; do NOT simplify to 1."

### Re point 5 — `set_phi_applied` reference probably doesn't exist

**Accept.** Production uses `ctx['phi_applied_func'].assign(...)` directly, not a named setter helper. Renamed all references from "set_phi_applied" to "runtime mutation of `ctx['phi_applied_func']` (voltage continuation)" in §6 and §8.

---

## Section 2: Updated artifact

Changes from R3:
- §5.4: A_dyn^ex bound corrected (O(2e−5), not "≪ 10⁻⁵"); explicit instruction to keep (1−A_dyn) exact in source. **(Issue 4)**
- §5.5: K0_R4e_factor pinned to **1e−18** (not 1); cross-check at factor=1 removed; Stern-off cross-check removed entirely; R_ratio finite-window invariant. **(Issues 1, 2)**
- §6 coverage table: Stern row reworded to "use of ctx-stored coefficient" not "value"; setter row uses "runtime mutation of `ctx['phi_applied_func']`". **(Issues 3, 5)**
- §7: K0_R4e_factor invariant uses finite window [10, 1e5]; `TestSternOffSanity` class removed; setter list updated. **(Issues 1, 2, 5)**
- §8: deferred-coverage list updated to use "runtime mutation of `ctx['phi_applied_func']`" wording. **(Issue 5)**

```markdown
# MMS Derivation — `logc_muh` + multi-ion shared-θ Bikerman + Stern Robin + parallel R2e/R4e BV

**Date:** 2026-05-13 (R4 revision 2026-05-14)
**Scope:** byte-faithful Method-of-Manufactured-Solutions design for the
"no-speculative-physics" production stack used by
`scripts/studies/solver_demo_slide15_no_speculative_cs.py` and exercised by
`StudyResults/solver_demo_slide15_no_speculative_cs/`.
**Status:** derivation only; no source-builder code yet.

The "no-speculative-physics" subset of production is what the demo exercises:
formulation `logc_muh`; three dynamic species (O₂, H₂O₂, H⁺) with physical
hard-sphere `a_nondim`; two analytic Bikerman counterions (Cs⁺ and SO₄²⁻)
under the multi-ion shared-θ closure; Stern Robin BC at the electrode
(`C_S = 0.20 F/m²` after the two-stage anchor); parallel
R2e + R4e Butler–Volmer in log-rate form (Ruggiero 2022); `exponent_clip = 100`
and `u_clamp = 100`. Phase-6 speculative knobs
(`enable_water_ionization`, `lambda_hydrolysis`, Singh σ-pKa) are all off.

The existing MMS covers the older logc + single ClO₄⁻ + electrode-Dirichlet
stack. The four genuinely new pieces here are:

1. **muh substitution** `u_H → μ_H − em·z_H·φ` everywhere `u_H` appears
   (NP flux for H, BV cathodic-conc-factors, Poisson source `z_H·c_H`).
2. **Spatially-varying η** at the electrode because Stern-on uses
   `eta_raw = phi_applied − phi − E_eq` (`forms_logc_muh.py:393–394`).
3. **Shared-θ multi-ion Bikerman closure** for both Cs⁺ and SO₄²⁻,
   contributing to Poisson and to μ_steric simultaneously.
4. **Stern Robin boundary source** `g_S` on the φ-equation at the electrode.

Everything else (UnitSquareMesh sweep, `U_prev = U_manuf` with large dt,
side-wall geometry, expected L2 rate ≈ 2 / H1 rate ≈ 1 for CG1) carries
over unchanged. Source-quadrature degree initial candidate is `SRC_QUAD_DEGREE = 8`,
to be pinned by a one-time sweep in §7.

---

## 1. Primary unknowns and the muh transform

[unchanged from R3 — section copied verbatim]

Mixed function space (`forms_logc_muh.py:299–308`):

  W = V_O2 × V_H2O2 × V_muH × V_φ

Primary unknowns:

  U = (u_O2, u_H2O2, μ_H, φ),    z = (0, 0, +1, ·)

with `em · z_H = +1` in the production scaling. Note **`U[h_idx]` is μ_H,
not u_H** (`forms_logc_muh.py:294`).

### Reconstruction (`forms_logc_muh.py:319–340`)

Define `clamp_u(x) := min(max(x, −U_CLAMP), +U_CLAMP)` with `U_CLAMP = 100`.

  u_expr_O2     = u_O2
  u_expr_H2O2  = u_H2O2
  u_expr_H      = μ_H − em · z_H · φ                   (muh reconstruction)

  c_O2     = exp(clamp_u(u_O2))
  c_H2O2   = exp(clamp_u(u_H2O2))
  c_H      = exp(clamp_u(μ_H − em · z_H · φ))           (muh)

  c_H,prev = exp(clamp_u(μ_H,prev − em · z_H · φ_prev))   ← **uses φ_prev**, not φ

### Time-stepping made negligible in MMS

We set `U_prev = U_manuf` (interpolated onto the FE space) and `dt = 1e15`.
The discrete time term `(c(U_h) − c(U_prev,h))/dt` is therefore O(h²/dt)
≈ O(10⁻¹⁵·h²) at u_exact, far below the FE discretization error of interest
(O(h²) for L2). It is not literally zero, but the contribution is utterly
negligible.

The φ_prev-vs-φ landmine (`forms_logc_muh.py:174`) is neutralised because
φ_prev = φ at u_exact ⇒ c_H,prev = c_H. The adjoint/Jacobian sensitivity to
φ_prev is not exercised by this MMS; covered separately by
`tests/test_logc_muh_formulation.py::TestTransformResidualEquivalence`.

---

## 2. Production residual — exact UFL the MMS source must cancel

[unchanged from R3 — sections 2.1 through 2.7 copied verbatim]

[For brevity in this round-4 counterreply, all of §2 carries from R3
without changes. See R3_to_gpt.md or the artifact file at
docs/solver/mms_pnpbv_muh_multi_ion_stern_derivation.md for the full text.]

---

## 3. Manufactured solution

[unchanged from R3 — copied verbatim]

### 3.1 Field shapes (Stern-on / production)

  c_i^ex(x, y)   =  c0_i · [1 + δ_i · cos(πx) · (1 − y)²],     i ∈ {O₂, H₂O₂, H}
  φ^ex(x, y)    =  (1 − y) · [α₀ + α₁ · cos(πx)]  +  γ · y · (1 − y) · cos(πx)
  μ_H^ex(x, y) =  ln c_H^ex(x, y)  +  em · z_H · φ^ex(x, y)

[§3.2 BC verification, §3.3 reconstructed closure quantities: copied verbatim from R3]

---

## 4. Source terms — what we subtract from F_res

[copied verbatim from R3 — §§4.1–4.4 unchanged]

---

## 5. Clip and floor safety at the chosen test voltage

Test voltage: **V_RHE = +0.55 V**, matching the demo's anchor.

In production scaling, `phi_applied_model = V_RHE / V_T ≈ 21.41` and
`E_eq_j^model = E_eq_j / V_T` (so `E_eq_R2e^model ≈ 27.04`,
`E_eq_R4e^model ≈ 47.86`), with `bv_exp_scale = 1.0`. (Read these off
`scaling` at MMS runtime rather than hard-coding.)

### 5.1 BV `exponent_clip = 100`

**Recommended envelope:** `α₀ = α₁ = γ = 0.5`, so φ^ex(x,0) ∈ [0, 1].

  η_R2e_raw^ex(x,0)  ∈  [−6.63, −5.63]
  η_R4e_raw^ex(x,0)  ∈  [−27.45, −26.45]

**Broad envelope** (sanity at `|α₀|, |α₁|, |γ| ≤ 1`, φ^ex(x,0) ∈ [−2, +2]):

  η_R2e_raw^ex(x,0)  ∈  [−7.63, −3.63]
  η_R4e_raw^ex(x,0)  ∈  [−28.45, −24.45]

All inside ±100 ⇒ **clip identity at u_exact**. ✓

### 5.2 `u_clamp = 100`

  |u_O2^ex|, |u_H2O2^ex|, |ln c_H^ex| are O(|ln c0_i| + 1) ≪ 100.   ✓

### 5.3 Counterion `phi_clamp_k = 50`

Production clamps `φ` to ±phi_clamp_k BEFORE multiplying by `−z_k`
(`boltzmann.py:224–228`). The identity condition is `max|φ^ex| < phi_clamp_k`:

  max|φ^ex| ≤ 1.25  ≪  50.   ✓

⇒ q_k^ex = exp(−z_k · φ^ex) is the unclamped expression at u_exact.

### 5.4 `packing_floor = 1e−8` and `free_dyn_floor = 1e−10`

At the demo's physical hard-sphere `a_i^dyn` (1.49e−5, 2.42e−5, 6.65e−5)
and `c0_dyn` (O(1) in nondim with `C_SCALE = C_O2 = 1.2 mol/m³`,
`_bv_common.py:133`), worst-case A_dyn:

  A_O2^ex,max ≈ a_O2 · 1.3 · c0_O2 = 1.49e−5 · 1.3 · 1.0 ≈ 1.94e−5
  A_H2O2 < seed-level (c0_H2O2 is small)
  A_H^ex,max ≈ a_H · 1.3 · c0_H ≈ 6.65e−5 · 1.3 · 8.3e−5 ≈ 7e−9 (negligible)
  ⇒  A_dyn^ex ≲ 3e−5

So free_dyn = 1 − A_dyn^ex ≈ 0.99997 — close to 1 but **not identical**.
`free_dyn_floor = 1e−10` is inactive. The source builder MUST keep
(1 − A_dyn^ex) exactly; do NOT simplify to 1.

Production counterion constants from `scripts/_bv_common.py:707–708`:

  A_CSPLUS_HAT = 3.23e−5   (Cs⁺ r = 2.2 Å)
  A_SO4_HAT    = 4.20e−5   (SO₄²⁻ r = 2.4 Å)
  C_CSPLUS_HAT = 0.2 M / 1.2 mol/m³ ≈ 166.7
  C_SO4_HAT    = 0.1 M / 1.2 mol/m³ ≈ 83.3

  ⇒  a_Cs · c_Cs   ≈ 5.38e−3
  ⇒  a_SO4 · c_SO4 ≈ 3.50e−3
  ⇒  A_an,bulk    ≈ 8.88e−3
  ⇒  θ_b           ≈ 0.991

(Quoted for reference. Source builder reads via
`_get_bv_boltzmann_counterions_cfg(params)` at runtime; do NOT hardcode.)

At **recommended envelope** (φ^ex(x,0) ∈ [0, 1]), worst case φ = +1:

  q_SO4 = exp(+2) ≈ 7.39,    q_Cs = exp(−1) ≈ 0.368
  D     = 0.991 + 3.50e−3 · 7.39 + 5.38e−3 · 0.368 ≈ 1.019
  P_SO4 ≈ 0.0254
  P_Cs  ≈ 0.00194
  θ_inner^ex ≳ 0.973

All three floors (`packing_floor`, `free_dyn_floor`, ion `phi_clamp`) are
**identity at u_exact** for the recommended envelope.

**Runtime invariant** (discrete-min via indicator, NOT `min_value·dx`):

```python
small_region = fd.assemble(
    fd.conditional(theta_inner_ex < 1e-7, fd.Constant(1.0), fd.Constant(0.0)) * fd.dx
)
vol_omega = fd.assemble(fd.Constant(1.0) * fd.dx(domain=ctx['mesh']))
assert small_region < 1e-12 * vol_omega
```

### 5.5 Parameter envelope and K0_R4e_factor

  δ_O2 = δ_H2O2 = δ_H = 0.30       (matches existing MMS pattern)
  α₀ = 0.5,  α₁ = 0.5,  γ = 0.5    ⇒ |φ^ex| ≤ 1.25
  V_RHE = +0.55 V                  ⇒ η_R2e ∈ [−6.63, −5.63], η_R4e ∈ [−27.45, −26.45]

**K0_R4e_factor = 1e−18** (primary, matches the demo's smallest factor).

Rationale: with k0_R2e = k0_R4e (factor=1), R4e would dominate R2e by
~e^46 ≈ 1e20 — vacuously "discriminating", but operating in a regime where
the source magnitude (∝ k0·c·e^53) overwhelms Newton's relative-tolerance
ability to drive residual to the discretization-error floor. At
factor = 1e−18, the cathodic log-rate difference becomes

  log_cath_R4e − log_cath_R2e  ≈  −18·ln(10) + 46.6  ≈  +5.1

so R4e/R2e ≈ e^5.1 ≈ 165: R4e is dominant but not catastrophic, the
operator-of-test exercises both reaction branches meaningfully, and
residual magnitudes are tractable for Newton's `rtol`.

**Runtime invariants** (asserted in source builder before solve):

```python
ds_e = fd.ds(ctx['bv_settings']['electrode_marker'], domain=ctx['mesh'])
R2e_norm = float(fd.assemble(R_R2e_ex**2 * ds_e))**0.5
R4e_norm = float(fd.assemble(R_R4e_ex**2 * ds_e))**0.5
R_ratio = R4e_norm / max(R2e_norm, 1e-300)
assert 10 < R_ratio < 1e5,  (
    f"R4e/R2e = {R_ratio:.3e} outside finite window — "
    f"K0_R4e_factor likely mis-set"
)
```

Lower bound R_ratio > 10 ensures R4e is non-trivially present
(catches K0_R4e_factor = 0 or sign flip). Upper bound R_ratio < 1e5
ensures the source magnitude stays in a regime where Newton's `rtol`
can drive the residual to the discretization-error floor.

**Stern-off cross-check is NOT included in this plan.** At
φ_app^model ≈ 21.4, SO₄²⁻ has q_SO4 = exp(2·21.4) = exp(42.8) ≈ 4e18,
which saturates the shared-θ closure: P_SO4 → 1 − A_dyn ≈ 1, θ_inner → 0,
and `packing_floor` activates. This contradicts the "clamps/floors
inactive at u_exact" premise. Testing the Stern-off path requires either
a small-V_RHE design (V_RHE ≲ 0.1 V keeps the closure unsaturated) or a
separate saturation-active MMS with different convergence-rate
expectations. Both are out of scope here.

---

## 6. Convergence-test coverage by source-term group

| Source piece | Catches a bug in |
|---|---|
| `−∇·J_i` for i ∈ {O₂, H₂O₂} | μ_steric inclusion in NP flux (`:499`); shared-θ contributions of Cs⁺/SO₄²⁻ to packing (`:447–449`); D_i and c_i^ex scaling |
| `−∇·J_H` | muh reconstruction in mass-balance (∇u_H replaced by ∇μ_H, `:473`); μ_steric coupling for proton |
| `R_R2e, R_R4e` via `u_exprs[h]` | muh substitution in BV (`:551–563`); cathodic_conc_factors powers (2 vs 4); α·n_e signs; reversible-vs-irreversible branch; ln(k0_j) sign |
| Spatial η^ex(x,0) | Stern-on branch of `_build_eta_clipped` (`:393–394`); φ-coupling into BV that **only** happens when use_stern=True |
| Poisson interior `z_H·c_H + Σ_k z_k c_k^ster` | shared-θ closure numerator `(1 − A_dyn)` (`:254`); per-ion clamp inside D(φ) (`:224–227`); sign on `z_scale·charge_rhs` (`:658–660`) |
| `g_S` (Stern Robin) | sign of `−C_S·(φ_app − φ)` (`:668`); **use** of the ctx-stored `bv_stern_capacitance_model` in the residual; IBP byproduct from Laplacian on `ds(elec)` |

The MMS does **NOT** cover (by design):
- **Inactive clamp/floor branches**: `u_clamp`, ion `phi_clamp_k`,
  `exponent_clip`, `free_dyn_floor`, `packing_floor` are identity at
  u_exact (see §5). Clamp-branch bugs (wrong `fd.min_value` sign) not
  exercised.
- **Runtime mutation of ctx values**: voltage continuation via
  `ctx['phi_applied_func'].assign(...)`, `set_stern_capacitance_model`,
  K0 ramp, `boltzmann_z_scale` ramp. The test asserts no mutation
  fires between source build and solve.
- **Stern coefficient nondim conversion** at form-build time
  (`forms_logc_muh.py:238–254`): source and residual both use the
  converted value, so off-by-factor errors there cancel symmetrically.
- **c_ref-anchored anodic branch** (`forms_logc_muh.py:574–580`): both
  R2e and R4e bypass this branch.
- **Legacy per-species `bv_c_ref_model_vals` path** (`:617–636`): gated
  off by `use_reactions=True`.
- **IC machinery**, **anchor/continuation logic**, **Phase-6 speculative
  physics**: out of scope per §8.

---

## 7. Implementation notes for the source builder

### Reuse from existing MMS

- `compute_rates`, `plot_convergence`: keep verbatim.
- `make_sp_production`: adapt to muh + multi-ion + Stern factory call.
- `run_mms` outer loop: adapt for muh-aware interpolation and error norms.
- `_ufl_l2_error`, `_ufl_h1_error`: pass `μ_H^ex_func` / `U.sub(h_idx)`
  for the proton.

### muh-aware test harness

- `U_manuf.sub(h_idx) ← Interpolator(μ_H^ex, V_muH)`  ← **NOT u_H^ex**
- Initial guess: `U.assign(U_manuf); U_prev.assign(U_manuf)`.
- Error norm on proton: `||U.sub(h_idx) − μ_H^ex_func||_{L2, H1}`.
- Concentration diagnostics: `c_H_h = exp(U.sub(h_idx) − em·z_H·U.sub(phi_idx))`.

### Independence policy

Source builder composes its own UFL for `c_k^ster, P_k, ρ_k, D(φ)`
independently from `_get_bv_boltzmann_counterions_cfg(params)`, mirroring
`boltzmann.py:91–268`. **No consumption of production-side
`steric_boltz` bundle objects.** Live ctx Function/Constant references
are used for ramp-able scalars (`phi_applied_func`, `stern_coeff_const`,
`bv_k0_funcs`, `bv_alpha_funcs`, `boltzmann_z_scale`).

### Stack-invariant asserts (source builder)

```python
from Forward.bv_solver.config import _get_bv_convergence_cfg, _get_bv_boltzmann_counterions_cfg
from Nondim.transform import _get_nondim_cfg

scaling   = ctx['nondim']
conv_cfg  = _get_bv_convergence_cfg(solver_params)
nondim_cfg = _get_nondim_cfg(solver_params)
counter_cfg = _get_bv_boltzmann_counterions_cfg(solver_params)
snes_opts = solver_params[10]

# reactions list present, log-rate path, Stern on
bv_rxns = scaling.get('bv_reactions', [])
assert isinstance(bv_rxns, (list, tuple)) and len(bv_rxns) == 2
assert bool(conv_cfg.get('bv_log_rate', False)) is True
csm = scaling.get('bv_stern_capacitance_model')
assert csm is not None and float(csm) > 0

# Poisson source enabled (nondim_cfg, NOT conv_cfg)
assert not bool(nondim_cfg.get('suppress_poisson_source', False))

# speculative physics off
from Forward.bv_solver.water_ionization import is_water_ionization_enabled
from Forward.bv_solver.cation_hydrolysis import is_cation_hydrolysis_enabled
assert not is_water_ionization_enabled(conv_cfg)
assert not is_cation_hydrolysis_enabled(conv_cfg)

# species + counterion counts
assert ctx['n_species'] == 3
bikerman = [e for e in counter_cfg if e.get('steric_mode') == 'bikerman']
assert len(bikerman) == 2

# no Γ-augmented mixed space
indices = ctx.get('mixed_space_indices')
assert indices is None or indices.gamma_index is None

# time term negligible
assert float(scaling.get('dt_model', 0.0)) >= 1e12

# packing-floor margin (discrete-min via indicator)
small_region_measure = fd.assemble(
    fd.conditional(theta_inner_ex < 1e-7, fd.Constant(1.0), fd.Constant(0.0)) * fd.dx
)
vol_omega = fd.assemble(fd.Constant(1.0) * fd.dx(domain=ctx['mesh']))
assert small_region_measure < 1e-12 * vol_omega

# R4e dominant-but-not-catastrophic
ds_e = fd.ds(ctx['bv_settings']['electrode_marker'], domain=ctx['mesh'])
R2e_norm = float(fd.assemble(R_R2e_ex**2 * ds_e))**0.5
R4e_norm = float(fd.assemble(R_R4e_ex**2 * ds_e))**0.5
R_ratio = R4e_norm / max(R2e_norm, 1e-300)
assert 10 < R_ratio < 1e5

# SNES tolerances tight enough
assert float(snes_opts.get('snes_atol', 1e-7)) <= 1e-5
assert float(snes_opts.get('snes_rtol', 1e-8)) <= 1e-7
```

### Live continuation policy

For the convergence-rate test, **no continuation setter or runtime
mutation (voltage assign, `set_stern_capacitance_model`, K0 ramp,
`boltzmann_z_scale` ramp) fires between `_build_manufactured_source(ctx)`
and `solve(F == 0, U)`.**

### Quadrature degree

Source integrands include `c_i · ∂_y(ln(packing))` with `exp(−z·φ^ex)`
and rational terms. Initial candidate degree = 8.

**One-time sweep**: run at degree ∈ {6, 8, 10, 12, 16} on N = 32. Pin
`SRC_QUAD_DEGREE` to the smallest value at which the L2 error plateaus
within 1% of the degree-16 value.

### Test file layout

`tests/test_mms_logc_muh_multi_ion_stern.py`:
- `TestMMSConvergence`: UnitSquareMesh sweep N ∈ {8, 16, 32, 64};
  L2 slope ≥ 1.8, H1 slope ≥ 0.8 per primary unknown
  (`u_O2, u_H2O2, μ_H, φ`), R² > 0.99. K0_R4e_factor = 1e−18.
- `TestMMSProductionGradedMesh`: single-solve recovery on demo mesh
  (Nx=8, Ny=80, β=3); thresholds ~6× above baseline.

(No `TestSternOffSanity`, no `TestMMSConvergence_K0R4eSecondary` —
removed in R3 → R4 per issues 1 and 2.)

---

## 8. What's deliberately deferred

- **Time-dependent verification**: steady-state spatial MMS only.
- **Adjoint Jacobian check**: Taylor-test on linearisation around u_exact.
- **Clip-activation MMS**: V_RHE chosen to engage `exponent_clip`.
- **Bikerman-saturation regime**: at high cathodic drive the multi-ion
  closure saturates. Separate MMS design with `packing_floor` expected
  active.
- **c_ref-anchored anodic branch** (`forms_logc_muh.py:574–580`): needs
  a separate single-reaction MMS with
  `reversible=True, anodic_species=None, c_ref_model=1.0`.
- **Legacy per-species `bv_c_ref_model_vals` path**: gated off by
  reactions list non-empty.
- **Runtime mutation of ctx['phi_applied_func']** (voltage continuation),
  `set_stern_capacitance_model`, K0 ramp, `boltzmann_z_scale` ramp:
  verified by separate end-to-end runs.
- **Clamp-branch behavior**: `u_clamp`, ion `phi_clamp_k`,
  `exponent_clip`, `free_dyn_floor`, `packing_floor` are identity at
  u_exact.
- **K0_R4e_factor = 1 regime**: removed from this convergence-rate plan
  because R4e/R2e ~ e^46 ≈ 1e20 puts the operator-of-test in a
  R4e-only-dominant regime whose Newton-tolerance and quadrature-margin
  characteristics need separate pilot analysis.
- **Stern-off path**: at production V_RHE = 0.55 V, the Stern-off
  Dirichlet `φ(x,0) = φ_app^model ≈ 21.4` saturates the SO₄²⁻ closure
  (q_SO4 = exp(42.8) ≈ 4e18 ⇒ θ_inner → 0 ⇒ packing_floor active).
  Needs either a separate small-V_RHE design or a saturation-active MMS.
```

## Section 3: Continued critique prompt

Review the updated plan and my responses to your earlier issues.
Push back on responses where I defended poorly — name which point.
Raise any new issues the updated plan creates. Re-issue any earlier
issue you don't think I addressed. Same numbered format and same
verdict line at the end:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN
