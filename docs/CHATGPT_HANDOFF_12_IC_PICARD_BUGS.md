# Handoff #12 — Two IC bugs in the BV‑PNP forward solver

**Date:** 2026‑05‑07  ·  **Status:** Diagnosed, not yet fixed  ·  **Author:** Claude (with Jake)

We need a sanity check on a diagnosis of the production forward solver's
initial-condition (IC) builder. The IC is the matched-asymptotic
``debye_boltzmann`` seed used at every cold-start in the C+D
orchestrator. It used to land Newton in basin almost everywhere; on the
current production stack (3sp + Bikerman ClO4- + ``logc_muh`` + log-rate
BV + Stern 0.10 F/m² + clip=100), only **1/19** voltages cold-converge
from the IC. The other 18 are warm-walked from a single +0.60 V anchor.

We've isolated **two independent IC bugs** by direct
residual-at-IC measurement and code reading. We want a critical review
of the diagnosis and the proposed fixes before touching the solver.

## 1. Stack & conventions

Coordinate y ∈ [0,1]: y=0 is the electrode boundary, y=1 is the bulk.
Dimensional → nondimensional uses RT/F = V_T ≈ 0.0258 V at 298 K.

| Object | Symbol | Value / definition |
|---|---|---|
| Mesh | graded rect | Nx=8, Ny=200, β=3 |
| Species | (O2, H2O2, H+) | z_vals = (0, 0, +1), 3 dynamic |
| Formulation | logc_muh | u_O2 = ln c_O2; mu_H = u_H + em·z_H·φ |
| Counterion | ClO4- analytic | bikerman closure (residual-side) |
| Stern | Robin | F_res -= C_s · (φ_app − φ) · w · ds |
| Bulk concs | c0_O2_hat=1, c0_H_hat=0.2, c0_ClO4=0.2, c0_H2O2_seed=1e-4 |
| BV kinetics | R1: O2→H2O2 (k0=1.2e-3, α=0.627, n_e=2, E_eq=0.68 V) |
| | R2: H2O2→H2O (k0=5e-5, α=0.5, n_e=2, E_eq=1.78 V) |
| BV form | log-rate | log_R = log(k0)+u_cat+Σpower(u_sp−ln c_ref) − α·n_e·η |
| Eta clip | exponent_clip=100 | clip ON eta_scaled BEFORE α·n_e factor |
| Initializer | debye_boltzmann | composite-ψ + multispecies-γ + Picard outer loop |

The orchestrator we use is ``solve_grid_per_voltage_cold_with_warm_fallback``
(the "C+D" strategy): per-V cold (with internal z-ramp), then warm-walk
substepping for cathodic/anodic edges that don't cold-converge.

The **cold-start path B** (debye_boltzmann IC): builds the analytical
IC, then runs Newton at z=1 directly. If Newton diverges, it sets
``initializer_fallback=True``, re-seeds with ``set_initial_conditions_logc_muh``
(linear-φ), and falls into the standard z=0 → z=1 ramp. So the analytical
IC's quality drives whether Path B succeeds.

The Picard outer loop inside the IC builder solves for (R1, R2, H_o, ψ_D)
at the matched-asymptotic outer-region balance:

```python
# forms_logc_muh.py:861–908 (Picard outer loop, abridged)
for k in range(1, MAX_ITERS+1):                       # MAX_ITERS=50
    H_s = max(H_o * exp(-psi_D), 1e-300)              # surface H+

    log_h_factor1 = sum(p*(log(H_s) - log(c_ref))     # H+ stoich factor (R1)
                        for p in cathodic_factors_R1)
    log_A1 = log(k1) + log_h_factor1 - α1·n_e·eta1
    log_B1 = log(k1)                + (1-α1)·n_e·eta1
    log_A2 = log(k2) + log_h_factor2 - α2·n_e·eta2
    A1, B1, A2 = exp(log_A1), exp(log_B1), exp(log_A2)

    # 2x2 diffusion-balance solve for R1, R2:
    m11 = 1 + A1/D_O + B1/D_P;   m12 = -B1/D_P
    m21 =     -A2/D_P        ;   m22 = 1 + A2/D_P
    rhs1 = A1·O_b - B1·P_b   ;   rhs2 = A2·P_b
    R1, R2 = solve(m, rhs)                            # (with ω=0.5 damping)

    O_s   = max(O_b - R1/D_O,  1e-300)                # surface c_O2
    P_s   = max(P_b + (R1-R2)/D_P, P_FLOOR)           # surface c_H2O2
    H_o   = max(H_b - (R1+R2)/D_H, 1e-300)            # outer-region H+
    phi_o = log(H_o / c_clo4_bulk)                    # outer reference φ
    psi_D = phi_applied_model - phi_o                 # diffuse-layer drop

    delta = |ΔR1|/|R1| + |ΔR2|/|R2|
    if delta < TOL:  break                            # TOL = 1e-6
```

Outputs of Picard: (R1, R2, H_o, O_s, P_s, ψ_D). ω=0.5 damped fixed-point.

## 2. The three IC code paths

```python
# IC dispatcher (forms_logc_muh.py:683–701):
def set_initial_conditions_debye_boltzmann_logc_muh(ctx, sp):
    ok, reason, picard_iters = _try_debye_boltzmann_ic_muh(...)
    if ok:
        ctx['initializer_fallback'] = False
        return
    ctx['initializer_fallback'] = True
    ctx['initializer_fallback_reason'] = reason
    set_initial_conditions_logc_muh(ctx, sp)        # linear-φ fallback
```

**Path 1 (Picard succeeds, bikerman branch)** at line 949–1024:

```python
phi_init_expr = ln(H_outer/c_clo4_bulk) + psi
gamma_psi = 1 / (1 + a_h ·H_outer    ·(exp(-psi) − 1)
                   + a_cl·c_cl_anchor·(exp(+psi) − 1))
log_gamma = ln(gamma_psi)

U_prev.sub(0).interpolate(ln(O_outer) + log_gamma)            # u_O2
U_prev.sub(1).interpolate(ln(P_outer) + log_gamma)            # u_H2O2
u_h_init   = ln(H_outer) − psi + log_gamma
mu_h_init  = u_h_init + em·z_H·phi_init_expr                  # mu_H
U_prev.sub(2).interpolate(mu_h_init)
U_prev.sub(n).interpolate(phi_init_expr)
```

At y=0:
* O_outer(0) = O_s (Picard's surface c_O2)
* ψ(0) = ψ_D
* φ(0) = ln(H_o/c_clo4_bulk) + ψ_D = phi_o + ψ_D = **phi_applied_model**
* u_O2(0) = ln O_s + log γ(0)  →  c_O2_residual_surface = O_s · γ(0)

**Path 2 (Picard succeeds, legacy ideal-counterion branch)** at line 1025–1041
(``apply_bikerman_ic = False``):

```python
psi = psi_gc                                                  # tanh-GC
phi_init_expr = ln(H_outer/c_clo4_bulk) + psi
U_prev.sub(0).interpolate(ln(O_outer))                        # NO log_gamma
U_prev.sub(1).interpolate(ln(P_outer))                        # NO log_gamma
u_h_init   = ln(H_outer) − psi                                # NO log_gamma
mu_h_init  = u_h_init + em·z_H·phi_init_expr
```

At y=0: c_O2_residual_surface = O_s (no γ correction).

**Path 3 (Picard fails, linear-φ fallback)** in
`set_initial_conditions_logc_muh` at line 597–656:

```python
phi_init_expr = phi_applied_model · (1 − y)
for i in range(n):
    if i in mu_species:                                       # H+
        U_prev.sub(i).interpolate(ln(c0_H) + em·z_H·phi_init_expr)
    else:
        U_prev.sub(i).assign(ln(c0_i))                        # bulk uniform
U_prev.sub(n).interpolate(phi_init_expr)
```

At y=0: φ(0) = phi_applied_model, c_i_surface = c0_i (bulk).

## 3. The residual the SS solver actually evaluates

```python
# forms_logc_muh.py:298–308
if use_stern:
    eta_raw = phi_applied_func - phi - E_eq_const             # Stern-aware
elif conv_cfg["use_eta_in_bv"]:
    eta_raw = phi_applied_func - E_eq_const                   # no-Stern
else:
    eta_raw = phi - E_eq_const
eta_scaled = bv_exp_scale * eta_raw
if conv_cfg["clip_exponent"]:
    eta_clipped = clip(eta_scaled, ±exponent_clip)
```

```python
# forms_logc_muh.py:413–429 (log-rate BV cathodic)
log_cathodic = log(k0_j) + u_exprs[cat_idx] − α_j · n_e · eta_j
for factor in cathodic_conc_factors:
    log_cathodic += power · (u_exprs[sp_idx] − ln c_ref)
cathodic = exp(log_cathodic)
R_j = cathodic - anodic
F_res -= stoi[i] · R_j · v_list[i] · ds(electrode_marker)     # for each species
```

Crucial detail: ``u_exprs[i]`` for non-mu species is ``U.sub(i)`` (the
log-c stored field, including any log_γ the IC put there). For the mu
species H+, ``u_exprs[i] = mu_H − em·z_H·φ`` — the muh reconstruction
of log c_H. So the residual sees whatever the IC writes into U.

## 4. Bug #1 — Stern-eta inconsistency

**Mechanism.** When Stern is on, the residual computes
``eta_raw = phi_applied − φ(0) − E_eq``. All three IC paths happen to set
``φ(0) = phi_applied_model``:

* Linear-φ fallback: φ(y) = phi_applied · (1−y) → φ(0) = phi_applied.
* Picard bikerman branch: φ(0) = ln(H_o/c_clo4_bulk) + ψ_D = phi_o + ψ_D.
  But Picard set ``ψ_D = phi_applied − phi_o`` on line 854/897, so
  φ(0) collapses to phi_applied.
* Picard ideal branch: same algebra.

Therefore ``eta_raw at IC = phi_applied − phi_applied − E_eq = −E_eq``,
**V-independent**, with ``α·n_e·E_eq_R2 ≈ 1·68.9`` giving
``exp(50) ≈ 5×10²¹`` — exactly the magnitude we measure (see §6).

The Picard outer loop independently uses the *no-Stern* eta on line 817:

```python
eta = bv_exp_scale * (phi_applied_model - E)         # no φ(0) correction
```

So with Stern on, the Picard solves a no-Stern problem (eta = phi_applied − E_eq)
but the residual at the same IC sees eta = −E_eq. **Even when Picard converges,
its R1/R2 don't match the residual's R1/R2** because they use different etas.

In the no-Stern case, the residual's eta is ``phi_applied − E_eq`` (line 301)
and the Picard agrees (line 817). So no-Stern + Picard-converged is internally
consistent (this is the ||F|| ≈ 1 regime — see §6).

## 5. Bug #2 — Bikerman-γ inconsistency in Picard

**Mechanism.** Commit `77ceff3` ("Bikerman-consistent IC, composite ψ +
multispecies γ") added `+ log_gamma` to the IC's u_i seeds (Path 1 above).
The Picard outer loop was *not* updated to include γ at the surface.

Picard's R1, R2 are solved against ``c_O2_surface = O_s``, ``c_H2O2_surface = P_s``
(no γ). The IC then writes ``u_O2(0) = ln O_s + log γ(0)``, so the residual
sees ``c_O2_surface = O_s · γ(0)``.

For our parameters at, say, V_RHE = +0.5 V (no Stern, ψ_D ≈ +20 V_T at the
Picard fixed point), the surface γ is dominated by the cation-side term:

```
γ(0) = 1/(1 + a_h·H_outer·(e^{-ψ_D} − 1) + a_cl·c_clo4_bulk·(e^{+ψ_D} − 1))
     ≈ 1/(a_cl · c_clo4_bulk · e^{ψ_D})
     ≈ 1/(0.01 · 0.2 · e^{20})  ≈  3 × 10⁻⁹
```

So the IC's c_O2_surface is ~10⁹× different from what Picard balanced
against. The diffusion-flux ↔ BV-rate balance is broken at the surface;
the residual block reflects this.

Empirically (§6): no-Stern + bikerman has ||F|| = 1.1×10³ at V=+0.5 V;
no-Stern + ideal-counterion (legacy γ-free path) has ||F|| = 0.81 at the
same V. **Three orders of magnitude** difference, attributable entirely
to the γ insertion in the IC seeds without Picard update.

We have not yet checked whether the bikerman residual side
(``add_boltzmann_counterion_residual`` + ``build_steric_boltzmann_expressions``)
expects γ-shifted u_i or γ-free u_i. If the residual's bikerman closure
expects γ-free u_i, then the IC's γ-shift may also be inconsistent with
the residual's bikerman closure, not just the Picard.

## 6. Diagnostic data

Diagnostic script: `scripts/diagnose_db_ic_distance.py`. It runs the C+D
orchestrator to obtain the converged SS at every V, then independently
rebuilds a fresh ctx per V, runs `set_initial_conditions(...)`, and
assembles `||F(U_ic; z=1)||` with full charge coupling.

V grid: V_RHE ∈ {−0.7, …, +1.0} (19 pts).

### 6.1 Production (bikerman + Stern 0.10): IC residuals

Picard converges only at V_RHE ≥ +0.3 V; below that it limit-cycles
with delta=2 (50 iters, no convergence) and the dispatcher falls back to
linear-φ.

```
V_RHE   fb (picard fail?)  ||F||           per-block: [u_O2, u_H2O2, mu_H, phi]
−0.700  True               4.94e+21        [1.12e+11, 2.21e+21, 4.42e+21, 6.20e-03]
−0.600  True               4.94e+21        [1.12e+11, 2.21e+21, 4.42e+21, 6.15e-03]
…       (V-independent species blocks across 11 cathodic/onset V's)…
+0.000  True               4.94e+21        [1.12e+11, 2.21e+21, 4.42e+21, 9.75e-20]
+0.200  True               4.94e+21        [1.12e+11, 2.21e+21, 4.42e+21, 6.94e-01]
+0.300  False              2.69e+04        [5.46e+02, 1.20e+04, 2.40e+04, 3.94e-03]
+0.400  False              3.07e+03        [3.07e+03, 2.58e-01, 4.88e+00, 5.78e-03]
+0.500  False              1.11e+03        [1.11e+03, 9.31e-02, 3.51e+00, 6.37e-03]
+0.600  False              3.10e+03        [3.10e+03, 2.61e-01, 5.70e+00, 7.06e-03]
+0.700  False              3.73e+03        [3.73e+03, 3.14e-01, 5.50e+00, 8.26e-03]
+0.800  False              3.54e+03        [3.54e+03, 2.98e-01, 4.30e+00, 9.86e-03]
+0.900  False              1.36e+03        [1.36e+03, 1.15e-01, 4.54e+00, 9.87e-03]
+1.000  False              3.04e+03        [3.04e+03, 2.56e-01, 6.74e+00, 1.00e-02]
```

The 11 fallback rows have **identical** species-block residuals to 3
significant figures, the 4.94e+21 ≈ exp(49.95) signature of α·n_e·E_eq_R2
(exactly Bug #1). The φ block varies with V (the linear-φ profile). At
V=0 the φ block hits machine zero because phi_applied=0.

### 6.2 logc vs logc_muh

logc and logc_muh produce **identical** residuals to many decimal places
across all 19 voltages. This rules out a muh-specific bug.

### 6.3 Same as 6.1 but Stern off

```
V_RHE   fb     ||F||           per-block
−0.700  True   3.36e+33        species-blocks much LARGER than 6.1
−0.100  True   2.42e+23
+0.000  True   4.94e+21        (matches 6.1 — phi_applied=0 makes Stern/no-Stern eta agree)
+0.300  False  5.46e+02        species-blocks much SMALLER than 6.1
+0.400  False  3.07e+03
+0.500  False  1.11e+03
+0.600  False  3.10e+03
+0.700  False  3.73e+03
…
```

* At V=0 the two modes give identical ||F||.  ✓ Bug #1 mechanism.
* For V<0 fallback the no-Stern residual is *larger* (eta = phi_applied−E_eq
  is more cathodic than eta = −E_eq).
* For V≥+0.3 V (Picard converged) the no-Stern residual is up to 50×
  smaller than with-Stern.  ✓ Bug #1 affects converged regime too.

### 6.4 Same as 6.3 but ideal counterion (legacy γ-free path)

```
V_RHE   fb     ||F||           per-block: [u_O2, u_H2O2, mu_H, phi]
+0.300  False  3.02e+06        [7.05e-01, 1.35e+06, 2.70e+06, 6.09e-03]
+0.400  False  2.57e+01        [7.07e-01, 1.15e+01, 2.29e+01, 4.10e-02]
+0.500  False  8.10e-01        [7.07e-01, 1.14e-04, 3.84e-01, 9.26e-02]
+0.600  False  1.96e+00        [7.07e-01, 5.96e-05, 3.84e-01, 1.78e+00]
+0.700  False  5.54e+01        [7.07e-01, 5.96e-05, 3.84e-01, 5.53e+01]
+0.800  False  1.66e+03        [7.07e-01, 5.96e-05, 3.84e-01, 1.66e+03]
+0.900  False  5.26e+04        [7.07e-01, 6.49e-05, 3.84e-01, 5.26e+04]
+1.000  False  1.71e+06        [7.07e-01, 4.73e-04, 3.84e-01, 1.71e+06]
```

* At V=+0.5–+0.6: ||F|| ≈ 1. **The legacy γ-free IC is essentially a SS
  solution** in this V range. (Compare 1.11e+03 at the same V with bikerman
  no-Stern.)
* For V > +0.7 the residual is dominated by the φ block — the
  tanh-Gouy-Chapman ψ profile fails when ψ_D > ψ_sat (saturation regime).
  This is exactly the regime bikerman was supposed to cover.
* For V = +0.3, the dominant residual is in u_H2O2 and mu_H blocks — at
  that V, R2 dominates and the IC's bulk-seed c_H2O2(0) doesn't match SS.

So the bikerman extension correctly addresses the high-|V| φ-block blow-up
(tanh-GC saturation) but introduces a worse problem at moderate V via
the γ-Picard inconsistency.

### 6.5 Picard limit cycle (cathodic, V ≤ +0.2)

For all 11 cathodic voltages the Picard hits ``MAX_ITERS=50`` with
``delta=2`` (oscillation, not slow convergence). The mechanism (verified
by inspection of the iteration variables):

```
iter k:    R1 large → H_o = max(H_b − (R1+R2)/D_H, 1e-300) → 1e-300
           phi_o = log(1e-300 / 0.2) = −688
           psi_D = phi_applied − phi_o ≈ +688
iter k+1:  H_s = H_o · exp(−psi_D) → 0
           A1, A2 → 0  (no driving rate)
           R1, R2 → (1−ω)·R1_old (decay with ω=0.5)
iter k+2:  R1, R2 small → H_o ≈ H_b → phi_o ≈ 0 → psi_D ≈ phi_applied
           H_s ≈ H_b → A1, A2 huge again → R1, R2 large → loop.
```

ω=0.5 isn't enough to break this. Looks like a third independent
issue, separate from Bugs #1 and #2 — it's intrinsic to the Picard
fixed-point dynamics with the H_o → 1e-300 floor.

## 7. Proposed fixes (ranked by blast radius)

### Fix A (smallest): drop log_gamma from IC seeds in the bikerman branch

Lines 1003–1004, 1011 (forms_logc_muh.py) and 931–933 (forms_logc.py):

```python
# replace
U_prev.sub(0).interpolate(ln(O_outer) + log_gamma)
U_prev.sub(1).interpolate(ln(P_outer) + log_gamma)
u_h_init  = ln(H_outer) - psi + log_gamma

# with
U_prev.sub(0).interpolate(ln(O_outer))
U_prev.sub(1).interpolate(ln(P_outer))
u_h_init  = ln(H_outer) - psi
```

This restores Picard ↔ IC consistency (Bug #2). The bikerman residual
will recover γ in the volume during the SS solve. The composite-ψ
profile is preserved (which is what fixes the high-|V| φ-block blowup
seen in 6.4). Does **not** address Bug #1 (Stern-eta).

Expected effect (extrapolating from 6.4): at V=+0.5 V no-Stern, ||F||
should drop from 1.11e+03 → ~1.

### Fix B (medium): Stern-aware φ anchoring in IC

The IC's φ profile should anchor at the *solution-side* surface
potential ``φ(0) = phi_applied − ψ_S`` (after the Stern drop) rather
than at phi_applied. Three sub-changes:

1. Add a Stern self-consistency step at IC build time:
   ``C_stern · ψ_S = σ_surf(ψ_D)`` and ``ψ_S + ψ_D = phi_applied − phi_o``.
   For Bikerman, σ_surf(ψ_D) = α_d (the BKSA factor already computed
   on line 979–982 of forms_logc_muh.py for the composite-ψ).
   1-D nonlinear solve, bisection on ψ_D, ~20 lines.

2. Shift the IC's φ profile by ψ_S so that φ(0) = phi_applied − ψ_S
   instead of phi_applied. Diffuse-layer shape unchanged.

3. Use the Stern-aware eta in the Picard's BV rates:
   ``eta = bv_exp_scale · (phi_applied − φ(0) − E)`` instead of
   ``bv_exp_scale · (phi_applied − E)``.

Expected effect: makes IC and residual agree on eta even when Stern
is enabled. Layered with Fix A, this should bring residuals into
the ||F|| ≈ 1 regime at converged-Picard voltages with the production
stack.

### Fix C (largest): break the cathodic Picard limit cycle

Independent of A/B. Options:

* Anderson acceleration on the Picard outer loop (~20 lines).
* V-continuation inside the IC builder: solve at +0.6 V first, march
  cathodically reseeding from previous Picard state (~30 lines).
* Soften the ``H_o → 1e-300`` floor: smooth log-domain clamp instead of
  hard floor; may eliminate the limit cycle at the floor boundary (~5
  lines, lowest blast radius — try first to see if the limit cycle is
  fundamental or just a floor artifact).

## 8. Open questions for review

1. **Fix A correctness.** Is dropping ``+ log_gamma`` from the IC seeds
   the right fix for the Picard mismatch, or should the Picard be
   updated to include γ at the surface (preserving γ in the IC)? The
   former is much smaller; the latter is "more right" theoretically.
   Are there cases where the volumetric γ structure in the IC is
   load-bearing for Newton convergence?

2. **Picard ↔ residual γ semantics.** Is the bikerman residual
   (``add_boltzmann_counterion_residual``) consistent with γ-shifted u_i
   or γ-free u_i? If shifted, then dropping γ from the IC will break
   the residual side too. (We have not yet read this code path
   carefully.)

3. **Stern self-consistency formula.** For Fix B, is the right closure:
   * Grahame (ideal Boltzmann): ``σ_surf = (2/λ_D)·sinh(ψ_D/2)``, or
   * Bikerman (BKSA): ``σ_surf = α_d`` from line 979–982?
   Which depends on whether the Stern interface "sees" the bikerman
   saturation or the ideal diffuse layer. Our reading suggests
   Bikerman, but we want a check.

4. **Cathodic Picard limit cycle.** Is the H_o → 1e-300 floor
   bouncing fundamental (needs reformulation) or a numerical artifact
   of the floor (smooth clamp suffices)? Specifically: does the
   matched-asymptotic outer-region problem at deep cathodic V have a
   finite well-posed fixed point in (R1, R2, H_o, ψ_D), or does H_o
   really go to zero in the asymptotic limit?

5. **What we may be missing.** We diagnosed two bugs and a Picard
   convergence issue. Are there cross-couplings between Stern and γ
   we haven't accounted for? Is there a third bug hiding in the
   high-|V| φ-block tail at +0.8/+0.9/+1.0 V (which still has ||F||
   ~ 10³ even in the legacy γ-free path)?

## 9. References (file:line)

* IC dispatcher (muh): `Forward/bv_solver/forms_logc_muh.py:683-701`
* Picard outer loop: `Forward/bv_solver/forms_logc_muh.py:861-908`
* IC seeds (bikerman, with γ): `Forward/bv_solver/forms_logc_muh.py:1003-1024`
* IC seeds (legacy ideal, γ-free): `Forward/bv_solver/forms_logc_muh.py:1025-1041`
* Linear-φ fallback IC: `Forward/bv_solver/forms_logc_muh.py:597-656`
* Residual eta selection: `Forward/bv_solver/forms_logc_muh.py:298-308`
* Residual BV term (log-rate): `Forward/bv_solver/forms_logc_muh.py:413-480`
* Residual Stern term: `Forward/bv_solver/forms_logc_muh.py:519-522`
* logc counterparts: `Forward/bv_solver/forms_logc.py:623-952` (mirror)
* Bikerman commit that introduced both bugs: `77ceff3`
  ("Bikerman-consistent IC: composite ψ + multispecies γ for both
  forms_logc and forms_logc_muh", 2026-05-04)
* Diagnostic script: `scripts/diagnose_db_ic_distance.py`
* Diagnostic outputs:
  * `StudyResults/diagnose_db_ic_distance_logc_muh/`        (bikerman + Stern)
  * `StudyResults/diagnose_db_ic_distance_logc/`            (logc, bikerman + Stern)
  * `StudyResults/diagnose_db_ic_distance_logc_muh_nostern/`(bikerman, no Stern)
* Production-stack run that motivated this:
  `StudyResults/iv_curve_full_prod_v07to10/iv_curve.{png,csv,json}`
  (19/19 V_RHE ∈ [-0.7, +1.0] converged, but only 1/19 cold; the rest
  are warm-walks from the +0.60 V cold anchor).
