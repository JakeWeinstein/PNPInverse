# Round 2 — Counterreply

## Section 1 — Per-issue acknowledgment

### Issue 1 — Cathodic prefactor sign

**Accept.** Plan's `R_j = A_j · O_s · (H_o)^n · γ_s` with
`A_j = ... · exp(α_j · n_e_j · η_j_clipped)` is sign-inconsistent with
the codebase. The existing convention in
`Forward/bv_solver/picard_ic.py:_build_picard_prefactors:716-725` is:

```python
log_alpha = log(k_j) + log_gamma + log_h_factor_j - α_j · n_e_j · η_j
```

with `η_j = bv_exp_scale · (eta_drop − E_j_model)` (signed, cathodic
when negative). The sign on `α·n·η` is **negative**: cathodic
enhancement comes from `−α·n·(negative η) > 0`. My handwritten form
flipped it.

**Fix:** Drop the handwritten Phase 1.2 math entirely (see Issue 5
fix).

### Issue 2 — Reversibility of R_2e

**Accept.** `PARALLEL_2E_4E_REACTIONS[0]` has `reversible=True,
anodic_species=1` (H₂O₂). The Picard's actual rate expression in
that branch is:

```text
R_j = α̂_j · O_s − β̂_j · P_s    (for cathodic_species=O₂, anodic_species=H₂O₂)
```

Dropping the `β̂·P_s` term is incorrect.

**Fix:** Drop the handwritten Phase 1.2 math (see Issue 5 fix); the
existing `picard_outer_loop_general` already handles the
three-branch anodic dispatch correctly via `_build_picard_prefactors`
(branch 1 surface-species linear, branch 2 affine constant, branch 3
irreversible).

### Issue 3 — γ powers and proton Boltzmann factor

**Accept.** The actual log-rate algebra is (from
`picard_ic.py:716-724` and the `_build_picard_prefactors` docstring
v3 §3):

- `log_h_factor_j = Σ_f power_f · (log_H_rxn − log(c_HP_ref))`
- `log_H_rxn = log(H_o) − ψ_D + log_γ`     ← Boltzmann shift!
- `log_α̂_j = log(k_j) + log_γ + log_h_factor_j − α·n·η`
- Net γ-power: `γ^{1 + Σ_f power_f}`

For R_2e with H⁺ power=2: `γ̂ = γ^3` and the rate expression
inherently carries `exp(−2·ψ_D)` from the proton Boltzmann shift.
For R_4e with H⁺ power=4: `γ̂ = γ^5` and `exp(−4·ψ_D)`.

My plan's `R_j = A_j · O_s · (H_o)^n · γ_s` was a hand-rolled
cartoon, not the actual derivation.

**Fix:** Drop the handwritten math (see Issue 5).

### Issue 4 — Spurious `· L_eff` factor

**Accept.** The existing nondim Picard in `picard_ic.py:911-919`
(`_surface_concs_from_rates`) uses:

```python
c_{i,s} = max(c_{i,b} + Σ_j s_{i,j} · R_j · λ_i, floor_i)
λ_i = 1/D_i  (or 1/(2·D_i) for H+ ambipolar)
```

There is **no** `L_eff` factor; the nondim setup folds `L_REF` into
the diffusion scale via `D_HAT_i = D_i / D_REF` and
`K_SCALE = D_REF / L_REF`. My plan added a spurious `· L_eff`
multiplication.

**Fix:** Drop the handwritten math (see Issue 5).

### Issue 5 — `picard_outer_loop_general` already exists

**Accept (this is a major plan revision).** Verified by
`grep -rn "picard_outer_loop_general"`:

- Defined: `Forward/bv_solver/picard_ic.py:941-1280`. Generic
  N-reaction signed flux balance + per-reaction stoichiometry +
  reversibility branches + ambipolar `1/(2·D_H)` for H⁺ +
  `topology_hint` dispatch (`"sequential_2e_h2o2"` |
  `"general"`).
- Wired: `forms_logc.py:703,882` and `forms_logc_muh.py:778,936`.
  The "topology gate" my plan referenced (per
  `docs/ruggiero_realignment_plan.md` "What landed" §M3a.2) **no
  longer exists**: the dispatch in `forms_logc_muh.py:809-818` does
  `is_sequential_template = (rxn1[H₂O₂_idx] > 0 AND
  rxn2[H₂O₂_idx] < 0 AND rxn2 not reversible)` and routes
  parallel topology to `topology_hint='general'`, NOT to a
  rejection.

So the realignment plan I was meant to be superseding was already
stale by 2026-05-08. The "M3a.3 will replace the gate with a
generalized Picard" sentence had already been retroactively executed
between when the long-form plan was written and now.

**This means the M3a.2 cold-fail-universally claim has a different
explanation than my plan assumed.** It is NOT "parallel rejected,
linear_phi falls in." It is one of:

- (a) `picard_outer_loop_general` *runs* with `topology_hint=
  'general'` but its Picard non-converges for parallel topology, OR
- (b) `picard_outer_loop_general` converges, but the spatial IC
  interpolation downstream uses a 1:1-incompatible composite-ψ
  closure that breaks Newton (per Issue 13), OR
- (c) The whole IC builds fine but Newton fails at the BV residual
  for some other reason (Stern, mu_h_idx, etc.).

The plan needs an instrumentation pass to distinguish these BEFORE
deciding what to fix.

**Fix:** Replace Phase 1.2 in its entirety with:

> **Phase 1.2 — Audit existing `picard_outer_loop_general` for
> parallel R_2e/R_4e (1 day).**
>
> 1. Add per-iteration logging to `picard_outer_loop_general`
>    behind a `verbose=False` param (off by default; on for the
>    diagnostic). Log: iter k, R_list, c_s_list, phi_o, psi_D,
>    psi_S, gamma_s, eta_list, delta. No code-path changes — just
>    instrumentation.
> 2. Run the existing
>    `scripts/studies/peroxide_window_3sp_parallel_2e_4e.py` driver
>    at one V (V_RHE = +0.55 V; weakest cathodic drive — see Issue
>    16 fix) with verbose Picard logging. Capture stdout to
>    `StudyResults/fast_realignment_2026-05-08/picard_audit/`.
> 3. Diagnose:
>    - Picard fails (delta doesn't shrink, or singular det, or
>      non-finite state) → bug in the generic Picard; investigate
>      and fix.
>    - Picard converges but its post-loop reconstruction goes
>      non-physical → bug in `topology_hint='general'` post-loop
>      reconstruction; investigate.
>    - Picard converges fine, downstream Newton fails → spatial-IC
>      interpolation issue (likely the 1:1 BKSA composite-ψ; see
>      Issue 13).
>    - All converge, page-15 sweep just hadn't been re-run after
>      `picard_outer_loop_general` landed → run the sweep, see what
>      we get.
>
> No new Picard code is written until step 3 says we need it.

### Issue 6 — Disabled-reaction handling not centralized

**Accept.** A single-source-of-truth for "this reaction is disabled"
needs to:

1. Skip the form-builder's `ln(k0_j)` and produce `R_j = 0` UFL.
2. Make `_build_picard_prefactors` skip the disabled rxn (set
   α̂_j = β̂_j = Ĉ_j = 0).
3. Make `_assemble_n_reaction_system` produce a trivial row
   (`M[j,j]=1`, `b[j]=0`) so the linear solve gives `R_j=0`.
4. Make the topology detector ignore disabled rxns when classifying
   `is_sequential_template` vs parallel.

**Fix:** Add a centralized helper in `picard_ic.py`:

```python
def _is_reaction_disabled(rxn: dict) -> bool:
    if not bool(rxn.get("enabled", True)):
        return True
    k0 = float(rxn.get("k0_model", rxn.get("k0", 0.0)))
    return k0 <= 0.0
```

Use at all four sites above. Fast plan adds 1 unit test that
`enabled=False` and `k0=0` produce identical (zero) downstream rate.

### Issue 7 — Topology detection too weak

**Accept.** `[+1, 0]` for H₂O₂ stoich is necessary but not
sufficient. Strict predicate for parallel 2e/4e:

```python
def _is_parallel_2e_4e(reactions: list, h_idx: int) -> bool:
    if len(reactions) != 2:
        return False
    r2e, r4e = reactions
    # n_electrons
    if int(r2e.get("n_electrons", -1)) != 2: return False
    if int(r4e.get("n_electrons", -1)) != 4: return False
    # H2O2 (assume H2O2_idx = 1 for 3sp ORR)
    s2 = r2e.get("stoichiometry", [])
    s4 = r4e.get("stoichiometry", [])
    if len(s2) < 3 or len(s4) < 3: return False
    if int(s2[1]) != +1: return False    # R_2e produces H2O2
    if int(s4[1]) != 0:  return False    # R_4e does NOT touch H2O2
    # O2 consumed by both
    if int(s2[0]) != -1 or int(s4[0]) != -1: return False
    # H+ stoichiometry
    if int(s2[h_idx]) != -2: return False  # 2 H+ per 2e
    if int(s4[h_idx]) != -4: return False  # 4 H+ per 4e
    # Reversibility
    if not bool(r2e.get("reversible", False)): return False
    if bool(r4e.get("reversible", False)):     return False
    return True
```

This is just used in adapter sites for safety asserts; the existing
`is_sequential_template` heuristic in
`forms_logc_muh.py:809-815` continues to do the actual dispatch.

### Issue 8 — Multi-steric Bikerman API change

**Accept and re-scope.** Verified
`Forward/bv_solver/boltzmann.py:159-165` raises
`NotImplementedError("multi-counterion bikerman closure not
supported: ... each appears in the others' denominator")` for
`len(bikerman) > 1`. The closed-form analytic reduction for one
steric ion does not generalize.

**Fast-path resolution per the user's "fast" mandate:**

- **Cs⁺ steric (Bikerman), SO₄²⁻ ideal (unbounded Boltzmann).**
  This avoids the coupled-denominator algebra entirely. The cost is
  that SO₄²⁻ won't saturate at the cathode, but at V_RHE > 0 the
  cathode repels SO₄²⁻ anyway (it's an anion), so its EDL profile
  is a depleted Boltzmann tail — saturation isn't the dominant
  physics there. At V_RHE < 0 (anodic relative to PZC) the plot
  starts to fail for sulfate, but the page-15 grid is dominantly
  cathodic.

- DEFER full multi-steric Bikerman closure (~1-2 weeks of
  derivation work) to post-fast-realignment.

**Fix to plan:** §2.2 entries change to:

```python
DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC: ...  # bikerman
DEFAULT_SULFATE_BOLTZMANN_COUNTERION_IDEAL: ...  # NOT bikerman
```

**Tradeoff acknowledged:** sulfate ideal is a worse approximation
than ClO₄⁻ Bikerman in some regimes; we'd be substituting one
imperfect model for another. But the chemical-identity correction
(SO₄²⁻ instead of ClO₄⁻, z=-2 instead of z=-1) plus correct
ionic strength still buys most of the value.

### Issue 9 — `a_nondim` values violate bulk packing (CRITICAL)

**Accept.** Verified arithmetic with C_SCALE=1.2:

- Cs⁺: a · c_nondim = 0.0044 · (199.9/1.2) = 0.0044 · 166.58 = 0.7330
- SO₄²⁻: a · c_nondim = 0.0048 · (100/1.2)  = 0.0048 · 83.33  = 0.4000
- Bulk pack = 1.133 ⇒ θ_b = -0.133 ⇒
  `boltzmann.py:182-189` raises `ValueError`.

The values I picked are ~150× too large for a 2.2 Å hydrated cation.

**Correct hard-sphere derivation:** for radius r,
`a_phys = (4/3)·π·r³·N_A` (m³/mol), then `a_nondim = a_phys ·
C_SCALE` (where C_SCALE has units mol/m³). For Cs⁺ at r=2.2 Å:

```text
a_phys = (4/3) · π · (2.2e-10)³ · 6.022e23
       = (4/3) · π · 1.06e-29  · 6.022e23
       = 2.69e-5 m³/mol
a_nondim_Cs = 2.69e-5 · 1.2 = 3.23e-5
```

For SO₄²⁻ at r=2.4 Å (Marcus): `a_phys = 3.50e-5 m³/mol`,
`a_nondim_SO4 = 4.20e-5`.

Bulk pack at I=0.3 M:
- Cs⁺: 3.23e-5 · 166.58 = 5.38e-3
- SO₄²⁻: 4.20e-5 · 83.33  = 3.50e-3
- Total = 8.88e-3 ⇒ θ_b ≈ 0.991 ✓

**Fix to plan:** §2.2 uses hard-sphere-derived `a_nondim` values
(documented derivation), not the placeholder `0.0044 / 0.0048` I had.
Also remove the line "probably wrong as written; will tune during
Phase 3 wiring" — it's not OK to land a config that crashes.

### Issue 10 — A_DEFAULT=0.01 continuation start

**Accept.** Same arithmetic: `0.01 · (199.9/1.2 + 100/1.2) = 0.01 ·
249.9 = 2.499` — over 1 even worse than the literature values.

**Fix:** §5b continuation in `a_nondim` ramps from `a_nondim = 0`
(ideal limit, both ions unbounded Boltzmann) toward the
hard-sphere-derived values. NOT from `A_DEFAULT`.

### Issue 11 — `phi_o = ln(H_o / c_clo4_bulk)` is 1:1-specific

**Accept and partially defer.** The existing Picard outer-region
update at `picard_ic.py:1142` is:

```python
phi_o = log(H_o / c_clo4_bulk)
```

This is the closed-form solution to outer-region electroneutrality
for a 1:1 H⁺ + ClO₄⁻ system. For Cs⁺ + SO₄²⁻ + H⁺ multi-ion, outer
electroneutrality is:

```text
H_o + c_Cs_b · exp(-φ_o) − 2 · c_SO4_b · exp(2·φ_o) = 0
```

This is transcendental in φ_o; needs a 1D bisection (cheap, ~30
iter at 1e-12 tolerance).

**Fast-path fix:**

- Add `_solve_outer_phi_multiion(H_o, c_b_list, z_list)` in
  `picard_ic.py` (~30 LOC, bisection on the electroneutrality
  residual).
- Modify `picard_outer_loop_general` to call it instead of the
  hardcoded log-ratio when `len(boltzmann_counterions) >= 2` is
  detected at the call site. For the single-counterion legacy path,
  preserve the closed-form (byte-equivalent regression).

This is a real piece of work — ~1 day. Folds into Phase 1.2 audit
+ multi-ion adaptation.

### Issue 12 — `compute_surface_gamma` is 1:1-specific

**Accept.** Verified:

```python
# picard_ic.py:124-129
denom = (
    1.0
    + a_h * H_o * (_safe_exp(-psi_D) - 1.0)
    + a_cl * c_cl_anchor * (_safe_exp(+psi_D) - 1.0)
)
```

This is hardcoded H⁺ (z=+1) + ClO₄⁻-like anion (z=-1). For multi-ion
the general form is:

```text
γ_s = 1 / (1 + a_h · H_o · (e^(-z_H·ψ_D) − 1)
             + Σ_k a_k · c_b_k · (e^(-z_k·ψ_D) − 1))
```

For our specific case (H⁺ z=+1, Cs⁺ z=+1, SO₄²⁻ z=-2):

```text
γ_s = 1 / (1 + a_h  · H_o   · (e^(-ψ_D)  − 1)
             + a_Cs · c_Cs  · (e^(-ψ_D)  − 1)
             + a_SO4· c_SO4 · (e^(2·ψ_D) − 1))
```

**Fast-path fix:**

- Add `compute_surface_gamma_multiion(H_o, ψ_D, ions: list)` in
  `picard_ic.py` (~25 LOC, list-based). Each `ions` entry is `{"z":
  ±k, "c_anchor": c, "a": a_nondim}`. H⁺ contributes one entry,
  each analytic counterion contributes one entry.
- Have `picard_outer_loop_general` call the new helper when
  `len(boltzmann_counterions) >= 2`. Single-counterion path
  preserved for byte-equivalent regression.

### Issue 13 — Asymmetric composite-ψ self-deception

**Accept (validated by reading the IC interpolation path).** The
Picard tracks scalar `(R_list, c_s, H_o, ψ_D, ψ_S, γ_s, η_list)`,
but the spatial IC profile in `forms_logc_muh.py:_try_debye_boltzmann_ic_muh`
(post-Picard) interpolates phi(y) using the BKSA composite-ψ
closure embedded in the C+D continuation. That closure is 1:1
symmetric (Gouy-Chapman cosh / BKSA log).

For 2:1 sulfate + 1:1 cation, the analytic shape is wrong, even if
the boundary scalars (ψ_D at OHP, ψ_S Stern drop, surface concs) are
correct. Newton then has to do extra work to repair the spatial
profile; in practice this often manifests as cold-fail.

**Fast-path fix options (pick one):**

A. **Numerical 1D PB pre-solve.** Solve the multi-ion 1D Poisson-
   Boltzmann ODE on the IC mesh once, using scipy.integrate.solve_bvp.
   Use that as the spatial phi(y) seed and let the Picard's surface
   scalars dictate the boundary conditions. ~3-5 days work.

B. **Linearized Debye seed.** Use phi(y) = ψ_D · exp(-y/λ_D_eff)
   with λ_D_eff = sqrt(ε / Σ z_i² c_b_i_phys). Loses the BKSA
   saturation tail in the strong-field regime but gives Newton a
   reasonable shape to start from. ~1 day work.

C. **Drop spatial IC entirely; use linear_phi.** The spatial profile
   is just a straight line phi_applied → 0; surface concs at bulk
   values. Worst seed but always available. The empirical failure
   here was at parallel topology + ClO₄⁻ surrogate; with multi-ion
   correct bulk concentrations, linear_phi may fail differently
   (or succeed where the wrong-shape composite-ψ failed).

**For the fast plan I propose Option B + warm-start from a freshly
solved legacy ctx (per H19 §3) as a fallback, in this order:**

1. Try linearized-Debye spatial IC + Picard scalars. (~1 day to
   implement.)
2. If that fails: warm-start from a legacy ClO₄⁻ + sequential ctx
   solved at the same V, then switch to multi-ion + parallel and
   re-solve. (~1 day to wire.)
3. If both fail: derive numerical PB pre-solve (Option A, ~3-5
   days).

Self-deception risk acknowledged: I was treating the spatial IC as
a black box ("Picard absorbs the mismatch") when in fact the
spatial profile is a separate first-class object the Picard doesn't
control.

### Issue 14 — Stern split is 1:1

**Accept.** `picard_ic.py:142-178` (`compute_surface_slope_signed`)
uses Gouy-Chapman / BKSA closures hardcoded for symmetric 1:1.
For 2:1, the GC first integral is asymmetric:

```text
|dφ/dy(0)|² ∝ Σ_k z_k² · c_b_k · (e^(-z_k·ψ_D) − 1) · (sign correction)
```

Multi-ion BKSA needs a separate derivation (with the Bikerman
denominator).

**Fast-path fix:** for the fast realignment, **disable Stern at IC
only** (`stern_split=None` in the Picard call). The Newton residual
still includes the Stern Robin BC; just the IC seed doesn't try to
split the layer. Document this as a transient crutch. The Stern
contribution is small relative to the diffuse layer at 0.55 nm Debye
length, so the IC mismatch is bounded.

### Issue 15 — Debye length `λ_D = sqrt(poisson_coefficient)` not
updated for I=0.3 M

**Need to verify.** Trace required:

- `poisson_coefficient` is computed in `scripts/_bv_common.py` from
  physical inputs `(ε_r, ε_0, RT/F², C_SCALE, L_REF)` per the nondim
  block. Must check whether changing `c_b` (Cs⁺=199.9, SO₄²⁻=100)
  forces a re-derivation of `poisson_coefficient` or whether the
  dimensional inputs remain anchored to the original
  electrolyte's `C_SCALE`.

**Provisional accept (action is verification, not coding change):**
add an explicit step in Phase 2.3:

> Print `poisson_coefficient`, `λ_D = sqrt(poisson_coefficient)` and
> the dimensional Debye length `λ_D · L_REF` after multi-ion
> wiring. Compare to the analytic
> `λ_D_phys = sqrt(ε_0·ε_r·R·T / (F² · Σ z_i²·c_b_i))` ≈ 0.55 nm
> for I=0.3 M, ε_r=80. If discrepant, fix the
> `poisson_coefficient` derivation in `_bv_common.py`.

Closing this issue requires confirming the math. Will report back
after Phase 2.3.

### Issue 16 — Anchor V backwards

**Accept.** Re-derived:

- E_eq_2e = 0.695 V; E_eq_4e = 1.23 V.
- For V_RHE in [-0.40, +0.55], both reactions are cathodic (η < 0).
- Weakest cathodic drive ⇒ V closest to E_eq from below.
- Within page-15 grid, V=+0.55 is the highest V, closest to E_eq_2e.
- V=+0.45 is *more* cathodic than +0.55, hence harder.

H19's "weak-reaction side" recommendation was probably written
without strict reference to E_eq; the fast plan should override.

**Fix:** §3 driver anchor_v_rhe = +0.55, walk +0.55 → +0.50 → … →
-0.40. §4 watch list updated accordingly. §5c V scan order: {+0.55,
+0.50, +0.45, +0.40, +0.35, +0.30, +0.20, 0.0}.

### Issue 17 — `K0_PHYS_R4E = K0_PHYS_R1` produces R_4e dominance

**Accept (with magnitude). Quantified:**

At V_RHE = +0.55, in nondim units (V_T = 0.025693 V):

- η_2e_model = (0.55 − 0.695) / 0.025693 = −5.64 (no clip)
- η_4e_model = (0.55 − 1.23)  / 0.025693 = −26.46 (no clip; clip=100)
- BV factor exp(−α·n·η):
    - 2e: exp(−0.627·2·(−5.64))  = exp(7.07)   ≈ 1.18e3
    - 4e: exp(−0.5·4·(−26.46))   = exp(52.9)   ≈ 9.4e22
- Ratio R_4e / R_2e in BV factor alone: ≈ 8e19.
- H⁺ correction: (H_HAT)² ≈ 6.94e-3 vs (H_HAT)⁴ ≈ 4.8e-5.
  Reduces ratio by ~144× → still ≈ 6e17.

R_4e dominates by ~17 orders of magnitude. Total disk current
saturates at the 4e Levich limit; gross R_2e is invisible noise.

**Fix to plan:** §3 driver runs THREE passes, not one:

1. **Pass A: pure-2e** (`K0_PHYS_R4E = 0`, k_R2e at literature).
   Confirms the structural plumbing (multi-ion + parallel topology
   + electron-weighted observables) works, with a clean gross R_2e
   curve to inspect.
2. **Pass B: pure-4e** (`K0_PHYS_R2E = 0`, k_R4e at literature).
   Confirms n_e_apparent = 4 and total disk current respects the
   4e Levich ceiling.
3. **Pass C: mixed** (both at literature). Acknowledged
   non-physical magnitude (R_4e dominates). LABEL the output as
   "structural validation only; not page-15 interpretable."

After fast realignment lands, calibrate K0_PHYS_R4E via Tafel-slope
fit to the experimental disk current — that's M4 in the long-form
plan, deferred per the fast plan.

### Issue 18 — Acceptance inconsistency

**Accept.** Phase 4 acceptance "≥1 V converges" is a bare smoke
test, not "page-15 sweep." The "Done" criterion "≥15/25 V" is the
real one.

**Fix:** Reconcile both to:

> **Phase 4 acceptance (smoke gate, must pass before Phase 5):**
> Pass A produces ≥ 5/25 V_RHE converged with non-trivial gross
> R_2e ≠ 0. (Ensures the multi-ion + parallel plumbing isn't
> silently broken.)
>
> **"Done" acceptance (single criterion):** Pass A + Pass B + Pass
> C all produce ≥ 10/25 V_RHE converged with the structural
> features described. No quantitative shape/magnitude bands. User
> iterates after.

(Lowered from 15/25 to 10/25 because three passes have to all clear
the bar; staffing 3 separate convergence regimes is harder than
one. Still substantive — 40% grid converged is meaningful.)

## Section 2 — Updated plan summary

Plan changes (will apply to
`docs/fast_realignment_plan_2026-05-08.md` after VERDICT APPROVED):

- **Phase 0** — unchanged.
- **Phase 1.1** — unchanged (disabled-rxn guard for `ln(k0)`).
- **Phase 1.2** — REPLACED. Was: "write minimal 2-rate parallel
  Picard." Now: "audit / instrument the existing
  `picard_outer_loop_general`; identify whether failure mode is
  Picard, post-loop reconstruction, or downstream Newton; fix
  *that* not write a new Picard." Drop the handwritten 2-rate math
  entirely (Issues 1-4 obsoleted).
- **Phase 1.3** — NEW (centralized disabled-reaction helper +
  strict topology predicate; Issues 6, 7).
- **Phase 2.1** — REPLACED scope. Drop "drop the single-counterion
  guard." Replace with: "implement `compute_surface_gamma_multiion`
  + `_solve_outer_phi_multiion` for two analytic counterions
  (Cs⁺ steric + SO₄²⁻ ideal)." Issues 11, 12.
- **Phase 2.2** — REPLACED `a_nondim` derivation: hard-sphere from
  hydrated radii, with Cs⁺ steric (Bikerman) + SO₄²⁻ ideal
  (unbounded Boltzmann). Issues 8, 9.
- **Phase 2.3** — Add explicit `poisson_coefficient` /
  `λ_D` print-and-verify step (Issue 15).
- **Phase 2.4** — REPLACED scope. Was "let Picard absorb shape
  mismatch." Now: linearized-Debye spatial IC seed + warm-start
  fallback from legacy ClO₄⁻ ctx. Issues 13, 14.
- **Phase 3** — Anchor V → +0.55 (Issue 16). Three-pass run
  structure (Pass A pure-2e, Pass B pure-4e, Pass C mixed; Issue 17).
- **Phase 4** — Acceptance: ≥ 5/25 V Pass A converged with gross
  R_2e ≠ 0. Pass B and Pass C plumbing-test only at this stage.
- **Phase 5** — `a_nondim` continuation starts at 0, not A_DEFAULT
  (Issue 10).
- **"Done" criterion** — Pass A + B + C all ≥ 10/25 V converged.
  (Issue 18.)

Estimated total: still 5-10 days, slightly shifted: Phase 1.2 is
shorter (audit, not new code) but Phase 2.1 + 2.4 are longer
(multi-ion gamma/phi_o + linearized-Debye spatial IC). Net wash.

## Section 3 — Continued critique prompt

Review the updated plan and my responses to your earlier issues.
Push back on responses where I defended poorly — name which point.
Raise any new issues the updated plan creates. Re-issue any earlier
issue you don't think I addressed. Same numbered format and same
verdict line at the end:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN
