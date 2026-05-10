# Option 2a′ — Bikerman-corrected IC for 4sp dynamic — implementation plan

**Date:** 2026-05-04
**Status:** ready to execute
**Background:**
* `docs/4sp_drop_boltzmann_investigation.md` (failure diagnosis)
* `docs/4sp_bikerman_ic_gpt_handoff.md` (initial proposal)
* `docs/4sp_bikerman_ic_gpt_assessment.md` (GPT review with two corrections)
* `docs/steric_sign_correction_plan.md` (the now-applied sign fix)

This plan implements the **GPT-corrected Option 2a′**: γ-correct every
species in the synthesised-4sp counterion branch of `_try_debye_boltzmann_ic`,
using `H_outer` as the local outer anchor for the charged pair, and
preserving `phi_init_expr`. The change is gated entirely on
`synthesised_4sp_counterion`; the 3sp+analytic-Boltzmann path is
byte-identical.

## 0. Goal in one sentence

Make the 4sp dynamic + `debye_boltzmann` IC seed a state with strictly
positive Bikerman packing at every node and `c_ClO₄ ≤ 1/a` at the
electrode, so that direct z=1 Newton converges at V_RHE = +0.3 V and
above instead of falling back to linear_phi.

## 1. Pre-flight (do these before writing any code)

### 1.1 ✅ Sign correction in `forms_logc.py`

Already applied at `forms_logc.py:266`:

```python
mu_steric = -fd.ln(packing)
```

Verified in this session.

### 1.2 ⚠ Sign status in other backends — do NOT touch in this plan

```
Forward/bv_solver/forms_logc.py        : -fd.ln(packing)  [corrected]
Forward/bv_solver/forms_logc_muh.py    : -fd.ln(packing)  [corrected]
Forward/dirichlet_solver.py:146        : +fd.ln(packing)  [OLD SIGN]
Forward/robin_solver.py:156            : +fd.ln(packing)  [OLD SIGN]
```

`dirichlet_solver.py` and `robin_solver.py` are the legacy 4sp
concentration-form path; the 3sp+Boltzmann production stack and the
4sp-dynamic path both run through `forms_logc.py`, so the legacy
sign mismatch does not block this work. **Leave them alone in this
plan.** Open a separate ticket / plan to fix or retire those two
files (they may be on the deletion list per the May 2026 cleanup).

### 1.3 Locate `a_vals` in `_try_debye_boltzmann_ic` scope

`solver_params` is unpacked at `forms_logc.py:49` as

```python
n_species, order, dt, t_end, z_vals, D_vals, a_vals, phi_applied, c0, phi0, params
```

so `solver_params[6]` is `a_vals`. The existing IC body already reads
`solver_params[4]` for `z_vals_full` (line 624), so the same indexing
pattern works.

### 1.4 Confirm `H_outer` is in scope at the seeding block

Yes — `H_outer` is defined at `forms_logc.py:790` (lines 790–793),
above the species-seeding block at lines 795–805. No refactor needed.

## 2. The math (final form)

### 2.1 γ definition (multispecies Bikerman, local-anchor)

```
γ(y) = 1 / [ 1
           + a_H    · H_outer(y) · (exp(−ψ(y)) − 1)
           + a_ClO₄ · H_outer(y) · (exp(+ψ(y)) − 1) ]
```

Two notes:

1. z=0 species (O₂, H₂O₂) drop out of the denominator because
   `exp(0) − 1 = 0`. They still get multiplied by γ in their c_i
   profile (so their occupancy contracts in the EDL), but they don't
   shape γ.
2. `H_outer(y)` is used as the outer anchor for **both** charged
   species. This is the consequence of outer electroneutrality
   propagated through `phi_o = ln(H_outer/c_ClO₄_bulk)`: the existing
   IC already places `c_ClO₄_outer(y) = H_outer(y)`. Using the same
   anchor in γ keeps the IC internally consistent.

### 2.2 Species seeding (final form)

```
u_0 = ln(O_outer)            + ln γ                        (O₂,    z=0)
u_1 = ln(P_outer)            + ln γ                        (H₂O₂,  z=0)
u_2 = ln(H_outer)      − ψ   + ln γ                        (H⁺,    z=+1)
u_3 = ln(c_ClO₄_bulk)        + phi_init_expr      + ln γ
    = ln(H_outer)        + ψ + ln γ                        (ClO₄⁻, z=−1)
u_n = phi_init_expr  =  ln(H_outer/c_ClO₄_bulk) + ψ        (φ; UNCHANGED — no ln γ)
```

φ is **not** modified. The γ correction lives entirely in the
concentration profiles; via `θ(ψ) = θ_bulk · γ`, the chemical
potential `ln(c_i) + z_i·φ − ln(θ)` has γ entering twice with
opposite signs and cancelling. Adding `ln γ` to φ would
double-count the steric correction. (This corrects an instinct
in the prior handoff — see `4sp_bikerman_ic_gpt_assessment.md`
"Stern compatibility".)

### 2.3 Behaviour summary at V_RHE = +0.3 V (ψ_D ≈ 11.5)

| species | c_i(0) | a · c_i(0) |
|---|---|---|
| O₂ | O_s · γ ≈ 0.7 · 2e-2 ≈ 1.4e-2 | 1.4e-4 |
| H₂O₂ | P_s · γ ≈ ~0 | ~0 |
| H⁺ | H_o · γ · e^(−ψ) ≈ ~0 | ~0 |
| ClO₄⁻ | H_o · γ · e^(+ψ) ≈ 98 | 0.98 |
| **Σ a_j c_j** | | **≈ 0.98** |
| **1 − Σ (= θ)** | | **≈ 0.02** |

Strictly positive packing margin (~2%). Newton-healthy.

## 3. Test-first sequencing (TDD per project rules)

Order:

1. Write/update tests **first**, run them — they should fail under the
   current code (because the existing IC is pure-Boltzmann seed).
2. Apply the code change.
3. Run the tests — they should now pass.
4. Run the regression gates.
5. Run the smoke solve.
6. Run the sweep.

### 3.1 Tests to update

**`tests/test_initializer_debye_boltzmann_4sp.py::test_ic_seeds_clo4_on_boltzmann_manifold`**

Currently asserts `max | u_3 − (ln c_ClO₄_bulk + ψ) | < 1e-6` (pure
Boltzmann). This will fail under the γ-corrected IC by design.
Replace with the assertions in §3.3.

### 3.2 Tests to add (new file `tests/test_steric_saturation.py`)

Five cases. All marked `@pytest.mark.slow` and `@skip_without_firedrake`
unless noted.

**P1 — total packing positive at every node.**
Run 4sp + `debye_boltzmann` IC at V_RHE = +0.3 V on small Ny (e.g. 80,
to keep the test under ~30 s). Extract `c_i(y)` from `U_prev` after
IC. Assert
```
all y: 1 − Σ_j a_j · c_j(y) > 1e-3
```

**P2 — bulk recovery: γ → 1, c_i → c_bulk_i at y=1.**
Same IC. At top-row nodes (y=1):
* `|c_O₂(1)   − c_O₂_bulk|   / c_O₂_bulk   < 1e-2`
* `|c_H₂O₂(1) − c_H₂O₂_bulk| / c_H₂O₂_bulk < 1e-2`
* `|c_H(1)    − c_H_bulk|    / c_H_bulk    < 1e-2`
* `|c_ClO₄(1) − c_ClO₄_bulk| / c_ClO₄_bulk < 1e-2`

**P3 — saturation visible at electrode.**
Same IC. At y=0:
* `c_ClO₄(0) ∈ [50, 1.0/a)` for `a = 0.01`. Lower bound rules out a
  γ-too-aggressive bug; upper bound rules out cap violation.

**P4 — symbolic γ sanity (fast, no Firedrake).**
Pure-Python, runs in `pytest -m "not slow"`. At a fixed `ψ = 11.5`
and bulk values, compute `γ` from the closed form and check:
* `γ(0) = 1` exactly,
* `γ(11.5) ∈ (0, 0.1)`,
* `γ(11.5) · c_ClO₄_bulk · exp(11.5) < 1.0/a`.

**P5 — 3sp+Boltzmann regression byte-identical.**
Reuses an existing 3sp + `debye_boltzmann` test fixture; asserts
`U_prev` after IC matches a pre-recorded snapshot to within
`rel_tol=1e-12` on every dof. Pre-record this snapshot **before**
making the code change, save as
`tests/fixtures/U_prev_3sp_debye_boltzmann_v0p1.npy`. Use a small
problem (Ny=64, V_RHE=+0.05) so the fixture is small.

### 3.3 Updated existing test assertions

In `tests/test_initializer_debye_boltzmann_4sp.py`:

* `test_ic_seeds_clo4_on_boltzmann_manifold` → rename to
  `test_ic_seeds_clo4_with_gamma_correction`. Assert
  ```
  c_ClO₄(y_node) ≈ H_outer(y_node) · γ(ψ(y_node)) · exp(+ψ(y_node))
  ```
  to FE interpolation tolerance (e.g. 1e-4 relative). Compute
  the RHS with the same closed form used by the IC.
* Other assertions in that file that check `c_H = H_outer · exp(−ψ)`
  pure-Boltzmann: update to multiply by γ.
* `TestRegression3spStillWorks::test_3sp_still_fires` should
  continue to pass byte-identically (the 3sp branch is gated out).

### 3.4 Regression gates (must continue to pass)

* `tests/test_initializer_debye_boltzmann.py` — 3sp+Boltzmann (3
  tests, slow). Byte-identical.
* `tests/test_solver_equivalence.py` — 4sp ↔ 3sp+Boltzmann at
  Ny=100, V_RHE ∈ [−0.5, +0.1]. The IC change applies to
  `debye_boltzmann`; equivalence test uses default `linear_phi`,
  so should be unaffected. **Run as a sanity check.**
* `tests/test_mms_convergence.py` — uses smooth manufactured field
  far from saturation. Steric residual term is small (γ ≈ 1).
  Expect h^p convergence within existing tolerance.
* `tests/test_stern_no_stern_snapshot.py` — pinned to 3sp+Boltzmann
  Stern. Should not change. Run as sanity.
* `tests/test_bv_common_config.py` — config wiring only. Should
  not change.

## 4. Code change

### 4.1 The diff

In `Forward/bv_solver/forms_logc.py`, replace the seeding block at
lines 795–806 with:

```python
# ----- Species seeding ---------------------------------------------
phi_init_expr = fd.ln(H_outer / fd.Constant(c_clo4_bulk)) + psi

if synthesised_4sp_counterion:
    # Multispecies Bikerman γ correction (matches the sign-corrected
    # residual in build_forms_logc — mu_steric = -ln(packing)).
    # Local outer anchor: H_outer for both charged species, since
    # outer electroneutrality plus phi_o = ln(H_outer/c_ClO4_bulk)
    # forces c_ClO4_outer = H_outer.
    a_vals_full = list(solver_params[6])
    a_h = float(a_vals_full[2])
    a_cl = float(a_vals_full[3])
    gamma_psi = fd.Constant(1.0) / (
        fd.Constant(1.0)
        + fd.Constant(a_h)  * H_outer * (fd.exp(-psi) - fd.Constant(1.0))
        + fd.Constant(a_cl) * H_outer * (fd.exp(+psi) - fd.Constant(1.0))
    )
    log_gamma = fd.ln(gamma_psi)

    U_prev.sub(0).interpolate(fd.ln(O_outer) + log_gamma)
    U_prev.sub(1).interpolate(fd.ln(P_outer) + log_gamma)
    U_prev.sub(2).interpolate(fd.ln(H_outer) - psi + log_gamma)
    U_prev.sub(n).interpolate(phi_init_expr)
    U_prev.sub(3).interpolate(
        fd.Constant(math.log(c_clo4_bulk)) + phi_init_expr + log_gamma
    )
else:
    # 3sp + analytic-Boltzmann counterion path: byte-identical to
    # the pre-2a′ behaviour.
    U_prev.sub(0).interpolate(fd.ln(O_outer))
    U_prev.sub(1).interpolate(fd.ln(P_outer))
    U_prev.sub(2).interpolate(fd.ln(H_outer) - psi)
    U_prev.sub(n).interpolate(phi_init_expr)

ctx["U"].assign(U_prev)
return True, "", picard_iters
```

### 4.2 Things explicitly NOT in scope

* No edit to `phi_init_expr` (φ stays as-is — no `ln γ`).
* No edit to the picard outer loop.
* No edit to `O_outer`, `P_outer`, `H_outer` definitions.
* No edit to `_try_debye_boltzmann_ic`'s preamble (Picard, GC ψ,
  outer linear envelopes).
* No edit to other backends (`dirichlet_solver.py`,
  `robin_solver.py`, `forms_logc_muh.py`).
* No edit to the 3sp+Boltzmann path at all.

### 4.3 Risk register

* **`a_h` or `a_cl` is zero in some configuration.** Then γ ≡ 1,
  `log_gamma ≡ 0`, and the IC reverts to pure-Boltzmann seeding.
  That's the right fallback (no Bikerman → no γ correction needed),
  so no special-casing required. But guard the test setup so P1–P3
  use a config with non-zero `a_h, a_cl`.
* **`H_outer` near zero at the surface (deep cathodic V?).** At
  cathodic V we don't expect ClO₄⁻ saturation, so `(exp(+ψ)−1)` is
  small and γ → 1 anyway. Still, sanity-check by adding `H_outer`
  floor of `1e-300` (already in the existing IC — line 790–793:
  `fd.max_value(... 1e-300)`).
* **FE interpolation on a piecewise-linear basis.** The symbolic
  IC satisfies `Σ a_j c_j ≤ 1` exactly; the nodal interpolant may
  not, especially in elements that span y∈[0, λ_D]. P1's `> 1e-3`
  margin is meant to absorb FE error; if it fails, refine Ny.

## 5. Smoke solve (after tests pass)

Single-voltage cold solve. Script (new):
`scripts/studies/smoke_4sp_bikerman_ic_v0p3.py`:

```
4sp dynamic + debye_boltzmann + Stern C_S=0.10 + V_RHE=+0.3 V + Ny=200
+ exponent_clip=50 + log-rate
```

Pass criteria:

1. `ctx["initializer_fallback"]` is False — IC fired and Newton at z=1
   converged from it (no `cold_z1_diverged`).
2. `surface_counterion_within_steric` is True (already a
   `Forward/bv_solver/diagnostics.py` flag).
3. CD at V=+0.3 within ±10% of the 3sp+Boltzmann + Stern reference
   from `StudyResults/peroxide_window_stern_test/` at the same
   voltage. (3sp+Boltzmann at this V has c_ClO₄ over the cap, so
   exact CD agreement is not expected; same order of magnitude is
   the right gate.)

If any of these fails, **stop and diagnose before running the
sweep**. Likely causes: (a) γ formula bug; (b) `H_outer` anchor
isn't right for ClO₄⁻ (revisit §2.1 with the actual numerical
profile); (c) ψ(y) in the EDL is too far from Bikerman (escalate
to Option 2b — composite asymptotic ψ).

## 6. Sweep validation (after smoke passes)

Re-run `scripts/studies/peroxide_window_4sp_extended.py debye_boltzmann`
on the V_RHE grid `[-0.5, -0.3, -0.1, 0.0, +0.1, +0.3, +0.5, +0.55,
+0.60, +0.65, +0.66, +0.68, +0.70, +0.75, +1.00]`. Compare against
the pre-2a′ artifact at
`StudyResults/peroxide_window_4sp_extended_debye_boltzmann/`.

Expected behaviour change:

| V_RHE | pre-2a′ | post-2a′ (target) |
|---|---|---|
| ≤ +0.1 V | converges 5/5 | converges 5/5, CD/PC unchanged within ~1% |
| +0.3 V | falls back to linear_phi | converges, c_ClO₄(0) ∈ [50, 100), packing > 1e-3 |
| +0.5 V | falls back | converges, saturating |
| +0.66 V (peroxide window centre) | falls back | converges or partial-z, document either |
| +1.0 V | falls back | best-effort; warm-walk should reach further than pre-2a′ |

Acceptance: at least V ∈ {+0.3, +0.5} converge with
`surface_counterion_within_steric = True` and direct z=1 (no
fallback). If V ≥ +0.7 still partial — that's the cue to escalate
to Option 2b (composite asymptotic ψ), per §8 of
`4sp_drop_boltzmann_investigation.md`. Not in scope here.

## 7. Acceptance criteria for merge

All of:

1. Regression gates (§3.4) pass.
2. New positive tests P1–P5 (§3.2) pass.
3. Smoke solve at V=+0.3 (§5) passes its three pass criteria.
4. Sweep at V ∈ {+0.3, +0.5} converges (§6).
5. `docs/4sp_drop_boltzmann_investigation.md` gets a "Resolution"
   section appended quoting the converged-V list and surface
   c_ClO₄ values from the sweep.
6. `docs/4sp_bikerman_ic_gpt_handoff.md` gets a one-line note at
   the top: "Superseded by `4sp_bikerman_ic_option_2a_plan.md` —
   §1.2 and §8 contained two errors; see GPT assessment doc for
   the corrections."

## 8. Rollback path

The diff is local to one block in `_try_debye_boltzmann_ic`.
Reverting:

1. Restore the pre-edit seeding block.
2. Revert the test changes (the snapshot fixture for P5 stays, since
   it's the pre-edit state).
3. Tests P1–P4 will fail under the revert; that's the correct signal
   that 2a′ is no longer in effect.
4. Open a new investigation if the failure mode at V=+0.3 returns.

## 9. Sequencing — concrete TODO order

```
[ ] 1. Pre-record P5 fixture (snapshot of 3sp+Boltzmann debye IC)
       BEFORE any code change.  Save as
       tests/fixtures/U_prev_3sp_debye_boltzmann_v0p1.npy.
[ ] 2. Update test_initializer_debye_boltzmann_4sp.py per §3.3
       (pure-Boltzmann assertions → γ-corrected assertions).  Run
       — these new assertions should FAIL on current code.
[ ] 3. Write tests/test_steric_saturation.py with P1–P4.  Run —
       slow tests should fail; P4 (fast, pure-Python) should pass
       since it doesn't touch the code.
[ ] 4. Apply the §4.1 diff to forms_logc.py.
[ ] 5. Run the failing tests from steps 2–3.  All should pass.
[ ] 6. Run regression gates (§3.4).  All should pass.  If P5
       regression fails — STOP, debug, do not proceed.
[ ] 7. Write scripts/studies/smoke_4sp_bikerman_ic_v0p3.py.
[ ] 8. Run smoke (§5).  Assert all three pass criteria.
[ ] 9. Run sweep (§6).  Compare to pre-2a′ artifact.
[ ] 10. Append "Resolution" sections to investigation doc and
        handoff doc (§7 step 5–6).
[ ] 11. Commit:
          test: gamma-corrected IC assertions, steric saturation tests
          feat: option 2a' bikerman gamma in 4sp debye_boltzmann IC
          docs: resolution of 4sp ic at high anodic V
        (Three commits, atomic per CLAUDE.md project rules.)
```

## 10. Open questions deferred until after sweep

These are NOT blockers for this plan; record answers in a follow-up
doc once 2a′ is in.

1. **Does V ≥ +0.7 converge under 2a′ alone, or is Option 2b
   needed?** Empirical question — the §6 sweep answers it.
2. **What does the inverse pipeline look like with a converging
   4sp + Bikerman + Stern stack?** Out of scope; the production
   inverse uses 3sp+Boltzmann.
3. **Should the `dirichlet_solver.py` / `robin_solver.py` sign be
   fixed too, or are those backends slated for deletion?** Open a
   separate plan.
4. **Should the same γ-corrected IC be added to
   `forms_logc_muh.py`?** That backend uses electrochemical
   potential as the primary variable; the IC structure differs.
   Out of scope here.

## 11. Pointers

* Code (target of the change): `Forward/bv_solver/forms_logc.py`
  — IC body lines 571–807, edited block at lines 795–806.
* Solver param indexing: `solver_params[4] = z_vals`,
  `solver_params[6] = a_vals` (see line 49).
* Sign-corrected residual: `forms_logc.py:266`
  (`mu_steric = -fd.ln(packing)`).
* Existing 4sp IC test:
  `tests/test_initializer_debye_boltzmann_4sp.py`.
* New test file: `tests/test_steric_saturation.py` (P1–P4).
* Pre-record fixture: `tests/fixtures/U_prev_3sp_debye_boltzmann_v0p1.npy`.
* Smoke script (new):
  `scripts/studies/smoke_4sp_bikerman_ic_v0p3.py`.
* Sweep script (existing):
  `scripts/studies/peroxide_window_4sp_extended.py`.
* Reference artifact (pre-fix):
  `StudyResults/peroxide_window_4sp_extended_debye_boltzmann/`.
* Borukhov 1997, Bazant 2009: see citations in
  `docs/steric_sign_correction_plan.md` §8.
