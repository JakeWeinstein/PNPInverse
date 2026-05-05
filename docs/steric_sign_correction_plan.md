# Steric sign correction — math, code change, and testing plan

**Date:** 2026-05-04
**Background:** see `docs/4sp_drop_boltzmann_investigation.md` §6, the
follow-up exchange with the human reviewer in
`docs/4sp_bikerman_corrected_ic_review.md`, and the literature
confirmation in `docs/4sp_drop_boltzmann_investigation.md` end-of-doc.

The implemented modified Nernst-Planck steric term in
`Forward/bv_solver/forms_logc.py` and the corresponding definition in
`docs/PNP Equation Formulations.tex` line 152 carry the **opposite
sign** from the standard Bikerman / lattice-gas modified
Poisson-Boltzmann (MPB) framework, as derived in Borukhov-Andelman-Orland
(1997) and Bazant-Kilic-Storey-Ajdari (2009). Behavioural consequence:
no equilibrium SS exists for the dynamic counterion above
`ψ_D ≈ ln(1/(4·a·c_bulk)) ≈ 4.83` nondim ≈ +0.124 V vs RHE — exactly
the point where the 4sp warm-walk has been failing.

This plan documents the math, prescribes the code change, and lists
the regression and positive tests required before the fix can ship.

## 1. Variational derivative — what it is and how it gives the sign

### 1.1 The concept

An ordinary derivative `df/dx` measures how a *number* `f(x)` responds
to a small nudge in another *number* `x`. A **variational derivative**
generalises this to *functionals* — quantities like the free energy
`F[c]` that take an entire function `c(r)` as input and return a
number. Notation: square brackets `F[c]` flag that the argument is a
function, not a value at a point.

The variational derivative `δF/δc(r)` answers the question:

> *If I add a tiny localised bump to `c` at the single point `r`, by
> how much does `F` change per unit bump amplitude?*

Formally, with `δ_r(·)` a delta-function bump at `r`,

```
F[c + ε · δ_r] − F[c]  =  ε · (δF/δc)(r)  +  O(ε²).
```

The result `δF/δc(r)` is itself a function of `r`, evaluated point by
point. Equilibrium of an open system requires `δF/δc(r) = μ` (constant
chemical potential everywhere) — that's how variational derivatives
generate equilibrium distributions.

### 1.2 Computing it for an integral functional

The Bikerman free energy has the structure

```
F[c⁺, c⁻, ψ] = ∫ ℒ(c⁺(r), c⁻(r), ψ(r), ∇ψ(r)) dr
```

— the integrand `ℒ` depends only on local values (and gradients of `ψ`,
which we won't need for the steric piece). When `ℒ` doesn't contain
gradients of `c`, the variational derivative reduces to an ordinary
partial derivative of the integrand:

```
δF/δc⁺(r)  =  ∂ℒ/∂c⁺  evaluated at the values c⁺(r), c⁻(r), ψ(r).
```

So the steric piece of the chemical potential comes from
*differentiating the entropy density at a point with respect to `c⁺`*.

### 1.3 Applying it to Borukhov-Andelman-Orland eq (2)

The entropy density (per unit volume) in their eq (2):

```
σ(c⁺, c⁻) = (k_B T / a³) · [ c⁺a³·ln(c⁺a³)
                            + c⁻a³·ln(c⁻a³)
                            + (1 − Φ)·ln(1 − Φ) ]      with  Φ = a³(c⁺ + c⁻)
```

is the integrand of `−TS`. The functional we differentiate is `F = U − TS`,
which contributes `+σ` to the integrand of `F`.

Take `∂σ/∂c⁺` term by term:

1. `∂/∂c⁺ [c⁺a³·ln(c⁺a³)] = a³·(ln(c⁺a³) + 1)`
2. `∂/∂c⁺ [c⁻a³·ln(c⁻a³)] = 0`  (no c⁺ dependence)
3. `∂/∂c⁺ [(1 − Φ)·ln(1 − Φ)] = (∂Φ/∂c⁺) · d/dΦ[(1 − Φ)·ln(1 − Φ)]
                            = a³ · [ −ln(1 − Φ) − 1 ]
                            = −a³·(ln(1 − Φ) + 1)`

Summing and multiplying by `(k_B T / a³)`:

```
∂σ/∂c⁺  =  k_B T · [ ln(c⁺a³) + 1 − ln(1 − Φ) − 1 ]
        =  k_B T · [ ln(c⁺a³) − ln(1 − Φ) ]
        =  k_B T · ln(c⁺a³)            ← ideal part
           +  ( −k_B T · ln(1 − Φ) )    ← excess part, "μ⁺ steric"
```

The `+1`s from the differentiations cancel exactly. The remaining
`−ln(1 − Φ)` is the **excess chemical potential**, with an explicit
**minus sign**. This is the sign that produces saturation.

Adding the U contribution `eψ − μ⁺` and setting the total variation to
zero gives Borukhov eq (3) directly:

```
μ⁺ = eψ + k_B T · [ ln(c⁺a³) − ln(1 − Φ) ]
```

— the "ideal Boltzmann" piece `(k_B T · ln(c⁺a³) + eψ)` plus
`(−k_B T · ln(1 − Φ))`. Bazant eq (20) writes the second term as
`μ⁺_ex = −k_B T · ln(1 − Φ)`, identical to Borukhov.

The single minus sign comes from differentiating
`(1 − Φ) · ln(1 − Φ)` once; there is no opportunity for double-negation
in this derivation. Anyone who can compute the variational derivative
of eq (2) gets the same answer.

## 2. The implemented form is opposite-sign

The current code in `Forward/bv_solver/forms_logc.py`:

| line | code | meaning |
|---|---|---|
| 266 | `mu_steric = fd.ln(packing)` | `μ_steric = +ln(1 − Φ)` |
| 290 | `drift = em · z · phi` | electromigration drift |
| 293 | `Jflux = D · c · (∇u + ∇drift + ∇mu_steric)` | flux uses `+∇mu_steric` |

Equivalent chemical potential: `μ_total = ln(c) + z·φ + ln(1 − Φ)`,
which has a `+ln(1 − Φ)` excess. **Opposite sign from
Borukhov/Bazant/standard MPB.**

The accompanying writeup at `docs/PNP Equation Formulations.tex`
line 152:

```
μ^{steric}(c) = k_B T · ln(1 − Σ_j a_j c_j)
```

— same wrong sign. The implementation faithfully matches the writeup;
the writeup itself is the source of the inversion. (See
`docs/4sp_drop_boltzmann_investigation.md` for the full record.)

## 3. The fix

### 3.1 Code change

**One-character change** at `Forward/bv_solver/forms_logc.py:266`:

```diff
-    mu_steric = fd.ln(packing)
+    mu_steric = -fd.ln(packing)
```

(Equivalent alternative: keep line 266 as-is and flip the sign at
line 293 from `+ fd.grad(mu_steric)` to `− fd.grad(mu_steric)`. The
proposed change keeps `Jflux = D·c·(∇u + ∇drift + ∇mu_steric)` as the
"all gradients positive" canonical form, which is more consistent with
how textbooks write the modified NP flux.)

This change is gated entirely by `if steric_active` at line 260, which
only fires when at least one `a_vals_hat[i]` is nonzero. The 3sp
production preset has `a_vals_hat = [0.0]*3`, so `steric_active = False`
and the 3sp path is **not exercised by this code** — the change is a
no-op for everything except 4sp dynamic.

### 3.2 Writeup change

**Two-line change** at `docs/PNP Equation Formulations.tex`:

Line 151–153 should read:

```latex
\mu^{\mathrm{steric}}(c) = -k_B T \ln\left(1 - \sum_j a_j\, c_j\right),
```

(adding the leading `-`).

Equation 15 (`eq:modified_NP`, lines 158–160) keeps the `+
μ^{\mathrm{steric}}(c)` term, since the negative sign is now baked into
the definition.

A short clarifying paragraph is worth adding pointing to
Borukhov-Andelman-Orland (1997) eq (3) and Bazant-Kilic-Storey-Ajdari
(2009) eq (20) for the standard form.

**This is the change that needs explicit advisor sign-off.** The
writeup is the project's specification document; the code change is
ratifying what the writeup says. Get advisor agreement on the writeup
edit *before* committing the code edit.

### 3.3 Order of operations

1. **Discuss with advisor** — bring the math derivation in §1 above and
   the cross-table from `docs/4sp_drop_boltzmann_investigation.md`
   (Borukhov '97 eq (2), Bazant '09 eq (20)). Get explicit agreement
   that the writeup definition should be `−ln(1 − Φ)`.
2. **Update the writeup** — single-line edit to line 152, plus a
   citation note pointing to the two literature references.
3. **Update the code** — single-character edit to `forms_logc.py:266`.
4. **Run the regression suite** — see §4.1.
5. **Add the new positive tests** — see §4.2.
6. **Re-run the 4sp + debye_boltzmann + Stern sweep** — see §5.
7. **Update the investigation log** — append a "Resolution" section to
   `docs/4sp_drop_boltzmann_investigation.md` quoting the verdicts of
   the new tests and the new sweep.

## 4. Testing requirements

### 4.1 Regression gates (must still pass after the change)

**Gate R1 — 3sp+Boltzmann path strictly unchanged.** The change only
modifies behaviour when `steric_active` is True (i.e., when some
`a_i ≠ 0`). The 3sp production preset has `a_vals_hat = [0.0]*3`, so
`steric_active = False` and the modified line is dead code for that
path. The following tests must pass byte-identically (no tolerance
slack, exact reproduction):

- `tests/test_initializer_debye_boltzmann.py` (3 tests, slow)
- `tests/test_initializer_debye_boltzmann_4sp.py::TestRegression3spStillWorks::test_3sp_still_fires`
- `tests/test_stern_no_stern_snapshot.py` (2 tests, slow) — pinned to the
  baseline at V_RHE=0.66 within `rel_tol=1e-6`
- `tests/test_bv_common_config.py` (12 tests, fast) — config-wiring only

If any of these fail, the change has unintended scope; investigate
before proceeding.

**Gate R2 — MMS convergence rates unchanged.** The manufactured solution
in `tests/test_mms_convergence.py` and `scripts/verification/mms_*.py`
is a smooth `c_i = c_0 · (1 + 0.3·cos(πx)·(1−y)²)` field that does not
approach saturation. The steric term `±ln(1 − Φ)` is small and bounded
on this manufactured state regardless of sign. So the MMS test should
still produce h^p convergence within its existing tolerance. Allow up
to a 5% drift in measured convergence rate; a larger drift indicates
an unintended interaction.

**Gate R3 — 4sp equivalence test still passes.** `tests/test_solver_equivalence.py`
runs 4sp dynamic vs 3sp+Boltzmann at Ny=100, V_RHE ∈ [−0.5, +0.1]. At
these voltages, `ψ ≤ 3.9 nondim`, well below `ψ_crit = 4.83` of the
old sign. Both signs give `μ_steric ≈ 0` to within 5% in this range,
so the equivalence test should still pass. Tolerance currently in the
test may need to widen by a small factor (1–2× current); document
any change.

### 4.2 Positive tests for the fix (would have failed under the old sign)

**Test P1 — saturation at high anodic V (NEW).** Add to a new file
`tests/test_steric_saturation.py`:

```python
@skip_without_firedrake
@pytest.mark.slow
def test_4sp_clo4_saturates_at_steric_cap():
    """At V_RHE = +0.3 V on a small problem, the SS surface c_ClO4
    should be bounded by 1/a (= 100 for a=0.01), not diverge.
    Pre-fix, this voltage was above ψ_crit = 4.83 and Newton diverged
    (no SS exists).  Post-fix, the conventional Bikerman saturation
    holds and Newton converges with c_ClO4_surf ≤ 1/a."""
    # Build 4sp + debye_boltzmann + Stern at V=+0.3, run cold solve.
    # Assert: ctx solves; max(c3_surface_mean) <= 1.0 / 0.01 + tol
    # Assert: max(c3_surface_mean) >= 50 (saturation should be visible)
```

**Test P2 — analytical Bikerman distribution match (NEW).** Add to
`tests/test_steric_saturation.py`:

```python
@skip_without_firedrake
@pytest.mark.slow
def test_clo4_matches_borukhov_eq5():
    """Drop the BV reaction terms and reactive species; just solve
    Poisson + NP for ClO4- (z=-1) at fixed phi at the boundary.
    Compare resulting c_ClO4(y) to Borukhov-Andelman-Orland 1997 eq (5)
    (Fermi-Dirac saturating distribution).  At psi_D = 5, converged
    c_ClO4 at the electrode should match 1/(a^3 * (1 + (1-phi0)/phi0 *
    exp(-z*beta*e*psi))) within a few percent FE error."""
```

**Test P3 — sign sanity (NEW, fast).** Add to a fast test file:

```python
def test_mu_steric_sign_at_saturation():
    """Symbolic sanity: the variational derivative of (1-Phi)·ln(1-Phi)
    is -(ln(1-Phi)+1), so the chemical potential gets a -ln(1-Phi)
    contribution.  At Phi = 0.95 (near saturation), mu_steric should be
    large and POSITIVE (repulsive), not negative."""
    import math
    phi = 0.95
    mu_old_sign = math.log(1 - phi)        # -3.0  (wrong sign)
    mu_new_sign = -math.log(1 - phi)       # +3.0  (right sign)
    assert mu_new_sign > 0
    assert mu_old_sign < 0
    # And the new sign should grow large positive as packing fills:
    assert -math.log(1 - 0.999) > -math.log(1 - 0.95)
```

This is a tiny test but it documents the intent — anyone looking at
the test sees the sign convention spelled out symbolically.

### 4.3 Sweep validation

**Sweep S1 — re-run `peroxide_window_4sp_extended.py` with
`debye_boltzmann` initializer** (see `StudyResults/peroxide_window_4sp_extended_debye_boltzmann/`
for the pre-fix run). Expected behaviour change:

| V_RHE | pre-fix | post-fix |
|---|---|---|
| ≤ +0.1 V | converges (5/5) | converges (5/5), CD/PC unchanged within ~1% |
| +0.3 V | diverges at z=1 (orchestrator falls back) | should converge with surface c_ClO4 ≤ 1/a |
| +0.5–0.7 V | diverges, no SS | should converge, c_ClO4 saturating |
| +1.0 V | diverges | best-effort; warm-walk should reach |

Acceptance criterion: at least V ∈ {+0.3, +0.5, +0.66, +0.68} V converge
with `surface_counterion_within_steric = True` (already a flag in
`Forward/bv_solver/diagnostics.py:collect_diagnostics`). If V ≥ +0.7
still fails, that's a separate investigation about the IC's `ψ(y)`
profile (composite asymptotic, Option 2b in the prior plan), not the
sign fix.

**Sweep S2 — Stern-test sweep re-run.** `peroxide_window_stern_test.py`
is on the 3sp+Boltzmann path, which is unaffected by this change. No
re-run needed; existing artifacts in
`StudyResults/peroxide_window_stern_test/` remain valid.

### 4.4 What NOT to test for

- **3sp+Boltzmann observable preservation under the change.** This is
  a dead code path for the 3sp preset; expect zero diff. If 3sp results
  change, the fix has wider scope than intended and needs separate
  investigation.
- **Inverse-pipeline regression at every voltage.** The inverse
  pipeline uses 3sp+Boltzmann, which is unaffected. Inverse fits at
  V ∈ [−0.5, +0.1] are not perturbed.

## 5. Acceptance criteria

The change can be merged when:

1. All gates in §4.1 pass (R1, R2, R3).
2. New positive tests P1, P2, P3 pass.
3. Sweep S1 shows convergence at V ∈ {+0.3, +0.5, +0.66, +0.68} V with
   `surface_counterion_within_steric = True`.
4. The investigation log (`docs/4sp_drop_boltzmann_investigation.md`)
   has a "Resolution" section appended quoting the verdicts.
5. Advisor has signed off on the writeup change (per §3.3 step 1).

## 6. Roll-back path

The code change is one character; the writeup change is one minus
sign. Reverting both is trivial. The new tests P1-P3 would fail under
revert, but that is the correct signal: the old sign is unphysical.

## 7. Open questions for advisor discussion

1. **Was the `+ln(1 − Φ)` sign in the writeup intentional?** If so,
   what physical model does it correspond to? (The behaviour — phase
   transition at moderate ψ, no SS above ψ_crit — is not
   physically standard for an electrolyte with hard-sphere ions, so
   intentional non-standard derivation seems unlikely, but worth
   asking.)
2. **Are there other writeups (older drafts, related notes) that use
   the conventional sign?** If yes, the inversion may be a recent
   editing accident in `docs/PNP Equation Formulations.tex` rather
   than a long-standing model choice.
3. **Has the steric term ever been validated against analytical
   Bikerman saturation in a separate forward problem?** If a
   validation case exists somewhere in the project history with
   `c_ClO4 → 1/a` at high ψ, that's strong evidence the implemented
   model used to have the conventional sign and was inverted later.
4. **Does the inverse-pipeline TRUE-parameter cache rely on numerical
   values that would shift?** Per §4.1 R3, equivalence-test tolerance
   may need a small (~5%) widen — confirm whether any downstream
   scripts hard-code values from the 4sp path.

## 8. Pointers

- Code: `Forward/bv_solver/forms_logc.py:260-300` (steric block)
- Spec: `docs/PNP Equation Formulations.tex:151-153, 158-160`
- Investigation: `docs/4sp_drop_boltzmann_investigation.md`
- External review: `docs/4sp_bikerman_corrected_ic_review.md`
- Literature:
  - Borukhov, Andelman, Orland (1997) *PRL* 79, 435.
    arXiv:cond-mat/9803258. Free-energy eq (2), modified PB eq (3),
    saturating distribution eq (5).
  - Bazant, Kilic, Storey, Ajdari (2009) *Adv. Colloid Interface Sci.*
    152, 48. arXiv:0903.4790. Excess chemical potential eq (20),
    Fermi-Dirac equilibrium eq (21), Bikerman MPB eq (22).
- Tests touched (regression):
  - `tests/test_initializer_debye_boltzmann.py`
  - `tests/test_initializer_debye_boltzmann_4sp.py`
  - `tests/test_stern_no_stern_snapshot.py`
  - `tests/test_bv_common_config.py`
  - `tests/test_solver_equivalence.py`
  - `tests/test_mms_convergence.py`
- Tests added (positive):
  - `tests/test_steric_saturation.py` (new file with P1, P2)
  - sign-sanity P3 — pick a fast test file to extend
- Sweep to re-run: `scripts/studies/peroxide_window_4sp_extended.py debye_boltzmann`
