# Handoff #13 — Response to Codex's review of Handoff #12

**Date:** 2026-05-07  ·  **Status:** Critique-of-the-critique, awaiting second review  ·  **Author:** Claude (with Jake)

This is a short follow-up to:
* `docs/CHATGPT_HANDOFF_12_IC_PICARD_BUGS.md` — original diagnosis of two
  IC bugs in the BV-PNP forward solver (Stern-eta mismatch and
  Bikerman-γ Picard mismatch).
* `docs/CODEX_REVIEW_HANDOFF_12_IC_PICARD_BUGS.md` — Codex's review of
  Handoff #12, including a substantive pushback on my "drop γ from IC"
  Fix A.

We've digested Codex's review, conceded the main correction, and want
a second pass to check (a) that we read the concession correctly and
(b) one residual concern about how to sequence verification.

## 1. What Codex pushed back on, and why we accept it

My original Handoff #12 ranked three fixes, with **Fix A (drop
`+ log_gamma` from bikerman IC seeds)** as cheapest. The empirical
support was: at V_RHE = +0.5 V, no-Stern, the *legacy ideal-counterion
γ-free path* gives `||F||(U_ic) = 0.81` — essentially a SS — while the
*bikerman γ-shifted* path gives `||F|| = 1.11×10³`. I read this as:
"the legacy IC was much closer; bikerman extension introduced the
regression; reverting fixes it."

Codex's pushback (their §"Why I would not drop log_gamma"):

> The current residual-side Bikerman closure is *not* gamma-free. It
> uses dynamic concentrations in the total packing `A_dyn = Σ aᵢcᵢ`
> and in the steric flux. A matched Bikerman zero-flux IC must use
> `cᵢ = c_outer_i · γ(ψ) · exp(−zᵢ·ψ)`. For neutral species this is
> `c_outer_i · γ`. So `log_gamma` on O₂ and H₂O₂ is not obviously a
> bug — it's the expected steric chemical-potential offset. Dropping
> it may recover a small no-Stern residual at moderate V, but that
> appears to be because it reverts the IC toward the older
> ideal-counterion manifold, not because it matches the current
> Bikerman residual.

We agree, and on closer inspection of our own diagnostic, **the
"legacy path is close to SS" datum is a configuration artefact**, not
evidence that γ-free IC matches bikerman residual:

* The legacy IC code path (`forms_logc_muh.py:1025-1041`,
  `forms_logc.py:942-949`) only fires when
  `apply_bikerman_ic = synthesised_4sp_counterion or
  bikerman_in_counterions` is False.
* Our "legacy γ-free" diagnostic was run with
  `DEFAULT_CLO4_BOLTZMANN_COUNTERION` (no `steric_mode='bikerman'`)
  rather than `DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC`.
* So that residual is the *ideal*-counterion bikerman residual — the
  bikerman closure was disabled for both IC and residual. IC and
  residual are internally consistent there, but it's a *different
  stack* than production (3sp + analytic Bikerman ClO₄⁻ + Stern).

We can't use a no-bikerman-residual run as evidence that a γ-free IC
matches the bikerman-residual production stack. **Fix A is rejected.**

## 2. Adopted plan (Codex's recommended order)

1. **γ-aware Picard.** Keep `+ log_gamma` in the IC seeds. Update the
   Picard outer loop's surface BV rate calculation to use the same
   γ-shifted reaction-plane concentrations the residual sees:

   ```
   γ_s   = 1 / (1 + a_h·H_o·(e^{-ψ_D}-1) + a_cl·c_clo4_bulk·(e^{+ψ_D}-1))
   log_O_rxn = log(O_s) + log(γ_s)
   log_P_rxn = log(P_s) + log(γ_s)
   log_H_rxn = log(H_o) - ψ_D + log(γ_s)
   ```

   Then build `A1, B1, A2` from these reaction-plane logs (rather than
   from `H_s = H_o · e^{-ψ_D}` and gamma-free `O_s, P_s`). Codex flags
   a double-count risk: `_h_factor_log(H_s, factors)` already folds in
   the H⁺ stoichiometric factor; if we *also* add `log_gamma` outside
   it, we double-count. Cleanest refactor is to pass `log_H_rxn` into
   the helper directly so it does the species-2 sum on top of the
   already-γ-shifted log.

2. **Stern split in IC + Picard eta.** When Stern is active, solve

   ```
   ψ_total = phi_applied − phi_o
   ψ_total = ψ_D + ψ_S
   stern_coeff · ψ_S = poisson_coeff · surface_slope(ψ_D)
   ```

   for ψ_S, ψ_D at IC-build time. The `surface_slope(ψ_D)` should be
   the BKSA `α_d` factor already computed at
   `forms_logc_muh.py:979-982` (or the GC first-integral analogue for
   the legacy ideal branch). Then anchor IC at
   `φ_surface = phi_applied − ψ_S` rather than `phi_applied`, and use
   `η = bv_exp_scale · (phi_applied − φ_surface − E_eq)` in the
   Picard, matching the residual.

3. **Linear-φ fallback also Stern-aware.** Same Stern-split helper,
   bulk outer values. Prevents fallback rows from all seeing
   `η = −E_eq`.

4. **Cathodic Picard limit cycle.** Separate problem. Address only
   after 1–3 demonstrably collapse `||F||(U_ic)` in the
   converged-Picard regime.

## 3. One augmentation Codex's plan doesn't fully resolve

Codex's Fix 1 (γ-aware Picard) is theoretically the right physical
patch. Empirically, here's the gap we'd like reviewer input on:

The diagnostic data (Handoff #12 §6.1) shows that with the *current*
production stack (bikerman + Stern, Picard converges at V_RHE ≥
+0.3 V), `||F||(U_ic)` is in the `10³`–`10⁴` band at converged-Picard
voltages, and the **dominant residual block is u_O2** (the surface
species residual). The hypothesised cause is the γ-Picard mismatch —
factor `γ(0) ≈ 10⁻⁹` at moderate ψ_D explains a multiplier-of-10⁹
mismatch in the surface BV rate, which roughly fits the observed `10³`
residual after assuming surface integration drops a few orders.

We want to confirm the dominant residual really is the γ mismatch
*before* layering Fix 2 on top. The cheap test:

* Implement Fix 1 alone, verify on a single voltage pair:
  * `V = +0.5 V`, no Stern, bikerman: predicted to collapse from
    `||F|| = 1.11×10³` to roughly `||F|| ≲ 10⁰`.
  * `V = +0.5 V`, with Stern, bikerman: predicted to remain
    `~10³` (Bug #1 still present).
* If both predictions hold, Fix 1 is doing what we expect and Fix 2
  is well-isolated. If the no-Stern post-Fix-1 residual is *not* near
  `10⁰`, there's a third gap (or our γ-block-magnitude estimate is
  off by orders of magnitude, which would itself be a clue).

We'd like the reviewer to weigh in on:

* Is this single-V "predicted collapse" the right empirical bisection
  for the bug isolation? Or is there a tighter scalar/per-block check
  that would isolate γ-mismatch from any other contribution?
* Codex's test plan calls for a scalar Picard helper unit test (γ
  helper, Stern-split helper, rate logs) before Firedrake runs. Are
  there scalar identities we should be testing that would catch
  off-by-one errors in the γ insertion (e.g., neutral species get one
  γ, charged species get one γ × Boltzmann)?

## 4. Smaller items in Codex's review we accept verbatim

* **Fast tests before Firedrake.** Pure-Python scalar tests for the γ
  helper, Stern-split helper, and Picard rate logs. We hadn't planned
  this in Handoff #12; it's a clear improvement.
* **Cathodic limit cycle is a separate issue.** We'd been considering
  bundling it in. Codex is right that the H_o → 1e-300 floor smoothing
  is a *diagnostic* — it tells us whether the fixed point exists at
  all — and that the production fix needs Anderson or a log-domain
  reformulation, not floor smoothing.
* **Bikerman vs ideal closure for `surface_slope(ψ_D)`.** We had
  flagged this as an open question. Codex's answer is unambiguous:
  bikerman branch uses BKSA `α_d`; ideal branch uses GC first
  integral. This matches our reading and resolves the question.

## 5. Open questions for second review

1. Is our concession (rejecting Fix A) correct, given that the
   "legacy γ-free path" we cited as evidence was running against an
   *ideal*-counterion residual, not a bikerman one?

2. Is the V=+0.5 V single-point bisection (§3 above) a sound enough
   test of Fix 1 alone to justify proceeding to Fix 2? Or should we
   demand that both no-Stern + Picard-converged voltages drop into
   the `||F|| ≲ 10⁰` band before believing Fix 1 worked?

3. Codex's `_h_factor_log` double-count warning is well-taken. Are
   there other places in the Picard where γ may need to enter and
   we'd risk double-counting (e.g., the `rhs1, rhs2` construction at
   line 880-881, the `H_s` definition that drives `log_h_factor1`)?

4. With a γ-aware Picard, the surface gamma `γ_s = γ(ψ_D)` becomes
   another iterated quantity. ω = 0.5 fixed-point damping currently
   limit-cycles with `delta = 2` at cathodic V. Adding γ_s (which
   can range from ~1 in the bulk to `1e-9` at saturated cathodic ψ_D)
   could plausibly *worsen* the cathodic limit cycle. Should the γ
   update inside Picard be its own under-relaxation, or should we
   couple it with Anderson before the limit cycle becomes worse?

## 6. References

* Original handoff: `docs/CHATGPT_HANDOFF_12_IC_PICARD_BUGS.md`
* Codex review: `docs/CODEX_REVIEW_HANDOFF_12_IC_PICARD_BUGS.md`
* Picard outer loop: `Forward/bv_solver/forms_logc_muh.py:861-908`
* IC seeds (bikerman branch): `Forward/bv_solver/forms_logc_muh.py:996-1024`
* Residual eta selection: `Forward/bv_solver/forms_logc_muh.py:298-308`
* Bikerman residual closure: `Forward/bv_solver/boltzmann.py:215-225`
* Diagnostic script: `scripts/diagnose_db_ic_distance.py`
* Diagnostic outputs:
  * `StudyResults/diagnose_db_ic_distance_logc_muh/` (bikerman + Stern)
  * `StudyResults/diagnose_db_ic_distance_logc_muh_nostern/` (bikerman, no Stern)
  * "Legacy γ-free at no-Stern + ideal counterion" was an inline check
    earlier in the conversation — see Handoff #12 §6.4. **Note for
    reviewer**: that data was generated against an *ideal*-counterion
    residual, not a bikerman residual; see §1 above for why this
    makes the comparison invalid as evidence for Fix A.
