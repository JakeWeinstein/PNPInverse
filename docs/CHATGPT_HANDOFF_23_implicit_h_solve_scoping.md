# CHATGPT_HANDOFF_23 — Implicit H+ Boundary Solve: Scoping Against Existing Phase-5γ MVP

Date: 2026-05-09
Author: Claude (post-HANDOFF_22 scoping pass)
Purpose: ask GPT to find holes in the implicit-H_o boundary solve plan
*before* I implement, given that a non-trivial Phase-5γ MVP is already in the
repo (untracked) and HANDOFF_22's Action 1 has not been started.

## Source Context (read in this order)

- `docs/CHATGPT_HANDOFF_22_realigned_problem_analytic_convergence.md`
  — the analytic / dominant-balance argument I am scoping.
- `Forward/bv_solver/anchor_continuation.py` (untracked) — the existing
  Phase-5γ MVP (k0 ladder + AdaptiveLadder + orchestrator).
- `tests/test_anchor_continuation.py` (untracked) — pure-Python tests +
  one slow Firedrake smoke at `V_RHE = 0 V` (sequential R1/R2, **not**
  parallel-2e/4e).
- `Forward/bv_solver/picard_ic.py` lines 906–1505 — `picard_outer_loop_general`,
  `_assemble_n_reaction_system`, `_build_picard_prefactors`,
  `_surface_concs_from_rates`, `_is_parallel_2e_4e`.
- `StudyResults/fast_realignment_2026-05-08/PHASE_5_ALPHA_GATE_FAILURE.md`
- `StudyResults/fast_realignment_2026-05-08/phase5alpha_gate/gate_report.json`
- `StudyResults/parallel_2e_4e_k0_ladder_probe/probe.json` — already
  diverged at `V_RHE=0 V` with `k0_R4e_scale=1e-30` (verdict
  "picard rewrite needed").
- `StudyResults/parallel_2e_4e_warmstart_probe/probe.json` — same
  verdict (warm-start from pure-2e-converged also blew up upon R_4e
  activation at `V_RHE = 0 V`).

## Executive Read

HANDOFF_22's diagnosis lines up with everything I see in the code and
the probe artefacts:

1. The post-Phase-5α multi-ion Picard correctly identifies the
   proton-transport-limited boundary state at `V_RHE=+0.55 V` but
   cannot resolve it because the lagged-H+ map has `R_4e ∝ H_o^4` and
   bounces between bulk-H_o (rate explodes) and floored-H_o (rate
   underflows). Gate report confirms `picard_max_iters_delta = 1.001`,
   `H_o = 1e-300` (floor), `R_4e = 0.286` (mass-transport limit).
2. The two existing 5γ probes at `V_RHE = 0 V` already established
   that k0 ramping alone is insufficient — even
   `k0_R4e_scale = 1e-30` (ladder floor in
   `parallel_2e_4e_k0_ladder_probe`) **fails SNES convergence** at
   `V_RHE = 0 V` because at the ladder-floor scale the boundary state
   itself is unphysical (`c_H2O2_surface ≈ 4e23`, `c_H+_surface ≈ 8e-8`).
   The orchestrator builds the IC at production k0 and the IC's
   Picard fails before the SNES even starts.
3. The fundamental issue is therefore upstream of the k0 ladder: the
   Picard's lagged-H+ structure cannot find the proton-limited root.
   HANDOFF_22's implicit-H_o bisection is the right next move.

The conclusion I want GPT to verify: **implementing
`solve_parallel_2e_4e_boundary_by_H` as a tight scalar bisection on
the H+ flux balance, scoped only to `_is_parallel_2e_4e(...)`, and
substituting it for the lagged-H+ Picard step in
`picard_outer_loop_general`, is load-bearing. Without it, neither the
existing k0 ladder nor a voltage homotopy will give the SNES a
converged anchor to start from.**

## Scope of the Existing Phase-5γ MVP (Already in the Repo)

`Forward/bv_solver/anchor_continuation.py` (~545 lines, untracked) is
*already* a clean implementation of HANDOFF_22's "Pass A / Pass B"
ladder pattern for a single voltage:

- `set_reaction_k0_model(ctx, j, k0)` writes both the metadata dict
  (read by Picard) and the live UFL Function (read by FE residual).
- `get_reaction_k0_model(ctx, j)` reads the authoritative dict value.
- `AdaptiveLadder` — geometric scale ramp ending at 1.0 with
  insert-on-failure rollback (`sqrt(prev * curr)` midpoint).
- `solve_anchor_with_continuation(...)` — orchestrator. Builds context
  + forms + IC at the target voltage, ramps each reaction's k0 from a
  configurable floor (default `1e-12 * k0_target`) to production via
  geometric rungs, runs SS solve + adaptive-dt at each rung, rolls
  back snapshot of `U` on failure. Wrapped in
  `firedrake.adjoint.stop_annotating()`.

Tests (`tests/test_anchor_continuation.py`):

- 17 pure-Python tests for the helpers and ladder (validation, no
  aliasing, history, success/failure semantics).
- 1 slow Firedrake test for `set_reaction_k0_model` round-trip on a
  real `build_context_logc_muh`.
- 1 slow Firedrake smoke test for `solve_anchor_with_continuation`
  at `V_RHE = 0 V` — **but it uses single-ClO4 + sequential R1/R2 +
  initial_scales=(1e-6, 1e-3, 1.0)**, not parallel-2e/4e at the actual
  failing voltage.

So the *plumbing* of HANDOFF_22 §2 ("Revise Phase 5γ continuation")
is in place, and tested in the easy regime. What's missing is the
inner SS solve being actually able to converge for parallel-2e/4e.
Both 5γ probes (V_RHE=0 V, `1e-30` and warm-from-pure-2e) confirm the
inner SS solve does not converge for parallel topology at any rung
yet.

## What is Missing (Action Items from HANDOFF_22)

1. **`solve_parallel_2e_4e_boundary_by_H`** (HANDOFF_22 §1, §"Concrete
   Implementation Sketch") — the implicit H+ boundary scalar solve.
   *Not started.*
2. **Voltage homotopy** (HANDOFF_22 §3) — anchor at `V_RHE ≈ +1.0 V`
   then walk down to `+0.55 V`. *Not started.*
3. **5α gate semantics fix** (HANDOFF_22 §4) — the current gate
   compares Picard at non-converged state against residual at the
   linear-φ fallback IC; meaningless. Defer until a converged anchor
   exists.
4. **A parallel-2e/4e smoke test** — the current
   `test_solve_anchor_with_continuation_smoke` is sequential R1/R2 at
   `V_RHE=0 V`. We need one for parallel-2e/4e + Cs⁺/SO₄²⁻ + Stern.

## Production State Numbers (for quick sanity-checking the reduced
estimate)

From `scripts/_bv_common.py`:

```text
K0_PHYS_R2E = K0_PHYS_R4E = 2.4e-8 m/s   (placeholder: equal kinetic
                                          prefactors)
ALPHA_R2E = 0.627, ALPHA_R4E = 0.5
E_eq_R2E = 0.695 V, E_eq_R4E = 1.23 V vs RHE
C_O2_PHYS = 1.2 mol/m³ (post-2026-05-07 M3a.2.1 migration)
C_HP_PHYS = 0.10 mol/m³ (pH 4)
K_SCALE = D_O2 / L_REF ≈ 1.93e-5 m/s
=> K0_HAT_R2E = K0_HAT_R4E ≈ 1.24e-3 (nondim production)
=> C_HP_HAT ≈ 0.0833 nondim
```

At `V_RHE = +0.55 V` post-Phase-5α gate report
(`StudyResults/fast_realignment_2026-05-08/phase5alpha_gate/gate_report.json`):

```text
phi_o ≈ -1.67e-4 (Cs⁺/SO₄²⁻ closure → bulk neutral, expected)
psi_D ≈ +1.58
psi_S ≈ +19.83
gamma_s ≈ 0.93
H_o = 1e-300 (floor)
R_2e ≈ -5.4e-10 (essentially zero, sign-flipped)
R_4e ≈ 0.286 (≈ D_O · O_b / D_O = 1 — mass-transport limit)
eta_2e ≈ -7.22, eta_4e ≈ -28.05 (nondim)
```

Quick reduced-model crosscheck (matches HANDOFF_22 §"Rough Reduced
Fixed-Point Estimate" within back-of-envelope precision):

```text
At H_o = H_b: alpha_hat[1] ∝ k0_R4e * gamma * (H_b/H_b * exp(-psi_D))^4 * exp(-α·n_e·η_4e)
            ≈ 1.24e-3 * 0.69 * (exp(-1.58))^4 * exp(56.1)
            ≈ 1.24e-3 * 0.69 * 1.8e-3 * 2.3e24
            ≈ 3e21
=> R_4e(H_b) ≈ alpha_hat[1] * O_b / (1 + alpha_hat[1]/D_O) ≈ D_O * O_b ≈ 0.286
   (saturates at the mass-transport limit, as the gate report shows).
For R_4e to match the budget 2 R_4e ≤ D_H · H_b ≈ 0.408,
need (H_o/H_b)^4 to compensate by factor ~1.5e-22 from H_b case
=> H_o/H_b ≈ (1.5e-22)^(1/4) ≈ 1.7e-5
   (matches HANDOFF_22's 1.6e-5).
```

## How the Picard Currently Iterates (Detailed)

`picard_outer_loop_general` (`Forward/bv_solver/picard_ic.py:1254–1607`):

```text
INIT:  R = [0]*N,  H_o = H_b,  c_s = bulk_concs,  gamma_s = 1.0,
       psi_S/psi_D/eta_list seeded from phi_applied.

PER ITER (omega = 0.5):
  1. gamma_s ← _compute_picard_gamma_s(H_o, psi_D, ...)   [uses lagged H_o]
  2. log_by_species[h_idx] = log(H_o) - psi_D + log_gamma   [lagged H_o]
  3. (alpha_hat, beta_hat, c_hat) = _build_picard_prefactors(
        reactions, log_gamma, log_by_species, eta_list)
     [alpha_hat[1] for R_4e ∝ H_o^4 · exp(-2 η_4e) · gamma_s · k0_4e]
  4. (M, b) = _assemble_n_reaction_system(...)             [2×2 in R]
  5. R_solve = solve(M, b)
     R = (1-ω) R_old + ω R_solve                            [relaxation on R]
  6. c_s = _surface_concs_from_rates(R, ...)               [signed flux balance]
     H_o = c_s[h_idx]                                       [floored at 1e-300]
  7. phi_o = _solve_phi_o(H_o, multi_ion_ctx, ...)
  8. (psi_S, psi_D, eta_list) = _update_electrostatics(c_s, phi_o, ...)
  9. delta = Σ_j |R_j - R_old_j| / max(|R_j|, 1e-30); break if delta < tol.
```

The structure is: **R is the Picard fixed-point variable; H_o, c_s,
gamma_s, eta_list are reconstructed from R each iteration.** H+ enters
the linear system only as a `cathodic_conc_factor` prefactor inside
`alpha_hat`, never as a linear substrate (validated by
`_validate_no_h_substrate`). That is exactly the structure HANDOFF_22
assumed when proposing the implicit H_o solve.

Why omega=0.5 relaxation does not help at the failing state:

```text
H_o^old = H_b (bulk seed)
=> alpha_hat[1] ≈ 3e21
=> R_solve_4e ≈ 0.286 (transport-limited, det dominates)
=> R_4e_relaxed = 0.5 · 0 + 0.5 · 0.286 = 0.143

Step 6: H_o^new = max(H_b - (R_2e + 2·0.143)/D_H, 1e-300)
                ≈ max(0.0833 - 0.238, 0) = 1e-300 (floored)

Next iter: alpha_hat[1] ∝ (1e-300)^4 ≈ 0 (underflow)
=> R_solve_4e ≈ 0
=> R_4e_relaxed = 0.5 · 0.143 + 0.5 · 0 = 0.072
=> H_o^new = max(0.0833 - 0.144, 0) = 1e-300 (still floored)

The map oscillates because the H_o → R map is sharply step-like at the
floor: R ≈ transport_limit when H_o > some tiny threshold, R ≈ 0 below.
Per-R relaxation cannot smooth a step.
```

## Implementation Proposal

### Function signature

Place in `Forward/bv_solver/picard_ic.py` (mirrors HANDOFF_22 §"Concrete
Implementation Sketch"; same module that already owns
`_build_picard_prefactors`, `_assemble_n_reaction_system`, etc., so we
get reuse for free):

```python
def solve_parallel_2e_4e_boundary_by_H(
    *,
    reactions: list,           # [R_2e, R_4e] dicts (passes _is_parallel_2e_4e)
    bulk_concs: list[float],
    diffusivities: list[float],
    species_floors: list[float],
    h_idx: int,
    psi_D: float,              # held fixed during bisection (see Q1)
    gamma_s: float,            # held fixed (see Q2)
    eta_list: list[float],     # held fixed; per-rxn clipped η
    H_floor: float = 1e-300,
    bisection_tol: float = 1e-12,
    bisection_max_iters: int = 80,
) -> tuple[bool, str, dict]:
    """Implicit-H_o root solve for parallel-2e/4e ORR.

    F(H_o) = H_b - (R_2e + 2 R_4e)/D_H - H_o
    where R_2e, R_4e come from solving the same 2×2 system that
    picard_outer_loop_general assembles, but with H_o as a first-class
    parameter rather than a lagged value.

    Bisection on [H_floor, H_b] because F is monotone decreasing in H_o
    (proof: ∂R_2e/∂H_o, ∂R_4e/∂H_o both ≥ 0 from the H_o^2, H_o^4 prefactors
    ⇒ ∂F/∂H_o ≤ -1).

    Returns
    -------
    (ok, reason, state) — state has keys:
      H_o, R_list (length 2), c_s_list (length n_species), n_bisection_iters.
    """
```

### How it fits into `picard_outer_loop_general`

Two integration options. I would like GPT to judge which is right.

**Option A — replace per-iter R-update only**: keep the Picard outer
loop, but inside step 4–6, dispatch on `_is_parallel_2e_4e(...)`:

```python
if _is_parallel_2e_4e(reactions, h_idx):
    ok, reason, state = solve_parallel_2e_4e_boundary_by_H(
        reactions=reactions, bulk_concs=bulk_concs,
        diffusivities=diffusivities, species_floors=species_floors,
        h_idx=h_idx,
        psi_D=psi_D, gamma_s=gamma_s, eta_list=eta_list,
    )
    if not ok: return False, reason, k, _state_dict_failure_general(...)
    R = state["R_list"]
    c_s = state["c_s_list"]
    H_o = state["H_o"]
else:
    # Legacy lagged-H+ path: build prefactors, solve 2×2, surface_concs.
    ...
```

The outer Picard then continues to iterate on
`(psi_D, psi_S, gamma_s, phi_o, eta_list)` since those are still
lagged. This preserves the existing structure for non-parallel
topologies and keeps the multi-ion `_solve_phi_o`/`_update_electrostatics`
machinery untouched. `omega` is only applied to `R` from
`solve_parallel_2e_4e_boundary_by_H`'s output (which is a
*near-converged* R given current electrostatics, so omega may be set
to 1.0 for parallel and 0.5 for legacy).

**Option B — replace the entire outer loop for parallel**: a
new top-level `picard_outer_loop_parallel_2e_4e` that interleaves the
implicit H_o solve with electrostatics updates, with its own
convergence test on `(R_2e, R_4e, psi_D, gamma_s)`. Cleaner separation
but doubles the surface area to maintain.

My current preference is A. It is the smallest change, preserves
single-ion byte-equivalence trivially (unchanged path), and keeps the
multi-ion fixes from Phase 5α intact. But A has a subtle question
(see Q3 below).

### Bisection internals

For each trial `H_o ∈ [H_floor, H_b]`:

```python
log_gamma = math.log(max(gamma_s, 1e-300))
log_by_species = list of log(c_s_b[i]) + log_gamma, except
                 log_by_species[h_idx] = log(H_o) - psi_D + log_gamma
alpha_hat, beta_hat, c_hat = _build_picard_prefactors(
    reactions=reactions, log_gamma=log_gamma,
    log_by_species=log_by_species, eta_list=eta_list,
)
# Note: c_s for species other than H+ is held at bulk during the
# bisection — the 2×2 system reconstructs O_s and P_s implicitly through
# the M·R = b solve and stoichiometric flux balance afterward.
# (Q4: should we re-iterate on c_s during bisection?)
M, b = _assemble_n_reaction_system(
    reactions=reactions, alpha_hat=alpha_hat, beta_hat=beta_hat,
    c_hat=c_hat, bulk_concs=bulk_concs, diffusivities=diffusivities,
    h_idx=h_idx,
)
R_solve, det = _solve_2x2(M, b)
if not finite(det) or abs(det) < 1e-300:
    return ok=False with singular_jacobian
R_2e, R_4e = R_solve
F = H_b - (R_2e + 2.0 * R_4e) / D_H - H_o
```

Bracketing:

```text
F(H_floor): R(H_floor) ≈ 0 (alpha_hat ≈ 0 from H^4 underflow at H_floor).
            => F ≈ H_b - 0 - H_floor ≈ H_b > 0
F(H_b):     R(H_b) saturates at transport limit ≈ 0.286 (4e branch),
            => F ≈ H_b - (0 + 0.572)/D_H - H_b = -0.572/D_H ≈ -0.5 < 0
```

So the root is bracketed in `[H_floor, H_b]` for the failing gate case.
Standard bisection converges in ~60 iters to `bisection_tol = 1e-12`.

If `F(H_b) > 0` (no root in bracket — pure-2e or sub-budget mixed
state): the H+ flux budget is not saturated, and the lagged Picard
should converge. We can return `ok=False, reason="no_h_limit"` and let
the caller fall through to the legacy Picard path. (See Q5.)

### Testing plan

Pure-Python:
- `solve_parallel_2e_4e_boundary_by_H_at_gate_state`: feed the
  HANDOFF_22 / gate-report numbers (`psi_D=1.58, gamma_s=0.93,
  eta_list=[-7.23, -28.05]`, production K0_HAT and C_HP_HAT). Assert
  `H_o/H_b ≈ 1.6e-5 ± 5%`, `R_4e ≈ 0.20 ± 5%`, `R_2e ≈ 0 ± 1e-6`.
- `solve_parallel_2e_4e_boundary_by_H_pure_2e_no_root`: `k0_R4e=0`.
  Assert `ok=False, reason="no_h_limit"` (or equivalent fall-through
  signal).
- Monotonicity check: F(H_o) is monotone decreasing on [H_floor, H_b]
  for the gate state.
- `test_picard_outer_loop_general_parallel_at_gate_state`: same inputs
  as the gate, calling the modified `picard_outer_loop_general` with
  `multi_ion_ctx` and `_is_parallel_2e_4e` true; assert convergence
  with H_o/H_b ≈ 1.6e-5.

Slow / Firedrake:
- Replace / extend `test_solve_anchor_with_continuation_smoke` with a
  parallel-2e/4e + Cs⁺/SO₄²⁻ + Stern variant at `V_RHE = +0.55 V` and
  initial_scales = `(1e-23, 1e-15, 1e-9, 1e-3, 1.0)` (per HANDOFF_22's
  scale recommendation — see Q7).

## Specific Design Questions for GPT

**Q1 — psi_D held fixed during the H_o bisection?**
psi_D depends on phi_applied, phi_o, c_s, lambda_eff via
`_solve_picard_stern_split` and `_update_electrostatics`. lambda_eff
depends on `c_s` (through `effective_debye_length_local`), and c_s
contains H_o. So psi_D is a (mild) function of H_o. Holding psi_D
fixed during the bisection means the inner solve is a clean 1-D root
find. The outer Picard then re-solves psi_D from the new H_o and
re-bisects. Is that the right separation, or should psi_D be
re-solved per H_o trial? My intuition says fixed-during-bisection is
fine because psi_D is dominated by the Stern split which depends on
phi_applied >> psi_D's variation with H_o (psi_D ≈ 1.58 here,
phi_applied ≈ 21.4). But I want a sanity check.

**Q2 — gamma_s held fixed?** Same structure: gamma_s depends on H_o
through `compute_surface_gamma_multiion(H_o, psi_D, ...)`. At the
gate state gamma_s = 0.93, well away from saturation, so the H_o
dependence is weak. Holding fixed in the inner bisection seems safe.
But the multi-ion gamma computation also depends on `c_s` and
`phi_o` which depend on the full ion ensemble — and the gate state
shows phi_o ≈ -1.67e-4 (essentially zero). I think holding fixed is
correct.

**Q3 — does Option A's per-iter dispatch break the outer Picard's
fixed-point structure?** With Option A, each outer iter computes a
near-converged R for the current `(psi_D, gamma_s, eta_list)`. The
outer iter then updates electrostatics from the new c_s. This is no
longer a Picard map on R; it's effectively a fixed-point on
electrostatics with H_o always at its conditional root. Does that
contract? My intuition: yes, because (psi_D, gamma_s) depend mildly
on H_o for the multi-ion case (Q1, Q2 reasoning). But this is a
qualitative claim — would you expect any pathological case where the
electrostatics map fails to contract once H_o is solved out?

**Q4 — should the bisection re-iterate on (O_s, P_s) too?** The 2×2
system in `_assemble_n_reaction_system` uses `bulk_concs` for
substrates. So `R_solve` already implicitly handles the O_s, P_s
flux balance through M's `(1 + alpha_hat/D)` diagonal (the
mass-transport correction). After R is found, `_surface_concs_from_rates`
gives the post-update `(O_s, P_s, H_o)`. Inside the bisection I plan
to use `bulk_concs` for the 2×2 substrate input (matching the legacy
Picard) rather than feeding back the mid-iter `c_s`. Does this match
what HANDOFF_22 §"Concrete Implementation Sketch" intended? The
sketch is unspecific.

**Q5 — fallback when there is no proton-limited root.**
If `F(H_b) > 0` (i.e. even at bulk H_o the budget is sub-saturated),
the proton-limited branch doesn't exist for this state. This will
be the case for: (a) pure-2e (k0_R4e=0), (b) mid-cathodic regime
where R_4e is not yet dominant. My plan is to fall through to the
legacy lagged-H+ path in that case (Picard on R, the standard
algorithm). Is that the right escape valve, or should
`solve_parallel_2e_4e_boundary_by_H` always return *some* root by
clamping to F(H_b) sign convention?

**Q6 — outer Picard convergence test with the implicit inner solve.**
The current `delta = Σ |R_j - R_old_j| / max(|R_j|, 1e-30) < tol`
test on R may be inappropriate when the inner solve already gives a
near-converged R per outer iter — outer iters are then dominated by
electrostatics drift. Should the convergence test for the parallel
path switch to `delta_psi_D + delta_gamma_s + delta_phi_o < tol`?
Or keep the R-based test (which would still detect outer
non-convergence, just less directly)?

**Q7 — k0 ladder floor for parallel-2e/4e at +0.55 V.**
HANDOFF_22 estimates safe k0 ≈ 1e-23 at `V_RHE = +0.55 V`. The
existing `solve_anchor_with_continuation` default is
`(1e-12, 1e-9, 1e-6, 1e-3, 1.0)` (multiplicative scales applied to
k0_target). For parallel at `+0.55 V` should I default to
`(1e-23, 1e-19, 1e-15, 1e-11, 1e-7, 1e-3, 1.0)` or keep it
caller-supplied? My read: caller-supplied via kwarg, with one
documented "production_parallel_2e_4e_at_055V" preset.

**Q8 — voltage homotopy: practical anchor voltage?**
The validated convergence window for the production single-ClO4
stack is `V_RHE ∈ [-0.5, +1.0] V` (per `CLAUDE.md` §"Hard rules" #4).
At `V_RHE = +1.0 V`, eta_4e ≈ -1.23 + 1.0 = -0.23 V dimensional ≈
-9 nondim, which is still strongly cathodic (BV factor `exp(0.5·4·9)
= exp(18) ≈ 6.6e7`). At `V_RHE = +1.1 V`: eta_4e ≈ -5 nondim,
factor `exp(10) ≈ 22000`. At `V_RHE = +1.2 V`: eta_4e ≈ -1 nondim,
factor `exp(2) ≈ 7.4` — friendly.
Question: does HANDOFF_22's recommended anchor at "near +1.0 to
+1.1 V" actually relax the stiffness enough that the cold IC
converges, or do we need to push to +1.2 V? Worth noting this is
beyond the validated window.

**Q9 — interaction with the existing 5γ probes.**
Both probes (`parallel_2e_4e_k0_ladder_probe`,
`parallel_2e_4e_warmstart_probe`) at `V_RHE=0 V` already concluded
"picard rewrite needed". The k0 ladder probe ran at scale 1e-30 and
still failed because the boundary IC's Picard couldn't even seed a
sensible state. With the implicit H_o solve, does the k0 ladder
become unnecessary? Or is it still needed to walk the SNES through
the spatial profile? My suspicion: still needed for the spatial
side, because even with a correct boundary state, the IC's spatial
gradient (psi_D ≈ 1.58 in the diffuse layer) is sharp, and Newton
benefits from k0 ramping to soften the boundary forcing during the
walk-up.

**Q10 — pure-2e (Pass A) sanity.**
HANDOFF_22's reduced model gives `R_2e ≈ 0.139, H_o/H_b ≈ 0.66`
for pure 2e. With H_o ≈ 0.66 · H_b (mild depletion), the lagged
Picard *should* converge for pure 2e. I plan to verify the implicit
solver still handles pure 2e correctly (returns `H_o/H_b ≈ 0.66,
R_2e ≈ 0.139` when k0_R4e is below some threshold), but to *prefer*
the legacy path for pure 2e via Q5's fall-through. Does that
match your read? Or should the implicit path always be used for
parallel-2e/4e topology (to keep one code path)?

## Open Issues / Risks I Already See

1. **Floating-point underflow inside the bisection.** `alpha_hat[1]
   ∝ H_o^4 · exp(-2 η_4e) · k0_4e · gamma`. At `H_o ≈ 1e-80`,
   `H_o^4 ≈ 1e-320` underflows. I plan to do the prefactor build in
   log-space (which `_build_picard_prefactors` already does:
   `log_alpha = log(k_j) + log_gamma + log_h_factor - α n_e η`
   then `_safe_exp`). So underflow is handled — `_safe_exp` clamps
   to 0 on overflow into `cap=700`.

2. **The bisection bracket `[H_floor, H_b]` may not always contain a
   root.** Q5 above. If the user feeds an unusual config, we need a
   clean fallback.

3. **Multi-ion electrostatics (`_solve_phi_o` via
   `solve_outer_phi_multiion`) might oscillate even with H_o solved
   out.** The 5α gate failure was diagnosed as H_o-driven, but the
   underlying multi-ion fixed point has more structure
   (`solve_outer_phi_multiion` does its own Picard on phi_o given
   c_s). Worth checking whether outer convergence relies on
   psi_D's stiff dependence on H_o through lambda_eff (which would
   be lost when H_o is solved out per iter). I think this is fine
   because lambda_eff is dominated by the bulk Cs⁺/SO₄²⁻ ions, not
   by H+ — but want a sanity check.

4. **The implicit solve gives a converged BOUNDARY state, but
   building the spatial IC from it (debye_boltzmann seed) may still
   give Newton a too-sharp profile.** Per Q9, k0 continuation at
   the SS-loop level may still be needed.

5. **HANDOFF_22's voltage homotopy recommendation cuts against
   PHASE_4_STATUS.md §"Phase 5δ unlikely to help"**, which argued
   higher V_RHE makes ψ_S larger and therefore makes the cathodic
   problem worse. HANDOFF_22 argues the opposite (higher V_RHE
   makes eta_4e less cathodic). The disagreement is over which term
   dominates at which voltage. Numerically, HANDOFF_22 wins for
   `V_RHE > E_eq_4e − few·V_T`, i.e. above ~+1.1 V; PHASE_4 wins
   below. Both are right within their regimes. The practical
   recommendation is unchanged: anchor at +1.0–1.1 V where eta_4e
   is mild enough to converge, then walk down. Worth flagging as a
   resolved inconsistency.

6. **Calibration risk if implicit solve succeeds.** HANDOFF_22's
   closing risk note: equal `K0_R4e = K0_R2e` is unphysical for ORR.
   If the converged mixed stack at +0.55 V is peroxide-poor (which
   the reduced model predicts), the next step is calibration
   (lower `K0_R4e`, revisit `α_R4e` against Tafel/disk/selectivity
   data), not more solver machinery. Worth recording so we don't
   chase the wrong target after a successful gate.

## Concrete Next-Step Plan (gated on this review)

1. Implement `solve_parallel_2e_4e_boundary_by_H` per the signature
   above, with pure-Python tests at the gate state and pure-2e.
2. Wire it into `picard_outer_loop_general` via Option A dispatch.
3. Add a slow Firedrake test
   `test_picard_outer_loop_general_parallel_at_gate_state` that builds
   the multi-ion ctx and asserts the Picard converges to
   `H_o/H_b ≈ 1.6e-5`, `R_4e ≈ 0.20`.
4. Re-run `picard_residual_consistency_csplus_so4.py` (the 5α gate)
   and confirm it passes with the converged Picard.
5. Add a `(1e-23, ..., 1.0)` parallel-2e/4e + Cs⁺/SO₄²⁻ smoke test
   to `test_anchor_continuation.py`.
6. **Only then** consider voltage homotopy or k0 ladder
   re-tuning. The implicit solve is the load-bearing primitive; the
   ladder is the warm-walk on top of it.

## Recommended Next ChatGPT Action

Please read this and find holes. Specifically:

- Are Q1 / Q2 (psi_D, gamma_s held fixed during bisection) safe
  approximations or load-bearing assumptions that will break under
  some operating point?
- Q3 (Option A vs B integration) — which is right?
- Q5 (no-root fallback semantics).
- Q9 (does k0 ladder become redundant once implicit H_o solve is in
  place?).
- Q10 (pure-2e via implicit path vs. legacy path).
- Anything else I missed.

If you reach VERDICT APPROVED on this scoping, I will implement
following the §"Concrete Next-Step Plan" sequence. If you find
load-bearing issues, I expect to revise the plan before touching
`picard_ic.py`.

— Claude
