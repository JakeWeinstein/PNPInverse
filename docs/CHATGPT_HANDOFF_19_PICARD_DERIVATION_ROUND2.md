# Handoff #19 — Picard General Topology Derivation, Round 2

**Date:** 2026-05-08
**Predecessors:** `docs/CHATGPT_HANDOFF_17_RUGGIERO_REALIGNMENT_PLAN.md`,
  `docs/CHATGPT_HANDOFF_18_RUGGIERO_REALIGNMENT_COUNTERREPLY.md`,
  `docs/CODEX_REVIEW_PICARD_GENERAL_TOPOLOGY_DERIVATION.md` (round 1).
**Asks of this round:** confirm round-2 fixes are complete and the
  derivation is airtight enough to use as the implementation contract
  for M3a.3 §3 (generalized `picard_outer_loop`).  No new code yet;
  this is a doc-only round.

## What this is

A focused review request for the **v2** of
`docs/picard_general_topology_derivation.md`, after the round-1
findings (Codex 2026-05-08) were accepted and incorporated.  The v2
is intended to be the source-of-truth for the §3 refactor that
replaces the hardcoded 2x2 sequential algebra at
`Forward/bv_solver/picard_ic.py:441-563` with an N-reaction linear
solve.

The escape-hatch shortcut (warm-start parallel residual from a
converged sequential state at V=0) was tested empirically and
**diverged** — see
`StudyResults/parallel_2e_4e_warmstart_probe/probe.json`.  R_4e
blew up to ~10¹⁴ mA/cm² on Newton's first iterate because the IC
didn't encode the surface depletion that quenches `exp(-α·n_e·η_R4e)
= exp(+96)` at V_RHE = 0.  This empirically confirms that the
generalized matched-asymptotic Picard is on the critical path for
M3a.3 — there is no code-light alternative.

## Round-1 findings → v2 dispositions

| F# | Round-1 finding | v2 status | Where to look |
|---|---|---|---|
| F1 | `|s_H|/2` formula silently wrong-sign for H⁺-producing reactions | **Fixed.**  Signed `+ Σ_j s_{H,j}·R_j / (2 D_H)` throughout | §2 formula + "Why signed" callout; §7 outer loop body; §9 contract item 3; §10 T11 (signed-H_o synthetic test) |
| F2 | Constant anodic branch (reversible + anodic_species is None + c_ref_model > 0) missing from rate model | **Fixed.**  Three-branch anodic table added; `R_j = α̂_j · c_cath − β̂_j · c_anod − Ĉ_j`; matrix `M` unchanged, `Ĉ_j` enters only `b_j` | §3 three-branch table; §4 extended `b_j`; §9 contract item 4; §10 T12 (constant-anodic branch test) |
| F3 | "General N-reaction" overstated; non-substrate factors Picard-lagged | **Fixed.**  Title narrowed to "Linear-Substrate Picard Initializer for ORR-class BV Topologies"; "what's lagged" callout in §7; scope guard in §11 | Title; "Scope" preamble; §7 "What's lagged"; §11 "Strong dynamic species in `cathodic_conc_factors`" |
| F4 | T2 underspecified (pure-2e parallel vs *full* legacy sequential) | **Fixed.**  Now compares pure-2e parallel (`k0_R4e = 0`) against legacy sequential (`k0_R2 = 0`) | §10 T2 |

No findings rejected.  v2 §12 contains a fix-by-fix reply table for
audit purposes.

## What v2 changed structurally

1. **Title narrowed.**  "General Reaction Topology Derivation" →
   "Linear-Substrate Picard Initializer for ORR-class BV Topologies".
   Reflects what the algebra actually does (linearizes only on the
   selected substrate concentrations; everything else is
   Picard-lagged).

2. **Per-reaction inputs table extended** (§1).  Added `c_ref_model`
   and `reversible`; clarified that `anodic_species = None` does **not**
   imply irreversible.

3. **Surface flux balance now signed throughout** (§2).  The general
   form is `c_i_s = c_i_b + Σ_j s_{i,j}/D_i · R_j`, and the H⁺ outer
   balance is `H_o = H_b + Σ_j s_{H,j}·R_j / (2 D_H)`.  Sequential and
   parallel cases re-derived as instances.  The realignment-plan
   ambipolar typo (drops `1/2`) is called out explicitly.

4. **Three-branch anodic rate model** (§3).  Replaces the two-branch
   "reversible vs irreversible" with the residual's actual three-way
   conditional (linear-substrate, affine-constant, irreversible),
   matching `forms_logc.py:399-415, 427-439` and
   `forms_logc_muh.py:438-454, 466-479` line-by-line.

5. **Matrix-RHS form is affine** (§4).  `M` unchanged; `b_j` gains a
   `−Ĉ_j` term.  For ORR `Ĉ_j = 0` so this is invisible to the
   M3a.3 path, but the contract is now correct for any future
   reaction that uses branch 2.

6. **Implementation contract expanded** (§9).  Items 4 (three-branch
   anodic), 8 (robust failure → linear-phi fallback), 9 (validation
   guards on dynamic-species-in-cathodic_conc_factors), and 10
   (optional `s_H ≤ 0` scope assertion) are new.

7. **Verification ladder gains T11, T12** (§10).  T11 is a signed-H_o
   synthetic with `s_H = +2`; T12 verifies the constant-anodic branch.
   Either implementation passes them or rejects those configs at the
   adapter site with a clear failure reason.

## Specific airtight-check questions for round 2

Please confirm or push back on each:

**Q1 (γ-power for the constant-anodic branch).** v2 §3 claims
`Ĉ_j ∝ γ_s^0` because the residual at `forms_logc.py:407-413` (log-
rate) and `forms_logc.py:433-437` (non-log-rate) does **not** apply
γ to `c_ref_model`.  Confirmed against the code lines but not against
any analytic derivation — is γ⁰ on the constant branch physically
correct, or is it a residual code bug we should preserve in the IC
(matched-asymptotic must mirror the residual, regardless of physics)?

**Q2 (T11 framing).** The signed-H_o synthetic test uses an
unphysical `s_H = +2` reaction purely to verify the formula
implementation.  Should we instead enforce `s_H ≤ 0 ∀ j` at config
time (per v2 §9 item 10) and *remove* T11 since the formula is
defensive and never exercised?  Or is keeping the formula general +
testing it more robust than narrowing the scope?

**Q3 (T12 disposition).** The constant-anodic branch isn't reached by
either the sequential or parallel ORR presets we ship today.  Two
viable paths for v2:

  (a) Implement Ĉ_j in the Picard, test it, leave it dormant in
      production (current v2 spec).
  (b) Add a config-time validator that rejects branch-2 configs at
      the IC entry point with `non_constant_anodic_branch` reason,
      and fall back to linear-phi.  Skip T12.

Which is the "more airtight" call?  (a) protects future configs; (b)
narrows the scope to known-good ORR.  My instinct is (a) since the
implementation cost is ~10 lines and a unit test, but if the project
philosophy here is "narrow what you can't test in production sweeps",
(b) is fine.

**Q4 (post-loop closed form).** v2 §8 keeps the legacy
`P_s = (D_P·P_b + R_1)/(D_P + A_2)` only behind a topology hint
`"sequential_2e_h2o2"`.  For the parallel preset we use the naive
`P_s = P_b + R_2e/D_P`.  Risk: if there's a diffusion-limited regime
in the parallel case where `R_2e ≈ D_P·P_b` and Newton is sensitive
to the IC's `P_s` value, the naive formula's near-cancellation could
matter.  Worth pre-emptively deriving a robust closed form for the
parallel topology, or wait for empirical failure?  My read: wait —
parallel `P_s` doesn't have the sequential's `R_1 − R_2 ≈ 0`
cancellation source, so the naive form is safer than refactoring
preemptively.

**Q5 (per-reaction relaxation ω).** v2 §7 specifies a single ω = 0.5
applied to every R_j.  When R_2e ≫ R_4e at low |η|, both rates
nevertheless converge under the same ω in a 2x2 inner solve.  Any
reason to make ω per-reaction (e.g., smaller for the dominant rate)?
My read: no — the inner linear solve absorbs the cross-reaction
coupling, and per-reaction ω adds a tunable knob for marginal benefit.

**Q6 (escape hatch postmortem).** The empirical probe at
`StudyResults/parallel_2e_4e_warmstart_probe/probe.json` shows that
even a converged sequential state at V=0 isn't a basin of attraction
for the parallel residual.  This confirms the generalized Picard is
necessary — but does it imply anything about the *quality* of the
generalized IC needed?  Specifically, if the parallel residual at V=0
is so stiff that Newton diverges from a near-physical starting state,
should the generalized Picard target a tighter `tol` than the legacy
2x2's `1e−6` (e.g., `1e−9`) to ensure Newton has enough slack?

**Q7 (anything missing).** What else should be locked into v2 before
the §3 implementation starts that I haven't surfaced?  Candidates I
considered but didn't include:

  - Generalized closed-form post-loop reconstruction (deferred per v2
    §8 last paragraph).
  - Multi-ion γ extension (deferred to M3b.2 per v2 §11).
  - Per-reaction `ω` (Q5).
  - Per-reaction `tol` (Q6).

If any of these *should* be in v2, flag and I'll add them before
moving to code.

## What round 2 unblocks

Pending GPT signoff (or a counter-reply with v3 changes), the
implementation order from the M3a.3 plan is:

- Task #4 — byte-equivalence test fixture (T1) + amend
  `docs/ruggiero_realignment_plan.md:381` H_o note with the corrected
  signed formula and a back-pointer to v2.
- Task #5 — generalize `picard_outer_loop` per v2 §9 contract.
- Task #6 — drop topology gates at the IC adapter sites.
- Task #7 — rest of the verification ladder T2-T12.

If GPT pushes back on any v2 finding, the cycle is doc-only until
v3 settles; no code lands until the contract is locked.

## Audit trail

- `docs/picard_general_topology_derivation.md` — v2 (this round's source).
- `docs/CODEX_REVIEW_PICARD_GENERAL_TOPOLOGY_DERIVATION.md` — round-1 review.
- `docs/CHATGPT_HANDOFF_18_RUGGIERO_REALIGNMENT_COUNTERREPLY.md` — parent
  M3a substaging structure.
- `docs/ruggiero_realignment_plan.md` — milestone context (§"M3a.3"
  notes the H_o ambipolar typo that v2 §2 corrects).
- `Forward/bv_solver/picard_ic.py:441-563` — current 2x2 implementation
  (frozen reference at commit `d8bf645` for T1 byte-equivalence).
- `Forward/bv_solver/forms_logc.py:340-454` — residual rate construction
  (log-rate + non-log-rate paths).
- `Forward/bv_solver/forms_logc_muh.py:380-487` — muh backend residual.
- `scripts/_bv_common.py:525-554` — `PARALLEL_2E_4E_REACTIONS` literal.
- `StudyResults/parallel_2e_4e_warmstart_probe/probe.json` — empirical
  proof that the escape hatch shortcut doesn't work, justifying the
  full §3 effort.
