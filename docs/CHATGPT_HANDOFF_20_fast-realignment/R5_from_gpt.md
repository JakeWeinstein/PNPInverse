1. WHAT: Pass C is no longer part of done.
WHY: That is acceptable only because Pass C is explicitly labeled exploratory and Pass D is the structural mixed-channel validation pass. The plan no longer claims Pass C is page-15 interpretable.
WHAT TO DO: In the final artifact, state plainly that the delivered “done” sweep is structural, not calibrated-to-literature-K0. Do not let Pass D plots be described as physical page-15 agreement.

2. WHAT: `phi_outer(y) = phi_o · (1 − y/L_REF)` is still an IC approximation, not exact multi-ion outer electroneutrality at every y.
WHY: It is acceptable for the fast plan because Phase 2.4 now separates `phi_outer` from `psi`, uses local `c_dyn_outer`, and lets Newton repair the profile.
WHAT TO DO: Document it as an IC approximation. If Newton failures localize to the initial EDL shape, the first escalation is solving `_solve_outer_phi_multiion()` at several y nodes and interpolating.

3. WHAT: The revised plan now has a coherent single source of truth for multi-ion closure, a correct ψ-vs-φ split, local Debye length, explicit opt-in validation, and per-pass acceptance.
WHY: The earlier blocking structural errors are addressed.
WHAT TO DO: Apply these revisions to `docs/fast_realignment_plan_2026-05-08.md` before implementation.

VERDICT: APPROVED