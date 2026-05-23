The plan is close, but I still see two blocking state-management holes caused by inserting a mutable ξ coefficient into continuation machinery that currently only checkpoints `U/U_prev`.

1. WHAT: `warm_walk_phi` rollback checkpoints are still U-only. Current bisection logic snapshots/restores `U` and `U_prev`; v5 adds ξ mutation but does not say `ckpt_inner` / `ckpt_outer` include ξ. WHY: A failed Picard-wrapped substep can leave ξ at a failed damped value while warm-walk restores only the PDE state, corrupting the retry/bisection path. WHAT TO DO: Make warm-walk checkpoints Picard-aware: snapshot and restore `(U, U_prev, xi)` whenever a Picard factory is active, or require `picard_run_ss(False)` to restore the exact entry state before returning False. Prefer both.

2. WHAT: The same rollback issue applies to anchor/k0/Stern ladder failures. WHY: `solve_anchor_with_continuation` has its own rung rollback snapshots; after adding the Picard factory, those snapshots must include ξ too. Otherwise failed ladder rungs can leave ξ inconsistent with the restored `U`. WHAT TO DO: Extend anchor-continuation checkpoint helpers to carry ξ snapshots when `bv_picard_mode=True`.

3. WHAT: `converged_overall = grid_converged AND all Picard wrap calls for that V returned True` is wrong. WHY: Warm-walk bisection normally creates failed attempts before a successful smaller step. Those failed Picard calls are rejected path attempts, not final per-V failure. WHAT TO DO: Track attempted vs accepted Picard results. Overall convergence should depend on `warm_walk_phi` returning True and the final accepted target-state Picard result being converged, not every failed intermediate attempt.

4. WHAT: `picard_run_ss` on final failure is not guaranteed to restore its entry state. WHY: On `max_picard_iters` or persistent `run_ss_failed_with_target`, it may leave the last attempted ξ/U in ctx. The caller may then do rollback, but only if every caller has been made ξ-aware. WHAT TO DO: Contractually require: every `picard_run_ss(max_steps)` returning False restores `(U, U_prev, xi)` to the state at entry.

5. WHAT: The `make_run_ss_factory=None` byte-equivalence test needs an explicit normalization rule. WHY: Your default is `make_run_ss`, so passing `None` will fail unless code treats `None` as bare factory. WHAT TO DO: Add `if make_run_ss_factory is None: make_run_ss_factory = make_run_ss` at every public entry point.

6. WHAT: The pinned-baseline test depends on `StudyResults/.../iv_curve.json`. WHY: Those results may be untracked/local and brittle under solver tolerances. WHAT TO DO: Use a small committed fixture or generate the baseline inside the test once with the bare factory, then compare bare vs modified path.

VERDICT: ISSUES_REMAIN