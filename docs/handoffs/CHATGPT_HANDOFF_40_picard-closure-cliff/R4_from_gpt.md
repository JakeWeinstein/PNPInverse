The core design is now right: wrapping `run_ss` is the correct insertion point. I still would not approve v4 because a few implementation contracts are internally inconsistent.

1. WHAT: The plan says modify `solve_grid_with_anchor`, but the actual `make_run_ss(...)` call for warm-walk lives inside `warm_walk_phi`. WHY: Adding a factory only at `solve_grid_with_anchor` will not affect the substep/final `run_ss` calls unless `warm_walk_phi` accepts and uses that factory. WHAT TO DO: Change `warm_walk_phi(..., make_run_ss_factory=make_run_ss)` and have both grid drivers pass either the bare or Picard factory through.

2. WHAT: The factory signature is inconsistent with `make_run_ss`. You list `dt_initial`, `dt_max`, and `max_ss_steps` in the factory, but current `make_run_ss` uses `dt_init`, `dt_growth_cap`, `dt_max_ratio`, and the returned closure receives `max_steps`. WHY: This will create a parallel API and likely wrong dt behavior. WHAT TO DO: Make the Picard factory accept exactly the same steady-state kwargs as `make_run_ss`; keep `max_steps` only on the returned `picard_run_ss(max_steps)`.

3. WHAT: `picard_run_ss` returns only `bool`, but the plan also says `PicardResult` is returned to caller and per-V JSON captures Picard history. WHY: `warm_walk_phi` only consumes `bool`; any result returned by the wrapper is lost. WHAT TO DO: Store the latest and cumulative Picard results on `ctx`, e.g. `ctx["_picard_run_ss_history"]`, and have diagnostics/reporting read from there.

4. WHAT: Picard rollback still does not explicitly snapshot/restore `U_prev`. WHY: In this pseudo-time formulation, restoring `U` without `U_prev` corrupts the next residual. WHAT TO DO: Every Picard rollback snapshot must include `U`, `U_prev`, and ξ; restore all three together.

5. WHAT: Returning `False` on first inner `run_ss` failure delegates to warm-walk bisection, but if bisection reaches its depth limit, Picard still never attempts a supply reduction. WHY: Deep-cathodic recovery depends on having a usable ξ from prior successful substeps; if that is not enough, the algorithm falls back to v1 failure. WHAT TO DO: Acceptable if intentional, but state it as a known limitation and log `run_ss_failed_before_picard_target`; stronger fix is an optional emergency ξ backoff before declaring substep failure.

6. WHAT: Anchor continuation also calls `make_run_ss` directly in several ladder paths. WHY: The plan mentions adding an optional wrapper arg, but the change is larger than a simple Stage 1 call-site tweak; every ladder `run_ss` must use the factory to keep anchor/Stern/k0 states Picard-consistent. WHAT TO DO: Apply the same factory pattern inside `solve_anchor_with_continuation`, not just the study script.

7. WHAT: `closure_picard.py` writing trajectory logs directly is a layer violation. WHY: Forward solver modules should not know study output directories; it makes tests and reuse messy. WHAT TO DO: Keep Picard history in returned/stored data structures; let the study script write JSON.

8. WHAT: The byte-equivalence test described only compares `None` versus omitted. WHY: That does not prove the new factory plumbing is inert through all modified call paths if the code branches on `picard_mode` elsewhere. WHAT TO DO: Also run a non-Picard config through the modified anchor/grid drivers and compare against the pre-wrapper baseline or a fixture with the factory path absent.

9. WHAT: `test_parallel_2e_4e_shared_supply` says “one Picard iter” and “matches the Picard target formula.” WHY: One iteration generally is not a fixed point. WHAT TO DO: Either assert wiring only after one iter, or run to convergence before asserting the closure residual.

10. WHAT: The `enabled=False` no-flux test may not exercise the log-rate replacement path if disabled reactions skip before closure substitution. WHY: It could pass while `log_c_cat = ln(packing)+xi` is broken. WHAT TO DO: Add a second zero-flux form test with an enabled reaction but `R_j` manufactured/overridden to zero after the closure path is built, or directly assert the rate expression changes with ξ in `test_function_update_no_rebuild`.

VERDICT: ISSUES_REMAIN