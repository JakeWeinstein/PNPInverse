# Critique Session 25 — phase-6a-water-ionization

- Started: 2026-05-10T00:29:07Z
- Round cap: 5
- Final round: 5
- Final verdict: APPROVED
- Codex session ID: 019e0f4c-f1e7-7350-aeac-7bab073eaad7
- Original artifact: /Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/docs/PHASE_6A_OH_WATER_IONIZATION_PLAN.md
- Output dir: /Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/docs/CHATGPT_HANDOFF_25_phase-6a-water-ionization
- Status: complete

## Round summaries

### R1 — verdict ISSUES_REMAIN (25 issues)
Major: Option B physically broken; Damköhler arithmetic; c_OH at pH 14 is 1 M, not 1e-5.

### R2 — verdict ISSUES_REMAIN (15 issues)
Major: NP migration term needs D_i; finite water source capacity is a real bound;
weak form sign convention; muh form of E equation cleanest.

### R3 — verdict ISSUES_REMAIN (10 issues)
Major: 10× arithmetic correction on i_max; finite-rate validity = R_w,req/(k_r·Kw)
not raw current; sulfate is steady-state, not "one-shot"; water-aware Picard required.

### R4 — verdict ISSUES_REMAIN (8 issues)
Major: Gate 3 (water as cap) wrong; sulfate transport off by 10×; stale water-rate
constants needed purging; reduced BC validation against full OH⁻ no-flux.

### R5 — verdict APPROVED (5 minor TODOs)
All architectural and numerical issues resolved; remaining items are revision
TODOs absorbed into the revised plan.

## Outcome

Plan revised end-to-end.  Major architectural pivot:
**Option B (algebraic c_OH = Kw/c_H slaving in Poisson) → Option C
(proton-condition variable E = c_H − c_OH)** — Option B was physically
incoherent because it added OH⁻ charge without sourcing H⁺, leaving
the H⁺ NP equation unchanged and surface c_H still unbounded.

Final plan: 5 acceptance gates (surface pH, plateau direction,
E conservation, fast-water validity ε < 0.1, Yash cross-check + reduced-BC
J_OH·n validation).  Path-B IC (approximate, relies on Newton + Kw_eff
continuation) with hard-trigger fallback to Path A (water-aware Picard
rewrite).  Implementation cost ~5-7 days, ~1000 LoC.

See `FINAL_REVISION.md` for the issue ledger.
