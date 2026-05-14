# Final revision — Critique session 38

**Revised artifact:** `/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/docs/solver/mms_pnpbv_muh_multi_ion_stern_derivation.md`
**Final verdict:** APPROVED at round 4
**Issues raised across rounds:** 12 + 10 + 5 + 4 = **31**
**Accepted:** 31 / 31
**Defended:** 0 / 31
**Unresolved:** 0 / 31

The loop converged steadily (issue counts 12 → 10 → 5 → 4). All 31 raised
issues were real flaws in the original derivation; none were rejected.
The fully-revised artifact is byte-faithful to the production code paths
in `Forward/bv_solver/forms_logc_muh.py`, `boltzmann.py`, and
`scripts/_bv_common.py`.

---

## Round 1 (12 issues, all accepted)

1. **NP electrode flux sign in §4.2 hand derivation** — accepted; corrected
   `J_i^ex · n |_{y=0} = −D_i · c_i^ex · ∂_y(...)` (was `+D_i · ...`).
   Lands at §4.2 of revised artifact.
2. **O₂ BV sign interpretation (Jprod vs physical N)** — accepted; added
   "Mobility vs physical flux" paragraph distinguishing
   `Jprod = +D · c · ∇(...)` (production) from `N = −Jprod` (physical).
   Lands at §4.2.
3. **"Exactly 0" overstated for time term** — accepted; reworded to
   "negligible by construction (O(h²/dt))" with `dt = 1e15` rationale.
   Lands at §1 "Time-stepping made negligible".
4. **muh-aware test harness reuse** — accepted; added explicit
   "muh-aware test harness" subsection requiring `μ_H^ex` (NOT `u_H^ex`)
   for the proton interpolation, error norms, and concentration
   diagnostics. Lands at §7.
5. **η arithmetic in §5.1 wrong** — accepted; recomputed for both
   recommended and broad envelopes. Lands at §5.1.
6. **Packing-floor proof overclaims** — accepted; recomputed
   θ_inner^ex bounds for recommended envelope (≳ 0.97) and broad envelope
   (≳ 0.84). Lands at §5.4.
7. **Clamp/floor coverage overstated** — accepted; reworded to
   "all clamps/floors are identity at u_exact; their branch logic is
   not exercised". Lands at §5 introductory text and §6.
8. **Hard asserts on stack assumptions** — accepted; added explicit
   `assert` block in the source builder for use_reactions, bv_log_rate,
   use_stern, suppress_poisson_source, water/cation hydrolysis off,
   species count, counterion count, no Γ slot, dt threshold. Lands at §7.
9. **K0_R4e_factor not pinned** — accepted; pinned (later revised in R3
   to **K0_R4e_factor = 1e−18** rather than R1's first-pass `1.0` choice).
   Lands at §5.5.
10. **`bv_c_ref_model_vals` and c_ref-anchored anodic branch uncovered**
    — accepted; added explicit "does NOT cover" rows in §6 and §8.
11. **Live continuation parameters can desynchronize source** —
    accepted; added "Live continuation policy" assertion that no
    setter/mutator fires between source build and solve. Lands at §7.
12. **Quadrature degree asserted, not demonstrated** — accepted; added
    one-time quadrature-degree sweep recommendation at §7.

## Round 2 (10 issues, all accepted)

1. **Wrong production a_nondim values in §5.4** — accepted; replaced
   radius-derived numbers (Cs r=3.3 Å, SO4 r=3.0 Å) with verified
   production constants from `_bv_common.py:707–708`
   (`A_CSPLUS_HAT = 3.23e−5`, `A_SO4_HAT = 4.20e−5`, with C_SCALE = 1.2).
   Lands at §5.4.
2. **Asserts not implementable as written** — accepted; corrected dict
   lookups: `use_reactions` ⇒ `len(scaling['bv_reactions']) == 2`;
   `suppress_poisson_source` lives on `nondim_cfg` (NOT `conv_cfg`);
   SNES tolerances live on `solver_params[10]`. Lands at §7.
3. **θ-min check mathematically wrong** — accepted; replaced
   `fd.assemble(min_value(θ, t)*dx)` (which is integral of clipped field,
   NOT minimum) with discrete-min via `conditional` indicator. Lands at §5.4.
4. **R4e magnitude assert dimensionally meaningless** — accepted; replaced
   pointwise `|R_R4e^ex(0.5,0)| > 10·atol` with assembled boundary
   L2-norm comparison vs R2e norm. Lands at §5.5 and §7.
5. **Coverage row gate wrong** — accepted; legacy `bv_c_ref_model_vals`
   path is gated by `use_reactions=True` (NOT `bv_log_rate=True`).
   Lands at §6.
6. **Stern-off cross-check still wrong** — accepted with restructure;
   in R2 the proposed `φ^ex_NoStern = (1−y)·φ_app + γ·y(1−y)·cos(πx)`
   was offered, then in R3 the entire Stern-off cross-check was dropped
   when GPT showed it saturates SO₄²⁻ closure at production V_RHE.
7. **Wrong ctx key + production-bundle independence** — accepted; removed
   `ctx['boltzmann_bundles']` claim (no such key exists); added explicit
   "Independence policy" requiring source builder to compose its own UFL
   from config rather than consuming production bundles. Lands at §2.1
   and §7.
8. **Continuation coverage overclaim** — accepted; moved Stern setter /
   continuation coverage from "catches" to "does NOT cover" in §6.
9. **Counterion clamp identity stated incorrectly** — accepted;
   production clamps `φ` (not `z·φ`), so identity condition is
   `max|φ^ex| < phi_clamp_k`. Lands at §5.3.
10. **Quadrature policy internally inconsistent** — accepted; opening
    abstract changed from "SRC_QUAD_DEGREE=8 carries over" to "initial
    candidate, pinned by sweep". Lands at opening abstract and §4.1.

## Round 3 (5 issues, all accepted)

1. **R4e catastrophically dominates at K0_R4e_factor=1** — accepted;
   re-derived: R4e/R2e ~ e^46 ≈ 10^20. Pinned K0_R4e_factor = 1e−18
   (matches demo's smallest factor; brings R4e/R2e to ~165, manageable).
   Replaced "R4e at least 10% of R2e" with finite window `10 < R_ratio < 1e5`.
   Removed K0_R4e_factor=1 secondary cross-check from primary scope.
   Lands at §5.5 and §7.
2. **Stern-off cross-check broken by SO₄²⁻ saturation** — accepted;
   at φ_app ≈ 21.4, q_SO4 = exp(42.8) ≈ 4·10^18 saturates closure ⇒
   packing_floor activates ⇒ contradicts "clamps inactive" premise.
   **Removed `TestSternOffSanity` from the plan entirely.** Lands at §5.5
   and §7.
3. **Stern coefficient coverage row internally inconsistent** —
   accepted; reworded to "**use** of ctx-stored coefficient" not "value
   of nondim coefficient". Lands at §6.
4. **§5.4 overstates A_dyn^ex** — accepted; A_dyn ≲ 3e−5 (not "≪10⁻⁵"),
   free_dyn = 1 − A_dyn ≈ 0.99997 (not "identical to 1"). Source
   builder must keep (1−A_dyn) exactly. Lands at §5.4.
5. **`set_phi_applied` reference probably nonexistent** — accepted;
   replaced with "runtime mutation of `ctx['phi_applied_func']`
   (voltage continuation)". Lands at §6 and §8.

## Round 4 (4 cleanup items, all accepted, VERDICT: APPROVED)

1. **§5.4 proton concentration typo** — accepted; corrected
   `c0_H_nondim = 0.0833` (was incorrectly `8.3e−5`, off by 10³).
   Conclusion (`A_dyn ≲ 3e−5`) unchanged. Lands at §5.4.
2. **Counterion concentration units sloppy** — accepted; made M → mol/m³
   conversion explicit (`200 mol/m³ / 1.2 mol/m³` instead of
   `0.2 M / 1.2 mol/m³`). Lands at §5.4.
3. **§2.7 natural-BC notation typo** — accepted; replaced
   "`∇·n = 0` implicit BC" with the correct natural conditions
   "`J_i · n = 0` for each species and `∇φ · n = 0` for Poisson".
   Lands at §2.7 and §3.2.
4. **Use domain-qualified `dx` in invariant snippets** — accepted;
   replaced bare `fd.dx` with `fd.dx(domain=mesh)` in all assertion
   code blocks. Lands at §5.4 and §7.

---

## Substantive structural changes from the loop

Beyond the 31 individual fixes, three **structural** decisions emerged
from the critique:

1. **K0_R4e_factor pinned to 1e−18, not 1**: avoids the e^46 R4e/R2e
   dominance regime. Lands at §5.5.
2. **Stern-off cross-check removed entirely**: SO₄²⁻ saturation at
   production V_RHE makes the cross-check incompatible with the
   "clamps inactive" baseline premise. Lands at §5.5, §7, §8.
3. **Independence policy made explicit**: source builder composes its
   own UFL for the multi-ion closure from config; does NOT consume
   production-side `steric_boltz` bundle expressions. Lands at §2.1, §7.

Each of these would have broken the implementation later if not caught
during derivation.
