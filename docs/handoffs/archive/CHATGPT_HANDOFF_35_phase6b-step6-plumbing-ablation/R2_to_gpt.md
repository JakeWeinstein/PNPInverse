# Round 2 counterreply — Step 6 plumbing ablation plan

## Section 1 — Acknowledgments (per R1 issue)

### Issue 1 (override not wired through Picard + diagnostics)
**Accept.** Centralizing the σ_singh override on `ctx` is the right fix.

Concrete plan: introduce
`ctx["_cation_hydrolysis_sigma_singh_override_counts_pm2"]` (set by
`set_reaction_sigma_singh_override_model(ctx, value | None)` accessor,
analogous to the existing `set_reaction_*_model` family).  Three call
sites consume it:

1. `forms_logc_muh.py` + `forms_logc.py` — when override active, build
   `pka_shift_expr` via the override path (see Issue 9 below).
2. `update_gamma_from_solution` — uses the same `build_pka_shift(...)`
   on the override path so Picard's Γ_ss formula uses the same ΔpKa.
3. `collect_v10a_rung_diagnostics` — same path so `pka_shift_avg` and
   `F0_decomposition.pka_factor_avg` reflect the override.

Add an integration test that asserts the three values agree at λ=1
when the override is active.

### Issue 2 (A3 self-contradictory on coupling)
**Accept.** Re-framing A3 in the plan:

**A3 is a RESIDUAL-PATH imposed-σ ablation.**  The override propagates
into `pka_shift_expr` → `build_proton_boundary_source` → `R_net`, so
the physical residual is computed with σ_singh fixed at the override
value instead of σ_singh = max(0, −σ_S_solved) · 6.2415e-6.

**The σ-mapping decoupling A3 verifies is**: residual `R_net` should
respond to the OVERRIDE (constant in counts/pm²) and **not** to the
PNP-solved σ_S(V).  In a clean run with σ_solved ≈ −0.017 C/m² at
V_kin (anodic-clamped to 0 counts/pm² by the existing `max(0, −σ_S)`
clamp) and override = SIGMA_SINGH_PLUMBING_SENTINEL (large), R_net
should respond to the sentinel.  This is the discriminator.

### Issue 3 (A1/A2 can't catch broken physical R_net path)
**Accept.** Adding a dedicated **physical-path sentinel**:

**A0b — Physical path zero check.**  Force the *physical* R_net to 0
by setting `cation_hydrolysis_bundle.k_hyd_func = 0.0` at A0 defaults
(no manufactured R_inj, no override) and comparing to A0
(`k_hyd = 1e-3`).  Expected: c_H at boundary differs measurably (the
A.2 baseline has γ=0.0405 with k_hyd=1e-3; setting k_hyd=0 ⇒ γ=0
⇒ no R_net injection ⇒ different c_H at the OHP).  Pass criterion:
`|Δc_H_vs_A0b_rel| > 1%` (sign-only check; the sign depends on
whether the physical R_net is net positive or negative at V_kin).

This sentinel runs the *physical* path with a quantitative knob that
should produce a measurable difference, catching a bug where
`build_proton_boundary_source` silently returns 0.0.

A0b replaces my earlier "manufactured path bypasses physical bug"
concern with a positive test of the physical path.

### Issue 4 (manufactured path still builds R_net_default)
**Accept.** Trivial fix in form-build:

```python
manufactured_R_inj = conv_cfg.get("manufactured_R_inj", None)
if manufactured_R_inj is not None:
    R_net = fd.Constant(float(manufactured_R_inj))
    # Skip building R_net_default — it's not used and could crash
    # on a physical-path bug we don't care about for A1/A2.
else:
    pka_shift_expr = ...
    R_net_default = build_proton_boundary_source(...)
    R_net = R_net_default
```

Same change in both `forms_logc_muh.py` and `forms_logc.py`.

### Issue 5 (apply_h_source/apply_k_sink not wired to Picard Γ update)
**Accept with refinement.**  Adding a validation guard:

* `apply_h_source` and `apply_k_sink` are valid **only** with
  `manufactured_R_inj is not None` (A1/A2).
* Driver raises if `apply_h_source=False` or `apply_k_sink=False` is
  set with `manufactured_R_inj=None`.

**Why Γ Picard doesn't need to be flag-aware for the manufactured
path:** in the manufactured path, `R_inj` is a *constant*, not
Γ-dependent (see `cation_hydrolysis.py:857-861`).  The residual uses
`λ·R_inj` regardless of Γ.  Γ's value is irrelevant to the residual.
Picard's manufactured-path closed form `Γ = λ·R_inj /
(λ·k_des + (1−λ))` gives the steady-state Γ value but doesn't feed
back into the residual.  So `apply_h_source=False` with
`manufactured_R_inj=X` produces a residual where the H+ source term
is omitted, independent of what Γ Picard outputs.

(For A3, the override is physical-path AND Γ-coupled, which is what
Issue 1's centralization addresses.)

### Issue 6 (A.2 diagnostics invalid for A1/A2)
**Accept.** Pass criteria for A1/A2 do NOT gate on
`mass_balance_residual_rel`, `F0_avg`, `amp_from_singh`, or
`amp_from_c_K`.  Those are physical-path diagnostics that don't apply
to manufactured runs.

A1/A2 emit a `manufactured_R_inj_used` field and a
`physical_diagnostics_skipped: true` flag in the per-ablation JSON
record.  The driver's `classify_ablation_status` ignores physical
diagnostics for these ablations and only gates on the c_H/c_K shifts
+ Newton convergence + concentration positivity.

For A0 / A0b / A3 (physical path), full A.2 diagnostics apply
verbatim.

### Issue 7 (σ conversion repo inconsistencies)
**Accept.**  Verified via grep — there are stale references:
* `tests/test_phase6b_v10a_langmuir_cap.py:297-304` uses `0.226 C/m²`
  (not `2.26 C/m²`); test expected value `≈ 1.41e-6` is internally
  consistent (`0.226 · 6.2415e-6 = 1.41e-6`) but does NOT correspond
  to Singh's Cu σ of 226 µC/cm² = 2.26 C/m² → 1.41e-5 counts/pm².
* `docs/phase6/singh_2016_pka_formula.md:241` lists `0.141` in the
  σ column — likely in counts/Å² (1 Å² = 1e4 pm², so 0.141
  counts/Å² = 1.41e-5 counts/pm² = 2.26 C/m² ✓).  Unit label
  ambiguous in the doc.

**Step 6 prerequisite (must land before A3 runs):** unit-label audit
of these three files.  Add a section to the plan listing the
prerequisite as a "do this first" item, with a one-paragraph fix
spec:

* `Forward/bv_solver/units.py`: confirm `sigma_C_m2_to_counts_pm2(σ)`
  returns counts/pm² (the codebase's chosen unit).  Add unit
  docstring.
* `tests/test_phase6b_v10a_langmuir_cap.py:297`: re-derive the
  σ value (likely a typo; either 2.26 C/m² → 1.41e-5 or 0.226 C/m² is
  intentional and not Singh-Cu-equivalent).
* `docs/phase6/singh_2016_pka_formula.md:241`: clarify σ column
  units; if currently counts/Å², state explicitly; if intended
  counts/pm², the values are wrong by 1e4.

### Issue 8 (Singh deck override is tiny — A3 plumbing weak)
**Accept.**  Splitting into two constants:

* **`SIGMA_SINGH_K_CU_DECK = 1.41e-5` counts/pm²** — deck Singh K⁺/Cu
  value.  Used in a separate ONE-LINE unit test that asserts the
  conversion path is correct; not used in A3.

* **`SIGMA_SINGH_PLUMBING_SENTINEL = 1.0` counts/pm²** — large
  sentinel value chosen so that:
  - `β_K_Cu · 1.0 ≈ -45.61` (per GPT's β ≈ -45.61 per counts/pm²)
  - `10^(+45.61) ≈ 4e45` — that's way too large; saturates clip.
  
  Actually with sentinel = 1.0, the Singh ΔpKa would be ~+45.6 and
  the pka_factor = 10^(-45.6) ≈ 0, which would zero out R_net.
  That's the opposite of "measurable response".
  
  Better sentinel: pick a value that gives a "1 unit shift in ΔpKa",
  i.e., `β · σ = ±1.0`.  With β ≈ -45.61, sentinel ≈ 0.022
  counts/pm².  This corresponds to σ_S ≈ 3500 C/m² (way unphysical
  but that's the point of a sentinel — it just exercises the
  plumbing).

  Final: **`SIGMA_SINGH_PLUMBING_SENTINEL = 0.022` counts/pm²**
  (gives `|ΔpKa| ≈ 1.0`, `pka_factor ≈ 10^(-1) = 0.1`).  R_net
  response to override should be ~10× smaller than A0's R_net.  This
  is measurable AND keeps Newton in a well-conditioned regime.

(I'm not 100% sure about the β sign — please push back if cathodic
ΔpKa should be NEGATIVE (lowering pKa, more proton produced).
Looking at `cation_hydrolysis.py:528`: `ΔpKa = 2·A·z·r_H_El·G·σ_singh`
where `G = (1 − r_M-O²/r_H_El²)` is positive for `r_H_El > r_M-O`
(typical case), so ΔpKa is positive (pKa increases, less proton
produced) when σ_singh > 0.  But Singh says cathodic pKa LOWERING.
Reconcile: with `σ_singh = max(0, -σ_S)`, cathodic σ_S < 0 →
σ_singh > 0 → ΔpKa < 0?  No — formula has `+2·A·z·...·σ_singh`,
positive σ_singh gives ΔpKa > 0.  This conflicts with cathodic
lowering.  EITHER the formula in code has a sign error OR Singh's
convention has σ_singh negative for cathodic and the clamp is
wrong.  Open: verify sign convention before A3 runs.)

### Issue 9 (build_pka_shift_from_override wrong parameter names)
**Accept.**  Dropping the new helper.  Implementation:

```python
# In forms_logc[_muh].py:
sigma_singh_override = conv_cfg.get(
    "override_sigma_singh_counts_pm2", None,
)
if sigma_singh_override is not None:
    # Existing build_pka_shift takes a signed σ_S (C/m²) and
    # applies max(0, -σ_S) · 6.2415e-6 to get counts/pm².  To pin
    # σ_singh_counts to `sigma_singh_override` (positive scalar),
    # pass a signed σ_S = -override / 6.2415e-6 (negative C/m²
    # value).  Then max(0, -σ_S) = max(0, +override/6.2415e-6)
    # = override/6.2415e-6.  Then conversion · 6.2415e-6 gives
    # override.  ✓
    FACTOR = 1.602176634e-19 / 1e-24  # = 1.602e5; reciprocal of 6.2415e-6
    fake_signed_sigma_S = fd.Constant(
        -float(sigma_singh_override) * FACTOR
    )
    pka_shift_expr = build_pka_shift(
        cation_params=cation_hydrolysis_bundle.cation_params,
        sigma_S=fake_signed_sigma_S,
        r_H_El_func=cation_hydrolysis_bundle.r_H_El_pm_func,
    )
else:
    pka_shift_expr = build_pka_shift(
        cation_params=cation_hydrolysis_bundle.cation_params,
        sigma_S=sigma_S_expr,
        r_H_El_func=cation_hydrolysis_bundle.r_H_El_pm_func,
    )
```

The factor `1.602e5` is the reciprocal of `6.2415e-6` (1/e × 1e24 →
e × 1e-24 = 1.602e-19 × 1e24 = 1.602e5).  No new helper, no
parameter naming drift, no formula duplication.

For Picard + diagnostics consumers (Issue 1), apply the same
override→fake_signed_sigma_S transform.  Best place: factor this
into `build_pka_shift_via_override_or_solved(cation_params, ...)`
helper that takes `override_counts_pm2 | None` AND `sigma_S_solved`,
returns the appropriate `pka_shift_expr`.  Single helper, used by
forms / Picard / diagnostics.

### Issue 10 (A3 pass criterion tautological)
**Accept.**  Adding a **residual-path** check:

A3 pass criteria (revised):

1. **`pka_shift_avg_A3 = β_K_Cu · σ_override` within rel 1%**
   (diagnostic-path correctness).
2. **`pka_factor_avg_A3 ≈ 10^(-β·σ_override)` within rel 5%**
   (this is `c_H_avg · pka_factor` divided by `forward_avg_no_k_hyd /
   k_hyd / c_M_avg` — i.e., the residual-side `pka_factor` matches
   the override prediction).
3. **`R_net_A3 / R_net_A0 ≈ pka_factor_A3 / pka_factor_A0` within
   rel 10%** (residual side responds to the override the way the
   formula says).
4. **`σ_S_C_per_m2_A3 - σ_S_C_per_m2_A0` falls in a bounded range**
   (the override changes R_net which changes c_H near OHP which
   weakly changes σ_S via Bikerman+Stern; this IS expected coupling
   per Risk #4 acknowledgment, NOT a "leak" gate).

Replaces the original tautological `|pka_shift_avg - β·σ_override|
< 5%` + ad-hoc 10% σ_S leak gate.

### Issue 11 (5%/1% thresholds not defensible)
**Accept.**  Reframing all pass criteria with sign + magnitude +
convergence:

* **A0 pass**: end-to-end reproduces A.2 baseline at k_hyd=1e-3
  with all defaults; per-observable rel tolerance 1e-9 (PETSc noise
  band) for γ, θ, σ_S, cd_mA_cm² (NOT byte 1e-12; see Issue 14).
  Newton converged; mass-balance rel < 5e-3.
* **A0b pass**: `Δc_H_vs_A0_rel` has measurable magnitude (`> 1%`);
  sign matches "no R_net injected" direction; Newton converged;
  c_H ≥ 0.
* **A1 pass**: `Δc_H_vs_A0_rel > +1%` (sign + magnitude); Newton
  converged; c_K(boundary) ≥ 0 (positivity).
* **A2 pass**: `Δc_K_vs_A0_rel < -1%` (sign + magnitude); Newton
  converged; c_H(boundary) ≥ 0.
* **A3 pass**: all four gates above (Issue 10).

The "1% magnitude" floor is a single-run noise estimate from the
existing solver's SNES rtol=1e-10 + Picard rel_tol=1e-4 — values
that change at <1% rel between identical runs are dominated by
floating-point noise.  No replicate noise estimates are needed; the
1% gate is a structural floor.

The "3σ < 1%" language from R1 is dropped.

### Issue 12 (R_inj bracketing only A1, no upper bound)
**Accept.**  Joint bracket selection for A1+A2:

Pre-pass runs A1 and A2 at each of `{1e-2, 1e-1, 1.0}` nondim (6
total ramps, ~4 min wall).  Selection criterion:

* Smallest R_inj such that **both** A1 and A2 satisfy:
  - `5% ≤ |Δc_H| ≤ 25%` (A1) and `5% ≤ |Δc_K| ≤ 25%` (A2)
  - Newton converged
  - c_H, c_K at boundary ≥ 0

If no R_inj in the bracket passes both, escalate to `{2.0, 5.0,
10.0}` (hard ceiling 10.0 nondim ≈ 250× A0's R_net).  If the
ceiling fails, report inconclusive — do NOT pass step 6.

### Issue 13 (soft clamp on c_K is bad mitigation)
**Accept.**  Dropping the soft clamp.  Newton failure for A2 with a
given R_inj is a SIGNAL that the manufactured break is overdriving
the system at that R_inj.  Mitigation chain:

1. λ continuation: ramp λ=0 → 0.25 → 0.5 → 0.75 → 1.0 (same
   AdaptiveLadder as A.2).  If Newton fails at λ=1, try smaller
   R_inj.
2. Lower R_inj (next-smallest in the bracket).
3. If no R_inj in the full bracket converges, report inconclusive.

No PDE modification.  No soft clamp.  Risk #10 is rewritten.

### Issue 14 (byte-equivalence 1e-12 too brittle)
**Accept.**  Tiered tolerance table:

| Comparison | Tier | Tolerance | Action on fail |
|---|---|---|---|
| Pure-helper unit test (same process, no Firedrake) | ≤ 1e-12 | exact | block |
| End-to-end A0 vs A.2 baseline | ≤ 1e-9 | pass | continue |
| End-to-end A0 vs A.2 baseline | 1e-9 to 1e-6 | re-run; if reproducible at <1e-6, continue with documented PETSc-determinism note | document |
| End-to-end A0 vs A.2 baseline | > 1e-6 | block; debug residual-side wiring | block |

Per-observable basis (γ, θ, σ_S, cd, c_H, c_K) — not a single scalar
RMS.

### Issue 15 (fast tests overclaim)
**Accept.**  Re-classified tests:

**Fast (pure Python, no Firedrake) — `tests/test_phase6b_step6_plumbing_ablation.py`**:
* `_parse_args` parsing of `--r-inj-prepass`, `--sigma-singh-override`,
  `--ablations A0,A0b,A1,A2,A3`.
* `_parse_args` rejects invalid override values (negative, NaN, str).
* `classify_ablation_status` covers each pass/fail combination per
  ablation.
* `_build_ablation_sp_overrides` returns the right config keys per
  ablation_id.
* `_select_r_inj_bracket(prepass_results)` picks the right value
  from synthetic prepass data.
* Override factor inverse: `_override_to_signed_sigma(override)` →
  `signed_sigma`; verify `build_pka_shift`-style algebra gives back
  the override post-clamp.

**Slow (Firedrake required) — same file or split into `*_slow.py`**:
* UFL form comparison: with `apply_h_source=False`, the H+ residual
  contribution is absent from `F_res`.
* Override path: `pka_shift_expr` symbolically equals
  `β · sigma_singh_override` after compilation.
* End-to-end A0 reproduces A.2 baseline at k_hyd=1e-3 within the
  tiered tolerance.
* End-to-end A0b (k_hyd=0) shows c_H shift from A0 by ≥ 1%.

### Issue 16 (bool / override validation underspecified)
**Accept.**  Validation in `config.py`:

```python
apply_h_source = _bool(raw.get("apply_h_source", True))
apply_k_sink   = _bool(raw.get("apply_k_sink",   True))
override_raw = raw.get("override_sigma_singh_counts_pm2", None)
if override_raw is None:
    override_sigma_singh_counts_pm2 = None
else:
    override_sigma_singh_counts_pm2 = float(override_raw)
    if not math.isfinite(override_sigma_singh_counts_pm2):
        raise ValueError(
            "override_sigma_singh_counts_pm2 must be finite "
            f"(got {override_raw!r})"
        )
    if override_sigma_singh_counts_pm2 < 0.0:
        raise ValueError(
            "override_sigma_singh_counts_pm2 must be ≥ 0 "
            f"(counts/pm² is a post-clamp non-negative scalar; "
            f"got {override_raw!r}); use None to disable."
        )
```

`_bool` already exists in `Nondim.transform` (used throughout
`config.py`).

### Issue 17 (A3 σ_S leak rule not clean detector)
**Accept.**  Structural test in slow-test tier:

UFL-introspection test:
```python
# After build_forms, inspect the residual:
# 1. The Stern σ_S coefficient in F_res must depend on
#    ctx['U'] (the solved PNP fields).
# 2. The pka_shift_expr used inside F_res must NOT depend on
#    ctx['U'] when override is active — it should be a Constant.
```

This is a symbolic check using `ufl.algorithms.extract_coefficients`.
Replaces the original numerical 10% σ_S leak gate (which was both
weak — could pass with a leak — and brittle — could fail with
expected feedback).

The numerical gate stays for monitoring but is downgraded to
"observation, not pass criterion".

---

## Section 2 — Updated plan (deltas)

The full plan in R1 stands except for the following revisions.  I'm
not re-pasting all 539 lines; the codex session has them in memory.

### New section: "Prerequisites (before step 6 runs)"

Add a §0 to the plan:

* **σ-conversion audit** (resolving Issue 7):
  * Audit `tests/test_phase6b_v10a_langmuir_cap.py:297-304` and
    `docs/phase6/singh_2016_pka_formula.md:241` for unit-label
    consistency with `Forward/bv_solver/cation_hydrolysis.py:516`'s
    `sigma_C_m2_to_counts_pm2` convention.
  * Lands as a 1-2 hour PR before step 6 driver runs.

* **Singh sign convention audit** (resolving Issue 8 subnote):
  * Verify `cation_hydrolysis.py:528`'s `+2·A·z·r_H_El·G·σ_singh`
    formula sign is consistent with Singh's cathodic-pKa-lowering
    convention.
  * If the sign is wrong, fix before step 6.

### Revised flags table

| Flag | Type | Default | Effect | Valid combos |
|---|---|---|---|---|
| `apply_h_source` | `bool` | `True` | Omit H+ residual term when False | requires `manufactured_R_inj is not None` |
| `apply_k_sink` | `bool` | `True` | Omit K+ residual term when False | requires `manufactured_R_inj is not None` |
| `override_sigma_singh_counts_pm2` | `Optional[float]` | `None` | When set, replace σ_S in `build_pka_shift` via fake-signed-σ trick (Issue 9 sketch) | physical path only (no manufactured_R_inj) |

Validation in driver:
```python
if (override_sigma_singh_counts_pm2 is not None
    and manufactured_R_inj is not None):
    raise ValueError(
        "override + manufactured_R_inj is undefined: override is "
        "physical-path; manufactured bypasses physical path."
    )
if ((not apply_h_source or not apply_k_sink)
    and manufactured_R_inj is None):
    raise ValueError(
        "apply_h_source=False or apply_k_sink=False requires "
        "manufactured_R_inj to be set (these flags only make sense "
        "for manufactured ablations)."
    )
```

### Revised ablation table

| Ablation | apply_h_source | apply_k_sink | manufactured_R_inj | override_σ_singh | Expected |
|---|---|---|---|---|---|
| **A0** (baseline) | True | True | None | None | Reproduces A.2 baseline at k_hyd=1e-3 within tiered tolerance |
| **A0b** (physical-path sentinel) | True | True | None | None | k_hyd=0 ⇒ Γ=0 ⇒ no R_net ⇒ c_H differs from A0 by ≥1% |
| **A1** (source-only) | True | False | R_INJ_MFG (bracketed) | None | c_H rises 5-25%; positivity + Newton converged |
| **A2** (sink-only) | False | True | R_INJ_MFG (same value) | None | c_K falls 5-25%; positivity + Newton converged |
| **A3** (σ_singh override) | True | True | None | SIGMA_SINGH_PLUMBING_SENTINEL = 0.022 counts/pm² | All four gates from Issue 10 |

### Revised pass criteria (table)

(See Issue 11 ack above — replaces original plan §12.)

### Revised R_inj bracket pre-pass

(See Issue 12 ack — joint A1+A2 selection.)

### Revised risks

* Risk #1 (σ-counts conversion ambiguity) → RESOLVED in
  Prerequisites.
* Risk #10 (soft clamp) → REMOVED (Issue 13 — no PDE modification).
* New Risk #11: Singh sign convention (Issue 8 subnote).  Mitigate
  via Prerequisites audit before step 6 runs.
* New Risk #12: A0b's "k_hyd=0 ⇒ measurable c_H shift" requires the
  physical R_net at A0 to actually move c_H at the OHP.  If
  A.2's k_hyd-independence claim turns out to be a TRUE clean
  physics result AND R_net is negligible at V_kin for all k_hyd
  (not just k_hyd-independent but also k_hyd-small in magnitude),
  A0b's 1% shift gate could fail with no bug.  Mitigation:
  pre-check at plan time using the A.2 record's
  `R_net_A0_nondim = k_des · γ = 1.0 · 0.0405 = 0.0405` (4% of
  unity O₂ flux) — that's >>1% expected, so A0b SHOULD see a
  measurable shift.  If A0b fails at 1%, that's evidence of the
  physical-path bug we're hunting for.

---

## Section 3 — Continued critique prompt

Review the updated plan and my responses to your earlier issues.
Push back on responses where I defended poorly — name which point.
Raise any new issues the updated plan creates. Re-issue any earlier
issue you don't think I addressed. Same numbered format and same
verdict line at the end:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN
