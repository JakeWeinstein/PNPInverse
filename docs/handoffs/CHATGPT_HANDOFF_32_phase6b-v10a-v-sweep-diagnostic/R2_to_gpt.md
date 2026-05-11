# Round 2 counterreply — V-sweep diagnostic plan

## 1. Acknowledgment per issue

### Re your point 1 — column is mislabeled. **Accept.**

You're right. The estimate is `(dR/dlogC_S) / (dσ_S/dlogC_S)` evaluated
on the Stern-capacitance-perturbation manifold at fixed V_RHE, k_hyd,
k_prot, k_des, Γ_max, r_H_El. Calling it `dR_net/dσ_S` invites Phase E
calibrators to read it as a partial derivative, which it isn't.

**Fix in plan:**

- Rename JSON key `dRnet_dsigma_along_stern_capacitance` (verbose but
  unambiguous). Driver also logs the intermediate quantities so the
  reader can reconstruct the ratio:
  - `dRnet_dlogCs`, `dsigma_dlogCs`
  - `epsilon = 0.05` (Stern-capacitance fractional step)
  - `Cs_perturbed = (Cs·(1−ε), Cs·(1+ε))`
  - `sigma_S_perturbed`, `R_net_perturbed`
  - `fixed_at_perturb`: `["V_RHE", "k_hyd", "k_prot", "k_des", "Γ_max", "r_H_El", "δ_OHP"]`
  - `relaxed_at_perturb`: `["φ", "U(species)", "Γ via Picard"]`
- Add a docstring on `select_v_kin` that names the column literally as
  "Stern-capacitance-manifold total derivative" and notes that a
  true partial `∂R_net/∂σ_S` would require frozen-fields evaluation
  (out of scope for v10a).

### Re your point 2 — C_S ≡ r_H_El equivalence is wrong. **Accept, but narrowly.**

To be precise about what I wrote earlier: in the R1 context bundle I
listed `(a)` and `(c)` as separate options and even noted in `(c)`
that perturbing `r_H_El_pm` "does NOT directly shift σ_S, however —
only shifts F₀ via 10^(-ΔpKa)". So I had distinguished them at the
mechanism level. What I conflated is the framing "calibration-knob
sensitivity"; that's the part you're right to flag.

**Fix in plan:**

- Plan documents the C_S perturbation strictly as **Stern-capacitance
  leverage** sensitivity. No mention of "calibration knob" or
  "equivalent to fitting Singh r_H_El".
- Phase E calibration knobs (Singh r_H_El, C_S) get their own
  sensitivity diagnostics if the plan A.2 / D / E phases need them.
  Out of scope for v10a.
- `dRnet_dr_H_El` is **not** added to v10a; deferred to Phase A.2
  (which runs at V_kin and needs identifiability against the
  primary calibration knobs).

### Re your point 3 — 50 % FD-vs-perturb exclusion isn't defensible. **Accept.**

You're right that FD and perturbation measure different total
derivatives along different constraint manifolds, so disagreement is
*path dependence*, not numerical noise. Hidden-fourth-filter argument
is fair: the acceptance bundle locked exactly three filters, so any
silent exclusion outside them is rule drift.

**Fix in plan:**

- Drop the FD-vs-perturbation 50 % exclusion entirely.
- Both columns logged informationally (and on the plot). A
  `path_mismatch_relative` column equal to
  `|FD − perturb| / max(|FD|, |perturb|, ε)` is logged for the
  reader, but **not** used as a filter.
- Numerical quality filter replaced by one-sided slope agreement
  on the perturbation column itself:
    - `S_minus = (R_net(Cs·(1−ε)) − R_net(Cs)) / (σ_S(Cs·(1−ε)) − σ_S(Cs))`
    - `S_plus  = (R_net(Cs·(1+ε)) − R_net(Cs)) / (σ_S(Cs·(1+ε)) − σ_S(Cs))`
    - `one_sided_disagreement = |S_plus − S_minus| / max(|S_plus|, |S_minus|, ε_quality)`
- Primary candidate filter (new `sensitivity_quality_ok`):
  `one_sided_disagreement ≤ 0.25`.
- Fallback filter: `one_sided_disagreement ≤ 0.50`.
- Plus a minimum `|σ_+ − σ_−| ≥ σ_min_threshold` (σ_min_threshold =
  median(|σ_+ − σ_−|) / 10 across V — adaptive floor; if all V are
  σ_S-clamped, this filter is no-op).
- Plus all three perturbed solves must converge (driver records
  `perturbation_converged: bool` per V).

### Re your point 4 — Levich D_O2 typo. **Accept.**

The plan's docstring quoted `D_O2 = 2.18e-9` (water-literature value);
the codebase uses `D_O2 = 1.9e-9` in `scripts/_bv_common.py:73`. At
`l_eff = 16e-6`, the correct `I_lim_4e ≈ 5.50 mA/cm²`.

**Fix in plan:**

```python
def _i_lim_4e_mA_cm2(l_eff_m):
    """I_lim_4e_phys = 4 · F · D_O2 · c_O2 / l_eff_m, converted to mA/cm².

    Imports D_O2, C_O2 from _bv_common to stay in lockstep with
    I_SCALE and the rest of the stack.  At l_eff = 16 µm this returns
    ≈ 5.50 mA/cm².
    """
    from scripts._bv_common import D_O2, C_O2
    F = 96485.3329
    return 4.0 * F * D_O2 * C_O2 / l_eff_m * 0.1     # A/m² → mA/cm²
```

### Re your point 5 — `|cd|/I_lim_4e` asymmetric for parallel 2e/4e. **Accept, with rule-preservation caveat.**

Your algebra is right: at O₂-flux limit,
`|cd|/I_lim_4e = (1 + x_4e) / 2`, so the 0.9 threshold is only
reachable when `x_4e ≥ 0.8` (pure-4e-rich). For 2e-rich or 50/50
plateaus the filter passes trivially — that's a real flaw in the
locked rule.

**However:** the rule is locked by the acceptance bundle (session 31
R5). Silently substituting an O₂-flux-normalised filter changes the
contract. The right move is to **implement the locked rule literally**
+ **flag the asymmetry informationally** so a downstream reader (or
follow-up plan-revision pass) sees the gap.

**Fix in plan:**

- Locked filter implemented exactly as written: `|cd|/I_lim_4e < 0.9`,
  with `I_lim_4e = 4·F·D_O2·c_O2/l_eff_m`.
- New informational column per V:
  - `o2_flux_levich_ratio = (R_2e_current + R_4e_current) / (D_O2_HAT · C_O2_HAT / domain_height_hat)`
    where the numerator is the boundary-integrated O₂ consumption rate
    (per-reaction current contributions from `collect_v10a_rung_diagnostics`),
    the denominator is the bulk Levich O₂ flux. Ratio is in [0, 1] at
    Levich limit regardless of selectivity.
- New informational flag per V:
  - `locked_filter_passes_but_o2_transport_limited`: True iff
    `|cd|/I_lim_4e < 0.9` AND `o2_flux_levich_ratio > 0.9`.
- Driver console summary prints the count of such V at the end. JSON
  records them. **No automatic exclusion** — the locked rule stays
  authoritative.
- Plan adds a "Known limitation" note pointing to the asymmetry; the
  follow-up `gsd:add-todo` (or v10a' patch) can revisit the locked
  rule with the experimental group if 2e-rich plateaus become a
  problem in Phase A.2.

## 2. Updated artifact

Changes since R1 (bulleted):

* Sensitivity column renamed `dRnet_dsigma_along_stern_capacitance`
  with intermediate quantities (`dRnet_dlogCs`, `dsigma_dlogCs`, ε,
  fixed/relaxed variable lists) logged alongside.
* Levich helper imports `D_O2, C_O2` from `_bv_common`; corrected
  value `≈ 5.50 mA/cm²` at `l_eff=16 µm`.
* FD-vs-perturbation 50 % exclusion **dropped**. `path_mismatch_relative`
  logged informationally; not a filter.
* Numerical quality replaced by one-sided slope agreement on the
  perturbation column: `≤ 0.25` (primary), `≤ 0.50` (fallback);
  plus adaptive `|σ_+ − σ_−|` floor and per-V `perturbation_converged`.
* New `o2_flux_levich_ratio` column + `locked_filter_passes_but_o2_transport_limited`
  flag; informational only. Locked filter stays literal.
* "Calibration knob equivalence" framing removed; C_S perturbation
  documented strictly as Stern-capacitance leverage.

### Revised sensitivity section (verbatim replacement)

```markdown
## Sensitivity computation (Stern-capacitance leverage)

Two columns logged; the **perturbation column** is primary for V_kin
selection. The **FD column** is informational cross-check only —
divergence between the two indicates path dependence (FD is along
the V_RHE manifold; perturbation is along the Stern-capacitance
manifold), NOT numerical noise, and does NOT exclude candidates.

| Column | What it measures | Cost |
|---|---|---|
| **Perturbation (primary)** — `dRnet_dsigma_along_stern_capacitance` | At each V, two extra solves at `C_S·(1±ε)` with `ε=0.05`, warm-restarted from the unperturbed converged U. Records `dRnet_dlogCs`, `dsigma_dlogCs`, `Cs_perturbed`, `sigma_S_perturbed`, `R_net_perturbed`, `fixed_at_perturb=["V_RHE","k_hyd","k_prot","k_des","Γ_max","r_H_El","δ_OHP"]`, `relaxed_at_perturb=["φ","U(species)","Γ via Picard"]`. The ratio `(R_net+ − R_net−)/(σ_S+ − σ_S−)` is the **Stern-capacitance-manifold total derivative**, not a partial `∂R_net/∂σ_S`. A true partial would require frozen-fields evaluation, out of scope for v10a. | ~2× wall on the λ=1 pass. |
| **FD (informational)** — `dRnet_dsigma_along_voltage` | `(R_net(V_{i+1}) − R_net(V_{i−1})) / (σ_S(V_{i+1}) − σ_S(V_{i−1}))` central difference. The V_RHE-manifold total derivative. | Free post-pass. |

### Numerical quality filter (NEW — replaces FD-vs-perturb mismatch)

For each V, compute one-sided slope agreement on the perturbation
column:

    S_minus = (R_net(Cs·(1−ε)) − R_net(Cs)) / (σ_S(Cs·(1−ε)) − σ_S(Cs))
    S_plus  = (R_net(Cs·(1+ε)) − R_net(Cs)) / (σ_S(Cs·(1+ε)) − σ_S(Cs))
    one_sided_disagreement
        = |S_plus − S_minus| / max(|S_plus|, |S_minus|, ε_quality)

Filter flags:
- `sensitivity_quality_primary`: `one_sided_disagreement ≤ 0.25 AND |σ_+ − σ_−| ≥ σ_min AND perturbation_converged`
- `sensitivity_quality_fallback`: `one_sided_disagreement ≤ 0.50 AND |σ_+ − σ_−| ≥ σ_min AND perturbation_converged`

where `σ_min = median(|σ_+ − σ_−|, all V) / 10` (adaptive floor;
prevents amplified noise where the C_S perturbation barely moves σ_S).
```

### Revised Levich section

```python
# scripts/studies/phase6b_v10a_v_sweep_diagnostic.py
from scripts._bv_common import D_O2, C_O2

F_CONST = 96485.3329  # C/mol

def _i_lim_4e_mA_cm2(l_eff_m: float) -> float:
    """4-electron Levich limit in mA/cm².

    I_lim_4e = 4 · F · D_O2 · c_O2 / l_eff_m [A/m²], times 0.1 → mA/cm².
    Imports D_O2 (1.9e-9 m²/s) and C_O2 (1.2 mol/m³) from `_bv_common`
    so this stays in lockstep with `I_SCALE` and the rest of the stack.

    At l_eff = 16 µm: I_lim_4e ≈ 5.50 mA/cm².
    """
    return 4.0 * F_CONST * D_O2 * C_O2 / l_eff_m * 0.1
```

### Revised V_kin selection rule implementation

```python
def select_v_kin(per_v_records, *, i_lim_4e_mA_cm2):
    """Implements the LOCKED acceptance-bundle rule literally:
        argmax(|dRnet_dsigma_along_stern_capacitance|) subject to {
            σ_S < 0,
            |cd|/I_lim_4e < 0.9,                     # locked threshold
            R_2e/(R_2e + R_4e) ∈ [0.05, 0.95],
            sensitivity_quality_primary,             # new numerical quality filter
        }
        with fallback (drop branch filter; relax to sensitivity_quality_fallback)
        and abort_to_v10c when no V has σ_S < 0.

    NOTE on the |cd|/I_lim_4e filter: at O₂-flux Levich limit,
    |cd|/I_lim_4e = (1 + x_4e)/2 where x_4e = R_4e/(R_2e+R_4e).
    The threshold 0.9 only triggers for x_4e ≥ 0.8 (pure-4e-rich
    plateaus).  For 2e-rich or mixed plateaus the filter passes
    trivially.  This is a documented asymmetry of the locked rule;
    `o2_flux_levich_ratio` is logged informationally to flag V where
    the locked filter passes but O₂ transport is actually saturated.
    Driver does NOT silently amend the locked rule.
    """
    # ... (rule implementation; logs decision provenance per V) ...
```

### New informational diagnostics

```python
def _compute_o2_flux_levich_ratio(record):
    """Boundary-integrated O₂ consumption / bulk Levich O₂ flux (nondim).

    Independent of branch selectivity; ratio ≈ 1 at O₂-transport limit
    regardless of R_2e/R_4e split.  Cross-check for the locked-rule
    asymmetry (see select_v_kin docstring).
    """
    o2_consumption = record["R_2e_current_nondim"] + record["R_4e_current_nondim"]
    levich_flux = D_O2_HAT * C_O2_HAT / DOMAIN_HEIGHT_HAT
    return o2_consumption / levich_flux

# Per-V flag set after the locked rule applies:
record["locked_filter_passes_but_o2_transport_limited"] = (
    locked_filter_passed and o2_flux_levich_ratio > 0.9
)
```

## 3. Continued critique prompt

Review the updated plan and my responses to your earlier issues.
Push back on responses where I defended poorly — name which point.
Raise any new issues the updated plan creates. Re-issue any earlier
issue you don't think I addressed. Same numbered format and same
verdict line at the end:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN

Scope reminder: still narrow on those three subtle physics/numerics
points (perturbation chain rule, FD/perturb cross-check threshold,
Levich formula + parallel-2e/4e asymmetry). Don't critique the locked
V_kin rule itself, driver structure, or V_RHE grid resolution.
