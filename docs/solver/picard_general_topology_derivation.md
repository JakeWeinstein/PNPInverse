# Linear-Substrate Picard Initializer for ORR-class BV Topologies

**Version:** v3 (2026-05-08, post round-2 Codex review)
**Round-1 review:** `docs/CODEX_REVIEW_PICARD_GENERAL_TOPOLOGY_DERIVATION.md`
**Round-2 handoff:** `docs/CHATGPT_HANDOFF_19_PICARD_DERIVATION_ROUND2.md`
**Round-2 review:** `docs/CODEX_REVIEW_PICARD_DERIVATION_ROUND2.md`

Source-of-truth for the matched-asymptotic Picard initializer (M3a.3
plan §3) when the BV reaction set is no longer the legacy sequential
`R_1: O₂→H₂O₂` followed by `R_2: H₂O₂→2H₂O`.

The current implementation at `Forward/bv_solver/picard_ic.py:441-563`
hard-codes the 2x2 sequential algebra.  This document derives the
N-reaction generalization that:

1. Reproduces the legacy 2x2 byte-for-byte when called with the
   sequential reaction set.
2. Specializes to the Ruggiero parallel `R_2e + R_4e` topology used by
   M3a.2/M3a.3.
3. Covers all three anodic branches the residual already supports
   (linear surface-species, affine constant, irreversible).

**Scope.**  This is not a fully general reaction-network solver.  It is
a Picard-lagged initializer in which the cathodic and anodic substrate
concentrations are linearized, while H⁺, γ, ψ_D, ψ_S, η_j, and all
non-substrate `cathodic_conc_factors` are frozen during each inner
linear solve and updated only across outer Picard iterations.  This
matches the legacy strategy and works well for ORR where the
non-substrate factor (H⁺) is itself coupled through a separate flux
balance and converges in 5-10 outer iterations.  It is **not** safe
for hypothetical reaction sets where a dynamic species enters
`cathodic_conc_factors` with a high power *and* is also strongly
modified by the rate balance — those would require a tighter coupling
than this initializer provides.  See §11.

Companion documents:

- `docs/PNP_BV_Analytical_Simplifications.md` — outer-region matched-
  asymptotic derivation; ambipolar `2 D_H` proton transport
  (lines 240-244 are the load-bearing reference).
- `docs/ruggiero_realignment_plan.md` — milestone context (§ "M3a.3").
- `docs/CHATGPT_HANDOFF_12_IC_PICARD_BUGS.md` and `_13_*` — γ-Picard
  consistency and Stern-η bugs that the algebra here must preserve.

## 1. Notation and per-reaction inputs

Per-reaction config fields (consumed by the residual at
`forms_logc.py:340-441` and the muh counterpart at `forms_logc_muh.py:380-487`):

| Field | Type | Meaning |
|---|---|---|
| `k0_model` | float | Nondim exchange rate constant |
| `alpha` | float | BV transfer coefficient (cathodic) |
| `n_electrons` | int | Electrons transferred per turnover |
| `E_eq_model` | float | Nondim equilibrium potential |
| `cathodic_species` | int | Species index of cathodic substrate (linear factor) |
| `anodic_species` | int \| None | Species index of anodic substrate; None ⇒ no surface-species anodic branch |
| `reversible` | bool | Whether the anodic backward branch contributes at all |
| `c_ref_model` | float | Nondim reference concentration used by the **constant-anodic** branch |
| `cathodic_conc_factors` | list of `{species, power, c_ref_nondim}` | Non-substrate cathodic concentration factors (typically H⁺) |
| `stoichiometry` | list[int] of length n_species | **Signed** coefficients (negative = consumed at cathode, positive = produced) |

Surface state:

- `c_b = (O_b, P_b, H_b, …)` — bulk concentrations (nondim).
- `c_s = (O_s, P_s, H_o, …)` — outer-region surface values at the OHP
  (`H_s = H_o · exp(−ψ_D)` after the diffuse-layer Boltzmann shift).
- `D = (D_O, D_P, D_H, …)` — nondim diffusivities.
- `R = (R_1, R_2, …, R_N)` — per-reaction cathodic-forward rates;
  positive ⇒ cathodic reduction.
- `γ_s` — multispecies Bikerman activity coefficient at the OHP
  (= 1 in the ideal-counterion limit).
- `ψ_D, ψ_S` — diffuse-layer / Stern drops; `η_drop = ψ_S` (Stern) or
  `φ_applied` (no-Stern).
- `η_j = bv_exp_scale · clip(η_drop − E_eq_j, ±exponent_clip)` — per-
  reaction overpotential after the eta-clip.

## 2. Outer-region surface flux balance (signed)

For each dynamic species `i`, with **signed** stoichiometry `s_{i,j}`
and signed cathodic-forward rate `R_j`:

```
c_i_s  =  c_i_b  +  Σ_j  s_{i,j} / D_i  ·  R_j        (signed throughout)
```

For protons under outer ambipolar transport (`J_H = -2 D_H ∇c_H`), the
matched-asymptotic derivation in `docs/PNP_BV_Analytical_Simplifications.md:240-244`
gives

```
H_o  =  H_b  +  Σ_j  s_{H,j}  ·  R_j  /  ( 2 D_H )           (signed, ambipolar)
```

The `1 / (2 D_H)` is **the ambipolar correction** and must not be
dropped when stoichiometries vary across reactions.

> **Why signed, not absolute.**  All current acid-form ORR reactions
> consume protons (`s_H ≤ 0`), so `|s_H| / 2` and `+ s_H / (2 D_H)`
> give the same answer.  But the absolute-value form silently inverts
> the sign for any reaction whose cathodic-direction stoichiometry
> *produces* H⁺.  We use the signed form to keep the derivation
> truthful for arbitrary stoichiometries and to match the same
> bookkeeping convention used for `c_i_s`.  Code that accepts only
> proton-consuming topologies should validate `s_{H,j} ≤ 0` at config
> time rather than relying on `abs(.)` in the formula.

**Sequential (legacy) case.**  Both reactions have `s_H = −2`:

```
H_o  =  H_b  +  ( −2 R_1  −  2 R_2 ) / ( 2 D_H )
     =  H_b  −  ( R_1 + R_2 ) / D_H .
```

Matches `picard_ic.py:516`.

**Parallel R_2e + R_4e.**  `s_{H, R_2e} = −2`, `s_{H, R_4e} = −4`:

```
H_o  =  H_b  +  ( −2 R_2e − 4 R_4e ) / ( 2 D_H )
     =  H_b  −  ( R_2e + 2 R_4e ) / D_H .
```

> **Correction note (round-1 Codex review).**  The realignment plan
> at `docs/ruggiero_realignment_plan.md:381` writes
> `H_b − (2·R_2e + 4·R_4e)/D_H`, which drops the ambipolar `1/2`
> factor and is wrong by `2×` for non-uniform `s_H`.  The corrected
> formula above is the implementation contract.

## 3. Linearized BV rate expressions (three-branch)

The residuals at `forms_logc.py:376-454` (log-rate path 376-441, non-log-rate
path 410-454) and `forms_logc_muh.py:415-487` produce the per-reaction
rate as a difference of cathodic and anodic terms.  The **anodic
branch is three-way conditional**:

| Branch | Condition | Anodic term |
|---|---|---|
| 1 (surface-species linear) | `reversible AND anodic_species is not None` | `k_j · c_anodsub_j · γ_s · exp((1 − α_j) · n_e_j · η_j)` |
| 2 (affine constant) | `reversible AND anodic_species is None AND c_ref_model > 1e-30` (log-rate) or `reversible AND anodic_species is None` (non-log-rate, with `c_ref_model = 0` collapsing to zero) | `k_j · c_ref_model · exp((1 − α_j) · n_e_j · η_j)` *(no γ factor)* |
| 3 (irreversible) | `reversible == False` (or both above conditions fail) | `0` |

Define per-reaction prefactors that absorb everything **except** the
linear substrate concentrations:

```
α̂_j  =  k_j · γ_s^{1 + Σ_f power_f}                                    (cathodic)
        · Π_f (c_factor_f / c_ref_f)^{power_f}
        · exp(−α_j · n_e_j · η_j)

β̂_j  =  ( branch 1 ) ? k_j · γ_s · exp((1 − α_j) · n_e_j · η_j) : 0    (anodic linear)

Ĉ_j  =  ( branch 2 ) ? k_j · c_ref_model · exp((1 − α_j) · n_e_j · η_j) : 0   (anodic constant)
```

Note that **branches 1 and 2 are mutually exclusive** by the
conditional structure above, so at most one of `β̂_j, Ĉ_j` is nonzero
per reaction.  Then the per-reaction affine rate model is:

```
R_j  =  α̂_j · c_{cathsub_j}_s   −   β̂_j · c_{anodsub_j}_s   −   Ĉ_j
```

Cathodic γ-power per reaction:
`1 + Σ_factors power_factor` — one for the catalyst (cathodic
substrate is itself γ-shifted) plus one per H⁺-style factor power.
Anodic linear-branch (β̂_j): γ¹.  Anodic constant-branch (Ĉ_j): γ⁰
(the residual does not multiply `c_ref_model` by activity).

For the M3a.3 ORR target, `cathodic_conc_factors` always carries
`power = n_electrons` (acid-form ORR consumes one H⁺ per electron):

```
R_2e  cathodic γ-power  =  1 + 2  =  3        (one catalyst + 2 H⁺ powers)
R_4e  cathodic γ-power  =  1 + 4  =  5
R_2e  anodic γ-power    =  1                  (branch 1 fires)
R_4e  anodic γ-power    =  0                  (branch 3 fires, anodic = 0)
```

## 4. Linear system M · R = b (affine)

Substitute the surface-flux balance from §2 into the linearized rates
from §3.  Surface concentrations become linear functions of the rate
vector R:

```
c_{cathsub_j}_s  =  c_{cathsub_j}_b  +  Σ_k  s_{cathsub_j, k} / D_{cathsub_j}  · R_k
c_{anodsub_j}_s  =  c_{anodsub_j}_b  +  Σ_k  s_{anodsub_j, k} / D_{anodsub_j}  · R_k          (only when β̂_j ≠ 0)
```

Substituting and gathering R_k terms gives **M · R = b** with

```
M_{j,k}  =  δ_{j,k}
         −  α̂_j · s_{cathsub_j, k} / D_{cathsub_j}
         +  β̂_j · s_{anodsub_j, k} / D_{anodsub_j}            (β̂_j = 0 ⇒ second term gone)

b_j      =  α̂_j · c_{cathsub_j}_b
         −  β̂_j · c_{anodsub_j}_b
         −  Ĉ_j                                               (Ĉ_j = 0 ⇒ third term gone)
```

`Ĉ_j` does not enter `M` — it is a per-reaction constant that
contributes only to the RHS.  Solve with `numpy.linalg.solve` for the
ORR-scale `N ≤ a small constant`.

> **Per-species transport coefficient λ_i.**  The flux-balance divisor
> in `M_{j,k}` and `b_j` is `λ_i = 1 / D_i` for ordinary species (O₂,
> H₂O₂, etc.), and `λ_H = 1 / (2 D_H)` for protons under the ambipolar
> outer transport law (matches `H_o` in §2).  Current ORR configs
> never use H⁺ as `cathodic_species` or `anodic_species` — H⁺ enters
> only via `cathodic_conc_factors` — so `λ_H` is not exercised in the
> inner solve.  If a future config uses H⁺ as a linear substrate, the
> implementation MUST dispatch `λ_i` per species; the adapter site
> rejects such configs until tested (see §9 item 11).

The H⁺ flux balance does **not** appear in `M`.  H⁺ enters only
through γ_s (via `H_o` in `compute_surface_gamma`) and through the
`cathodic_conc_factors` log term inside `α̂_j`, both treated as fixed
during the inner linear solve and updated in the outer Picard
iteration (§7).

## 5. Byte-equivalence to the legacy 2x2 sequential

Sequential reaction set:

- `R_1: O₂ + 2H⁺ + 2e⁻ → H₂O₂`,
  `cathsub_1 = O (idx 0)`, `anodsub_1 = P (idx 1)`,
  `s_O = −1, s_P = +1, s_H = −2`, `n_e = 2`, **reversible**, branch 1
  (`A_1, B_1` ≠ 0; `C_1 = 0`).
- `R_2: H₂O₂ + 2H⁺ + 2e⁻ → 2H₂O`,
  `cathsub_2 = P (idx 1)`, `anodsub_2 = None`,
  `s_O = 0, s_P = −1, s_H = −2`, `n_e = 2`, **irreversible** in
  legacy presets (`reversible = False` ⇒ branch 3, `B_2 = C_2 = 0`).

Substituting into §4 (using `α̂ → A`, `β̂ → B`, `Ĉ → C` to match code
notation):

| Entry | Generalized formula | Reduction | Current code |
|---|---|---|---|
| `M_11` | `1 − A_1·(−1)/D_O + B_1·(+1)/D_P` | `1 + A_1/D_O + B_1/D_P` | `m11 = 1 + A1/D_O + B1/D_P` |
| `M_12` | `−A_1·(0)/D_O + B_1·(−1)/D_P` | `−B_1/D_P` | `m12 = -B1/D_P` |
| `M_21` | `−A_2·(+1)/D_P + 0` | `−A_2/D_P` | `m21 = -A2/D_P` |
| `M_22` | `1 − A_2·(−1)/D_P + 0` | `1 + A_2/D_P` | `m22 = 1 + A2/D_P` |
| `b_1` | `A_1·O_b − B_1·P_b − 0` | `A_1·O_b − B_1·P_b` | `rhs1 = A1*O_b - B1*P_b` |
| `b_2` | `A_2·P_b − 0 − 0` | `A_2·P_b` | `rhs2 = A2*P_b` |

Every entry matches `picard_ic.py:489-494`.  The H_o flux balance from
§2 collapses to `H_b − (R_1 + R_2)/D_H` (each `s_H = −2` ⇒ ambipolar
`(−2 + −2)/2 = −2`), matching `picard_ic.py:516`.  The post-Picard
closed-form `P_s = (D_P·P_b + R_1)/(D_P + A_2)` at `picard_ic.py:611`
arises from the same fixed-point identity (see §8).

## 6. Parallel R_2e + R_4e (Ruggiero topology)

Reaction set per `scripts/_bv_common.py:525-554`
(`PARALLEL_2E_4E_REACTIONS`):

- `R_2e: O₂ + 2H⁺ + 2e⁻ → H₂O₂`,
  `cathsub = O (idx 0)`, `anodsub = P (idx 1)`,
  `stoich = [−1, +1, −2]`, `n_e = 2`, **reversible**, branch 1
  (`α̂_2e, β̂_2e` ≠ 0; `Ĉ_2e = 0`).
- `R_4e: O₂ + 4H⁺ + 4e⁻ → 2H₂O`,
  `cathsub = O (idx 0)`, `anodsub = None`,
  `stoich = [−1, 0, −4]`, `n_e = 4`, **irreversible**
  (`reversible = False` ⇒ branch 3, `β̂_4e = Ĉ_4e = 0`).

§4 yields:

```
M = [ 1 + α̂_2e/D_O + β̂_2e/D_P        α̂_2e/D_O             ]
    [ α̂_4e/D_O                        1 + α̂_4e/D_O         ]

b = [ α̂_2e·O_b − β̂_2e·P_b − 0 ]
    [ α̂_4e·O_b − 0 − 0          ]
```

(All `s_O_*` = −1 ⇒ `−α̂·(−1)/D_O = +α̂/D_O`.)

H_o flux balance from §2 (signed):

```
H_o  =  H_b  +  (−2 R_2e − 4 R_4e) / (2 D_H)
     =  H_b  −  (R_2e + 2 R_4e) / D_H .
```

The cathodic γ-power on R_4e is γ⁵ (1 catalyst + 4 from `power = 4`).
A numerical concern at saturation — γ_s can be O(10⁻²) in the diffuse
layer, so γ⁵ ~ 10⁻¹⁰ — but it does not break the algebra; double
precision tolerates it as long as α̂_4e is computed in log-space
(matching `picard_ic.py:481`).

## 7. Outer Picard iteration

The matrix coefficients depend on the surface state
`(γ_s, H_o, ψ_D, η_j)` which themselves depend on the rates.  The
outer Picard iteration is unchanged in structure from the legacy 2x2:

```
loop k = 1 .. max_iters:
    γ_s              ←  compute_surface_gamma(H_o, c_clo4_bulk, ψ_D, a_h, a_cl)
    log_by_species   ←  per-species log-conc at OHP, γ-shifted
    α̂_j, β̂_j, Ĉ_j   ←  build per-reaction prefactors from log-rates (§3)
    M, b             ←  assemble §4 matrix and RHS
    R_solve          ←  numpy.linalg.solve(M, b)         (or fail → singular_jacobian)
    R_old            ←  copy(R)
    R                ←  (1−ω) · R_old  +  ω · R_solve    (per-reaction relaxation)
    c_i_s            ←  signed surface flux balance from §2 (with floors; §9 item 8)
    H_o              ←  H_b  +  Σ_j s_{H,j}·R_j / (2 D_H) (with floor; §9 item 8)
    φ_o, ψ_D, ψ_S    ←  Stern split (unchanged from picard_ic.py:520-540)
    η_j              ←  per-reaction eta clip (§1)
    δ                ←  Σ_j |R_j − R_old_j| / max(|R_j|, 1e−30)
    if δ < tol: break
```

`R_solve` is the raw linear-solve output for iteration k; `R_old` is the
previous iteration's (already-relaxed) rate vector; `R` is the relaxed
state used for everything downstream including δ.  Mirrors
`picard_ic.py:441-563` exactly: `R1, R2 = (1−ω)·old + ω·solve` with δ
computed from the relaxed values.

**What's lagged.**  H⁺ surface concentration, γ_s, ψ_D, ψ_S, η_j, and
all `cathodic_conc_factors` species' log-concentrations enter
`α̂_j, β̂_j, Ĉ_j` only through their previous-iteration values — they
are *frozen* during the inner `M·R = b` solve.  This is the standard
matched-asymptotic Picard pattern.  Convergence is empirical:
5-10 outer iterations suffice for the legacy 2x2 ORR config; we
expect similar for parallel R_2e/R_4e.

For N>2 reactions whose magnitudes span many decades (e.g., the
Ruggiero case where `R_2e ≫ R_4e` at low |η|), keep `ω = 0.5` and
rely on the inner linear solve to absorb most of the cross-reaction
coupling.  Picard relaxation at the *outer* loop level is not
sensitive to individual rate magnitudes — what matters is the
Jacobian conditioning of `M`, which the inner solve handles directly.

## 8. Post-loop surface concentrations

After convergence, the naive `c_i_s` from §2 can suffer
near-cancellation in the diffusion-limited regime when
`Σ_k s_{i,k} R_k` is small relative to its individual terms.  The
legacy code at `picard_ic.py:585-612` works around this for `P_s` in
the sequential template by absorbing `R_2` into the denominator:

```
P_s  =  (D_P · P_b  +  R_1)  /  (D_P + A_2)        (sequential template only)
```

derivation: `P_s = P_b + (R_1 − R_2)/D_P` and `R_2 = A_2 · P_s` ⇒
`P_s · (1 + A_2/D_P) = P_b + R_1/D_P`.

For the parallel case, `P_s = P_b + R_2e/D_P` (R_4e doesn't touch P)
— no cancellation source, naive form is robust.

**Implementation contract**: keep the naive
`c_i_s = c_i_b + Σ_k s_{i,k}·R_k/D_i` everywhere by default; preserve
the legacy closed-form `P_s` formula behind a topology hint
(`"sequential_2e_h2o2"`) so the byte-equivalence test passes.  Do
**not** generalize the closed form to arbitrary topologies until
empirical numerical-stability evidence demands it.

## 9. Implementation contract

The generalized `picard_outer_loop` MUST satisfy:

1. **Sequential byte-equivalence.**  Called with the legacy 2-reaction
   list (any combination of α̂_j, β̂_j, Ĉ_j, η_j, γ_s, ψ_D, ψ_S that
   the current code accepts) it must return `R_list[:2], O_s, P_s,
   H_o, ψ_D, ψ_S, γ_s` byte-identical to the current 2x2
   implementation up to floating-point reordering — tolerance ≤ 1e−12
   on each scalar.
2. **Per-reaction n_e.**  `α̂_j, β̂_j, Ĉ_j` MUST read `n_electrons`
   from `reactions[j]`, not from a single hardcoded value.  The
   current implementation at `forms_logc_muh.py:873` reads `n_e` from
   `reactions[0]` only — that breaks for parallel R_2e/R_4e.
3. **Signed ambipolar H_o.**  Use
   `H_o = H_b + Σ_j s_{H,j}·R_j / (2 D_H)`.  Do NOT use either
   `|s_{H,j}|` (silently wrong-sign for proton-producing reactions)
   or the realignment plan's `Σ_j s_{H,j}·R_j / D_H` (drops ambipolar
   `1/2`).
4. **All three anodic branches.**  Implement the affine
   `R_j = α̂_j · c_cathsub_j - β̂_j · c_anodsub_j - Ĉ_j` with branch
   selection matching the residual (§3).  In particular, when
   `reversible AND anodic_species is None AND c_ref_model > 1e-30`
   (log-rate) or `reversible AND anodic_species is None` (non-log-rate,
   with `c_ref_model = 0` collapsing to zero), populate `Ĉ_j` (no
   γ factor) and leave `β̂_j = 0`.
5. **γ-power.**  α̂_j carries γ_s^{1 + Σ_factors power_factor};
   β̂_j carries γ_s¹; Ĉ_j carries γ_s⁰.  Verifiable against
   `forms_logc.py:376-454` and `forms_logc_muh.py:415-487` line-by-line.
6. **Topology hint.**  A `topology_hint` string (e.g.,
   `"sequential_2e_h2o2"`, `"parallel_2e_4e"`, `"general"`) selects
   the post-loop surface-concentration reconstruction.  Default to
   `"general"` (naive flux balance); enable the legacy closed-form
   `P_s` only for `"sequential_2e_h2o2"`.
7. **Adjoint hygiene.**  All Picard work is wrapped in
   `firedrake.adjoint.stop_annotating()` at the adapter sites
   (`forms_logc.py:661`, `forms_logc_muh.py:729`); the refactor must
   not introduce any pyadjoint-tracked operations inside the Picard
   body.
8. **Robust failure with legacy-compatible floors.**  After the inner
   linear solve and rate relaxation, surface concentrations are
   reconstructed with the same floors used by the current 2x2 loop:

   ```
   O_s = max(O_b − R_1 / D_O,            1e−300)
   P_s = max(P_b + (R_1 − R_2) / D_P,    P_FLOOR = 1e−30)
   H_o = max(H_b − (R_1 + R_2) / D_H,    1e−300)
   ```

   (and the equivalent generalized signed-flux form for non-sequential
   topologies; floors per species).  Floors are applied **first** —
   negative raw `O_s`, `P_s`, or `H_o` is **not** an automatic failure
   on the sequential-compatible path.  Failure is reserved for: (i)
   singular `M` (`det → 0` or `numpy.linalg.LinAlgError`), (ii) non-
   finite solve output, (iii) non-finite post-update state, (iv)
   `max_iters` reached without convergence, or (v) post-floor states
   that remain physically unrecoverable (e.g., NaN/Inf surviving the
   floor).  On any of these, return
   `(False, reason_string, n_iters, partial_state)` so the adapter
   site falls back to the linear-phi IC unchanged.  This preserves
   the existing fallback contract at `forms_logc.py:880` and
   `forms_logc_muh.py:945`.
9. **Validation guards.**  At config time, validate that no reaction
   places a cathodic substrate (`cathodic_species`) into its own
   `cathodic_conc_factors` with a non-trivial power — that case
   would make the rate nonlinear in the substrate concentration and
   break the linear-substrate assumption.  The current ORR configs
   are safe (H⁺ is in `cathodic_conc_factors`, never the catalyst).
10. **Scope assertion (optional).**  If we want to enforce the
    proton-consuming-only ORR scope, add a config-time assertion
    `s_{H, j} ≤ 0 ∀ j`.  Otherwise the signed formula in §2 handles
    the general case correctly; the assertion is a defensive narrowing
    rather than a correctness fix.
11. **Per-species λ_i.**  The matrix divisor for the cathodic / anodic
    substrate of reaction `j` MUST be `λ_i = 1 / D_{cathsub_j}` (or
    `1 / D_{anodsub_j}`) for ordinary species, and
    `λ_H = 1 / (2 D_H)` if the substrate is H⁺ (ambipolar outer
    transport).  At adapter entry, validate that no reaction lists H⁺
    as `cathodic_species` or `anodic_species`; reject with reason
    `h_plus_as_linear_substrate` and fall back to the linear-phi IC.
    Current ORR configs never trigger this — H⁺ enters only via
    `cathodic_conc_factors` — so the guard is a forward-compatibility
    rejection, not an active dispatch path.

## 10. Verification ladder

(Mirrors §4 of the M3a.3 plan handoff; T2 fixed per round-1 review.)

- **T1 — sequential byte-equivalence (unit, no Firedrake).**
  Generalized call with the legacy 2-reaction list reproduces a
  frozen reference scalar state `(R_1, R_2, O_s, P_s, H_o, ψ_D, ψ_S,
  γ_s)` to ≤ 1e−12 (tolerance: floating-point reordering only).
  Reference frozen at commit `d8bf645`.

- **T2 — pure-2e parallel vs. legacy sequential with R_2 disabled
  (unit).**  Parallel preset with `k0_R4e = 0` against legacy
  sequential with `k0_R2 = 0`.  Both runs have only the O₂ → H₂O₂
  rate active.  Required identities:
  ```
  R_4e  ≈  0
  R_2e  =  R_1 (legacy with k0_R2 = 0)        ≤ 1e−10
  O_s   =  O_b − R_2e / D_O                   ≤ 1e−10
  P_s   =  P_b + R_2e / D_P                   ≤ 1e−10
  H_o   =  H_b − R_2e / D_H                   ≤ 1e−10
  ```

- **T3 — pure-4e (unit).**  `k0_R2e = 0` ⇒ `R_2e ≈ 0`,
  `P_s ≈ P_b` (R_4e doesn't touch P), `H_o = H_b − 2·R_4e/D_H`
  (ambipolar with |s_H|=4), `O_s = O_b − R_4e / D_O`.

- **T4 — γ-power probe (unit).**  Verify by perturbing γ_s in the
  Picard inputs:
  ```
  R_2e cathodic prefactor (α̂_2e)  ∝  γ_s^3
  R_4e cathodic prefactor (α̂_4e)  ∝  γ_s^5
  R_2e anodic linear prefactor (β̂_2e)  ∝  γ_s^1
  R_2e anodic constant prefactor (Ĉ_2e), if synthetically tested  ∝  γ_s^0
  ```

- **T5 — singular Jacobian (unit).**  Pathological config returns
  `(False, "singular_jacobian_iter_k", k, partial_state)` and the
  adapter site falls back to linear-phi IC.

- **T6 — adjoint tape (slow).**  Annotated run on
  `PARALLEL_2E_4E_REACTIONS` produces zero Picard-iteration tape
  entries.

- **T7 — Firedrake single-V cold (slow, sequential).**  V=0.0 V on
  the legacy reaction list reproduces Run C `cd_mA_cm2 = −0.175`
  (post-C_O2-fix value to be re-anchored; pre-fix reference is
  −0.175 from `StudyResults/parallel_2e_4e_warmstart_probe/probe.json`
  Step A).  Retained as the full-sequential regression anchor
  independent of T8.

- **T7b — Firedrake single-V cold (slow, legacy sequential with
  R_2 disabled).**  V=0.0 V on the legacy reaction list with
  `k0_R2 = 0`.  Establishes the disabled-R_2 reference current that
  T8 compares against (mathematically equivalent to a pure-2e
  parallel run, modulo any numerical / IC slack).

- **T8 — Firedrake single-V cold (slow, parallel pure-2e).**  V=0.0 V
  on the parallel preset with `k0_R4e = 0`; compare against **T7b**
  (legacy sequential with `k0_R2 = 0`), not T7.  Required:
  `cd_mA_cm2` matches T7b within ≤ 1% relative.  This keeps the
  pure-2e parallel comparison mathematically equivalent at the slow-
  Firedrake level, fixing the round-1 ambiguity that T2 already
  fixed at the unit level.

- **T9 — Firedrake single-V cold (slow, parallel pure-4e).**  V=0.0
  V with `k0_R2e = 0`; `n_e_apparent → 4`, peroxide current → 0.

- **T10 — Firedrake full sweep (production).**  Mixed parallel,
  ≥ 22/25 V converged on the page-15 grid via C+D.  Acceptance
  bands from realignment plan §M3a.3.

- **T11 — Signed-H_o synthetic (unit).**  Synthetic 1-reaction
  config with `s_H = +2` (proton-producing cathodic — pathological,
  used only to verify the formula, not a physical config).  Required
  identity: `H_o = H_b + R_1 / D_H`.  Confirms code uses the signed
  form, not `|s_H|`.

- **T12 — Constant-anodic branch validator (unit).**  Synthetic
  1-reaction config with `reversible = True, anodic_species = None,
  c_ref_model = 0.5`.  Required identities: `Ĉ_j ≠ 0`, `β̂_j = 0`,
  `b_j = α̂_j · O_b − Ĉ_j`.  Confirms branch 2 is wired correctly.
  Either this test passes, or the implementation explicitly rejects
  this config at the adapter site (with a clear failure reason).

## 11. Open derivation work

Deferred until empirical evidence demands it:

- **Closed-form post-loop reconstruction for non-sequential
  topologies with cancellation.**  §8.  Wait for a numerical-
  stability test failure on the parallel preset before generalizing.

- **Generalized γ activity for multi-counterion electrolyte (M3b.2).**
  `compute_surface_gamma` currently hardcodes one H⁺ + one ClO₄⁻;
  multi-ion (Cs⁺/H⁺/SO₄²⁻) is a separate piece of work.

- **Strong dynamic species in `cathodic_conc_factors`.**  The
  Picard-lagged factor pattern degrades when a factor species is
  itself strongly modified by the rate balance with a high power.
  No current ORR config triggers this (H⁺ is the only factor and its
  flux balance is solved in the same outer loop), but a future
  reaction that puts e.g. O₂ into another reaction's
  `cathodic_conc_factors` with `power ≥ 2` would exit the validity
  window of this initializer.  Add a config-time validator that flags
  such configs and rejects the analytical IC (falling back to
  linear-phi cleanly) until the inner solve is generalized.

- **Generalized closed-form for branch 2 (Ĉ_j) post-loop
  reconstruction.**  Current spec uses naive flux balance for all
  topologies except sequential.  If branch 2 ever fires in
  production, re-evaluate whether a robust closed form is needed.

## 12. Round-1 Codex review reply

Source: `docs/CODEX_REVIEW_PICARD_GENERAL_TOPOLOGY_DERIVATION.md`
(2026-05-08).  All four findings accepted; this v2 incorporates the
fixes.

| Finding | Disposition | Where in v2 |
|---|---|---|
| F1 — `|s_H|` form silently wrong-sign for H⁺-producing reactions | **Accepted.**  Replaced with signed form `+ Σ_j s_{H,j}·R_j / (2 D_H)` everywhere | §2 (formula + "Why signed" callout); §7 (loop body); §9 contract item 3; §10 T11 |
| F2 — Constant anodic branch (`reversible AND anodic_species is None AND c_ref_model > 0`) was missing from rate model | **Accepted.**  Added as branch 2 with prefactor `Ĉ_j`; rate model is now affine `R_j = α̂_j · c_cath − β̂_j · c_anod − Ĉ_j`; matrix `M` unchanged, `Ĉ_j` enters only `b_j` | §3 (three-branch table); §4 (extended `b_j`); §9 contract item 4; §10 T12 |
| F3 — "General N-reaction" overstated; non-substrate factors are Picard-lagged | **Accepted.**  Title narrowed to "Linear-Substrate Picard Initializer for ORR-class BV Topologies"; explicit "what's lagged" callout added; scope guard added in §11 | Title; §"Scope" preamble; §7 ("What's lagged"); §11 ("Strong dynamic species in `cathodic_conc_factors`") |
| F4 — T2 underspecified (compared pure-2e parallel to full legacy sequential) | **Accepted.**  Rewrote T2 to compare against legacy sequential with `k0_R2 = 0` | §10 T2 |

**No findings rejected.**  Codex's "what is mathematically sound"
section confirms the M·R = b structure, byte-equivalence to the
legacy 2x2, the parallel R_2e/R_4e matrix, and the γ-power
accounting — those are unchanged in v2.

**Implementation contract changes** (relative to round-1):

- `picard_outer_loop` signature now reads `c_ref_model` and
  `reversible` per reaction and builds three-branch anodic terms
  (was: two-branch).
- `picard_state` returned by the loop now includes `R_list`, `α̂_list`,
  `β̂_list`, `Ĉ_list` for diagnostics and the constant-anodic γ⁰ test.
- Verification ladder gains T11 (signed-H_o synthetic) and T12
  (constant-anodic branch).

## 13. Round-2 Codex review reply

Source: `docs/CODEX_REVIEW_PICARD_DERIVATION_ROUND2.md` (2026-05-08).
All three findings accepted; the round-2 question (Q7) is also accepted
as a new contract item.  This v3 incorporates the fixes.

| Finding | Disposition | v3 location |
|---|---|---|
| F1 — δ uses raw `R_new` instead of relaxed `R` (ambiguity in §7 pseudo-code, would shift effective tolerance by ~ω if implemented literally) | **Accepted.**  Renamed `R_new → R_solve`, added explicit `R_old ← copy(R)` snapshot, `R ← (1−ω)·R_old + ω·R_solve`, and δ uses the relaxed `R`.  Mirrors `picard_ic.py:441-563`. | §7 pseudo-code (A1) |
| F2 — Failure contract said "negative `O_s`" is failure, conflicting with legacy floors and breaking sequential byte-equivalence | **Accepted.**  Reworded item 8: floors first (`O_s ≥ 1e−300`, `P_s ≥ P_FLOOR = 1e−30`, `H_o ≥ 1e−300`); fail only on singular `M` / non-finite solve / non-finite post-update / max-iters / unrecoverable post-floor states. | §9 item 8 (A2) |
| F3 — Slow-Firedrake T8 still compared parallel pure-2e to full legacy sequential (T7), reintroducing the round-1 ambiguity at the slow-test level | **Accepted.**  Inserted T7b (legacy sequential with `k0_R2 = 0`); T8 now compares to T7b within ≤ 1% relative.  T7 retained as the full-sequential regression anchor. | §10 T7b/T8 (A3) |
| Q7 — Missing per-species transport coefficient guard.  Future configs could use H⁺ as a linear substrate, in which case `λ_H = 1/(2 D_H)` (ambipolar) is required, not `1/D_H`. | **Accepted.**  Added explicit `λ_i` per-species note in §4; new contract item 11 in §9 dispatches `λ_i` per species and rejects H⁺-as-substrate at the adapter site (`reason = h_plus_as_linear_substrate`). | §4 sub-note + §9 item 11 (A4) |

**Round-2 question dispositions (Q1–Q6) baked into v3 unchanged from
v2.**  Per Codex's round-2 answers:

- **Q1** (γ-power for `Ĉ_j`): retained as `γ⁰` because the residual at
  `forms_logc.py:376-454` does not multiply `c_ref_model` by activity;
  the IC must mirror the residual.  See §3 γ-power table.
- **Q2** (signed-H_o framing): kept the signed formula and T11; did
  **not** narrow scope to `s_H ≤ 0`.  Item 10 in §9 is left as an
  **optional** scope assertion.
- **Q3** (T12 disposition): chose path (a) — implement `Ĉ_j`, test it,
  leave dormant in current ORR presets.  See §10 T12.
- **Q4** (post-loop closed form for parallel): deferred until empirical
  failure.  `P_s = P_b + R_2e/D_P` has no cancellation source.  Listed
  in §11 "Open derivation work".
- **Q5** (per-reaction relaxation): kept single `ω = 0.5`.  Inner N×N
  solve absorbs the cross-reaction coupling.
- **Q6** (Picard tolerance): kept `tol = 1e-6`; tighten only after
  T7-T10 show Newton sensitivity.

**No findings rejected.**  Codex's round-2 approval status confirms the
v2 algebra is sound for the M3a.3 ORR target; only doc-level fixes were
required.

**Implementation contract changes** (relative to v2):

- `picard_outer_loop` MUST snapshot `R_old = copy(R)` before relaxation
  and compute δ from the relaxed `R`, not from the raw solve output.
- `picard_outer_loop` MUST apply legacy-compatible floors before
  classifying a post-update state as "unrecoverable"; floors are not
  failures.
- Adapter sites (`forms_logc.py:711`, `forms_logc_muh.py:786`) MUST
  reject configs with H⁺ as `cathodic_species` or `anodic_species`
  (reason `h_plus_as_linear_substrate`).
- T7b is added to the slow Firedrake portion of the verification
  ladder; T8 acceptance now references T7b instead of T7.
