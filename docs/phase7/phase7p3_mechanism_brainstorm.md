# Phase 7.3 brainstorm — mechanisms for the pH-resolved ORR discrepancies

**Status:** brainstorm, GPT-hardened (Handoff 44, 3 rounds, APPROVED).
Not a plan, not a lock — a ranked, defensible mechanism shortlist.
**Inputs:** `phase7p2_dual_series_summary.md` (single-condition pH-6.39
fit LOCKED), `StudyResults/phase7p2_ph_series_generalization/metrics.json`
(frozen-θ_L water-only across pH 1.65–6.39 vs Brianna `Exp Info`),
slide-15 v2 SVG target (`project-slide15-exact-source-v2`).
**Critique record:** `docs/handoffs/CHATGPT_HANDOFF_44_phase7p3-mechanism-ranking/`.

> **Reading guide.** §1 quantifies the discrepancy (with the honest error
> bars GPT forced). §2 is the central reframe (SHE-axis onset). §3 ranks
> the mechanisms. §4 is the **final taxonomy** (Tier 1 hypotheses / Tier 2
> required infrastructure / Tier 3 confounds / Tier 4 near-dead) — this
> supersedes any flat ranking. §5 = identifiability ceiling. §6 = ordered
> data-asks. §7 = confounds. Skip to §4 for the answer.

---

## 1. The discrepancy, quantified (with error bars)

Frozen θ_L (log f_2w −1.009, log f_4w −12.309, α_2w 0.577, α_4w 0.305,
L_eff 21.7 µm), water routes only, applied across bulk pH (only c_H and
per-pH OCP changed). Model vs Brianna `Exp Info`:

| pH | onset_data | onset_model | sel_data % | sel_model % | ring_data | ring_model |
|----|-----------|------------|-----------|------------|-----------|-----------|
| 1.65 | 0.251 | 0.531 | 19.9 | 57.3 | 0.079 | 0.356 |
| 2.35 | 0.363 | 0.529 | 47.6 | 57.8 | 0.416 | 0.359 |
| 3.42 | 0.417 | 0.526 | 72.1 | 58.3 | 0.406 | 0.361 |
| 4.21 | 0.434 | 0.523 | 91.2 | 54.2 | 0.353 | 0.362 |
| 5.21 | 0.458 | 0.519 | 75.3 | 54.6 | 0.504 | 0.364 |
| 6.39 | 0.472 | 0.514 | 73.2 | 55.0 | 0.355 | 0.365 |

**The model is structurally FLAT in pH; the data is not.** Signatures,
with the robustness caveats the critique demanded:

- **S1 — onset shifts with pH (robust SIGN, soft magnitude).**
  d(onset_RHE)/dpH = +41 mV/pH but **R² = 0.80**, and leave-one-out
  slopes span **26–53 mV/pH** (endpoints pH 1.65 / 6.39 drive the swing).
  So: *higher pH ⇒ earlier onset on RHE is robust; the precise slope is
  not.* Model ≈ −4 mV/pH (flat, slightly wrong sign).
  **Gate:** S1 must be re-extracted at multiple small *absolute*
  current thresholds, after background subtraction, on raw / sheet
  (+I·Rs) / physical (−I·Rs) axes, with CIs, before any mechanism is
  credited (see §7).
- **S2 — peroxide selectivity rises with pH then plateaus.** 20%
  (pH 1.65) → ~73–91% (pH ≥ 3.4); transition midpoint ≈ pH 2.5–3. Model
  flat ~55%. *(Selectivity is a derived quantity — fit raw ring, not
  this, see §7.)*
- **S3 — ring (peroxide) current collapses in strong acid.** max_ring
  0.079 at pH 1.65 vs ~0.35–0.50 above pH 2.3. Model flat ~0.36.
- **S2′ (NOT trusted).** A possible selectivity peak at pH 4.21 (91%)
  above the plateau. peak_sel is a curve-MAX (small-denominator
  near-onset transients); ring at 4.21 is *lower* than at 5.21, so the
  91% is probably artifactual. Explain only if it survives in raw ring.

Slide-15 (Cs⁺ pH 4) residuals from the Phase-7 fit, for cross-checking:
peak position **+0.09 V too anodic** (model −0.391@+0.194 vs deck
−0.368@+0.101), a secondary bump at +0.22–0.27 the model misses, sharp
cliff at +0.28–0.35.

### Why the current model CANNOT produce these
- **Water route is c_H-independent by construction** (`_bv_common.py`
  `PARALLEL_2E_4E_DUAL_PATHWAY`, `cathodic_conc_factors: []`). Its
  overpotential `eta = phi_applied − phi − E_eq` (`forms_logc_muh.py:396`)
  uses an **RHE-referenced** E_eq ⇒ η = V_RHE − E°(RHE) has zero explicit
  pH dependence ⇒ everything pH-flat. (The −4 mV/pH residual is an
  OCP-convention leak, not kinetics.)
- **Un-freezing the acid route does NOT fix onset — wrong sign.** *As
  encoded* the acid route is RHE-fixed E_eq (0.695 V) **plus** a `c_H²`
  kinetic factor, giving onset slope ≈ −0.0592·m/(αn) with m=2,n=2 ≈
  **−102 mV/pH on RHE** (lower pH → much earlier onset). Caveat (GPT #5):
  RHE-fixed-formal + explicit c_H² may *over-count* proton dependence vs
  a properly-derived PCET step; do not reject PCET wholesale on this — it
  requires a single-convention re-derivation (§7).

---

## 2. The central reframe (read S1 on the SHE axis)

V_RHE = V_SHE + 0.0592·pH. Convert the onsets:

| pH | onset_SHE (V) |
|----|---------------|
| 1.65 | 0.153 |
| 2.35 | 0.224 |
| 3.42 | 0.215 |
| 4.21 | 0.185 |
| 5.21 | 0.150 |
| 6.39 | 0.094 |

mean **+0.17 V**, std 44 mV, range 130 mV, average slope −18 mV/pH.

**Reading.** On the SHE axis the onset is *roughly* pH-independent
(−18 mV/pH) — the signature of a rate-determining step that transfers an
electron but **little/no proton**. But it is **not flat: it humps** (peak
~+0.22 at pH 2.35–3.4, falling at both ends). So the SHE-anchoring is
*approximate*, and the residual hump is itself a secondary feature
(candidate causes: buffering/H, local-pH/G, or noise — do not
over-interpret given S1's R²=0.80).

**The split this motivates:**
- **Onset (S1)** ← proton coupling of the rate-determining first ET. §3-A.
- **Branching (S2/S3)** ← fate of the peroxide intermediate, which *is*
  c_H-dependent. §3-C.

### The onset slope, derived correctly
For an irreversible cathodic step `j ∝ c_H,surf^m · exp(−αnF(E−E_ref)/RT)`
with E_ref **SHE-anchored**, at fixed current:

> **dE_RHE/dpH = 0.0592·[1 − m·β/(αn)]**,  β ≡ −∂log₁₀(c_H,surf)/∂pH_bulk

- Pure proton-uncoupled RDS: **m = 0 ⇒ +59 mV/pH on RHE** (parameter-free).
- Sub-Nernstian data (≈41) ⇒ a small nonzero `m·β/(αn) ≈ 0.30`.
- With β = 1 (surface tracks bulk): m ≈ 0.17–0.18 (n=1) — but this is a
  **bulk-pH limiting estimate only**. If the interface is buffered
  (β ≪ 1, plausible given surface pH 9–10 under load), the same slope
  implies a larger or non-identifiable m. **The real inference must use
  modeled surface c_H, not bulk** — which is why G (§3) is mandatory, not
  optional.
- *Do not conflate* this kinetic m with a thermodynamic proton
  stoichiometry in the formal potential (which would give 59·(1−m),
  m≈0.31). Pick one convention and derive the combined slope once; never
  apply both an E_eq shift and a c_H factor without the combined formula.

---

## 3. Candidate mechanisms

Each: basis · target signature · sign check · PNP-BV mapping ·
identifiability/cost · risk.

### A — SHE-anchored first electron transfer (the onset driver) ★ lead
Renamed (GPT #2): **"SHE-anchored first ET with optional proton-dependent
prefactor / pre-equilibrium."** Pure A has m = 0; any nonzero kinetic
proton order means a protonated precursor, a coupled pre-equilibrium, or
a mixed RDS — state which.

Split into two falsifiability tiers (GPT #3):
- **A0 — pure SHE anchor, m = 0.** Parameter-free prediction
  **+59 mV/pH on RHE**. Status: *consistent with the currently-extracted
  sign and order of magnitude, but over-predicts the nominal slope*
  (≈41 < 59) — qualified pending raw-axis re-extraction (§7).
- **A1 — A0 + proton-dependent prefactor / local-pH(β) / field-site
  correction.** Fits the sub-Nernstian 30–55 mV/pH. A *family*, not a
  point prediction.

- **Sign:** ✔ SHE-fixed reference ⇒ +59 mV/pH RHE; data positive.
- **Mapping:** per condition set `E_eq_RHE = E0_SHE + 0.0592·pH_bulk`
  (E_eq is already a per-reaction field) and/or a `c_H,surf^m` factor via
  `cathodic_conc_factors` reading the **boundary** c_H. E0_SHE is a fit
  parameter (the literal aqueous O₂/O₂·⁻ at −0.33 V SHE is ~0.5 V more
  cathodic than the observed +0.17 V SHE onset, so it is *not* the
  number — adsorption/field stabilization is folded into E0_SHE).
- **Identifiability:** A0's slope is the clean test; A1 trades m against β
  against k0 — needs surface-c_H modeling (G) to be identifiable.
- **Risk:** the fingerprint is not unique (GPT #2) — pzc/field, site
  speciation (D), cation effect (E), iR/OCP artifacts (§7) can all mimic
  an SHE-anchored onset. A is the **leading hypothesis, not proof**;
  axis-validation gates it.

### C — pH-dependent peroxide consumption (the selectivity/ring driver) ★ co-lead
Split into two observably-distinct channels (GPT #8):
- **C1 — electrochemical disk H₂O₂ reduction:** H₂O₂ + 2H⁺ + 2e⁻ → 2H₂O,
  rate ∝ surface c_H, facile on carbon in acid. **Adds disk current**,
  consumes peroxide.
- **C2 — chemical/catalytic peroxide decomposition:** 2H₂O₂ → 2H₂O + O₂.
  Consumes ring-detectable peroxide with **no direct faradaic disk
  signature** — BUT it **regenerates O₂** (GPT #4): implement as a
  coupled H₂O₂-sink **and** O₂-source in the transport residual, and let
  transport decide whether that O₂ escapes or re-enters ORR (so C2 can
  leak into disk current and bias the partition). Its pH law need not be
  a simple `c_H^m` (GPT-nit #2): it may depend on potential, peroxide
  concentration, surface state, or metal impurities — *any* pH-dependent
  C2 law must use **local interfacial** conditions, not bulk.

- **Targets:** S2 (selectivity rise) AND S3 (acid ring collapse), one
  mechanism family.
- **Sign:** ✔ low pH ⇒ fast consumption ⇒ low ring, low apparent 2e
  selectivity (looks 4e); high pH ⇒ peroxide survives ⇒ high ring, high
  sel. Monotonic-rising sigmoid matches robust S2. **NB (GPT #11):** the
  driver is *acid-accelerated consumption*, **not** HO₂⁻ stability —
  H₂O₂ pKa₁ ≈ 11.6 is far above the window (and above surface pH 9–10),
  so HO₂⁻ is negligible everywhere here.
- **Mapping:** add C1 as a third reaction (`cathodic_species = 1` H₂O₂,
  c_H factor on boundary c_H) and C2 as a homogeneous sink/source pair.
- **Identifiability:** C1's rate+order pinned by the S2 midpoint and the
  pH-1.65 ring collapse — but degenerate with direct-4e at one rpm (§5).
- **Risk:** competes with the direct-4e route; the LOCKED "direct 2e/4e
  competition" finding (pH 6.39) is *not* established against the series
  alternative without rpm (§5).

### B — Superoxide pKa-4.8 branching ☠ near-dead (GPT #6, #7)
HO₂· ⇌ H⁺ + O₂·⁻ at pKa 4.8 controlling 2e-desorb vs further-reduce.
- **Why demoted:** (i) pKa 4.8 ≠ the robust S2 midpoint (pH 2.5–3); it
  could only explain the *untrusted* S2′ peak. (ii) Sign problem:
  disproportionation is fastest near the pKa (HO₂·+O₂·⁻ ~10⁸ M⁻¹s⁻¹),
  which would *lower* selectivity at the pKa — opposite to S2′. (iii)
  Under load the interface is pH 9–10, so the intermediate never samples
  bulk pH 4.8 — the bulk-pKa coincidence is **not evidence**.
- **Resurrect only if:** a raw, potential-aligned curve proves a real
  pH-4–5 non-monotonicity AND it is modeled at *local* pH (or as
  post-desorption solution-phase branching).

### D — Surface functional-group acid-base speciation (confound for A)
Quinone/carboxyl pKa sets active-site fraction; deprotonated sites more
active ⇒ higher pH earlier onset. ✔ sign, but a single pKa gives a
*sigmoid* onset-vs-pH, not the (soft-)linear S1 — weaker on S1's shape
than A. **Can mimic A's onset shift**; the A-vs-D discriminator (reversible
surface redox/titration + independent surface-chemistry readout, NOT
irreversible modification — GPT #8) is the clean test.

### E — Non-covalent cation / field effect (cross-condition)
Hydrated K⁺ at the OHP stabilizes intermediates; strength scales with
(E−pzc) hence pH (the group's Singh-2016 field-pKa hypothesis). The right
tool for the **Cs⁺ slide-15 cross-condition** and the +0.09 V anodic peak
residual. Under-constrained from K⁺-only data; the Stern/Bikerman +
Boltzmann machinery already computes OHP cation enrichment to couple to.

### F — pzc / Frumkin ψ₁ coupling (bookkeeping, not a standalone knob)
Microscopic justification for A1/E (the field correction in the BV
exponent). Can give either sign without an independent pzc measurement;
keep as the mechanism *behind* A1/E.

### G — Local/surface-pH kinetic coupling (required infrastructure)
Unbuffered K₂SO₄ + proton-consuming ORR ⇒ surface pH 9–10 under load.
The model transports μ_H but the rates only see c_H via the frozen acid
route. **Every** c_H-dependent rate (A1, C1, any pH-dependent C2) must
read the **surface** value `c_H = exp(μ_H − em·z_H·φ)` at the boundary.
G sets β; without it, A1's m and C's order are bulk-limiting only and
not identifiable. Cheap wiring, load-bearing for *magnitudes*.

### H — Bisulfate buffering at the acid end (required, acid points)
At pH 1.65, [HSO₄⁻]/[SO₄²⁻] = 10^(1.99−1.65) ≈ 2.2 (~69% bisulfate;
reservoir comparable to or above free H⁺); at pH 2.35, ~30%. So
bisulfate materially sets *local* proton availability at the two acid
points, sustains C1/C2 under load (→ pH-1.65 ring collapse), and shifts
the apparent proton order/midpoint assigned to C. **Caveat (GPT #5):**
the reservoir number assumed unchanged total sulfate and ideal
activities — if acidification used H₂SO₄, total sulfate *rises* at low pH,
and I≈0.3 M activities are non-ideal. "Likely important; quantify after
confirming acid recipe + activity model."

---

## 4. Final taxonomy (the shortlist) ★

**Tier 1 — hypotheses to pursue (mechanisms):**
- **A0** SHE-anchored first ET, m=0 → predicts +59 mV/pH. *Falsifiable;
  consistent with extracted sign/order, over-predicts nominal slope.*
- **A1** A0 + proton-dependent prefactor/pre-equilibrium + local-pH(β) +
  field/site correction → sub-Nernstian 30–55 mV/pH. *Family, not point.*
- **C1** electrochemical disk H₂O₂ reduction (∝ surface c_H; adds disk
  current) **+ C2** chemical peroxide decomposition (coupled H₂O₂-sink /
  O₂-source; local-pH law if any). *Robust owner of S2 + S3.*

**Tier 2 — required quantitative infrastructure (not optional, not
standalone mechanisms):**
- **G** local/surface-pH kinetic coupling — sets β; makes A1 and C
  identifiable rather than bulk-limiting.
- **H** bisulfate buffering at the acid end — bound buffer capacity
  (after acid-recipe + activity confirmation) before fitting C's order.

**Tier 3 — confounds / refinements (can mimic Tier 1; rule out, don't
lead with):**
- **D** surface-group speciation (onset curvature; mimics A — needs the
  reversible-surface discriminator).
- **E** cation/field effect (the Cs⁺ slide-15 cross-condition + anodic
  peak residual).
- **F** pzc/Frumkin ψ₁ — microscopic backing for A1/E.

**Tier 4 — near-dead:**
- **B** superoxide pKa-4.8 branching — only if raw, potential-aligned
  curves prove a real pH-4–5 non-monotonicity AND it's modeled at local
  pH.

**Minimal model to build first:** **A (A0→A1) + C (C1+C2) on top of G**,
with H bounded at the acid end. This is a **minimal mechanism family**
(not "two parameters" — A carries E0_SHE/k0/α coupling; C carries
rate + order + potential + possibly site/transport terms). It aims to
predict the whole 3-panel from a small set of physically-named knobs,
each falsifiable against a feature it was not fit to: A0 against the
onset slope; C against the S2 midpoint and the pH-1.65 ring collapse.

---

## 5. Identifiability ceiling (applies to everything)

At a **single rotation rate**, direct-4e ORR and series (2e → C1) enter
disk/ring currents as a **sum** — n_e(V) at one rpm **cannot** separate
them (corrects the earlier "n_e(V) distinguishes" claim). Therefore the
LOCKED pH-6.39 finding "the volcano is a *direct* 2e/4e O₂ competition"
is established only **up to the direct/series degeneracy**. Breaking it
needs varying residence time or an independent peroxide-reduction
measurement (§6).

---

## 6. Data-asks (priority order)

1. **rpm series** — breaks the direct/series degeneracy (§5),
   sets L_eff hence absolute k0.
2. **N₂-saturated H₂O₂ disk-reduction scans + peroxide-spike recovery
   during ORR** — bounds C1 (note: N₂ scans bound but do not transfer
   the exact rate to ORR conditions — no O₂, different surface state and
   local-pH generation — so pair with rpm or in-ORR spike recovery).
3. **ring-potential plateau check / H₂O₂ calibration at each pH** — raw
   ring is necessary but not sufficient; if ring H₂O₂ oxidation isn't
   mass-transport-limited at every pH, raw-ring collapse can be a
   detection artifact mimicking S3.
4. **acid recipe (H₂SO₄ vs other) + ionic-strength/activity model** —
   for H and for the absolute proton budget.
5. **reversible surface redox/titration + surface-chemistry readout** —
   the clean A-vs-D discriminator (irreversible modification also moves
   pzc/wettability/cation accumulation/transport).
6. **catalyst identity (CMK-3 vs GC), ring calibration/N, OCP component
   (0.47 V), iR sign used in the paper pipeline.**

---

## 7. Risks & confounds that could masquerade as "missing physics"

- **S1 axis/extraction (the gate on A).** Re-extract onset at multiple
  small *absolute* current thresholds, after background subtraction, on
  raw / sheet (+I·Rs) / physical (−I·Rs) axes, with CIs. iR at *onset* is
  small (the 0.19 V is a *plateau* figure, iR ∝ I) and largely
  common-mode across pH — but only if onset is a true small-fixed-current
  feature, not a "fraction of plateau" or noisy threshold. The
  slope-relevant axis risk is the **Ag/AgCl→RHE calibration** (sheet
  0.549 vs theoretical 0.574; 25 mV) and **per-pH OCP** — a *constant*
  offset cancels in the slope; a *pH-dependent* calibration error fakes
  S1. A survives only if the positive slope survives this.
- **Single-convention re-derivation (gate on acid/PCET ranking).**
  Re-derive every elementary step — formal-potential stoichiometry,
  local activities, BV exponent, explicit c_H factors — in one
  convention before ranking acid/PCET routes in or out (the "wrong-sign"
  acid claim is true for the current encoding but may be a convention
  artifact).
- **Derived-quantity fragility.** Fit **raw disk + raw ring** (model-side
  N), as Phase 7.2 did; treat selectivity and n_e as diagnostics only.
  S2′ in particular is untrusted.
- **N (ring calibration) and rpm (L_eff)** set the *absolute* scales;
  unknown rpm ⇒ absolute k0 conditional (does not affect A0's slope test).

---

## 8. Open questions carried forward (for the next round / experiments)

(i) Does the +59 mV/pH (A0) survive raw-axis onset re-extraction, and is
the residual hump in onset_SHE real? (ii) Once G supplies surface c_H,
what is β, and does the implied m stay physical? (iii) Can C1 vs direct-4e
be separated at all without rpm? (iv) Does A+C transfer to slide-15
(Cs⁺ pH 4) with only an E (cation) term, explaining the bump + anodic
offset — the cross-condition win the project is after?
