# Missing data — running tracker

Living doc of experimental / literature data the project still needs
from external sources (Seitz/Mangan group, papers behind paywalls,
etc.). Each entry says: what we need, why it's blocking, what's the
fallback, and how it gets requested.

When an item is delivered, **don't delete it** — flip its status
to `RESOLVED` and add the resolution date + delivery path. Keeps
the audit trail.

**Also see:**
- `docs/seitz_mangan_data_folder_audit_2026-05-08.md` — what's
  already in `data/EChem Reactor Modeling-Seitz-Mangan/`
- `docs/CONJECTURE_AUDIT_2026-05-09.md` — flags model-side
  conjectures; some pull on missing data items below
- `docs/phase6b_next_steps_plan.md` §8 — Phase 6β v9
  unresolved-physics ledger (L1–L9); some entries cross-reference
  here

---

## Status legend

- `OPEN — REQUESTED` : asked the source, waiting
- `OPEN — UNREQUESTED` : haven't asked yet, would unblock something
- `OPEN — PAYWALLED` : public/literature item, may need
  institutional access
- `OPEN — WORKAROUND IN PROGRESS` : we have a path that doesn't
  need the data; fallback being built
- `RESOLVED` : delivered or extracted; entry kept for audit
- `WONTFIX` : decided not to chase; explain why

---

## Open items

### M1. Tafel slope analysis cation-pH-Li-K-Cs.xlsx

- **Status:** `OPEN — REQUESTED — K⁺ workaround delivered 2026-05-10`.
  Original ask 2026-05-08; pH 6.39 K⁺ slopes extracted via Phase 6β v9
  post-Gate-4 plan §F.
- **Filename expected:** `Tafel slope analysis cation-pH-Li-K-Cs.xlsx`
- **Source:** Linsey Seitz / Brianna Ruggiero (Seitz group at NU)
- **What it should contain:** per-(cation, pH) Tafel slopes for ORR
  in M₂SO₄ electrolyte (M ∈ Li, Na, K, Cs; pH ∈ ~1–6) measured on
  CMK-3 carbon at the disk electrode of an RRDE. Yash's
  `Yash-Trends/.../plotting.ipynb` explicitly reads this file via
  `pd.read_excel(..., sheet_name='Cs+')` so the column schema
  Yash expects is also a constraint:
  - `V_RHE_iRdisk_LSV (V vs RHE)`, `J_ring_LSV_bl`, …
  - one sheet per cation (`Cs+`, `K+`, `Na+`, `Li+`)
  - data start row ~27 (header at top)
- **Why we need it:** unblocks Phase 6β v9 6β.2 calibration of
  `K0_R4e` (currently 1e-18 placeholder ratio against `K0_R2e`)
  and `ALPHA_R4e` (currently 0.5 placeholder). Pulls on ledger
  L9 + L7 (C_S calibration as a side effect — they share the
  cathodic kinetics fit).
- **What's blocked:** 6β.2 K0_R4e / α_R4e fitting; Yash's `plotting.ipynb`
  pipeline is non-runnable without it.
- **Workaround delivered (2026-05-10, Phase 6β v9 post-Gate-4 plan §F):**
  K⁺-only Tafel slopes extracted from
  `Brianna/0,1M K2SO4 data 8-15-19.xlsx` and written to
  `data/derived/k_plus_tafel_slopes_from_brianna_2019.xlsx`
  (+`.json` companion). Extractor:
  `scripts/derive/extract_k_plus_tafel_slopes.py`. **Scope caveat
  found during extraction:** the source xlsx documents 6 disks at
  6 pH values (6.39, 5.21, 4.21, 3.42, 2.35, 1.65) in its `Exp Info`
  sheet, but **only Disk 1 (pH 6.39) has its raw E-vs-j LSV traces
  in the workbook** (`cycle 1`/`2`/`3` sheets). Disks 2–6 have
  summary statistics only (ring onset potential, max ring current,
  peroxide selectivity) — no LSV scan tables.
  Delivered fits at pH 6.39 (cycles 1–3, R² = 0.996, |j| ∈ [10%, 60%]
  |j_lim|): slope ≈ 270–310 mV/decade (mean 282), intercept
  log10|j| at E=0 V ≈ 1.22–1.41. The 270–310 mV/decade is on the
  high end of the textbook 60–120 mV/decade for kinetic-controlled
  ORR — consistent with mixed kinetic+mass-transport regime; a
  Koutecky–Levich correction would pull this lower but requires the
  RRDE rotation rate (not in `Exp Info`; would need to read it
  from the lab notebook or a separate file).
  All-cation extraction from the `*_10-9-20.mat` files is **NOT
  feasible** for Tafel — those are CP (fixed-current) waveforms,
  not LSV (E-vs-j); they don't contain the needed kinetic-region
  E vs. j sweep.
- **Status remaining:** still need the original xlsx (or the
  pH 5.21/4.21/3.42/2.35/1.65 LSV traces) to extend Tafel beyond
  pH 6.39. K⁺ pH 6.39 is delivered.
- **How to request:** ping Linsey/Brianna directly with
  "still missing the Tafel xlsx; can you re-send when convenient?"
  Reference the 2026-05-08 ask.

### M2. C_S (Stern compact-layer specific capacitance) for CMK-3 carbon under ORR conditions

- **Status:** `OPEN — UNREQUESTED`
- **Source:** Seitz group (Brianna's experimental measurements) or
  an external CMK-3 capacitance reference
- **What we'd need:** measured `C_S` value (in F/m² or µF/cm²) for
  the CMK-3 carbon catalyst at the operating points of interest
  (V_RHE in [−0.4, +0.55], 0.1 M K₂SO₄ at pH 4). Ideally a single
  number, possibly with a V-dependence if it varies across the
  scan window.
- **Why we need it:** the production solver currently uses
  `C_S = 0.10 F/m²` which has **no Ruggiero / deck citation**
  (verified by full-text grep — see `docs/CONJECTURE_AUDIT_2026-05-09.md` §5.3).
  It traces to `docs/stern_layer_physics_and_next_steps.md`
  (2026-05-03) line 214 — a sweep design picked from the
  textbook 5–100 µF/cm² range; the May 2026 sweep selected
  0.10 because it was the smallest finite-Stern that crossed the
  +1.0 V wall. It's a convergence-pinned engineering choice, NOT
  a deck-calibrated parameter.
- **What's blocked:** Phase 6β v9 ledger L7 (Stern capacitance
  calibration). Currently being treated as a labelled tunable
  with sensitivity sweep `C_S ∈ {0.05, 0.10, 0.20}` at Gate 4B.
- **Workaround in progress:** sensitivity sweep at Gate 4B + 6β.2.
  If the v9 architecture works robustly across the [0.05, 0.20]
  bracket, we don't strictly need a measured value. If sensitivity
  is high, we'd want either a measured value or a Bohra 2019-style
  literature anchor.
- **How to request:** future ask; lower priority than M1 because
  the sensitivity sweep gives partial cover. Bohra 2019
  (`10.1039/c9ee02485a`, already in `data/.../Articles/`) may
  provide a literature-anchored Cu/CO₂RR value that informs but
  doesn't directly transfer (different cathode material).

### M3. IrOx local-pH probe measurements (ring-current → ring-pH calibration)

- **Status:** `OPEN — UNREQUESTED`
- **Source:** Linsey 2025 ACS-CATL deck slides 5–9 (per
  `CHATGPT_HANDOFF_26_phase6a_outcome_and_phase6b_scoping.md` §9
  + `CLAUDE.md` Hard Rule #6)
- **What we'd need:** raw or processed IrOx ring-current data with
  the conversion to local pH at the disk surface, ideally as a
  function of disk current density and cation. The deck has
  schematics and analysis but the underlying data may be in a
  separate xlsx / .mat that wasn't included in the data drop.
- **Why we need it:** ground-truth for our solver's *predicted*
  surface pH (Phase 6α currently predicts 10.58 at L=16 µm at
  V_RHE = −0.40 V on Cs⁺/SO₄ — does the IrOx measurement agree?).
  Without this, the surface-pH-side validation of v9 is purely
  theoretical (deck slide 27 cation-series predictions).
- **What's blocked:** Phase 6α empirical-truth check (also queued as
  Phase 6α.1 finite-rate Kw refinement trigger — only worth doing
  if IrOx measurement disagrees with our fast-equilibrium prediction).
  Phase 6β v9 6β.2 holdout validation against measured local pH
  (currently we compare model surface pH to the deck slide 27
  *predicted* values, not to measured values).
- **Workaround in progress:** none directly. The deck's slide-27
  per-cation pKa-near-cathode table (Li 13.16, Na 11.44, K 8.49,
  Cs 4.32) effectively encodes the IrOx-measured local pH at one
  operating point — but it's a single-point summary, not the
  full V × cation × pH dataset.
- **How to request:** ask Linsey if the underlying data behind
  slides 5–9 of `Trienens_Report_2025/20250818-ACS-CATL-EChem Rxn Enviro for ORR-LSeitz.pptx`
  is available as a standalone file.

### M4. Cathode-side specific capacitance for the ORR/CMK-3 system (independent of M2)

Marked separately from M2 because M2 asks for a *single* `C_S`
value while there's a deeper question: does CMK-3 carbon have
strong V-dependence in the diffuse-layer capacitance that the
flat `C_S = 0.10` model can't capture? If so, the v9 solver's
Stern BC is the wrong functional form, not just the wrong number.

- **Status:** `OPEN — UNREQUESTED` (but lower priority — we'd only
  chase this if the C_S sensitivity sweep at Gate 4B / 6β.2 shows
  the [0.05, 0.20] flat-C_S bracket can't reproduce the deck
  shape)
- **Source:** Bohra 2019 EES (Ruggiero ref 71, in `data/.../Articles/`)
  is the literature precedent for V-dependent C_S in
  CO₂-electrocatalysis double-layer modeling — start there.
- **Workaround in progress:** the flat C_S model in v9 is
  good-enough until the Gate 4B sensitivity sweep says otherwise.

---

## Resolved items (kept for audit)

### M0. Singh 2016 JACS SI (`10.1021/jacs.6b07612`)

- **Status:** `RESOLVED 2026-05-10`
- **Delivered as:** `data/CO2RR Cation Supplementary Information.pdf`
  (315 KB, user fetched via institutional credentials after ACS
  paywall returned 403 to direct download)
- **What it gave us:** field-dependent pKa formula (Eq. 4),
  per-cation Singh Table S1 (z_eff, r_M, n_hyd, pKa_bulk),
  Singh constants (A=620.32 pm, B=17.154, r_O=63 pm), and the
  per-cathode-material `r_H_El` Cu values back-fitted from
  Table S3.
- **Documented at:** `docs/singh_2016_pka_formula.md`
- **What it unblocked:** Phase 6β v9 ledger L1; Gate 4A's
  `pka_shift_form="singh_2016_eq_4"` is now a physical formula,
  not a falsification placeholder.

---

## Update protocol

When an item is delivered or extracted, edit the entry in place:
1. Flip status to `RESOLVED <date>`.
2. Move the entry to the "Resolved items" section at the bottom.
3. Add a "Delivered as:" path or DOI.
4. Note what it unblocked.

When a new item is identified:
1. Add to the "Open items" section with the next sequential
   `M<N>` ID (M1, M2, …; resolved items keep their ID).
2. Fill in all five fields (status, source, what, why, blocked,
   workaround, request path).
3. Cross-reference any ledger items in `docs/phase6b_next_steps_plan.md` §8
   that the data would close.
