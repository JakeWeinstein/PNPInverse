# K₂SO₄ pH 6.39 dual-series RRDE target — provenance (Phase 7.2 Stage 0)

Source: `data/EChem Reactor Modeling-Seitz-Mangan/Brianna/0,1M K2SO4
data 8-15-19.xlsx` (Brianna Ruggiero, 2019-08-15). Extractor:
`scripts/studies/_extract_k2so4_ph6p39_rrde.py`. Machine-readable
numbers: `StudyResults/phase7p2_stage0_extraction/extraction_report.json`.

## What the file contains

- Raw LSV sheets `cycle 1/2/3` (2103–2105 rows each = descending
  sweep ~1053 pts + return sweep), columns Edisk/V (vs Ag/AgCl),
  Idisk/mA, Iring/mA, plus in-sheet processing formulas.
- In-sheet constants (verified from cells): A_disk = 0.19635 cm²
  (Ø 5 mm), A_ring = 0.109956 cm² (7.5/6.5 mm OD/ID), Rs = 125 Ω,
  N = 0.224, Ag/AgCl→RHE cal = +0.549 V (measured; theoretical
  0.197 + 0.059·6.39 = 0.574 V — 25 mV discrepancy, documented,
  drives the OCP-shift bracket in the plan).
- `Exp Info` catalogs six same-day pH conditions (6.39, 5.21, 4.21,
  3.42, 2.35, 1.65); only pH 6.39 traces are in this file.
- Cycle 1 excluded: in-sheet remark "pH changed to 6.11 while ring
  cleaning" + grossly different trace (conditioning). Cycles 2+3
  are near-identical replicates and form the target.

## Verification

- Every processed column (V_cal, V_iR, j_disk, j_ring, Sel%, n_e)
  re-derived from raw + constants matches the workbook's cached
  values with |Δ| ≤ max(1e-8 abs, 1e-6 rel): **0 failures** across
  all three cycles. Main-sheet group 2 = cycle 2 rounded to 5 dp
  (checked at 1e-4 abs); main-sheet group 1 = cycle 1 full
  precision.
- Sign conventions (verified from the data, not assumed): raw I is
  **anodic-positive** (+0.35 mA/cm² scan-start transient at
  V_RHE ≈ 1.09 pre-ORR; negative on the ORR plateau). Canonical
  output: disk cathodic-NEGATIVE, ring anodic-POSITIVE (raw ring).
  NOTE: the approved plan's parenthetical guess "raw Idisk is
  cathodic-positive" was wrong; no sign flip exists anywhere in the
  pipeline. Conventions above are the file's own.

## iR-correction audit (substantive finding)

The sheet's "iR New" column computes **V_sheet = V_cal +
(I_mA/1000)·Rs**. With the file's anodic-positive current
convention, the physical interface potential is **V_phys = V_cal −
(I_mA/1000)·Rs** (measured minus ohmic drop). The sheet sign is
nonstandard; at the ORR plateau (I ≈ −0.76 mA, Rs = 125 Ω) the two
axes differ by **0.19 V**. Consequences:

- **Canonical fit axis = V_phys** (physically corrected). The sheet
  axis is carried in every CSV as `v_sheet` and is a REFIT
  sensitivity (axis-variant) in Stage 4.
- The group's published figures presumably inherit the sheet
  convention; cross-condition comparisons (e.g. slide-15, Rs ≈ 50 Ω
  → up to ~30 mV) carry this caveat.
- This audit was *required* by the plan (Stage 0 item 2); the
  plan's "canonical V = their iR-corrected axis" presumption is
  overridden by the audit outcome, documented here. DATA ASK: ask
  Brianna/Linsey which sign their final paper pipeline used.

## Ring baseline

Per-cycle median of raw j_ring over the pre-ORR shelf V_phys ∈
[0.60, 1.00] (ring flat there; ~350 pts): 0.00464 (c2) / 0.00458
(c3) mA/cm²_ring, spreads ~2e-4. Subtracted; spread folded into
σ_ring. (The plan's original "disk non-cathodic" region caught only
11 scan-start transient points — replaced, documented deviation.)

## Disk LSV background (documented deviation, load-bearing)

The descending sweep carries a **ring-silent cathodic shelf**:
j_disk ≈ −0.21 → −0.33 mA/cm² over V_phys 1.0 → 0.6, ~60× the
Exp-Info capacitance band, ~7–9% of the plateau, with the ring flat
at baseline throughout (no H₂O₂ ⇒ not 2e ORR; smooth non-Tafel ramp
⇒ not 4e onset). This is capacitive/pseudocapacitive LSV background
(consistent with a high-surface-area porous carbon disk) — a
scan-rate artifact a steady-state model must not fit.

Treatment: linear background bg(V) fitted on V_phys ∈ [0.65, 1.00]
(slopes ≈ +0.13 mA/cm²/V, anchor MAD ≈ 0.001–0.0014), subtracted
across the sweep; extrapolation uncertainty σ_bg = sqrt(MAD² +
(0.5·|slope|·span)²) (≈ 0.033 mA/cm² at the cathodic end) added in
quadrature to σ_disk. **Validation:** the (down+up)/2 sweep average
cancels first-order capacitive current; background-corrected
down-sweep agrees with it to median |Δ| ≈ 0.026–0.031 mA/cm² over
the window (p90 ≈ 0.19, concentrated at onset where faradaic
hysteresis is real). A background-scale ×{0.5, 1.5} REFIT joins the
Stage 4 sensitivity table. The plan's original window+σ-only
treatment assumed the background ≈ cap band (0.004); reality is 60×
that, so windowing alone would have either fit background as ORR or
discarded the onset.

## Canonical RRDE quantities

From CURRENTS with baseline-corrected ring:
Sel% = 200·(I_r/N)/(|I_d| + I_r/N); n_e = 4|I_d|/(|I_d| + I_r/N).
The sheet's own formulas mix densities normalized by different
areas, overweighting the ring term by A_d/A_r = 1.786: sheet peak
Sel 73.2% (Exp Info) vs canonical **63.4%** (background-corrected
disk); sheet n_e ≈ 2.8 vs canonical band 2.85–3.52. Sheet columns
are provenance only.

## Target definition

- Window (predeclared rule, background-aware): V_phys ∈
  [**0.1317, 0.5591**], 30 bins; descending sweep only.
- Binned mean of cycles 2+3; σ = max(|c2−c3|/√2, floor, 0.02·|j|) ⊕
  σ_bg (disk, quadrature). σ is a conservative single-observation
  predictive scale (NOT SEM); absolute reduced-χ² is not
  interpreted.
- Files: `data/k2so4_ph6p39_rrde_{cycle2,cycle3,binned}.csv`
  (git add -f per repo precedent). Binned columns include pc_equiv
  (= −j_ring·A_r/(N·A_d), PLOTS ONLY — the objective fits raw ring
  with model-side N) and the n_e diagnostic band.

## Gates and checks

- Cross-check: PASS (0 failures). σ > 0 every bin: PASS.
- n_e diagnostic band: [2.85, 3.52] within alarm bounds [1.8, 4.2]
  → no alarm; mixed 2e/4e everywhere, 2e-richest (n_e ≈ 2.9) at the
  ring peak (~0.42 V), 4e share growing cathodically.
- Air-saturation impossibility: corrected plateau 3.50 mA/cm² >
  air-sat 4e ceiling at every L_eff bracket end (1.53 @ 12 µm,
  1.19 @ 15.4, 0.84 @ 21.7) ⇒ **air-saturated electrolyte excluded
  by the data**; O₂-saturated assumption stands (ask remains for T/
  solubility detail).

## Data asks (carried from the plan, plus one new)

rpm; gas/T/O₂-saturation protocol; catalyst identity (CMK-3 vs GC);
ring hold potential + calibration; sibling pH files; **iR sign used
in the final paper pipeline (new, from the audit)**; raw Tafel
xlsx; 0.47 V OCP component.
