# Conjecture audit — fast-realignment branch (2026-05-09)

Survey of recent changes on `fast-realignment-2026-05-08` that were
**Claude- or GPT-conjecture**, classified by status after grepping
`data/EChem Reactor Modeling-Seitz-Mangan/` for clarifying citations.
Companion to `docs/PHASE_6A_INVESTIGATION_SUMMARY.md`.

The audit was triggered by the Phase 6α handoff originally pointing
Phase 6β at HSO₄⁻/SO₄²⁻ buffering before checking the data folder,
when the group's documentation actually proposes cation hydrolysis.
Hard rule #6 in `CLAUDE.md` exists to prevent that pattern.

## Verdict table

| Item | Status | Risk | Where |
|---|---|---|---|
| **Cs⁺ as production cation** vs. deck-baseline K⁺ | ❌ CONFIRMED MISALIGNED | HIGH | Linsey 2025 ACS-CATL deck slide 9: `[SO₄²⁻]=0.1 M & [H⁺]+[K⁺]=0.2 M`. Linsey 2020 deck slide 2: "ORR in pH-adjusted **K2SO4** electrolyte". Brianna's `0,1M K2SO4 data 8-15-19.xlsx`. Our Cs⁺ is one of four cations in the cation-comparison study (slide 27), **not** the production baseline. |
| **K0_R4E ratio = 1e-18** (qualitative fit) | ⚠ UNVERIFIED | MED | No measured value in folder. Linsey 2020 ButlerVolmer MATLAB has `J0_ORR_2e = 1 A/m²`, `α=0.1` — both **labeled "FITTING VALUE"**, not measured. R_4e isn't parameterized in Linsey's MATLAB at all. Our 1e-18 was Claude-fit to qualitative selectivity. Calibration target: missing `Tafel slope analysis cation-pH-Li-K-Cs.xlsx` (flagged in audit memo, not yet in folder). |
| **ALPHA_R4E = 0.5** (placeholder) | ⚠ STILL PLACEHOLDER | MED | Group docs discuss Tafel slopes generically as a diagnostic but don't give specific α₄ₑ. Defer to M4 Tafel calibration. |
| **L_eff transport-domain sweep framing** | ✅ PARTIALLY VALIDATED | downgrade to LOW | CESR Seed Proposal: "*The thickness of the boundary layer (10s of µm) is determined by the bulk flowrate*". Trienens 2025 Report: extracts limiting current as fn of "rotation rate (which impacts boundary layer thickness)". Linsey 2020 deck uses 1600 rpm → Levich δ_H≈26 µm. Our `L_eff ∈ {16, 21, 66, 100} µm` brackets this. Concept is deck-grounded; values are Claude's pick. |
| **Stern capacitance = 0.10 F/m²** | ⚠ STILL UNVERIFIED | LOW-MED | NSF proposal describes inner layer "(< 1 nm)" and screening layer "(~50 nm)" qualitatively. Linsey 2025 deck slide on "Stern Modification (1924) Inner & Outer Helmholtz Planes" describes the concept. **No specific C_S value cited in folder.** Worth checking `docs/Ruggiero2022_JCatal_manuscript.pdf` directly. |
| **SO₄²⁻ Bikerman radius = 2.4 Å** | ⚠ STILL UNVERIFIED | LOW | No SO₄²⁻ radius mentioned anywhere in folder. Acknowledged placeholder in `_bv_common.py:594`. |
| **Pass A V_RHE band [+0.10, +0.80]** | ⚠ debug convenience | LOW | Narrower than deck page-15 band; production sweep `mangan_full_grid` uses correct `linspace(-0.40, +0.55, 25)`. |
| **Phase 6α water self-ionization as buffer mechanism** | ⚠ caught & corrected | n/a | Documented in §9 of handoff #26 — group's primary buffer is cation hydrolysis at the OHP, not water self-ionization. Phase 6α is universal aqueous physics but sub-leading for this electrolyte. |

## Verified-grounded (no audit concern)

| Item | Source |
|---|---|
| Parallel 2e/4e ORR topology | Ruggiero 2022 §1 |
| `E_eq_2e = 0.695 V`, `E_eq_4e = 1.23 V` | Ruggiero 2022 §1 Eqs 1-2 |
| `C_O2 = 1.2 mol/m³` (salting-out at I=0.3 M) | Ruggiero 2022 §2.4 (Linsey 2020 MATLAB had a wrong 5 mol/m³ "dry H₂O saturation") |
| `ALPHA_R2E = 0.627` | Ruggiero 2022 (Linsey 2020 MATLAB had a "fitting" α=0.1, superseded) |
| `N_collection = 0.224` | Ruggiero 2022 §2; Linsey 2020 deck slide 13 confirms |
| pH 4 bulk operating point | Ruggiero 2022 §2 |
| Bulk K₂SO₄ at 0.1 M (I = 0.3 M) | Linsey 2025 ACS-CATL deck slide 9; Linsey 2020 deck; Ruggiero 2022 §2 |
| Boundary-layer thickness as mass-transport handle | CESR Seed Proposal; Trienens 2025 Report |
| 1600 rpm RDE rotation rate | Linsey 2020 deck slide 11 |
| RRDE catalyst = CMK-3 (carbon) | CESR docs; Linsey deck |

## Bonus finding: Phase 6α water-self-ionization is precedented

The Linsey 2020 `ButlerVolmer MATLAB/ButlerVolmerKinetics.m` already
tracks `c_OH(y)` via the Kw closure:

```matlab
pH_0(ii) = -log10(C_H_0(ii)/1000);
C_OH_0(ii) = 10^(-(14-pH_0(ii)))*1000;
```

with an explicit comment acknowledging the gap that motivated Phase 6α:

> *"Currently, it is possible to calculate a current that would
> consume more protons than are available. This is not physically
> possible. We need to incorporate possibility of both acid &
> alkaline mechanisms (allow either/both consumption of protons or
> production of hydroxide ions). Not yet sure how to do this in a
> physically accurate way."*

So Phase 6α water-ionization is **precedented** in the group's earlier
modeling code — just not as a primary buffer mechanism, more as a way
to track c_OH alongside c_H. Linsey acknowledged the H⁺-only model
was insufficient in 2020 but didn't have a clean solution.

## Recommendations for Phase 6β planning

1. **Run a K⁺/SO₄²⁻ sweep alongside Cs⁺/SO₄²⁻** — apples-to-apples
   with the deck baseline (`[SO₄²⁻]=0.1 M, [K⁺]=0.2 M` per Linsey
   2025) and a falsifiable test of the cation-hydrolysis story
   (predicted: pH(K⁺) ≫ pH(Cs⁺) at L=16 µm cathodic, since K⁺'s
   near-cathode pKa is 8.49 vs Cs⁺'s 4.32). Brianna's
   `0,1M K2SO4 data 8-15-19.xlsx` and the four-cation CP data are
   the ground truth to compare.
2. **Defer K0_R4E and ALPHA_R4E calibration** until cation hydrolysis
   lands — without correct local pH the Tafel slope and selectivity
   would calibrate to wrong-regime physics.
3. **Confirm Stern C_S against `docs/Ruggiero2022_JCatal_manuscript.pdf`**
   before accepting 0.10 F/m² as production. If it's not in Ruggiero
   either, flag and treat as another tunable.
4. **Reframe L_eff sweep semantics in writeups** — the boundary-
   layer-thickness concept *is* deck-grounded (CESR Seed, Trienens
   2025); the specific sweep values were our pick. Future write-ups
   should cite the boundary-layer concept, not the L_eff parameter
   as a deck quantity.
5. **Request the missing `Tafel slope analysis cation-pH-Li-K-Cs.xlsx`**
   from Seitz/Mangan again — it's the calibration source for items 2
   and 3, and it's the only piece of the data audit (per
   `docs/seitz_mangan_data_folder_audit_2026-05-08.md`) that's still
   missing from the folder.

## Net status

The HIGH-risk Cs⁺ vs K⁺ misalignment is now **definitively confirmed**
by the deck's own electrolyte spec. **First Phase 6β validation step
should be a K⁺/SO₄²⁻ rerun on the same sweep grid**, before any
further calibration.

Remaining uncited items (Stern_C, SO₄²⁻ radius, K0_R4E, ALPHA_R4E)
are calibration-target placeholders, not misaligned conjectures.
The L_eff framing is more grounded than initially rated.
