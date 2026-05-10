# Data-folder code inventory — `data/EChem Reactor Modeling-Seitz-Mangan/`

What's in the folder besides the experimental data, organized by what
it's useful for. All paths relative to that folder.

## 1. Linsey/ButlerVolmer MATLAB/  (Linsey Seitz, ~2020)

| File | Size | What it is | Useful for |
|---|---|---|---|
| `ButlerVolmerKinetics.m` | 6.3 KB | 1-D forward-Euler surface-kinetics solver: applies BV at the disk, tracks `C_O2 / C_H2O2 / C_H / C_OH` over 20 timesteps. **Includes water self-ionization closure** (`c_OH = Kw/c_H` via `10^(-(14-pH))*1000`). Comment block acknowledges H⁺-only insufficiency. | Phase 6α validation (different implementation of the same Kw closure); 2020-vintage acknowledgment of the gap that Phase 6α addresses |
| `ButlerVolmerFitting.m` | 2.0 KB | Generic fsolve-based BV fitter for LSV → (η, J). FITTING values: `J0=0.1 mA/cm²`, `α=0.1`, `redox=0.695 V`, `n=2`. | Tafel-slope fitting reference; not a production reference (J0/α are explicitly fitting handles) |
| `CurrentVector.m` | 1.5 KB | Small helper that builds log-spaced current vectors for BV plotting. | Utility only |

**Don't pull as ground truth**: the J₀ and α values in this MATLAB are
explicitly labeled "FITTING VALUE" — they're not deck measurements.
The Ruggiero 2022 values supersede them in production.

**Pull-able insight**: the Kw closure pattern in
`ButlerVolmerKinetics.m` is the same one Phase 6α landed in
`Forward/bv_solver/water_ionization.py`. Linsey's 2020 acknowledgment
of the H⁺-only-insufficiency gap predates Phase 6α by 6 years.

## 2. Brianna/20201024 CP Experiment Data-Code/

| File | Size | What it is | Useful for |
|---|---|---|---|
| **`CP_data.csv`** | 1.6 KB | **Clean tabular steady-state CP results: 60 rows = 4 cations × 3 pH × 5 currents.** Columns: `Cation, Bulk pH, Disk CP, Ring OCP, Disk Potential`. **`Ring OCP` is the IrOx pH-probe potential** — the experimental local-pH signal we should compare our `surface_pH_proxy` against. | **HIGH VALUE** — direct ground truth for Phase 6β cation-hydrolysis validation |
| `{Cs,K,Na,Li}2SO4_10-9-20.mat` | ~MB each | Full time-series CP data (3600 s × 5 currents × 3 pHs per cation). Same content as `CP_data.csv` but with raw temporal sweeps. | Source for the .csv averages; useful only if temporal artifacts need investigation |
| `Summary Data-Error.xlsx` | — | Error-bar summary of the CP data | Variability bounds for the .csv numbers |
| `plot_CP_data.m` | 14 KB | MATLAB plotting script for the .mat files | Reference for how Brianna organized the .mat structure |

**Pull-able insight: CP_data.csv is the most concise calibration target
in the whole folder**. Plot Ring OCP vs Disk CP across the 4 cations at
each bulk pH and you have an experimental analog of our `surface_pH_proxy`
vs `cd` curves. Phase 6β acceptance criterion should be: model predicts
the Cs⁺/K⁺/Na⁺/Li⁺ ordering of Ring OCP at a given current.

Specific flagged data: **`Cs,4,-0.65, ..., -1.5537`** — at pH 4, highest
cathodic current, the disk drops to −1.55 V on Cs₂SO₄ but stays at
−0.50 V on K₂SO₄. That's the buffer-active signature for Cs⁺ at its
near-cathode pKa = 4.32.

## 3. Yash-Trends/Data and Plotting.zip  (Yash, recent)

| File | What it is | Useful for |
|---|---|---|
| `plotting.ipynb` | 5-cell Jupyter notebook. **Reads `Tafel slope analysis cation-pH-Li-K-Cs.xlsx`** (the file flagged as missing in the data audit — Yash had it). Processes Cs⁺ pH 4 sub-block, plots experimental vs simulation. | **HIGH VALUE** — confirms the missing Tafel file existed; provides RRDE conversion math; gives the Cs⁺ pH 4 LSV processing pipeline |
| `Data/best_fit_base/sim_*.npy` | ~200 pre-computed simulation snapshots from Yash's 6-species PNP+BV. Each `.npy` is a dict with `estimated_voltage`, `current_density`. | **MED VALUE** — direct numerical comparison target for our Cs⁺ pH 4 simulation |
| `Data/{0.16_0.5e-9_0.065_4e-6_6e-6}/sim_*.npy` | Parameter-swept sim results, named by parameters | Parameter-sweep reference; folder name gives parameter values |

**Pull-able from `plotting.ipynb`**:

1. **RRDE peroxide flux conversion**:
   ```python
   j_H2O2 = j_ring * 0.11 / (0.224 * 0.196)
   ```
   `0.224` = N (collection efficiency, matches our `N_COLLECTION`),
   `0.196 cm²` = disk area, `0.11 cm²` = ring area. This is the
   canonical conversion for the deck's RRDE setup.

2. **V_OCP / Ag-AgCl-to-RHE correction**:
   ```python
   V_OCP = 0.47 + 0.197 + 0.059 * pH
   ```
   `0.197 V` = Ag/AgCl reference vs SHE; `0.059·pH` = SHE→RHE shift;
   `0.47 V` = a per-electrode offset (their measurement convention).
   At pH 4: V_OCP ≈ 0.903 V.

3. **Cs⁺ pH 4 data is at sheet 'Cs+', rows 27-1668, columns 18-29 of
   `Tafel slope analysis cation-pH-Li-K-Cs.xlsx`.** The xlsx itself
   isn't in the folder, but Yash's code shows it was structured as
   one sheet per cation with multi-pH column groups.

## 4. Yash-Trends/Results.zip

15 SVG plots. No code. Plot titles imply Yash already swept:
`L_bulk`, `Peroxide_j0`, `Peroxide_Tafel`, `Stern`, `size_factor`
across `Disk / H2O / H2O2` — i.e., he has results for L_eff sweeps,
J₀ sensitivity, Tafel-slope sensitivity, Stern-capacitance sensitivity,
and Bikerman-size-factor sensitivity. **Look at these SVGs before
re-running our equivalent sweeps** — Yash may have already explored
the parameter ranges we're considering.

## What we should pull / use

1. **`CP_data.csv`** — make this the primary Phase 6β acceptance
   dataset. 60 rows, 4 cations × 3 pH × 5 currents, Ring OCP =
   experimental local pH. **No new experiments needed, calibration
   target is already on disk.**

2. **`Yash-Trends/Data and Plotting.zip`** — open it in a real Python
   env, regenerate Yash's Cs⁺ pH 4 simulation comparison plot, and
   put our Phase 6α Cs⁺ pH 4 IV curve next to it. If Yash's
   pre-Phase-6α model already matches Cs⁺ pH 4 better than our
   Phase 6α run does, that's a clue our infrastructure has a bug
   (or Yash had additional physics we don't).

3. **`ButlerVolmerKinetics.m`** — port the Kw-closure block to
   Python and confirm it matches our `water_ionization.py`. Two
   independent implementations of the same closure should agree.

4. **`ButlerVolmerFitting.m`** — port to Python as a Tafel-slope
   fitter. We need this for the M4 Tafel calibration step (currently
   blocked by missing `Tafel slope analysis cation-pH-Li-K-Cs.xlsx`,
   but the fitter itself is reusable).

## What we should not pull

- The fitting J₀ and α values from `ButlerVolmerKinetics.m`
  (`J0_ORR_2e=1, alpha_ORR_2e=0.1`) — these are 2020 fitting handles,
  not deck measurements. Ruggiero 2022 values supersede.
- C_O2_bulk = 5 mol/m³ from `ButlerVolmerKinetics.m` — that's
  saturation in dry water; Ruggiero §2.4's 1.2 mol/m³ (salt-corrected
  at I=0.3 M) is the deck-correct value.

## Provenance

This inventory was generated 2026-05-09 23:30 CDT after the
conjecture-audit pass. CLAUDE.md hard rule #6 covers when to consult
the data folder; this doc covers what's actually pull-able from it.
