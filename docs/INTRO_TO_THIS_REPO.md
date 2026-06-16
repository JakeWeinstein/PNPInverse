# An intro to this repo

A cold-start tour of **PNPInverse**. Read this first; then `REPO_LAYOUT.md`
(file map), `CLAUDE.md` (hard rules), and the full research narrative in
`README.md`. This doc is the broad strokes: what the project is, how the
research got here, where to start reading code, and how to call the solver.

---

## 1. What this is, in one breath

We simulate the **oxygen reduction reaction (ORR)** on a porous carbon
rotating-ring-disk electrode by solving the **Poisson–Nernst–Planck / Butler–Volmer
(PNP-BV)** equations with Firedrake finite elements, and compare against the
Seitz/Mangan group's experimental data (K₂SO₄ electrolyte, pH 2–6). The eventual
goal is the *inverse* problem — infer kinetic parameters from measured current —
but that is **paused**; the **forward solver is the active surface** today.

The chemistry is two **parallel** electron-transfer branches (Ruggiero 2022 §1):

```
R_2e:  O2 + 2H+ + 2e-  ->  H2O2     E° = 0.695 V_RHE   (peroxide)
R_4e:  O2 + 4H+ + 4e-  ->  2 H2O    E° = 1.23  V_RHE   (water)
```

The science question is **selectivity**: how much current goes to peroxide vs.
water, and how that splits with potential and pH. The four kinetic unknowns are
`[log_k0_2e, log_k0_4e, alpha_2e, alpha_4e]`.

## 2. The production model (what the solver actually solves)

- **3 dynamic species** transported by PNP: O₂, H₂O₂, H⁺ — plus **analytic
  Bikerman counterions** (deck baseline **K⁺/SO₄²⁻**; Cs⁺/Na⁺/Li⁺ for the
  cation-comparison study). Counterions are a closed-form steric profile, not
  extra PDE unknowns.
- **`formulation="logc_muh"`** — log-concentration primary variables, with the
  proton carried as an electrochemical potential `mu_H = u_H + em·z_H·phi`.
- **Log-rate Butler–Volmer**, the two reactions above in **parallel**.
- **Stern compact-layer Robin BC** on Poisson at the electrode, `C_S = 0.20 F/m²`
  (literature-locked).
- **`debye_boltzmann` initial condition** (composite-ψ + multispecies-γ).
- Optional opt-in physics: **Phase 6α** water self-ionization, **Phase 6β** cation
  hydrolysis (field-dependent pKa). Both default OFF and are byte-equivalent to
  the base stack when off.

Everything is **2-D** (x,y mesh) and **non-dimensionalized** (see `Nondim/`).

## 3. The research arc (why the code looks like it does)

The git history and `docs/phase*/` track a multi-month campaign. Broad strokes:

| Stage | What happened | Key artifacts |
|---|---|---|
| **Forward rebuild** (Apr–May 2026) | Rebuilt the solver around `logc_muh`, parallel 2e/4e, Bikerman counterions, Stern Robin BC. Removed the legacy concentration backend. | `Forward/bv_solver/forms_logc_muh.py`, `docs/solver/` |
| **Phase 5 (α/γ)** | Got the multi-ion + Stern stack to *converge*. The cold C+D dispatcher fails on this stack, so we built **anchor-and-grid** continuation (converge one anchor voltage, then warm-walk the grid). Pass A = 8/8. | `Forward/bv_solver/anchor_continuation.py`, `picard_ic.py` |
| **Phase 6α** | Plumbed water self-ionization. **Retired as primary** — the surface-pH gate failed (pH 10.58, unphysical). | `water_ionization.py`, `docs/phase6/PHASE_6A_INVESTIGATION_SUMMARY.md` |
| **Phase 6β** | Group's real hypothesis: **cation hydrolysis at the polarized OHP** (Singh 2016 field-dependent pKa). Calibrated v9 → v10a → **v10b** (Γ_max=0.047, k_des=1.0, C_S=0.20, literature-locked). | `cation_hydrolysis.py`, `calibration/v10b.py`, `docs/phase6/v10b_calibration_summary.md` |
| **Phase 6β Step 10 (Phase D)** | K-only Δ_β fit → **`OUTCOME_C_NON_IDENTIFIABLE`**. Root-caused: total current was **pinned at the H⁺ Levich (transport) cap**, so every fit knob was tuning a transport-capped current — not kinetics. | `docs/phase6/phase6b_step10_phase_D_summary.md` |
| **Phase 7** (active lineage) | Pivot: **dual-pathway water-as-proton-donor kinetics + RRDE-correct `L_eff` + slide-15 volcano fit**. Reproduced the volcano. | `docs/handoffs/CHATGPT_HANDOFF_41_*` |
| **Phase 7.2** | **Locked** a K₂SO₄ pH-6.39 disk+ring dual-series fit against *real* LSV data: the water-route model fits, kinetics transfer across pH, the disk/ring partition is ring-determined. | `docs/handoffs/CHATGPT_HANDOFF_43_*` |
| **Phase 7.3** (in flight) | Rank the *missing* mechanism behind pH-flatness. Finding: onset shifts **+41 mV/pH on the RHE scale** ⇒ a **proton-uncoupled first electron transfer**; peroxide-consumption owns selectivity. | `docs/handoffs/CHATGPT_HANDOFF_4{4,5}_*`, `tasks/todo.md` |
| **Inverse / surrogate** | Direct-PDE inverse + surrogate inference stacks — **paused** until the forward model is settled. Source packages remain; the driver scripts were pruned in the 2026-06-15 cleanup. | `docs/inverse/CHATGPT_HANDOFF_10_*` |

If you only read one narrative artifact, read `writeups/May13th/phase_6_overview.pdf`
(the selectivity-gap story through Phase D) and then the Phase 7 handoffs #41–#45.

## 4. Where the production solver lives

```
Forward/                         core package
  bv_solver/
    dispatch.py                  routes build_context / build_forms / set_initial_conditions by formulation
    forms_logc_muh.py            THE production weak form (logc + muh + Stern Robin + parallel 2e/4e)
    forms_logc.py                logc (non-muh) variant
    boltzmann.py                 analytic Bikerman counterion closure
    multi_ion.py                 multi-ion shared-θ steric closure (≥2 counterions)
    picard_ic.py                 scalar Picard for the debye_boltzmann IC
    anchor_continuation.py       *** orchestrator: solve_anchor_with_continuation + solve_grid_with_anchor + AdaptiveLadder
    grid_per_voltage.py          legacy C+D orchestrator (single-counterion ClO₄⁻ stack)
    water_ionization.py          Phase 6α opt-in
    cation_hydrolysis.py         Phase 6β opt-in
    observables.py / rrde_observables.py   current/selectivity/ring observables
scripts/_bv_common.py            make_bv_solver_params factory + species presets + reaction bundles + constants
```

### Suggested reading order for the code

1. **`scripts/_bv_common.py`** — the vocabulary: species presets
   (`THREE_SPECIES_LOGC_BOLTZMANN`, `FOUR_SPECIES_LOGC_DYNAMIC_K2SO4`), reaction
   bundles (`PARALLEL_2E_4E_REACTIONS`), counterion presets, and the
   `make_bv_solver_params` factory that wires a run.
2. **`Forward/bv_solver/dispatch.py`** — how a `solver_params` becomes context +
   forms + IC.
3. **`Forward/bv_solver/forms_logc_muh.py`** — the actual residual (NP interior +
   electrode BV flux + Poisson + Stern Robin).
4. **`Forward/bv_solver/anchor_continuation.py`** — how a full IV curve gets solved
   robustly (anchor → grid walk, with ladders for hard parameters).
5. **A driver wired end-to-end**:
   `scripts/studies/solver_demo_slide15_no_speculative_cs.py` (and its visual call
   graph `writeups/May13th/forward_codepath_demo_slide15.pdf`).
6. **Correctness**: `tests/test_mms_logc_muh_multi_ion_stern.py` (method of
   manufactured solutions for the production stack).

## 5. How to call the production solver

Full API: `docs/solver/bv_solver_unified_api.md`. The canonical recipe — build
params with the factory, then **anchor + grid** (do NOT cold-start the grid on the
multi-ion + Stern stack; its Phase-1 cold start fails ~13/13 near V ≈ +0.55 V):

```python
from scripts._bv_common import (
    make_bv_solver_params,
    THREE_SPECIES_LOGC_BOLTZMANN,
    PARALLEL_2E_4E_REACTIONS,
    DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,   # ⚠ deck baseline is K⁺ — use the K⁺ preset for deck-aligned runs
    DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
)
from Forward.bv_solver import (
    solve_anchor_with_continuation,
    extract_preconverged_anchor,
    solve_grid_with_anchor,
)

sp = make_bv_solver_params(
    species=THREE_SPECIES_LOGC_BOLTZMANN,
    formulation="logc_muh", log_rate=True,
    bv_reactions=PARALLEL_2E_4E_REACTIONS,
    boltzmann_counterions=[
        DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,
        DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
    ],
    multi_ion_enabled=True,             # required for >=2 counterions
    stern_capacitance_f_m2=0.20,
    initializer="debye_boltzmann",
    l_eff_m=100e-6,
    enable_water_ionization=False,      # Phase 6α opt-in
)

anchor = solve_anchor_with_continuation(sp, ...)   # converge one voltage robustly
pre = extract_preconverged_anchor(anchor)
results = solve_grid_with_anchor(sp, pre, ...)     # warm-walk the V_RHE grid
```

The fastest way to *see* the solver work end-to-end is to run a driver rather than
hand-assemble the above:

```bash
python -u scripts/studies/solver_demo_slide15_no_speculative_cs.py
```

## 6. Environment & running

- **Activate the Firedrake venv from `PNPInverse/`:**
  `source ../venv-firedrake/bin/activate`. Conda envs will NOT run Firedrake
  correctly. Run everything from the `PNPInverse/` directory.
- Cache env vars (set before any run):
  `MPLCONFIGDIR=/tmp XDG_CACHE_HOME=/tmp PYOP2_CACHE_DIR=/tmp/pyop2 FIREDRAKE_TSFC_KERNEL_CACHE_DIR=/tmp/firedrake-tsfc OMP_NUM_THREADS=1`.
- Tests: `pytest -m "not slow"` (fast, pure-Python/PyTorch — no Firedrake) and
  `pytest -m slow` (Firedrake FEM). Always stream long runs (`-s -vv`, `python -u`)
  — a single solve can stall for minutes.
- Forward studies are **expensive** (minutes to hours). Check existing
  `summary.md` in `StudyResults/` before regenerating.

## 7. Things a newcomer will trip on (condensed from `CLAUDE.md`)

- **Use the Ruggiero parallel topology** (R2e E°=0.695, R4e E°=1.23) via
  `PARALLEL_2E_4E_REACTIONS`. The legacy sequential R1/R2 (0.68/1.78) was *wrong*;
  never `E_eq = 0`.
- **`exponent_clip = 100`** is the only PC-trustworthy clip setting (clip=50
  manufactures fictitious peroxide current).
- **The IC and the residual must agree about steric saturation** — a `bikerman` IC
  needs the matching `bikerman` residual, or it cold-fails.
- **Anchor-and-grid for the multi-ion + Stern stack; C+D only for the legacy
  ClO₄⁻ single-counterion stack.**
- **Deck baseline electrolyte is K⁺/SO₄²⁻**, not Cs⁺ or ClO₄⁻ — use the K⁺ preset
  for apples-to-apples deck comparisons.
- **`StudyResults/` is the working record, not a build-artifact dir** — active
  Phase 7 results live at top level; closed phases under `StudyResults/archive/`.

## 8. Where the data is

`data/EChem Reactor Modeling-Seitz-Mangan/` is the **primary experimental data**
(K₂SO₄ RRDE, pH 1–6). It is **gitignored** (~273 MB, lives on disk only). Inventory:
`docs/papers/data_folder_code_inventory.md`. The peer-reviewed physics source for
the deck is `docs/papers/Ruggiero2022_JCatal_source_paper.md`.

---

*Maintenance note: this doc was written at the 2026-06-15 repo reorg. If the phase
status or file paths drift, reconcile against `tasks/todo.md` (active plan),
`CLAUDE.md` (conventions), and `REPO_LAYOUT.md` (file map).*
