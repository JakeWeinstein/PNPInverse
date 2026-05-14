# pnpbv Package Fork — Scoping Outline

**Status:** Tabled 2026-05-10. Resume by re-reading this doc + handing to `sci-planner` (or `gsd:new-project`) to produce a verified, actionable plan.

## Goal

Fork the PNP-BV forward solver currently living at `Forward/bv_solver/` in
PNPInverse into a standalone, general-purpose Python package called
**`pnpbv`**. The package supports every feature of the current solver in a
user-customizable way: variable dynamic species, variable analytic Boltzmann
counterions, variable reaction parameters (stoich, E°, alphas, n_e, k0,
log-rate), composable special-physics mechanisms, and explicit continuation
strategies. v1 is **forward-only**; the inverse layer is deferred to v2.

**v1.0's stable surface excludes water ionization (`HomogeneousReaction`)
and cation hydrolysis (`SurfaceAdsorbate`)** — these ship under
`pnpbv.experimental.*` and graduate post-v1 once Phase 6β identifiability is
resolved and a literature-calibrated parameter set is in hand. **v1.0
builds a publication-ready wheel + sdist but does NOT upload to PyPI** —
the release is delivered as GitHub Release assets; PyPI upload is one
documented command away, gated on explicit user authorization.

## Scope Decisions (all settled)

| Decision | Choice |
|---|---|
| Generality | **Medium — composable mechanisms.** 2D graded rectangle geometry; Firedrake runtime; `logc` and `logc_muh` formulations only. Broader generality (3D, axisymmetric, abstract solver backend, custom kinetics families) is explicitly out of scope for v1. |
| Repo model | **Soft fork.** `pnpbv` lives in a new repo. This repo (PNPInverse) becomes its first research consumer, importing the lib once it's ready. Original `Forward/bv_solver/` paths get deprecated post-cutover. |
| v1 cutline | **Forward only.** No adjoint / inversion infrastructure. `scripts/Inference/` and `scripts/studies/v*.py` stay in PNPInverse. |
| API surface | **Composable objects.** Typed classes: `Species`, `AnalyticCounterion`, `BVReaction`, `HomogeneousReaction`, `SurfaceAdsorbate`, `SternElectrode`, `DebyeBoltzmannIC`, `Problem`, `Solver`. No nested-dict factory. |
| Continuation | **First-class strategy objects.** `AnchorContinuation`, `ColdWithWarmFallback`, `ChargeContinuation`. Solver auto-suggests default based on problem; user can override. Internal ladders (`k0_ladder`, `kw_eff_ladder`) are knobs on the strategy. |
| Name | **`pnpbv`** (imports as `import pnpbv as pb`). |
| Special physics | **Experimental primitives + reference factories.** `pb.experimental.reactions.HomogeneousReaction(...)` / `pb.experimental.reactions.SurfaceAdsorbate(...)` as experimental primitives, plus `pb.experimental.physics.water_ionization(...)` and `pb.experimental.physics.cation_hydrolysis_with_langmuir_cap(...)` as preconfigured experimental factories with literature pKa defaults baked in. **All four live under `pnpbv.experimental.*`** because Phase 6β identifiability is still open as of 2026-05-12. |
| Stable vs experimental surface | **Two-tier API.** v1.0 stable surface = `Species`, `AnalyticCounterion`, `BVReaction`, `SternElectrode`, `DebyeBoltzmannIC`, `Problem`, `Solver`, `Solution` + multi-ion + parallel BV + all continuation strategies. **`HomogeneousReaction`/`SurfaceAdsorbate` + their factories ship under `pnpbv.experimental.*` and are excluded from v1.0 stable surface.** Experimental imports emit an `ExperimentalWarning`; API stability not guaranteed. Graduation gate post-v1: literature-calibrated parameter set + independent reviewer sign-off + Phase 6β identifiability resolved. |
| Numerical knobs | **Documented defaults + advanced override.** `exponent_clip=100`, `u_clamp=100`, `H2O2_SEED_NONDIM=1e-4`, ladder schedules, Newton tolerances live in `pnpbv.numerical` with documented physical meaning. `Solver`/`Problem` accept `numerical_overrides=...` for advanced users. Default path = current production values. |
| Logging | **Structured + human-readable, treated as API.** Every `solve_grid(...)` writes a per-run directory: `results.parquet` + `results.json` (machine-readable per-voltage rows), `run.log` (human summary), `config.yaml` (full reproducer dump). Provenance metadata (`pnpbv` version, git SHA, Firedrake version, config hash) headers every artifact. Verbosity via `pnpbv.logging.set_level(...)`. See "Output Logging Requirements". |
| Documentation | **First-class v1 deliverable, not a v1.x backlog item.** Every public class/function carries a full docstring (param types, units, defaults, valid ranges, physical meaning, literature citation where applicable). API reference auto-generates from docstrings. One tutorial per major workflow. Migration guide for the legacy `Forward/bv_solver/` API. Docstring contract is lint-enforced; doctests run in CI. See "Documentation Requirements". |

## Composable Abstractions

Five user-facing mechanism classes cover everything the current solver does:

1. **`Species`** — dynamic concentration solved by the PDE. Carries `z`, `D`,
   `c_bulk`, optional `seed` (for `ln c_i` finite-seeding), optional `role`
   tag (`"proton"` for IC/water-ionization wiring) so the IC can introspect
   the problem instead of hardcoding species names.
2. **`AnalyticCounterion`** — Boltzmann-distributed counterion. Carries
   `z`, `c_bulk`, and a `steric` argument: `pb.Bikerman(a=...)` or
   `pb.IdealSteric()`. Multiple instances → multi-ion shared-θ closure.
3. **`BVReaction`** — Butler-Volmer surface reaction with explicit
   reactants/products dicts (stoich), `n_e`, `E_eq`, `alpha_a`/`alpha_c`,
   `k0`, `log_rate` flag. Generalizes both the legacy sequential R₁/R₂
   pair and the current parallel R2e/R4e pair.
4. **`HomogeneousReaction`** *(experimental — `pnpbv.experimental.reactions`; excluded from v1.0 stable surface)* — bulk-distributed reaction (source/sink in
   the residual). Carries forward/reverse stoich dicts, `K_eq`,
   `kf_kw_factor`. The current `enable_water_ionization=True` becomes one
   of these.
5. **`SurfaceAdsorbate`** *(experimental — `pnpbv.experimental.reactions`; excluded from v1.0 stable surface)* — surface coverage Γ with Langmuir saturation
   cap. Carries forward/reverse stoich (referencing both `Species` and
   `AnalyticCounterion` for cation hydrolysis), `pKa0`, `dpKa_dE`,
   `gamma_max`. The current `cation_hydrolysis.py` becomes one of these.

Composition root: `pb.Problem(species=[...], counterions=[...],
reactions=[...], homogeneous=[...], adsorbates=[...], electrode=...,
formulation=..., initializer=...)`.

## API Sketch

```python
import pnpbv as pb
import numpy as np

# --- Species ---
H    = pb.Species("H+",   z=+1, D=9.31e-9, c_bulk=1e-4, role="proton")
O2   = pb.Species("O2",   z=0,  D=2.0e-9,  c_bulk=1.0)
H2O2 = pb.Species("H2O2", z=0,  D=1.0e-9,  c_bulk=0.0, seed=1e-4)

# --- Analytic Boltzmann counterions ---
Cs  = pb.AnalyticCounterion("Cs+",   z=+1, c_bulk=0.2, steric=pb.Bikerman(a=3e-10))
SO4 = pb.AnalyticCounterion("SO4--", z=-2, c_bulk=0.1, steric=pb.Bikerman(a=3e-10))

# --- BV reactions ---
r2e = pb.BVReaction("ORR_2e",
        reactants={O2:1, H:2}, products={H2O2:1},
        n_e=2, E_eq=0.695, alpha_a=0.5, alpha_c=0.5, k0=K0_R2E, log_rate=True)
r4e = pb.BVReaction("ORR_4e",
        reactants={O2:1, H:4}, products={},
        n_e=4, E_eq=1.23,  alpha_a=0.5, alpha_c=0.5, k0=K0_R4E, log_rate=True)

# --- Optional: homogeneous reaction (water self-ionization) — EXPERIMENTAL ---
import pnpbv.experimental as pbe
pbe.acknowledge_experimental()  # silence ExperimentalWarning for batch runs
water_ion = pbe.physics.water_ionization(K_w=1e-14)  # experimental reference factory

# --- Optional: surface adsorbate (cation hydrolysis with Langmuir cap) — EXPERIMENTAL ---
moh = pbe.physics.cation_hydrolysis_with_langmuir_cap(
    cation=Cs, proton=H, pKa0=4.3, gamma_max=5.6e-6,
)

# --- Compose problem ---
prob = pb.Problem(
    species=[H, O2, H2O2],
    counterions=[Cs, SO4],
    reactions=[r2e, r4e],
    homogeneous=[water_ion],     # optional
    adsorbates=[moh],            # optional
    electrode=pb.SternElectrode(C_stern=0.10, l_eff=100e-6),
    formulation="logc_muh",
    initializer=pb.DebyeBoltzmannIC(),
)

# --- Solver + continuation strategy ---
solver = pb.Solver(prob, mesh=pb.graded_rectangle(Nx=8, Ny=80, beta=3.0))
sol = solver.solve_grid(
    v_rhe=np.linspace(-0.5, 1.0, 16),
    continuation=pb.AnchorContinuation(
        v_anchor=0.55,
        k0_ladder=(1e-12, 1e-9, 1e-6, 1e-3, 1.0),
        kw_eff_ladder=None,  # auto-enabled when HomogeneousReaction present
    ),
)

# --- Rich diagnostics ---
sol.current_density               # total j(V_RHE)
sol.reaction_currents["ORR_2e"]   # per-reaction currents
sol.surface_coverages["MOH"]      # surface adsorbate Γ
sol.surface_pH
sol.diagnostics                   # F0, Γ, θ, R_forward_capped, σ_S, etc.
```

## Module Layout

```
pnpbv/
├── __init__.py            — Public API re-exports
├── species.py             — Species, AnalyticCounterion, Bikerman, IdealSteric
├── reactions/             — Stable: BVReaction only
│   ├── __init__.py
│   └── bv.py              — BVReaction
├── experimental/          — Excluded from v1.0 stable surface; ExperimentalWarning on import
│   ├── __init__.py        — acknowledge_experimental(), silence_warnings()
│   ├── reactions/
│   │   ├── __init__.py
│   │   ├── homogeneous.py — HomogeneousReaction (v0.4)
│   │   └── adsorbate.py   — SurfaceAdsorbate (v0.5)
│   └── physics/           — Experimental reference factories
│       ├── __init__.py
│       ├── water_ionization.py     — water_ionization() (v0.4)
│       └── cation_hydrolysis.py    — cation_hydrolysis_with_langmuir_cap() (v0.5)
├── electrode.py           — SternElectrode (BC abstraction layer for future variants)
├── initializers/
│   ├── __init__.py
│   └── debye_boltzmann.py — DebyeBoltzmannIC
├── formulations/
│   ├── __init__.py
│   ├── logc.py            — Current forms_logc.py, ported
│   └── logc_muh.py        — Current forms_logc_muh.py, ported
├── problem.py             — Problem (composition root, role-based introspection)
├── mesh.py                — graded_rectangle, structured_rectangle
├── continuation/
│   ├── __init__.py
│   ├── anchor.py          — AnchorContinuation
│   ├── cold_warm.py       — ColdWithWarmFallback
│   ├── charge.py          — ChargeContinuation
│   └── ladders.py         — k0_ladder, kw_eff_ladder utilities
├── solver.py              — Solver
├── solution.py            — Solution, SolutionDiagnostics
├── nondim.py              — V_T, L_REF, C_SCALE helpers
├── numerical.py           — Clips, clamps, Newton tolerances, validate_solution_state
├── logging.py             — Structured run logging, results-table writer, provenance metadata
└── _internal/             — Form builders, Bikerman closure, multi-ion θ closure, Stern, Newton wrapper
```

Each public class is a thin, immutable record. `_internal/` holds the
Firedrake/UFL form-building code (ported with minimal math change).

## Output Logging Requirements

Forward runs are unattended for hours over dozens of voltages × continuation
strategies. Logging is part of the API, not incidental stdout chatter — both
PNPInverse (the research consumer) and external users will read these
artifacts long after the run finishes, often to debug a discrepancy at a
single voltage. Treat the schema as a public contract.

### Per-run artifacts

Every `solver.solve_grid(...)` writes a run directory containing:

- **`results.parquet`** + **`results.json`** — Per-voltage row table:
  `V_RHE`, `converged`, `newton_iters`, `final_residual`, `j_total`,
  per-reaction `j_*`, per-adsorbate Γ, surface pH, σ_S, F0, every
  `SolutionDiagnostics` field. Schema versioned in the parquet metadata.
- **`run.log`** — Human-readable. Per-voltage outcome, continuation
  ladder trajectory (rungs hit, fallback triggers), clip activations,
  IC↔residual consistency checks, timings.
- **`config.yaml`** — Full `Problem` + `Solver` config dump (species,
  counterions, reactions, electrode, formulation, initializer,
  continuation strategy, numerical overrides). Sufficient to reproduce
  the run on a different machine.
- **Provenance metadata** in every artifact: `pnpbv` version, git SHA (if
  available), Firedrake version, config hash, timestamp.
- **Optional `checkpoint.h5`** — Firedrake checkpoint for warm-restart or
  post-processing.

### What must be captured

- **Per-voltage**: Newton iterations, final nonlinear residual, wall
  time, converged/failed flag, full fallback chain (which strategy was
  tried, in what order, with what outcome).
- **Per-continuation step**: ladder rung, anchor cache hits, `k0_ladder`
  / `kw_eff_ladder` / `lambda_hydrolysis` schedules.
- **Clip activations**: count of cells where `exponent_clip` / `u_clamp`
  triggered, max pre-clip magnitude — so users know when they are in a
  regime where clipping affects physics (CLAUDE.md Hard Rule #2).
- **IC↔residual consistency**: warn at startup if IC steric mode and
  residual steric mode disagree (CLAUDE.md Hard Rule #3).
- **Validation**: every solve runs `validate_solution_state` and logs
  charge imbalance + max-|c| diagnostics.
- **Bikerman `a` provenance**: log which species use physical
  Marcus/Stokes radii vs the placeholder `A_DEFAULT = 0.01` (CLAUDE.md
  Hard Rule #7), so downstream readers can spot the discrepancy without
  re-deriving it from source.

### Verbosity levels

| Level | Captures |
|---|---|
| `QUIET` | Errors and fatal warnings only |
| `INFO` (default) | Per-voltage outcomes, continuation events, clip warnings, validation summary |
| `DEBUG` | Every Newton step, every ladder-rung evaluation, residual history |
| `TRACE` | Form-assembly events, expression rebuilds (rarely needed) |

Global default via `pnpbv.logging.set_level(...)`; per-run override via
`solver.solve_grid(..., log_level="DEBUG")`.

### Cross-milestone constraint

The results-table schema lands by **v0.1** (needed for byte-equivalence
checks against legacy `Forward/bv_solver/` runs), extends additively
through v0.2–v0.5 as new mechanisms come online (per-reaction currents in
v0.3, kw-ladder telemetry in v0.4, adsorbate Γ in v0.5), and is **frozen
at v0.6 under semver** — any incompatible schema change after that
requires a major version bump, because PNPInverse and other consumers
will be reading these tables programmatically.

## Documentation Requirements

The library exists so that researchers other than the original author can
compose PNP-BV problems without reading Firedrake source. That only works
if docs are a first-class deliverable from day one, not a v1.x backlog
item bolted on after the API is set. **Treat undocumented public symbols
as broken.**

### Docstring contract (enforced by lint in CI)

Every public class, function, and parameter carries:

- **Type** (Python hint; units in parens for physical quantities, e.g.
  `D: float  # m²/s — diffusivity`)
- **Default**, where applicable
- **Valid range** or physical/numerical bounds, where applicable
- **Physical meaning** in one sentence
- **Literature citation** for reference factories and literature-anchored
  defaults (e.g. `C_stern = 0.20 F/m²` cites Bohra-Koper-Choi per
  `docs/phase6/CMK3_capacitance_literature.md`; `pKa0 = 4.3` cites
  Linsey 2025 deck slide 9)
- **Cross-references** to related symbols (`See Also:` block)

### Numerical-knob provenance

Every default in `pnpbv.numerical` carries a comment explaining *why*
the value, citing Phase 6 lessons where applicable. Example:

```python
EXPONENT_CLIP = 100  # Clips eta_scaled = (V_RHE - E_eq)/V_T pre α·n_e mult.
                     # At 100, R2 unclips at V_RHE > -0.79V on production grid.
                     # 50 (historical) produces fictitious peroxide current.
                     # See docs/numerics/clipping_conventions.md.
```

### Library-level deliverables (must land by v0.6)

- **`README.md`** — Overview, install (with Firedrake prereq note),
  20-line hello-world, pointers to tutorials.
- **API reference** — Sphinx + autodoc, auto-generated from docstrings,
  published to GitHub Pages on every tagged release.
- **Tutorials** (Jupyter notebooks, also CI-tested as scripts):
  1. Single counterion + sequential reactions (v0.1/v0.2 reference)
  2. Multi-ion + parallel 2e/4e (v0.3 reference)
  3. Adding water ionization via `pb.physics.water_ionization()` (v0.4)
  4. Adding cation hydrolysis via
     `pb.physics.cation_hydrolysis_with_langmuir_cap()` (v0.5)
  5. Writing a custom `HomogeneousReaction` from scratch
  6. Choosing and tuning a continuation strategy
- **Migration guide** — Side-by-side old → new API for every PNPInverse
  reference case (`peroxide_window_3sp_bikerman_muh.py`,
  `mangan_full_grid_csplus_so4.py`,
  `l_eff_transport_sweep_csplus_so4.py`,
  `phase6b_v9_gate4_finite_hydrolysis_smoke.py`).
- **Physics & numerics guide** — Codifies the CLAUDE.md Hard Rules into
  library docs: nondim conventions, IC↔residual coupling, clipping
  conventions, C+D vs anchor continuation choice, validation-state
  machinery, Bikerman `a_nondim` physicality. This is the doc a new
  contributor reads to avoid the same time-sinks Phase 6 already paid
  for.
- **`CHANGELOG.md`** — Per-version, breaking changes flagged with
  before/after snippets.

### Doctest discipline

Every code example in the docs (README, tutorials, docstrings) is a
runnable doctest, run in CI on every PR using the upstream Firedrake
Docker image. No example rots silently between releases.

## What Survives vs. What Gets Refactored

### Ports nearly as-is (numerical math unchanged)

- `Forward/bv_solver/forms_logc.py` → `pnpbv/formulations/logc.py` (+ `_internal/forms.py`)
- `Forward/bv_solver/forms_logc_muh.py` → `pnpbv/formulations/logc_muh.py` (+ `_internal/forms.py`)
- `Forward/bv_solver/anchor_continuation.py` → `pnpbv/continuation/anchor.py`
- C+D cold-with-warm-fallback dispatcher → `pnpbv/continuation/cold_warm.py`
- Charge continuation → `pnpbv/continuation/charge.py`
- Multi-ion shared-θ closure → `pnpbv/_internal/multi_ion.py`
- Stern compact-layer machinery → `pnpbv/_internal/stern.py`
- `debye_boltzmann` IC machinery → `pnpbv/initializers/debye_boltzmann.py`
- Bikerman analytic counterion residual closure → `pnpbv/_internal/bikerman.py`
- MMS verification suite (`scripts/verification/`) → `pnpbv/tests/mms/`
- `Forward.bv_solver` test suite → `pnpbv/tests/unit/` + `pnpbv/tests/integration/`

### Gets refactored / rewritten

- `make_bv_solver_params` factory → composable objects via `pb.Problem(...)`
- Named constants `THREE_SPECIES_LOGC_BOLTZMANN`, `PARALLEL_2E_4E_REACTIONS`,
  `DEFAULT_*_COUNTERION_STERIC` → user-constructed `Species` / `BVReaction` /
  `AnalyticCounterion` instances (the defaults move to docs/examples as
  reference recipes, not into the library namespace)
- `enable_water_ionization=True` flag + manual `kw_eff_ladder` outer loop →
  `HomogeneousReaction` mechanism (in `pnpbv.experimental.reactions`) +
  `pb.experimental.physics.water_ionization()` factory +
  `AnchorContinuation` auto-detects and enables the kw ladder.
  **Experimental — excluded from v1.0 stable surface.**
- `Forward/bv_solver/cation_hydrolysis.py` (named feature) → `SurfaceAdsorbate`
  mechanism (in `pnpbv.experimental.reactions`) +
  `pb.experimental.physics.cation_hydrolysis_with_langmuir_cap()` factory.
  **Experimental — excluded from v1.0 stable surface.**
- Hardcoded species-name lookups in IC ("H", "O2", "H2O2") → role-based
  introspection (`problem.species_by_role("proton")` etc.)
- Positional reaction binding (`k0_targets={0: K0_R2E, 1: K0_R4E}`) → object-
  or name-keyed (`k0_targets={r2e: K0_R2E, r4e: K0_R4E}`)
- `scripts/_bv_common.py` helper module → its useful pieces (named defaults,
  BV reaction definitions) become reference examples in docs, not library API

### Stays out of v1 entirely

- Inverse layer (`scripts/Inference/`, `scripts/studies/v*.py`) — paused/broken
- Mangan-specific reference scripts (`scripts/studies/l_eff_transport_sweep_csplus_so4.py`,
  `mangan_full_grid_csplus_so4.py`, `peroxide_window_3sp_bikerman_muh.py`) —
  these stay in PNPInverse as the lib's research consumer
- `data/EChem Reactor Modeling-Seitz-Mangan/` — stays in PNPInverse
- `docs/phase*/`, `docs/papers/`, `docs/ic_studies/`, `docs/realignment/`, etc. —
  stay in PNPInverse (research narrative, not library docs)
- Concentration backend — already removed May 2026; not resurrected

## Milestone Breakdown (sequential, each milestone = working artifact)

| Tag | Scope | Estimate |
|---|---|---|
| **v0.1** | Repo bootstrap (`pyproject.toml`, CI, README, license) + form-builder port + raw-function continuation. Reproduce one legacy ClO₄⁻ + sequential R₁/R₂ + Stern run, byte-equivalent (or within numerical noise). | 1–2 wk |
| **v0.2** | Composable objects land: `Species`, `AnalyticCounterion`, `BVReaction`, `SternElectrode`, `DebyeBoltzmannIC`, `Problem`. Migrate v0.1 reference case to new API. Role-based introspection. | 2–3 wk |
| **v0.3** | Multi-ion shared-θ closure exposed via multiple `AnalyticCounterion` + parallel-2e/4e `BVReaction` demo + first-class continuation strategies (`AnchorContinuation`, `ColdWithWarmFallback`, `ChargeContinuation`). Cs⁺/SO₄²⁻ + parallel-2e/4e reference case lands. | 2 wk |
| **v0.4** *(experimental)* | `HomogeneousReaction` (in `pnpbv.experimental.reactions`) + `pb.experimental.physics.water_ionization()` reference factory + `kw_eff_ladder` strategy hook. Reproduces Phase 6α 8/8 sweep. **Excluded from v1.0 stable surface.** | 2 wk |
| **v0.5** *(experimental)* | `SurfaceAdsorbate` (in `pnpbv.experimental.reactions`) w/ Langmuir cap + `pb.experimental.physics.cation_hydrolysis_with_langmuir_cap()` reference factory. Reproduces Phase 6β v10a smoke. **Excluded from v1.0 stable surface.** | 2–3 wk |
| **v0.6** | Documentation pass per "Documentation Requirements": Sphinx autodoc API reference, six tutorial notebooks, migration guide, physics & numerics guide, doctest-in-CI. Results-table logging schema locked under semver. | 1–2 wk |
| **v0.7** | PNPInverse cutover: `Forward/bv_solver/` imports replaced with `import pnpbv as pb`. All Mangan-specific scripts adapted. Tests pass on both sides. | 1 wk |
| **v1.0** | Wheel + sdist built; `twine check` validates PyPI metadata; tagged GitHub release with wheel + sdist as downloadable assets. **No PyPI upload in v1.0** — publish command documented in `docs/release/PYPI_PUBLISH_COMMAND.md` ready to execute on user authorization. Optional: Docker/conda recipes. | 1 wk |

**Total:** ~12–15 wk sequential, ~6–8 wk if v0.4 and v0.5 run in parallel
after v0.3 and v0.6 overlaps with v0.5.

## Reproduction Targets per Milestone (acceptance gates)

These are the legacy runs each milestone must reproduce before merging:

| Milestone | Reproduction target |
|---|---|
| v0.1 | `scripts/studies/peroxide_window_3sp_bikerman_muh.py` — ClO₄⁻ single counterion, sequential R₁/R₂, Stern, log-c muh |
| v0.2 | Same as v0.1, via the new composable-objects API |
| v0.3 | `scripts/studies/mangan_full_grid_csplus_so4.py` — Cs⁺/SO₄²⁻ multi-ion + parallel 2e/4e |
| v0.4 | Phase 6α 8/8 sweep result from `scripts/studies/l_eff_transport_sweep_csplus_so4.py --enable-water-ionization` |
| v0.5 | Phase 6β v10a smoke: `scripts/studies/phase6b_v9_gate4_finite_hydrolysis_smoke.py` (post-v10a Langmuir cap) |
| v0.7 | All of the above runs reproduced from PNPInverse's adapter scripts importing `pnpbv` |

## Key Risks / Refactor Concerns

### 1. IC ↔ residual ↔ continuation coupling (v0.2 hot spot)

The current `set_initial_conditions_debye_boltzmann_*` family hardcodes
species-name lookups ("H", "O2"). The CLAUDE.md is emphatic: *"The IC and the
residual must agree about steric saturation."* Generalizing to role-based
introspection means making sure the IC and the residual receive consistent
species/counterion declarations across all sterics × multi-ion × initializer
combos. Reserve buffer time in v0.2; ship with regression tests covering at
least: single-counterion ClO₄⁻ + ideal/bikerman; multi-ion Cs⁺/SO₄²⁻ +
bikerman; with and without role-tagged proton.

### 2. Adjoint-friendliness as a non-functional requirement (all milestones)

Inverse work resumes in v2. The forward solver needs `pyadjoint`-friendly
seams: no in-place mutation in user-facing code, no untaped Firedrake
operations in the hot path, immutable problem records. Costs almost nothing
now; saves a refactor in v2. `pyadjoint` itself stays out of v1 imports.

### 3. Positional → object-keyed reaction binding (v0.3)

Current `k0_targets={0: K0_R2E, 1: K0_R4E}` is positional. Migrating to
`k0_targets={r2e: K0_R2E, r4e: K0_R4E}` (where `r2e`, `r4e` are
`BVReaction` instances) means continuation code needs to look up reactions
by identity, not index. Plan should call out the bookkeeping.

### 4. Convergence-strategy auto-default heuristic (v0.3)

Solver picks a default continuation strategy based on problem features:
multi-ion + Stern → `AnchorContinuation`; single-counterion + Stern → C+D;
etc. The heuristic itself needs unit tests with synthetic problems.

### 5. Reference-factory parameter provenance (v0.4, v0.5 — experimental)

`pb.experimental.physics.water_ionization()` and
`pb.experimental.physics.cation_hydrolysis_with_langmuir_cap()` ship with
literature-defended defaults (K_w=1e-14; Cs⁺ pKa₀ ≈ 4.3 per Linsey deck;
gamma_max from `docs/phase6/CMK3_capacitance_literature.md` once that lands).
The docstrings must cite sources so users know what they're getting. The
current `gamma_max_nondim = 0.047` is a smoke baseline — v0.5 ships with it
under the experimental flag, and the literature-calibrated value becomes a
**graduation gate** for moving these factories from `pnpbv.experimental.*`
to the v1.x stable surface (post-v1.0).

## Firedrake Dependency

`pyproject.toml` will declare `firedrake>=...` as a documented system
prerequisite, **not** a hard pip dep (since `pip install firedrake` doesn't
work — users run `firedrake-install` or use the firedrake-pip channel). The
README points at upstream install instructions. CI uses the upstream
Firedrake Docker image. Optional install extras for dev/test/docs.

## How to Resume

1. Re-read this doc.
2. Sanity-check that nothing has shifted in PNPInverse that invalidates the
   reproduction targets (especially the post-v10a Phase 6β state).
3. Hand to `sci-planner` with this doc as the task prompt, asking for a
   verified, actionable plan with per-milestone task breakdown, dependencies,
   risks, and verification gates. Or use `gsd:new-project` if you want the
   full GSD project-init treatment.
4. Pick a starting milestone (v0.1 is the obvious entry).
5. Bootstrap the new repo and begin.
