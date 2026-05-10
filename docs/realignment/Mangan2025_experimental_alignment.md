# Mangan 2025 Catalysis Deck: Experimental Alignment Notes

Date: 2026-05-04

## Purpose

This note summarizes how the current PNP-BV solver differs from the
experimental and modeling setup in `docs/Mangan2025_Catalysis.pdf`, and what
would need to change to make the solver more directly represent that setup.

The deck describes pH and cation effects in oxygen reduction reaction (ORR)
experiments for peroxide production on mesoporous carbon black CMK-3 using a
rotating ring-disk electrode (RRDE) and iridium oxide (IrOx) local pH sensing.
The modeling slides describe a steady electrocatalytic reactor model with a
diffusive outer region, a nanometer-scale screening/double-layer region,
modified Poisson-Boltzmann physics, steric effects, and a Butler-Volmer/Tafel
electrode boundary law.

The current solver is aimed at the same broad physical problem, but it is not
yet a direct model of the deck's experiment.

## Current Solver Match

The current production solver is already aligned with several core pieces of
the deck:

- It models ORR to peroxide at pH 4.
- It uses a two-step reaction model:
  - R1: `O2 + 2H+ + 2e- -> H2O2`
  - R2: `H2O2 + 2H+ + 2e- -> 2H2O`
- It uses PNP-style charged transport coupled to BV electrode kinetics.
- It computes current-density and peroxide-current observables from the
  reaction rates.
- It has a diffusion-layer length scale, currently `L_REF = 100 um`, which is
  conceptually close to the deck's varying diffusive-region length.
- It includes steric/Bikerman machinery and an analytic inert counterion path.
- It has optional finite-Stern-layer support, which is closer to the deck's
  compact/diffuse-layer schematic than the default no-Stern boundary condition.

Relevant implementation anchors:

- `README.md`
- `scripts/_bv_common.py`
- `Forward/bv_solver/forms_logc.py`
- `Forward/bv_solver/boltzmann.py`
- `Forward/bv_solver/grid_per_voltage.py`

## Main Differences

### Experimental observables

The deck's experiment is not just a disk-current experiment. It uses:

- RRDE disk current.
- RRDE ring signal for peroxide.
- IrOx local pH sensing.
- Local-vs-bulk pH comparisons.
- Selectivity trends.

The current solver mainly produces:

- total current density, assembled as a sum of BV reaction rates;
- peroxide current, assembled as `R1 - R2`;
- inverse-problem synthetic observables derived from those currents.

It does not yet model an RRDE ring collection efficiency, an IrOx sensor
calibration, or local-pH data as a first-class observable.

### Electrolyte chemistry

The deck and the cited Ruggiero et al. work focus on alkali cations
`Li+`, `Na+`, `K+`, and `Cs+`, especially pH 4 `Cs+` behavior.

The current production solver uses three explicit transported species:

- `O2`
- `H2O2`
- `H+`

and replaces the inert anion with an analytic `ClO4-` Boltzmann counterion.
That is a useful numerical and physical reduction for a generic inert
supporting electrolyte, but it is not the same electrolyte identity as the
experiment in the deck.

### Balancing anion

Including a balancing anion is experimentally and physically backed. It is not
just a numerical device: any electrolyte must satisfy electroneutrality in the
bulk, and charged transport requires the corresponding countercharge.

However, the specific anion matters.

The experimental electrolyte source is sulfate-family, not perchlorate-family.
The accepted manuscript source for the Ruggiero et al. study reports
electrolytes prepared from `H2SO4`, `MOH`, and `M2SO4`, with
`M+ = Li+, Na+, K+, Cs+`.

Practical implication:

- Keep an anion in the model.
- Do not present `ClO4-` as the experimentally matched anion for the deck.
- Treat `ClO4-` as an inert supporting-electrolyte surrogate unless the
  experiment being simulated actually used perchlorate.
- For a closer deck match, model sulfate/bisulfate either explicitly or through
  an effective inert anion approximation.

Useful source:

- OSTI accepted manuscript page for Ruggiero et al.:
  `https://www.osti.gov/servlets/purl/2418971`

### Cation effects and sterics

The deck's cation story is explicitly ion-specific. Larger effective ion
radii shift the peroxide-current curve and reduce current via steric exclusion.

The current solver has a generic steric size parameter, currently represented
by defaults such as `A_DEFAULT = 0.01`, but does not yet use experimentally
specific cation radii or hydration-shell parameters. A deck-aligned model
should make steric parameters depend on the chosen cation.

### Mass transport and RRDE hydrodynamics

The deck varies the diffusive-region length and relates that to
transport-limited current. The current solver has a reference length scale
`L_REF`, and planned multi-experiment work already considers `L_ref`/rotation
variation.

For a closer RRDE model, `L_REF` should be tied to rotation rate through a
Levich-style transport relation or through a calibrated effective diffusion
layer thickness. Otherwise it remains a useful but abstract transport knob.

### Numerical strategy

The deck describes a reduced one-dimensional strategy:

- solve the outer diffusive regime explicitly;
- integrate inward from the bulk;
- switch from PDE to ODE;
- use nonlinear spatial mapping to resolve nanometer-scale screening;
- use spectral methods for spatial resolution.

The current solver is different:

- Firedrake finite elements;
- coupled log-concentration PNP weak form;
- graded interval/rectangle meshes;
- pseudo-time stepping to steady state;
- Newton/SNES nonlinear solves;
- per-voltage z-ramp and warm-walk continuation.

This is not necessarily wrong, but it is not the same solver setup described
in the deck.

## Recommended Shift Toward the Deck Setup

### Stage 1: Add experiment-mode configuration

Add a high-level experiment config that names the physical condition instead
of only the solver numerics:

```text
experiment:
  catalyst: CMK-3
  geometry: RRDE
  pH_bulk: 4
  cation: Cs+
  anion_model: sulfate_effective
  rotation_rate_rpm: ...
  L_eff_m: ...
  observables:
    - disk_current
    - ring_peroxide_current
    - peroxide_selectivity
    - local_pH
```

This config should drive solver parameter construction rather than requiring
each study script to manually assemble species, boundary markers, and scales.

### Stage 2: Add local pH observable

Do not initially model IrOx as a separate electrochemical species. First add
the pH observable implied by the IrOx measurement:

```text
pH_local = -log10(c_H_surface / 1000)
```

where `c_H_surface` is in `mol/m^3`. If using nondimensional concentration,
convert back through the concentration scale before computing pH.

Then compare this local pH observable against IrOx-inferred data.

### Stage 3: Replace the anion surrogate for deck-matched runs

For a pH 4 `Cs+` deck condition, use one of these levels:

1. Minimal: an effective inert sulfate-family anion with charge and bulk
   concentration chosen to enforce electroneutrality.
2. Better: dynamic `Cs+` plus effective sulfate-family anion.
3. Best: acid-base speciation with `HSO4- <-> H+ + SO4^2-`, plus `Cs+`.

The minimal route keeps the solver tractable while avoiding the misleading
claim that `ClO4-` is the experimental anion.

### Stage 4: Make cation sterics explicit

Promote steric size from a generic solver constant to an experiment parameter:

```text
cation = Cs+
a_Cs = f(effective_radius_Cs)
a_anion = f(effective_radius_anion)
a_H = f(effective_radius_H)
```

Then reproduce the deck-style radius sweep by varying the cation steric
parameter while holding bulk pH and transport fixed.

### Stage 5: Tie `L_REF` to RRDE transport

The deck's length variation should be represented as an experimental
transport condition rather than only a nondimensionalization scale.

Short-term route:

- fit or prescribe `L_eff` values around the deck's range, e.g. `66-86 um`;
- report diffusion-limited current sensitivity to `L_eff`.

Longer-term route:

- compute `L_eff` from RRDE rotation rate and fluid properties;
- use the same mapping for disk and ring-current interpretation.

### Stage 6: Add finite Stern as a physical parameter

The deck schematic separates compact/Stern and diffuse layers. For
deck-aligned studies, no-Stern should be treated as the infinite-capacitance
limit, not as the only physical model.

Use `stern_capacitance_f_m2` as either:

- a fixed plausible compact-layer capacitance; or
- a calibrated nuisance parameter with bounds.

### Stage 7: Only then consider a reduced 1D ODE/spectral solver

The current Firedrake solver can still be a valid way to solve similar
physics. Replacing it with the deck's PDE-to-ODE/spectral strategy should be
deferred until the experimental closure is cleaner.

The highest-value first changes are observables, electrolyte identity, cation
sterics, and RRDE transport. The numerical method is a second-order mismatch
until those physical links are corrected.

## Suggested Development Order

1. Add a local pH observable from surface `H+`.
2. Add experiment metadata and a pH 4 `Cs+` config path.
3. Replace `ClO4-` with a sulfate-family effective anion for deck-matched runs.
4. Add dynamic `Cs+` or a cation-specific analytic closure.
5. Make steric parameters cation-specific.
6. Tie `L_REF` to an RRDE effective diffusion length or rotation-rate model.
7. Add ring-current/selectivity post-processing.
8. Revisit finite Stern calibration.
9. Consider the deck's reduced ODE/spectral solver only if the FE solver
   remains too stiff or too expensive after the physical model is aligned.

## Bottom Line

The current solver is close to the deck at the level of broad ORR/PBNP/BV
physics, but not at the level of experimental closure. The most important
shift is not to change Newton or Firedrake first. It is to make the model
represent the actual measured experiment:

- pH 4 `Cs+` electrolyte;
- sulfate-family balancing anion;
- cation-specific sterics;
- RRDE transport and ring/disk observables;
- local pH observable corresponding to IrOx sensing;
- finite Stern layer as an explicit physical option.

Only after those changes are in place will it be meaningful to decide whether
the current finite-element numerical strategy needs to be replaced by the
deck's reduced 1D inward-integration method.
