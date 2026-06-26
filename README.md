# PNPInverse

**A generalizable finite-element solver for the Poisson–Nernst–Planck equations with Butler–Volmer boundary conditions.**

PNPInverse simulates strongly-coupled, nonlinear electrochemical systems from first principles — ion transport, electrostatics, and reaction kinetics solved together as one coupled PDE system — to help uncover the unknown reaction mechanisms behind processes such as the oxygen reduction reaction (ORR).

## Why this exists

Electrochemistry at an electrode is governed by the **Poisson–Nernst–Planck (PNP)** equations for coupled ion transport and electrostatics, closed by **Butler–Volmer (BV)** reaction kinetics at the surface. This system is:

- **strongly coupled** — ion concentrations, electric potential, and reaction rates all feed back on one another;
- **stiff and nonlinear** — thin interfacial layers and exponential reaction kinetics span many orders of magnitude;
- **hard to solve robustly** — naive nonlinear solves diverge across much of the physically interesting operating range.

There was no general, robust solver assembled for this coupled PNP–BV system. PNPInverse fills that gap: a solver framework built to handle this class of problems **robustly and generally**, so the model can be used as a scientific instrument — fit to experimental data, used to test competing mechanistic hypotheses, and pushed to reveal what physics is actually required to explain what is measured.

## What it does

- **Forward simulation** — a finite-element solver (built on [Firedrake](https://www.firedrakeproject.org/)) for the coupled PNP–BV system, with multi-species ion transport, electrostatic coupling, and configurable surface kinetics.
- **Robust nonlinear solves** — continuation/homotopy and warm-start strategies carry the solver through regimes where direct Newton iterations fail.
- **Generalizable by design** — built to extend across electrolytes, ion species, and reaction topologies rather than being hard-wired to a single experiment.
- **Mechanism discovery** — gradient-based calibration and identifiability analysis to infer kinetic parameters from data and separate what is identifiable from what is not.
- **Verified** — convergence validated against analytic benchmarks via the method of manufactured solutions, guarded by an extensive regression test suite.

## Project status

Active research code developed in **Prof. Niall Mangan's group at Northwestern University**. The forward solver is the primary, actively-developed surface; the inverse and surrogate components are part of the same framework and continue to be matured.

## Documentation

This README is a high-level overview of the project's purpose. The full technical reference — modeling choices, solver internals, calibration conventions, phase-by-phase research notes, and detailed setup — is archived in [`docs/old_README.md`](docs/old_README.md), alongside the rest of [`docs/`](docs/).

## Setup

The solver depends on [Firedrake](https://www.firedrakeproject.org/), which is installed through its own installer (it is not pip-installable from PyPI). With the Firedrake virtual environment active, install this package on top:

```bash
pip install -e ".[dev]"
```

See [`docs/old_README.md`](docs/old_README.md#setup) for the complete, step-by-step environment setup and the full list of commands.
