# Coworker Repo Diff Notes (PNP-Numerical-Solver)

Comparison of `tempBaenanclone/PNP-Numerical-Solver` against this repo
(`PNPInverse`). Focus: things worth porting in either direction. Date of
read: 2026-05-03.

## Their state, in one paragraph

Single-file Firedrake forward solver (`python-solvers/firedrake_solvers/pnp_solver.py`),
4-species **Li2SO4** chemistry (Li⁺, SO4²⁻, H⁺, OH⁻), CG1 mixed
`[c_1,...,c_n, φ]`, BDF1 with `dt=1e-9`, `t_end=2 µs`. Three BC modes via
CLI: Dirichlet / Butler-Volmer (single one-electron, scalar `j0`/`α`) /
Robin. Steric drift `μ_steric = ln(1 − Σ a_i c_i)` is default-on. MMS is a
separate driver for 2-species pure diffusion (no BV path). Adjoint file is
a Burgers' tutorial copy, not yet wired to PNP.

## Gaps relative to us (large)

Things their stack does not have and would need before it can solve our
inverse problem:

- Two-reaction ORR BV with per-reaction `k0, α, n_e, E_eq, stoichiometry,
  cathodic_conc_factors`.
- Log-c reformulation with `u_clamp` / `exponent_clip`.
- Log-rate BV evaluation (V19-V24 fix; `cond(F)` 2e11 → 1.8e7).
- Analytic Boltzmann counterion for inert spectator ion.
- CD + PC observables, FIM, multi-experiment screening, Tikhonov.
- Real Pyadjoint reduced-functional driving steady-state forward solves.
- MMS coverage of the BV boundary (their MMS is diffusion-only).

## Things they have that we don't (worth importing)

1. **CG2 MMS rate verification.** `run_convergence_study.py` sweeps mesh ×
   `order ∈ {1, 2}` and checks against `L2 ~ O(h^{p+1})`, `H1 ~ O(h^p)`.
   Our `scripts/verification/mms_bv_3sp_logc_boltzmann.py` and
   `mms_voltage_sweep.py` are CG1 only. Cheap to add; would catch any
   silent degradation of higher-order rates introduced by log-c / log-rate /
   Boltzmann.

2. **Combinatorial PETSc sweep harness.** `pnp_utils.generate_solver_params()`
   builds `itertools.product` of SNES × line-search × KSP/PC (~36 combos).
   We rely on named strategies (hybrid / stabilized / Gummel / robust). A
   brute-force harness is useful as a first-pass diagnostic when a new
   regime starts diverging.

3. **Runtime steric-occupancy diagnostic.** `pnp_solver.py:303-306` prints
   `μ_steric` min/max/mean and `∫ Σ a_i c_i / |Ω|` each output interval.
   Our `forms_logc.py:251-288` gates steric on `steric_active` but has no
   comparable runtime monitor. The packing argument can go singular long
   before Newton complains; a 5-line monitor would catch it.

4. **Transient driver (not pseudo-time-to-steady).** Their solver runs a
   real transient with VTK time-series + animation (`pnp_plotter.create_animations`).
   We have BDF1 in the residual (`forms_logc.py:271,293`,
   `robin_solver.py:168`, `dirichlet_solver.py:158`) but always drive to
   steady state via continuation. If we ever need double-layer charging,
   onset transients, or EIS-like response, the harness — not the residual —
   is what's missing.

5. **Documented dolfinx blocker.** `python-solvers/dolfin_issue.py` and
   `yiyao_code/pnp-dolfin-solver.py` record the concrete dolfinx attempt.
   Our README explains the firedrake choice but carries no worked
   artifact. Useful if anyone asks "why not dolfinx?".

6. **Independent chemistry case (Li2SO4).** Once a multi-electrolyte switch
   exists in our stack, running a converged Li2SO4 problem against their
   solver — or any reference — is a sanity check beyond MMS.

## Things I checked and we already have

- Steric μ: `Forward/bv_solver/forms_logc.py:251-288`.
- Robin BCs: `Forward/robin_solver.py`, `FluxCurve/`, Nondim Robin scaling.
- Transient time derivative in residual: `forms_logc.py:271,293`,
  `dirichlet_solver.py:158`, `robin_solver.py:168`.

## Suggested order of action

1. Add `order=2` to the existing MMS verification sweeps (cheap, immediate
   evidence of higher-order correctness).
2. Add a 5-line runtime monitor for `Σ a_i c_i` whenever `steric_active`
   is on.
3. Skip everything else unless a study calls for it.

## What we could share back

If we want to help the coworker reach the same problem class:

- The two-reaction BV BC abstraction (`Forward/bv_solver/...`).
- The log-c forms and `exponent_clip` / `u_clamp` conventions.
- A minimal Pyadjoint reduced-functional wired to a steady-state forward
  solve (their `pnp_adjoint.py` is currently a Burgers' tutorial stub).
- MMS that exercises the BV boundary, not just diffusion.
