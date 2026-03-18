# Physics Model Change Log — Anodic Species Concentration

**Date:** 2026-03-18
**Decision:** Implement full concentration-dependent anodic BV term

## Background

The Butler-Volmer anodic (reverse) term in `Forward/bv_solver/forms.py:296-302` uses a
**constant** `c_ref_j` instead of the dynamic surface concentration `c_surf[anodic_species]`.

The `anodic_species` config field is parsed in `config.py:135` and stored per reaction, but
**never read** during weak form assembly. This makes it dead code.

## Standard BV kinetics

For Rxn 1: O2 + 2H+ + 2e- -> H2O2

```
R = k0 * [c_O2_surf * exp(-alpha*n*eta) - c_H2O2_surf * exp((1-alpha)*n*eta)]
                                            ^^^^^^^^^^^^
                                            Should be dynamic, was constant c_ref
```

## Impact assessment

- **Large overpotentials (transport-limited regime):** Negligible — cathodic exponential dominates
  by orders of magnitude. Existing I-V curve fits in this regime are unaffected.
- **Near equilibrium (eta ~ 0):** Moderate — exchange current density and open-circuit potential
  affected because anodic term doesn't respond to product (H2O2) accumulation.
- **Reaction 2 (H2O2 -> H2O):** Unaffected — configured as `reversible: False`, anodic = 0.
- **Surrogate models:** Trained on PDE data with this approximation, so they are self-consistent.
  Retraining may be needed after the fix if near-equilibrium accuracy matters.

## Decision rationale

Implementing the full physics is a one-line change that makes the model more correct without
downside. The `anodic_species` config field already exists and is parsed — it just needs to be
used. All existing results remain valid for the transport-limited regime where they were generated.

## Change

```python
# Before (constant c_ref):
anodic = k0_j * c_ref_j * fd.exp((1-alpha_j) * n_e_j * eta_clipped)

# After (dynamic surface concentration):
anod_idx = rxn["anodic_species"]
anodic = k0_j * c_surf[anod_idx] * fd.exp((1-alpha_j) * n_e_j * eta_clipped)
```

When `anodic_species` is None (not configured), fall back to constant `c_ref_j` for backward
compatibility.
