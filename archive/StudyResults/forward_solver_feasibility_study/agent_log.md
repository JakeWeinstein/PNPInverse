# Forward Solver Feasibility Study - Agent Log

**Date:** 2026-02-28
**Target:** eta_hat = -46.5 (V_RHE = -0.5 V)
**Solver:** PNP-BV with steric a=0.05, 300-cell graded mesh, 300 continuation steps

## Summary

| Configuration | z_vals | max eta_hat | V_RHE | Points | Time | Status |
|---|---|---|---|---|---|---|
| Full charged | [0,0,+1,-1] | -46.51 | -0.500 V | 300/300 | 185s | PASS |
| 2-sp H2O2 charged | [0,+1] | 0.00 | +0.695 V | 0/300 | 12s | FAIL |
| H+ only charged | [0,0,+1,0] | 0.00 | +0.695 V | 0/300 | 1s | FAIL |
| ClO4- only charged | [0,0,0,-1] | 0.00 | +0.695 V | 0/300 | 0s | FAIL |

**Only the fully electroneutral configuration z=[0,0,+1,-1] converges.**
All other configurations (which break electroneutrality) fail at the very first voltage step (eta=-0.16, which is only 0.3% of the target).

## Detailed Results

### z=[0,0,+1,-1] (full charged) -- PASS

- Reached full target: eta_hat=-46.51 (V_RHE=-0.500 V) in 184.5s
- All 300 continuation steps converged
- Final electrode concentrations:
  - c_O2 = 0.0033 (depleted from 1.0 bulk -- transport-limited)
  - c_H2O2 = 1.1213 (product accumulation)
  - c_H+ = 0.0000 (essentially zero at electrode -- depleted by BV reaction)
  - c_ClO4- = 0.0000 (swept away by electric field)
- Current density evolution: I_total approaches -0.188 mA/cm2 at V_RHE=-0.5

### z=[0,+1] (2-sp, H2O2 charged) -- FAIL

- DIVERGED_DTOL at first step (eta=-0.16)
- Failure mode: No counterion to balance H2O2 charge
- The Poisson equation creates unbounded electric field with no electroneutrality
- Even with steric regularization, the solver cannot find a physical solution
- **Root cause:** Single charged species without counterion is mathematically ill-posed for PNP

### z=[0,0,+1,0] (H+ only charged) -- FAIL

- DIVERGED_DTOL at first step (eta=-0.16), 12 SNES iterations
- H+ charge creates positive space charge with no balancing anion
- Poisson equation drives phi to extreme values trying to satisfy charge balance
- **Root cause:** Net positive charge density with no neutralization mechanism

### z=[0,0,0,-1] (ClO4- only charged) -- FAIL

- DIVERGED_DTOL at first step (eta=-0.16), only 5 SNES iterations (fastest failure)
- ClO4- charge creates negative space charge
- Same fundamental issue as H+-only but opposite sign
- **Root cause:** Net negative charge density with no neutralization mechanism

## Key Findings

1. **Electroneutrality is essential.** The PNP solver requires balanced charge vectors (sum of z_i * c_i = 0 in bulk) to converge. Any charge configuration that breaks electroneutrality fails immediately at the first voltage step.

2. **The z=[0,0,1,-1] baseline is the only viable charged configuration.** It maintains electroneutrality through the H+/ClO4- pair and converges robustly to the full experimental range.

3. **The z=[0,1] configuration is fundamentally broken.** Assigning charge to H2O2 without a counterion violates the mathematical structure of PNP. This is not a convergence issue that can be fixed with better solver parameters -- it is a modeling error.

4. **All single-charge configs fail identically** at eta=-0.16 with DIVERGED_DTOL, confirming the issue is the charge balance, not the magnitude of overpotential.

5. **For inverse problems**, the forward solver should always use z=[0,0,1,-1] (full charged) or z=[0,0,0,0] (all neutral). Partial charge assignments are not physically meaningful for PNP.

## Recommendations

- Use z=[0,0,1,-1] for all production runs (already validated in Phase 1)
- For simplified/faster models, use z=[0,0,0,0] (all neutral) which avoids Poisson entirely
- Do not attempt partial charge assignments -- they are ill-posed
- The 2-species z=[0,1] configuration should be documented as "not viable" for PNP
