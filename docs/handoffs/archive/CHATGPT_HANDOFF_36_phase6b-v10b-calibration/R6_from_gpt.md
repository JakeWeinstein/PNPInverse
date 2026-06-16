1. **One implementation constraint still needs to be explicit.**  
WHAT: D7 drivers now call `gamma_ss_langmuir`, which lives in `Forward.bv_solver.cation_hydrolysis` and imports Firedrake.  
WHY: The plan also requires fast CLI/schema tests for the new bracket/matrix drivers. If those drivers import `gamma_ss_langmuir` at module top level, the fast tests become Firedrake-dependent.  
WHAT TO DO: Add one sentence to P36/P9: import `gamma_ss_langmuir` lazily inside the solver-running analytic-audit path, not at module import time. CLI parse/schema helpers must remain importable without Firedrake.

VERDICT: ISSUES_REMAIN