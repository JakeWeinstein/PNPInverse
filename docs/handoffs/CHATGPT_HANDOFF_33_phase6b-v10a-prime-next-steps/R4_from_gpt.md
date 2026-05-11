1. WHAT: Minor nit: the R4e scalar decomposition still says to verify `log_R4e_predicted ≈ log_R4e_measured`.

WHY: Since it is explicitly a scalar approximation over boundary-averaged nonlinear terms, mismatch can come from averaging/covariance, not only a missing Stern/diffuse term.

WHAT TO DO: Keep it, but interpret only order-of-magnitude discrepancies as diagnostic. `log_R4e_measured` remains authoritative. Not blocking.

2. WHAT: Minor wording inconsistency: Risk #5 says v10c is “the next physical knob” after dense bracket failure, while the decision tree says Case D escalates to acceptance-bundle review.

WHY: Could confuse the handoff reader about whether v10c is automatic.

WHAT TO DO: Change Risk #5 to “v10c is a possible next physical knob after review.” Not blocking.

All prior material blockers are addressed: K0/α/c_H math is separated, C_S/cap routing is deterministic, bracket coverage now hits the relevant branch windows, transport artifacts are routed, and the decomposition language is appropriately cautious.

VERDICT: APPROVED