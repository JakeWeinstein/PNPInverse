No blocking issues remain. Carry these non-blocking implementation guards into the patch:

1. β units: `β = 2·A·z·r_H_El·(...)` has units of `pm²` if σ is `counts/pm²`, not `pm`. The formula is right; fix the unit label.

2. Ring-current basis: define whether `I_ring_model` is actual collected ring current, disk-basis H2O2 current, or current density on ring area. The RRDE formulas need the same basis as `Summary Data-Error.xlsx`.

3. β sign: add a guard/test that the fitted β values preserve cathodic pKa lowering unless an exploratory rule explicitly allows sign reversal.

VERDICT: APPROVED