1. WHAT: Re 4’s migration sign test is wrong: “HSO4− migrates TOWARD the cathode.”

WHY: A cathode under ORR is the negative electrode; anions should be repelled from it, not attracted. If your sign test encodes “toward cathode” as expected behavior, it will bless a reversed electromigration term.

WHAT TO DO: Rewrite the sign test in coordinate-free terms: for a negative electrode potential relative to bulk, HSO4− and SO4²− deplete near the electrode and their migrational flux points away from the cathode. If your convention gives the opposite, stop and fix the sign ledger.

2. WHAT: Re 1’s “finite sulfur” closure still uses `S_tot(x)` as the existing analytic SO4²− counterion pool, then splits it pointwise.

WHY: That freezes total sulfur to the old divalent-sulfate spatial distribution. But after protonation, part of the pool is monovalent HSO4− with different electrostatic weight and mobility. This is not the same as a physical sulfur reservoir/speciation equilibrium.

WHAT TO DO: Derive `S_tot`, `c_SO4`, and `c_HSO4` jointly from sulfur chemical potential, electrostatic potential, sterics, and acid equilibrium. If you keep pointwise splitting of the old SO4 pool, label B′ as a local-buffer sensitivity, not a physical buffer-transport closure.

3. WHAT: Re 1’s closed-cell sulfur conservation test is nearly trivial if `S_tot(x)` is pointwise imposed.

WHY: Conserving an imposed field does not prove the buffer cannot launder protons. The real question is whether bound-proton delivery by HSO4 is balanced by finite sulfur/bisulfate transport.

WHAT TO DO: Add an integrated buffer balance: domain-integrated HSO4 deprotonation/proton delivery must equal the appropriate boundary flux of bound-proton carrier, with sign, to tight tolerance. Report this beside the sulfur ceiling.

4. WHAT: Re 5 gates B′ on `pH = 6.5`, but Re 10 only credits acid-2e chemistry at `pH ≤ 5.0`.

WHY: A pH 6.5 midpoint inside +0.22..+0.31 does not mean acid kinetics are active in the bump window. The pH 5 crossing could still sit too anodic.

WHAT TO DO: Gate acid handoff on the pH threshold that actually permits acid credit: require the `pH ≤ 5.0` region to overlap the bump/cliff window for acid-2e. Keep pH 6.5 only as a descriptive midpoint.

5. WHAT: Re 7’s tail-threshold conversion needs an explicit unit ledger.

WHY: The formula is correct only if `0.001 mA/cm²(ring)` is first multiplied by ring area `0.11 cm²`, then divided by `N * A_disk`. If the source quantity is already a ring current, the threshold is wrong by the ring-area factor.

WHAT TO DO: State the raw unit of the threshold from the processing notebook and include the dimensional conversion line in the plan.

6. WHAT: Re 8 scores outer-boundary H2O2 flux as `D_H2O2·∂c/∂n`, but the sign convention is not specified.

WHY: Cathodic peroxide production has been carried as negative current throughout. A flipped outward normal or missing minus sign can silently invert the R3 diagnostic.

WHAT TO DO: Add a one-line sign test: a pure producing 2e run must produce the same cathodic sign in the boundary-flux observable as the original ring-derived target.

VERDICT: ISSUES_REMAIN