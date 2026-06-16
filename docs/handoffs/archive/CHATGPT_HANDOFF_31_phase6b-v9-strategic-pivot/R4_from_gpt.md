1. WHAT: Re point 10, the proposed `G` parameterization still mixes geometry with σ.  
WHY: You define `G = 2·A·z·σ·r_H_El·(1 − r_M-O²/r_H_El²)` and propose storing `G_per_cation_Cu` at Cu σ. That bakes Singh’s Cu surface charge into what is supposed to be the transferable geometry term.  
DO: Store a σ-independent coefficient, e.g. `q = r_H_El·(1 − r_M-O²/r_H_El²)` or `β = 2·A·z·q`, then compute `ΔpKa = β·σ` for whatever σ mapping is active.

2. WHAT: Re point 11, the max-selectivity extraction contradicts itself.  
WHY: You first say model max is taken over a fixed V window, then say use the experimental `Corr Pot` as the model lookup voltage. Those are different observables. The second one biases the model toward the experimental argmax.  
DO: Extract model max over the fixed overlap window. Separately compare the model argmax voltage to experimental `Corr Pot` with the ±50 mV tolerance.

3. WHAT: Re point 4, “smallest V where current exceeds threshold” is directionally ambiguous.  
WHY: In a cathodic sweep from high V to low V, onset is the first crossing as V decreases, which is usually the highest V satisfying the threshold, not the smallest numeric V.  
DO: Define onset as “first threshold crossing encountered when sweeping from anodic to cathodic potential”; interpolate between the bracketing points.

4. WHAT: Re point 4, `n_e = j_total/(F·O2 flux)` is not the same extraction as the experimental RRDE `Number of e-` unless you prove the flux mapping.  
WHY: The table’s `n_e` is almost certainly computed from disk/ring currents using the RRDE formula, not from a directly measured O2 flux.  
DO: Compute model `n_e_rrde = 4|I_D|/(|I_D| + I_R/N)` for comparison to the table. Keep flux-derived n_e as an internal diagnostic only.

5. WHAT: Re point 3, fallback `V_RHE = +0.30 V` is still inherited from the invalid v9 scan.  
WHY: v10a cap, branch currents, and σ_S may shift the best voltage. Hard-coding +0.30 V reintroduces the old regime assumption.  
DO: If no point passes filters, choose the best-scoring point from the actual v10a V-sweep and label it “filter-failed fallback”; do not hard-code +0.30 V.

6. WHAT: Re point 7, “typical CMK-3 RF ~50–500” is still an uncited prior.  
WHY: It can silently become the new knob that rescales C_S into the Singh σ range.  
DO: Remove the numeric roughness range from the plan until the literature note justifies it with normalization and area basis.

VERDICT: ISSUES_REMAIN