# Optimum robustness — perturbed restarts (session-43 plan, Stage 4)

Main fit: theta_K = (-0.294, -9.001, 0.439, 0.236), J = 293.365
(bins objective, L_eff 15.4).

| Restart | start (theta_K + delta) | result | verdict |
|---|---|---|---|
| B (−0.5, −0.5, −0.05, −0.05) | (−0.794, −9.50, 0.389, 0.186) | J = 293.60 at (−0.329, −9.12, 0.445, 0.239) | SAME basin (ΔJ 0.08%; Δθ within the flat k0-ratio valley) |
| A (+0.5, +0.5, +0.05, +0.05) | (+0.206, −8.50, 0.489, 0.286) | J = 1474.4 at (+0.782, −8.0 bound, 0.198, 0.050 α-floor) | DIFFERENT basin — high-k0_2e / low-alpha compensation regime, 5× worse J; REJECTED on objective |

Verdict: landscape flagged MULTI-MODAL per the plan; the secondary
basin is decisively inferior (J 1474 vs 293) and boundary-pinned in
two coordinates. theta_K stands. The flat valley between the B
solution and theta_K couples (log f_2w, log f_4w) — the profile
protocol quantifies its width.
