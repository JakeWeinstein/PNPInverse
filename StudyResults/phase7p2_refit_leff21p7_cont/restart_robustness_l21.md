# Optimum robustness at the ACCEPTED theta_L (L_eff 21.7)

theta_L = (-1.009, -12.309, 0.577, 0.305), J = 12.195.

| Restart | start | result | verdict |
|---|---|---|---|
| A (+0.5, +0.5, +0.05, +0.05) | (-0.509, -11.81, 0.627, 0.355) | J = 12.195 at (-1.0097, -12.314, 0.5771, 0.3049) | EXACT return (4 decimals) |
| B (-0.5, -0.5, -0.05, -0.05) | (-1.509, -12.81, 0.527, 0.255) | J = 12.633 at (-0.947, -11.93, 0.565, 0.297), maxiter-capped | same basin, inching along the flat k0-ratio valley toward theta_L (dJ 3.6%) |

Verdict: UNIMODAL at 21.7 within the perturbation box (the 15.4
condition's secondary basin does not reappear). B's offset lies along
the expected sloppy valley; the profile protocol quantifies its
width.
