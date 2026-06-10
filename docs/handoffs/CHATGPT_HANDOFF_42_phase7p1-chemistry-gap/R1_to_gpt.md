# Handoff 42 — Adversarial review: Phase 7.1 plan for the residual model/chemistry gap

## Section 1: Context bundle

System: steady-state PNP + Butler-Volmer FEM (Firedrake) of ORR on CMK-3
carbon RRDE, Cs2SO4 pH 4, 1600 rpm (single stagnant film L_eff = 15.4 um =
O2 Levich equivalent; bulk Dirichlet at the film edge). 3 dynamic species
(O2, H2O2, H+ via electrochemical potential mu_H), analytic Bikerman
counterions (Cs+, SO4 2-), Stern Robin BC (0.20 F/m2), water-ionization
fast-equilibrium closure c_OH = Kw/c_H with proton-condition variable
E = c_H − c_OH (known caveat: this closure supplies H+ from water at up to
~20x the real dissociation rate — "Kw-laundered H+"). Deck OCP convention:
V grid and all E° shifted by −0.903 V so the rest state has a flat double
layer; eta_BV preserved exactly.

Reactions (current fitted config — WATER ROUTES ONLY, acid k0=0):
- R2e_water: O2 + 2H2O + 2e− → H2O2 + 2OH−, rate k0·c_O2·exp(−α·2·η/V_T),
  irreversible, no c_H factor (empirical cathodic Tafel branch; E_ref
  0.695 V is a formal onset parameter).
- R4e_water: same form, n=4, E_ref 1.23 V.
Acid-route variants (rates ∝ c_O2·c_H², c_O2·c_H⁴, production-calibrated
k0/α: k0_acid2e basis 2.4e-8 m/s, α=0.627; reversible 2e) exist in the
model but were EXCLUDED from the fit for clean parametrization.

Target: exact vector extraction of the experimental H2O2-current curve
(754 SVG vertices; 33 bins with measurement-scatter σ; "thresholded-zero"
tail flagged). Shape: left plateau −0.165 (data ends at V=−0.33), broad
flat trough −0.35..−0.37 over V ∈ [0.0, +0.15], a LOCAL BUMP at
+0.22..+0.27 (rises to −0.31, dips to −0.33), sharp cliff +0.28..+0.35,
zero (thresholded) beyond +0.45. All V vs RHE, cathodic negative.

Fit result (adjoint L-BFGS-B, gradients FD-verified to 1.2e-2 with
h-convergence; 25 it): θ* = (k0_2w = 2.07e-4×base, k0_4w = 2.90e-14×base,
α_2w = 0.550, α_4w = 0.2854); fine grid 25/25, chi2/pt 29.8 (relative
metric). Key mechanism finding: the optimizer moved both α's but PRESERVED
α₄n₄ − α₂n₂ = +0.042 (pre-fit analytic estimate +0.045): the data stiffly
constrains (k0 ratio, Δ(αn)), weakly the individual α's. The volcano
exists in the model as: anodic flank = 2e Tafel onset; cathodic flank =
4e channel slowly out-competing 2e for surface O2 (slope difference
+0.04); both channels transport-coupled through one O2 pool.

Structured residuals (the "gap" under review):
R1. Trough +0.08 V anodic of data (−0.391@+0.194 vs −0.368@+0.101); cliff
    ~0.04 V early. Mid-section compressed.
R2. The +0.22..+0.27 bump is ABSENT and is structurally unreachable for
    two smooth competing Tafel branches (monotone share ratio in V ⇒ pc
    has a single interior extremum; a local max-dip-cliff sequence needs a
    third process). Bump amplitude ~3σ of the extraction scatter.
R3. Data's left plateau flattens below −0.25 V; model still declines.
R4. Model total cd −5.5 at cathodic end vs deck total ~3 (partition only
    sanity-hinged: NO disk-current series exists in the target; the raw
    xlsx with disk LSV columns is identified but not yet obtained).
Model-internal fact: the model's surface pH collapses 9.3 → 3 precisely
over +0.27..+0.35 — the same window as the bump and cliff. Surface pH
under load is 9.0–9.7 (Ruggiero's IrOx-measured band is ~8–9 at pH 4).

## Section 2: The artifact under review (the Phase 7.1 plan, verbatim)

Hypotheses (ranked), each with falsifiable signature:
H1 Acid/water route handoff missing (acid excluded from fit; real system
   is acid-kinetic at onset where local pH≈4, water-kinetic under load).
   Signature: re-enabling acid R2e adds 2e structure in the pH-transition
   window (+0.15..+0.35), decompresses trough/cliff.
H2 Two-branch family limit: flat-trough+sharp-cliff outside the 4-param
   family even with handoff. Signature: 6-param refit leaves R1 with χ²
   within ~2× of current.
H3 Potential-dependent surface transition (Frumkin cation blocking or
   CMK-3 quinone/hydroquinone redox, known to sit at +0.2..0.5 V_RHE)
   modulating the 2e k_eff. Signature: logistic θ(V) multiplier reproduces
   bump+dip with fitted V_θ inside +0.2..0.3 ONLY if H1/H2 fail.
H4 R3 peroxide re-reduction (consumes H2O2). Its anodic shutoff creates
   structure near onset BUT deepens the cathodic H2O2 deficit, likely
   conflicting with the left plateau; source paper's topology says no
   surface consumption of free H2O2 on carbon. Falsification-only arm.
H5 Single-film ionic-δ bias (δ_H/δ_O2=1.7, δ_OH/δ_O2=1.4 — we use one
   O2-based film): biases WHERE in V the local-pH transition sits, hence
   the bump/cliff window. Signature: L_eff 21.7/26.2 um reruns shift
   cliff/trough ≥0.03 V.

Stage A (diagnosis, ~1 day): A1 θ*+acid at locked production k0/α (fine
grid; in flight). A2 L_eff {21.7, 26.2} at θ*. A3 bisulfate stress bracket
(bulk H+ 0.1→1.1 mol/m3 = free H+ + full HSO4− pool as protons; ceiling
~0.83 mA/cm2) at θ*. A4 profile slices from the 35 checkpointed fit evals
(no compute) → stiff/sloppy directions documented.
Gate A: measured magnitude per hypothesis; pick ≤2 levers that move R1/R2.

Stage B (6-param adjoint refit, H1): controls += acid-R2e (k0, α);
x0 = (θ*, f_acid=1.0, α=0.627); FD-gate new components (fresh-walk
h-convergence) before trusting. Acceptance: chi2/pt ≥2× better AND trough
error ≤0.05 V AND model becomes non-monotonic in +0.15..+0.35. AIC-style:
2 extra params must buy Δχ²·n/2 ≥ 4 else H1 insufficient → Stage C.

Stage C (H3 closure, only on B-failure): k0_eff(V) = k0·[1 − A·σ((V−V_θ)/w)]
as a per-V algebraic multiplier on the k0 R-space Function — assigned by
the driver per grid point, NO weak-form change; the V-schedule's parameter
dependence is closed-form so the outer gradient composes with the adjoint
dJ/dk0 per point. Fit (A, V_θ, w) + Stage-B 6. Falsification guard: V_θ
outside +0.15..+0.40 or A saturating 0/1 ⇒ H3 rejected, R2 documented as
open surface chemistry (quinone hypothesis + CV literature pointer to the
group). H4 runs as a 1-run falsification sweep only.

Stage D (lock + document): final fine-grid θ, residual-panel figure,
ledger checks (acid-share per V, E/O2 closures, electron consistency,
anodic share <1%), surface-pH overlay vs Ruggiero, memory/summary updates;
the existing Phase-6 ablation matrix then runs at θ_final.

Parallel data asks: the raw xlsx (ring AND disk LSV → pins partition,
kills R4 hinge); confirm the 0.47 V OCP component.

Risks: (1) acid branch re-opens Kw-laundered-H+ artifact cathodically —
mitigate: ledger acid-share per V, acid k0 box-bounded ≤ production,
identification comes from the onset window; (2) 8-param identifiability —
profile slices first, AIC discipline, FD gates; (3) bump could be
measurement structure — ~3σ argues real; report the alternative; (4)
maxiter-limited fits — warm-restart from prior x*.

## Section 3: Critique prompt

You are an adversarial reviewer. Be critical. Be argumentative.
Find every hole: missing steps, wrong algebra, untested assumptions,
edge cases not addressed, implicit dependencies, claims without
evidence, off-by-one errors, sign errors, dimensional errors. Don't
be polite — if something is wrong, say so. Concision over hedging.

For each issue, state:
  - WHAT is wrong (specific, not vague — name the line or symbol)
  - WHY it matters (what breaks downstream if uncorrected)
  - WHAT to do (concrete fix, or what evidence would close the gap)

Number your issues. After all issues, end your response with exactly
one of these lines, no other text after it:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN

Use APPROVED only when there are no issues you would block on.
Minor nitpicks alone do not justify ISSUES_REMAIN — call them out
but still verdict APPROVED. Use ISSUES_REMAIN whenever any of your
issues are genuinely blocking.
