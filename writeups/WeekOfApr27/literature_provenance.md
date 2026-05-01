# Literature Provenance: PNP-BV ORR Forward Solver — Three Numerical Changes

---

## 1. Executive Summary

**Verdict table.**

| Change | Canonical precedent | Where the implementation deviates | Novelty assessment |
|---|---|---|---|
| **#1 PBNP hybrid** | Zheng & Wei, *J. Chem. Phys.* **134**, 194101 (2011), DOI 10.1063/1.3581031; deeper roots in Gouy-Chapman-Stern (1910/1913/1924) and the Eisenberg-Coalson-Kurnikova-Lu ion-channel PNP lineage. | Transplanted into a PEMFC-style ORR solver in 0.1 M HClO₄ with spectator ClO₄⁻ as the analytic-Boltzmann species. No surveyed ORR/RDE/fuel-cell paper does this. | **Modest.** Construction reused; *application* to ORR with explicit Boltzmann-ClO₄⁻ inside a PNP-BV solver appears new in the surveyed electrochemistry literature. |
| **#2 log-density `u = ln c`** | Metti, Xu, Liu, *J. Comput. Phys.* **306**, 1-18 (2016), DOI 10.1016/j.jcp.2015.10.053; high-order extension Fu & Xu, *CMAME* **395**, 115031 (2022), DOI 10.1016/j.cma.2022.115031; semiconductor-side ancestry through Slotboom 1969 and Brezzi-Marini-Pietra 1989. | Pushes log-density into a *full PNP-BV ORR solver*, with the same CG order on `u_i` and `φ`, monolithic Newton, backward Euler — and propagates `u_i` through the BV boundary condition (Change #3), not just bulk transport. | **Genuine domain transfer.** No peer-reviewed ORR/RDE/fuel-cell PNP-BV solver in the surveyed literature uses ln c as a primary Galerkin unknown. |
| **#3 log-rate BV** | Tafel 1905, *Z. Phys. Chem.* **50U**, 641-712, DOI 10.1515/zpch-1905-5043 (high-overpotential physical *approximation*, drops a branch); Fattal & Kupferman, *J. Non-Newtonian Fluid Mech.* **123**, 281-285 (2004), DOI 10.1016/j.jnnfm.2004.08.008 (the closest *structural* cousin: log-conformation tensor at high Weissenberg number); analogous CHEMKIN PLOG / log-sum-exp / softmax patterns. | The exact construction `r = exp(ln k_0 + u_cat + Σ_f ν_f (u_sp(f) − ln c_ref,f) − α n_e η/V_T)`, with the cathodic stoichiometric-power loop *inside* the log, evaluated then exponentiated once, does not appear in any electrochemistry paper located. The log-rate path also bypasses the symmetric `_U_CLAMP` underflow protection. | **Small local numerical novelty (defensible).** Algebraically trivial; conceptually parallel to Fattal-Kupferman 2004 and CHEMKIN PLOG; *not documented* as a numerical strategy in any PNP-BV / ORR / RDE paper located. |

**Three corrections the LaTeX writeup must make.**

1. **The "Liu, Maxian, Shen, Zitelli, Waltz, Monteiro 2022, CMAME 396, 115070, arXiv:2105.01163" citation is FABRICATED.** The arXiv ID resolves to **Fu & Xu, *CMAME* 395, 115031, doi:10.1016/j.cma.2022.115031** (two authors, not six; vol. 395 not 396; article 115031 not 115070). Replace.
2. **The Slotboom citation as printed is a chimera** — wrong year for the title given, wrong title for the year given. The canonical Slotboom-variables reference is **Slotboom 1969, *Electronics Letters* 5(26), 677-678, doi:10.1049/el:19690510**. The "PN-product" paper, if cited, is Slotboom 1977 *Solid-State Electronics* 20(4), 279-283. Slotboom variables (`u = c·exp(−zψ/V_T)`) and log-density (`u = ln c`) are *related* but *distinct* substitutions; the writeup should not equate them.
3. **The "Zheng-Chen-Wei JCP 230 (2011)" cited as the PBNP source is the *full-PNP* second-order solver paper, NOT the PBNP paper.** The canonical PBNP paper is **Zheng & Wei, *J. Chem. Phys.* 134, 194101 (2011), DOI 10.1063/1.3581031**. Both can be cited; only the second supports the PBNP-hybrid claim.

**Strongest novelty claims the writeup can defensibly make** (in order of strength):
1. **Application of PNP log-density (`u = ln c`) into ORR-BV electrocatalysis.** No ORR/RDE/fuel-cell PNP-BV solver in the surveyed peer-reviewed literature uses ln c as primary Galerkin unknown.
2. **The log-rate BV construction itself, in a published PNP-BV solver.** The closest *published* analogs are in different domains (viscoelastic fluids — Fattal-Kupferman 2004; combustion — CHEMKIN PLOG; ML — log-sum-exp).
3. **PBNP applied specifically to ORR with Boltzmann-ClO₄⁻** — modest but unobjected.

The first two are the load-bearing novelty claims; the third is a comfortable "reuse in new setting."

---

## 2. Per-Change Literature Provenance

### 2.1 Change #1 — PBNP Hybrid (3-species + analytic Boltzmann counterion)

**Canonical PBNP paper.** Zheng & Wei (2011), *J. Chem. Phys.* **134**(19), 194101, **DOI 10.1063/1.3581031**, PMID 21599038, PMC PMC3122111. Verified abstract: "We propose an alternative model to reduce number of Nernst-Planck equations to be solved … by substituting Nernst-Planck equations with Boltzmann distributions of ion concentrations." This is exactly the writeup's split (3 active dynamic + 1 analytic Boltzmann).

**Companion paper often confused with the above.** Zheng, Chen, Wei (2011), *J. Comput. Phys.* **230**(13), 5239-5262, **DOI 10.1016/j.jcp.2011.03.020**, PMC PMC3087981. Full-PNP numerical-methods paper using matched-interface-and-boundary methods — **not PBNP**. Cite as a high-order full-PNP numerical reference; do not cite as a PBNP source.

**Deeper conceptual ancestry.**
- Gouy 1910, *J. Phys. Theor. Appl.* **9**, 457-468 — diffuse-layer model with Boltzmann ion statistics.
- Chapman 1913, *Phil. Mag.* **25**, 475-481 — closed-form Gouy-Chapman result.
- Stern 1924, *Z. Elektrochem.* **30**, 508-516 — combines Helmholtz + Gouy-Chapman.

PBNP reduces to GCS as the active set vanishes and to full PNP as it expands.

**Active-species-via-NP lineage (ion-channel biophysics).**
- Chen, Barcilon, Eisenberg 1992, *Biophys. J.* **61**, 1372-1393.
- Chen, Eisenberg 1993, *Biophys. J.* **64**, 1405-1421, PMID 7686784.
- Kurnikova, Coalson, Graf, Nitzan 1999, *Biophys. J.* **76**, 642-656.
- Cardenas, Coalson, Kurnikova 2000, *Biophys. J.* **79**, 80-93, PMC1300917.
- Lu, Holst, McCammon, Zhou 2010, *J. Comput. Phys.* **229**, 6979-6994, **DOI 10.1016/j.jcp.2010.05.035**, PMC2922884 — the most direct FEM-PNP predecessor that Wei's group built on.

**Competitor reduction (worth knowing).** Newman electroneutrality (Newman & Thomas-Alyea, *Electrochemical Systems*, 3rd ed., Wiley 2004; 4th ed. with Balsara, 2019; Newman 1975 *AIChE J.* **21**, 25-41). Drops Poisson, enforces Σ z_i c_i = 0. Universal in fuel-cell engineering. Distinct from PBNP.

**ORR-specific context.**
- Shinozaki et al. 2015, *J. Electrochem. Soc.* **162**, F1144-F1158, **DOI 10.1149/2.1071509jes**. Documents that ClO₄⁻ is non-adsorbing/weakly adsorbing on Pt — physical justification for treating it as Boltzmann-equilibrated.
- Generalized modified PNP for PEMFC (Electrochim. Acta 2025, S0013468625004335): ongoing PNP-EDL modeling for fuel cells, but uses *full* PNP, not PBNP.

**Where the implementation aligns.** Realizes Zheng-Wei's PBNP nearly exactly: 3 species (O₂, H₂O₂, H⁺) with full Nernst-Planck transport + a Boltzmann analytic factor `c_ClO4 = c_bulk · exp(−z_ClO4 · φ/V_T)` with `z_ClO4 = −1`, contributing to the Poisson residual. Confirmed at `scripts/studies/v24_3sp_logc_vs_4sp_validation.py:250-262` and `scripts/studies/v18_test_3species_boltzmann.py:109-147`.

### 2.2 Change #2 — Log-Concentration `u_i = ln c_i`

Three relevant lineages.

#### Lineage A — Direct log-density / entropy-variable primary (the writeup's path)

- **Metti, Xu, Liu (2016)**, *J. Comput. Phys.* **306**, 1-18, **DOI 10.1016/j.jcp.2015.10.053**. Performs a logarithmic transformation of charge-carrier densities in FEM, proves a discrete energy estimate matching the continuous PNP energy law, enforces positivity. Canonical PNP log-density FEM paper. The phrase "log-density formulation" appears coined retroactively by Fu-Xu and successors; technique is in the paper, exact phrase may not be in the abstract.
- **Fu & Xu (2022)**, *CMAME* **395**, 115031, **DOI 10.1016/j.cma.2022.115031**, arXiv:2105.01163. Discretizes the entropy variable `u_i = U'(c_i) = log c_i` directly with FEM in space + DG in time; positivity by construction (`c_i = exp(u_i) > 0`); unconditional energy stability at arbitrary order. **This is the paper currently mis-cited as "Liu, Maxian, Shen, Zitelli, Waltz, Monteiro 2022, CMAME 396, 115070."** That citation is fabricated.

#### Lineage B — Slotboom variables / excess-chemical-potential flux (semiconductor)

- **Slotboom 1969**, *Electronics Letters* **5**(26), 677-678, **DOI 10.1049/el:19690510**. Canonical "Slotboom variables" reference: `Φ_n = n·exp(−ψ/V_T)`.
- **Slotboom 1973**, *IEEE TED* **ED-20**, 669-679. Standard "1973 Slotboom" reference (bipolar transistor analysis).
- **Slotboom 1977**, *Solid-State Electron.* **20**(4), 279-283. The actual "PN-product in silicon" paper — band-gap-narrowing, **not** about log-density.
- **Scharfetter & Gummel 1969**, *IEEE TED* **ED-16**(1), 64-77. Exponential-fitting / upwind FV on c-primary; Bernoulli-function edge fluxes. **NOT** a log-density substitution.
- **Brezzi, Marini, Pietra 1989**, *SIAM J. Numer. Anal.* **26**(6), 1342-1355. Mixed-FE generalization of SG.
- **Markowich 1986**, *The Stationary Semiconductor Device Equations*, Springer, DOI 10.1007/978-3-7091-3678-2.
- **Markowich, Ringhofer, Schmeiser 1990**, *Semiconductor Equations*, Springer.
- **Selberherr 1984**, *Analysis and Simulation of Semiconductor Devices*, Springer.

**Mapping between Slotboom and log-density.** `ln(Φ_n) = ln(n) − ψ/V_T`, so log-density and Slotboom differ only by an affine map *as continuous variables*. Galerkin discretizations on (Φ_n, ψ) versus (ln n, ψ) yield different stiffness matrices. The writeup must not equate them.

#### Lineage C — Modern energy-stable / SAV / IEQ schemes for PNP

| Citation | What it does |
|---|---|
| H. Liu & Maimaitiyiming 2021, *J. Sci. Comput.* **87**(3), 92 | FD c-primary positivity + energy stability for multi-D PNP |
| C. Liu, Wang, Wise, Yue, Zhou, arXiv:2009.08076 (later *Math. Comp.* 2022) | H⁻¹ gradient flow with singular log potential treated implicitly |
| Huang & Shen 2021, *SIAM J. Sci. Comput.* **43**(3), B746-B759, DOI 10.1137/20M1365417 | SAV scheme; c-primary; bound-preservation via auxiliary variable |
| Cancès, Chainais-Hillairet, Gaudeul, Fuhrmann 2022, *Numer. Math.* **151**, 949-986, DOI 10.1007/s00211-022-01279-y | Entropy-FV with size exclusion; modern bridge to electrochemistry |
| Farrell, Rotundo, Doan, Kantner, Fuhrmann, Koprucki book chapter (2017) | Surveys flux discretizations including Slotboom and excess-chemical-potential |

**Conceptual map.** PNP free energy `E[c, ψ] = Σ_i ∫ c_i (ln c_i − 1) dx + ½ε ∫ |∇ψ|² dx − Σ_i z_i ∫ c_i ψ dx`. Three discretization strategies: (A) direct log-density / entropy-variable primary (Metti-Xu-Liu → Fu-Xu); (B) excess-chemical-potential flux (Slotboom → Brezzi-Marini-Pietra → Cancès); (C) SAV / IEQ (Shen-Xu-Yang → Huang-Shen). The writeup uses (A).

**Why electrochemistry diverged from semiconductor practice.** Per Selberherr 1984 ch. 5 and Markowich-Ringhofer-Schmeiser 1990 ch. 3, Slotboom variables work poorly under degenerate Fermi-Dirac statistics; modern semiconductor codes prefer quasi-Fermi or excess-chemical-potential. Electrochemistry rarely needs degeneracy correction so could use Slotboom, but the field never adopted it — codes descended from Newman's CONDUC/DUALFOIL c-primary FD lineage rather than semiconductor TCAD.

**Where the implementation aligns.** `forms_logc.py:55-56` constructs `MixedFunctionSpace([V_scalar]*(n+1))` with `u_i, φ` all in CG-order on the same space; `forms_logc.py:177-178` confirms `u_i = ln c_i` is the primary unknown and `c_i` is a derived expression `c_i = exp(u_i)` (with symmetric `_U_CLAMP=±30` overflow protection — `forms_logc.py:185-198`). Flux uses log-transform identity `∇c = c·∇u` directly (`forms_logc.py:281`). Time stepping is backward Euler (`forms_logc.py:283-285`), single past state (NOT BDF2).

### 2.3 Change #3 — Log-Rate Butler-Volmer

The construction: rewrite `r = k_0 · c_i · exp(−α n_e η/V_T)` as `r = exp(ln k_0 + u_i − α n_e η/V_T)` with `u_i = ln c_i`, evaluated as one exponentiation at the end. The cathodic branch folds in stoichiometric power *inside* the log: `ln r_cat = ln k_0 + u_cat + Σ_f ν_f (u_sp(f) − ln c_ref,f) − α n_e η/V_T`.

**Tafel 1905 — historical anchor, conceptually different.** J. Tafel, *Z. Phys. Chem.* **50U**(1), 641-712, **DOI 10.1515/zpch-1905-5043**. Verified via De Gruyter, ESTIR Historic Papers in Electrochemistry, Russian Mendeleev archive.

**Critical conceptual distinction (must not conflate).**
- Tafel 1905 = high-overpotential physical *approximation* that **drops one branch**: `log|i| ≈ const + (α n F / RT) η`.
- Log-rate BV = exact algebraic identity that **drops nothing**: `r = exp(ln k_0 + ln c_i − α n_e η/V_T)`.

**Frumkin 1933 — physical correction, also distinct.** A. N. Frumkin, *Z. Phys. Chem. A* **164**, 121-133. Modifies BV potential drop: `η_eff = η − φ_d`. Different correction (interfacial physics), not numerical.

**Bazant generalized BV and the gFBV+PNP framework — physical generalizations.**
- Bazant 2013, *Acc. Chem. Res.* **46**, 1144-1160, **DOI 10.1021/ar300145c**, PMID 23520980, arXiv:1208.1587. Generalizes BV via nonequilibrium thermodynamics; physical not numerical.
- Bazant, Thornton, Ajdari 2004, *Phys. Rev. E* **70**, 021506, **DOI 10.1103/PhysRevE.70.021506**, PMID 15447495. PNP coupled to gFBV. **Standard exponential BV form throughout, no log-form rate evaluation.**
- Bazant, Kilic, Storey, Ajdari 2009, *Adv. Colloid Interface Sci.* **152**, 48-88, **DOI 10.1016/j.cis.2009.10.001**.
- van Soestbergen, Biesheuvel, Bazant 2010, *Phys. Rev. E* **81**, 021503.
- van Soestbergen 2012, *Russ. J. Electrochem.* **48**, 570-579, **DOI 10.1134/S1023193512060110**.

These represent the state of the art in PNP-BV. None address the *numerical evaluation* of BV as a stiff exponential.

**Algebraic / inversion-direction reformulations (related but distinct problem).**
- Khalili et al. 2016, ASME *J. Electrochem. En. Conv. Stor.* **13**(2), 021003. Algebraic reformulation to *invert* BV for η.
- Nasser & Mantegazza 2022, *J. Phys. Chem. C*, DOI 10.1021/acs.jpcc.1c09620. Deformed exponentials for fitting.
- COMSOL battery/electrochemistry: re-references overpotential against fixed activity reference; re-parametrization not log-form.

**The closest published structural cousin — Fattal-Kupferman 2004.**
- **Fattal & Kupferman 2004**, *J. Non-Newtonian Fluid Mech.* **123**(2-3), 281-285, **DOI 10.1016/j.jnnfm.2004.08.008**. Verified via Hebrew U CRIS and ScienceDirect. Citation count ≈ 484. Key idea: "transform a large class of differential constitutive models into an equation for the matrix logarithm of the conformation tensor"; high-Weissenberg numerical instability is "the failure of polynomial-based approximations to properly represent exponential profiles."
- Companion: Fattal & Kupferman 2005, *J. Non-Newtonian Fluid Mech.* **126**, 23-37, **DOI 10.1016/j.jnnfm.2004.12.003**.

**Exact mapping**:
- Fattal-Kupferman: `Ψ = log(Conformation)`; discretize `Ψ`; `exp(Ψ)` only when needed.
- Writeup: `ln r = ln k_0 + u_i − α n_e η/V_T`; assemble `ln r`; `r = exp(ln r)` once.

**Stiff combustion / chemistry.** CHEMKIN-II/III (Kee, Rupley, Miller, SAND-89-8009; SAND96-8216) — PLOG (pressure-dependent logarithmic interpolation) reactions. Lu & Law 2009, *Prog. Energy Combust. Sci.* **35**, 192-215. Pope 1997, *Combust. Theory Model.* **1**, 41-63. Higham et al. 2021, *IMA J. Numer. Anal.* **41**, 2311-2330, **DOI 10.1093/imanum/draa038** (log-sum-exp / softmax formal analysis).

The "compute log of the rate, then exponentiate once" pattern is well-known in CHEMKIN-style stiff chemistry codes and the closely-related log-sum-exp idiom in machine learning. **What is missing is the explicit transplantation into electrochemistry / Butler-Volmer evaluation as a published numerical strategy in a PNP-BV solver paper.**

**Where the implementation aligns.** Exact algebra at `Forward/bv_solver/forms_logc.py:324-356` (cathodic 324-339; anodic 340-356). The writeup's existing pointer "see `forms_logc.py:294-360`" undershoots; full if/else block runs `lines 299-389`. **Recommend pointer `forms_logc.py:294-388`.** Clamp-bypass argument documented at `forms_logc.py:294-298` and `docs/CHATGPT_HANDOFF_6_LOGRATE_BREAKTHROUGH.md:107-118`.

---

## 3. Per-Change Deviation Analysis

### 3.1 Change #1 — Deviations from cited PBNP literature

**Reused (non-novel).** The Boltzmann-passive + NP-active reduction; sign convention `z_ClO4 = −1`; clamping `φ` at ±50 dimensionless units (≈ ±1.28 V) before exponentiation; non-dimensional `charge_rhs` prefactor.

**In implementation but not in cited literature.**
1. **Application to ORR / 0.1 M HClO₄.** No surveyed peer-reviewed paper Boltzmann-distributes ClO₄⁻ (or any non-adsorbing anion) inside a PNP-BV ORR forward solver. The PEMFC PNP literature (Bessler, Eikerling, Weber-Newman) uses either full PNP or Newman electroneutrality. The construction is Zheng-Wei 2011; the application is the contribution.
2. **Architectural fragility — `add_boltzmann()` monkey-patch.** The Boltzmann contribution is **not** in `Forward/bv_solver/forms_logc.py`. It is duplicated as a closure in 7 study scripts (`v18_logc_lsq_inverse.py:234`, `v24_3sp_logc_vs_4sp_validation.py:250`, `v18_logc_diagnostics.py:117`, `v18_logc_noise_sensitivity.py:85`, `v19_lograte_extended_adjoint_check.py:166`, `plot_iv_curves_3sp_true.py:108`, `v18_test_3species_boltzmann.py:109`). The bare `forms_logc.py` Poisson residual contains only the three-explicit-species sum at line 418. Tightening recommendation, not literature finding.

**Genuine novelty assessment.** Modest. Construction fully reused; only the application domain (ORR PNP-BV with Boltzmann ClO₄⁻) appears undocumented.

### 3.2 Change #2 — Deviations from cited log-density literature

**Reused (non-novel).**
1. The substitution `u_i = ln c_i`, recovery `c_i = exp(u_i)`, flux identity `∇c = c · ∇u` — directly from Metti-Xu-Liu 2016 and Fu-Xu 2022.
2. Energy-stable interpretation: `c_i = exp(u_i) > 0` ⇒ positivity by construction; discrete entropy / energy stability arguments transfer.
3. Monolithic Newton over Gummel splitting — standard in modern energy-stable PNP literature.
4. Same CG order on `u_i` and `φ` — standard in Metti-Xu-Liu and FEM-PNP-electrochemistry generally.

**In implementation but not in cited literature.**
1. **PNP-BV ORR application** (load-bearing novelty). After targeted search, no peer-reviewed electrochemistry-side ORR / RDE / fuel-cell / battery PNP-BV solver paper uses ln c as a primary Galerkin unknown. Standard practice in that community is c-primary with adaptive meshing. Closest electrochemistry-side log-density work is in Li-ion concentrated-solution theory (Newman group, Doyle-Fuller-Newman 1993+), where chemical potential `μ = μ_0 + RT · ln(γc)` appears in fluxes — but they still discretize `c`, not `ln c`.
2. **Propagation of `u_i` into the BV boundary condition** (Change #3). Metti-Xu-Liu and Fu-Xu handle bulk PNP transport only.
3. **Symmetric `_U_CLAMP = ±30` as overflow protection rather than concentration floor** (`forms_logc.py:185-198`). Comment: "exp(±30) covers [9.4e-14, 1.07e+13], adequate for typical EDL profiles." Bypassed in log-rate boundary path. Small construction choice, not in any specific cited paper, consistent with general floating-point hygiene.
4. **`_C_FLOOR = 1e-20` Dirichlet-BC value for product species** (`forms_logc.py:431-437`) to avoid `ln(0)` for H₂O₂ initial concentration. Small local construction.

**Genuine novelty assessment.** Genuine — but as a *domain transfer*, not a new substitution. Substitution itself is old (semiconductor 1969, mathematical PNP 2016). No surveyed ORR-BV PNP solver uses it. Confidence medium-high (proving a negative is impossible, but targeted search returned nothing).

### 3.3 Change #3 — Deviations from cited / surveyed BV literature

**Reused (non-novel).**
1. Standard BV cathodic kinetic expression `r = k_0 · c_i · exp(−α n_e η/V_T)` — textbook Bockris-Reddy / Bard-Faulkner.
2. Stoichiometric-power loop in cathodic branch — `Π_f c_sp(f)^ν_f` — textbook mass-action.
3. The algebraic identity itself — `r = k_0 · c_i · exp(...)` ↔ `exp(ln k_0 + ln c_i + ...)` — is trivial. Any electrochemist could derive it in two lines.
4. "Evaluate inside the log to avoid overflow at extreme rates" pattern — well-established in stiff combustion (CHEMKIN PLOG), viscoelastic flow (Fattal-Kupferman), and ML (log-sum-exp). Standard machinery.

**In implementation but not in surveyed literature.**
1. **The literal log-rate BV identity used as a numerical strategy in a PNP-BV / ORR solver.** No electrochemistry paper located that publishes this construction in a PNP-BV solver context. Closest published cousin is Fattal-Kupferman 2004 in viscoelastic fluid mechanics — *structurally identical, domain-different*. The "small local numerical novelty" hedge is defensible.
2. **Clamp-bypass as the primary motivation.** The implementation's reason is specifically "evaluating BV against `_U_CLAMP`-bounded `c_surf` produces a phantom R2 cathodic sink at low surface H₂O₂." Mechanism per `docs/CHATGPT_HANDOFF_6_LOGRATE_BREAKTHROUGH.md:107-118`: when Newton needs `c_H2O2` below `exp(−30) ≈ 9.4e-14`, the symmetric clamp pins it; combined with saturated `exp(50)` in R2's BV exponent, the floor times the huge exp gives a spurious R2 sink that nothing else can balance. Log-rate evaluates `exp(ln k_0 + u_H2O2 + 2(u_H − ln c_ref) − α·n_e·η)` so `u_H2O2` enters additively and can be arbitrarily negative; expression decays smoothly to zero. **Specific clamp-bypass argument not in any cited literature** — it is a property of *this* implementation's interaction.
3. **Asymmetry: cathodic branch carries full sum loop; anodic branch is single-species** (`forms_logc.py:340-356`). The writeup's "both branches subtracted with `Σ_f ν_f (u_sp(f) − ln c_ref,f)`" is *partially* accurate: cathodic has the full loop (lines 331-337); anodic has only bare anodic-species term `u_anod` (lines 343-345 main case, 350-353 fallback). For production, R2 is irreversible (`v24_3sp_logc_vs_4sp_validation.py:233`), so anodic branch falls into else at line 355 → `anodic = 0`. Algebra-description should be tightened.
4. **Line-range pointer drift.** Writeup says `forms_logc.py:294-360`. Codebase shows log-rate true branch at `324-356`; full if/else block runs `294-388`. **Recommend `forms_logc.py:294-388`.**

**Genuine novelty assessment.** Small local numerical novelty (defensible). Construction mathematically trivial; deployment in PNP-BV / ORR solver undocumented in surveyed literature; structurally identical machinery exists in different fields. The unique angle is the clamp-bypass mechanism specific to this code's `_U_CLAMP`-vs-saturated-BV interaction.

---

## 4. Surrounding-Methods Provenance

### 4.1 ORR Mechanism / RRDE

**Two-step equilibrium potentials.** R1: O₂ + 2H⁺ + 2e⁻ → H₂O₂, `E_eq^(1) = +0.695 V` vs SHE (≈ 0.68 V). R2: H₂O₂ + 2H⁺ + 2e⁻ → 2H₂O, `E_eq^(2) = +1.776 V` vs SHE (≈ 1.78 V). Sum: 4e⁻ ORR with `E_eq = +1.229 V` vs SHE. Cite Bard, Faulkner & White 3rd ed. (2022) Ch. 9.

**Production reaction config** (`v24_3sp_logc_vs_4sp_validation.py:222-237`): R1 has `E_eq_v=0.68`, `n_electrons=2`, cathodic O₂, anodic H₂O₂, stoichiometry `[-1, +1, -2]` with H⁺ power-2; R2 has `E_eq_v=1.78`, `n_electrons=2`, irreversible.

**Foundational mechanism papers.**
- Wroblowa, Yen-Chi-Pan, Razumney 1976, *J. Electroanal. Chem.* **69**(2), 195-201, DOI 10.1016/S0022-0728(76)80250-1. THE classic 2-step ORR paper.
- Damjanovic, Genshaw, Bockris 1967, *J. Electrochem. Soc.* **114**(5), 466-472. RRDE H₂O₂-on-Pt mechanism.
- Marković, Schmidt, Stamenković, Ross 2001, *Fuel Cells* **1**(2), 105-116, DOI 10.1002/1615-6854(200107)1:2<105::AID-FUCE105>3.0.CO;2-9.
- Marković & Ross 2002, *Surf. Sci. Rep.* **45**(4-6), 117-229.
- Nørskov, Rossmeisl et al. 2004, *J. Phys. Chem. B* **108**(46), 17886-17892, DOI 10.1021/jp047349j.
- Stamenković et al. 2007, *Science* **315**(5811), 493-497, DOI 10.1126/science.1135941.
- Kulkarni, Siahrostami, Patel, Nørskov 2018, *Chem. Rev.* **118**(5), 2302-2312, DOI 10.1021/acs.chemrev.7b00488.

### 4.2 Kinetic Inverse Problems and Tafel-Ridge Identifiability

**Battery / Newman-style PDE-constrained ID.**
- Bizeray, Kim, Duncan, Howey 2017/2018, *IEEE Trans. Control Syst. Technol.* (preprint arXiv:1702.02471). SPM Li-ion has only six identifiable parameter groups.
- Forman, Moura, Stein, Fathy 2012, *J. Power Sources* **210**, 263-275, DOI 10.1016/j.jpowsour.2012.03.009. DFN parameter ID + post-hoc FIM.
- Laue, Röder, Krewer 2021, *J. Appl. Electrochem.* **51**, 1253-1265, DOI 10.1007/s10800-021-01579-5.
- Park, Kato, Gima, Klein, Moura 2018, *J. Electrochem. Soc.* **165**(7), A1309. Convex OED for SPM via FIM.

**Identifiability theory.**
- Raue et al. 2009, *Bioinformatics* **25**(15), 1923-1929, DOI 10.1093/bioinformatics/btp358. Profile likelihood; structural vs practical non-identifiability cleavage.
- Brun, Reichert, Künsch 2001, *Water Resour. Res.* **37**(4), 1015-1030, DOI 10.1029/2000WR900350. Sensitivity index + collinearity index = `1/σ_min(J_normalized)`.
- Quaiser & Mönnigmann 2009, *BMC Syst. Biol.* **3**, 50, DOI 10.1186/1752-0509-3-50. Eigenvalue method.

**Tafel-ridge directly.** In BV Tafel branch, `log|i| ≈ log i₀ + (1−α) f η`: log i₀ enters additively, α as slope; both identifiable. Ridge emerges when (a) η window narrow or (b) mass transport / R2 coupling bends apparent Tafel slope.
- Schalenbach et al. 2024, ACS Energy Lett., DOI 10.1021/acsenergylett.4c00266. Variance in (α, i₀) across groups.
- Anantharaj & Noda 2021, *Sci. Rep.* **11**, 8915, DOI 10.1038/s41598-021-87951-z. Differential Tafel decoupler.
- Bilodeau, Gibson, Garnett 2021, *Nat. Commun.* **12**, 825, DOI 10.1038/s41467-021-20924-y. Bayesian (i₀, α) elongated posterior in CO₂RR.

### 4.3 Charge / Voltage Continuation

- Keller 1977, "Numerical solution of bifurcation and nonlinear eigenvalue problems," in *Applications of Bifurcation Theory* (Rabinowitz, ed.), Academic Press. Pseudo-arclength continuation.
- Allgower & Georg 1990/2003 SIAM Classics, *Numerical Continuation Methods*.
- Doedel et al. AUTO-07P / AUTO-2000.
- Uecker 2021, *Jahresber. DMV* **123**, 199-248, DOI 10.1365/s13291-021-00241-5.
- Markowich, Ringhofer, Schmeiser 1990, *Semiconductor Equations*, Springer, DOI 10.1007/978-3-7091-6961-2. **Textbook origin of bias-ramping for drift-diffusion / PNP.**
- Hyon, Eisenberg, Liu 2010, *Commun. Math. Sci.* **9**(2), 459-475. Energetic-variational PNP with steric.
- Kilic, Bazant, Ajdari 2007, *Phys. Rev. E* **75**, 021502 (I), 021503 (II), DOI 10.1103/PhysRevE.75.021502. Why classical PB/PNP fails > 25 mV.
- Schmuck, Bazant 2015, *SIAM J. Appl. Math.* **75**(3), 1369-1401, DOI 10.1137/140968082.
- EchemFEM (Roy, Lin, Hahn 2024), JOSS / LLNL-JRNL-860653. Most direct contemporary Firedrake-PNP reference.

### 4.4 Adjoint / Discrete-Adjoint in Firedrake

- Farrell, Ham, Funke, Rognes 2013, *SIAM J. Sci. Comput.* **35**(4), C369-C393, DOI 10.1137/120873558. Algorithmic basis of dolfin-adjoint.
- Mitusch, Funke, Dokken 2019, *J. Open Source Softw.* **4**(38), 1292, DOI 10.21105/joss.01292. pyadjoint software paper.
- Plessix 2006, *Geophys. J. Int.* **167**(2), 495-503, DOI 10.1111/j.1365-246X.2006.02978.x. Adjoint-state method review.
- Gunzburger 2003, *Perspectives in Flow Control and Optimization*, SIAM.
- Hinze, Pinnau, Ulbrich, Ulbrich 2009, *Optimization with PDE Constraints*, Springer.
- Ascher & Petzold 1998, *Computer Methods for ODEs and DAEs*, SIAM.

Taylor test verification documented in Farrell-Ham-Funke-Rognes 2013 §5.

### 4.5 Local Fisher Information & Canonical-Ridge Cosines

- Fisher 1922, *Phil. Trans. R. Soc. A* **222**, 309-368, DOI 10.1098/rsta.1922.0009.
- Lehmann & Casella 1998, *Theory of Point Estimation*, 2nd ed., Springer.
- Brun, Reichert, Künsch 2001 (above). Their collinearity index is `1/σ_min(J_normalized)`.
- Quaiser, Mönnigmann 2009 (above). Eigenvalue method = SVD of FIM.
- Pant 2018, *J. R. Soc. Interface* **15**, 20170871, DOI 10.1098/rsif.2017.0871. Most modern v_min(FIM) inspection.
- Hotelling 1936 (Biometrika), canonical correlation analysis.
- van der Vaart 2000, *Asymptotic Statistics* §5.
- Vajda & Rabitz 1989, *J. Phys. Chem.* **93**, 5043. Overlap angles between sensitivity-derived eigenvectors.
- Komorowski, Costa, Rand, Stumpf 2011, *PNAS* **108**(21), 8645-8650, DOI 10.1073/pnas.1015814108.

The canonical-ridge cosine is the user's naming for the dot-product of v_min(FIM) with a chosen analytical Tafel-ridge direction. Pant 2018 is the closest pre-existing diagnostic.

**Joshi, Seidel-Morgenstern, Tiller 2006** — not located in any database. Drop or replace with Brun 2001 + Quaiser 2009 + Pant 2018.

### 4.6 Monolithic vs Splitting PNP Solvers

- Gummel 1964, *IEEE TED* **11**(10), 455-465. Original Gummel iteration.
- Markowich, Ringhofer, Schmeiser 1990 (above), Ch. 7.
- Yu, Holst, McCammon 2007, *J. Comput. Chem.* **28**, 1827-1839. Monolithic PNP, biological geometries.
- Liu & Wang 2014, *J. Comput. Phys.* **268**, 363-376. Energy-stable structure-preserving FD.
- EchemFEM (Roy, Lin, Hahn 2024).

The writeup uses monolithic Newton on the mixed `(u_1, ..., u_n, φ)` space.

---

## 5. Annotated Bibliography (verified)

### 5.1 Log-density / entropy-variable PNP

| Citation | Status | One-sentence summary |
|---|---|---|
| Slotboom 1969, *Electronics Letters* **5**(26), 677-678, DOI 10.1049/el:19690510 | **Verified** | Canonical Slotboom-variables paper; introduces `Φ_n = n·exp(−ψ/V_T)` |
| Slotboom 1973, *IEEE TED* **ED-20**, 669-679 | **Verified** | Bipolar transistor analysis; not the PN-product paper |
| Slotboom 1977, *Solid-State Electron.* **20**(4), 279-283 | **Verified** | Band-gap-narrowing paper, not log-density |
| Scharfetter & Gummel 1969, *IEEE TED* **ED-16**(1), 64-77 | **Verified** | Exponential-fitting / upwind FV on c-primary; NOT log-density substitution |
| Brezzi, Marini, Pietra 1989, *SIAM J. Numer. Anal.* **26**(6), 1342-1355 | **Verified** | Mixed-FE generalization of SG |
| Markowich 1986, *The Stationary Semiconductor Device Equations*, Springer, DOI 10.1007/978-3-7091-3678-2 | **Verified** | Surveys variable choices |
| Markowich, Ringhofer, Schmeiser 1990, *Semiconductor Equations*, Springer | **Verified** | Drift-diffusion + iterative schemes textbook |
| Selberherr 1984, *Analysis and Simulation of Semiconductor Devices*, Springer | **Verified** | Industrial Slotboom + SG + Gummel |
| Metti, Xu, Liu 2016, *J. Comput. Phys.* **306**, 1-18, DOI 10.1016/j.jcp.2015.10.053 | **Verified** | Canonical PNP log-density FEM paper |
| Fu & Xu 2022, *CMAME* **395**, 115031, DOI 10.1016/j.cma.2022.115031, arXiv:2105.01163 | **Verified** | High-order space-time FEM for PNP via entropy variable; **the paper currently mis-cited as "Liu, Maxian, Shen, Zitelli, Waltz, Monteiro 2022"** |
| H. Liu & Maimaitiyiming 2021, *J. Sci. Comput.* **87**(3), 92, DOI 10.1007/s10915-021-01503-1 | **Verified** | FD c-primary positivity + energy stability for multi-D PNP |
| C. Liu, Wang, Wise, Yue, Zhou, arXiv:2009.08076 | **Verified** | H⁻¹ gradient flow with singular log potential |
| Huang & Shen 2021, *SIAM J. Sci. Comput.* **43**(3), B746-B759, DOI 10.1137/20M1365417 | **Verified** | SAV scheme for PNP/Keller-Segel |
| Cancès, Chainais-Hillairet, Gaudeul, Fuhrmann 2022, *Numer. Math.* **151**, 949-986, DOI 10.1007/s00211-022-01279-y | **Verified** | Entropy-FV for PNP with size exclusion |

### 5.2 PBNP hybrid and ion-channel PNP

| Citation | Status | Summary |
|---|---|---|
| Zheng & Wei 2011, *J. Chem. Phys.* **134**(19), 194101, DOI 10.1063/1.3581031, PMID 21599038, PMC PMC3122111 | **Verified** | Canonical PBNP paper |
| Zheng, Chen, Wei 2011, *J. Comput. Phys.* **230**(13), 5239-5262, DOI 10.1016/j.jcp.2011.03.020, PMC PMC3087981 | **Verified** | Second-order full-PNP solver, NOT PBNP |
| Gouy 1910, *J. Phys. Theor. Appl.* **9**, 457-468 | **Verified** | Diffuse-layer model, Boltzmann ion statistics |
| Chapman 1913, *Phil. Mag.* **25**, 475-481 | **Verified** | Closed-form Gouy-Chapman |
| Stern 1924, *Z. Elektrochem.* **30**, 508-516 | **Verified** | Combines Helmholtz + GC |
| Chen, Barcilon, Eisenberg 1992, *Biophys. J.* **61**, 1372-1393 | **Verified** | Open ionic channel PNP |
| Chen, Eisenberg 1993, *Biophys. J.* **64**, 1405-1421, PMID 7686784 | **Verified** | Foundational PNP for one-conformation channels |
| Kurnikova, Coalson, Graf, Nitzan 1999, *Biophys. J.* **76**, 642-656 | **Verified** | 3D PNP for gramicidin A |
| Cardenas, Coalson, Kurnikova 2000, *Biophys. J.* **79**, 80-93, PMC1300917 | **Verified** | 3D PNP gramicidin A |
| Lu, Holst, McCammon, Zhou 2010, *J. Comput. Phys.* **229**, 6979-6994, DOI 10.1016/j.jcp.2010.05.035, PMC2922884 | **Verified** | FEM-PNP for biomolecular diffusion-reaction |
| Newman & Thomas-Alyea 2004, *Electrochemical Systems*, 3rd ed., Wiley, ISBN 0-471-47756-7 | **Verified** | Electroneutrality reference |
| Newman & Balsara 2019, *Electrochemical Systems*, 4th ed., Wiley-ECS | **Verified** | Current edition |
| Newman 1975, *AIChE J.* **21**, 25-41 | **Verified** | Porous-electrode theory |
| Shinozaki et al. 2015, *J. Electrochem. Soc.* **162**, F1144-F1158, DOI 10.1149/2.1071509jes | **Verified** | RDE/ORR HClO₄ benchmarking |

### 5.3 Butler-Volmer, Tafel, Frumkin, generalized BV

| Citation | Status | Summary |
|---|---|---|
| Tafel 1905, *Z. Phys. Chem.* **50U**(1), 641-712, DOI 10.1515/zpch-1905-5043 | **Verified** | Original empirical Tafel law |
| Frumkin 1933, *Z. Phys. Chem. A* **164**, 121-133 | **Verified** | Frumkin correction `η_eff = η − φ_d` |
| Bazant, Thornton, Ajdari 2004, *Phys. Rev. E* **70**, 021506, DOI 10.1103/PhysRevE.70.021506, PMID 15447495 | **Verified** | Diffuse-charge dynamics; PNP+gFBV |
| Kilic, Bazant, Ajdari 2007, *Phys. Rev. E* **75**, 021502/021503, DOI 10.1103/PhysRevE.75.021502 | **Verified** | Steric effects; classical PB fails > 25 mV |
| Bazant, Kilic, Storey, Ajdari 2009, *Adv. Colloid Interface Sci.* **152**, 48-88, DOI 10.1016/j.cis.2009.10.001 | **Verified** | Induced-charge electrokinetics |
| van Soestbergen, Biesheuvel, Bazant 2010, *Phys. Rev. E* **81**, 021503 | **Verified** | PNP + gFBV transient response |
| van Soestbergen 2012, *Russ. J. Electrochem.* **48**, 570-579, DOI 10.1134/S1023193512060110 | **Verified** | Frumkin-BV + mass transfer |
| Bazant 2013, *Acc. Chem. Res.* **46**, 1144-1160, DOI 10.1021/ar300145c, PMID 23520980, arXiv:1208.1587 | **Verified** | Generalized BV via nonequilibrium thermodynamics |
| Bockris, Reddy, Gamboa-Aldeco 2002, *Modern Electrochemistry* **2A**, 2nd ed., Kluwer/Plenum | **Verified (Internet Archive)** | Canonical BV/Tafel textbook |
| Bard, Faulkner, White 2022, *Electrochemical Methods*, 3rd ed., Wiley | **Verified** | Canonical RRDE/Levich/BV textbook |
| Khalili et al. 2016, ASME *J. Electrochem. En. Conv. Stor.* **13**(2), 021003 | **Verified** | Algebraic BV reformulation (inverts for η) |

### 5.4 Closest log-rate-evaluation cousins (different domains)

| Citation | Status | Summary |
|---|---|---|
| **Fattal & Kupferman 2004**, *J. Non-Newtonian Fluid Mech.* **123**(2-3), 281-285, DOI 10.1016/j.jnnfm.2004.08.008 | **Verified** | Log-conformation tensor at high Weissenberg; **closest published structural cousin to log-rate BV** |
| Fattal & Kupferman 2005, *J. Non-Newtonian Fluid Mech.* **126**, 23-37, DOI 10.1016/j.jnnfm.2004.12.003 | **Verified** | Application paper for log-conformation |
| Kee, Rupley, Miller (CHEMKIN-II SAND-89-8009; CHEMKIN-III SAND96-8216) | **Verified** | PLOG (logarithmic interpolation) reactions |
| Lu & Law 2009, *Prog. Energy Combust. Sci.* **35**, 192-215 | **Verified at metadata level** | Stiff-chemistry numerics review |
| Pope 1997, *Combust. Theory Model.* **1**, 41-63 | **Verified at metadata level** | Tabulates log of rates |
| Higham et al. 2021, *IMA J. Numer. Anal.* **41**, 2311-2330, DOI 10.1093/imanum/draa038 | **Verified** | Log-sum-exp / softmax numerical analysis |

### 5.5 ORR mechanism

| Citation | Status |
|---|---|
| Wroblowa, Yen-Chi-Pan, Razumney 1976, *J. Electroanal. Chem. Interfacial Electrochem.* **69**(2), 195-201, DOI 10.1016/S0022-0728(76)80250-1 | **Verified** |
| Damjanovic, Genshaw, Bockris 1967, *J. Electrochem. Soc.* **114**(5), 466-472 | **Verified at citation level** |
| Marković, Schmidt, Stamenković, Ross 2001, *Fuel Cells* **1**(2), 105-116 | **Verified** |
| Marković & Ross 2002, *Surf. Sci. Rep.* **45**(4-6), 117-229 | **Verified** |
| Nørskov, Rossmeisl et al. 2004, *J. Phys. Chem. B* **108**(46), 17886-17892, DOI 10.1021/jp047349j | **Verified** |
| Stamenković et al. 2007, *Science* **315**(5811), 493-497, DOI 10.1126/science.1135941 | **Verified** |
| Kulkarni, Siahrostami, Patel, Nørskov 2018, *Chem. Rev.* **118**(5), 2302-2312, DOI 10.1021/acs.chemrev.7b00488 | **Verified** |

### 5.6 Inverse problems / identifiability / FIM

| Citation | Status |
|---|---|
| Fisher 1922, *Phil. Trans. R. Soc. A* **222**, 309-368, DOI 10.1098/rsta.1922.0009 | **Verified** |
| Lehmann & Casella 1998, *Theory of Point Estimation*, 2nd ed., Springer | textbook |
| Brun, Reichert, Künsch 2001, *Water Resour. Res.* **37**(4), 1015-1030, DOI 10.1029/2000WR900350 | **Verified** |
| Quaiser & Mönnigmann 2009, *BMC Syst. Biol.* **3**, 50, DOI 10.1186/1752-0509-3-50 | **Verified** |
| Raue et al. 2009, *Bioinformatics* **25**(15), 1923-1929, DOI 10.1093/bioinformatics/btp358 | **Verified** |
| Pant 2018, *J. R. Soc. Interface* **15**, 20170871, DOI 10.1098/rsif.2017.0871 | **Verified** |
| Komorowski, Costa, Rand, Stumpf 2011, *PNAS* **108**(21), 8645-8650, DOI 10.1073/pnas.1015814108 | **Verified** |
| Vajda & Rabitz 1989, *J. Phys. Chem.* **93**, 5043 | **Verified at citation level** |
| Bizeray, Kim, Duncan, Howey 2018, *IEEE Trans. Control Syst. Technol.* (arXiv:1702.02471) | **Verified** |
| Forman, Moura, Stein, Fathy 2012, *J. Power Sources* **210**, 263-275, DOI 10.1016/j.jpowsour.2012.03.009 | **Verified** |
| Laue, Röder, Krewer 2021, *J. Appl. Electrochem.* **51**, 1253-1265, DOI 10.1007/s10800-021-01579-5 | **Verified** |
| Park, Kato, Gima, Klein, Moura 2018, *J. Electrochem. Soc.* **165**(7), A1309 | **Verified** |
| Bilodeau, Gibson, Garnett 2021, *Nat. Commun.* **12**, 825, DOI 10.1038/s41467-021-20924-y | **Verified** |
| Anantharaj & Noda 2021, *Sci. Rep.* **11**, 8915, DOI 10.1038/s41598-021-87951-z | **Verified** |
| Schalenbach et al. 2024, ACS Energy Lett., DOI 10.1021/acsenergylett.4c00266 | **Verified** |

### 5.7 Continuation and adjoint

| Citation | Status |
|---|---|
| Keller 1977, pseudo-arclength continuation chapter | book chapter |
| Allgower & Georg 1990/2003, *Numerical Continuation Methods*, Springer / SIAM | textbook |
| Doedel et al. 2007 AUTO-07P manual | Concordia listing |
| Uecker 2021, *Jahresber. DMV* **123**, 199-248, DOI 10.1365/s13291-021-00241-5 | **Verified** |
| Hyon, Eisenberg, Liu 2010, *Commun. Math. Sci.* **9**(2), 459-475 | **Verified** |
| Schmuck, Bazant 2015, *SIAM J. Appl. Math.* **75**(3), 1369-1401, DOI 10.1137/140968082 | **Verified** |
| Roy, Lin, Hahn 2024 EchemFEM, JOSS / LLNL-JRNL-860653 | **Verified** |
| Farrell, Ham, Funke, Rognes 2013, *SIAM J. Sci. Comput.* **35**(4), C369-C393, DOI 10.1137/120873558 | **Verified** |
| Mitusch, Funke, Dokken 2019, *J. Open Source Softw.* **4**(38), 1292, DOI 10.21105/joss.01292 | **Verified** |
| Plessix 2006, *Geophys. J. Int.* **167**(2), 495-503, DOI 10.1111/j.1365-246X.2006.02978.x | **Verified** |
| Gunzburger 2003, *Perspectives in Flow Control and Optimization*, SIAM | textbook |
| Hinze, Pinnau, Ulbrich, Ulbrich 2009, *Optimization with PDE Constraints*, Springer | textbook |
| Ascher & Petzold 1998, *Computer Methods for ODEs and DAEs*, SIAM | textbook |

### 5.8 Monolithic vs splitting PNP

| Citation | Status |
|---|---|
| Gummel 1964, *IEEE TED* **11**(10), 455-465 | original |
| Yu, Holst, McCammon 2007, *J. Comput. Chem.* **28**, 1827-1839 | monolithic PNP, biological |
| Liu & Wang 2014, *J. Comput. Phys.* **268**, 363-376 | energy-stable FD |

### 5.9 NOT located / verification failed

- **"Liu, Maxian, Shen, Zitelli, Waltz, Monteiro 2022, CMAME 396, 115070, arXiv:2105.01163"** — **FABRICATED**. arXiv:2105.01163 resolves to Fu & Xu 2022, CMAME 395, 115031. Replace.
- **"Joshi, Seidel-Morgenstern, Tiller 2006"** — Not located in any database. If used in writeup, drop or replace with Brun-Reichert-Künsch 2001 + Quaiser-Mönnigmann 2009 + Pant 2018.

---

## 6. Recommendations for the LaTeX Writeup's `\thebibliography`

### 6.1 Critical corrections (blocking publication)

**Correction 1 — Remove the fabricated CMAME 2022 reference, replace with Fu & Xu 2022.**

```latex
\bibitem{FuXu2022}
G.~Fu and Z.~Xu,
``High-order space-time finite element methods for the
Poisson--Nernst--Planck equations: positivity and unconditional energy
stability,''
{\em Computer Methods in Applied Mechanics and Engineering}, vol.~395,
art.~115031, 2022.
\href{https://doi.org/10.1016/j.cma.2022.115031}{doi:10.1016/j.cma.2022.115031}.
arXiv:2105.01163.
```

**Correction 2 — Disambiguate the Slotboom citation.**

```latex
\bibitem{Slotboom1969}
J.~W. Slotboom,
``Iterative scheme for 1- and 2-dimensional d.c.-transistor simulation,''
{\em Electronics Letters}, vol.~5, no.~26, pp.~677--678, 1969.
\href{https://doi.org/10.1049/el:19690510}{doi:10.1049/el:19690510}.
```

Add a one-sentence disambiguation in methods text: "Slotboom variables and the log-density substitution are related by a logarithm and additive shift — `ln(Φ_n) = ln(n) − ψ/V_T` — so they share continuous structure but yield distinct Galerkin discretizations."

**Correction 3 — Disambiguate Zheng-Wei vs Zheng-Chen-Wei.**

```latex
\bibitem{ZhengWei2011}
Q.~Zheng and G.-W. Wei,
``Poisson--Boltzmann--Nernst--Planck model,''
{\em The Journal of Chemical Physics}, vol.~134, no.~19, art.~194101,
12~pp., 2011.
\href{https://doi.org/10.1063/1.3581031}{doi:10.1063/1.3581031}.
PMID:~21599038.

\bibitem{ZhengChenWei2011}
Q.~Zheng, D.~Chen, and G.-W. Wei,
``Second-order Poisson--Nernst--Planck solver for ion transport,''
{\em Journal of Computational Physics}, vol.~230, no.~13, pp.~5239--5262, 2011.
\href{https://doi.org/10.1016/j.jcp.2011.03.020}{doi:10.1016/j.jcp.2011.03.020}.
```

Do **not** cite Zheng-Chen-Wei 2011 as the PBNP source.

### 6.2 Recommended additions

```latex
\bibitem{MettiXuLiu2016}
M.~S. Metti, J.~Xu, and C.~Liu,
``Energetically stable discretizations for charge transport and
electrokinetic models,''
{\em Journal of Computational Physics}, vol.~306, pp.~1--18, 2016.
\href{https://doi.org/10.1016/j.jcp.2015.10.053}{doi:10.1016/j.jcp.2015.10.053}.

\bibitem{CancesEtAl2022}
C.~Canc\`es, C.~Chainais-Hillairet, B.~Gaudeul, and J.~Fuhrmann,
``Entropy and convergence analysis for two finite-volume schemes for a
Nernst--Planck--Poisson system with ion-volume constraints,''
{\em Numerische Mathematik}, vol.~151, pp.~949--986, 2022.
\href{https://doi.org/10.1007/s00211-022-01279-y}{doi:10.1007/s00211-022-01279-y}.

\bibitem{Tafel1905}
J.~Tafel,
``\"Uber die Polarisation bei kathodischer Wasserstoffentwicklung,''
{\em Zeitschrift f\"ur Physikalische Chemie}, vol.~50U, no.~1, pp.~641--712, 1905.
\href{https://doi.org/10.1515/zpch-1905-5043}{doi:10.1515/zpch-1905-5043}.

\bibitem{FattalKupferman2004}
R.~Fattal and R.~Kupferman,
``Constitutive laws for the matrix-logarithm of the conformation tensor,''
{\em Journal of Non-Newtonian Fluid Mechanics}, vol.~123, no.~2--3,
pp.~281--285, 2004.
\href{https://doi.org/10.1016/j.jnnfm.2004.08.008}{doi:10.1016/j.jnnfm.2004.08.008}.

\bibitem{LuHolstMcCammonZhou2010}
B.~Lu, M.~J. Holst, J.~A. McCammon, and Y.~C. Zhou,
``Poisson--Nernst--Planck equations for simulating biomolecular
diffusion-reaction processes I: Finite element solutions,''
{\em Journal of Computational Physics}, vol.~229, pp.~6979--6994, 2010.
\href{https://doi.org/10.1016/j.jcp.2010.05.035}{doi:10.1016/j.jcp.2010.05.035}.

\bibitem{Wroblowa1976}
H.~Wroblowa, Yen-Chi-Pan, and G.~Razumney,
``Electroreduction of oxygen: a new mechanistic criterion,''
{\em Journal of Electroanalytical Chemistry and Interfacial
Electrochemistry}, vol.~69, no.~2, pp.~195--201, 1976.
\href{https://doi.org/10.1016/S0022-0728(76)80250-1}{doi:10.1016/S0022-0728(76)80250-1}.

\bibitem{KulkarniSiahrostami2018}
A.~Kulkarni, S.~Siahrostami, A.~Patel, and J.~K. N{\o}rskov,
``Understanding catalytic activity trends in the oxygen reduction reaction,''
{\em Chemical Reviews}, vol.~118, no.~5, pp.~2302--2312, 2018.
\href{https://doi.org/10.1021/acs.chemrev.7b00488}{doi:10.1021/acs.chemrev.7b00488}.

\bibitem{FarrellHamFunkeRognes2013}
P.~E. Farrell, D.~A. Ham, S.~W. Funke, and M.~E. Rognes,
``Automated derivation of the adjoint of high-level transient finite
element programs,''
{\em SIAM Journal on Scientific Computing}, vol.~35, no.~4, pp.~C369--C393, 2013.
\href{https://doi.org/10.1137/120873558}{doi:10.1137/120873558}.

\bibitem{MituschFunkeDokken2019}
S.~K. Mitusch, S.~W. Funke, and J.~S. Dokken,
``dolfin-adjoint 2018.1: automated adjoints for FEniCS and Firedrake,''
{\em Journal of Open Source Software}, vol.~4, no.~38, p.~1292, 2019.
\href{https://doi.org/10.21105/joss.01292}{doi:10.21105/joss.01292}.

\bibitem{MarkowichRinghoferSchmeiser1990}
P.~A. Markowich, C.~A. Ringhofer, and C.~Schmeiser,
{\em Semiconductor Equations}.
Wien: Springer, 1990.
\href{https://doi.org/10.1007/978-3-7091-6961-2}{doi:10.1007/978-3-7091-6961-2}.

\bibitem{KilicBazantAjdari2007a}
M.~S. Kilic, M.~Z. Bazant, and A.~Ajdari,
``Steric effects in the dynamics of electrolytes at large applied voltages.
I. Double-layer charging,''
{\em Physical Review E}, vol.~75, art.~021502, 2007.
\href{https://doi.org/10.1103/PhysRevE.75.021502}{doi:10.1103/PhysRevE.75.021502}.

\bibitem{BrunReichertKunsch2001}
R.~Brun, P.~Reichert, and H.~R. K\"unsch,
``Practical identifiability analysis of large environmental simulation models,''
{\em Water Resources Research}, vol.~37, no.~4, pp.~1015--1030, 2001.
\href{https://doi.org/10.1029/2000WR900350}{doi:10.1029/2000WR900350}.

\bibitem{Pant2018}
S.~Pant,
``Information sensitivity functions to assess parameter information
gain and identifiability of dynamical systems,''
{\em Journal of the Royal Society Interface}, vol.~15, art.~20170871, 2018.
\href{https://doi.org/10.1098/rsif.2017.0871}{doi:10.1098/rsif.2017.0871}.
```

### 6.3 Recommended deletions

- The fabricated **Liu-Maxian-Shen-Zitelli-Waltz-Monteiro 2022** entry must be removed.
- If the writeup cites **Joshi, Seidel-Morgenstern, Tiller 2006**, drop or replace with Brun 2001 + Quaiser 2009 + Pant 2018.
- The misattributed Slotboom citation as printed (chimera 1973+1977 conflation) must be replaced with Slotboom 1969.
- Reassign Zheng-Chen-Wei 2011 JCP from PBNP-source role to full-PNP-numerical-methods role.

### 6.4 Code-pointer corrections (writeup prose, not bibliography)

1. **Line range for the log-rate branch**: change `forms_logc.py:294-360` to `forms_logc.py:294-388`.
2. **Cathodic-vs-anodic algebra description**: tighten "both branches subtracted with `Σ_f ν_f (u_sp(f) − ln c_ref,f)`" to specify "the cathodic branch carries the full stoichiometric-power loop (`forms_logc.py:331-337`); the anodic branch is single-species (lines 343-345 and fallback 350-353). For the production setup R2 is irreversible (`v24_3sp_logc_vs_4sp_validation.py:233`), so its anodic branch is zero, and the algebraic asymmetry has no production-output consequence."
3. **Architectural note on `add_boltzmann()`**: the writeup's pointer "see `add_boltzmann()` in the current study scripts" is technically correct but architecturally fragile (function is monkey-patched in 7 study scripts). Tightening recommendation: future work should migrate the Boltzmann contribution into `Forward/bv_solver/forms_logc.build_forms_logc()` as an option.

---

# Synthesis Summary

**Document length.** ~700 lines as rendered above (executive summary 60 lines, per-change provenance ~250 lines, deviation analysis ~140 lines, surrounding methods ~150 lines, annotated bibliography ~120 lines structured as verification tables, recommendations ~80 lines including ready-to-paste BibTeX).

**Most important corrections to the writeup's existing bibliography.**
1. The **"Liu-Maxian-Shen-Zitelli-Waltz-Monteiro 2022, CMAME 396, 115070" entry is FABRICATED**. The arXiv ID 2105.01163 resolves to Fu & Xu 2022, CMAME 395, 115031, doi:10.1016/j.cma.2022.115031 — two authors not six, vol. 395 not 396, article 115031 not 115070. Replace.
2. The **Slotboom citation as printed is a chimera** conflating 1969/1973/1977. The canonical Slotboom-variables reference is Slotboom 1969, *Electronics Letters* 5(26), 677-678, doi:10.1049/el:19690510. Slotboom variables and log-density are related but distinct substitutions; do not equate them.
3. The **Zheng-Chen-Wei JCP 230 (2011) cited as PBNP source is the wrong paper** — that is the full-PNP numerical methods paper. The canonical PBNP paper is Zheng & Wei, *J. Chem. Phys.* 134, 194101 (2011), doi:10.1063/1.3581031.
4. **Add Tafel 1905** with corrected DOI 10.1515/zpch-1905-5043, and explicitly distinguish Tafel (high-η physical approximation that drops a branch) from log-rate BV (exact algebraic identity that drops nothing).
5. **Add Fattal-Kupferman 2004** as the closest published structural cousin to log-rate BV.

**Strongest defensible novelty claims.**
1. **Log-density `u = ln c` as primary Galerkin unknown in an ORR-BV PNP solver.** No surveyed peer-reviewed ORR/RDE/fuel-cell PNP-BV solver does this; the technique is borrowed from semiconductor (Slotboom 1969) and mathematical-PNP (Metti-Xu-Liu 2016, Fu-Xu 2022) literature.
2. **Log-rate BV evaluation in a published PNP-BV solver** — small local numerical novelty. The construction is mathematically trivial, but no electrochemistry paper publishes it. Closest published cousins live in viscoelastic fluids (Fattal-Kupferman 2004) and stiff combustion (CHEMKIN PLOG / log-sum-exp).
3. **PBNP applied to ORR with explicit Boltzmann-ClO₄⁻** — modest reuse of Zheng-Wei 2011 in a new application domain.
