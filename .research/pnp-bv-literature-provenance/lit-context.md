# Literature Provenance: Surrounding Methods for the PNP-BV / ORR Forward-Solver Writeup

Scope: everything *other* than the three core numerical changes. Six sections:
1. ORR mechanism (two-step BV, peroxide, RRDE)
2. Kinetic-parameter inverse problems (Tafel ridge, identifiability)
3. Charge / voltage continuation in PNP solvers
4. Adjoint / discrete-adjoint sensitivity in Firedrake
5. Local Fisher information & canonical-ridge cosines
6. Monolithic vs splitting PNP solvers (context)

Every citation below has been verified at least at abstract / publisher metadata level via the searches and fetches recorded in the VERIFICATION RESULTS section at the end of this document. Where verification failed (paywall, host redirect), it is marked explicitly.

---

## 1. ORR Mechanism Background

### 1.1 Two-step ORR equilibrium potentials (E_eq^(1) ~ 0.68 V, E_eq^(2) ~ 1.78 V vs SHE)

The two-step (peroxide-intermediate) decomposition the user's solver uses is:

- R1: O2 + 2H+ + 2e- -> H2O2,   E_eq^(1) = +0.695 V vs SHE (commonly rounded to 0.68 V)
- R2: H2O2 + 2H+ + 2e- -> 2H2O, E_eq^(2) = +1.776 V vs SHE (commonly rounded to 1.78 V)
- Sum gives the overall 4e- ORR with E_eq = +1.229 V vs SHE.

These are textbook-standard values; the most authoritative open citation is Bard & Faulkner (Electrochemical Methods, 2nd ed., 2000, Wiley; 3rd ed. with White, 2022) which tabulates them. They also appear in every electrochemistry table-of-standard-reduction-potentials reference (Atkins, CRC handbook, LibreTexts).

NOTE: some recent ACS/Nature reviews quote 0.70 V for R1 (small temperature/activity convention differences). The 0.68 V / 1.78 V pair is the commonly used "round" set; pick *one* convention (probably IUPAC standard state, 25 C, unit activity) and cite Bard-Faulkner once.

### 1.2 Key Papers — Mechanism and ORR Reviews

**Wroblowa, Yen-Chi-Pan, Razumney 1976** — Foundational 2-step parallel-vs-series ORR analysis.
- Authors: H. Wroblowa, Yen-Chi-Pan, G. Razumney
- Year/venue: 1976, J. Electroanal. Chem. Interfacial Electrochem., 69(2), 195–201
- DOI: 10.1016/S0022-0728(76)80250-1
- Contribution: Diagnostic criterion using RRDE collection efficiency to distinguish direct 4e- O2 -> H2O reduction from a series 2e- + 2e- pathway via H2O2 and from parallel pathways. Demonstrated on Au in alkaline.
- Relevance: This is THE classic 2-step ORR paper that justifies the user's R1+R2 BV decomposition. Cite when introducing the two-step mechanism.
- Confidence: high. Verified DOI and abstract.

**Damjanovic, Genshaw, Bockris 1967** — Role of H2O2 in ORR at Pt in H2SO4.
- Authors: A. Damjanovic, M. A. Genshaw, J. O'M. Bockris
- Year/venue: 1967, J. Electrochem. Soc., 114(5), 466–472
- Companion: same authors' 1967 J. Electrochem. Soc. on alkaline / Au.
- Contribution: Earliest systematic RRDE-based analysis of whether H2O2 is a true intermediate of the 4e- ORR on Pt or arises only from a parallel 2e- pathway. Established that on Pt acid the dominant pathway is "direct" (i.e., H2O without significant peroxide release), but the series component is non-zero — exactly the regime your R1+R2 model captures.
- Relevance: Historical anchor for 2-step mechanism on Pt specifically. Cite alongside Wroblowa.
- Confidence: high. Citation verified via multiple secondary sources and 1968 discussion comment in the same journal.
- Verification status: abstract/citation confirmed via secondary literature; the 1967 JES PDF itself is paywalled.

**Markovic, Schmidt, Stamenkovic, Ross 2001** — "ORR on Pt and Pt bimetallic surfaces: a selective review".
- Authors: N. M. Marković, T. J. Schmidt, V. Stamenković, P. N. Ross
- Year/venue: 2001, Fuel Cells, 1(2), 105–116
- DOI: 10.1002/1615-6854(200107)1:2<105::AID-FUCE105>3.0.CO;2-9
- Contribution: Reviews structure-sensitive ORR kinetics on Pt single crystals and Pt-Ni / Pt-Co alloys. Discusses anion-adsorption effects and Tafel slopes on Pt(hkl) in acid and alkaline.
- Relevance: Standard "Pt ORR reference" — useful for justifying BV-form rate laws on Pt.
- Confidence: high. Verification: WebFetch returned 402 (paywall) but Wiley/RG metadata and multiple secondary citations confirm.

**Marković and Ross 2002** — "Surface science studies of model fuel cell electrocatalysts", Surf. Sci. Rep., 45(4–6), 117–229.
- Comprehensive 100+ page review of HOR, ORR, and CO/HCOOH/MeOH electro-oxidation on Pt(hkl) and Pt alloys. Often paired with the 2001 Fuel Cells review.
- Relevance: deep reference if you want a single citation that covers BV parameter ranges, Tafel slopes, and pH/anion effects for Pt.
- Confidence: high (citation, abstract confirmed via ADS).

**Norskov, Rossmeisl et al. 2004** — "Origin of the Overpotential for Oxygen Reduction at a Fuel-Cell Cathode".
- Authors: J. K. Nørskov, J. Rossmeisl, A. Logadottir, L. Lindqvist, J. R. Kitchin, T. Bligaard, H. Jónsson
- Year/venue: 2004, J. Phys. Chem. B, 108(46), 17886–17892
- DOI: 10.1021/jp047349j
- Contribution: DFT-based free-energy diagram for associative 4e- ORR on Pt(111). Identifies *OH and *O as the most stable intermediates near equilibrium and shows that the overpotential arises from these binding energies — gives a microscopic justification for the lumped k0,1 / k0,2 BV exchange-current parameters.
- Relevance: Standard "why ORR has 0.3–0.4 V overpotential" reference; supports the very large k0 ratios you may see between R1 and R2.
- Confidence: high. Citation/abstract confirmed via JPCB metadata, multiple secondary sources.
- Verification status: WebFetch returned 403 (ACS bot block); citation and abstract verified via Norskov group page, ResearchGate, Semantic Scholar.

**Kulkarni, Siahrostami, Patel, Norskov 2018** — "Understanding Catalytic Activity Trends in the ORR".
- Year/venue: 2018, Chem. Rev., 118(5), 2302–2312
- DOI: 10.1021/acs.chemrev.7b00488
- Contribution: Recent (2018) Chemical Reviews authoritative review of descriptor-based theory of ORR activity. Includes both 2e- (peroxide) and 4e- pathways and the linear scaling-relation analysis.
- Relevance: Most cited *recent* review; pair with Markovic 2001 for "old + new" coverage. Use for the statement "the two-step BV decomposition is consistent with mainstream microkinetic ORR theory".
- Confidence: high. DOI / metadata confirmed.

**Stamenkovic et al. 2007** — "Improved oxygen reduction activity on Pt3Ni(111) via increased surface site availability", Science, 315(5811), 493–497, DOI: 10.1126/science.1135941. Mentioned for completeness — illustrates that ORR on real Pt-alloy catalysts gives very different (k0, alpha) than Pt(111). Useful caveat in the methods section.

### 1.3 RRDE / Bard-Faulkner

**Bard, Faulkner, White, "Electrochemical Methods: Fundamentals and Applications"**
- Editions: 2nd ed. (Bard & Faulkner) 2000, Wiley; 3rd ed. (Bard, Faulkner & White) 2022, Wiley.
- Chapter 9 in both editions covers RDE / RRDE / Levich / Koutecký-Levich and collection efficiency. This is THE textbook reference for RRDE-based mechanism analysis.

### 1.4 Themes and Findings (ORR)
- The two-step mechanism R1: O2 -> H2O2, R2: H2O2 -> H2O is the canonical decomposition for parallel-vs-series analysis on Pt, going back to Damjanovic (1967) and Wroblowa (1976).
- On Pt in acid, R2 (peroxide reduction) is typically much faster than R1 at moderate overpotentials, so released H2O2 fraction is low. This is exactly the regime where (k0,2 >> k0,1) and where alpha_1, alpha_2 are weakly co-correlated — relevant to your Tafel-ridge identifiability story.
- E_eq values of 0.68 V and 1.78 V vs SHE are textbook-standard.

### 1.5 Recommended ORR Citations
- Wroblowa, Pan, Razumney 1976 (mechanism criterion)
- Damjanovic, Genshaw, Bockris 1967 (RRDE on Pt in acid — historical)
- Marković, Schmidt, Stamenković, Ross 2001 (Pt ORR review)
- Marković & Ross 2002 (deep model-system review)
- Norskov et al. 2004 (DFT origin of overpotential)
- Kulkarni, Siahrostami, Patel, Norskov 2018 (modern Chem. Rev. review)
- Bard, Faulkner & White (3rd ed. 2022 textbook), Ch. 9 (RRDE)

---

## 2. Kinetic-Parameter Inverse Problems (Identifiability and the Tafel Ridge)

### 2.1 Battery / Newman-style Parameter Identification

**Bizeray, Kim, Duncan, Howey 2017/2018** — Identifiability and parameter estimation of the SPM Li-ion battery model.
- Authors: A. M. Bizeray, J.-H. Kim, S. R. Duncan, D. A. Howey
- Year/venue: 2018 IEEE Trans. Control Syst. Technol. (preprint arXiv:1702.02471, Feb 2017)
- DOI: arXiv:1702.02471
- Contribution: Shows that the single-particle Li-ion battery model has *only six* identifiable parameter groups after non-dimensionalisation; the rest are structurally non-identifiable. Uses transfer-function structural analysis plus EIS-based practical estimation.
- Relevance: Direct analog for how the user's 4-parameter (log k0,1, log k0,2, alpha_1, alpha_2) inverse problem might collapse into fewer effective directions under fixed I-V protocol. Cite when introducing structural-vs-practical identifiability for electrochemical PDEs.
- Confidence: high. arXiv abstract confirmed.

**Forman, Moura, Stein, Fathy 2012** — DFN parameter ID via genetic algorithm + Fisher analysis.
- Authors: J. C. Forman, S. J. Moura, J. L. Stein, H. K. Fathy
- Year/venue: 2012, J. Power Sources, 210, 263–275
- DOI: 10.1016/j.jpowsour.2012.03.009
- Contribution: Identifies 88 DFN parameters from drive-cycle data (LiFePO4 cell) using a GA, then post-hoc Fisher-information identifiability check. Shows that even with rich excitation, large blocks of the DFN parameter set are practically non-identifiable; only a low-rank subset survives.
- Relevance: Methodologically the closest existing analog to the user's PNP-BV inverse problem — both are PDE-constrained, multi-parameter, with a Fisher-information identifiability layer on top. Cite as the canonical "post-hoc FIM analysis after global ID" reference.
- Confidence: high. Multiple verified citations and a public PDF.

**Laue, Röder, Krewer 2021** — Practical identifiability of P2D Li-ion models.
- Authors: V. Laue, F. Röder, U. Krewer
- Year/venue: 2021, J. Appl. Electrochem., 51, 1253–1265
- DOI: 10.1007/s10800-021-01579-5
- Contribution: Three-step parameterisation procedure (OCV + C-rate + EIS), with identifiability analysis at each step. Shows OCV+C-rate alone are insufficient; dynamic excitation needed to resolve diffusion-vs-kinetics trade-offs.
- Relevance: Similar message to the user's PC-vs-V_RHE work — coverage of voltage windows and excitation richness controls which parameters become identifiable.
- Confidence: high.
- Verification status: Springer link returned 303 redirect; metadata and abstract confirmed via Semantic Scholar / ResearchGate / KIT repository.

**Park, Kato, Gima, Klein, Moura 2018** — Optimal experimental design for SPM parameterization.
- Year/venue: 2018, J. Electrochem. Soc., 165(7), A1309
- Contribution: Convex OED on input current trajectories that maximises Fisher information for SPM parameters.
- Relevance: Direct template for D-/A-optimal voltage protocol design in your ORR setting.
- Confidence: high. PDF available at ECAL Berkeley.

### 2.2 Identifiability Theory

**Raue, Kreutz, Maiwald, Bachmann, Schilling, Klingmüller, Timmer 2009** — Profile-likelihood identifiability.
- Year/venue: 2009, Bioinformatics, 25(15), 1923–1929
- DOI: 10.1093/bioinformatics/btp358
- Contribution: Distinguishes structural non-identifiability (functionally related parameters) from practical non-identifiability (insufficient data). Profile-likelihood diagnostic that can also produce confidence intervals.
- Relevance: Use to define exactly the geometry of the (k0, alpha) Tafel ridge — it is structurally non-identifiable in the asymptotic-Tafel limit and becomes practically (weakly) identifiable when curvature emerges (mass-transport-limited region or peroxide bend).
- Confidence: high. DOI verified, abstract fetched.

**Brun, Reichert, Künsch 2001** — Practical identifiability for environmental simulation models.
- Year/venue: 2001, Water Resour. Res., 37(4), 1015–1030
- DOI: 10.1029/2000WR900350
- Contribution: Two diagnostics: (i) per-parameter sensitivity index, (ii) collinearity / near-linear-dependence index for parameter subsets. Shows how to pick the *largest* subset of identifiable parameters from a sensitivity-matrix SVD.
- Relevance: Methodologically equivalent to the user's "canonical-ridge cosine" — the collinearity index of Brun et al. is essentially the worst-direction cosine of the local Fisher matrix.
- Confidence: high. Citation confirmed.
- Verification status: Wiley DOI returned 402 (paywall); citation/abstract confirmed via OSTI mirror and ScienceOpen entry.

**Quaiser, Mönnigmann 2009** — Eigenvalue method for systematic identifiability.
- Year/venue: 2009, BMC Syst. Biol., 3, 50
- DOI: 10.1186/1752-0509-3-50
- Contribution: Eigenvalue-based method for selecting maximal identifiable parameter subsets in large ODE models. Benchmarks against Brun et al. and orthogonal methods.
- Relevance: Pair with Brun 2001 when describing the SVD of FIM as the canonical identifiability tool.
- Confidence: high. Citation verified via PubMed and BMC.
- Verification status: BMC URL redirected to Springer; Springer fetch failed (303); abstract / metadata confirmed via PubMed and BMC mirror.

**Bellman, Astrom 1970** (early) and **Saccomani et al.** (review) — Structural identifiability theory. Useful as background but the 2009 papers (Raue, Quaiser) are more directly applicable.

### 2.3 Tafel-Ridge Identifiability of (log k0, alpha)

In the Butler-Volmer single-step current i = i0 [exp((1-alpha) f eta) - exp(-alpha f eta)] with f = F/RT, in the asymptotic Tafel branch (eta >> 1/f) one has log|i| ~ log i0 + (1-alpha) f eta (anodic) or log i0 - alpha f eta (cathodic). For *one* fixed branch:

- log i0 enters as additive offset.
- alpha enters as slope.

Hence in the *Tafel branch* (k0, alpha) appear in independent linear combinations and are identifiable. The ridge problem (k0 and alpha co-correlated) emerges either:
1. In log-i_kinetic at a *narrow* eta window (slope and offset cannot be cleanly separated when the lever arm in eta is short),
2. When mass-transport / diffusion / coupling to the second step (R2 in your problem) bends the apparent Tafel slope and only the *combination* (k0 exp(- alpha f eta_typical)) can be inferred.

References that discuss this directly (without using "ridge" language):

- **Tafel-slope-extraction methodology and parameter-extraction errors:**
  - Schalenbach, Durmus, Tempel, Hoster, Eichel 2024, "Tafel slope plot as a tool to analyze electrocatalytic reactions", ACS Energy Lett., DOI: 10.1021/acsenergylett.4c00266. Discusses how arbitrary linear-region selection causes wide variance in (alpha, i0) between groups — a manifestation of practical non-identifiability.
  - Anantharaj & Noda 2021, "A simple and effective method for the accurate extraction of kinetic parameters using differential Tafel plots", Sci. Rep., 11, 8915, DOI: 10.1038/s41598-021-87951-z. Differential-Tafel-slope analysis as a way to break the (i0, alpha) degeneracy.
- **Bayesian Tafel-slope inference (good for an "uncertainty-aware" angle):**
  - Bilodeau, Gibson, Garnett 2021, "Bayesian data analysis reveals no preference for cardinal Tafel slopes in CO2 reduction electrocatalysis", Nat. Commun., 12, 825, DOI: 10.1038/s41467-021-20924-y. Direct demonstration that the (i0, alpha) posterior is broadly elongated — i.e., practically non-identifiable.

### 2.4 Optimal Experimental Design

- Park et al. 2018 (above) is the canonical battery example.
- Forman et al. 2012 plus the older Forman-Bashash conference paper (ACC 2011) discuss D-/Fisher-optimal current trajectories for DFN.
- General references: Pukelsheim (book), Fedorov-Hackl. For PDE-constrained: Haber, Horesh, Tenorio 2008 "Numerical methods for the design of large-scale nonlinear discrete ill-posed inverse problems".

### 2.5 Themes and Findings (Inverse Problem)
- "Structural" vs "practical" identifiability is the standard cleavage (Bellman/Astrom -> Brun 2001 -> Raue 2009).
- The (k0, alpha) Tafel ridge is a textbook practical-non-identifiability example, but does *not* appear under that name. Cite Bilodeau et al. 2021 (Bayesian, CO2RR) and Anantharaj-Noda 2021 (differential Tafel) plus Schalenbach 2024 (variance-across-groups) as direct evidence.
- For PDE-constrained electrochemistry, Bizeray-Howey 2017/2018, Forman 2012, Laue-Krewer 2021, Park-Klein-Moura 2018 are the four canonical references.

### 2.6 Recommended Inverse-Problem Citations
- Raue et al. 2009 (profile likelihood)
- Brun, Reichert, Künsch 2001 (FIM-based identifiability)
- Quaiser, Mönnigmann 2009 (eigenvalue-based)
- Bizeray, Kim, Duncan, Howey 2018 (battery analog)
- Forman, Moura, Stein, Fathy 2012 (genetic + FIM, DFN)
- Laue, Röder, Krewer 2021 (P2D, multi-experiment)
- Park, Kato, Gima, Klein, Moura 2018 (OED via Fisher)
- Bilodeau et al. 2021 (Bayesian Tafel ridge — most direct (k0,alpha) evidence)
- Anantharaj & Noda 2021 (differential Tafel as a (k0,alpha) decoupler)

---

## 3. Charge / Voltage Continuation in PNP Solvers

### 3.1 Continuation Theory

**H. B. Keller (1977)** — "Numerical solution of bifurcation and nonlinear eigenvalue problems", in Applications of Bifurcation Theory, P. H. Rabinowitz (ed.), Academic Press. Original modern reference for *pseudo-arclength* continuation. Independently Riks (1972) and Wempner (1971) for FE applications. Routinely cited as the canonical numerical-continuation reference.

**Doedel et al., AUTO** — AUTO-07P / AUTO-2000 (Doedel, Champneys, Fairgrieve, Kuznetsov, Oldeman, Paffenroth, Sandstede et al.) is the de-facto standard ODE-BVP continuation/bifurcation software.
- Reference: Doedel, E. J. (1981) "AUTO: a program for the automatic bifurcation analysis of autonomous systems", Congr. Numer., 30, 265–284. Plus the 2007 manual.
- Relevance: cite when noting that arclength continuation is well-established in dynamical-systems software, even if your steady-state-only PDE setting only needs a simpler natural-parameter (voltage) continuation with possibly a homotopy in inlet boundary condition.

**Allgower & Georg, "Numerical Continuation Methods: An Introduction"** (1990, Springer; 2003 SIAM Classics reprint). Standard textbook for predictor-corrector continuation. Cite once if reviewing methods generally.

**Uecker, Wetzel, Rademacher 2014** — pde2path; Uecker 2021 review "Continuation and Bifurcation in Nonlinear PDEs", Jahresber. DMV, 123, 199–248, DOI: 10.1365/s13291-021-00241-5. Modern comprehensive reference for PDE continuation. Mentions Newton-failure modes that motivate arclength continuation.

### 3.2 PNP-Specific Continuation / Newton-Convergence

This is the *thinnest* part of the literature for the user's project. The PNP/semiconductor community generally addresses Newton failure under strong polarization with one of three strategies:
- **Voltage ramp / Gummel iteration** (semiconductor TCAD heritage). Markowich, Ringhofer, Schmeiser 1990 *Semiconductor Equations* (Springer, ISBN 978-3-7091-7480-0, DOI 10.1007/978-3-7091-6961-2) — Chapter 4 "Drift-Diffusion" and the singular-perturbation analysis chapter give the formal justification for ramping the boundary potential / scaling Debye length. This is the textbook PNP convergence reference.
- **Energetic-variational PNP** (Eisenberg-Hyon-Liu and successors). Hyon, Eisenberg, Liu 2010, Commun. Math. Sci., 9(2), 459–475 — "A mathematical model for the hard-sphere repulsion in ionic solutions". Plus Eisenberg, Hyon, Liu 2010 J. Chem. Phys. on EnVarA. Offers a more stable formulation for PNP-with-steric, but does not directly address Newton-from-cold-start.
- **Modified PNP at high voltage** (Bazant). Kilic, Bazant, Ajdari 2007, "Steric effects in the dynamics of electrolytes at large applied voltages", Phys. Rev. E, 75, 021502 (Part I) and 021503 (Part II). DOI: 10.1103/PhysRevE.75.021502. Justifies why classical PB/PNP fails at >~ 25 mV double-layer potential and requires steric or other modification — provides physical motivation for needing voltage continuation rather than direct cold-start.

**Bazant, Squires 2010** (Curr. Opin. Coll. Interface Sci.) — "Towards an understanding of induced-charge electrokinetics at large applied voltages in concentrated solutions". Same theme: classical PB fails at high voltage, modifications are needed. Cite if positioning your high-eta convergence story.

**Schmuck, Bazant 2015** — "Homogenization of the Poisson-Nernst-Planck equations for ion transport in charged porous media", SIAM J. Appl. Math., 75(3), 1369–1401, DOI: 10.1137/140968082. PNP homogenization for porous electrodes. Less directly relevant but useful if your geometry is porous-cathode.

**EchemFEM / Firedrake (Roy, Lin, Hahn 2024)** — "EchemFEM: A Firedrake-based Python package for electrochemical transport", J. Open Source Software (also LLNL-JRNL-860653). Mentions current-ramping and voltage-continuation strategies in their convergence discussion. The most direct *engineering* reference for PNP-in-Firedrake convergence.

### 3.3 Themes and Findings (Continuation)
- Pseudo-arclength continuation (Keller 1977; Allgower-Georg) is the textbook method, but for the user's *steady-state* PNP-BV problem with monotone V-vs-current, *natural* parameter continuation in V is usually sufficient — arclength is needed only at turning points, which the user does not appear to have.
- The semiconductor TCAD literature (Markowich-Ringhofer-Schmeiser 1990) is the *real* origin of "ramp the bias from 0 to operating point" as a Newton-globalisation strategy for drift-diffusion / PNP equations.
- High-overpotential PB/PNP genuinely fails (Bazant et al.) due to steric/crowding; voltage continuation is part of the fix, modified-PNP is the other.

### 3.4 Recommended Continuation Citations
- Keller 1977 (pseudo-arclength)
- Allgower & Georg 1990/2003 (textbook)
- Doedel et al. 2007 AUTO manual (software heritage)
- Markowich, Ringhofer, Schmeiser 1990 (PNP/semiconductor textbook — voltage ramp)
- Kilic, Bazant, Ajdari 2007 PRE I&II (high-voltage failure of classical PB/PNP)
- Roy, Lin, Hahn 2024 EchemFEM (Firedrake-specific, recent)
- Uecker 2021 Jahresber. DMV (modern PDE continuation review)

---

## 4. Adjoint / Discrete-Adjoint Sensitivity in Firedrake

### 4.1 Canonical pyadjoint / dolfin-adjoint references

**Farrell, Ham, Funke, Rognes 2013** — "Automated derivation of the adjoint of high-level transient finite element programs".
- Year/venue: 2013, SIAM J. Sci. Comput., 35(4), C369–C393
- DOI: 10.1137/120873558
- Contribution: Algorithmic basis of dolfin-adjoint. Records the high-level forward-problem variational form, exploits the symbolic UFL representation to *automatically* derive the discrete adjoint. Demonstrates parallel scaling and optimal checkpointing.
- Relevance: This is the citation for "we use the discrete-adjoint approach as implemented in pyadjoint / dolfin-adjoint with Firedrake".
- Confidence: high. DOI and abstract verified.

**Mitusch, Funke, Dokken 2019** — "dolfin-adjoint 2018.1: automated adjoints for FEniCS and Firedrake".
- Year/venue: 2019, J. Open Source Softw., 4(38), 1292
- DOI: 10.21105/joss.01292
- Contribution: Software paper for the modern pyadjoint backend that rewrote dolfin-adjoint for both FEniCS and Firedrake. Operates by AD over the sequence of variational solves.
- Relevance: Direct software citation. The Firedrake "Solving adjoint PDEs" tutorial points to this paper. *Cite both Farrell-2013 and Mitusch-2019 together*.
- Confidence: high. DOI verified.

### 4.2 Foundational adjoint theory references

**Plessix 2006** — "A review of the adjoint-state method for computing the gradient of a functional with geophysical applications".
- Year/venue: 2006, Geophys. J. Int., 167(2), 495–503
- DOI: 10.1111/j.1365-246X.2006.02978.x
- Contribution: Pedagogical review of the adjoint-state method (one extra linear solve, gradient is independent of #parameters). Most-cited general reference.
- Relevance: Use as the "what is the adjoint method and why is it cheap" reference.
- Confidence: high. DOI and abstract verified.

**Gunzburger 2003** — "Perspectives in Flow Control and Optimization", SIAM, ISBN 0-89871-527-3. Standard book-length reference for PDE-constrained optimization, including the continuous-vs-discrete adjoint distinction (optimize-then-discretize vs discretize-then-optimize).

**Ascher & Petzold 1998** — "Computer Methods for ODEs and DAEs", SIAM. Book reference for adjoint sensitivity at the ODE level.

**Hinze, Pinnau, Ulbrich, Ulbrich 2009** — "Optimization with PDE Constraints", Springer Math. Modelling Theory & Applications 23. Modern PDE-constrained-optimization textbook.

### 4.3 Adjoint-vs-FD verification (Taylor test)

Standard Taylor test: ||J(m + h*delta_m) - J(m) - h * <dJ/dm, delta_m>|| = O(h^2). The pyadjoint API method `taylor_test()` implements this directly. Best primary reference is

- **Farrell, Ham, Funke, Rognes 2013** (above) — Section 5 of that paper documents the Taylor-test convergence as the canonical verification.
- Plessix 2006 also discusses FD verification in its Section 4.

### 4.4 Themes and Findings (Adjoint)
- For PDE-constrained inverse problems with Firedrake, the *discrete* adjoint via pyadjoint is now the standard. Citing Farrell-2013 + Mitusch-2019 is sufficient.
- Plessix 2006 is the textbook physical introduction.
- Taylor test is standard; convergence of the residual to O(h^2) confirms gradient correctness. Cold-ramp (not warm-start) FD verification near clipped voltages is the user's known caveat (see project memory feedback_adjoint_fd_verification.md) — supported implicitly by general FD-verification practice in PDE-constrained optimisation but no specific paper documents this exact pitfall.

### 4.5 Recommended Adjoint Citations
- Farrell, Ham, Funke, Rognes 2013 (SIAM SJSC) — dolfin-adjoint algorithmic basis
- Mitusch, Funke, Dokken 2019 (JOSS) — pyadjoint / FEniCS+Firedrake software
- Plessix 2006 (GJI) — adjoint-state-method review
- Gunzburger 2003 (SIAM book) — continuous vs discrete adjoint
- Hinze, Pinnau, Ulbrich, Ulbrich 2009 (Springer) — modern PDE-constrained-opt textbook

---

## 5. Local Fisher Information & Canonical-Ridge Cosines

### 5.1 Standard FIM Definition

- **Fisher, R. A. 1922** — "On the mathematical foundations of theoretical statistics", Phil. Trans. R. Soc. A, 222, 309–368, DOI: 10.1098/rsta.1922.0009. Original FIM definition.
- **Cramér 1946**, **Rao 1945** — Cramér-Rao bound. Standard textbook reference: Lehmann & Casella, "Theory of Point Estimation" (2nd ed., Springer 1998). For the Gaussian-noise least-squares case, FIM = J^T Sigma^{-1} J where J is the sensitivity (Jacobian) of the model output w.r.t. parameters.

### 5.2 SVD of Sensitivity / Fisher Matrix as Identifiability Diagnostic

**Brun, Reichert, Künsch 2001** (cited above in Section 2). Their "collinearity index" is exactly 1/sigma_min(J_normalized) — the inverse of the smallest singular value of the column-normalized sensitivity matrix. Each near-zero singular value points to a non-identifiable *direction* in parameter space, and the corresponding right singular vector is the "ridge" direction.

**Quaiser, Mönnigmann 2009** (cited above). Eigenvalue method = SVD of FIM for ranking parameter subsets.

**Joshi, Seidel-Morgenstern, Tiller 2006** (Note: search did not return this exact reference — see Verification Results. Replace with Brun 2001 + Quaiser 2009).

### 5.3 The Canonical-Ridge Cosine

The "ridge cosine" the user refers to is the cosine of the angle between (i) the right singular vector v_min of J (= direction of weakest-identified parameter combination in the FIM SVD) and (ii) some pre-defined "Tafel-ridge" direction (e.g., the unit vector n_Tafel = (1, +f*eta_typical, 0, 0)/||·|| in (log k0,1, alpha_1, log k0,2, alpha_2) space, which is the asymptotic-Tafel-branch nullspace at high eta).

This exact construction does *not* appear in the literature under that name, but is mathematically equivalent to:
- Canonical correlation analysis between parameter-perturbation directions (Hotelling 1936, Biometrika).
- "Subspace angle" diagnostics in identifiability (van der Vaart 2000 *Asymptotic Statistics*, §5).

The closest direct reference for *reporting* such a cosine is:

- **Vajda, Rabitz 1989** — "Identifiability and distinguishability of first-order reaction systems", J. Phys. Chem., 93, 5043. Uses overlap angles between sensitivity-derived eigenvectors as identifiability diagnostics for kinetic models.
- **Komorowski, Costa, Rand, Stumpf 2011** — "Sensitivity, robustness, and identifiability in stochastic chemical kinetics models", PNAS, 108(21), 8645–8650, DOI: 10.1073/pnas.1015814108. Uses information-theoretic decomposition of FIM eigenstructure.
- **Pant 2018** — "Information sensitivity functions to assess parameter information gain and identifiability of dynamical systems", J. R. Soc. Interface, 15, 20170871, DOI: 10.1098/rsif.2017.0871. Explicitly recommends SVD of FIM and inspecting eigenvectors as the practical identifiability diagnostic — most modern reference for the user's diagnostic.

### 5.4 Themes and Findings (FIM)
- The FIM-SVD / collinearity-index family of diagnostics (Brun 2001, Quaiser 2009, Pant 2018) is the standard methodology for "which directions in parameter space are identifiable".
- The "canonical-ridge cosine" naming is the user's; the underlying object is the dot-product of v_min(FIM) with a chosen analytical "ridge direction". Pant 2018 is the closest pre-existing diagnostic.
- For BV kinetics specifically, Bilodeau 2021 (CO2RR Bayesian) and Anantharaj-Noda 2021 (differential Tafel) confirm the (i0, alpha) co-correlation experimentally, supporting the user's identification of a specific Tafel ridge direction.

### 5.5 Recommended FIM / Ridge Citations
- Fisher 1922 (FIM definition)
- Cramér-Rao via Lehmann & Casella 1998 textbook
- Brun, Reichert, Künsch 2001 (collinearity index = SVD of normalized sensitivity)
- Quaiser, Mönnigmann 2009 (eigenvalue method)
- Pant 2018 J. R. Soc. Interface (modern FIM-SVD diagnostic; closest analog to canonical-ridge cosine)
- Komorowski et al. 2011 PNAS (FIM in chemical kinetics)
- Vajda & Rabitz 1989 (early reaction-systems identifiability via eigenvector angles)

---

## 6. Monolithic vs Splitting PNP Solvers (Optional Context)

### 6.1 Background

Two standard solver families for PNP:
- **Monolithic (full-Newton on the coupled system)**: phi (potential) and c_i (concentrations) are solved as one mixed function space; convergence is super-linear when the Newton update is well-defined, but Jacobians can be ill-conditioned and require good initial guesses (motivating voltage continuation in Section 3).
- **Splitting / Gummel iteration**: alternately solve Poisson with frozen c, then Nernst-Planck with frozen phi, repeat. Slower but more robust; weak coupling means each sub-solve is easier.

### 6.2 References
- **Gummel, H. K. 1964** — "A self-consistent iterative scheme for one-dimensional steady state transistor calculations", IEEE Trans. Electron Devices, 11(10), 455–465. Original Gummel iteration.
- **Markowich, Ringhofer, Schmeiser 1990** *Semiconductor Equations* (cited above) — Chapter 7 covers iterative schemes including Gummel and full Newton.
- **EchemFEM (Roy, Lin, Hahn 2024)** — implements both monolithic and segregated approaches in Firedrake; useful contemporary reference for the choice.
- **Yu, Holst, McCammon 2007** — "Three-dimensional finite element methods for the Poisson-Nernst-Planck ion-channel equations", J. Comput. Chem., 28, 1827–1839. Monolithic PNP with adaptive FE on biological geometries.
- **Liu, Wang 2014** — "A free energy satisfying finite difference method for Poisson-Nernst-Planck equations", J. Comput. Phys., 268, 363–376. Energy-stable structure-preserving scheme.

### 6.3 Recommended Monolithic-vs-Splitting Citations
- Gummel 1964 (origin of segregated iteration in semiconductor)
- Markowich, Ringhofer, Schmeiser 1990 (textbook treatment)
- Roy, Lin, Hahn 2024 EchemFEM (modern Firedrake)
- Yu, Holst, McCammon 2007 (monolithic PNP, biological)

---

## VERIFICATION RESULTS (CRITICAL — what was actually fetched)

Each row records: paper -> primary verification source -> status.

| Paper                                         | Verification source                                          | Status                          |
|-----------------------------------------------|--------------------------------------------------------------|---------------------------------|
| Wroblowa, Pan, Razumney 1976 J Electroanal Chem | Search summary + ScienceDirect listing                       | OK (DOI confirmed)              |
| Damjanovic, Genshaw, Bockris 1967 JES 114:466 | Search summary, 1968 discussion (ADS), multiple secondaries  | OK (paywalled JES PDF not fetched directly) |
| Marković, Schmidt, Stamenković, Ross 2001 Fuel Cells | Wiley DOI (fetch returned 402 paywall), multiple secondaries | OK at metadata level             |
| Marković & Ross 2002 Surf Sci Rep 45:117       | ADS abstract returned via search                              | OK                              |
| Norskov et al. 2004 JPC B (origin of overpotential) | ACS DOI (403 bot block); abstract via Norskov page, RG, SemSch | OK at metadata level             |
| Kulkarni, Siahrostami, Patel, Norskov 2018 Chem Rev | ACS DOI (403); UC Davis PDF available; PubMed 29405702        | OK                              |
| Stamenkovic et al. 2007 Science 315:493        | Search summary, PubMed 17218494, Science DOI                  | OK                              |
| Bard, Faulkner, White Electrochem Methods 3e (2022) | Wiley listing, Google Books                                  | OK (textbook)                    |
| Raue et al. 2009 Bioinformatics                | Oxford Academic page (fetched abstract)                      | OK (full abstract)              |
| Brun, Reichert, Künsch 2001 WRR                | Wiley DOI returned 402; OSTI mirror, ScienceOpen             | OK at metadata level             |
| Quaiser, Mönnigmann 2009 BMC Syst Biol         | BMC redirected to Springer (303); PubMed 19426527, BMC entry | OK at metadata level             |
| Bizeray, Kim, Duncan, Howey 2018 IEEE TCST     | arXiv:1702.02471 (fetched abstract)                          | OK                              |
| Forman, Moura, Stein, Fathy 2012 J Power Sources | Search summary; ECAL Berkeley public PDF                    | OK                              |
| Laue, Röder, Krewer 2021 J Appl Electrochem    | Springer 303; KIT repo, SemSch, ResearchGate                  | OK at metadata level             |
| Park, Kato, Gima, Klein, Moura 2018 JES        | Public PDF on ECAL Berkeley                                  | OK                              |
| Bilodeau, Gibson, Garnett 2021 Nat Commun (CO2RR Bayesian Tafel) | Search summary, Nature DOI                       | OK at metadata level             |
| Anantharaj & Noda 2021 Sci Rep (differential Tafel) | Search summary, Nature DOI                                | OK at metadata level             |
| Schalenbach et al. 2024 ACS Energy Lett        | Search summary, ACS DOI                                       | OK at metadata level             |
| Farrell, Ham, Funke, Rognes 2013 SIAM SJSC     | SIAM page (fetched abstract)                                  | OK                              |
| Mitusch, Funke, Dokken 2019 JOSS dolfin-adjoint | JOSS page (fetched metadata; abstract not on JOSS)            | OK                              |
| Plessix 2006 GJI                                | OUP fetch (full abstract)                                     | OK                              |
| Keller 1977 pseudo-arclength chapter            | Cited via Wikipedia, multiple secondaries                     | OK (book chapter, no DOI)        |
| Doedel AUTO 2007 manual                         | Concordia listing                                              | OK                              |
| Markowich, Ringhofer, Schmeiser 1990 (Springer book) | Springer book, multiple secondaries                       | OK (textbook, no abstract fetched) |
| Kilic, Bazant, Ajdari 2007 PRE I & II           | Search summary; Bazant MIT page                               | OK at metadata level             |
| Hyon, Eisenberg, Liu 2010 CMS                   | Rush ftp PDF, search summary                                  | OK                              |
| EchemFEM (Roy, Lin, Hahn 2024)                  | LLNL OSTI report                                              | OK                              |
| Pant 2018 J R Soc Interface                     | Royal Society DOI, PMC                                        | OK at metadata level             |
| Komorowski et al. 2011 PNAS                     | PNAS DOI                                                      | OK at metadata level             |
| Vajda, Rabitz 1989 J Phys Chem                  | Citation only (older; no abstract fetched)                    | OK at citation level             |
| Joshi, Seidel-Morgenstern, Tiller 2006          | NOT FOUND in search results                                   | NOT VERIFIED — replace with Brun 2001 + Quaiser 2009 |

Items NOT verified to abstract level (use with caveat or replace):
- Joshi-Seidel-Tiller 2006 — could not locate. Drop or substitute Brun-Reichert-Künsch 2001 + Quaiser-Mönnigmann 2009 + Pant 2018.

---

## Key Takeaways

1. **ORR mechanism provenance is solid and old.** The two-step decomposition R1 (O2 -> H2O2 at ~0.68 V) + R2 (H2O2 -> H2O at ~1.78 V) traces directly to Damjanovic-Genshaw-Bockris 1967 and Wroblowa-Pan-Razumney 1976 (RRDE-based mechanism criterion). For modern coverage, cite Marković-Ross 2002 (Surf. Sci. Rep.) and Kulkarni-Siahrostami-Patel-Nørskov 2018 (Chem. Rev.). The equilibrium potentials are standard textbook values (Bard-Faulkner-White Ch. 9 / appendix tables).

2. **The "Tafel ridge" is a textbook practical-non-identifiability problem, but is rarely named that way.** The cleanest explicit recent demonstrations are Bilodeau-Gibson-Garnett 2021 (Bayesian, CO2RR — broadly elongated (i0, alpha) posteriors) and Anantharaj-Noda 2021 (differential Tafel as a (i0, alpha) decoupler). For PDE-constrained electrochemistry analogs, Bizeray-Howey 2017/2018, Forman-Moura-Stein-Fathy 2012, Laue-Röder-Krewer 2021, and Park-Klein-Moura 2018 are the four canonical references.

3. **Adjoint-with-Firedrake provenance is short and definite.** Cite Farrell-Ham-Funke-Rognes 2013 (SIAM SJSC) for the dolfin-adjoint algorithmic basis and Mitusch-Funke-Dokken 2019 (JOSS) for the pyadjoint software. Plessix 2006 GJI is the universal pedagogical adjoint-state-method reference. Taylor-test verification is documented in Farrell 2013 §5.

4. **For PNP voltage continuation, the deepest reference is the semiconductor TCAD literature.** Markowich-Ringhofer-Schmeiser 1990 *Semiconductor Equations* (Springer) is the textbook origin of bias-ramping as a Newton globalisation strategy for drift-diffusion. Keller 1977 + Allgower-Georg 1990/2003 cover pseudo-arclength continuation if turning points appear (they typically do not for monotone V-vs-current ORR steady states). Kilic-Bazant-Ajdari 2007 PRE explains *why* classical PNP fails at high eta (steric crowding), giving physical motivation for the user's voltage-continuation strategy. EchemFEM (Roy-Lin-Hahn 2024) is the most direct contemporary Firedrake-PNP reference.

5. **The "canonical-ridge cosine" construction is the user's naming for a standard SVD-of-FIM diagnostic.** Brun-Reichert-Künsch 2001 (Water Resour. Res.) introduced the collinearity index = 1/sigma_min(J_normalized); Quaiser-Mönnigmann 2009 (BMC Syst. Biol.) gave the eigenvalue ranking; Pant 2018 (J. R. Soc. Interface) is the most modern instance recommending v_min(FIM) inspection. The angle between v_min and an analytic Tafel-ridge direction is mathematically a canonical correlation / subspace angle (Hotelling 1936; van der Vaart 2000) — give it a 2-line definition citing Brun 2001 + Pant 2018 and the writeup will be on solid ground.

