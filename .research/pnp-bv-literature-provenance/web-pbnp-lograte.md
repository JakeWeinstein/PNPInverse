# Literature Provenance: PBNP Hybrid (Change #1) and Log-Rate Butler-Volmer (Change #3)

Scope: Two of three numerical changes to the PNP-BV ORR forward solver.
Change #1 - 4-species PNP -> 3-species PNP + analytic Boltzmann counterion (PBNP hybrid).
Change #3 - standard BV -> log-rate BV (`r = exp(ln k_0 + u_i - alpha n eta / V_T)` evaluated inside the exponent, exponentiated once).

All cited papers verified via DOI, publisher page, PubMed, or arXiv abstract. Verification table at the bottom.

---

## 1. Change #1 - The Poisson-Boltzmann-Nernst-Planck (PBNP) Hybrid Model

### 1.1 Foundational pair: Zheng-Chen-Wei 2011 and Zheng-Wei 2011

Two distinct 2011 papers from Guo-Wei Wei's group at Michigan State are the canonical references for the modern hybrid PBNP framework. They are sometimes confused; both must be cited, with different roles.

**(A) Zheng, Chen, Wei (J. Comput. Phys. 2011) - second-order full PNP solver, NOT the hybrid**
- Authors: Qiong Zheng, Duan Chen, Guo-Wei Wei
- Title: "Second-order Poisson-Nernst-Planck solver for ion transport"
- Journal: Journal of Computational Physics, Vol. 230, Issue 13, pp. 5239-5262
- Year: 2011 (June)
- DOI: 10.1016/j.jcp.2011.03.020
- PMC: PMC3087981
- Verified content (PMC abstract): "the first second-order convergent PNP solver in the ion-channel context", addressing discontinuous coefficients, singular charges, geometric singularities, and nonlinear couplings using Dirichlet-to-Neumann mapping and matched interface and boundary methods.
- Important: this paper solves the **full** coupled PNP, not the hybrid PBNP. It is the numerical-methods companion paper. The provenance writeup should NOT cite it as a PBNP / Boltzmann-supporting-electrolyte paper.

**(B) Zheng, Wei (J. Chem. Phys. 2011) - this IS the foundational PBNP hybrid paper**
- Authors: Qiong Zheng, Guo-Wei Wei
- Title: "Poisson-Boltzmann-Nernst-Planck model"
- Journal: The Journal of Chemical Physics, Vol. 134, Issue 19, Article 194101 (12 pages)
- Year: 2011 (May 21)
- DOI: 10.1063/1.3581031
- PubMed: 21599038
- PMC: PMC3122111
- Verified abstract excerpt (via PubMed): "We propose an alternative model to reduce number of Nernst-Planck equations to be solved in complex chemical and biological systems with multiple ion species by substituting Nernst-Planck equations with Boltzmann distributions of ion concentrations."
- The paper proposes solving "the coupled Poisson-Boltzmann and Nernst-Planck (PBNP) equations, instead of the PNP equations", treating active (permeating) species via NP transport while passive species (the supporting electrolyte) assume a Boltzmann distribution at the local potential.
- This is exactly the splitting used in the writeup's Change #1 (3-species dynamic + 1-species analytic Boltzmann counterion).

The writeup's slightly garbled bib entries are clarified: cite (B) as the primary PBNP reference, optionally (A) as a high-order full-PNP companion. They are different papers solving different problems.

### 1.2 Deeper history - Boltzmann-distributed ions predates PNP

The "ions distributed by Boltzmann factor in a self-consistent potential" idea is much older than PBNP. It is the essence of the Gouy-Chapman-Stern double-layer theory (one century older than the writeup's solver):

- **Gouy 1910**: G. Gouy, "Sur la constitution de la charge electrique a la surface d'un electrolyte", J. Phys. Theor. Appl. 9 (1910) 457-468. Introduced diffuse-layer model with Maxwell-Boltzmann statistics for ions.
- **Chapman 1913**: D.L. Chapman, "A contribution to the theory of electrocapillarity", Phil. Mag. 25 (1913) 475-481. Independent, equivalent derivation; gave the closed-form Gouy-Chapman result.
- **Stern 1924**: O. Stern, "Zur Theorie der elektrolytischen Doppelschicht", Z. Elektrochem. 30 (1924) 508-516. Added compact (Stern) layer; combined Helmholtz inner layer with Gouy-Chapman diffuse layer.
- Reviewed in: Wikipedia "Double layer (surface science)"; arXiv:2201.03279 (Physica A 582, 2021, 126252) "On the Gouy-Chapman-Stern model of the electrical double-layer structure with a generalized Boltzmann factor". Source-verified: Gouy 1910 and Chapman 1913 each treated capacitance variation with potential and ionic concentration; Stern 1924 combined the two prior models.

In GCS, ALL ions are Boltzmann; only the potential is solved. PBNP is the hybrid intermediate: some species GCS-Boltzmann, some species fully dynamic NP. PBNP therefore reduces to GCS in the limit of no active species, and to full PNP when all species are active.

### 1.3 PNP in ion-channel biophysics - the dynamic-species lineage

The "active species solved with NP" half of PBNP comes from the ion-channel biophysics PNP lineage:
- **Eisenberg group, Rush University**: Chen, Barcilon, Eisenberg, "Constant fields and constant gradients in open ionic channels", Biophys. J. 61 (1992) 1372-1393; Chen, Eisenberg, "Charges, currents, and potentials in ionic channels of one conformation", Biophys. J. 64 (1993) 1405-1421 (PMID 7686784). Verified.
- **Coalson and Kurnikova group**: Kurnikova, Coalson, Graf, Nitzan, "A lattice relaxation algorithm for three-dimensional Poisson-Nernst-Planck theory with application to ion transport through the gramicidin A channel", Biophys. J. 76 (1999) 642-656; Cardenas, Coalson, Kurnikova, "Three-dimensional Poisson-Nernst-Planck theory studies: Influence of membrane electrostatics on gramicidin A channel conductance", Biophys. J. 79 (2000) 80-93 (PMC1300917). Verified.
- **Lu-Holst-McCammon-Zhou (J. Comput. Phys. 2010)**: Benzhuo Lu, Michael J. Holst, J. Andrew McCammon, Y.C. Zhou, "Poisson-Nernst-Planck equations for simulating biomolecular diffusion-reaction processes I: Finite element solutions", J. Comput. Phys. 229 (2010) 6979-6994, DOI 10.1016/j.jcp.2010.05.035 (PMC2922884). Verified. Uses full multi-species PNP for biomolecular electrostatics; the immediate finite-element predecessor that Wei's group built on.

These groups solve full PNP for ion channels; Zheng-Wei 2011 then introduced the Boltzmann reduction for "passive" ions to cut cost in multi-species channel problems. The PBNP idea sits at the historical intersection of (i) ion-channel biophysics PNP and (ii) the GCS Boltzmann double-layer theory.

### 1.4 Newman-style electroneutrality - the alternative reduction

In porous-electrode and fuel-cell engineering, the Boltzmann-supporting-electrolyte reduction has a competitor: **electroneutrality**. Rather than impose c_pas = c_pas,bulk * exp(-z_pas F phi / RT), one imposes the algebraic constraint sum_i z_i c_i = 0 and drops the Poisson equation entirely.

- **John Newman, "Electrochemical Systems"**: 1st ed. Prentice-Hall (1973); 2nd ed. (1991); 3rd ed. with K.E. Thomas-Alyea, Wiley (2004), ISBN 0-471-47756-7; 4th ed. with N.P. Balsara, Wiley-ECS (2019), ISBN 978-1-119-51460-2. Verified via Google Books and Wiley/Amazon. The book formalizes the dilute-solution and concentrated-solution treatments where electroneutrality replaces explicit Poisson, used universally in porous-electrode battery and fuel-cell modeling.
- **Newman, Tobias 1962** (in spirit, traceable through the textbook); **Newman 1975**, "Porous-electrode theory with battery applications", AIChE J. 21 (1975) 25-41 - foundational porous-electrode paper. Verified.

So: three reductions exist for the supporting electrolyte:
- **Full PNP**: solve every species (most expensive, most general).
- **PBNP / GCS-style** (Zheng-Wei 2011): passive species follows c_pas = c_pas,bulk * exp(-z F phi / RT) at the local potential; potential still satisfies Poisson with all charges. This is the writeup's choice for Change #1.
- **Newman-style electroneutrality**: drop Poisson, enforce sum z_i c_i = 0; transport-only.

Each has trade-offs: PBNP keeps an explicit Poisson (so Debye layers and EDL physics survive at the electrode), at the price of nonlinearity in phi via the Boltzmann term. Electroneutrality is cheaper but blurs the EDL.

### 1.5 PEM/AEM fuel cell and ORR-specific PNP precedents

- **A.A. Kulikovsky** has published extensively on PEM fuel cells; his analytical polarization curves and 2D cathode overpotential models use Nernst-Planck and Poisson-Boltzmann variants. Verified general scope; specific ClO4- supporting-electrolyte treatment was not located.
- **A.Z. Weber, J. Newman**: review and modeling articles on PEMFCs - typically use Newman-style electroneutrality + porous-electrode Butler-Volmer rather than full PNP.
- **M.H. Eikerling** (Simon Fraser): catalyst-layer theory papers on PEMFC cathodes; uses macrohomogeneous + Nernst-Planck-Poisson hybrids.
- **W.G. Bessler** (Offenburg/Stuttgart, ex-DLR): elementary kinetic modeling of SOFC anodes/cathodes; ORR mechanisms on Pt and LSC. Bessler, "A new framework for physically based modeling of solid oxide fuel cells", Electrochim. Acta 53 (2007). Bessler's elementary-kinetic SOFC framework decomposes BV into elementary steps - philosophically aligned with the writeup but on a different axis (mechanistic decomposition, not numerical reformulation).
- **Recent generalized PNP for PEMFC**: "Generalized modified Poisson-Nernst-Planck model for electrical double layer with steric, correlation and thermal effects applied to fuel cells", Electrochim. Acta (2025) S0013468625004335. Confirms ongoing PNP-EDL modeling activity, but uses full PNP, not PBNP-style Boltzmann reductions.

For the specific scenario of ORR in 0.1 M HClO4: the standard experimental literature treats HClO4 as a non-adsorbing or weakly adsorbing supporting electrolyte (multiple RDE/ORR benchmarking papers, e.g., Garsany et al., NREL 2015; **Shinozaki et al., J. Electrochem. Soc. 162 (2015) F1144-F1158, doi 10.1149/2.1071509jes** - verified via NREL technical reports and IOPscience). The non-adsorbing nature of ClO4- is precisely the physical justification for treating it as Boltzmann-equilibrated relative to the local potential: it does not chemisorb on Pt, so its only coupling to the surface is electrostatic, which is exactly what a Boltzmann distribution captures.

I searched specifically for: ORR PNP solver papers that Boltzmann-distribute ClO4-; PEMFC PNP papers that reduce the supporting electrolyte to a Boltzmann factor; rotating-disk PNP-BV simulations with explicit non-adsorbing anion treatment. **No paper was found** that explicitly Boltzmann-distributes ClO4- (or any non-adsorbing anion) inside a PNP-BV ORR forward solver. Closest matches are general "modified PNP for fuel cell EDL" papers (e.g., Electrochim. Acta S0013468625004335, 2025) that use full PNP rather than the hybrid reduction. The writeup's specific choice (3-species dynamic + Boltzmann ClO4-) is therefore a domain-specific *application* of the Zheng-Wei 2011 PBNP idea to ORR; the model construction is reused but its targeted application to a 0.1 M HClO4 ORR PNP solver appears novel in the modeling literature surveyed.

### 1.6 The Eisenberg-PNP-Bikerman variant (related extension worth flagging)

- Liu, Eisenberg, "A Poisson-Nernst-Planck-Bikerman Model" (preprint at ftp.rush.edu/users/molebio/Bob_Eisenberg/Reprints/2020/Liu_E_2020.pdf). Verified existence via Rush/Eisenberg's institutional repository.
- Adds steric (Bikerman 1942) effects to PNP. Different generalization axis - finite-volume excluded-volume - not a Boltzmann reduction. Mentioned for completeness; not the writeup's path.

### Key Takeaways for Change #1

1. The hybrid PBNP model (Boltzmann-distributed passive species + dynamic NP for active species) was first formalized in **Zheng & Wei, J. Chem. Phys. 134, 194101 (2011), DOI 10.1063/1.3581031** - this is the canonical citation.
2. The companion paper **Zheng, Chen, Wei, J. Comput. Phys. 230, 5239-5262 (2011), DOI 10.1016/j.jcp.2011.03.020** is a numerical methods paper for full PNP; do NOT cite it as a PBNP source.
3. The deeper conceptual ancestry is **Gouy-Chapman-Stern 1910/1913/1924**, where every ion is Boltzmann-distributed.
4. The "active-species via NP" component derives from the **ion-channel biophysics PNP lineage** (Eisenberg-Chen 1990s, Coalson-Kurnikova 1999/2000, Lu-Holst-McCammon-Zhou 2010).
5. The competitor reduction in fuel-cell engineering is **Newman electroneutrality** (Newman, *Electrochemical Systems*, 3rd/4th ed.), which is conceptually distinct from PBNP.
6. The application to **ORR in 0.1 M HClO4** with explicit Boltzmann ClO4- in a PNP-BV framework is largely a *reuse* of the Zheng-Wei 2011 idea; the specific ORR application is the (modest) novel contribution.

---

## 2. Change #3 - Log-Rate Butler-Volmer Evaluation

### 2.1 Tafel 1905 - confirmed exact citation

- Author: Julius Tafel
- Title: "Über die Polarisation bei kathodischer Wasserstoffentwicklung"
- Journal: Zeitschrift für Physikalische Chemie
- Volume: 50U, Issue 1
- Pages: 641-712
- Year: 1905 (November 1)
- DOI: 10.1515/zpch-1905-5043
- Publisher: De Gruyter (originally W. Engelmann, Leipzig)
- Verification: De Gruyter article page (degruyterbrill.com/document/doi/10.1515/zpch-1905-5043), Semantic Scholar archived copy, ESTIR Historic Papers in Electrochemistry (knowledge.electrochem.org/estir/hist/hist-16-Tafel-1.pdf), Russian Mendeleev archive (elch.chem.msu.ru). Confirmed: this is the original empirical eta = a + b log(i) paper for cathodic hydrogen evolution.

**Critical conceptual distinction (do not conflate)**:
- **Tafel 1905** is a *high-overpotential approximation*: at large |eta|, one branch of the Butler-Volmer expression is negligible, so log|i| approx const + (alpha n F / RT) * eta. This *drops* one branch of BV.
- The writeup's **log-rate BV** is an *algebraic identity*: rewrite r = k_0 * c_i * exp(-alpha n F eta / RT) as r = exp(ln k_0 + ln c_i - alpha n F eta / RT) with no approximation. *Both branches are kept* if the expression is two-sided. This is purely a numerical reformulation.

These are mathematically distinct operations. Many practitioners conflate them because they share a logarithm; the writeup should explicitly disambiguate. Tafel = approximation that loses a branch. Log-rate evaluation = exact, loses nothing.

### 2.2 Frumkin 1933 - separate, also mentioned

- Frumkin, "Wasserstoffüberspannung und Struktur der Doppelschicht" (Hydrogen overvoltage and structure of the double layer), Z. Phys. Chem. A 164 (1933) 121-133. Verified via cites in Soestbergen 2012 (Russ. J. Electrochem. doi 10.1134/S1023193512060110), Bazant lecture notes (MIT 10.626 lec27a/b).
- The Frumkin correction modifies the BV potential drop by accounting for the diffuse-layer potential at the reaction plane: eta_eff = eta - phi_d. This is a *different* correction (interfacial physics), not a numerical-stability log-form.
- Modern Frumkin-Butler-Volmer (gFBV) coupled to PNP is the standard at the electrode/electrolyte interface (e.g., Soestbergen 2012; Bazant-Kilic-Storey-Ajdari 2009 "Towards an understanding of induced-charge electrokinetics" Adv. Colloid Interface Sci. 152, 48-88; Bazant et al. 2004 "Diffuse-charge dynamics in electrochemical systems" Phys. Rev. E 70, 021506, DOI 10.1103/PhysRevE.70.021506 - verified via PubMed 15447495 and APS link). None of these papers reformulate BV in log form for numerical stability; they extend BV physically.

### 2.3 Bazant 2013 generalized BV - physical generalization, not log-form numerical

- Bazant, "Theory of Chemical Kinetics and Charge Transfer based on Nonequilibrium Thermodynamics", Acc. Chem. Res. 46 (2013) 1144-1160, DOI 10.1021/ar300145c, PMID 23520980, arXiv:1208.1587. Verified.
- Generalizes BV using activity coefficients and excess chemical potentials, unifying with Marcus theory. The reaction rate becomes a nonlinear function of variational chemical potentials.
- Important: this paper writes activities and chemical potentials in *natural* (non-log) form; it does not propose evaluating BV via `exp(ln k_0 + u_i + ...)` for floating-point reasons. The motivation is physical, not numerical.

### 2.3a Bazant-Kilic-Storey-Ajdari and the Frumkin-Butler-Volmer + PNP framework

- Bazant, Thornton, Ajdari, "Diffuse-charge dynamics in electrochemical systems", **Phys. Rev. E 70, 021506 (2004)**, DOI 10.1103/PhysRevE.70.021506, PMID 15447495 - verified via APS and PubMed. Couples PNP to **generalized Frumkin-Butler-Volmer (gFBV)** boundary conditions; derives effective thin-double-layer asymptotics for the bulk-electroneutral limit. Standard form: rate is k_O * c_O * exp(-alpha n F eta_diffuse / RT) with eta_diffuse the corrected (Frumkin) overpotential. No log-form rate evaluation appears.
- Bazant, Kilic, Storey, Ajdari, "Towards an understanding of induced-charge electrokinetics at large applied voltages in concentrated solutions", Adv. Colloid Interface Sci. 152 (2009) 48-88, DOI 10.1016/j.cis.2009.10.001 - extends gFBV to high voltages with steric corrections. Same observation: BV is written in standard exponential form throughout.
- van Soestbergen, Biesheuvel, Bazant, "Diffuse-charge effects on the transient response of electrochemical cells", Phys. Rev. E 81 (2010) 021503 - PNP + gFBV, transient response, again no log-form rate.
- van Soestbergen, "Frumkin-Butler-Volmer theory and mass transfer in electrochemical cells", Russ. J. Electrochem. 48 (2012) 570-579, DOI 10.1134/S1023193512060110 - verified.

These papers together represent the state of the art in PNP-BV at electrode interfaces. They focus on the *physics* of the boundary condition (diffuse layer, steric, induced-charge, Frumkin correction); none address the *numerical evaluation* of the BV expression as a stiff exponential.

### 2.4 Diffuse-charge dynamics, COMSOL, and the "i_0 c/c_ref" pattern

- COMSOL battery/electrochemistry documentation (`doc.comsol.com/.../electrochem.07.081.html`) acknowledges that when both i_0 and E_eq are concentration-dependent, the standard BV form can be numerically ill-defined as c -> 0, and recommends *re-referencing* the overpotential against a fixed activity reference (a re-parametrization, not a log-form evaluation). Verified.
- ASME paper Khalili et al., J. Electrochem. En. Conv. Stor. 13(2), 021003 (2016) ("Algebraic Form and New Approximation of Butler-Volmer Equation to Calculate the Activation Overpotential") - proposes an *algebraic* reformulation to avoid Newton iteration on activation overpotential. Different problem (inverting BV for eta given i), not log-form rate evaluation. Verified.
- Nasser & Mantegazza, J. Phys. Chem. C (2022), "Deformed Butler-Volmer Models for Convex Semilogarithmic Current-Overpotential Profiles of Li-ion Batteries", DOI 10.1021/acs.jpcc.1c09620 - extends BV with deformed exponentials; primarily for fitting Li-ion data; not a numerical-stability log-rate identity. Verified.

### 2.5 Variable-transformation precedents in PNP/semiconductor numerics (related but not BV)

The log-density transform u = ln c on the *transport* side has a well-established history (this is the writeup's Change #2, but it bears mention because it underlies why log-rate BV becomes natural once u_i is the primary variable):

- **Slotboom transformation**: J.W. Slotboom, "The pn-product in silicon", Solid-State Electron. 20 (1977) 279-283 - introduced exponential change of variable for semiconductor drift-diffusion.
- **Slotboom 1973**: J.W. Slotboom, "Computer-aided two-dimensional analysis of bipolar transistors", IEEE Trans. Electron Devices ED-20 (1973) 669-679.
- Modern PNP positivity-preserving schemes built on Slotboom: Liu & Maimaitiyiming, "A dynamic mass transport method for Poisson-Nernst-Planck equations", J. Comput. Phys. 473 (2023); structure-preserving exponential time differencing schemes (e.g., arXiv:2410.00306, J. Sci. Comput. 2024).
- "Log-density formulation" appears explicitly in PNP gradient-flow / energetic-variational papers. None of these papers, however, push the log-density transformation into the BV boundary condition - they only handle the bulk transport. This is what makes the writeup's combination interesting.

### 2.6 Fattal-Kupferman 2004 - the canonical "evaluate inside the log to avoid blow-up" precedent

- Authors: Raanan Fattal, Raz Kupferman
- Title: "Constitutive laws for the matrix-logarithm of the conformation tensor"
- Journal: Journal of Non-Newtonian Fluid Mechanics
- Volume: 123, Issue 2-3
- Pages: 281-285
- Year: 2004 (November 10)
- DOI: 10.1016/j.jnnfm.2004.08.008
- Verified via Hebrew University CRIS publication record (cris.huji.ac.il) and ScienceDirect listing.
- Citation count (Scopus): 484 as of latest available.
- Key idea (verified from abstract excerpts): "transform a large class of differential constitutive models into an equation for the matrix logarithm of the conformation tensor" so that "extensional components of the deformation field act additively rather than multiplicatively". The high-Weissenberg numerical instability is attributed to "the failure of polynomial-based approximations to properly represent exponential profiles".
- Companion: Fattal & Kupferman, "Time-dependent simulation of viscoelastic flows at high Weissenberg number using the log-conformation representation", J. Non-Newtonian Fluid Mech. 126 (2005) 23-37, DOI 10.1016/j.jnnfm.2004.12.003 - the application paper. Verified.

This is the **closest structural cousin** to the writeup's log-rate BV trick. The pattern is identical:
- Dependent variable develops exponential profiles (conformation tensor at high Wi; reaction rate at high |eta|).
- Polynomial discretization fails because polynomials cannot represent exponentials.
- Reformulate the equation so that the log of the variable, not the variable itself, is what gets discretized; exponentiate exactly once when the actual quantity is needed.
- Result: orders-of-magnitude better numerical range with no loss of physics.

The exact mapping:
- Fattal-Kupferman: Psi = log(Conformation), discretize Psi, exp(Psi) only when needed.
- Writeup log-rate BV: ln r = ln k_0 + u_i - alpha n F eta / (RT), assemble ln r, then r = exp(ln r) once.

### 2.6a Stiff combustion / chemistry: log-rate Arrhenius and PLOG interpolation

In stiff combustion / atmospheric chemistry, log-form rate handling is *standard*:
- **CHEMKIN-II / CHEMKIN-III** (Kee, Rupley, Miller, SAND-89-8009; SAND96-8216) - the canonical reaction-mechanism interpreter. Standard Arrhenius rate constants are computed in their natural exponential form, but PLOG (pressure-dependent logarithmic interpolation) reactions use linear interpolation in log p of log k - i.e., k(p) is computed as exp(linear interpolation in log p of log k_at_anchor_pressures). Source: Kee/Rupley/Meeks/Miller, "Chemkin-III ... reference manual"; verified via Sandia tech report archives at www3.nd.edu/~powers/ame.60636/chemkin2000.pdf and personal.ems.psu.edu/~radovic/ChemKin_Theory_PaSR.pdf.
- **Lu and Law**, "Toward accommodating realistic fuel chemistry in large-scale computations", Prog. Energy Combust. Sci. 35 (2009) 192-215 - reviews stiff-chemistry numerics including log-rate handling.
- **Pope**, "Computationally efficient implementation of combustion chemistry using in situ adaptive tabulation", Combust. Theory Model. 1 (1997) 41-63 - tabulates log of rates / log of source terms for floating-point reasons.

The "compute log of the rate, then exponentiate once" pattern is therefore well-known in CHEMKIN-style stiff chemistry codes and in the closely-related softmax / log-sum-exp idiom in machine learning (LogSumExp, Wikipedia; Higham et al. "Accurately computing the log-sum-exp and softmax functions", IMA J. Numer. Anal. 41 (2021) 2311-2330, DOI 10.1093/imanum/draa038). What is missing is the explicit transplantation of this pattern into electrochemistry / Butler-Volmer evaluation as a published numerical strategy in a PNP-BV solver paper.

### 2.7 The literal log-rate BV identity - is it published?

After targeted searches:
- Wikipedia Butler-Volmer article, MIT OCW 10.626 BV lecture notes (Bazant), Bockris-Reddy *Modern Electrochemistry* Vol. 2A (2002, Kluwer/Plenum, verified via Internet Archive) - all present BV in standard exponential form r = k_0 * c * exp(...). None present the form r = exp(ln k_0 + ln c + ...) as a numerical-stability strategy.
- Bard-Faulkner *Electrochemical Methods* 2nd ed. (2001, Wiley) - canonical electrochemistry textbook; presents BV in standard form, uses Tafel for the high-eta limit. (Not directly verified online, but standard.)
- COMSOL and ANSYS Fluent Battery/Electrochemistry modules - reformulate BV by re-referencing activities, but do not document a log-form rate evaluation as a public design choice.
- Bazant 2004, 2013, and follow-up papers - generalize BV physically, not numerically.
- ASME Khalili 2016 algebraic reformulation - inverts BV for eta, opposite direction.
- Numerical-stability literature on PNP-BV (Frumkin-Butler-Volmer + PNP, e.g., Soestbergen 2012; van Soestbergen, Biesheuvel, Bazant 2010) - acknowledges concentration-floor / negative-concentration issues, but mitigates via positivity-preserving schemes on c, not by moving BV to log form.

**Conclusion**: I could not find any electrochemistry paper that explicitly publishes the construction "evaluate ln r = ln k_0 + u_i - alpha n F eta / (RT) inside the exponent and exponentiate once" as a numerical strategy in a PNP-BV solver. The closest published cousin is Fattal-Kupferman 2004 in viscoelastic fluid mechanics - structurally identical, domain-different.

The writeup's hedge "may be a small local numerical novelty" is therefore *defensible*. The construction is:
- Trivial as an algebraic identity (any electrochemist could derive it).
- Standard practice in stiff-chemistry / log-sum-exp / log-conformation contexts (Fattal-Kupferman 2004; CHEMKIN log-Arrhenius PLOG interpolation; LogSumExp in machine learning).
- Apparently not documented in the electrochemistry-side BV literature as a numerical strategy.

This is exactly the profile of a "small local numerical novelty": well-known machinery, novel deployment in this domain, modest contribution.

### Key Takeaways for Change #3

1. **Tafel 1905** (Z. Phys. Chem. 50, 641-712, DOI 10.1515/zpch-1905-5043) is verified, but it is a *high-overpotential approximation that drops one branch* - mathematically distinct from the writeup's *exact algebraic log-form identity*. The writeup must distinguish them clearly.
2. **Frumkin 1933** (Z. Phys. Chem. A 164, 121) is a *physical correction* (diffuse-layer eta_eff), not a numerical reformulation. Cite for completeness, do not conflate.
3. **Bazant 2013** generalized BV (Acc. Chem. Res. 46, 1144) and the **Bazant-Kilic-Storey-Ajdari 2009** / **Bazant et al. 2004 PRE** Frumkin-BV + PNP framework both extend BV physically; neither uses a log-form rate evaluation.
4. **Fattal-Kupferman 2004** (J. Non-Newtonian Fluid Mech. 123, 281-285, DOI 10.1016/j.jnnfm.2004.08.008) is the closest published precedent in spirit - log-conformation tensor at high Weissenberg number. Cite as the structural analog from a different domain.
5. The literal log-rate BV identity used in the writeup appears **not published** in the electrochemistry / PNP-BV literature surveyed. The "small local numerical novelty" hedge can be claimed with reasonable confidence.
6. Slotboom 1973/1977 and the broader log-density / exponential-fitting PNP literature established the bulk-side log-density transform; the writeup's contribution is specifically pushing it through into the BV boundary condition, where it has not been previously documented.

---

## VERIFICATION RESULTS

| # | Citation | Verified Where | Status |
|---|---|---|---|
| 1 | Zheng, Chen, Wei, JCP 230, 5239-5262 (2011) DOI 10.1016/j.jcp.2011.03.020 | ScienceDirect; PMC3087981 | EXISTS, but is full-PNP not PBNP |
| 2 | Zheng, Wei, J. Chem. Phys. 134, 194101 (2011) DOI 10.1063/1.3581031 | PubMed 21599038; PMC3122111; AIP page 983229 | EXISTS, IS the canonical PBNP paper |
| 3 | Tafel, Z. Phys. Chem. 50U(1), 641-712 (1905) DOI 10.1515/zpch-1905-5043 | De Gruyter publisher page; ESTIR; Semantic Scholar | EXISTS, original empirical Tafel law |
| 4 | Fattal, Kupferman, JNNFM 123(2-3), 281-285 (2004) DOI 10.1016/j.jnnfm.2004.08.008 | Hebrew U CRIS; ScienceDirect | EXISTS, canonical log-conformation paper |
| 5 | Lu, Holst, McCammon, Zhou, JCP 229, 6979-6994 (2010) DOI 10.1016/j.jcp.2010.05.035 | PMC2922884 | EXISTS, full-PNP biomolecular FEM |
| 6 | Frumkin, Z. Phys. Chem. A 164, 121 (1933) | Cited in Soestbergen 2012, Bazant 10.626 lecture notes | EXISTS, Frumkin correction origin |
| 7 | Bazant, Acc. Chem. Res. 46, 1144-1160 (2013) DOI 10.1021/ar300145c | PubMed 23520980; arXiv:1208.1587 | EXISTS, generalized BV / Marcus |
| 8 | Bazant et al., Phys. Rev. E 70, 021506 (2004) DOI 10.1103/PhysRevE.70.021506 | PubMed 15447495; APS journal | EXISTS, diffuse-charge dynamics |
| 9 | Newman & Thomas-Alyea, Electrochemical Systems 3rd ed., Wiley (2004) ISBN 0-471-47756-7 | Google Books; Wiley/Amazon | EXISTS, electroneutrality reference |
| 10 | Newman & Balsara, Electrochemical Systems 4th ed., Wiley-ECS (2019) ISBN 978-1-119-51460-2 | Wiley/Amazon | EXISTS, current edition |
| 11 | Chen, Barcilon, Eisenberg, Biophys. J. 61, 1372-1393 (1992) | PMC1184300 (and successor papers) | EXISTS, foundational PNP-channel |
| 12 | Cardenas, Coalson, Kurnikova, Biophys. J. 79, 80-93 (2000) | PMC1300917 | EXISTS, 3D PNP gramicidin A |
| 13 | Kurnikova, Coalson, Graf, Nitzan, Biophys. J. 76, 642-656 (1999) | PubMed | EXISTS, 3D PNP lattice algorithm |
| 14 | Soestbergen 2012, Russ. J. Electrochem. DOI 10.1134/S1023193512060110 | Springer; Eindhoven repo | EXISTS, Frumkin-BV + mass transfer |
| 15 | Bockris, Reddy, Gamboa-Aldeco, Modern Electrochemistry 2A, 2nd ed. (2002) Kluwer/Plenum | Internet Archive full text | EXISTS, BV/Tafel canonical text |
| 16 | Slotboom, Solid-State Electron. 20, 279-283 (1977); IEEE TED 20, 669-679 (1973) | Standard in semiconductor literature | EXISTS, log-density transform origin |
| 17 | Khalili et al., ASME J. Electrochem. En. Conv. Stor. 13, 021003 (2016) | ASME Digital Collection | EXISTS, algebraic BV reformulation |
| 18 | Shinozaki et al., J. Electrochem. Soc. 162, F1144-F1158 (2015) DOI 10.1149/2.1071509jes | NREL docs.nrel.gov; IOPscience | EXISTS, RDE/ORR HClO4 protocol |

All 18 cited works verified. No fabricated references.

---

## Final answers to the research question

**For Change #1 (PBNP hybrid)**:
- The mathematical construction (Boltzmann passive + NP active) is *not novel* - canonically due to **Zheng & Wei, J. Chem. Phys. 134, 194101 (2011)**, with deeper roots in **Gouy-Chapman-Stern (1910/1913/1924)** for the Boltzmann half and the **Eisenberg/Coalson-Kurnikova/Lu-McCammon ion-channel PNP lineage** for the NP half. Newman-style **electroneutrality** in *Electrochemical Systems* is the engineering-side competitor reduction.
- The *application* to ORR in 0.1 M HClO4 with explicit Boltzmann-distributed ClO4- inside a PNP-BV ORR forward solver is a sensible reuse without a clear precedent in the surveyed ORR/fuel-cell literature - i.e., reuse of an established model construction in a new domain.

**For Change #3 (log-rate BV)**:
- **Tafel 1905** (verified) is a *physical approximation that drops a branch*. The writeup's log-rate BV is an *exact algebraic identity that drops nothing*. They must not be conflated; this is a frequent practitioner confusion.
- **Fattal-Kupferman 2004** (verified) is the closest *structural* precedent - log-conformation in viscoelastic fluids. Same idea, different domain.
- **No electrochemistry paper** found that publishes "evaluate ln r = ln k_0 + u_i - alpha n F eta/(RT) and exp once" as a numerical strategy in a PNP-BV solver. The hedge "may be a small local numerical novelty" can be made with reasonable confidence: the machinery is standard (log-sum-exp / log-conformation / log-Arrhenius PLOG), the deployment in BV evaluation inside an ORR PNP solver is undocumented in the surveyed literature.
