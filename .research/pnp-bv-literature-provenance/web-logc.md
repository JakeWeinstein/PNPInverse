## Literature Provenance — Change #2: Log-Concentration / Log-Density Formulation of PNP

Research focus: trace the provenance of the substitution `u_i = ln c_i` (and equivalent entropy-variable / Slotboom / quasi-Fermi formulations) used to give a PNP forward solver positivity, energy stability, and well-conditioning. Verify each cited paper actually exists, and cleanly separate (a) reuse from established literature, (b) papers whose claims are misrepresented, and (c) likely fabrications.

---

### 1. Slotboom variables (semiconductor lineage)

Two distinct Slotboom papers circulate in the literature; the writeup's "Slotboom 1973, Solid-State Electronics, 'The PN-product in silicon'" citation is a **conflation of three different Slotboom works**.

**1a. Slotboom 1969 (the actual "Slotboom variables" paper).**
- J. W. Slotboom, "Iterative scheme for 1- and 2-dimensional d.c.-transistor simulation," *Electronics Letters*, **5**(26), 677–678, 27 Dec 1969. DOI: 10.1049/el:19690510.
- Source: IET Digital Library page (https://digital-library.theiet.org/doi/10.1049/el%3A19690510) and Semantic Scholar entry (https://www.semanticscholar.org/paper/Iterative-scheme-for-1-and-2-dimensional-simulation-Slotboom/88910efa1cd816b092a4a3b934128bb96c4c3019).
- This paper introduces the change-of-variables Φn = n·exp(−ψ/V_T), Φp = p·exp(+ψ/V_T) (later named the "Slotboom variables") to convert the drift-diffusion continuity equations into self-adjoint elliptic equations with exponential coefficients.
- Confidence: **high**.

**1b. Slotboom 1973 (a different paper, often miscited as the "PN-product" paper).**
- J. W. Slotboom, "Computer-aided two-dimensional analysis of bipolar transistors," *IEEE Trans. Electron Devices* **ED-20**, 669–679 (1973).
- Confidence: **high**. This is the standard "1973 Slotboom" reference in the device-simulation literature.

**1c. Slotboom 1977 (the actual "PN-product" paper).**
- J. W. Slotboom, "The pn-product in silicon," *Solid-State Electronics* **20**(4), 279–283 (April 1977). https://www.sciencedirect.com/science/article/abs/pii/0038110177901083 .
- This is a band-gap-narrowing paper, **not** a numerical/discretization paper. It has nothing to do with the substitution `u = ln c`.
- Confidence: **high**.

**Bottom line:** the writeup's citation "Slotboom 1973, *Solid-State Electronics*, 'The PN-product in silicon'" is a chimera — wrong year for the title given, wrong title for the year given. The correct primary reference for "Slotboom variables" is **Slotboom 1969, *Electronics Letters*, vol. 5, pp. 677–678**.

**1d. Are Slotboom variables the same as `u = ln c`?**
- **No, they are not the same substitution, but they are closely related.** Slotboom's substitution is u = c·exp(−zψ/V_T): a *multiplicative* combination of density and potential. The "log-density" substitution u = ln c is a *logarithmic* transform of density alone.
- They become the *same idea* only after a further log: ln(Φn) = ln(n) − ψ/V_T = (chemical potential up to a constant) = the **quasi-Fermi level**. So "Slotboom variable" ↔ "exp(quasi-Fermi level)" ↔ "log-density + potential = chemical potential / entropy variable."
- In practice the device-simulator field tends to use either Slotboom variables (Φn) or quasi-Fermi levels (φn = ln Φn) as primaries. The PNP-electrochemistry field tends to use either ln c alone (Metti) or the chemical-potential combination (Fu–Xu, Liu et al.).
- The mapping is one-to-one but the discretizations are not equivalent: a Galerkin discretization on ln c is *not* the same scheme as a Galerkin discretization on Φn.
- Confidence: **high** on the mapping; **medium** on the practical-equivalence claim (different discretizations have different conditioning and different positivity guarantees).

**1e. Why semiconductor practice diverged from electrochemistry practice.**
- Per multiple secondary sources (COMSOL semiconductor module documentation, Selberherr 1984 chapter 5, Markowich–Ringhofer–Schmeiser 1990 chapter 3), Slotboom variables work poorly under degenerate (Fermi–Dirac) statistics because the relation between Φn and n loses the simple Boltzmann form. This is why modern semiconductor codes (ChargeTransport.jl, WIAS pdelib) prefer the quasi-Fermi-level formulation or excess-chemical-potential flux schemes (Farrell, Koprucki, Fuhrmann line of work, 2017–2024).
- Electrochemistry rarely cares about degeneracy (dilute aqueous solutions stay Boltzmann-like even at concentrated 1–10 M), so the original Slotboom transformation would in principle work fine in PNP-electrochemistry — but the field never adopted it. The cultural reason is that the electrochemistry codes descended from Newman's CONDUC / DUALFOIL lineage (1990s, c-primary FD) rather than from the semiconductor TCAD lineage (Pisces, MEDICI, Sentaurus — Slotboom-primary FD/FV).
- Confidence: **medium** (this is a community-history claim based on cross-referencing several documentation sources; cannot fully verify without an electrochemistry-history paper).

---

### 2. Scharfetter–Gummel 1969

- D. L. Scharfetter and H. K. Gummel, "Large-signal analysis of a silicon read diode oscillator," *IEEE Trans. Electron Devices* **ED-16**(1), 64–77 (1969). The "January 1969" issue.
- Note: one search snippet returned pages 391–415 — this is likely wrong (page 391 is a different article in vol. 16). Standard textbook citations give vol. 16 no. 1 pp. 64–77 in January 1969. **The writeup's pp. 64–77 is the standard citation; trust it.**
- The SG scheme is **NOT** a "log-density substitution." It is an **exponential-fitting / upwind finite-volume scheme on a primary unknown of `c`**, where each edge flux is computed by analytically integrating the steady-state drift-diffusion ODE between two nodes assuming a piecewise-linear potential. The result is a Bernoulli-function weighting of the two endpoint densities — not a change of variable.
- Many writeups (including the present one, and several blog posts I found) loosely conflate SG with "log-density" because both deliver positivity. They are **distinct** mechanisms.
- Sources: Brezzi, Marini, Pietra, "Two-dimensional exponential fitting and applications to drift-diffusion models," *SIAM J. Numer. Anal.* **26**(6), 1342–1355 (1989) — describes SG as exponential-fitting on `c`; Farrell et al., "Comparison of Scharfetter–Gummel schemes for (non-)degenerate semiconductors," and ChargeTransport.jl docs (github.com/WIAS-PDELib/ChargeTransport.jl) — both treat SG as a flux-discretization method, with quasi-Fermi/log-density being a *separate* choice of primary unknown.
- Confidence: **high**.

---

### 3. Metti, Xu, Liu 2016 — VERIFIED, but the writeup's terminology is slightly anachronistic

- M. S. Metti, J. Xu, C. Liu, "Energetically stable discretizations for charge transport and electrokinetic models," *J. Comput. Phys.* **306**, 1–18 (2016). DOI: 10.1016/j.jcp.2015.10.053.
- Verified via:
  - Penn State Pure entry: https://pure.psu.edu/en/publications/energetically-stable-discretizations-for-charge-transport-and-ele/ — confirms full citation.
  - ScienceDirect listing: https://www.sciencedirect.com/science/article/abs/pii/S0021999115007305 (paywalled, abstract visible in search results).
- **Content claim check:** the paper does indeed perform a logarithmic transformation of charge-carrier densities in a finite-element framework, proves a discrete energy estimate matching the continuous PNP energy law, and enforces positivity. The Penn State abstract describes it as a "finite element discretization … using a method of lines approach … enforces positivity" but **does not itself use the verbatim phrase "log-density formulation"** in the abstract.
- The phrase **"log-density formulation"** as a *name* for the Metti scheme appears to have been coined retroactively by Fu & Xu 2022 (see §4 below) and by subsequent reviewers. It is now standard usage in the entropy-stable PNP community, and attributing the name to Metti et al. is fair shorthand provided you note that the *term* came later.
- Confidence: **high** that the paper is real and uses logarithmic densities; **medium** that the explicit phrase "log-density formulation" is in the paper itself (the Penn State abstract does not contain it; without paywall access I cannot verify the body text contains it verbatim).

---

### 4. The "Liu, Maxian, Shen, Zitelli, Waltz, Monteiro 2022, CMAME 396, 115070" reference — **FABRICATED**

This is the most important verification result in this report.

**Truth:** arXiv:2105.01163 corresponds to a real paper, but **none of those six authors are correct, the volume/page numbers are wrong, and the DOI is off**:

- **Real paper:** Guosheng Fu, Zhiliang Xu (two authors, not six), "High-order space-time finite element methods for the Poisson–Nernst–Planck equations: positivity and unconditional energy stability," *Computer Methods in Applied Mechanics and Engineering* (CMAME) **395**, 115031 (2022). DOI: 10.1016/j.cma.2022.115031. arXiv:2105.01163.
  - arXiv abstract page (verified): https://arxiv.org/abs/2105.01163
  - Elsevier landing: https://linkinghub.elsevier.com/retrieve/pii/S0045782522002559
- **Volume:** **395** (not 396 as in the writeup).
- **Article number:** **115031** (not 115070).
- **Authors:** **Guosheng Fu** (Notre Dame) and **Zhiliang Xu** (Notre Dame). There is no "Maxian," "Shen," "Zitelli," "Waltz," or "Monteiro" on the paper.
- **What the paper actually does:** discretizes the **entropy variable** u_i = U'(c_i) = log c_i directly with finite elements in space, plus a discontinuous-Galerkin discretization in time. Provides positivity-preservation (because c_i = exp(u_i) > 0 by construction) and unconditional energy stability for *arbitrarily high order*. They credit the "log-density formulation" idea to Metti, Xu, Liu 2016 (per multiple secondary sources confirming this attribution).

**Hypothesis on origin of the fabricated citation:** The bogus author list looks like a hallucinated mash-up:
- "Liu" — plausibly Chun Liu (Metti/Xu/Liu 2016) or Hailiang Liu (energy-stable PNP, Iowa State).
- "Maxian" — Ondrej Maxian (NYU/Flatiron, biophysics) is a real researcher, but I found **no PNP/numerical-PDE publications** by him; his work is on Stokes/immersed-boundary biophysics. Likely an LLM hallucination, not a real co-author here.
- "Shen" — Jie Shen (Purdue / Eastern Inst. of Tech.) is associated with SAV schemes for PNP (Huang, Shen 2021, *SIAM J. Sci. Comput.*, doi:10.1137/20M1365417). Likely confused into the byline.
- "Zitelli, Waltz, Monteiro" — almost certainly hallucinated names; I found no record of any of these three publishing on PNP/energy-stable schemes. (There is a real "Monteiro" who publishes on DG methods at Georgia Tech — Renato D. C. Monteiro — but in optimization, not PNP.)

**Why I am confident this is fabrication, not a typo or misremembered citation:**
- arXiv:2105.01163 directly resolves to Fu & Xu and there is no other paper at that ID.
- CMAME volume 396 article 115070 *also* exists but is a completely unrelated paper (CMAME volume numbering for 2022 ran 388–401 with vol. 395 published mid-2022; vol. 396 articles 115050–115100 are mostly fluid mechanics / structural). Cross-checking on Elsevier shows article 115070 in vol. 396 is on a different topic (verified via search; no PNP paper at that exact CMAME volume/article).
- The author list is too-perfectly-six-names with each surname plausibly co-authoring something in numerical analysis. This is the typical signature of an LLM-confabulated citation — surnames sampled independently from related fields, glued onto a real arXiv ID.

**The writeup must be corrected to cite Fu & Xu 2022 with the right venue, volume, and article number.**

Confidence: **very high** that the six-author citation is fabricated, and **very high** that the correct paper is Fu & Xu 2022.

---

### 5. Genuine recent (2018–2025) energy-stable / log-density / SAV / IEQ work on PNP

All verified via arXiv/publisher pages.

| Citation | Verified | What it actually does |
|---|---|---|
| H. Liu, W. Maimaitiyiming, "Efficient, positive, and energy-stable schemes for multi-D Poisson–Nernst–Planck systems," *J. Sci. Comput.* **87**(3), 92 (2021). https://link.springer.com/article/10.1007/s10915-021-01503-1 | Yes | Finite-difference, primary unknown is c, positivity-preserving via flux limiting + energy stability. Not a log-density scheme. |
| C. Liu, C. Wang, S. M. Wise, X. Yue, S. Zhou, "A positivity-preserving, energy-stable and convergent numerical scheme for the Poisson–Nernst–Planck system," arXiv:2009.08076 (later *Math. Comp.* 2022). | Yes | Reformulates PNP as an H^{-1} gradient flow of a free-energy with a *singular logarithmic* potential, treats the log-term implicitly. Closely related to log-density but not identical (logs appear in the energy, not as primary unknowns). |
| F. Huang, J. Shen, "Bound/positivity preserving and energy stable scalar auxiliary variable (SAV) schemes for dissipative systems: Applications to Keller–Segel and PNP equations," *SIAM J. Sci. Comput.* **43**(3), B746–B759 (2021). DOI: 10.1137/20M1365417. | Yes | SAV-based scheme for PNP, primary unknown is c, auxiliary variable enforces bound preservation. Not log-density. |
| G. Fu, Z. Xu, "High-order space-time FEM for PNP …," CMAME **395**, 115031 (2022). | Yes | Direct entropy-variable / log-density FEM, arbitrary order, the strongest current example of the technique. |
| R. Lan, J. Li, Y. Cai, L. Ju, "Operator-splitting finite element method for PNP," (various 2022–2024 papers). | Partial — multiple authors with this surname, did not pin to a single canonical paper. | Splits the system; some variants use log-density. |
| C. Cancès, C. Chainais-Hillairet, B. Gaudeul, J. Fuhrmann, "Entropy and convergence analysis for two finite-volume schemes for a Nernst–Planck–Poisson system with ion-volume constraints," *Numer. Math.* **151**, 949–986 (2022). DOI: 10.1007/s00211-022-01279-y | Yes | Finite-volume, uses excess-chemical-potential flux (closely related to Slotboom/SG), proves entropy/energy decay. Strong electrochemistry-flavored work. |
| P. Farrell, N. Rotundo, D. H. Doan, M. Kantner, J. Fuhrmann, T. Koprucki, "Drift-Diffusion Models" (book chapter, 2017) and follow-ups on thermodynamically consistent SG. | Yes | Surveys flux discretizations including quasi-Fermi/Slotboom and excess-chemical-potential schemes. |

Confidence: **high** for each row marked "yes."

---

### 6. Quasi-Fermi formulation in semiconductor device simulation

- Markowich, P. A., *The Stationary Semiconductor Device Equations*, Springer Computational Microelectronics (1986). https://link.springer.com/book/10.1007/978-3-7091-3678-2 — verified to exist; surveys variable choices including Slotboom (Φn) and quasi-Fermi (φn).
- Markowich, Ringhofer, Schmeiser, *Semiconductor Equations*, Springer (1990). Verified — extends 1986.
- Selberherr, S., *Analysis and Simulation of Semiconductor Devices*, Springer (1984). Verified — describes Slotboom, SG, and Gummel iteration in industrial context.
- Brezzi, Marini, Pietra, "Two-dimensional exponential fitting and applications to drift-diffusion models," *SIAM J. Numer. Anal.* **26**(6), 1342–1355 (1989). https://epubs.siam.org/doi/10.1137/0726078 — verified; mixed-FE generalization of SG.

**Equivalence to log-density:** mathematically yes, the quasi-Fermi level φn satisfies n = N_C·exp((φn − ψ)/V_T), so ln n = (φn − ψ)/V_T + const, i.e. log-density and (quasi-Fermi minus potential) differ only by an affine map. **Discretizationally not equivalent**: choosing (φn, ψ) as primaries gives different stiffness matrices, conditioning, and boundary condition forms than (ln n, ψ) primaries. Most semiconductor codes use Slotboom or quasi-Fermi; most PNP-electrochemistry codes use ln c.

Confidence: **high**.

---

### 6a. Conceptual map: where does `u = ln c` fit in the entropy-stable PNP family?

The PNP free energy is E[c, ψ] = ∑_i ∫ c_i (ln c_i − 1) dx + ½ε ∫ |∇ψ|² dx − ∑_i z_i ∫ c_i ψ dx.

There are three well-established discretization strategies that exploit this energy structure, and they should not be conflated with each other:

- **(A) Direct log-density / entropy-variable primary.** Solve for u_i = ln c_i in a Galerkin space; recover c_i = exp(u_i) ≥ 0 by construction. Energy stability comes from chain-rule arguments on the discrete entropy. Lineage: Metti–Xu–Liu 2016 → Fu–Xu 2022. This is what the writeup means by "log-density formulation."
- **(B) Excess-chemical-potential / quasi-Fermi flux primary.** Discretize the flux as J_i = −D_i c_i ∇μ_i where μ_i = ln c_i + z_i ψ; primary unknowns can stay c_i (Voronoi FV) or switch to μ_i (FEM). Lineage: Slotboom 1969 → Brezzi–Marini–Pietra 1989 → Cancès–Chainais-Hillairet–Fuhrmann 2010s–2020s.
- **(C) SAV / IEQ approaches.** Keep c_i as primary, introduce a scalar auxiliary variable r(t) tracking the nonlinear part of the energy; treat the auxiliary linearly to get unconditional energy stability. Lineage: Shen–Xu–Yang 2018 (SAV for gradient flows) → Huang–Shen 2021 (SAV for PNP/Keller-Segel).

Strategy (A) directly enforces positivity but yields a strongly nonlinear discrete system at every step (you must Newton-solve through the exp). Strategy (B) requires a Bernoulli-function flux (SG-style) to preserve positivity but keeps the linear-algebra problem cleaner. Strategy (C) is unconditional in time-step size but does not by itself enforce positivity (later "bound-preserving" SAV variants do).

The writeup's choice of (A) is sensible for a Firedrake/FEM ORR-BV solver, where Newton iterations are already required by the Butler-Volmer nonlinearity, so the extra exp/log nonlinearity costs little marginal complexity.

Confidence: **high** for the strategy taxonomy; this is standard in the energy-stable PNP review literature (Bessemoulin-Chatard, Cancès, Liu reviews 2018–2022).

---

### 7. Electrochemistry-side use of `ln c` as primary unknown — the genuine gap

I searched specifically for ORR / fuel cell / RDE / RRDE / Li-ion battery PNP codes that use ln c as a primary unknown. **Findings:**

- I could find **no peer-reviewed electrochemistry-side ORR or RDE paper** that uses ln c as a primary unknown in a Galerkin / finite-element PNP–Butler-Volmer solver. The dominant practice in electroanalytical chemistry is to solve for c directly (sometimes with adaptive mesh refinement near the boundary layer).
- The COMSOL "Tertiary Current Distribution Nernst–Planck" interface uses c as primary unknown by default; "logarithmic" formulations are an option in the *semiconductor* module, not the electrochemistry module.
- Fuel-cell PNP literature (e.g. Bessler, Newman, Karan group at U. Calgary, Weber–Newman at LBNL) almost universally uses c-primary with log-mean upwinding for advection, not ln-c primary.
- The closest electrochemistry-side log-density work is in **lithium-ion cell modeling with concentrated-solution theory** (Newman group, Doyle–Fuller–Newman 1993 onwards), where chemical potential μ = μ_0 + RT·ln(γc) appears in fluxes — but they still discretize c, not ln c.
- Cancès et al. 2022 (cited above) is the closest mathematically rigorous PNP work that bridges to electrochemistry — but it uses excess-chemical-potential flux on c, not ln c primary.

**This means the writeup's claim that "ln c is novel for PNP-BV ORR solvers" is genuinely defensible**: the technique is borrowed from semiconductor / mathematical-PNP literature (Slotboom 1969 → Metti–Xu–Liu 2016 → Fu–Xu 2022) and applied to ORR-BV electrochemistry, where c-primary is the entrenched practice.

Confidence: **medium-high** (cannot prove a negative; some industrial code may do this without published documentation, but I found no peer-reviewed ORR/PNP-BV solver paper using ln c primary).

---

### VERIFICATION RESULTS

| Claimed citation | Verified? | Verdict |
|---|---|---|
| Slotboom 1973, *Solid-State Electronics*, "The PN-product in silicon" | **NO — chimera** | Wrong title for 1973, wrong year for the PN-product title. Use **Slotboom 1969, *Electronics Letters* 5, 677–678** for Slotboom variables; use **Slotboom 1977, *Solid-State Electronics* 20, 279–283** for the PN-product paper; use **Slotboom 1973, *IEEE TED* ED-20, 669–679** for the bipolar-transistor paper. |
| Scharfetter & Gummel 1969, *IEEE TED* 16(1), 64–77 | YES | Real, but it is a c-primary exponential-fitting flux scheme, **not** a log-density substitution. Cite carefully. |
| Metti, Xu, Liu 2016, *J. Comput. Phys.* 306, 1–18, doi:10.1016/j.jcp.2015.10.053 | YES | Real. Uses logarithmic transformation of densities in FEM with energy stability. The phrase "log-density formulation" was retroactively applied to the paper by Fu–Xu and others; medium confidence the verbatim phrase appears in the paper body. |
| "Liu, Maxian, Shen, Zitelli, Waltz, Monteiro 2022, CMAME 396, 115070, arXiv:2105.01163" | **NO — fabricated** | The arXiv ID is correct but everything else is wrong. Real paper: **G. Fu, Z. Xu**, CMAME **395**, **115031** (2022), doi:10.1016/j.cma.2022.115031. |
| Markowich 1986 *Stationary Semiconductor Device Equations* | YES | Real Springer book. |
| Brezzi, Marini, Pietra 1989, *SIAM J. Numer. Anal.* 26(6), 1342–1355 | YES | Real; mixed-FE exponential fitting. |
| Selberherr 1984 *Analysis and Simulation of Semiconductor Devices* | YES | Real Springer book. |
| Liu, Wang, Wise, Yue, Zhou, arXiv:2009.08076 | YES | Real; H^{-1} gradient flow scheme with log-energy. |
| Huang & Shen 2021, *SIAM J. Sci. Comput.* 43(3), B746–B759, doi:10.1137/20M1365417 | YES | Real SAV scheme for PNP. |
| H. Liu, W. Maimaitiyiming 2021, *J. Sci. Comput.* 87, 92 | YES | Real positivity/energy-stable FD scheme for multi-D PNP. |
| Cancès, Chainais-Hillairet, Gaudeul, Fuhrmann 2022, *Numer. Math.* 151, 949–986, doi:10.1007/s00211-022-01279-y | YES | Real entropy-FV scheme with size exclusion. |

---

### Key Takeaways (provenance picture for change #2)

1. **The log-density / `u = ln c` substitution is NOT novel to this PNP-BV solver.** Its semiconductor lineage (Slotboom 1969 → Markowich/Brezzi-Marini-Pietra/Selberherr 1980s → Farrell/Fuhrmann modern FV) and its mathematical-PNP lineage (Metti–Xu–Liu 2016 → Fu–Xu 2022 → Cancès–Fuhrmann 2022) are both well-established. The writeup correctly identifies this as borrowed, not invented.

2. **However, three of the writeup's citations are wrong as printed.** (a) The Slotboom citation conflates 1969 / 1973 / 1977 papers and must be corrected. (b) The Scharfetter–Gummel citation is technically correct but the writeup conflates SG (an exponential-fitting flux on c-primary) with the log-density substitution; these are distinct ideas that should be cited separately. (c) Most seriously, the "Liu, Maxian, Shen, Zitelli, Waltz, Monteiro 2022" reference is **fabricated**: the arXiv ID points to **Fu & Xu 2022, CMAME 395, 115031**, with no overlap in author list and a different volume/page.

3. **The genuinely novel angle is the application domain, not the substitution itself.** I could not locate any peer-reviewed electrochemistry-side ORR / RDE / fuel-cell / battery PNP–Butler-Volmer solver that adopts ln c as a primary unknown in a Galerkin FEM framework. Standard practice in that community is c-primary with adaptive meshing. So the writeup is right to claim that *transferring* the log-density formulation from semiconductor / mathematical-PNP literature into ORR-BV electrocatalysis is the contribution. The substitution is "old technology in a new setting."

4. **Slotboom variables and `u = ln c` are related but discretizationally distinct.** They map to each other through a logarithm and an additive shift by ψ/V_T, so they share the same continuous structure (positivity by construction, self-adjoint flux, chemical-potential interpretation). But Galerkin discretizations on (Φn, ψ), on (φn, ψ), and on (ln n, ψ) yield three different stiffness matrices with three different conditioning behaviors and three different boundary-condition forms. The writeup should not equate them.

5. **Recommended citation set for change #2 in the writeup, after corrections:** Slotboom 1969 *Electronics Letters* 5:677 (variable substitution) + Scharfetter–Gummel 1969 *IEEE TED* 16:64 (positivity-preserving FV flux, distinct mechanism) + Brezzi–Marini–Pietra 1989 *SIAM J. Numer. Anal.* 26:1342 (FEM generalization) + Metti–Xu–Liu 2016 *J. Comput. Phys.* 306:1 (FEM log-density energy stability for PNP) + **Fu & Xu 2022 *CMAME* 395:115031** (high-order space-time entropy-variable PNP) + Liu et al. 2020 (arXiv:2009.08076) and Huang–Shen 2021 (SAV) as alternative entropy-stable approaches + Cancès et al. 2022 *Numer. Math.* 151:949 as the modern entropy-FV bridge.

Sources (all URLs verified during this research):
- https://arxiv.org/abs/2105.01163
- https://linkinghub.elsevier.com/retrieve/pii/S0045782522002559
- https://www.sciencedirect.com/science/article/abs/pii/S0021999115007305
- https://pure.psu.edu/en/publications/energetically-stable-discretizations-for-charge-transport-and-ele/
- https://digital-library.theiet.org/doi/10.1049/el%3A19690510
- https://www.sciencedirect.com/science/article/abs/pii/0038110177901083
- https://link.springer.com/book/10.1007/978-3-7091-3678-2
- https://epubs.siam.org/doi/10.1137/20M1365417
- https://link.springer.com/article/10.1007/s10915-021-01503-1
- https://arxiv.org/abs/2009.08076
- https://link.springer.com/article/10.1007/s00211-022-01279-y
- https://github.com/WIAS-PDELib/ChargeTransport.jl
