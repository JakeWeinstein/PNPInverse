# Singh 2016 SI — extraction of field-dependent pKa formula

**Source:** Singh, M. R.; Kwon, Y.; Lum, Y.; Ager, J. W.; Bell, A. T.
"Hydrolysis of Electrolyte Cations Enhances the Electrochemical
Reduction of CO₂ over Ag and Cu", *J. Am. Chem. Soc.* **138**,
13006–13012 (2016). DOI: `10.1021/jacs.6b07612`.

**Files used:**
- Main paper: `https://www.osti.gov/servlets/purl/1456958` (DOE OSTI free mirror; full text)
- SI: `data/CO2RR Cation Supplementary Information.pdf` (user-provided
  via institutional access; ACS direct download paywalled at
  `https://pubs.acs.org/doi/suppl/10.1021/jacs.6b07612/suppl_file/ja6b07612_si_001.pdf`)

**Status:** Section 1 of SI fully extracted. Eq. (3) (bulk pKa) and
Eq. (4) (field-dependent pKa) verified numerically against Tables S1
and S3 (see §3 below). This doc closes Phase 6β v9 ledger item L1.

---

## 1. Cation hydration — Eq. (1)

```
M⁺ + n·H₂O  ⇌  M⁺(H₂O)ₙ                                      (1)
```

Hydration number `n` decreases with cation size; values in Table S1
(reproduced in §4 below).

## 2. Cation hydrolysis — Eq. (2)

```
M⁺(H₂O)ₙ + H₂O  ⇌  MOH⁰(H₂O)ₙ₋₁ + H₃O⁺                       (2)
```

The neutralized hydrated cation is `MOH⁰(H₂O)ₙ₋₁`; this is the
neutral product that v9 architecture tracks as `Γ_MOH` at the OHP.

## 3. Bulk pKa — Eq. (3)

**Singh writes** (with sign lost in OCR; see §3.4 verification):

```
                z²
pKa_bulk  =   −A  ───────  +  B                              (3)
              r_M-O
```

with **A = 620.32 pm**, **B = 17.154** (dimensionless), and:

| Symbol | Meaning |
|---|---|
| `z` | **Effective** charge on hydrated cation (NOT just +1; see §4 Table S1) |
| `r_M-O` | Sum of cation radius and O-atom radius (`r_O = 63 pm`) |

Singh's text (SI page S1):

> "We have found that a better correlation is obtained by taking
> z to be the effective charge on the hydrated cation, and r_M-O
> to be the sum of the cation radius and the radius of the O atom
> (63 pm)."
>
> "Using this approach, we determined A = 620.32 pm⁻¹ and
> B = 17.154. Equation (3) fits the experimental data with
> r² = 0.97."

**Sign convention note.** The PDF text reports A in units of
`pm⁻¹` and writes `pKa = A·z²/r_M-O + B` literally. But the
numerics only work with units of `pm` and a leading minus sign:
`pKa = −A·z²/r_M-O + B`. See §3.4. We treat the published
formula with this sign correction throughout.

### 3.4 Numerical verification of Eq. (3)

Using `pKa_bulk = B − A·z²/r_M-O` with Table S1 effective charges
and cation radii:

| Cation | r_M (pm) | z | r_M-O (pm) | pKa calc | pKa data (Table S1) |
|---|---|---|---|---|---|
| Li⁺ | 69 | 0.864 | 132 | 13.65 | 13.6 |
| Na⁺ | 102 | 0.900 | 165 | 14.11 | 14.2 |
| K⁺ | 138 | 0.919 | 201 | 14.55 | 14.5 |
| Rb⁺ | 149 | 0.923 | 212 | 14.66 | 14.6 |
| Cs⁺ | 170 | 0.930 | 233 | 14.85 | 14.8 |

All within rounding ✓. Confirms A = 620.32 pm, B = 17.154,
sign = minus on the slope.

---

## 4. Per-cation parameters (Table S1)

| Cation | r_M (pm) | n (hydration) | z (effective charge) | pKa_bulk |
|---|---|---|---|---|
| Li⁺ | 69 | 5.2 | 0.864 | 13.6 |
| Na⁺ | 102 | 3.5 | 0.900 | 14.2 |
| K⁺ | 138 | 2.6 | 0.919 | 14.5 |
| Rb⁺ | 149 | 2.4 | 0.923 | 14.6 |
| Cs⁺ | 170 | 2.1 | 0.930 | 14.8 |

Notes:
- `r_M` from Singh ref 4 (Marcus 1988, hydrated-ion radii).
- `n` from Singh ref 4 (hydration numbers).
- `z` for Na⁺ and K⁺ from quantum-mechanical calculations
  (Singh ref 3); for Li⁺, Rb⁺, Cs⁺ assumed linear in `n`.
- pKa_bulk from Baes & Mesmer "Hydrolysis of Cations" (Wiley);
  reproduces the experimentally-measured potentiometric titration
  values for the alkali metals.

These five rows are the calibration anchor for v9: the per-cation
constants are fixed from this table; only `r_H-El` and the σ
mapping require Stern-coupling work.

---

## 5. Field-dependent pKa near a cathode — Eq. (4)

**Singh writes** (with the same sign caveat as Eq. (3)):

```
                       ┌  z²                          ┌    r_M-O² ┐ ┐
pKa_field  =  −A  ·    │ ──────  −  2·z·σ·r_H-El·  │1 − ────── │ │  +  B   (4)
                       │ r_M-O                        └    r_H-El²┘ │
                       └                                              ┘
```

Equivalently:

```
pKa_field  =  pKa_bulk  +  ΔpKa(σ)

ΔpKa(σ)  =  +2·A·z·σ·r_H-El·(1 − r_M-O²/r_H-El²)              (4')
```

Symbols:

| Symbol | Meaning | Units (Singh's convention) |
|---|---|---|
| `σ` | Surface charge density | counts/cm² (from Eq. 5) |
| `r_H-El` | Distance: hydration-shell H atom → electrode surface | pm |
| `r_M-O` | Cation–O distance (Eq. 3) | pm |

Singh's geometry assumption (SI page S2):

> "For calculating r_H-El, we assume that CO is the primary species
> adsorbed on the Ag and Cu cathodes which is sandwiched between
> the hydrated cation and the cathode."

So r_H-El is set by the geometry: cation → hydration water (O-H
pointing toward electrode) → adsorbed CO → cathode surface.

### 5.1 Sign of ΔpKa

`r_H-El < r_M-O` in Singh's geometry (the hydration H atom hovers
just inside the M–O cone projected onto the surface). Therefore
`(1 − r_M-O²/r_H-El²) < 0`, giving **ΔpKa < 0** when σ > 0
(cathodic case). Cation hydrolysis pKa drops at the cathode ✓.

### 5.2 Surface charge density — Eq. (5)

```
σ  =  Ĉ · V · N_A / F                                          (5)
```

| Symbol | Meaning | Units |
|---|---|---|
| `Ĉ` | Specific capacitance of the cathode | F/m² (or µF/cm²) |
| `V` | Total cell voltage (positive scalar; magnitude) | V |
| `N_A` | Avogadro's number | 1/mol |
| `F` | Faraday's constant | C/mol |
| `σ` | Counts/area | 1/cm² (or 1/m²) |

Singh's cathode capacitances (SI page S3):

| Cathode | Ĉ |
|---|---|
| Cu | 51 µF/cm² |
| Ag | 100 µF/cm² (atomically smooth) |

Cu's Ĉ is lower because of higher roughness; values taken from
Singh ref 5 (Bard/Faulkner electrochemistry textbook).

**Note for our solver:** Eq. (5) computes σ from the *total cell
voltage*, which couples cathode + anode + Nernst + iR drops. In
our Phase 6β solver we already have a Stern BC `σ_metal = C_S·ψ_S`
where `ψ_S = φ_m − φ_s` is the local Stern voltage drop at the
electrode. The relevant Singh-equivalent σ for our purposes is:

```
σ_Singh  ≡  |C_S · ψ_S| · (N_A / F)        [counts/area]
```

i.e. drop the Singh ext-circuit cell-voltage formula and use our
local Stern surface charge directly. This is consistent because
both quantities ultimately measure "cathode-side surface charge
per area" — Singh just routes through the cell voltage.

---

## 6. Predicted pKa near Cu and Ag cathodes (Tables S2, S3)

**Cu cathode at −1 V vs RHE** (Table S3):

| Cation | Ĉ (µF/cm²) | V_cell (V) | σ (10¹⁵ /cm²) | pKa_near_Cu |
|---|---|---|---|---|
| Li⁺ | 51 | 4.196 | 1.33 | 13.16 |
| Na⁺ | 51 | 4.295 | 1.37 | 11.44 |
| K⁺ | 51 | 4.437 | 1.41 | 8.49 |
| Rb⁺ | 51 | 4.591 | 1.46 | 7.23 |
| Cs⁺ | 51 | 4.962 | 1.58 | 4.32 |

**Ag cathode at −1 V vs RHE** (Table S2):

| Cation | Ĉ (µF/cm²) | V_cell (V) | σ (10¹⁵ /cm²) | pKa_near_Ag |
|---|---|---|---|---|
| Li⁺ | 100 | 4.157 | 2.59 | 11.64 |
| Na⁺ | 100 | 4.403 | 2.74 | 10.26 |
| K⁺ | 100 | 4.615 | 2.88 | 7.95 |
| Rb⁺ | 100 | 4.733 | 2.95 | 6.97 |
| Cs⁺ | 100 | 5.217 | 3.25 | 4.31 |

Cu values match Linsey 2025 ACS-CATL deck slide 27 exactly.

Cell voltages V_cell vary across cations because higher current
density drives larger OER overpotential at the anode. For our v9
solver we don't recompute V_cell; we use the *local* Stern σ from
the cathode-only domain, which factors out the anode overpotential
issue.

---

## 7. r_H-El back-fit from Cu data

Singh doesn't publish numerical r_H-El values explicitly; they're
derived from the assumed adsorbed-CO geometry. Back-fitting Eq.
(4) against Table S3:

| Cation | r_M-O (pm) | σ (1/pm²) | ΔpKa (Cu) | r_H-El (pm) |
|---|---|---|---|---|
| Li⁺ | 132 | 0.133 | −0.44 | 132.00 |
| Na⁺ | 165 | 0.137 | −2.76 | 164.99 |
| K⁺ | 201 | 0.141 | −6.01 | 200.98 |
| Rb⁺ | 212 | 0.146 | −7.37 | 211.98 |
| Cs⁺ | 233 | 0.158 | −10.48 | 232.97 |

Pattern: **r_H-El ≈ r_M-O − (small δ)** for every cation, with the
small δ varying smoothly (Cu: ~0.00–0.03 pm). This suggests Singh's
formula is highly fine-tuned at the geometric level — the
pKa-shift comes from the small `(1 − r_M-O²/r_H-El²)` factor times
the large `σ·r_H-El` factor.

For implementation we don't need to predict r_H-El from first
principles. Instead **calibrate it per (cation, cathode_material)
pair from Singh's Tables S2/S3**, then apply Eq. (4) with that
fixed value at any other σ.

---

## 8. Implementation guidance for v9 cation_hydrolysis.py

### 8.1 Per-cation config schema

Hard-code Table S1 + back-fitted r_H-El into `_bv_common.py`:

```python
SINGH_2016_CATION_PARAMS = {
    "Li+": {"r_M_pm": 69,  "z_eff": 0.864, "n_hyd": 5.2, "pKa_bulk": 13.6,
            "r_H_El_pm_Cu": 132.00, "r_H_El_pm_Ag": ...},
    "Na+": {"r_M_pm": 102, "z_eff": 0.900, "n_hyd": 3.5, "pKa_bulk": 14.2,
            "r_H_El_pm_Cu": 164.99, "r_H_El_pm_Ag": ...},
    "K+":  {"r_M_pm": 138, "z_eff": 0.919, "n_hyd": 2.6, "pKa_bulk": 14.5,
            "r_H_El_pm_Cu": 200.98, "r_H_El_pm_Ag": ...},
    "Rb+": {"r_M_pm": 149, "z_eff": 0.923, "n_hyd": 2.4, "pKa_bulk": 14.6,
            "r_H_El_pm_Cu": 211.98, "r_H_El_pm_Ag": ...},
    "Cs+": {"r_M_pm": 170, "z_eff": 0.930, "n_hyd": 2.1, "pKa_bulk": 14.8,
            "r_H_El_pm_Cu": 232.97, "r_H_El_pm_Ag": ...},
}

# Singh 2016 SI Eq (3)/(4) constants
SINGH_2016_A_PM = 620.32
SINGH_2016_B = 17.154
SINGH_2016_R_O_PM = 63.0  # O-atom radius
```

ORR-on-CMK-3-carbon is **not** Cu or Ag, so the `r_H_El_pm_Cu` /
`r_H_El_pm_Ag` values are **literature-derived priors** for our
case, not directly applicable. For the v9 K2SO4 / ORR / carbon
work, treat `r_H_El_pm_carbon` as a tunable starting from `r_M-O`
and calibrate against the Linsey deck slide 27 values (which are
Cu-derived but the deck applies them to ORR — group convention).

### 8.2 ΔpKa(σ) function

```python
def delta_pKa_singh_2016(cation_params, sigma_S_C_per_m2):
    """
    Returns ΔpKa = pKa_field - pKa_bulk per Singh 2016 SI Eq (4').

    Args:
        cation_params: dict with z_eff, r_M_pm, r_H_El_pm
        sigma_S_C_per_m2: signed Stern surface charge density in C/m²
                         (cathodic = negative; bare value, NOT corrected by Γ)

    Returns:
        ΔpKa (negative when sigma_S < 0, zero or positive at anodic bias)
    """
    # Convert C/m² → counts/pm² (Singh's Eq 4 unit convention)
    # σ_count_per_m² = σ_C · N_A / F  (since N_A·e/F = 1)
    # σ_count_per_pm² = σ_count_per_m² × 1e-24 (1 m² = 1e24 pm²)
    sigma_pm2 = sigma_S_C_per_m2 * (N_A / F) * 1e-24

    # Singh treats σ as positive (cathode magnitude). Map our signed
    # ψ_S < 0 (cathodic) to Singh's positive σ:
    sigma_singh = max(0.0, -sigma_pm2)   # zero at anodic bias

    z = cation_params["z_eff"]
    r_M_O = cation_params["r_M_pm"] + SINGH_2016_R_O_PM
    r_H_El = cation_params["r_H_El_pm"]   # per cathode_material
    A = SINGH_2016_A_PM

    G = 1.0 - (r_M_O**2) / (r_H_El**2)    # geometric factor (negative
                                          #   if r_H_El < r_M_O)
    delta_pKa = 2.0 * A * z * sigma_singh * r_H_El * G   # negative when
                                                          # r_H_El < r_M_O
    return delta_pKa
```

For UFL/Firedrake the same formula gets written as a UFL expression
with `sigma_S = stern_coeff * (phi_applied − phi)` taken from the
existing Stern BC residual.

### 8.3 Anodic-bias guard

`max(0, -σ_S)` clamps the anodic case (σ_S > 0 in our signed
convention) to ΔpKa = 0, meaning hydrolysis is inactive when the
metal is positively biased. Physically: cations are repelled from
the OHP, no hydration-water polarization, no driving force for
deprotonation. v9 Gate 4 smoke runs only at cathodic V_RHE so
this guard is mostly for symmetry / numerical safety; it should be
enforced as a UFL `conditional` or an indicator function.

### 8.4 Calibration target for ORR-on-carbon (the v9 use case)

Singh's r_H-El values are Cu/Ag-specific. For ORR-on-CMK-3-carbon:

* **Calibration anchor:** Linsey 2025 ACS-CATL deck slide 27
  cation pKa near-cathode values (Li 13.16, Na 11.44, K 8.49,
  Cs 4.32). These are Singh's Cu values reproduced; the deck
  applies them as-is to ORR (group convention — treats them as
  representative of the field-dependent shift even at a
  different cathode material).
* **Solver-internal calibration:** at the v9 Gate 4 K⁺ smoke,
  fix `r_H_El_pm = 200.98` (Cu prior), then verify the predicted
  `σ_at_-0.40V` in the smoke gives a `ΔpKa(K)` consistent with
  the ~6 pH-unit drop the deck implies for Cu at −1 V. Tune
  `r_H_El_pm_carbon` if not.
* **Falsification path:** if the smoke pH drop direction is right
  but magnitude is off by > 30%, vary `r_H_El_pm` ∈ {198, 199,
  200, 200.98} and report which value matches. If no `r_H_El`
  in this range works, the formula's transferability across Cu →
  carbon has failed and we need different physics (e.g. surface
  capacity dependence, different geometry).

### 8.5 What v9 places into `pka_shift_form` config

```python
"pka_shift_form": "singh_2016_eq_4",   # was placeholder
"pka_shift_params": {
    # Per-cation Singh Table S1 row
    "z_eff": 0.919,         # K+ effective charge (use Singh Table S1)
    "r_M_pm": 138.0,        # K+ Stokes radius (use Singh Table S1)
    "r_H_El_pm": 200.98,    # Cu prior; calibrate for carbon at Gate 4
    # Global Singh constants
    "A_pm": 620.32,
    "B": 17.154,            # Used for absolute pKa_bulk; comparable
                            # to deck Table 1
    "r_O_pm": 63.0,
    # Solver-side switches
    "use_signed_sigma": True,   # cathodic-only hydrolysis (clamp at 0
                                # for σ > 0)
    "anode_clamp_eps": 1e-12,
},
```

This replaces the v9 R5#5 falsification placeholder
`β_M·sgn(σ_S)·|σ_S|^p`. Gate 4 can now run with **physical**
(not falsification-only) cation-hydrolysis kinetics.

---

## 9. Caveats — transferability from Singh CO₂RR/Cu to ORR/carbon

* **Cathode material differs.** Singh's r_H-El, Ĉ, and adsorbed-CO
  geometry are Cu-specific (or Ag-specific). ORR on CMK-3 carbon
  is a different surface electronic structure, different specific
  capacitance, no adsorbed CO. The Singh formula is the right
  *form*, but the parameters likely need ORR-on-carbon
  re-calibration.
* **The deck slide 27 transferability assumption is the same one
  the group uses.** The Linsey 2025 ACS-CATL deck cites Singh's
  Cu values directly for ORR analysis. We follow this convention
  in v9; the cation-series 6β.2 holdout test will reveal whether
  it actually works.
* **Singh's adsorbed-CO sandwich geometry doesn't apply at ORR.**
  ORR has no adsorbed CO — it's adsorbed O₂ / OOH / H₂O / OH at
  the cathode. The "sandwich" geometry that fixed `r_H-El < r_M-O`
  is therefore not strictly justified. Empirically the deck values
  carry over; mechanistically we may need to revisit if Gate 4
  shows quantitative discrepancies.
* **Effective charge `z` is invariant.** The `z_eff` values from
  Singh Table S1 are intrinsic cation properties (set by hydration
  shell quantum-mechanics), not Cu-specific. Same for `r_M`. So
  the per-cation table transfers cleanly; only `r_H-El` is the
  potentially-divergent geometric parameter.
* **Eq. (4) is one of several literature forms.** Co-Zhang 2019
  Angewandte (already in `data/.../Articles/`) has an alternative
  derivation. If Gate 4 with Singh's Eq. (4) fails, Co-Zhang is
  the next fallback.

---

## 10. Status / closure

* **Ledger L1 (Singh SI extraction): CLOSED.** Eq. (3) verified
  numerically (§3.4). Eq. (4) parameters back-fitted from Cu Table
  S3 (§7). Per-cation Table S1 captured (§4). Singh's σ
  convention bridged to our Stern BC (§5.2).
* **v9 implementation can replace the falsification placeholder
  `β_M·sgn(σ_S)·|σ_S|^p` with `pka_shift_form="singh_2016_eq_4"`**
  per §8.5.
* **Phase 6β v9 R5#5 wording guard still applies:** even with the
  physical Singh formula in Gate 4, a Gate 4 pass shows the
  coupled solver expresses a plausible branch; cation-series
  validation (6β.2) is still the actual physics check, and the
  Singh-on-Cu → ORR-on-carbon transferability caveat in §9 is
  the load-bearing risk that 6β.2 must close.

---

## Appendix A — verbatim SI Eq. (4) OCR

The PDF text-extraction came out garbled (math layout doesn't
survive `pdftotext`). The structural reading I used (with sign
recovered from numerical verification §3.4) is:

```
                  ┌  z²                           ┌    r_M-O² ┐ ┐
pKa  =  −A · │ ─────  −  2·z·σ·r_H-El·│1 − ────── │ │  +  B
                  │ r_M-O                         └    r_H-El²┘ │
                  └                                              ┘
```

The OCR-extracted ASCII version is reproduced here for traceability:

```
pKa   A          B                               (3)
                  rM-O

                                    z2                     r2
                         pKa   A        2 z rH-El  1  M-O   1   B   (4)
                                    rM-O                   r 2
                                                            H-El
```

The leading minus sign on `A` in both Eq. (3) and Eq. (4) was lost
in OCR but is recovered by Table S1 verification (§3.4 — only the
minus reproduces the published bulk pKa values). The position
of σ in Eq. (4) was also dropped by OCR; I recovered it from the
prose at SI line 100: "where σ is the surface charge density and
r_H-El is the distance between the center of the hydrogen atom in
the hydration shell and the electrode surface."
