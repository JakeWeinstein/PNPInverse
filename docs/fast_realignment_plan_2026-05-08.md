# Fast Realignment Plan — Real Experimental Setup

Date: 2026-05-08
Status: APPROVED via gpt-critique-loop session 20 (5 rounds, 41 issues
        raised, all addressed). Replaces the M3a/M3b cascade in
        `docs/ruggiero_realignment_plan.md` for purposes of *getting
        the forward solver onto the actual Seitz/Mangan experimental
        setup as fast as possible*. Not a substitute for the long-form
        plan once the dust settles — it deliberately defers regression
        testing, comprehensive unit tests, M4 cation-series
        calibration, M5 L_eff retune, and M6 Stern+cation joint
        sensitivity.
Author: Claude (under instruction "fuck it — too slow, just implement
        everything to the real experimental setup; I'll fix later").
Critique audit trail: `docs/CHATGPT_HANDOFF_20_fast-realignment/`.

## Pre-implementation note (from critique loop)

The "done" sweep this plan delivers is **structural validation**, not
a calibrated quantitative match to Mangan deck page 15. Pass D plots
(see Phase 3) demonstrate that the multi-ion + parallel topology
machinery converges and produces inspectable observables; they
should NOT be described as physical page-15 agreement. Calibration
(K0_R4e, ALPHA_R4E, Stern, cation radii) is M4-M6 work, deferred.

## TL;DR

Three structural mismatches separate the current production stack
from the real Seitz/Mangan experiment (per
`docs/seitz_mangan_data_folder_audit_2026-05-08.md` +
`docs/Ruggiero2022_JCatal_source_paper.md`):

1. **Counterion identity.** ClO₄⁻ analytic Bikerman → Cs⁺ + SO₄²⁻
   multi-ion (multi-steric Bikerman; closed-form shared-theta
   ansatz), at I=0.3 M, λ_D ≈ 0.55 nm.
2. **Reaction topology.** Sequential R_0 + R_1 → parallel
   R_2e + R_4e (E°=0.695 V, 1.23 V; free H₂O₂ never forms in 4e).
3. **IC adapter sites need multi-ion refactor.** Existing
   `picard_outer_loop_general` already handles parallel topology
   (M3a.3 silently landed); but the adapter sites in
   `forms_logc{,_muh}.py` overload `c_clo4_bulk` as the
   single-counterion anchor and the spatial IC interpolation
   uses 1:1-symmetric closures throughout.

Plan rips all three together with a structured `counterion_ctx`
refactor, anchored on V=+0.55 V (weakest cathodic drive within the
page-15 grid), driven by a single script that runs four passes
(pure-2e, pure-4e, mixed-literature, mixed-reduced-K0_R4e).

## Hard constraints (non-negotiable)

- **Begin with git checkpointing.**
- **No regression testing** vs Run C / legacy stacks.
- **No extensive unit tests.** Smoke tests only.
- venv-firedrake activation, C+D orchestrator, exponent_clip=100,
  physical E_eq, all per CLAUDE.md hard rules.
- **Multi-ion code path is opt-in.** Existing single-counterion
  ClO₄⁻ runs MUST take the legacy code path unchanged
  (byte-equivalent regression for `peroxide_window_3sp_bikerman_muh.py`).

## Source authority hierarchy

1. **Ruggiero 2022 J. Catal.** (`docs/Ruggiero2022_JCatal_source_paper.md`)
   — peer-reviewed source paper for the deck. Authoritative for:
   electrolyte composition, mass-transport constants (C(O₂)=1.2
   mol/m³ at pH 4-13, D(O₂)=1.9e-5 cm²/s), parallel 2e/4e reactions
   and E° values (0.695 V, 1.23 V), ionic strength I=0.3 M, RRDE
   constants (N=0.224, 1600 rpm).
2. **Seitz/Mangan data folder audit** — multi-document confirmation
   2019→2025. K₂SO₄/M₂SO₄ electrolyte (never ClO₄⁻), parallel
   2e/4e topology, Linsey 2025 ACS-CATL deck slide 13 cation
   hydrated radii.
3. **Linsey 2025 ACS-CATL deck slide 13** — cation pKa near
   cathode (Li⁺ 13.16, Na⁺ 11.44, K⁺ 8.49, Cs⁺ 4.32) and hydrated
   radii (Li⁺ 3.4, Na⁺ 2.8, K⁺ 2.3, Cs⁺ 2.2 Å).

## What the existing code already does (verified during critique)

- `picard_outer_loop_general` (`Forward/bv_solver/picard_ic.py:941`)
  is generic over N reactions with stoichiometry-aware signed flux
  balance, three-branch reversibility (`anodic_species` linear /
  affine constant / irreversible), and ambipolar `1/(2·D_H)` for
  H⁺. `topology_hint` dispatches to sequential closed-form vs
  general flux balance for the post-loop reconstruction. **The
  topology gate the realignment plan referenced has been replaced
  with this dispatch already.**
- `_build_picard_prefactors` correctly handles the three-branch
  reversibility per derivation v3 §3.
- `_assemble_n_reaction_system` builds the linear N×N system per
  v3 §4.
- `_validate_no_h_substrate` rejects H⁺ as cathodic/anodic
  substrate (v3 §9 item 11).
- `compute_surface_gamma` (single-ion) and `solve_stern_split`
  (single-ion 1:1 BKSA) exist but are 1:1-specific.
- `boltzmann.py:159-165` raises `NotImplementedError` for
  `len(bikerman) > 1`, but the comment is misleading: the
  multi-counterion Bikerman closed-form (shared-theta ansatz) IS
  closed-form per the project's own derivation in
  `docs/steric_analytic_clo4_reduction_handoff.md`.

## What's NOT in the existing code (the gap this plan closes)

- ⏳ Disabled-reaction guard for `ln(k0)` in log-rate forms.
- ⏳ Strict topology predicate for parallel 2e/4e.
- ⏳ Multi-counterion Bikerman closed-form (drop
  `len > 1` rejection in `boltzmann.py`).
- ⏳ `Forward/bv_solver/multi_ion.py` module (counterion_ctx
  builder + multi-ion phi_o solve + multi-ion γ + local λ_eff).
- ⏳ Multi-ion-aware spatial IC in both `forms_logc.py` and
  `forms_logc_muh.py`.
- ⏳ `multi_ion_enabled` opt-in flag with hard validation.
- ⏳ Cs⁺ + SO₄²⁻ analytic-counterion entries in `_bv_common.py`.

## Phase 0 — Git checkpointing (10 min, FIRST)

Before any code change.

1. `git status` — confirm dirty-file state. Many modified +
   untracked files from M3a.0/M3a.1/M3a.2/M3a.2.1 work; treat as
   pre-realignment baseline.
2. `git add -A` selectively: include `Forward/`, `scripts/`,
   `tests/`, `docs/`. Exclude `StudyResults/` (run outputs).
3. Commit:
   ```
   chore: checkpoint pre-fast-realignment state (M3a.0–M3a.2.1 landed)
   ```
4. Tag: `git tag pre-fast-realignment-2026-05-08`.
5. Branch: `git checkout -b fast-realignment-2026-05-08`.
6. No push.

Acceptance: `git log -1`, `git tag -l 'pre-fast-realignment*'`,
`git branch --show-current` returns `fast-realignment-2026-05-08`.

## Phase 1 — Disabled-rxn safety + Picard audit (1-2 days)

### 1.1 Disabled-reaction guard for log-rate forms (~half day)

Inside the `for j, rxn in enumerate(rxns_scaled):` loop in both
`forms_logc.py` and `forms_logc_muh.py`, before any
`fd.ln(k0_j)`:

```python
k0_model_j = float(rxn["k0_model"])
if k0_model_j <= 0.0 or bool(rxn.get("enabled", True)) is False:
    R_j = fd.Constant(0.0)
    bv_rate_exprs.append(R_j)
    bv_k0_funcs.append(_zero_k0_placeholder(R_space, j))
    bv_alpha_funcs.append(_zero_alpha_placeholder(R_space, j))
    continue
```

Smoke test: 2-rxn config with `k0_R4e = 0` → forms compile, no NaN.

### 1.2 Audit existing `picard_outer_loop_general` for parallel topology (1 day)

The realignment plan said parallel topology cold-fails
universally. That claim was based on a stale code path; M3a.3
(generic Picard) silently landed between the long-form plan and
2026-05-08. Need to determine the actual failure mode.

1. Add `verbose: bool = False` parameter to
   `picard_outer_loop_general`. When True, log per-iteration
   `(k, R_list, c_s_list, phi_o, psi_D, psi_S, gamma_s, eta_list,
   delta)`.
2. Run the existing
   `scripts/studies/peroxide_window_3sp_parallel_2e_4e.py` driver
   at one V (V_RHE = +0.55 V; weakest cathodic drive — Pass A
   first). Capture stdout to
   `StudyResults/fast_realignment_2026-05-08/picard_audit/`.
3. Diagnose:
   - Picard's `delta` doesn't shrink → bug in the generic Picard.
   - Picard converges but post-loop reconstruction goes
     non-physical → bug in `topology_hint='general'` reconstruction.
   - Picard converges, downstream Newton fails → spatial-IC issue
     (likely the 1:1 BKSA composite-ψ; addressed in Phase 2.4).
   - Everything converges → page-15 sweep just hadn't been re-run
     after `picard_outer_loop_general` landed.

No new Picard code is written until this audit identifies a
specific issue.

### 1.3 Centralized disabled-rxn helper + strict topology predicate (~half day)

In `picard_ic.py`:

```python
def _is_reaction_disabled(rxn: dict) -> bool:
    if not bool(rxn.get("enabled", True)):
        return True
    k0 = float(rxn.get("k0_model", rxn.get("k0", 0.0)))
    return k0 <= 0.0


def _is_parallel_2e_4e(reactions: list, h_idx: int) -> bool:
    """Strict predicate. Classifies from NOMINAL config (ignores enabled)."""
    if len(reactions) != 2:
        return False
    r2e, r4e = reactions
    if int(r2e.get("n_electrons", -1)) != 2: return False
    if int(r4e.get("n_electrons", -1)) != 4: return False
    s2 = r2e.get("stoichiometry", [])
    s4 = r4e.get("stoichiometry", [])
    if len(s2) < 3 or len(s4) < 3: return False
    if int(s2[1]) != +1: return False    # R_2e produces H2O2
    if int(s4[1]) != 0:  return False    # R_4e doesn't touch H2O2
    if int(s2[0]) != -1 or int(s4[0]) != -1: return False
    if int(s2[h_idx]) != -2: return False
    if int(s4[h_idx]) != -4: return False
    if not bool(r2e.get("reversible", False)): return False
    if bool(r4e.get("reversible", False)):     return False
    return True
```

Used at adapter sites for safety asserts AND in the topology
classifier (which classifies from nominal config; disabled
reactions don't change topology). In
`_assemble_n_reaction_system`, disabled rxns produce trivial
rows: `M[j,j] = 1`, `b[j] = 0` ⇒ `R_j = 0`. Pure-2e probe is
"parallel topology + R_4e disabled," not "1-rxn sequential" —
preserves topology dispatch.

## Phase 2 — Multi-ion infrastructure (3-5 days)

### 2.1 Multi-steric Bikerman closed-form in `boltzmann.py` (1 day)

Re-derive the multi-counterion shared-theta closure (already in
`docs/steric_analytic_clo4_reduction_handoff.md` for K=1; trivial
generalization):

```text
For each analytic ion k (steric):
  c_k(φ) = c_b_k · exp(-z_k·φ) · (1 - A_dyn(φ))
                 / (θ_b + Σ_{k'} a_{k'}·c_b_{k'}·exp(-z_{k'}·φ))

with A_dyn(φ) = Σ_dyn a_i · c_i_dyn(φ)
     θ_b = 1 - A_dyn_bulk - Σ_k a_k · c_b_k

The denominator is the same for every ion (shared theta);
no coupled local NL solve needed.
```

In `boltzmann.py`:

- Replace `len(bikerman) > 1` rejection with a generalized loop.
- For each bikerman entry, build a `StericBoltzmannBundle` with
  numerator `c_b_k · exp(-z_k·φ_clamped) · (1 - A_dyn_local)` and
  shared denominator `θ_b + Σ_k' a_k'·c_b_k'·exp(-z_k'·φ_clamped)`.
- Return a list of bundles instead of one. Caller sums each
  bundle's `charge_density` into the Poisson residual and each
  bundle's `packing_contribution` into the dynamic-species `theta`.
- Bulk pack-fraction check: `θ_b > 0` (ValueError on violation).
- Double-counting guard: no analytic entry's (z, c_b) duplicates
  any dynamic species'.

### 2.2 Hard-sphere `a_nondim` from physical hydrated radii

Add to `scripts/_bv_common.py`:

```python
# Hard-sphere excluded volume from hydrated radii.
# a_phys = (4/3)·π·r³·N_A in m³/mol; a_nondim = a_phys · C_SCALE.

# Cs+ at r=2.2 Å (Linsey 2025 deck slide 13)
A_CSPLUS_HAT = 3.23e-5

# SO4²⁻ at r=2.4 Å (Marcus, placeholder for fast plan)
A_SO4_HAT = 4.20e-5

C_CSPLUS = 199.9        # mol/m³, electroneutrality with H+ + SO4²⁻
C_SO4    = 100.0        # mol/m³, Ruggiero §2 sulfate concentration
C_CSPLUS_HAT = C_CSPLUS / C_SCALE
C_SO4_HAT    = C_SO4 / C_SCALE


DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC: Dict[str, Any] = {
    "z": +1,
    "c_bulk_nondim": C_CSPLUS_HAT,
    "phi_clamp": 50.0,
    "steric_mode": "bikerman",
    "a_nondim": A_CSPLUS_HAT,
    "label": "Cs+",
}

DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC: Dict[str, Any] = {
    "z": -2,
    "c_bulk_nondim": C_SO4_HAT,
    "phi_clamp": 50.0,
    "steric_mode": "bikerman",
    "a_nondim": A_SO4_HAT,
    "label": "SO4--",
}
```

Bulk-packing check (passes):
- a_Cs · c_Cs_nondim = 3.23e-5 · 166.58 = 5.38e-3
- a_SO4 · c_SO4_nondim = 4.20e-5 · 83.33 = 3.50e-3
- Total = 8.88e-3 ⇒ θ_b ≈ 0.991 ✓

### 2.3 New `Forward/bv_solver/multi_ion.py` module (1-2 days)

Single source of truth for multi-ion machinery:

```python
@dataclass(frozen=True)
class CounterionConfig:
    z: int
    c_bulk_nondim: float
    a_nondim: float
    steric_mode: str  # "ideal" | "bikerman"
    phi_clamp: float = 50.0
    label: str = ""


def build_counterion_ctx(
    counterions: list[CounterionConfig | dict],
    a_dyn: list[float],
    c_dyn_bulk: list[float],
    z_dyn: list[int],
) -> dict:
    """Build canonical counterion context with derived theta_b.

    Single producer of theta_b; downstream reads from this ctx.
    """
    ions = [_normalize(e) for e in counterions]
    A_dyn_bulk = sum(a * c for a, c in zip(a_dyn, c_dyn_bulk))
    A_an_bulk = sum(ion["a_nondim"] * ion["c_bulk_nondim"]
                    for ion in ions if ion["steric_mode"] == "bikerman")
    theta_b = 1.0 - A_dyn_bulk - A_an_bulk
    if theta_b <= 0:
        raise ValueError(f"theta_b <= 0 (got {theta_b})")
    return {
        "ions": ions,
        "z_dyn": list(z_dyn),
        "c_dyn_bulk": list(c_dyn_bulk),
        "a_dyn": list(a_dyn),
        "theta_b": theta_b,
    }


def _solve_outer_phi_multiion(
    *,
    c_dyn_outer: list[float],     # NOT bulk — Picard-side OHP-edge values
    a_dyn: list[float],
    z_dyn: list[int],
    ions: list[dict],
    theta_b: float,
    bracket: tuple[float, float] = (-50.0, +50.0),
    tol: float = 1e-12,
    max_iter: int = 100,
) -> float:
    """Solve outer-region electroneutrality for phi_o via bisection.

    Uses full A_dyn(phi_o) sum (NOT just a_H·H_o); analytic-ion
    concentrations from the multi-steric closure.
    """
    A_dyn_outer = sum(a * c for a, c in zip(a_dyn, c_dyn_outer))
    def residual(phi_o: float) -> float:
        denom = theta_b + sum(
            ion["a_nondim"] * ion["c_bulk_nondim"]
            * math.exp(-ion["z"] * phi_o) for ion in ions
        )
        ck = [ion["c_bulk_nondim"] * math.exp(-ion["z"] * phi_o)
              * (1 - A_dyn_outer) / denom for ion in ions]
        rho_dyn = sum(z * c for z, c in zip(z_dyn, c_dyn_outer))
        rho_an = sum(ion["z"] * c for ion, c in zip(ions, ck))
        return rho_dyn + rho_an
    # standard bisection + bracket fallback
    ...


def compute_surface_gamma_multiion(
    *,
    H_o: float,                     # outer-region H+ scalar
    psi_D: float,                   # diffuse-layer drop at OHP
    a_H: float,                     # H+ Bikerman size
    ions: list[dict],               # each entry has {"z","a","c_outer"}
                                    # c_outer is the outer-region value,
                                    # NOT bulk
) -> float:
    """Multispecies γ_s at OHP using outer-region anchors.

    γ_s = 1 / (1 + a_H·H_o·(e^(-ψ_D)−1)
                 + Σ_k a_k·c_k_outer·(e^(-z_k·ψ_D)−1))
    """
    if a_H == 0.0 and not ions:
        return 1.0
    denom = 1.0 + a_H * H_o * (_safe_exp(-psi_D) - 1.0)
    for ion in ions:
        denom += ion["a_nondim"] * ion["c_outer"] * (
            _safe_exp(-ion["z"] * psi_D) - 1.0
        )
    if not math.isfinite(denom) or denom <= 0.0:
        return 1e-300
    return 1.0 / denom


def effective_debye_length_local(
    *, phi_o: float, ions: list, theta_b: float,
    z_dyn: list[int], c_dyn_outer: list[float], a_dyn: list[float],
    poisson_coeff: float, dphi: float = 1e-4,
) -> float:
    """Local λ_eff = sqrt(eps / |dρ/dφ|_outer) via finite difference.

    Uses the same multi-steric closure as Phase 2.1.  Differs from the
    bulk Σz²c form by up to ~20× when |φ_o| ≳ 2 (tested on 2:1
    sulfate); at I=0.3 M with Stern dominating, the correct screening
    coefficient is critical for the linear-Debye Stern split.
    """
    def rho_at(phi: float) -> float:
        denom = theta_b + sum(
            ion["a_nondim"] * ion["c_bulk_nondim"]
            * math.exp(-ion["z"] * phi) for ion in ions
        )
        A_dyn = sum(a * c for a, c in zip(a_dyn, c_dyn_outer))
        ck = [ion["c_bulk_nondim"] * math.exp(-ion["z"] * phi)
              * (1 - A_dyn) / denom for ion in ions]
        rho_dyn = sum(z * c for z, c in zip(z_dyn, c_dyn_outer))
        rho_an = sum(ion["z"] * c for ion, c in zip(ions, ck))
        return rho_dyn + rho_an
    drho_dphi = (rho_at(phi_o + dphi) - rho_at(phi_o - dphi)) / (2 * dphi)
    inv_lambda_sq = max(-drho_dphi, 1e-30) / poisson_coeff
    return math.sqrt(1.0 / inv_lambda_sq)
```

### 2.4 Spatial IC: ψ-vs-φ split + multi-ion interpolation (1-2 days)

In `forms_logc{,_muh}.py:_try_debye_boltzmann_ic*`, replace the
1:1-symmetric composite-ψ closure with a properly-decomposed
multi-ion seed:

```python
# After Picard converges → (R_list, c_s, H_o, psi_D, psi_S, gamma_s, phi_o)

# Spatial profile decomposition (matched-asymptotic):
#   phi(y) = phi_outer(y) + psi(y)
# where phi_outer is slow (varies on L_REF scale), psi is fast (λ_eff scale).
#
# IC APPROXIMATION: we use linear phi_outer(y) instead of solving
# _solve_outer_phi_multiion at every y.  Acceptable because Phase 2.4
# uses local c_dyn_outer and lets Newton repair the profile.  If
# Newton failures localize to the initial EDL shape, the first
# escalation is solving _solve_outer_phi_multiion at several y
# nodes and interpolating.

lambda_eff = effective_debye_length_local(
    phi_o=phi_o, ions=ions, theta_b=theta_b,
    z_dyn=z_dyn, c_dyn_outer=c_dyn_outer_at_ohp, a_dyn=a_dyn,
    poisson_coeff=poisson_coefficient,
)

def psi_profile(y):
    """Diffuse-layer drop, linear-Debye seed.  ψ(0)=ψ_D, ψ(∞)=0."""
    return psi_D * exp(-y / lambda_eff)

def phi_outer_profile(y):
    """Slow outer-region profile.  Linear from phi_o at OHP to 0 at bulk."""
    return phi_o * (1.0 - min(y / L_REF, 1.0))

def c_dyn_outer_profile(y):
    """Linear interp from Picard surface to bulk.  c_O2_outer, c_H2O2_outer,
    c_H_outer at the OHP-side edge are O_s, P_s, H_o (NOT bulk)."""
    frac = min(y / L_REF, 1.0)
    return [
        (1 - frac) * O_s + frac * O_b,
        (1 - frac) * P_s + frac * P_b,
        (1 - frac) * H_o + frac * H_b,
    ]

def gamma_psi_profile(y):
    """Spatial gamma from shared-theta closure.  γ_psi(y) = θ(y)/θ_outer."""
    phi_y = phi_outer_profile(y) + psi_profile(y)
    A_dyn_y = sum(a * c for a, c in zip(a_dyn, c_dyn_outer_profile(y)))
    denom_y = theta_b + sum(
        ion["a_nondim"] * ion["c_bulk_nondim"]
        * math.exp(-ion["z"] * phi_y) for ion in ions
    )
    theta_y = (1 - A_dyn_y) * theta_b / denom_y
    A_dyn_outer = sum(a * c for a, c in zip(a_dyn, c_dyn_outer_at_ohp))
    denom_outer = theta_b + sum(
        ion["a_nondim"] * ion["c_bulk_nondim"]
        * math.exp(-ion["z"] * phi_o) for ion in ions
    )
    theta_outer = (1 - A_dyn_outer) * theta_b / denom_outer
    return theta_y / theta_outer

# Spatial IC for each species:
#   log_c_i_seed(y) = log(c_i_outer(y)) - z_i·psi(y) + log_gamma_psi(y)
# Boltzmann shift uses ψ (NOT absolute φ).  Neutrals (z_i=0) only get
# the gamma_psi factor.
for i in range(n_species):
    z_i = z_dyn[i]
    log_c_init = lambda y, i=i: (
        log(max(c_dyn_outer_profile(y)[i], 1e-300))
        - z_i * psi_profile(y)
        + log(gamma_psi_profile(y))
    )
    # Interpolate to FE Function on log-c (or log-c plus em·z·phi for muh)

# Poisson primary variable phi_init(y):
phi_init = lambda y: phi_outer_profile(y) + psi_profile(y)

# muh formulation: mu_H_init(y) = u_H_init(y) + em·z_H·phi_init(y)
mu_H_init = lambda y: log(max(c_dyn_outer_profile(y)[h_idx], 1e-300)) - psi_profile(y) \
                      + log(gamma_psi_profile(y)) + em * z_H * phi_init(y)

# Stern split: linear-Debye fallback with lambda_eff_local.
psi_D_via_linear_debye = stern_coeff * full_drop * lambda_eff \
                       / (poisson_coefficient + stern_coeff * lambda_eff)
```

Single-ion legacy path: preserved via the `multi_ion_enabled=False`
adapter branch — calls existing 1:1 BKSA code unchanged.

## Phase 3 — One-shot driver with four passes (~half day)

New `scripts/studies/peroxide_window_3sp_parallel_2e_4e_csplus_so4.py`:

```python
from scripts._bv_common import (
    make_bv_solver_params,
    THREE_SPECIES_LOGC_BOLTZMANN,
    DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,
    DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
    PARALLEL_2E_4E_REACTIONS,
    K0_HAT_R2E, K0_HAT_R4E,
)

PAGE_15_V_RHE_GRID = [...]  # 25 points spanning [-0.40, +0.55]
ANCHOR_V_RHE = +0.55          # weakest cathodic drive (R2 Issue 16)

def make_sp(*, k0_r2e_factor=1.0, k0_r4e_factor=1.0):
    rxns = [dict(r) for r in PARALLEL_2E_4E_REACTIONS]
    rxns[0]["k0"] = K0_HAT_R2E * k0_r2e_factor
    rxns[1]["k0"] = K0_HAT_R4E * k0_r4e_factor
    return make_bv_solver_params(
        eta_hat=eta_hat,
        dt=DT_HAT, t_end=T_END_HAT,
        species=THREE_SPECIES_LOGC_BOLTZMANN,
        formulation="logc_muh",
        log_rate=True,
        bv_reactions=rxns,
        boltzmann_counterions=[
            DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,
            DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
        ],
        multi_ion_enabled=True,    # OPT-IN; required when len > 1
        stern_capacitance_f_m2=0.10,
        initializer="debye_boltzmann",
        exponent_clip=100.0,
        u_clamp=100.0,
    )

# Pass A: pure-2e (k0_R4e disabled via Phase 1.1 guard)
sp_A = make_sp(k0_r2e_factor=1.0, k0_r4e_factor=0.0)
result_A = solve_grid_per_voltage_cold_with_warm_fallback(
    sp_A, v_rhe_grid=PAGE_15_V_RHE_GRID, anchor_v_rhe=ANCHOR_V_RHE, ...
)

# Pass B: pure-4e
sp_B = make_sp(k0_r2e_factor=0.0, k0_r4e_factor=1.0)
result_B = solve_grid_per_voltage_cold_with_warm_fallback(sp_B, ...)

# Pass C: mixed at literature K0 (exploratory only; expected R_4e domination)
sp_C = make_sp(k0_r2e_factor=1.0, k0_r4e_factor=1.0)
result_C = solve_grid_per_voltage_cold_with_warm_fallback(
    sp_C, v_rhe_grid=[ANCHOR_V_RHE], anchor_v_rhe=ANCHOR_V_RHE, ...
)

# Pass D: mixed with reduced K0_R4e (LADDER, not fixed factor)
PASS_D_LADDER = [1e-12, 1e-15, 1e-18, 1e-21, 1e-24]
PASS_D_TEST_VOLTAGES = [-0.40, -0.20, 0.0, +0.20, +0.55]
result_D = {}
for ratio in PASS_D_LADDER:
    sp_D = make_sp(k0_r2e_factor=1.0, k0_r4e_factor=ratio)
    res = solve_grid_per_voltage_cold_with_warm_fallback(
        sp_D, v_rhe_grid=PASS_D_TEST_VOLTAGES, anchor_v_rhe=ANCHOR_V_RHE, ...
    )
    # Record whether ≥3 of 5 V points show non-zero R_2e AND R_4e
    if _ratio_passes_smoke(res):
        # Promote to full grid
        sp_D_full = make_sp(k0_r2e_factor=1.0, k0_r4e_factor=ratio)
        result_D[ratio] = solve_grid_per_voltage_cold_with_warm_fallback(
            sp_D_full, v_rhe_grid=PAGE_15_V_RHE_GRID,
            anchor_v_rhe=ANCHOR_V_RHE, ...
        )
        break
```

Output: `StudyResults/fast_realignment_2026-05-08/`:
- `pass_A/summary.json`, `pass_A/overlay.png`, ...
- `pass_B/summary.json`, ...
- `pass_C/summary.json` (single-V exploratory)
- `pass_D/{ratio}/summary.json`, ...
- `summary.md` — overall what-happened.

**Important labeling caveat (per critique R5):** Pass D plots
demonstrate that the multi-ion + parallel topology machinery
produces inspectable observables; they DO NOT represent calibrated
agreement with Mangan deck page 15. State this plainly in the
output `summary.md`. Calibration is M4 work, deferred.

## Phase 4 — First production sweep (1 day if it converges)

Run the driver. Per-pass smoke gate:

| Pass | Smoke acceptance | Done acceptance |
|---|---|---|
| A | ≥ 5/25 V_RHE converged with non-zero gross R_2e | ≥ 15/25 |
| B | anchor + warm-walk single-V smoke (1/25) | ≥ 15/25 |
| C | anchor + warm-walk single-V smoke (1/25; reports R_4e dominance) | NOT REQUIRED |
| D | smallest ladder factor with ≥ 5/25 converged + non-zero R_2e AND R_4e | ≥ 15/25 at chosen ratio |

If any of {A, B, D} ≥ 15/25 is unreachable in the time budget,
ASK THE USER before lowering. Do not silently downgrade.

Watch list:

- Picard convergence rate (Phase 1.2 audit findings).
- Newton convergence rate (multi-ion γ behaving sanely).
- NaN / unbounded residuals (likely failure modes: H_o → 0,
  pack-fraction γ → 0, exponent overflow).
- Anchor convergence at V_RHE = +0.55 V. If anchor cold-fails, V
  scan {+0.55, +0.50, +0.45, +0.40, +0.35, +0.30, +0.20, 0.0}.

## Phase 5 — Fix breaks (variable; expected longest)

### 5a. Picard non-convergence
- omega → 0.3 / 0.2.
- max_iters → 100.
- Per-iter printout to localize divergent sub-iterate.

### 5b. Newton non-convergence at multi-ion EDL
- Continuation in I (ramp from legacy ClO4⁻ bulk concentrations
  toward Ruggiero values across N pseudo-time steps).
- Continuation in `a_nondim` starting from 0 (NOT A_DEFAULT — the
  literature values, scaled by 0...1 ramp).
- Mesh refinement Ny 200 → 400 → 800.

### 5c. Anchor relocation
V_RHE scan {+0.55, +0.50, +0.45, +0.40, +0.35, +0.30, +0.20, 0.0}.

### 5d. `a_nondim` calibration
If `theta_b > 0` is satisfied by the hard-sphere derivation but
EDL pack-fraction violates `Σ a_k c_k(EDL) ≤ 1` (Bikerman
saturation), reduce `a_nondim` proportionally until bounded.
Document empirical vs literature.

### 5e. Legacy driver regression
If Phase 1 / Phase 2 changes break legacy
`peroxide_window_3sp_bikerman_muh.py` despite the
`multi_ion_enabled=False` opt-in: the regression is unintended
(byte-equivalent legacy was a hard requirement). Investigate and
fix; legacy must remain green.

### 5f. Concrete legacy warm-start (R3 Issue 11 fallback)

```python
def _legacy_warmstart_to_target(*, sp_legacy, sp_target, v_anchor, mesh_args):
    # 1. Build legacy ctx (ClO4 + sequential + 3sp logc_muh)
    ctx_legacy = build_ctx(sp_legacy, mesh_args)
    set_initial_conditions(ctx_legacy, sp_legacy)  # debye_boltzmann
    solve_steady_state(ctx_legacy, v_anchor)
    # 2. Build target ctx (Cs+/SO4 + parallel + 3sp logc_muh)
    ctx_target = build_ctx(sp_target, mesh_args)
    # 3. Copy U state subfunction-by-subfunction
    for src_sub, dst_sub in zip(
        ctx_legacy["U"].subfunctions, ctx_target["U"].subfunctions
    ):
        dst_sub.dat.data[:] = src_sub.dat.data_ro
    ctx_target["U_prev"].assign(ctx_target["U"])
    # 4. Solve target residual at v_anchor
    solve_steady_state(ctx_target, v_anchor)
    return ctx_target
```

Both stacks use `THREE_SPECIES_LOGC_BOLTZMANN` ⇒ subfunction
indices align ([O₂, H₂O₂, H⁺, φ]) and `mu_h_idx = 2` for both.

### 5g. IC approximation escalation (R5 Note 2)

If Newton failures localize to the initial EDL shape (large jumps
in the first few Newton iters at high-|φ_o| V_RHE points), escalate
the spatial IC: solve `_solve_outer_phi_multiion()` at several y
nodes (e.g. {0, λ_eff, 5·λ_eff, L_REF/2, L_REF}) and interpolate
between, instead of the linear `phi_outer(y) = phi_o · (1 − y/L_REF)`.

## What this plan deliberately skips

- All M3a.0/M3a.1/M3a.2/M3a.2.1/M3a.3/M3b.1-5/M3c/M4/M5/M6
  substaging from `docs/ruggiero_realignment_plan.md`.
- Generic N-reaction Picard. (Already exists; just audit.)
- Asymmetric composite-ψ derivation. Replaced by linearized-Debye +
  ψ-vs-φ split + Newton repair.
- HSO₄⁻ / SO₄²⁻ acid-base equilibrium. Negligible at pH 4
  (pKa₂≈1.99).
- OH⁻ as tracked species. K_w-coupled.
- Cation-swap parameterization (Li/Na/K). Hardcode Cs⁺.
- L_eff retune (M5).
- Stern + cation joint sensitivity (M6).
- Inverse work.
- Full BKSA nonlinear Stern split for 2:1 multi-ion. Replaced by
  linearized-Debye Stern split.
- Adjoint tape hygiene (verify on inverse re-entry; not blocking).
- Regression vs Run C.
- Full pytest sweep.
- Quantitative shape/magnitude bands vs page-15.
- The Tafel slope analysis xlsx request to Mangan team (per data
  audit); for now match against derived figure from Yash-Trends.

## Risks (acknowledged, not exhaustively mitigated)

| Risk | Where | Likelihood | Worst case | Triage |
|---|---|---|---|---|
| Picard's `general` topology path has a bug | 1.2 | Medium | Need to rewrite generic Picard | 5a fallbacks |
| Multi-ion bulk anchor at I=0.3 M doesn't cold-converge | 4 | Medium-high | Zero converged points | 5b/5c |
| Multi-ion γ FD-derivative is inaccurate at large \|φ_o\| | 2.3 | Low | Stern split jumps branches | Use larger `dphi`; sanity-check vs analytic in dilute limit |
| Spatial IC linear phi_outer is too crude | 2.4 | Medium | Newton fails at high-\|V_RHE\| points | 5g escalation |
| K0_PHYS_R4E placeholder produces R_4e dominance | 4 | High | Pass C output is uninterpretable | Acknowledged; Pass D ladder gives inspectable mixed |
| Anchor at +0.55 V cold-fails | 4 | High | Need anchor V scan | 5c |
| Legacy regression despite opt-in flag | 1, 2 | Low | Existing 25/25 stack breaks | 5e: investigate and fix |

## Sequencing

```
Phase 0 (10 min)
  → Phase 1.1 (~half day)
  → Phase 1.2 (1 day)
  → Phase 1.3 (~half day)
  → Phase 2.1 (1 day)
  → Phase 2.2 (couple hours)
  → Phase 2.3 (1-2 days)
  → Phase 2.4 (1-2 days)
  → Phase 3 (~half day)
  → Phase 4 (1 day)
  → Phase 5 (variable; expected longest)
```

Estimated total: **7-12 days** to first end-to-end Cs⁺/SO₄²⁻ +
parallel-2e/4e converged page-15 sweep across Pass A + Pass B +
Pass D. Original M3a/M3b plan was ~4-6 weeks.

## Code-surface checklist

- ⏳ `Forward/bv_solver/forms_logc.py:707-` — disabled-rxn guard
  (Phase 1.1).
- ⏳ `Forward/bv_solver/forms_logc_muh.py:782-` — same (Phase 1.1).
- ⏳ `Forward/bv_solver/picard_ic.py:941-` — add `verbose` param +
  per-iter logging (Phase 1.2).
- ⏳ `Forward/bv_solver/picard_ic.py` — add `_is_reaction_disabled`,
  `_is_parallel_2e_4e` helpers (Phase 1.3).
- ⏳ `Forward/bv_solver/boltzmann.py:159-165` — drop
  `len(bikerman) > 1` rejection; multi-steric closed-form
  (Phase 2.1).
- ⏳ NEW `Forward/bv_solver/multi_ion.py` — counterion ctx +
  multi-ion phi_o + multi-ion γ + local λ_eff (Phase 2.3).
- ⏳ `scripts/_bv_common.py` — Cs⁺ + SO₄²⁻ entries with hard-sphere
  `a_nondim`; `multi_ion_enabled` flag in
  `make_bv_solver_params` with hard validation (Phase 2.2 + 2.3).
- ⏳ `Forward/bv_solver/forms_logc{,_muh}.py:_try_debye_boltzmann_ic*`
  — multi-ion spatial IC with ψ-vs-φ split (Phase 2.4).
- ⏳ NEW `scripts/studies/peroxide_window_3sp_parallel_2e_4e_csplus_so4.py`
  — driver (Phase 3).
- ⏳ NEW `tests/test_disabled_reaction_log_rate.py` — smoke
  (Phase 1.1).
- ✅ Already landed: electron-weighted observables, parallel
  reaction config plumbing, C_O2 = 1.2 mol/m³, generic
  `picard_outer_loop_general` (M3a.3 silently landed).

## Acceptance for "done" (single criterion, per-pass)

- **Pass A**: ≥ 15/25 V_RHE converged with non-zero gross R_2e.
- **Pass B**: ≥ 15/25 V_RHE converged.
- **Pass D**: ≥ 15/25 V_RHE converged with non-zero R_2e AND R_4e
  contributions, at the chosen ladder factor.
- **Pass C**: NOT required for done. Exploratory only.

If any of {A, B, D} ≥ 15/25 is unreachable in the time budget, ask
the user before lowering.

The delivered sweep is **structural**, not calibrated to literature
K0. Pass D plots demonstrate machinery; do not describe them as
physical page-15 agreement. Calibration is M4 work, deferred.

## Cross-references

- `docs/ruggiero_realignment_plan.md` — long-form plan being
  superseded.
- `docs/seitz_mangan_data_folder_audit_2026-05-08.md` — multi-doc
  experimental setup audit.
- `docs/Ruggiero2022_JCatal_source_paper.md` — source paper.
- `docs/CHATGPT_HANDOFF_19_RUGGIERO_CONVERGENCE_RECOVERY_PLAN.md` —
  H19 convergence rescue ladder.
- `docs/picard_general_topology_derivation.md` v3 — derivation of
  signed ambipolar form (already implemented in
  `picard_outer_loop_general`).
- `docs/steric_analytic_clo4_reduction_handoff.md` — single-ion
  Bikerman closed-form derivation (generalizes to multi-ion via
  shared-theta ansatz; see Phase 2.1).
- `docs/CHATGPT_HANDOFF_20_fast-realignment/` — full critique audit
  trail (5 rounds, 41 issues, APPROVED).
- `CLAUDE.md` — hard rules; production stack flags; environment.
- `memory/project_ic_stern_bug.md` — anchor fragility memory.
