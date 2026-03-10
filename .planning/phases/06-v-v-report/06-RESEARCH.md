# Phase 6: V&V Report - Research

**Researched:** 2026-03-09
**Domain:** LaTeX report generation, matplotlib publication figures, programmatic table generation
**Confidence:** HIGH

## Summary

Phase 6 is a pure report-generation phase: no new tests, no new verification work. The task is to produce a publication-grade LaTeX V&V report that programmatically consumes all JSON/CSV data from `StudyResults/` and generates vector figures (PDF) and LaTeX table snippets. The report covers 4 verification layers (forward solver, surrogate, inverse, pipeline) plus a synthesis summary.

The data sources are well-defined and stable: 6 JSON files across 3 `StudyResults/` subdirectories plus 1 pipeline reproducibility JSON. All schemas have been inspected and documented below. The existing `scripts/generate_presentation_plots.py` provides reusable data-loading patterns but uses different styling goals (presentation vs publication).

**Primary recommendation:** Build a single `generate_figures.py` script that reads all JSON/CSV sources, outputs PDF figures to `writeups/vv_report/figures/` and LaTeX table `.tex` snippets to `writeups/vv_report/tables/`, then write the LaTeX report using `\input{}` for all tables and `\includegraphics{}` for all figures.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- LaTeX standalone V&V report (not appendix, not Jupyter)
- Self-contained document at `writeups/vv_report/vv_report.tex`
- Report structure: 7 sections (Introduction, Pipeline Architecture, Forward Solver, Surrogate, Inverse, Pipeline Reproducibility, Summary)
- Fresh `writeups/vv_report/generate_figures.py` script (not extending existing presentation plots)
- Reads all data from `StudyResults/` JSON/CSV files
- Outputs PDF vector figures to `writeups/vv_report/figures/`
- Essential figures: MMS convergence log-log, parameter recovery vs noise, worst-case I-V overlay
- Surrogate fidelity as table only (no figure)
- Auto-generated LaTeX table snippets via Python to `writeups/vv_report/tables/*.tex`
- Tables: MMS convergence rates, surrogate fidelity, parameter recovery, gradient consistency, summary
- Brief methodology per section (1-2 paragraphs + references)
- Governing PNP-BV equations in introduction/architecture
- Summary section with synthesis table + concluding prose
- TikZ pipeline architecture diagram

### Claude's Discretion
- Publication figure styling (serif fonts, color palette, axis labels)
- TikZ pipeline architecture diagram design
- Exact LaTeX document class and package choices
- Table formatting details (significant figures, column alignment)
- Bibliography entries and citation style
- Page length and layout decisions

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| RPT-01 | Written V&V report with convergence tables, convergence plots, and GCI uncertainty bounds suitable for journal supplementary material | All data sources documented, figure/table specifications defined, LaTeX structure planned |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| LaTeX (article class) | texlive 2024+ | Document typesetting | Standard for academic publication |
| matplotlib | (project env) | PDF vector figure generation | Already used throughout project |
| numpy | (project env) | Data manipulation for plots | Already used throughout project |

### Supporting
| Library/Package | Purpose | When to Use |
|-----------------|---------|-------------|
| booktabs (LaTeX) | Publication-quality tables | All tables (`\toprule`, `\midrule`, `\bottomrule`) |
| siunitx (LaTeX) | Number formatting in tables | Aligning decimal points, scientific notation |
| graphicx (LaTeX) | Figure inclusion | `\includegraphics` for PDF figures |
| tikz (LaTeX) | Pipeline architecture diagram | Section 2 flowchart |
| amsmath/amssymb (LaTeX) | Math typesetting | PNP-BV governing equations |
| hyperref (LaTeX) | Cross-references and PDF metadata | Internal references |
| geometry (LaTeX) | Page margins | Standard journal-compatible margins |
| natbib or biblatex (LaTeX) | Bibliography | Citation management |
| caption/subcaption (LaTeX) | Figure captions | Multi-panel figure captions |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| article class | revtex4-2 | Only needed if targeting APS/AIP journals specifically |
| TikZ pipeline | matplotlib pipeline figure | TikZ integrates better with LaTeX text, user chose TikZ |
| siunitx | manual formatting | siunitx handles alignment automatically, less error-prone |

## Architecture Patterns

### Recommended Project Structure
```
writeups/vv_report/
├── vv_report.tex          # Main LaTeX document
├── generate_figures.py    # Single script: figures + tables
├── figures/               # PDF vector figures (generated)
│   ├── mms_convergence.pdf
│   ├── parameter_recovery.pdf
│   └── worst_case_iv.pdf
├── tables/                # LaTeX table snippets (generated)
│   ├── mms_rates.tex
│   ├── surrogate_fidelity.tex
│   ├── parameter_recovery.tex
│   ├── gradient_consistency.tex
│   └── summary.tex
└── references.bib         # Bibliography file
```

### Pattern 1: Programmatic Table Generation
**What:** Python script writes LaTeX table code directly as `.tex` files
**When to use:** Every table in the report
**Example:**
```python
def write_mms_table(data, outpath):
    """Write MMS convergence rates table as LaTeX snippet."""
    lines = []
    lines.append(r"\begin{tabular}{l S[table-format=1.3] S[table-format=1.7] S[table-format=1.3] S[table-format=1.7] S[table-format=1.3]}")
    lines.append(r"\toprule")
    lines.append(r"Species & {$L^2$ Rate} & {$L^2$ $R^2$} & {$H^1$ Rate} & {$H^1$ $R^2$} & {GCI (finest)} \\")
    lines.append(r"\midrule")
    for key, info in data["fields"].items():
        gci_finest = info["gci"][-1]["gci"]
        lines.append(
            f"{info['label']} & {info['L2_rate']:.3f} & {info['L2_r_squared']:.7f} "
            f"& {info['H1_rate']:.3f} & {info['H1_r_squared']:.7f} & {gci_finest:.3f} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    Path(outpath).write_text("\n".join(lines))
```

### Pattern 2: PDF Vector Figure Output
**What:** matplotlib saves figures as PDF for LaTeX inclusion
**When to use:** All figures
**Example:**
```python
fig.savefig("figures/mms_convergence.pdf", bbox_inches="tight", backend="pdf")
```

### Pattern 3: Publication Figure Styling
**What:** Serif fonts matching LaTeX document, appropriate sizing
**When to use:** All figures
**Example:**
```python
import matplotlib
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
    "text.usetex": False,  # safer: avoid requiring LaTeX in Python env
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.figsize": (6.5, 4.5),  # fits single-column journal width
})
```

### Anti-Patterns to Avoid
- **Hardcoded numbers in LaTeX:** Never type data values directly in `.tex` — always `\input{tables/...}` from generated files
- **Raster figures in LaTeX:** Always use PDF vector output, never PNG
- **`text.usetex: True` in matplotlib:** Requires full LaTeX installation in Python environment, fragile — use `mathtext.fontset: "cm"` instead for Computer Modern math rendering without LaTeX dependency

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Table formatting | Manual `\\` alignment | `booktabs` + `siunitx` | Consistent spacing, decimal alignment |
| Number formatting | `f"{x:.3f}"` everywhere | `siunitx` `\num{}` or Python-side consistent formatting | Handles scientific notation, alignment |
| Cross-references | Hardcoded "Table 1" | `\label{}`/`\ref{}` | Auto-numbering survives restructuring |
| Pipeline diagram | matplotlib boxes | TikZ | User chose TikZ; better text integration |

## Common Pitfalls

### Pitfall 1: LaTeX Not Installed
**What goes wrong:** `pdflatex` or `latexmk` not available on the machine
**Why it happens:** Scientific Python environments often lack LaTeX
**How to avoid:** The Python script should be runnable without LaTeX. LaTeX compilation is a separate step the user runs. Document both steps clearly.
**Warning signs:** `which pdflatex` returns nothing (confirmed: no LaTeX found in current env)

### Pitfall 2: Figure Sizing Mismatch
**What goes wrong:** Figures look too small or text is unreadable when included in LaTeX
**Why it happens:** matplotlib default figsize doesn't match LaTeX column width
**How to avoid:** Use `figsize=(6.5, 4.5)` for single-column (fits standard 6.5in text width) and scale with `\includegraphics[width=\textwidth]`

### Pitfall 3: Surrogate PC NRMSE Outliers
**What goes wrong:** Peroxide current (PC) NRMSE values are extremely large (max 151-510) due to near-zero-range samples
**Why it happens:** NRMSE normalizes by range; when PC range is tiny, small absolute errors become huge relative errors
**How to avoid:** Report median and 95th percentile (not mean/max) as primary metrics for PC. Note the near-zero-range issue in the table footnote.

### Pitfall 4: Parameter Recovery 5% Noise Row
**What goes wrong:** Including 5% noise as a pass/fail gate when it exceeds surrogate approximation limits
**Why it happens:** The 5% noise level is informational only
**How to avoid:** Mark the 5% row distinctly (e.g., italic, footnote) and note it is informational, not a gate

### Pitfall 5: GCI Values Are Ratios, Not Percentages
**What goes wrong:** Misinterpreting GCI values (~1.25) as percentages
**Why it happens:** The GCI in the data is the ratio `error_coarse / error_fine`, not a percentage uncertainty bound
**How to avoid:** Present as safety factor or compute percentage GCI = Fs * |e_fine| / (r^p - 1) * 100 if needed

## Code Examples

### Data Loading Pattern (from existing codebase)
```python
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
STUDY = ROOT / "StudyResults"

def load_mms_data():
    with open(STUDY / "mms_convergence" / "convergence_data.json") as f:
        return json.load(f)

def load_surrogate_fidelity():
    with open(STUDY / "surrogate_fidelity" / "fidelity_summary.json") as f:
        return json.load(f)

def load_inverse_verification():
    inv = STUDY / "inverse_verification"
    return {
        "parameter_recovery": json.load(open(inv / "parameter_recovery_summary.json")),
        "gradient_fd": json.load(open(inv / "gradient_fd_convergence.json")),
        "gradient_pde": json.load(open(inv / "gradient_pde_consistency.json")),
        "multistart": json.load(open(inv / "multistart_basin.json")),
    }

def load_pipeline_reproducibility():
    with open(STUDY / "pipeline_reproducibility" / "regression_baselines.json") as f:
        return json.load(f)
```

### MMS Log-Log Convergence Plot
```python
def plot_mms_convergence(data, outdir):
    h = np.array(data["metadata"]["h_values"])
    fig, (ax_l2, ax_h1) = plt.subplots(1, 2, figsize=(12, 5))

    colors = {"c0": "#1f77b4", "c1": "#d62728", "c2": "#2ca02c", "c3": "#ff7f0e", "phi": "#9467bd"}

    for key, info in data["fields"].items():
        ax_l2.loglog(h, info["L2_errors"], "o-", color=colors[key],
                     label=f"{info['label']} (rate={info['L2_rate']:.2f})", lw=1.5, ms=5)
        ax_h1.loglog(h, info["H1_errors"], "s--", color=colors[key],
                     label=f"{info['label']} (rate={info['H1_rate']:.2f})", lw=1.5, ms=5)

    # Reference slopes
    h_ref = np.array([h[0], h[-1]])
    for ax, p, label in [(ax_l2, 2, r"$O(h^2)$"), (ax_h1, 1, r"$O(h)$")]:
        scale = ax.get_lines()[0].get_ydata()[0] / h[0]**p * 1.5
        ax.loglog(h_ref, scale * h_ref**p, "k:", alpha=0.5, lw=1, label=label)
        ax.set_xlabel("Mesh spacing $h$")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    ax_l2.set_ylabel(r"$L^2$ error")
    ax_h1.set_ylabel(r"$H^1$ error")
    ax_l2.set_title(r"$L^2$ Convergence")
    ax_h1.set_title(r"$H^1$ Convergence")

    fig.savefig(outdir / "mms_convergence.pdf", bbox_inches="tight")
    plt.close(fig)
```

### Worst-Case I-V Overlay (Surrogate vs PDE)
```python
def plot_worst_case_iv(outdir):
    """Read P2 fit CSV for surrogate-vs-PDE overlay at worst parameter samples."""
    csv_path = STUDY / "master_inference_v13" / "P2_pde_full_cathodic" / "multi_obs_fit.csv"
    # Load and plot target vs simulated I-V curves
    # This shows how well the surrogate prediction matches PDE truth
```

## Data Source Inventory

All data sources consumed by the report, with schema summaries:

| Source File | Section | Key Fields |
|-------------|---------|------------|
| `mms_convergence/convergence_data.json` | Forward Solver | 5 fields x {L2/H1 errors, rates, R^2, GCI} at 4 mesh sizes |
| `surrogate_fidelity/fidelity_summary.json` | Surrogate | 4 models x {CD, PC} x {max, mean, median, 95th NRMSE}, n=479 test points |
| `inverse_verification/parameter_recovery_summary.json` | Inverse | 4 noise levels x {median/mean/std max relative error, gate, pass/fail} |
| `inverse_verification/gradient_fd_convergence.json` | Inverse | FD gradients at 3 step sizes, convergence rates, analytic comparison |
| `inverse_verification/gradient_pde_consistency.json` | Inverse | PDE gradients at 3 eval points x 3 step sizes, convergence rates |
| `inverse_verification/multistart_basin.json` | Inverse | 20 candidates, basin uniqueness CV, functional fit NRMSE |
| `pipeline_reproducibility/regression_baselines.json` | Pipeline | Surrogate-only + full pipeline reference values |
| `master_inference_v13/P2_pde_full_cathodic/multi_obs_fit.csv` | Inverse (figure) | phi, target/simulated primary+secondary I-V curves |

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual tables in LaTeX | Programmatic table generation from JSON | This phase | Eliminates transcription errors |
| PNG figures | PDF vector figures | This phase | Publication-quality, resolution-independent |
| Scattered results | Unified V&V report | This phase | Single artifact for peer review |

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (existing) |
| Config file | existing pytest configuration |
| Quick run command | `python writeups/vv_report/generate_figures.py` (generates all artifacts) |
| Full suite command | `python writeups/vv_report/generate_figures.py && pdflatex vv_report.tex` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| RPT-01 | Figures generated from JSON data | smoke | `python writeups/vv_report/generate_figures.py` | Wave 0 |
| RPT-01 | Tables generated from JSON data | smoke | `python writeups/vv_report/generate_figures.py` (same script) | Wave 0 |
| RPT-01 | LaTeX compiles without errors | manual-only | `cd writeups/vv_report && pdflatex vv_report.tex` (requires LaTeX) | Wave 0 |
| RPT-01 | All figures/tables referenced in report | manual-only | Visual inspection of PDF | N/A |

### Sampling Rate
- **Per task commit:** `python writeups/vv_report/generate_figures.py` (verify no crashes)
- **Per wave merge:** Same + visual inspection of generated PDFs
- **Phase gate:** All figures/tables generated; LaTeX compiles (if LaTeX available)

### Wave 0 Gaps
- [ ] `writeups/vv_report/generate_figures.py` — the figure/table generation script
- [ ] `writeups/vv_report/figures/` directory — output location
- [ ] `writeups/vv_report/tables/` directory — output location
- [ ] `writeups/vv_report/vv_report.tex` — the main LaTeX document

## Open Questions

1. **LaTeX availability**
   - What we know: `pdflatex` and `latexmk` are not found in the current PATH
   - What's unclear: Whether the user has a LaTeX distribution installed elsewhere or uses Overleaf
   - Recommendation: Write the `.tex` file and `generate_figures.py` as standalone artifacts. The user compiles LaTeX separately. Do not make the plan depend on LaTeX compilation succeeding in CI.

2. **Python environment for figure generation**
   - What we know: The project uses matplotlib/numpy (evidenced by existing scripts) but the exact virtualenv path wasn't found
   - What's unclear: Which Python interpreter activates the project environment
   - Recommendation: The `generate_figures.py` script should only depend on `matplotlib`, `numpy`, and `json` (stdlib). The user runs it in their existing environment.

## Sources

### Primary (HIGH confidence)
- Direct inspection of all 7 JSON data files in `StudyResults/` — schema fully documented
- Direct inspection of `scripts/generate_presentation_plots.py` — reusable patterns identified
- CONTEXT.md — user decisions locked

### Secondary (MEDIUM confidence)
- LaTeX package recommendations (booktabs, siunitx, tikz) — standard academic practice, widely documented

### Tertiary (LOW confidence)
- None — this phase is straightforward report generation with no uncertain dependencies

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - LaTeX + matplotlib are well-established, no version-sensitive APIs needed
- Architecture: HIGH - File structure and data flow are fully defined by user decisions
- Pitfalls: HIGH - Data schemas inspected, edge cases (PC outliers, 5% noise) documented from actual data
- Data sources: HIGH - All 7 JSON files read and schemas verified

**Research date:** 2026-03-09
**Valid until:** Indefinite (stable tooling, locked data sources)
