# Bug Report: Forward/Inverse Error Handling

**Focus:** Code quality and logic bugs in Forward/ and Inverse/ packages
**Agent:** Forward/Inverse Error Handling

---

## BUG 1 -- Bare `except` swallowing import errors
**File:** `Forward/plotter.py:22`
**Severity:** MEDIUM
**Description:** `except Exception` catches all exceptions when importing `imageio`, including `ImportError`, `SyntaxError`, `PermissionError`, etc. While the intent is to handle `ImportError`, the broad catch hides genuine errors (e.g., a corrupt or incompatible `imageio` install that raises `AttributeError` or `RuntimeError` during import).
**Suggested fix:** Change to `except ImportError:`.

## BUG 2 -- Bare `except Exception` swallowing adjoint import errors
**File:** `Forward/steady_state/common.py:28`
**Severity:** MEDIUM
**Description:** `except Exception` when importing `firedrake.adjoint` will catch non-import errors (e.g., `RuntimeError` from MPI misconfiguration). If `adj` is silently set to `None` due to a real configuration error, `_maybe_stop_annotating()` will be a no-op, causing silent tape leaks in adjoint-driven inverse workflows.
**Suggested fix:** Change to `except ImportError:`.

## BUG 3 -- `conv_cfg` accessed with `[]` but can be empty dict
**File:** `Forward/bv_solver/forms.py:190, 198, 231`
**Severity:** HIGH
**Description:** `_get_bv_convergence_cfg` returns `{}` when `params` is not a dict or when the `bv_convergence` key is absent. The code then accesses `conv_cfg["use_eta_in_bv"]`, `conv_cfg["clip_exponent"]`, `conv_cfg["regularize_concentration"]`, `conv_cfg["exponent_clip"]`, and `conv_cfg["conc_floor"]` with bracket notation. When `conv_cfg` is `{}`, these will raise `KeyError`.
**Suggested fix:** Either have `_get_bv_convergence_cfg` always return all keys with defaults (it currently does when `raw` is a dict, but returns `{}` when `params` is not a dict), or use `.get()` with defaults at the call sites. The most targeted fix: in `_get_bv_convergence_cfg`, replace `if not isinstance(params, dict): return {}` with returning the full defaults dict.

## BUG 4 -- `write_phi_applied_flux_csv` crashes on empty dirname
**File:** `Forward/steady_state/common.py:222`
**Severity:** LOW
**Description:** `os.makedirs(os.path.dirname(csv_path), exist_ok=True)` will be called with an empty string `""` when `csv_path` is a bare filename like `"results.csv"` (no directory component). `os.makedirs("")` raises `FileNotFoundError` on some platforms.
**Suggested fix:** Guard with `dirname = os.path.dirname(csv_path); if dirname: os.makedirs(dirname, exist_ok=True)`.

## BUG 5 -- `configure_bv_solver_params` loses `bv_bc` mutations when key is missing
**File:** `Forward/steady_state/bv.py:50-51, 65-66`
**Severity:** HIGH
**Description:** When `k0_values` is provided in the `SolverParams` branch, `bv_cfg = opts.get("bv_bc", {})` creates a new empty dict if the key is missing, but this new dict is never stored back into `opts`. All mutations to `bv_cfg` (setting `k0`, `alpha`, etc.) are lost. The same pattern repeats for the `alpha_values` branch (line 65). In contrast, when `bv_bc` already exists, mutations propagate because `get` returns the existing reference.
**Suggested fix:** Replace `bv_cfg = opts.get("bv_bc", {})` with `bv_cfg = opts.setdefault("bv_bc", {})`, which inserts the empty dict into `opts` if missing.

## BUG 6 -- `configure_bv_solver_params` re-fetches `bv_cfg` independently for `alpha_values`
**File:** `Forward/steady_state/bv.py:64-66 (SolverParams branch), 122-128 (legacy list branch)`
**Severity:** MEDIUM
**Description:** Both the `k0_values` and `alpha_values` branches independently fetch `bv_cfg = opts.get("bv_bc", {})`. If the `bv_bc` key does not exist, each branch creates a separate ephemeral dict, so even if Bug 5 were fixed for `k0_values`, the `alpha_values` branch would create a second separate dict. Both should share the same `bv_cfg` reference.
**Suggested fix:** Hoist the `bv_cfg = opts.setdefault("bv_bc", {})` call to before the `if k0_values` / `if alpha_values` blocks so both branches operate on the same dict.

## BUG 7 -- `SolverParams.__setitem__` bypasses frozen dataclass without updating `solver_options`
**File:** `Forward/params.py:90-100`
**Severity:** MEDIUM
**Description:** `__setitem__` handles type coercion for scalars and lists but does not normalize `solver_options` (index 10). If `params[10] = some_new_dict` is called, the dict is set without any deepcopy, allowing aliasing bugs where external code mutates the solver options through the shared reference while the `SolverParams` instance is supposed to be frozen.
**Suggested fix:** Add deepcopy for dict values in `__setitem__`.

## BUG 8 -- `_AttemptMonitor.eval_cb_post` silently swallows all exceptions
**File:** `Inverse/inference_runner/objective.py:123-126`
**Severity:** MEDIUM
**Description:** The bare `except Exception: return` silently swallows any error in `estimate_from_controls`, including bugs like `AttributeError`, `TypeError`, or `IndexError`. This means the monitor will silently fail to track optimization progress and `best_estimate` / `last_estimate` will remain `None` without any warning, making debugging difficult.
**Suggested fix:** At minimum, log a warning. Or narrow the catch to the specific expected exceptions.

## BUG 9 -- `read_phi_applied_flux_csv` accesses `row["flux_clean"]` without `.get()` safety
**File:** `Forward/steady_state/common.py:285`
**Severity:** MEDIUM
**Description:** When the preferred `flux_column` is empty/NaN, the fallback is `row["flux_clean"]` using bracket access. If the CSV was written by an external tool or has different headers, this will raise `KeyError` without a useful message.
**Suggested fix:** Use `row.get("flux_clean", "nan")` with a descriptive fallback.

## BUG 10 -- Integer truncation in `num_steps` calculation
**File:** `Forward/dirichlet_solver.py:232`, `Forward/robin_solver.py:279`
**Severity:** LOW
**Description:** `num_steps = int(t_end / dt)` uses `int()` which truncates toward zero rather than rounding. For floating-point values like `t_end=1.0, dt=0.01`, the division gives `99.99999999999999` which truncates to `99` instead of `100`. The BV solver at `Forward/bv_solver/solvers.py:54` correctly uses `int(round(...))`.
**Suggested fix:** Use `int(round(t_end / dt))` for consistency with the BV solver.

## BUG 11 -- Matplotlib figures not closed on early return in `plotter.py`
**File:** `Forward/plotter.py:130-134`
**Severity:** LOW
**Description:** If an exception occurs between `plt.subplots()` and the `plt.close(fig)` call, the figure leaks. Minor resource leak in a visualization utility.
**Suggested fix:** Wrap the body in try/except with `plt.close(fig)` in the except branch.

---

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 0     |
| HIGH     | 2     |
| MEDIUM   | 5     |
| LOW      | 3     |

**Top priority fixes:**
- **Bug 3** (`conv_cfg` KeyError in `bv_solver/forms.py`): Will crash any BV solver invocation where `solver_params[10]` is not a dict.
- **Bug 5** (`configure_bv_solver_params` losing mutations in `steady_state/bv.py`): When `bv_bc` is not already present in solver options, injected `k0`/`alpha` values are silently lost.
