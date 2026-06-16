
## 2026-06-10 — `global` declaration placement (REPEAT)
Second occurrence of `SyntaxError: name used prior to global
declaration` (first: L_EFF_M in the dual-pathway driver). Rule: when a
module-level name appears ANYWHERE in a function that also declares it
`global` (including in argparse defaults), put `global X` as the FIRST
statement of the function, at write time — don't wait for the compile
error. Always py_compile new scripts before launching runs.

## 2026-06-11 — argparse eats negative-number arguments
`--x0 -0.29,...` dies with "expected one argument" (leading minus
parsed as a flag); two background runs were lost silently as
"completed". Rules: (1) always pass potentially-negative values as
`--flag=value`; (2) after launching a background run, verify it
actually STARTED (check the log for the first expected line) before
trusting the completion notification.

## 2026-06-11 — don't shell-loop background launches with unquoted flag vars
A `for tag in "name:--flag val"` loop with `python ... $flag ...`
under nohup mangled the args ("unrecognized arguments: --w-ring-scale
2.0") and the runs died instantly while ps still showed siblings.
Rule: launch each background run with the dedicated background tool and
EXPLICIT args (negatives as --flag=value); never build flag strings in
a shell loop.

## 2026-06-15 — repo cleanup: import rewrites miss PATH STRINGS
Restructuring `scripts/studies/` into `drivers/plot/extract/` and
rewriting `scripts.studies.X` → `scripts.studies.drivers.X` left ONE
test red: `test_phase6b_v10b_calibration.py` opens drivers by hardcoded
FILE PATH (`open("scripts/studies/<mod>.py")`, AST source-audit), not by
import. Rule: when moving a module, grep for BOTH the dotted import path
AND the slash file path (`scripts/studies/<mod>.py`) across tests+scripts;
AST/source-introspection tests and provenance metadata strings reference
the latter. Also: before declaring a cleanup green, baseline the failing
tests on a `main` worktree — 16 of 17 failures were pre-existing
(autograd-FD, jithin_picard_closure, multistart), only 1 was the regression.

## 2026-06-15 — parallel cleanup: git add -A sweeps in untracked results
After fs-only parallel agents moved files, `git add -A` staged 98
previously-UNTRACKED result files (phase7_fit evals, smoke JSONs) as new
adds — conflating "organize" with "start version-controlling outputs."
Rule: after a mv/rm reorg, `git diff --cached --diff-filter=A` and unstage
anything that was untracked before; a cleanup commit should be renames +
deletes + edits, not new data. Keep only intentional adds (e.g. new
package `__init__.py`).
