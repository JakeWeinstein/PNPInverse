
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
