
## 2026-06-10 — `global` declaration placement (REPEAT)
Second occurrence of `SyntaxError: name used prior to global
declaration` (first: L_EFF_M in the dual-pathway driver). Rule: when a
module-level name appears ANYWHERE in a function that also declares it
`global` (including in argparse defaults), put `global X` as the FIRST
statement of the function, at write time — don't wait for the compile
error. Always py_compile new scripts before launching runs.
