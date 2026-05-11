"""Top-level lightweight calibration package.

This package holds literature-anchored numeric constants and their
provenance metadata for the PNP-BV forward stack.  It is deliberately
Firedrake-free: importing any module under :mod:`calibration` must not
pull ``firedrake`` (or any Forward.bv_solver submodule) into
``sys.modules``.  That invariant lets unit tests verify constant
consistency and metadata schemas without a Firedrake interpreter.

Modules:

- :mod:`calibration.v10b` — Phase 6β v10b literature calibration of
  ``Gamma_max``, ``k_des``, and ``C_S`` (2026-05-10).
"""
