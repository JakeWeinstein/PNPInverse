"""Unit tests for Forward.params.SolverParams.

These tests exercise pure-Python logic and do NOT require Firedrake.
"""

from __future__ import annotations

import copy
import dataclasses

import pytest

from Forward.params import SolverParams


# ===================================================================
# Construction
# ===================================================================

class TestSolverParamsConstruction:
    """Test the various ways to create a SolverParams instance."""

    def test_keyword_construction(self):
        sp = SolverParams(
            n_species=2,
            order=1,
            dt=0.1,
            t_end=10.0,
            z_vals=[1, -1],
            D_vals=[1.0, 1.1],
            a_vals=[0.0, 0.0],
            phi_applied=0.05,
            c0_vals=[0.1, 0.1],
            phi0=0.05,
            solver_options={"snes_type": "newtonls"},
        )
        assert len(sp) == 11
        assert isinstance(sp, SolverParams)

    def test_from_list(self):
        raw = [
            2, 1, 0.1, 10.0,
            [1, -1], [1.0, 1.1], [0.0, 0.0],
            0.05, [0.1, 0.1], 0.05,
            {"snes_type": "newtonls"},
        ]
        sp = SolverParams.from_list(raw)
        assert len(sp) == 11
        assert isinstance(sp, SolverParams)

    def test_from_list_wrong_length_raises(self):
        with pytest.raises(ValueError, match="Expected 11-entry"):
            SolverParams.from_list([1, 2, 3])

    def test_from_list_too_long_raises(self):
        with pytest.raises(ValueError, match="Expected 11-entry"):
            SolverParams.from_list(list(range(12)))

    def test_is_frozen_dataclass(self):
        sp = SolverParams(
            n_species=2, order=1, dt=0.1, t_end=10.0,
            z_vals=[0, 0], D_vals=[1.0, 1.0], a_vals=[0.0, 0.0],
            phi_applied=0.0, c0_vals=[1.0, 1.0], phi0=0.0,
            solver_options={},
        )
        assert dataclasses.is_dataclass(sp)
        # Direct attribute assignment should raise FrozenInstanceError
        with pytest.raises(dataclasses.FrozenInstanceError):
            sp.n_species = 5


# ===================================================================
# Property accessors
# ===================================================================

class TestSolverParamsProperties:
    """Test named-attribute getters."""

    def test_n_species_getter(self, sample_solver_params):
        assert sample_solver_params.n_species == 2

    def test_order_getter(self, sample_solver_params):
        assert sample_solver_params.order == 1

    def test_dt_getter(self, sample_solver_params):
        assert sample_solver_params.dt == pytest.approx(0.1)

    def test_t_end_getter(self, sample_solver_params):
        assert sample_solver_params.t_end == pytest.approx(10.0)

    def test_z_vals_getter(self, sample_solver_params):
        assert sample_solver_params.z_vals == [0, 0]

    def test_D_vals_getter(self, sample_solver_params):
        assert sample_solver_params.D_vals == [1.0, 1.1]

    def test_a_vals_getter(self, sample_solver_params):
        assert sample_solver_params.a_vals == [0.0, 0.0]

    def test_phi_applied_getter(self, sample_solver_params):
        assert sample_solver_params.phi_applied == pytest.approx(0.05)

    def test_c0_vals_getter(self, sample_solver_params):
        assert sample_solver_params.c0_vals == [1.0, 0.0]

    def test_phi0_getter(self, sample_solver_params):
        assert sample_solver_params.phi0 == pytest.approx(0.0)

    def test_solver_options_getter(self, sample_solver_params):
        opts = sample_solver_params.solver_options
        assert isinstance(opts, dict)
        assert opts["snes_type"] == "newtonls"


# ===================================================================
# Mutation helpers (with_* methods)
# ===================================================================

class TestSolverParamsMutationHelpers:
    """Test with_*() methods that return new frozen instances."""

    def test_with_phi_applied(self, sample_solver_params):
        new = sample_solver_params.with_phi_applied(0.2)
        assert new.phi_applied == pytest.approx(0.2)
        assert sample_solver_params.phi_applied == pytest.approx(0.05)

    def test_with_dt(self, sample_solver_params):
        new = sample_solver_params.with_dt(0.5)
        assert new.dt == pytest.approx(0.5)
        assert sample_solver_params.dt == pytest.approx(0.1)

    def test_with_D_vals(self, sample_solver_params):
        new = sample_solver_params.with_D_vals([2.0, 3.0])
        assert new.D_vals == [2.0, 3.0]
        assert sample_solver_params.D_vals == [1.0, 1.1]

    def test_with_solver_options(self, sample_solver_params):
        new_opts = {"snes_type": "nrichardson"}
        new = sample_solver_params.with_solver_options(new_opts)
        assert new.solver_options["snes_type"] == "nrichardson"
        assert sample_solver_params.solver_options["snes_type"] == "newtonls"


# ===================================================================
# Index-based access (backward compat)
# ===================================================================

class TestSolverParamsIndexAccess:
    """Verify that index-based access is consistent with named attributes."""

    def test_index_matches_attributes(self, sample_solver_params):
        sp = sample_solver_params
        assert sp[0] == sp.n_species
        assert sp[1] == sp.order
        assert sp[2] == sp.dt
        assert sp[3] == sp.t_end
        assert sp[4] == sp.z_vals
        assert sp[5] == sp.D_vals
        assert sp[6] == sp.a_vals
        assert sp[7] == sp.phi_applied
        assert sp[8] == sp.c0_vals
        assert sp[9] == sp.phi0
        assert sp[10] == sp.solver_options

    def test_index_setitem_updates_attribute(self, sample_solver_params):
        """__setitem__ provides backward compat for deep-copy-then-mutate."""
        sp = sample_solver_params.deep_copy()
        sp[7] = 0.99
        assert sp.phi_applied == pytest.approx(0.99)
        assert sp[7] == pytest.approx(0.99)

    def test_index_setitem_n_species(self, sample_solver_params):
        sp = sample_solver_params.deep_copy()
        sp[0] = 3
        assert sp.n_species == 3
        assert sp[0] == 3


# ===================================================================
# Iteration / unpacking
# ===================================================================

class TestSolverParamsIteration:
    """Test iteration and unpacking backward compat."""

    def test_unpacking(self, sample_solver_params):
        n_s, order, dt, t_end, z_vals, D_vals, a_vals, phi_applied, c0, phi0, opts = sample_solver_params
        assert n_s == 2
        assert order == 1
        assert dt == pytest.approx(0.1)
        assert phi_applied == pytest.approx(0.05)
        assert isinstance(opts, dict)

    def test_list_conversion(self, sample_solver_params):
        as_list = sample_solver_params.to_list()
        assert isinstance(as_list, list)
        assert len(as_list) == 11
        assert as_list[0] == 2
        assert as_list[7] == pytest.approx(0.05)


# ===================================================================
# deep_copy
# ===================================================================

class TestSolverParamsDeepCopy:
    """Test the deep_copy() convenience method."""

    def test_deep_copy_is_independent(self, sample_solver_params):
        original = sample_solver_params
        cloned = original.deep_copy()

        # Values match
        assert cloned.n_species == original.n_species
        assert cloned.D_vals == original.D_vals
        assert cloned.phi_applied == original.phi_applied
        assert cloned.solver_options == original.solver_options

        # Mutating clone via __setitem__ does not affect original
        cloned[7] = 999.0
        assert original.phi_applied == pytest.approx(0.05)

    def test_deep_copy_nested_independence(self, sample_solver_params):
        """Mutating nested structures in the clone should not affect original."""
        original = sample_solver_params
        cloned = original.deep_copy()

        cloned.D_vals[0] = 999.0
        assert original.D_vals[0] == pytest.approx(1.0)

        cloned.solver_options["new_key"] = "new_val"
        assert "new_key" not in original.solver_options

    def test_deep_copy_returns_solver_params_type(self, sample_solver_params):
        cloned = sample_solver_params.deep_copy()
        assert isinstance(cloned, SolverParams)


# ===================================================================
# __repr__
# ===================================================================

class TestSolverParamsRepr:
    """Test the __repr__ method."""

    def test_repr_contains_key_info(self, sample_solver_params):
        r = repr(sample_solver_params)
        assert "SolverParams" in r
        assert "n_species=2" in r
        assert "order=1" in r
        assert "dt=0.1" in r
        assert "phi_applied=0.05" in r

    def test_repr_does_not_contain_solver_options(self, sample_solver_params):
        """Repr omits verbose solver_options dict for readability."""
        r = repr(sample_solver_params)
        assert "snes_type" not in r


# ===================================================================
# Type coercion
# ===================================================================

class TestSolverParamsTypeCoercion:
    """Verify that constructor coerces types properly."""

    def test_int_coercion_for_n_species(self):
        sp = SolverParams(
            n_species=2.9,   # will be cast to int
            order=1,
            dt=0.1,
            t_end=10.0,
            z_vals=[0, 0],
            D_vals=[1.0, 1.0],
            a_vals=[0.0, 0.0],
            phi_applied=0.0,
            c0_vals=[1.0, 1.0],
            phi0=0.0,
            solver_options={},
        )
        assert sp.n_species == 2
        assert isinstance(sp[0], int)

    def test_float_coercion_for_dt(self):
        sp = SolverParams(
            n_species=2,
            order=1,
            dt=1,   # int, should be coerced to float
            t_end=10,
            z_vals=[0, 0],
            D_vals=[1.0, 1.0],
            a_vals=[0.0, 0.0],
            phi_applied=0,
            c0_vals=[1.0, 1.0],
            phi0=0,
            solver_options={},
        )
        assert isinstance(sp.dt, float)
        assert isinstance(sp.t_end, float)
        assert isinstance(sp.phi_applied, float)

    def test_list_coercion_for_z_vals(self):
        sp = SolverParams(
            n_species=2,
            order=1,
            dt=0.1,
            t_end=10.0,
            z_vals=(1, -1),  # tuple, should be coerced to list
            D_vals=(1.0, 1.0),
            a_vals=(0.0, 0.0),
            phi_applied=0.0,
            c0_vals=(1.0, 1.0),
            phi0=0.0,
            solver_options={},
        )
        assert isinstance(sp.z_vals, list)
        assert isinstance(sp.D_vals, list)
        assert isinstance(sp.a_vals, list)
        assert isinstance(sp.c0_vals, list)
