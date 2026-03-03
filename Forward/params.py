"""SolverParams — frozen dataclass for PNP solver parameters.

Entry layout
------------
Index  Name             Type   Description
-----  ---------------  -----  ------------------------------------------
  0    n_species        int    Number of ionic species
  1    order            int    Finite-element polynomial order
  2    dt               float  Time step (physical or model, see nondim)
  3    t_end            float  Final simulation time
  4    z_vals           list   Per-species charge numbers
  5    D_vals           list   Per-species diffusivities
  6    a_vals           list   Per-species activities (unused, kept for API compat)
  7    phi_applied      float  Applied boundary voltage
  8    c0_vals          list   Initial bulk concentrations
  9    phi0             float  Reference potential (echoed, kept in sync with phi_applied
                               by configure_robin_solver_params)
 10    solver_options   dict   PETSc/SNES options + ``robin_bc`` + ``nondim`` sub-dicts
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union


@dataclass(frozen=True)
class SolverParams:
    """Frozen dataclass for PNP solver parameters.

    Construct via :func:`build_default_solver_params` in :mod:`Inverse.inference_runner`,
    or directly::

        params = SolverParams(
            n_species=2, order=1, dt=0.1, t_end=20.0,
            z_vals=[1, -1], D_vals=[1.0, 1.0], a_vals=[0.0, 0.0],
            phi_applied=0.05, c0_vals=[0.1, 0.1], phi0=0.05,
            solver_options={...},
        )
        # Immutable: use replace helpers for mutation
        new_params = params.with_phi_applied(0.10)
    """

    n_species: int
    order: int
    dt: float
    t_end: float
    z_vals: List[float] = field(default_factory=list)
    D_vals: List[float] = field(default_factory=list)
    a_vals: List[float] = field(default_factory=list)
    phi_applied: float = 0.0
    c0_vals: List[float] = field(default_factory=list)
    phi0: float = 0.0
    solver_options: Dict[str, Any] = field(default_factory=dict)

    _NAMES = (
        "n_species", "order", "dt", "t_end",
        "z_vals", "D_vals", "a_vals",
        "phi_applied", "c0_vals", "phi0", "solver_options",
    )
    _N = len(_NAMES)  # 11

    def __post_init__(self) -> None:
        # Normalize types on construction (using object.__setattr__ since frozen)
        object.__setattr__(self, "n_species", int(self.n_species))
        object.__setattr__(self, "order", int(self.order))
        object.__setattr__(self, "dt", float(self.dt))
        object.__setattr__(self, "t_end", float(self.t_end))
        object.__setattr__(self, "z_vals", list(self.z_vals))
        object.__setattr__(self, "D_vals", list(self.D_vals))
        object.__setattr__(self, "a_vals", list(self.a_vals))
        object.__setattr__(self, "phi_applied", float(self.phi_applied))
        object.__setattr__(self, "c0_vals", list(self.c0_vals))
        object.__setattr__(self, "phi0", float(self.phi0))

    # ------------------------------------------------------------------
    # Backward compat: iteration, indexing, length
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator:
        """Yield values in the same order as the original 11-entry list."""
        return iter(self.to_list())

    def __getitem__(self, index: int) -> Any:
        """Support ``params[i]`` index access for backward compat."""
        return self.to_list()[index]

    def __setitem__(self, index: int, value: Any) -> None:
        """Support ``params[i] = value`` for backward compat with code that
        deep-copies then mutates. Uses object.__setattr__ to bypass frozen."""
        name = self._NAMES[index]
        if name in ("z_vals", "D_vals", "a_vals", "c0_vals"):
            value = list(value)
        elif name in ("n_species", "order"):
            value = int(value)
        elif name in ("dt", "t_end", "phi_applied", "phi0"):
            value = float(value)
        object.__setattr__(self, name, value)

    def __len__(self) -> int:
        return self._N

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def to_list(self) -> List[Any]:
        """Return a plain 11-entry list (same order as the original layout)."""
        return [
            self.n_species,
            self.order,
            self.dt,
            self.t_end,
            self.z_vals,
            self.D_vals,
            self.a_vals,
            self.phi_applied,
            self.c0_vals,
            self.phi0,
            self.solver_options,
        ]

    @classmethod
    def from_list(cls, params: Sequence[Any]) -> "SolverParams":
        """Construct from a legacy 11-entry list or any sequence."""
        if len(params) != cls._N:
            raise ValueError(
                f"Expected {cls._N}-entry solver params list; got length {len(params)}."
            )
        return cls(
            n_species=params[0],
            order=params[1],
            dt=params[2],
            t_end=params[3],
            z_vals=params[4],
            D_vals=params[5],
            a_vals=params[6],
            phi_applied=params[7],
            c0_vals=params[8],
            phi0=params[9],
            solver_options=params[10],
        )

    # ------------------------------------------------------------------
    # Deep copy
    # ------------------------------------------------------------------

    def deep_copy(self) -> "SolverParams":
        """Return an independent deep copy."""
        return SolverParams(
            n_species=self.n_species,
            order=self.order,
            dt=self.dt,
            t_end=self.t_end,
            z_vals=copy.deepcopy(self.z_vals),
            D_vals=copy.deepcopy(self.D_vals),
            a_vals=copy.deepcopy(self.a_vals),
            phi_applied=self.phi_applied,
            c0_vals=copy.deepcopy(self.c0_vals),
            phi0=self.phi0,
            solver_options=copy.deepcopy(self.solver_options),
        )

    # ------------------------------------------------------------------
    # Mutation helpers (return new frozen instances)
    # ------------------------------------------------------------------

    def with_phi_applied(self, phi: float) -> "SolverParams":
        """Return a new SolverParams with updated phi_applied."""
        return dataclasses.replace(self, phi_applied=float(phi))

    def with_phi0(self, phi0: float) -> "SolverParams":
        """Return a new SolverParams with updated phi0."""
        return dataclasses.replace(self, phi0=float(phi0))

    def with_dt(self, dt: float) -> "SolverParams":
        """Return a new SolverParams with updated dt."""
        return dataclasses.replace(self, dt=float(dt))

    def with_solver_options(self, opts: Dict[str, Any]) -> "SolverParams":
        """Return a new SolverParams with updated solver_options."""
        return dataclasses.replace(self, solver_options=opts)

    def with_D_vals(self, D_vals: Sequence[float]) -> "SolverParams":
        """Return a new SolverParams with updated D_vals."""
        return dataclasses.replace(self, D_vals=list(D_vals))

    def with_a_vals(self, a_vals: Sequence[float]) -> "SolverParams":
        """Return a new SolverParams with updated a_vals."""
        return dataclasses.replace(self, a_vals=list(a_vals))

    def with_c0_vals(self, c0_vals: Sequence[float]) -> "SolverParams":
        """Return a new SolverParams with updated c0_vals."""
        return dataclasses.replace(self, c0_vals=list(c0_vals))

    def with_z_vals(self, z_vals: Sequence[float]) -> "SolverParams":
        """Return a new SolverParams with updated z_vals."""
        return dataclasses.replace(self, z_vals=list(z_vals))

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SolverParams(n_species={self.n_species}, order={self.order}, "
            f"dt={self.dt}, t_end={self.t_end}, "
            f"z_vals={self.z_vals}, D_vals={self.D_vals}, "
            f"phi_applied={self.phi_applied}, c0_vals={self.c0_vals})"
        )
