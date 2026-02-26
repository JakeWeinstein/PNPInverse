"""SolverParams — a named-attribute view of the PNP 11-entry parameter list.

Why a list subclass?
--------------------
All forward solvers (dirichlet_solver, robin_solver, steady_state) expect and
mutate the classic 11-entry list.  Subclassing ``list`` means:

* All existing code that unpacks ``n_s, order, dt, ... = solver_params`` or
  reads ``solver_params[7]`` continues to work with zero changes.
* New code can use readable attribute access: ``params.phi_applied``,
  ``params.D_vals``, etc.
* ``deep_copy_solver_params()`` and ``configure_robin_solver_params()``
  return ``SolverParams`` instances automatically.

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
from typing import Any, Dict, List, Optional, Sequence, Union


class SolverParams(list):
    """Named-attribute list for PNP solver parameters.

    Construct via :func:`build_default_solver_params` in :mod:`Inverse.inference_runner`,
    or directly::

        params = SolverParams(
            n_species=2, order=1, dt=0.1, t_end=20.0,
            z_vals=[1, -1], D_vals=[1.0, 1.0], a_vals=[0.0, 0.0],
            phi_applied=0.05, c0_vals=[0.1, 0.1], phi0=0.05,
            solver_options={...},
        )
        params.phi_applied = 0.10   # mutates in-place; params[7] also changes
    """

    _NAMES = (
        "n_species", "order", "dt", "t_end",
        "z_vals", "D_vals", "a_vals",
        "phi_applied", "c0_vals", "phi0", "solver_options",
    )
    _N = len(_NAMES)  # 11

    def __new__(cls, *args, **kwargs) -> "SolverParams":
        return list.__new__(cls)

    def __init__(
        self,
        n_species: int,
        order: int,
        dt: float,
        t_end: float,
        z_vals: Sequence[float],
        D_vals: Sequence[float],
        a_vals: Sequence[float],
        phi_applied: float,
        c0_vals: Sequence[float],
        phi0: float,
        solver_options: Dict[str, Any],
    ) -> None:
        super().__init__(
            [
                int(n_species),
                int(order),
                float(dt),
                float(t_end),
                list(z_vals),
                list(D_vals),
                list(a_vals),
                float(phi_applied),
                list(c0_vals),
                float(phi0),
                solver_options,
            ]
        )

    # ------------------------------------------------------------------
    # Named getters / setters for each slot
    # ------------------------------------------------------------------

    @property
    def n_species(self) -> int:
        return self[0]

    @n_species.setter
    def n_species(self, v: int) -> None:
        self[0] = int(v)

    @property
    def order(self) -> int:
        return self[1]

    @order.setter
    def order(self, v: int) -> None:
        self[1] = int(v)

    @property
    def dt(self) -> float:
        return self[2]

    @dt.setter
    def dt(self, v: float) -> None:
        self[2] = float(v)

    @property
    def t_end(self) -> float:
        return self[3]

    @t_end.setter
    def t_end(self, v: float) -> None:
        self[3] = float(v)

    @property
    def z_vals(self) -> List[float]:
        return self[4]

    @z_vals.setter
    def z_vals(self, v: Sequence[float]) -> None:
        self[4] = list(v)

    @property
    def D_vals(self) -> List[float]:
        return self[5]

    @D_vals.setter
    def D_vals(self, v: Sequence[float]) -> None:
        self[5] = list(v)

    @property
    def a_vals(self) -> List[float]:
        return self[6]

    @a_vals.setter
    def a_vals(self, v: Sequence[float]) -> None:
        self[6] = list(v)

    @property
    def phi_applied(self) -> float:
        return self[7]

    @phi_applied.setter
    def phi_applied(self, v: float) -> None:
        self[7] = float(v)

    @property
    def c0_vals(self) -> List[float]:
        return self[8]

    @c0_vals.setter
    def c0_vals(self, v: Sequence[float]) -> None:
        self[8] = list(v)

    @property
    def phi0(self) -> float:
        return self[9]

    @phi0.setter
    def phi0(self, v: float) -> None:
        self[9] = float(v)

    @property
    def solver_options(self) -> Dict[str, Any]:
        return self[10]

    @solver_options.setter
    def solver_options(self, v: Dict[str, Any]) -> None:
        self[10] = v

    # ------------------------------------------------------------------
    # Convenience helpers
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

    def __repr__(self) -> str:
        return (
            f"SolverParams(n_species={self.n_species}, order={self.order}, "
            f"dt={self.dt}, t_end={self.t_end}, "
            f"z_vals={self.z_vals}, D_vals={self.D_vals}, "
            f"phi_applied={self.phi_applied}, c0_vals={self.c0_vals})"
        )
