"""Adapter utilities for plugging forward solvers into unified inverse workflows.

A compatible forward solver module should expose the following callables:
- ``build_context(solver_params)``
- ``build_forms(ctx, solver_params)``
- ``set_initial_conditions(ctx, solver_params, blob=True)``
- ``forsolve(ctx, solver_params, print_interval=...)``

If the solve function has a different name (for example ``forsolve_robin``),
pass that name when constructing :class:`ForwardSolverAdapter`.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import importlib
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# Shared aliases used across the unified inverse modules.
Context = Dict[str, Any]
SolverParams = List[Any]


def as_species_list(values: Any, n_species: int, name: str) -> List[float]:
    """Return a float list of length ``n_species``.

    Parameters
    ----------
    values:
        A scalar or sequence. Scalars are broadcast to all species.
    n_species:
        Number of species expected by the solver.
    name:
        Parameter label used in validation messages.
    """
    if np.isscalar(values):
        return [float(values) for _ in range(int(n_species))]

    try:
        out = [float(v) for v in values]
    except TypeError as exc:
        raise ValueError(
            f"{name} must be a scalar or a sequence of length {n_species}."
        ) from exc

    if len(out) != int(n_species):
        raise ValueError(
            f"{name} must have length {n_species}; got length {len(out)}."
        )
    return out


def deep_copy_solver_params(solver_params: Sequence[Any]) -> SolverParams:
    """Deep-copy the mixed list/dict solver parameter structure.

    The inverse pipeline mutates values such as diffusion coefficients and
    boundary parameters. Deep-copying ensures a request can reuse a base
    parameter object without side effects.
    """
    return copy.deepcopy(list(solver_params))


def extract_solution_vectors(U_final: Any, n_species: int) -> Tuple[List[np.ndarray], np.ndarray]:
    """Extract final-state concentration and potential vectors from ``U_final``.

    Returns
    -------
    tuple
        ``(c_vectors, phi_vector)`` where ``c_vectors`` has one array per species.
    """
    c_vectors = [np.array(U_final.sub(i).dat.data_ro, dtype=float) for i in range(int(n_species))]
    phi_vec = np.array(U_final.sub(int(n_species)).dat.data_ro, dtype=float)
    return c_vectors, phi_vec


@dataclass(frozen=True)
class ForwardSolverAdapter:
    """Runtime adapter for a forward solver module.

    Attributes
    ----------
    module_path:
        Import path for the forward solver module.
    build_context:
        Callable that allocates mesh/function spaces/state.
    build_forms:
        Callable that assembles weak forms and boundary conditions.
    set_initial_conditions:
        Callable that populates initial state.
    solve:
        Callable that advances the system to final time.
    solve_function_name:
        Original name used to resolve the solve callable.
    """

    module_path: str
    build_context: Callable[[SolverParams], Context]
    build_forms: Callable[[Context, SolverParams], Context]
    set_initial_conditions: Callable[[Context, SolverParams], None]
    solve: Callable[..., Any]
    solve_function_name: str

    @classmethod
    def from_module_path(
        cls,
        module_path: str,
        solve_function_name: Optional[str] = None,
        solve_candidates: Sequence[str] = ("forsolve", "forsolve_robin"),
    ) -> "ForwardSolverAdapter":
        """Construct an adapter by importing a compatible forward solver module.

        Parameters
        ----------
        module_path:
            Python import path, for example ``"Forward.robin_solver"``.
        solve_function_name:
            Optional explicit solve function name. When ``None``, candidate names
            in ``solve_candidates`` are searched in order.
        solve_candidates:
            Candidate solve function names used when
            ``solve_function_name is None``.
        """
        module = importlib.import_module(module_path)

        build_context = _require_callable(module, "build_context", module_path)
        build_forms = _require_callable(module, "build_forms", module_path)
        set_initial_conditions = _require_callable(
            module, "set_initial_conditions", module_path
        )

        solve_name = solve_function_name
        if solve_name is None:
            for candidate in solve_candidates:
                if hasattr(module, candidate) and callable(getattr(module, candidate)):
                    solve_name = candidate
                    break
        if solve_name is None:
            names = ", ".join(solve_candidates)
            raise ValueError(
                f"Could not find a solve function in '{module_path}'. "
                f"Tried candidates: {names}."
            )

        solve = _require_callable(module, solve_name, module_path)

        return cls(
            module_path=module_path,
            build_context=build_context,
            build_forms=build_forms,
            set_initial_conditions=set_initial_conditions,
            solve=solve,
            solve_function_name=solve_name,
        )

    def run_forward(
        self,
        solver_params: SolverParams,
        *,
        blob_initial_condition: bool = True,
        print_interval: int = 100,
    ) -> Tuple[Context, Any]:
        """Execute one forward solve and return ``(ctx, U_final)``.

        Parameters
        ----------
        solver_params:
            Standard solver parameter list.
        blob_initial_condition:
            Passed through to ``set_initial_conditions`` as ``blob``.
        print_interval:
            Progress print frequency passed to the solve function.
        """
        ctx = self.build_context(solver_params)
        ctx = self.build_forms(ctx, solver_params)
        self.set_initial_conditions(ctx, solver_params, blob=blob_initial_condition)
        U_final = self.solve(ctx, solver_params, print_interval=print_interval)
        return ctx, U_final


def _require_callable(module: Any, attr: str, module_path: str) -> Callable[..., Any]:
    """Return a callable attribute or raise a descriptive contract error."""
    if not hasattr(module, attr) or not callable(getattr(module, attr)):
        raise ValueError(
            f"Module '{module_path}' is missing required callable '{attr}'."
        )
    return getattr(module, attr)
