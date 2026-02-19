"""Parameter-target registry for unified inverse inference.

Each target describes:
- how to inject true/guess values into ``solver_params``
- which controls should be optimized from solver context ``ctx``
- how to map optimizer outputs back to physical parameter values
- default bounds and callback behavior

To add a new parameter in the future, define a new :class:`ParameterTarget`
entry and register it in :func:`build_default_target_registry`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .solver_interface import as_species_list, deep_copy_solver_params


ObjectiveField = str


def ensure_sequence(value: Any) -> List[Any]:
    """Normalize scalar/list optimizer outputs into a list."""
    # firedrake.adjoint can return either a single control or a sequence
    # depending on the number of control variables.
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _format_parameter_value(value: Any) -> str:
    """Format parameter values in fixed-width decimal (no scientific notation)."""
    try:
        v = float(value)
    except Exception:
        return f"{str(value):>12}"
    if not np.isfinite(v):
        return f"{str(v):>12}"
    # Fixed width keeps optimizer callback columns stable across iterations.
    if abs(v) < 5e-7:
        v = 0.0
    return f"{v:>12.6f}"


@dataclass(frozen=True)
class ParameterTarget:
    """Definition of an inferable solver parameter.

    Attributes
    ----------
    key:
        Stable target identifier used by the CLI/interface.
    description:
        Human-readable description for docs and help text.
    objective_fields:
        Which measured fields should appear in the objective. Supported values:
        ``"concentration"`` and ``"phi"``.
    apply_value_inplace:
        Function that mutates ``solver_params`` with a true/guess value.
    controls_from_context:
        Function that returns Firedrake control functions from solver context.
    estimate_from_controls:
        Function that converts optimized controls into physical parameter values.
    default_bounds_factory:
        Optional factory for optimizer bounds when caller does not provide bounds.
    eval_cb_pre_factory:
        Optional callback factory for ``ReducedFunctional(eval_cb_pre=...)``.
    eval_cb_post_factory:
        Optional callback factory for ``ReducedFunctional(eval_cb_post=...)``.
    """

    key: str
    description: str
    objective_fields: Tuple[ObjectiveField, ...]
    apply_value_inplace: Callable[[List[Any], Any], None]
    controls_from_context: Callable[[Dict[str, Any]], Sequence[Any]]
    estimate_from_controls: Callable[[Any], Any]
    default_bounds_factory: Optional[Callable[[int], Any]] = None
    eval_cb_pre_factory: Optional[Callable[[Dict[str, Any]], Callable[[Any], None]]] = None
    eval_cb_post_factory: Optional[
        Callable[[Dict[str, Any]], Callable[[float, Any], None]]
    ] = None

    def apply_value(self, solver_params: Sequence[Any], value: Any) -> List[Any]:
        """Return a deep-copied ``solver_params`` with ``value`` applied."""
        params_copy = deep_copy_solver_params(solver_params)
        self.apply_value_inplace(params_copy, value)
        return params_copy

    def default_bounds(self, n_species: int):
        """Return target-specific default bounds or ``None``."""
        if self.default_bounds_factory is None:
            return None
        return self.default_bounds_factory(int(n_species))


def build_default_target_registry() -> Dict[str, ParameterTarget]:
    """Build the default target registry used across scripts and studies."""
    return {
        "diffusion": _build_diffusion_target(),
        "dirichlet_phi0": _build_phi0_target(),
        "robin_kappa": _build_robin_kappa_target(),
    }


def _build_diffusion_target() -> ParameterTarget:
    def apply_value_inplace(solver_params: List[Any], value: Any) -> None:
        n_species = int(solver_params[0])
        # Solver parameter slot 5 stores D_vals in the canonical 11-entry list.
        solver_params[5] = as_species_list(value, n_species, "D_vals")

    def controls_from_context(ctx: Dict[str, Any]) -> Sequence[Any]:
        # Forward-form builder exposes logD functions as optimizable controls.
        return ctx["logD_funcs"]

    def estimate_from_controls(controls: Any) -> List[float]:
        control_list = ensure_sequence(controls)
        # Diffusion is parameterized in log-space, so map controls back with exp.
        return [float(np.exp(ctrl.dat.data[0])) for ctrl in control_list]

    def eval_cb_post_factory(_ctx: Dict[str, Any]) -> Callable[[float, Any], None]:
        def eval_cb_post(j: float, m: Any) -> None:
            m_list = ensure_sequence(m)
            logd_values = [float(v.dat.data[0]) for v in m_list]
            d_values = [float(np.exp(v)) for v in logd_values]
            logd_str = ", ".join(_format_parameter_value(v) for v in logd_values)
            d_str = ", ".join(_format_parameter_value(v) for v in d_values)
            print(f"[opt] j={float(j):>14.6e}  logD=[{logd_str}]  D=[{d_str}]")

        return eval_cb_post

    return ParameterTarget(
        key="diffusion",
        description="Infer per-species diffusion coefficients D in log-parameter space.",
        objective_fields=("concentration",),
        apply_value_inplace=apply_value_inplace,
        controls_from_context=controls_from_context,
        estimate_from_controls=estimate_from_controls,
        default_bounds_factory=None,
        eval_cb_pre_factory=None,
        eval_cb_post_factory=eval_cb_post_factory,
    )


def _build_phi0_target() -> ParameterTarget:
    def apply_value_inplace(solver_params: List[Any], value: Any) -> None:
        # Solver parameter slot 9 stores scalar phi0.
        solver_params[9] = float(value)

    def controls_from_context(ctx: Dict[str, Any]) -> Sequence[Any]:
        return [ctx["phi0_func"]]

    def estimate_from_controls(controls: Any) -> float:
        return float(ensure_sequence(controls)[0].dat.data[0])

    def default_bounds_factory(_n_species: int):
        # Keep phi0 positive in bounded optimizers by default.
        return (1e-8, None)

    def eval_cb_pre_factory(ctx: Dict[str, Any]) -> Callable[[Any], None]:
        phi0_func = ctx["phi0_func"]

        def eval_cb_pre(m: Any) -> None:
            # Synchronize trial value from optimizer into the boundary-control function.
            m0 = ensure_sequence(m)[0]
            phi0_func.assign(m0)

        return eval_cb_pre

    def eval_cb_post_factory(_ctx: Dict[str, Any]) -> Callable[[float, Any], None]:
        def eval_cb_post(j: float, m: Any) -> None:
            m0 = ensure_sequence(m)[0]
            print(f"[opt] j={float(j):>14.6e}  phi0={_format_parameter_value(float(m0.dat.data[0]))}")

        return eval_cb_post

    return ParameterTarget(
        key="dirichlet_phi0",
        description="Infer Dirichlet electric-potential boundary value phi0.",
        objective_fields=("phi",),
        apply_value_inplace=apply_value_inplace,
        controls_from_context=controls_from_context,
        estimate_from_controls=estimate_from_controls,
        default_bounds_factory=default_bounds_factory,
        eval_cb_pre_factory=eval_cb_pre_factory,
        eval_cb_post_factory=eval_cb_post_factory,
    )


def _build_robin_kappa_target() -> ParameterTarget:
    def apply_value_inplace(solver_params: List[Any], value: Any) -> None:
        n_species = int(solver_params[0])
        kappa_vals = as_species_list(value, n_species, "robin_kappa")

        params = solver_params[10]
        if not isinstance(params, dict):
            raise ValueError(
                "solver_params[10] must be a dict when optimizing robin_kappa."
            )

        # Robin settings live under params["robin_bc"] and are created if missing.
        robin_cfg = params.setdefault("robin_bc", {})
        if not isinstance(robin_cfg, dict):
            raise ValueError("params['robin_bc'] must be a dictionary.")
        robin_cfg["kappa"] = kappa_vals

    def controls_from_context(ctx: Dict[str, Any]) -> Sequence[Any]:
        # Robin forward-form builder exposes one kappa control per species.
        return ctx["kappa_funcs"]

    def estimate_from_controls(controls: Any) -> List[float]:
        control_list = ensure_sequence(controls)
        return [float(ctrl.dat.data[0]) for ctrl in control_list]

    def default_bounds_factory(n_species: int):
        # Per-species lower bounds enforce nonnegative transfer coefficients.
        return [[1e-8 for _ in range(n_species)], [None for _ in range(n_species)]]

    def eval_cb_post_factory(_ctx: Dict[str, Any]) -> Callable[[float, Any], None]:
        def eval_cb_post(j: float, m: Any) -> None:
            m_list = ensure_sequence(m)
            kappa_vals = [float(v.dat.data[0]) for v in m_list]
            kappa_str = ", ".join(_format_parameter_value(v) for v in kappa_vals)
            print(f"[opt] j={float(j):>14.6e}  kappa=[{kappa_str}]")

        return eval_cb_post

    return ParameterTarget(
        key="robin_kappa",
        description="Infer Robin boundary transfer coefficient(s) kappa.",
        objective_fields=("concentration", "phi"),
        apply_value_inplace=apply_value_inplace,
        controls_from_context=controls_from_context,
        estimate_from_controls=estimate_from_controls,
        default_bounds_factory=default_bounds_factory,
        eval_cb_pre_factory=None,
        eval_cb_post_factory=eval_cb_post_factory,
    )
