# Backward-compatibility shim. New code should import from Forward.dirichlet_solver directly.
from Forward.dirichlet_solver import *  # noqa: F401, F403
from Forward.dirichlet_solver import (  # noqa: F401
    build_context,
    build_forms,
    set_initial_conditions,
    forsolve,
)
