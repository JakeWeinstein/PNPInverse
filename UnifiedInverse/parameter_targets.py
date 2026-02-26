# Backward-compatibility shim. New code should import from Inverse.parameter_targets directly.
from Inverse.parameter_targets import *  # noqa: F401, F403
from Inverse.parameter_targets import (  # noqa: F401
    ParameterTarget,
    build_default_target_registry,
    ensure_sequence,
)
