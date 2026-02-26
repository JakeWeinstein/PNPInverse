# Backward-compatibility shim. New code should import from Forward.noise directly.
from Forward.noise import generate_noisy_data_robin  # noqa: F401
