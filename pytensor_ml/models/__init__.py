# Import from the submodule, never the package: this file runs first, so the package is still incomplete.
from pytensor_ml.models.registry import (
    architecture_name,
    build_from_config,
    register_builder,
)

__all__ = [
    "architecture_name",
    "build_from_config",
    "register_builder",
]
