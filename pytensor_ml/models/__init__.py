# Import from the submodule, never the package: this file runs first, so the package is still incomplete.
import pytensor_ml.models.clip  # registers the CLIP builders
import pytensor_ml.models.gpt2  # registers the GPT-2 builder

from pytensor_ml.models.keys import KeyMap, channels_last
from pytensor_ml.models.registry import (
    architecture_name,
    build_from_config,
    register_builder,
)

__all__ = [
    "KeyMap",
    "architecture_name",
    "build_from_config",
    "channels_last",
    "register_builder",
]
