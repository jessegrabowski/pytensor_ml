import logging
import sys

from pytensor_ml._version import get_versions
from pytensor_ml.pytensorf import function
from pytensor_ml.rewriting.layers import inline_layers, remove_dropout_for_prediction  # noqa: F401

_log = logging.getLogger(__name__)

if not logging.root.handlers:
    _log.setLevel(logging.INFO)
    if len(_log.handlers) == 0:
        handler = logging.StreamHandler(sys.stderr)
        _log.addHandler(handler)


__version__ = get_versions()["version"]

__all__ = ["function"]
