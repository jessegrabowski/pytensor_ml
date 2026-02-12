from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np

from pymc.util import RandomSeed
from pytensor.compile.sharedvalue import SharedVariable

from pytensor_ml.params import NonTrainableParameter, TrainableParameter

RandomState = RandomSeed | np.random.RandomState | np.random.Generator

InitializationScheme = Literal["zeros", "xavier_uniform", "xavier_normal", "unit_uniform"]

SamplingFunction = Callable[[tuple[int, ...], str, np.random.Generator], np.ndarray]


class Initializer(ABC):
    """
    Base class for parameter initializers.

    Can be used in two ways:
    - As a class: `XavierNormalInitializer(param, rng)` - directly initializes
    - As an instance: `init = XavierNormalInitializer(); init(param, rng)`
    """

    def __new__(cls, param: SharedVariable | None = None, rng: RandomState | None = None):
        # If called with a param, act as a function and initialize directly
        if param is not None:
            instance = object.__new__(cls)
            instance.__init__()
            return instance(param, rng)
        # Otherwise, return an instance for later use
        return object.__new__(cls)

    def __call__(self, param: SharedVariable, rng: RandomState | None = None) -> SharedVariable:
        param.set_value(self._sample_like(param, rng))
        return param

    @abstractmethod
    def sample(self, shape: tuple[int, ...], dtype: str, rng: RandomState) -> np.ndarray: ...

    def _sample_like(self, param: SharedVariable, rng: RandomState | None = None) -> np.ndarray:
        rng = np.random.default_rng(rng)
        value = param.get_value()
        return self.sample(value.shape, str(value.dtype), rng)


class ZeroInitializer(Initializer):
    def sample(self, shape: tuple[int, ...], dtype: str, rng: RandomState) -> np.ndarray:
        return np.zeros(shape, dtype=dtype)


class UnitUniformInitializer(Initializer):
    def sample(self, shape: tuple[int, ...], dtype: str, rng: RandomState) -> np.ndarray:
        return rng.uniform(0.0, 1.0, size=shape).astype(dtype)


class XavierUniformInitializer(Initializer):
    def sample(self, shape: tuple[int, ...], dtype: str, rng: RandomState) -> np.ndarray:
        scale = np.sqrt(6.0 / np.sum([x for x in shape if x is not None]))
        return rng.uniform(-scale, scale, size=shape).astype(dtype)


class XavierNormalInitializer(Initializer):
    def sample(self, shape: tuple[int, ...], dtype: str, rng: RandomState) -> np.ndarray:
        scale = np.sqrt(2.0 / np.sum([x for x in shape if x is not None]))
        return rng.normal(0, scale, size=shape).astype(dtype)


class CustomInitializer(Initializer):
    def __new__(
        cls,
        sample_fn: SamplingFunction | None = None,
        param: SharedVariable | None = None,
        rng: RandomState | None = None,
    ):
        instance = object.__new__(cls)
        if sample_fn is not None:
            instance._sample_fn = sample_fn
        if param is not None:
            instance.__init__(sample_fn)
            return instance(param, rng)
        return instance

    def __init__(self, sample_fn: SamplingFunction):
        self._sample_fn = sample_fn

    def sample(self, shape: tuple[int, ...], dtype: str, rng: RandomState) -> np.ndarray:
        return self._sample_fn(shape, dtype, rng)


_INIT_FUNCTIONS: dict[str, type[Initializer]] = {
    "zeros": ZeroInitializer,
    "xavier_uniform": XavierUniformInitializer,
    "xavier_normal": XavierNormalInitializer,
    "unit_uniform": UnitUniformInitializer,
}


def initialize_params(
    params: Sequence[SharedVariable],
    scheme: InitializationScheme = "xavier_normal",
    rng: RandomState | None = None,
) -> list[np.ndarray]:
    """
    Initialize parameter values using the specified scheme.

    Parameters
    ----------
    params
        SharedVariables to initialize values for.
    scheme
        Initialization scheme to use.
    rng
        Random number generator. If None, a new one is created.

    Returns
    -------
    list of np.ndarray
        Initialized values matching the shapes and dtypes of params.
    """
    rng = np.random.default_rng(rng)

    initializer = _INIT_FUNCTIONS[scheme]()
    results = []
    for var in params:
        value = var.get_value()
        results.append(initializer.sample(value.shape, str(value.dtype), rng))
    return results


class OptimizerState:
    """
    Container for all mutable training state using SharedVariables.

    This class manages:
    - Model parameters (trainable weights)
    - Optimizer state (momentum buffers, adaptive learning rate accumulators, etc.)
    - Non-trainable layer state (running statistics for batch normalization, etc.)
    """

    def __init__(
        self,
        params: Sequence[TrainableParameter],
        optimizer_params: Sequence[TrainableParameter] | None = None,
        non_trainable_params: Sequence[NonTrainableParameter] | None = None,
    ):
        """
        Create an OptimizerState with the given parameters.

        Parameters
        ----------
        params
            Trainable parameters.
        optimizer_params
            Optimizer states (e.g., momentum, variance estimates).
        non_trainable_params
            Non-trainable state (e.g., running mean/var for BatchNorm).
        """
        self._params = list(params)
        self._optimizer_params = list(optimizer_params) if optimizer_params else []
        self._non_trainable_params = list(non_trainable_params) if non_trainable_params else []

    @property
    def params(self) -> list[TrainableParameter]:
        return self._params

    @property
    def optimizer_params(self) -> list[TrainableParameter]:
        return self._optimizer_params

    @property
    def non_trainable_params(self) -> list[NonTrainableParameter]:
        return self._non_trainable_params

    @property
    def param_values(self) -> list[np.ndarray]:
        return [p.get_value() for p in self._params]

    @param_values.setter
    def param_values(self, values: Sequence[np.ndarray]) -> None:
        for param, value in zip(self._params, values, strict=True):
            param.set_value(value)

    @property
    def optimizer_values(self) -> list[np.ndarray]:
        return [p.get_value() for p in self._optimizer_params]

    @optimizer_values.setter
    def optimizer_values(self, values: Sequence[np.ndarray]) -> None:
        for param, value in zip(self._optimizer_params, values, strict=True):
            param.set_value(value)

    @property
    def non_trainable_values(self) -> list[np.ndarray]:
        return [p.get_value() for p in self._non_trainable_params]

    @non_trainable_values.setter
    def non_trainable_values(self, values: Sequence[np.ndarray]) -> None:
        for param, value in zip(self._non_trainable_params, values, strict=True):
            param.set_value(value)

    def initialize(
        self,
        param_scheme: InitializationScheme = "xavier_normal",
        optimizer_scheme: InitializationScheme = "zeros",
        non_trainable_scheme: InitializationScheme = "unit_uniform",
        seed: Any = None,
    ) -> "OptimizerState":
        """
        Initialize all state values.

        Parameters
        ----------
        param_scheme
            Initialization scheme for model parameters.
        optimizer_scheme
            Initialization scheme for optimizer state (typically zeros).
        non_trainable_scheme
            Initialization scheme for non-trainable state.
        seed
            Random seed for reproducibility.

        Returns
        -------
        self
            Returns self for method chaining.
        """
        rng = np.random.default_rng(seed)

        param_init = _INIT_FUNCTIONS[param_scheme]
        for p in self._params:
            param_init(p, rng)

        optimizer_init = _INIT_FUNCTIONS[optimizer_scheme]
        for p in self._optimizer_params:
            optimizer_init(p, rng)

        non_trainable_init = _INIT_FUNCTIONS[non_trainable_scheme]
        for p in self._non_trainable_params:
            non_trainable_init(p, rng)

        return self

    def state_dict(self) -> dict[str, np.ndarray]:
        """
        Return all state as a dictionary for checkpointing.

        Returns
        -------
        dict
            Dictionary mapping parameter names to their values.
        """
        result = {}

        for var in self._params:
            key = var.name or f"param_{id(var)}"
            result[key] = var.get_value().copy()

        for var in self._optimizer_params:
            key = f"optimizer/{var.name or id(var)}"
            result[key] = var.get_value().copy()

        for var in self._non_trainable_params:
            key = f"non_trainable/{var.name or id(var)}"
            result[key] = var.get_value().copy()

        return result

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        """
        Restore state from a dictionary.

        Parameters
        ----------
        state
            Dictionary mapping parameter names to values, as returned by state_dict().
        """
        for var in self._params:
            key = var.name or f"param_{id(var)}"
            if key in state:
                var.set_value(state[key])

        for var in self._optimizer_params:
            key = f"optimizer/{var.name or id(var)}"
            if key in state:
                var.set_value(state[key])

        for var in self._non_trainable_params:
            key = f"non_trainable/{var.name or id(var)}"
            if key in state:
                var.set_value(state[key])

    def save(self, path: str) -> None:
        """Save state to a .npz file."""
        np.savez(path, **self.state_dict())

    def load(self, path: str) -> None:
        """Load state from a .npz file."""
        with np.load(path) as data:
            self.load_state_dict(dict(data))
