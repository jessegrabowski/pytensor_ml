from functools import partial
from typing import Any, Literal, cast

import numpy as np

from pytensor import config
from pytensor.printing import debugprint
from pytensor.tensor import TensorVariable

from pytensor_ml.params import (
    collect_non_trainable_updates,
    collect_trainable_params,
)
from pytensor_ml.pytensorf import function, rewrite_for_prediction

InitializationSchemes = Literal["zeros", "xavier_uniform", "xavier_normal"]


class Model:
    def __init__(
        self,
        X: TensorVariable,
        y: TensorVariable,
        random_state: Any | None = None,
        compile_kwargs: dict | None = None,
    ):
        self.X = X
        self.y = y

        self.rng = np.random.default_rng(random_state)

        self._compile_kwargs = compile_kwargs if compile_kwargs else {}
        self._weight_values: list[np.ndarray[float]] | None = None
        self._non_trainable_values: list[np.ndarray[float]] | None = None
        self._predict_fn = None

    def initialize_weights(
        self,
        scheme: InitializationSchemes = "xavier_normal",
        random_seed: int | str | np.random.Generator | None = None,
    ):
        self._weight_values = initialize_weights(self, scheme, random_seed)

    def initialize_non_trainable_values(self) -> list[np.ndarray]:
        if self.updates[0]:
            values = [
                self.rng.uniform(0, 1, param.type.shape).astype(param.type.dtype)
                for param in self.updates[0]
            ]
            self._non_trainable_values = values
        else:
            self._non_trainable_values = []

    @property
    def weights(self) -> list[TensorVariable]:
        return collect_trainable_params(self.y, exclude={self.X})

    @property
    def updates(self) -> tuple[list[TensorVariable], list[TensorVariable]]:
        update_dict = collect_non_trainable_updates(self.y)
        if update_dict:
            return list(update_dict.keys()), list(update_dict.values())
        return [], []

    @property
    def weight_values(self) -> list[np.ndarray[float]]:
        if self._weight_values is None:
            self.initialize_weights()
        return self._weight_values

    @weight_values.setter
    def weight_values(self, values: list[np.ndarray[float]]):
        for i, new_value in enumerate(values):
            self._weight_values[i][:] = new_value

    @property
    def non_trainable_values(self) -> list[np.ndarray[float]]:
        if self._non_trainable_values is None:
            self.initialize_non_trainable_values()
        return self._non_trainable_values

    @non_trainable_values.setter
    def non_trainable_values(self, values: list[np.ndarray[float]]):
        for i, new_value in enumerate(values):
            self._non_trainable_values[i][:] = new_value

    def predict(self, X_values: np.ndarray) -> np.ndarray:
        y_pred = rewrite_for_prediction(self.y)
        if self._predict_fn is None:
            f = function(
                [*self.weights, *self.updates[0], self.X],
                y_pred,
                **self._compile_kwargs,
            )
            self._predict_fn = partial(f, *self.weight_values, *self.non_trainable_values)

        return cast(np.ndarray, self._predict_fn(X_values))

    def __str__(self):
        return debugprint(self.y, file="str")


def _zero_init(shape: tuple[int], *args) -> np.ndarray:
    return np.zeros(shape, dtype=config.floatX)


def _xavier_uniform_init(shape: tuple[int], dtype, rng: np.random.Generator) -> np.ndarray:
    scale = np.sqrt(6.0 / np.sum([x for x in shape if x is not None]))
    return rng.uniform(-scale, scale, size=shape).astype(dtype)


def _xavier_normal_init(shape: tuple[int], dtype, rng: np.random.Generator) -> np.ndarray:
    scale = np.sqrt(2.0 / np.sum([x for x in shape if x is not None]))
    return rng.normal(0, scale, size=shape).astype(dtype)


initialization_factory = {
    "zeros": _zero_init,
    "xavier_uniform": _xavier_uniform_init,
    "xavier_normal": _xavier_normal_init,
}


def initialize_weights(
    model, scheme: InitializationSchemes, random_seed: int | str | np.random.Generator | None
):
    rng = np.random.default_rng(random_seed)

    initial_values = []
    for var in model.weights:
        shape = var.type.shape
        dtype = var.type.dtype
        f_initialize = initialization_factory[scheme]
        initial_values.append(f_initialize(shape, dtype, rng))

    return initial_values
