from collections.abc import Generator
from functools import partial
from typing import Literal, cast

import numpy as np

from pytensor import config, graph_replace
from pytensor.compile.function import function
from pytensor.compile.sharedvalue import SharedVariable
from pytensor.graph import graph_inputs
from pytensor.graph.basic import Constant
from pytensor.printing import debugprint
from pytensor.tensor import TensorVariable

InitializationSchemes = Literal["zeros", "xavier_uniform", "xavier_normal"]


def required_graph_inputs(tensor: TensorVariable) -> Generator[TensorVariable, None, None]:
    return (
        cast(TensorVariable, var)
        for var in graph_inputs([tensor])
        if not isinstance(var, Constant | SharedVariable)
    )


class Model:
    def __init__(self, X: TensorVariable, y: TensorVariable | None = None):
        self.X = X
        self.y = y

        self._weight_values: list[np.ndarray[float]] | None = None
        self._predict_fn = None
        self._updates = {}

    def initialize(
        self,
        scheme: InitializationSchemes = "xavier_normal",
        random_seed: int | str | np.random.Generator | None = None,
    ):
        self._weight_values = initialize_weights(self, scheme, random_seed)
        # self._updates = initialize_updates(self)

    def initialize_from_pymc(
        self, outputs: TensorVariable | list[TensorVariable] | None = None, model=None
    ):
        import pymc as pm

        model = pm.modelcontext(model)

        if outputs is None:
            if self.y is None:
                raise ValueError(
                    "Must provide explicit output if y was not provided during Model construction"
                )
            outputs = self.y
        else:
            self.y = outputs

        self.initialize()
        init_dict = dict(zip([w.name for w in self.weights], self.weight_values))

        replacement_dict = {}
        for var in required_graph_inputs(self.y):
            if var.name not in model.named_vars:
                raise ValueError(f"Variable {var.name} not found in PyMC model")
            replacement_dict[var] = model[var.name]
            model.set_initval(model[var.name], init_dict[var.name])

        return graph_replace(outputs, replacement_dict, strict=True)

    @property
    def weights(self) -> list[TensorVariable]:
        return [var for var in required_graph_inputs(self.y) if var is not self.X]

    @property
    def weight_values(self) -> list[np.ndarray[float]]:
        if self._weight_values is None:
            raise ValueError("Weights have not been initialized")
        return self._weight_values

    @weight_values.setter
    def weight_values(self, values: list[np.ndarray[float]]):
        for i, new_value in enumerate(values):
            self._weight_values[i][:] = new_value

    @property
    def updates(self) -> dict[SharedVariable:SharedVariable]:
        if self._updates is None:
            raise ValueError("Updates have not been initialized")
        return self._updates

    @updates.setter
    def updates(self, new_updates: dict[SharedVariable:SharedVariable]):
        self._updates = self._updates | new_updates

    def predict(self, X_values: np.ndarray) -> np.ndarray:
        if self._predict_fn is None:
            f = function([*self.weights, self.X], self.y, updates=self._updates)
            self._predict_fn = partial(f, *self.weight_values)

        return self._predict_fn(X_values)

    def __str__(self):
        return debugprint(self.y, file="str")


def _zero_init(shape: tuple[int], *args) -> np.ndarray:
    return np.zeros(shape, dtype=config.floatX)


def _xavier_uniform_init(shape: tuple[int], rng: np.random.Generator) -> np.ndarray:
    scale = np.sqrt(6.0 / np.sum([x for x in shape if x is not None]))
    return rng.uniform(-scale, scale, size=shape).astype(config.floatX)


def _xavier_normal_init(shape: tuple[int], rng: np.random.Generator) -> np.ndarray:
    scale = np.sqrt(2.0 / np.sum([x for x in shape if x is not None]))
    return rng.normal(0, scale, size=shape).astype(config.floatX)


initialization_factory = {
    "zeros": _zero_init,
    "xavier_uniform": _xavier_uniform_init,
    "xavier_normal": _xavier_normal_init,
}


def initialize_weights(
    model, scheme: InitializationSchemes, random_seed: int | str | np.random.Generator | None
):
    if isinstance(random_seed, str):
        random_seed = sum(map(ord, random_seed))
    if isinstance(random_seed, int) or random_seed is None:
        random_seed = np.random.default_rng(random_seed)

    initial_values = []
    for var in model.weights:
        shape = var.type.shape
        f_initialize = initialization_factory[scheme]
        initial_values.append(f_initialize(shape, random_seed))

    return initial_values
