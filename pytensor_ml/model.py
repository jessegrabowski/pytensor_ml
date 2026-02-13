from typing import Any

import numpy as np

from pytensor.printing import debugprint
from pytensor.tensor.variable import TensorVariable

from pytensor_ml.params import (
    NonTrainableParameter,
    TrainableParameter,
    collect_non_trainable_params,
    collect_non_trainable_updates,
    collect_trainable_params,
)
from pytensor_ml.pytensorf import function, rewrite_for_prediction
from pytensor_ml.state import InitializationScheme, OptimizerState


class Model:
    def __init__(
        self,
        X: TensorVariable,
        y: TensorVariable,
        compile_kwargs: dict | None = None,
    ):
        self.X = X
        self.y = y

        self._compile_kwargs = compile_kwargs if compile_kwargs else {}
        self._state: OptimizerState | None = None
        self._predict_fn = None

    def _ensure_state(self) -> OptimizerState:
        if self._state is None:
            self._state = OptimizerState(
                params=self.weights,
                non_trainable_params=collect_non_trainable_params(self.y),
            )
        return self._state

    def initialize(
        self,
        scheme: InitializationScheme = "xavier_normal",
        seed: Any = None,
    ) -> "Model":
        """
        Initialize model weights.

        Parameters
        ----------
        scheme
            Initialization scheme for model parameters.
        seed
            Random seed for reproducibility.

        Returns
        -------
        self
            Returns self for method chaining.
        """
        self._ensure_state().initialize(param_scheme=scheme, seed=seed)
        return self

    @property
    def weights(self) -> list[TrainableParameter]:
        return collect_trainable_params(self.y)

    @property
    def non_trainable(self) -> list[NonTrainableParameter]:
        return collect_non_trainable_params(self.y)

    @property
    def updates(self) -> dict[NonTrainableParameter, TensorVariable]:
        return collect_non_trainable_updates(self.y)

    @property
    def state(self) -> OptimizerState:
        return self._ensure_state()

    @property
    def weight_values(self) -> list[np.ndarray]:
        return self._ensure_state().param_values

    @weight_values.setter
    def weight_values(self, values: list[np.ndarray]) -> None:
        self._ensure_state().param_values = values

    @property
    def non_trainable_values(self) -> list[np.ndarray]:
        return self._ensure_state().non_trainable_values

    @non_trainable_values.setter
    def non_trainable_values(self, values: list[np.ndarray]) -> None:
        self._ensure_state().non_trainable_values = values

    def predict(self, X_values: np.ndarray) -> np.ndarray:
        if self._predict_fn is None:
            y_pred = rewrite_for_prediction(self.y)
            self._predict_fn = function(
                [self.X],
                y_pred,
                **self._compile_kwargs,
            )

        result = self._predict_fn(X_values)
        return result if isinstance(result, np.ndarray) else np.asarray(result)

    def __str__(self):
        return debugprint(self.y, file="str")
