import numpy as np

from pytensor.printing import debugprint
from pytensor.tensor.variable import TensorVariable

from pytensor_ml.params import TrainableParameter, collect_trainable_params
from pytensor_ml.pytensorf import compile_predict
from pytensor_ml.state import InitializationScheme, initialize_params


class Model:
    """A network's input and output, with conveniences to initialize its weights and run inference."""

    def __init__(self, X: TensorVariable, y: TensorVariable, compile_kwargs: dict | None = None):
        self.X = X
        self.y = y
        self._compile_kwargs = compile_kwargs or {}
        self._predict_fn = None

    @property
    def weights(self) -> list[TrainableParameter]:
        return collect_trainable_params(self.y)

    def initialize(
        self,
        scheme: InitializationScheme = "xavier_normal",
        seed: int | np.random.Generator | None = None,
    ) -> "Model":
        """
        Initialize the trainable weights in place and return self.

        Parameters
        ----------
        scheme : str
            Initialization scheme for the weights. Default 'xavier_normal'.
        seed : int or numpy Generator, optional
            Seed for reproducible initialization.
        """
        parameters = self.weights
        for parameter, value in zip(parameters, initialize_params(parameters, scheme, rng=seed)):
            parameter.set_value(value)
        return self

    def predict(self, X_values: np.ndarray) -> np.ndarray:
        if self._predict_fn is None:
            self._predict_fn = compile_predict(
                self.y, inputs=[self.X], compile_kwargs=self._compile_kwargs
            )

        result = self._predict_fn(X_values)
        return result if isinstance(result, np.ndarray) else np.asarray(result)

    def __str__(self):
        return debugprint(self.y, file="str")
