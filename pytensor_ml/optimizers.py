from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from functools import wraps

import numpy as np
import pytensor
import pytensor.tensor as pt

from pytensor import config
from pytensor.compile.function.types import Function
from pytensor.gradient import grad
from pytensor.tensor import TensorLike, TensorVariable, sqrt, tensor

from pytensor_ml.model import Model
from pytensor_ml.pytensorf import function, rewrite_pregrad


def split_output_wrapper(fn: Callable, *splits) -> Callable:
    """
    Wraps a function to return the weights and optimizer weights separately.

    Parameters
    ----------
    fn: Function
        The function to wrap.
    splits: int

    Returns
    -------
    Function
        A wrapped function that returns the model weights and optimizer weights separately.
    """
    splits = np.cumsum(splits)

    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        result = fn(*args, **kwargs)
        output = []
        cursor = 0
        for split in splits:
            output.append(result[cursor:split])
            cursor = split
        output.append(result[cursor:])
        return output

    return wrapped_fn


class Optimizer(ABC):
    def __init__(
        self,
        model: Model,
        loss_fn: Callable,
        ndim_out: int = 1,
        optimizer_weights: list[TensorVariable] | None = None,
        compile_kwargs: dict | None = None,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.ndim_out = ndim_out
        self.optimizer_weights = optimizer_weights if optimizer_weights is not None else []
        self._optimizer_weights_values = self._initialize_weights()

        self.update_fn = self.build_update_fn(compile_kwargs)

    @abstractmethod
    def update_parameters(self, params: Sequence[TensorVariable], loss: TensorVariable): ...

    @property
    def optimizer_weights_values(self) -> list[np.ndarray]:
        return self._optimizer_weights_values

    @optimizer_weights_values.setter
    def optimizer_weights_values(self, values: list[np.ndarray]):
        for i, new_value in enumerate(values):
            self._optimizer_weights_values[i][:] = new_value

    def _initialize_weights(self) -> list[np.ndarray]:
        if self.optimizer_weights:
            return [np.zeros(param.type.shape) for param in self.optimizer_weights]
        return []

    def _split_weights(
        self, all_weights: list[TensorLike]
    ) -> tuple[list[TensorLike], list[TensorLike]]:
        n_params = len(self.model.weights)
        return all_weights[:n_params], all_weights[n_params:]

    def build_update_fn(self, compile_kwargs) -> Function:
        """
        Compile a function to update model weights

        Compile a pytensor function that maps data (x), targets (y), and current parameter values to new parameter
        values. By default, the functions also returns the loss value. To return additional diagnostics, this method
        should be overloaded.

        Parameters
        ----------
        compile_kwargs: dict
            Keyword arguments passed to pytensor.function

        Returns
        -------
        update_fn: Function
            A function that updates the model weights given a batch of data, targets, and current weights.
        """
        compile_kwargs = compile_kwargs if compile_kwargs else {}
        x, y_hat = self.model.X, self.model.y

        label_slice = (slice(None),) * self.ndim_out + (0,) * (y_hat.ndim - self.ndim_out)
        y = y_hat[np.s_[label_slice]].type()

        weights = self.model.weights
        optimizer_weights = self.optimizer_weights
        update_inputs, update_outputs = self.model.updates

        loss = self.loss_fn(y, y_hat)
        loss = rewrite_pregrad(loss)

        all_weights = self.update_parameters(weights + optimizer_weights, loss)

        weights_fn_inputs = [pytensor.In(w, mutable=True) for w in self.model.weights]
        optimizer_weights_fn_inputs = [pytensor.In(w, mutable=True) for w in self.optimizer_weights]
        non_trainable_fn_inputs = [pytensor.In(w, mutable=True) for w in update_inputs]

        fn = function(
            [x, y, *weights_fn_inputs, *optimizer_weights_fn_inputs, *non_trainable_fn_inputs],
            [*all_weights, *update_outputs, loss],
            trust_input=True,
            **compile_kwargs,
        )

        return split_output_wrapper(fn, len(weights), len(optimizer_weights), len(update_outputs))

    def step(self, x_values, y_values) -> np.ndarray:
        """
        Update model weights in-place given a new batch of x_values and y_values.

        Returns
        -------
        loss
        """

        new_weights, new_optimizer_weights, new_non_trainable_values, loss_values = self.update_fn(
            x_values,
            y_values,
            *self.model.weight_values,
            *self.optimizer_weights_values,
            *self.model.non_trainable_values,
        )

        self.model.weight_values = new_weights
        self.model.non_trainable_values = new_non_trainable_values

        self.optimizer_weights_values = new_optimizer_weights

        return loss_values


class SGD(Optimizer):
    def __init__(
        self,
        model,
        loss_fn,
        *,
        ndim_out: int = 1,
        learning_rate: TensorLike = 0.01,
        compile_kwargs: dict | None = None,
    ):
        self.learning_rate = learning_rate
        super().__init__(model, loss_fn, ndim_out=ndim_out, compile_kwargs=compile_kwargs)

    def update_parameters(
        self, params: list[TensorVariable], loss: TensorVariable
    ) -> list[TensorVariable]:
        grads = grad(loss, params)
        new_params = []
        for param, d_loss_d_param in zip(params, grads):
            new_params.append(param - self.learning_rate * d_loss_d_param)

        return new_params


class ADAGrad(Optimizer):
    def __init__(
        self,
        model,
        loss_fn,
        *,
        ndim_out: int = 1,
        learning_rate: TensorLike = 0.01,
        epsilon: TensorLike = 1e-8,
        compile_kwargs: dict | None = None,
    ):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        g2_weights = [param.type() for param in model.weights]

        super().__init__(
            model,
            loss_fn,
            ndim_out=ndim_out,
            optimizer_weights=g2_weights,
            compile_kwargs=compile_kwargs,
        )

    def update_parameters(
        self, weights: list[TensorVariable], loss: TensorVariable
    ) -> list[TensorVariable]:
        weights, optimizer_weights = self._split_weights(weights)

        grads = grad(loss, weights)

        new_weights = []
        new_optimizer_weights = []

        for param, d_loss_d_param, g2 in zip(weights, grads, optimizer_weights):
            new_g2 = g2 + d_loss_d_param**2
            weight_update = d_loss_d_param / pt.sqrt(new_g2 + self.epsilon)
            new_weights.append(param - self.learning_rate * weight_update)
            new_optimizer_weights.append(new_g2)

        return new_weights + new_optimizer_weights


class Adam(Optimizer):
    def __init__(
        self,
        model,
        loss_fn,
        *,
        ndim_out: int = 1,
        learning_rate: TensorLike = 0.01,
        beta1: TensorLike = 0.9,
        beta2: TensorLike = 0.999,
        epsilon: TensorLike = 1e-8,
        compile_kwargs: dict | None = None,
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        m_weights = [param.type() for param in model.weights]
        v_weights = [param.type() for param in model.weights]
        t = tensor("t", shape=(1,), dtype=config.floatX)

        optimizer_weights = m_weights + v_weights + [t]
        super().__init__(
            model,
            loss_fn,
            ndim_out=ndim_out,
            optimizer_weights=optimizer_weights,
            compile_kwargs=compile_kwargs,
        )

    def update_parameters(
        self, weights: list[TensorVariable], loss: TensorVariable
    ) -> list[TensorVariable]:
        weights, optimizer_weights = self._split_weights(weights)
        t = optimizer_weights.pop(-1)
        m_weights, v_weights = self._split_weights(optimizer_weights)

        grads = grad(loss, weights)

        new_weights = []
        new_m_weights = []
        new_v_weights = []

        new_t = t + 1
        a_t = sqrt(1 - self.beta2**new_t) / (1 - self.beta1**new_t)

        for param, d_loss_d_param, m, v in zip(weights, grads, m_weights, v_weights):
            weight_update = a_t * m / (sqrt(v) + self.epsilon)
            new_weights.append(param - self.learning_rate * weight_update)

            new_m = self.beta1 * m + (1 - self.beta1) * d_loss_d_param
            new_v = self.beta2 * v + (1 - self.beta2) * d_loss_d_param**2
            new_m_weights.append(new_m)
            new_v_weights.append(new_v)

        return new_weights + new_m_weights + new_v_weights + [new_t]
