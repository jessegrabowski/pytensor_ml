from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import numpy as np
import pytensor.tensor as pt

from pytensor import config
from pytensor.compile.function.types import Function
from pytensor.gradient import grad
from pytensor.tensor import TensorLike, TensorVariable, sqrt

from pytensor_ml.model import Model
from pytensor_ml.params import non_trainable, trainable
from pytensor_ml.pytensorf import function, rewrite_pregrad


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

        self.update_fn = self.build_update_fn(compile_kwargs)

    @abstractmethod
    def update_parameters(self, params: Sequence[TensorVariable], loss: TensorVariable): ...

    def _split_weights(
        self, all_weights: list[TensorLike]
    ) -> tuple[list[TensorLike], list[TensorLike]]:
        n_params = len(self.model.weights)
        return all_weights[:n_params], all_weights[n_params:]

    def build_update_fn(self, compile_kwargs) -> Function:
        """
        Compile a function to update model weights

        Compile a pytensor function that maps data (x), targets (y) to loss, updating
        all SharedVariables (weights, optimizer state, non-trainable state) in place.

        Parameters
        ----------
        compile_kwargs: dict
            Keyword arguments passed to pytensor.function

        Returns
        -------
        update_fn: Function
            A function that updates the model weights given a batch of data and targets.
        """
        compile_kwargs = compile_kwargs if compile_kwargs else {}
        x, y_hat = self.model.X, self.model.y

        label_slice = (slice(None),) * self.ndim_out + (0,) * (y_hat.ndim - self.ndim_out)
        y = y_hat[np.s_[label_slice]].type()

        weights = self.model.weights
        optimizer_weights = self.optimizer_weights

        loss = self.loss_fn(y, y_hat)
        loss = rewrite_pregrad(loss)

        # Get new weight values from the optimizer
        new_all_weights = self.update_parameters(weights + optimizer_weights, loss)
        new_weights = new_all_weights[: len(weights)]
        new_optimizer_weights = new_all_weights[len(weights) :]

        # Build updates dict: old_shared -> new_value
        updates = {}
        for old, new in zip(weights, new_weights):
            updates[old] = new
        for old, new in zip(optimizer_weights, new_optimizer_weights):
            updates[old] = new

        # Add non-trainable updates (e.g., BatchNorm running stats)
        non_trainable_updates = self.model.updates
        updates.update(non_trainable_updates)

        fn = function(
            [x, y],
            [loss],
            updates=updates,
            trust_input=True,
            **compile_kwargs,
        )

        return fn

    def step(self, x_values, y_values) -> np.ndarray:
        """
        Update model weights in-place given a new batch of x_values and y_values.

        Returns
        -------
        loss
        """
        (loss_values,) = self.update_fn(x_values, y_values)
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
        g2_weights = [
            trainable(np.zeros_like(param.get_value()), name=f"{param.name}_g2")
            for param in model.weights
        ]

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


class Adadelta(Optimizer):
    def __init__(
        self,
        model,
        loss_fn,
        *,
        ndim_out: int = 1,
        learning_rate: TensorLike = 1.0,
        rho: TensorLike = 0.9,
        epsilon: TensorLike = 1e-8,
        compile_kwargs: dict | None = None,
    ):
        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon

        u_weights = [
            trainable(np.zeros_like(param.get_value()), name=f"{param.name}_u")
            for param in model.weights
        ]
        v_weights = [
            trainable(np.zeros_like(param.get_value()), name=f"{param.name}_v")
            for param in model.weights
        ]
        optimizer_weights = u_weights + v_weights
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
        u_weights, v_weights = self._split_weights(optimizer_weights)

        grads = grad(loss, weights)

        new_weights = []
        new_u_weights = []
        new_v_weights = []

        for param, d_loss_d_param, u, v in zip(weights, grads, u_weights, v_weights):
            new_v = v * self.rho + d_loss_d_param**2 * (1 - self.rho)
            weight_update = (sqrt(u + self.epsilon) / sqrt(new_v + self.epsilon)) * d_loss_d_param
            new_u = u * self.rho + weight_update**2 * (1 - self.rho)

            new_weights.append(param - self.learning_rate * weight_update)
            new_u_weights.append(new_u)
            new_v_weights.append(new_v)

        return new_weights + new_u_weights + new_v_weights


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

        m_weights = [
            trainable(np.zeros_like(param.get_value()), name=f"{param.name}_m")
            for param in model.weights
        ]
        v_weights = [
            trainable(np.zeros_like(param.get_value()), name=f"{param.name}_v")
            for param in model.weights
        ]
        t = non_trainable(np.array(0, dtype=config.floatX), name="t")

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


class AdamW(Optimizer):
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
        weight_decay: TensorLike = 0.01,
        compile_kwargs: dict | None = None,
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay

        m_weights = [
            trainable(np.zeros_like(param.get_value()), name=f"{param.name}_m")
            for param in model.weights
        ]
        v_weights = [
            trainable(np.zeros_like(param.get_value()), name=f"{param.name}_v")
            for param in model.weights
        ]
        t = non_trainable(np.array(0, dtype=config.floatX), name="t")

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
            decayed_param = (1 - self.weight_decay * self.learning_rate) * param
            weight_update = a_t * m / (sqrt(v) + self.epsilon)
            new_weights.append(decayed_param - self.learning_rate * weight_update)

            new_m = self.beta1 * m + (1 - self.beta1) * d_loss_d_param
            new_v = self.beta2 * v + (1 - self.beta2) * d_loss_d_param**2
            new_m_weights.append(new_m)
            new_v_weights.append(new_v)

        return new_weights + new_m_weights + new_v_weights + [new_t]
