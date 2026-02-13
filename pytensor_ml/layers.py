from abc import ABC
from collections.abc import Callable
from typing import Any

import numpy as np
import pytensor.tensor as pt
import pytensor.tensor.random as ptr

from pytensor import config
from pytensor.compile.sharedvalue import shared

from pytensor_ml.params import non_trainable, trainable
from pytensor_ml.pytensorf import LayerOp


def shape_to_str(shape):
    inner = ",".join([str(st_dim) if st_dim is not None else "?" for st_dim in shape])
    return f"({inner})"


class Layer(ABC):
    def __call__(self, x: pt.TensorLike) -> pt.TensorVariable: ...


class LinearLayer(LayerOp):
    __props__ = ("n_in", "n_out", "bias")


class Linear(Layer):
    def __init__(self, name: str | None, n_in: int, n_out: int, bias: bool = True):
        self.name = name if name else "Linear"
        self.n_in = n_in
        self.n_out = n_out
        self.bias = bias

        W_value = np.zeros((n_in, n_out), dtype=config.floatX)
        self.W = trainable(W_value, f"{self.name}_W")

        if self.bias:
            b_value = np.zeros(n_out, dtype=config.floatX)
            self.b = trainable(b_value, f"{self.name}_b")

    def __call__(self, X: pt.TensorLike) -> pt.TensorLike:
        X = pt.as_tensor(X)

        init_st_shape = shape_to_str(X.type.shape)

        res = X @ self.W
        inputs = [X, self.W]
        if self.bias:
            res += self.b
            inputs.append(self.b)

        final_st_shape = shape_to_str(res.type.shape)

        ofg = LinearLayer(
            inputs=inputs,
            outputs=[res],
            name=f"{self.name}[{init_st_shape} -> {final_st_shape}]",
            n_in=self.n_in,
            n_out=self.n_out,
            bias=self.bias,
        )
        out = ofg(*inputs)
        out.name = f"{self.name}_output"

        return out


class DropoutLayer(LayerOp):
    __props__ = ("p",)


class Dropout(Layer):
    def __init__(self, name: str | None = None, p: float = 0.5, random_state: Any | None = None):
        if p < 0.0 or p > 1.0:
            raise ValueError(f"Dropout probability has to be between 0 and 1, but got {p}")
        self.name = name if name else "Dropout"
        self.p = p
        self.rng = shared(np.random.default_rng(random_state))

    def __call__(self, X: pt.TensorLike) -> pt.TensorLike:
        X = pt.as_tensor(X)
        p = pt.as_tensor(self.p, dtype=config.floatX)
        new_rng, mask = ptr.bernoulli(p=1 - p, size=X.shape, rng=self.rng).owner.outputs
        mask = mask.astype(config.floatX)

        X_masked = DropoutLayer(
            inputs=[X, mask],
            outputs=[pt.where(mask, ift=X / (1 - p), iff=0)],
            name=f"{self.name}[p = {self.p}]",
            p=self.p,
        )(X, mask)
        X_masked.name = f"{self.name}_output"

        return X_masked


class BatchNormLayer(LayerOp):
    __props__ = ("n_in", "epsilon", "momentum", "affine")

    def update_map(self):
        return {1: 3, 2: 4}


class NoRunningStatsBatchNormLayer(LayerOp):
    __props__ = ("n_in", "epsilon", "momentum", "affine")


class PredictionBatchNormLayer(LayerOp):
    __props__ = ("n_in", "epsilon", "momentum", "affine")


class BatchNorm2D(Layer):
    def __init__(
        self,
        name: str | None = None,
        n_in: int | None = None,
        epsilon: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        self.name = name if name else "BatchNorm"
        self.n_in = n_in
        self.epsilon = epsilon
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.scale = None
        self.loc = None
        self.running_mean = None
        self.running_var = None

        self.initialized = False
        self._initialize_params(None)

    def _initialize_params(self, X: pt.TensorLike | None):
        if self.initialized:
            return

        if self.n_in is None and X is None:
            return

        if X is not None:
            n_in = X.type.shape[-1]
        else:
            n_in = self.n_in

        if self.affine:
            scale_value = np.ones(n_in, dtype=config.floatX)
            loc_value = np.zeros(n_in, dtype=config.floatX)
            self.scale = trainable(scale_value, f"{self.name}_scale")
            self.loc = trainable(loc_value, f"{self.name}_loc")

        if self.track_running_stats:
            running_mean_value = np.zeros(n_in, dtype=config.floatX)
            running_var_value = np.ones(n_in, dtype=config.floatX)
            self.running_mean = non_trainable(running_mean_value, f"{self.name}_running_mean")
            self.running_var = non_trainable(running_var_value, f"{self.name}_running_var")

        self.initialized = True

    def __call__(self, X: pt.TensorLike) -> pt.TensorLike:
        X = pt.as_tensor(X)
        inputs = [X]

        self._initialize_params(X)

        mu = X.mean(axis=0)
        sigma_sq = X.var(axis=0)

        X_normalized = (X - mu) / pt.sqrt(sigma_sq + self.epsilon)

        if self.affine:
            X_rescaled = X_normalized * self.scale + self.loc
            inputs.extend([self.loc, self.scale])
        else:
            X_rescaled = X_normalized

        if self.track_running_stats:
            new_running_mean = self.momentum * mu + (1 - self.momentum) * self.running_mean
            new_running_var = self.momentum * sigma_sq + (1 - self.momentum) * self.running_var

            batch_norm_op = BatchNormLayer(
                inputs=[*inputs, self.running_mean, self.running_var],
                outputs=[X_rescaled, new_running_mean, new_running_var],
                name=f"{self.name}",
                n_in=self.n_in,
                epsilon=self.epsilon,
                momentum=self.momentum,
                affine=self.affine,
            )

            X_transformed, self.new_running_mean, self.new_running_var = batch_norm_op(
                *inputs, self.running_mean, self.running_var
            )

        else:
            batch_norm_op = NoRunningStatsBatchNormLayer(
                inputs=inputs,
                outputs=[X_rescaled],
                name=f"{self.name}",
                n_in=self.n_in,
                epsilon=self.epsilon,
                momentum=self.momentum,
                affine=self.affine,
            )

            X_transformed = batch_norm_op(*inputs)

        X_transformed.name = f"{self.name}_output"

        return X_transformed


def Input(name: str, shape: tuple[int]) -> pt.TensorLike:
    if not all(isinstance(dim, int) for dim in shape):
        raise ValueError("All dimensions must be integers")

    return pt.tensor(name=name, shape=shape)


def Sequential(*layers: Callable) -> Callable:
    def forward(x: pt.TensorLike) -> pt.TensorLike:
        for layer in layers:
            x = layer(x)
        return x

    return forward


Squeeze = pt.squeeze
Concatenate = pt.concatenate


__all__ = ["BatchNorm2D", "Concatenate", "Dropout", "Input", "Linear", "Sequential", "Squeeze"]
