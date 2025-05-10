from abc import ABC
from collections.abc import Callable
from typing import Any

import numpy as np
import pytensor.tensor as pt
import pytensor.tensor.random as ptr

from pytensor import config
from pytensor.compile.sharedvalue import shared

from pytensor_ml.pytensorf import LayerOp


def shape_to_str(shape):
    inner = ",".join([str(st_dim) if st_dim is not None else "?" for st_dim in shape])
    return f"({inner})"


class Layer(ABC):
    def __call__(self, x: pt.TensorLike) -> pt.TensorVariable: ...


class LinearLayer(LayerOp):
    __props__ = ("n_in", "n_out", "bias")


class Linear(Layer):
    __props__ = ("name", "n_in", "n_out")

    def __init__(self, name: str | None, n_in: int, n_out: int):
        self.name = name if name else "Linear"
        self.n_in = n_in
        self.n_out = n_out

        self.W = pt.tensor(f"{self.name}_W", shape=(n_in, self.n_out))
        self.b = pt.tensor(f"{self.name}_b", shape=(self.n_out,))

    def __call__(self, X: pt.TensorLike) -> pt.TensorLike:
        X = pt.as_tensor(X)

        init_st_shape = shape_to_str(X.type.shape)

        res = X @ self.W + self.b

        final_st_shape = shape_to_str(res.type.shape)

        ofg = LinearLayer(
            inputs=[X, self.W, self.b],
            outputs=[res],
            name=f"{self.name}[{init_st_shape} -> {final_st_shape}]",
            n_in=self.n_in,
            n_out=self.n_out,
            bias=self.bias,
        )
        out = ofg(X, self.W, self.b)
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


__all__ = ["Concatenate", "Dropout", "Input", "Linear", "Sequential", "Squeeze"]
