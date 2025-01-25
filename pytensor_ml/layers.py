from abc import ABC
from collections.abc import Callable

import numpy as np
import pytensor.tensor.random as ptr

from pytensor.compile.builders import OpFromGraph
from pytensor.compile.sharedvalue import shared
from pytensor.tensor import TensorLike, as_tensor, concatenate, squeeze, tensor


def shape_to_str(shape):
    inner = ",".join([str(st_dim) if st_dim is not None else "?" for st_dim in shape])
    return f"({inner})"


class Layer(ABC):
    def __call__(self, x: TensorLike) -> TensorLike: ...


class LinearLayer(OpFromGraph): ...


class Linear(Layer):
    __props__ = ("name", "n_in", "n_out")

    def __init__(self, name: str | None, n_in: int, n_out: int):
        self.name = name if name else "Linear"
        self.n_in = n_in
        self.n_out = n_out

        self.W = tensor(f"{self.name}_W", shape=(n_in, self.n_out))
        self.b = tensor(f"{self.name}_b", shape=(self.n_out,))

    def __call__(self, X: TensorLike) -> TensorLike:
        X = as_tensor(X)

        init_st_shape = shape_to_str(X.type.shape)

        res = X @ self.W + self.b

        final_st_shape = shape_to_str(res.type.shape)

        ofg = LinearLayer(
            inputs=[X, self.W, self.b],
            outputs=[res],
            inline=True,
            name=f"{self.name}[{init_st_shape} -> {final_st_shape}]",
        )
        out = ofg(X, self.W, self.b)
        out.name = f"{self.name}_output"

        return out


class DropoutLayer(OpFromGraph): ...


class Dropout(Layer):
    __props__ = ("name", "p")

    def __init__(self, name: str | None, p: float = 0.5):
        self.name = name if name else "Dropout"
        self.p = p
        self.rng = shared(np.random.default_rng())

    def __call__(self, X: TensorLike) -> TensorLike:
        X = as_tensor(X)
        new_rng, mask = ptr.bernoulli(p=1 - self.p, size=X.shape, rng=self.rng).owner.outputs

        X_masked, _ = DropoutLayer(inputs=[X, self.rng], outputs=[X * mask, new_rng])(X, self.rng)
        X_masked.name = f"{self.name}[p = {self.p}]"

        return X_masked


def Input(name: str, shape: tuple[int]) -> TensorLike:
    if not all(isinstance(dim, int) for dim in shape):
        raise ValueError("All dimensions must be integers")

    return tensor(name=name, shape=shape)


def Sequential(*layers: Callable) -> Callable:
    def forward(x: TensorLike) -> TensorLike:
        for layer in layers:
            x = layer(x)
        return x

    return forward


Squeeze = squeeze
Concatenate = concatenate


__all__ = ["Concatenate", "Input", "Linear", "Sequential", "Squeeze"]
