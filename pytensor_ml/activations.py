import numpy as np
import pytensor.tensor as pt

from pytensor_ml.layers import Layer


class Activation(Layer): ...


class ReLU(Activation):
    def __call__(self, x: pt.TensorLike) -> pt.TensorVariable:
        out = pt.maximum(0, x)
        out.name = "ReLU"
        return out


class LeakyReLU(Activation):
    def __init__(self, negative_slope: pt.TensorLike = 0.01):
        self.negative_slope = negative_slope

    def __call__(self, x: pt.TensorLike) -> pt.TensorVariable:
        out = pt.switch(x > 0, x, -self.negative_slope * x)
        out.name = "LeakyReLU"
        return out


class Tanh(Activation):
    def __call__(self, x: pt.TensorLike) -> pt.TensorVariable:
        out = pt.tanh(x)
        out.name = "TanH"
        return out


class Sigmoid(Activation):
    def __call__(self, x: pt.TensorLike) -> pt.TensorVariable:
        out = pt.sigmoid(x)
        out.name = "Sigmoid"
        return out


class SoftPlus(Activation):
    def __call__(self, x: pt.TensorLike) -> pt.TensorVariable:
        out = pt.softplus(x)
        out.name = "SoftPlus"
        return out


class GELU(Activation):
    r"""
    Gaussian Error Linear Unit.

    Compute :math:`\mathrm{GELU}(x) = x \, \Phi(x)`, where :math:`\Phi` is the standard normal
    cumulative distribution function:

    .. math::

        \mathrm{GELU}(x) = \frac{x}{2} \left(1 + \operatorname{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right).

    Parameters
    ----------
    approximate : bool, optional
        Use the tanh approximation

        .. math::

            \mathrm{GELU}(x) \approx \frac{x}{2}
            \left(1 + \tanh\!\left[\sqrt{2/\pi}\,(x + 0.044715\,x^3)\right]\right)

        This is the variant HuggingFace calls ``"gelu_new"`` / ``"gelu_pytorch_tanh"``, PyTorch exposes
        as ``nn.GELU(approximate="tanh")``, and Flax as ``gelu(approximate=True)``; GPT-2 uses it. It is
        cheaper to evaluate than the exact :math:`\operatorname{erf}` form. Default is True.
    """

    def __init__(self, approximate: bool = True):
        self.approximate = approximate

    def __call__(self, x: pt.TensorLike) -> pt.TensorVariable:
        x = pt.as_tensor(x)
        if self.approximate:
            out = 0.5 * x * (1 + pt.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
        else:
            out = 0.5 * x * (1 + pt.erf(x / np.sqrt(2.0)))
        out.name = "GELU"
        return out


class Softmax(Activation):
    def __init__(self, axis: int = -1):
        self.axis = axis

    def __call__(self, x: pt.TensorLike) -> pt.TensorVariable:
        out = pt.special.softmax(x, axis=self.axis)
        out.name = "Softmax"
        return out


__all__ = ["GELU", "LeakyReLU", "ReLU", "Sigmoid", "SoftPlus", "Softmax", "Tanh"]
