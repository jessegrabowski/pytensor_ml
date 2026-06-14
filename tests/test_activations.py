import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from scipy.special import erf

from pytensor_ml.activations import GELU, Swish


def gelu_exact(x):
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))


def gelu_tanh(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


@pytest.mark.parametrize(
    "approximate, reference", [(False, gelu_exact), (True, gelu_tanh)], ids=["exact", "tanh"]
)
def test_gelu_matches_reference(approximate, reference):
    x = pt.vector("x")
    f = pytensor.function([x], GELU(approximate=approximate)(x))
    values = np.linspace(-6, 6, 101)
    np.testing.assert_allclose(f(values), reference(values), rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
def test_swish_matches_reference(beta):
    x = pt.vector("x")
    f = pytensor.function([x], Swish(beta=beta)(x))
    values = np.linspace(-6, 6, 101)
    reference = values / (1 + np.exp(-beta * values))
    np.testing.assert_allclose(f(values), reference, rtol=1e-6, atol=1e-8)


def test_swish_beta_one_is_silu():
    x = pt.vector("x")
    f = pytensor.function([x], Swish()(x))
    values = np.linspace(-6, 6, 101)
    np.testing.assert_allclose(
        f(values), values * (1 / (1 + np.exp(-values))), rtol=1e-6, atol=1e-8
    )


def test_gelu_tanh_approximates_exact():
    x = pt.vector("x")
    exact = pytensor.function([x], GELU(approximate=False)(x))
    approx = pytensor.function([x], GELU(approximate=True)(x))
    values = np.linspace(-6, 6, 101)
    np.testing.assert_allclose(approx(values), exact(values), atol=1e-3)
