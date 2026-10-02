import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor.compile.mode import Mode

from pytensor_ml.layers import AvgPool1D, AvgPool2D, MaxPool1D, MaxPool2D
from pytensor_ml.layers.conv import PoolLayer, PoolLayerGrad

floatX = pytensor.config.floatX
ATOL = 1e-6 if floatX == "float64" else 1e-4


@pytest.fixture
def rng():
    return np.random.default_rng(sum(map(ord, "pytensor_ml pool")))


@pytest.mark.parametrize(
    "reduction, expected",
    [("max", [[5.0, 7.0], [13.0, 15.0]]), ("mean", [[2.5, 4.5], [10.5, 12.5]])],
)
def test_pooling_reduces_the_windows_a_kernel_would_visit(reduction, expected):
    """The windows written out by hand. A 4x4 counting up, tiled by 2x2 windows, has an answer you can
    read off the grid, which is what catches a stride mistaken for a kernel extent."""
    X = pt.tensor("X", shape=(1, 4, 4, 1))
    X_np = np.arange(16, dtype=floatX).reshape(1, 4, 4, 1)

    pooled = PoolLayer((2, 2), (2, 2), (1, 1), reduction)(X)

    np.testing.assert_allclose(pooled.eval({X: X_np})[0, :, :, 0], expected, atol=ATOL)


def test_pooling_rejects_a_reduction_it_does_not_have():
    """The reduction is a prop rather than a subclass, so a typo would otherwise build an op that fails
    only when its inner graph is traced."""
    with pytest.raises(ValueError, match="reduction must be one of"):
        PoolLayer((2,), (2,), (1,), "median")


def test_max_pooling_pads_with_negative_infinity():
    """Padding a max pool with zeros lets a padded position win any window whose real activations are
    all negative, which returns a plausible 0 instead of the true maximum."""
    X = pt.tensor("X", shape=(1, 3, 1))
    X_np = -np.arange(1, 4, dtype=floatX).reshape(1, 3, 1)

    # `same` over three elements with a width-2 window pads one element after, so the final window
    # covers the last activation and the pad.
    pooled = MaxPool1D("pool", kernel_size=2, padding="same")(X)

    np.testing.assert_allclose(pooled.eval({X: X_np})[0, :, 0], [-1.0, -3.0], atol=ATOL)


def test_average_pooling_counts_padding_as_zero():
    """An average pools over the window it was given, padding included, which is torch's
    `count_include_pad` default and the reason the two reductions cannot share a fill value."""
    X = pt.tensor("X", shape=(1, 3, 1))
    X_np = -np.arange(1, 4, dtype=floatX).reshape(1, 3, 1)

    pooled = AvgPool1D("pool", kernel_size=2, padding="same")(X)

    np.testing.assert_allclose(pooled.eval({X: X_np})[0, :, 0], [-1.5, -1.5], atol=ATOL)


def test_pooling_steps_by_the_kernel_unless_told_otherwise():
    """Pooling tiles by default where convolution slides, so the default stride is the kernel extent.
    Anyone reading the two signatures together will take that for a mistake, so it is pinned here."""
    X = pt.tensor("X", shape=(2, 8, 8, 3))

    assert MaxPool2D("tiled", kernel_size=2)(X).type.shape == (2, 4, 4, 3)
    assert MaxPool2D("slid", kernel_size=2, stride=1)(X).type.shape == (2, 7, 7, 3)


def test_max_pooling_routes_its_gradient_to_the_window_maximum():
    """A max pool's pullback sends the whole cotangent to the position that won and nothing anywhere
    else, so the gradient of a summed pool is an indicator of the per-window argmax. Max pooling
    usually follows a rectifier, so a window clamped entirely to zero is routine rather than a
    measure-zero curiosity. Crediting every tied tap would return more gradient than the window
    received, so the earliest one takes it, as jax, torch and mlx all do."""
    X = pt.tensor("X", shape=(1, 6, 1))
    X_np = np.array([3.0, 1.0, 2.0, 5.0, 1.0, 1.0], dtype=floatX).reshape(1, 6, 1)

    gradient = pt.grad(MaxPool1D("pool", kernel_size=2)(X).sum(), X)

    np.testing.assert_allclose(
        gradient.eval({X: X_np})[0, :, 0], [1.0, 0.0, 0.0, 1.0, 1.0, 0.0], atol=ATOL
    )


def test_average_pooling_spreads_its_gradient_over_the_window(rng):
    """The counterpart to the max routing: every position in a window contributed equally, so each
    receives the same share rather than one taking all of it. The windows overlap, so a position
    collects one share from every window that covers it, and a pullback that wrote its shares
    instead of adding them would keep only the last."""
    X = pt.tensor("X", shape=(1, 6, 1))
    X_np = rng.normal(size=(1, 6, 1)).astype(floatX)

    gradient = pt.grad(AvgPool1D("pool", kernel_size=3, stride=1)(X).sum(), X)

    np.testing.assert_allclose(
        gradient.eval({X: X_np})[0, :, 0], np.array([1, 2, 3, 3, 2, 1]) / 3, atol=ATOL
    )


def test_dilated_pooling_reads_every_other_element():
    """A dilation of 2 spaces a window's taps two apart, so with a stride of 1 the windows are
    `(0, 2), (1, 3), (2, 4), (3, 5)`. The stride and dilation differ, so a kernel that swapped them
    reads other windows, and one that ignored the dilation reads adjacent pairs."""
    X = pt.tensor("X", shape=(1, 6, 1))
    X_np = np.array([3.0, 0.0, 1.0, 4.0, 2.0, 5.0], dtype=floatX).reshape(1, 6, 1)

    pooled = MaxPool1D("pool", kernel_size=2, stride=1, dilation=2)(X)
    values, gradient = pytensor.function([X], [pooled, pt.grad(pooled.sum(), X)])(X_np)

    np.testing.assert_allclose(values[0, :, 0], [3.0, 4.0, 2.0, 5.0], atol=ATOL)
    np.testing.assert_allclose(gradient[0, :, 0], [1.0, 0.0, 0.0, 1.0, 1.0, 1.0], atol=ATOL)


@pytest.mark.parametrize(
    "layers", [(MaxPool1D, MaxPool2D), (AvgPool1D, AvgPool2D)], ids=["max", "avg"]
)
def test_pooling_over_one_axis_matches_a_degenerate_second(layers, rng):
    """Both ranks are `_PoolNd` with a different `n_spatial`, and pooling a width-1 image with a
    width-1 window is the 1-D case, so the two have to agree rather than merely match in shape."""
    one_d, two_d = layers
    X_np = rng.normal(size=(2, 12, 3)).astype(floatX)
    X1 = pt.tensor("X1", shape=(None, None, 3))
    X2 = pt.tensor("X2", shape=(None, None, None, 3))

    np.testing.assert_allclose(
        one_d("a", kernel_size=3, stride=2)(X1).eval({X1: X_np}),
        two_d("b", kernel_size=(3, 1), stride=(2, 1))(X2).eval({X2: X_np[:, :, None]})[:, :, 0],
        atol=ATOL,
    )


def test_pooling_rejects_an_input_too_small_for_one_window():
    """Shared with the convolution layers: a window that does not fit yields no windows at all, and an
    empty answer is worse than an error."""
    X = pt.tensor("X", shape=(2, 3, 3, 1))

    with pytest.raises(ValueError, match="needs at least 4 elements along spatial axis 0"):
        MaxPool2D("pool", kernel_size=4)(X)


# No linker runs the pooling ops' inner graph once every backend dispatches them, so it is reached
# deliberately here. It is the portable definition of what pooling means, and a backend's kernel is
# only correct if it agrees with it.
FALLBACK = Mode(linker="py", optimizer="fast_compile")


@pytest.mark.parametrize("layer_cls", [MaxPool2D, AvgPool2D], ids=["max", "avg"])
def test_the_portable_graph_agrees_with_the_kernel_that_replaces_it(layer_cls, rng):
    """Every backend dispatches past the inner graph, so nothing else exercises it -- and it is what
    runs wherever a kernel is missing. A kernel and a fallback that disagree is a result that changes
    with the backend. Every geometry argument differs per axis, and the input is drawn from three
    values so that windows tie: `pt.max` would hand the cotangent to every tied tap, and the fallback
    is where that would go unnoticed."""
    X_np = rng.integers(0, 3, size=(2, 7, 6, 3)).astype(floatX)
    X = pt.tensor("X", shape=X_np.shape)
    out = layer_cls("pool", kernel_size=(3, 2), stride=(2, 1), dilation=(1, 2), padding="same")(X)
    outputs = [out, pt.grad(out.sum(), X)]

    # The py linker still builds C thunks unless told not to, and those compiles dominate the cost.
    with pytensor.config.change_flags(cxx=""):
        fallback = pytensor.function([X], outputs, mode=FALLBACK)(X_np)
    for portable, kernel in zip(fallback, pytensor.function([X], outputs)(X_np)):
        np.testing.assert_allclose(portable, kernel, atol=ATOL)


def test_the_pooling_gradient_op_rejects_a_reduction_it_does_not_have():
    """`PoolLayerGrad` is built by `PoolLayer.pullback` today, but it is a public op, and a bad
    reduction would otherwise surface as a KeyError from a lookup inside its inner graph."""
    with pytest.raises(ValueError, match="reduction must be one of"):
        PoolLayerGrad((2,), (2,), (1,), "median")
