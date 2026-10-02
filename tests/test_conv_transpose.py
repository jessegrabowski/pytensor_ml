import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor.gradient import verify_grad
from pytensor.graph import graph_replace

from pytensor_ml.layers import Conv2D, ConvTranspose1D, ConvTranspose2D

floatX = pytensor.config.floatX
ATOL = 1e-4 if floatX == "float32" else 1e-8


@pytest.fixture
def rng():
    return np.random.default_rng(sum(map(ord, "conv transpose")))


def scatter_through_kernel(X, W, stride, padding, output_padding, dilation):
    """
    Scatter each input position through the kernel, which is the definition of the operation.

    Written out as loops rather than built from the library's own ops, so the tests below compare the
    layer against the arithmetic it claims to do rather than against another spelling of itself.
    ``padding`` is one amount per axis because a transposed convolution always crops the same number
    of elements from both ends.
    """
    batch, *spatial, _ = X.shape
    kernel, out_channels = W.shape[:-2], W.shape[-1]
    uncropped = [
        (length - 1) * step + spacing * (extent - 1) + 1 + extra
        for length, step, spacing, extent, extra in zip(
            spatial, stride, dilation, kernel, output_padding
        )
    ]
    out = np.zeros((batch, *uncropped, out_channels), dtype=X.dtype)
    for source in np.ndindex(*spatial):
        for tap in np.ndindex(*kernel):
            target = tuple(i * s + t * d for i, s, t, d in zip(source, stride, tap, dilation))
            out[(slice(None), *target)] += X[(slice(None), *source)] @ W[tap]
    cropped = tuple(slice(before, size - before) for before, size in zip(padding, uncropped))
    return out[(slice(None), *cropped, slice(None))]


@pytest.mark.parametrize(
    "stride, padding, output_padding, dilation, bias",
    [
        (1, 0, 0, 1, False),
        (2, 1, 1, 1, True),
        ((2, 3), (1, 2), (1, 2), (2, 1), False),
    ],
    ids=["plain", "strided_cropped", "per_axis"],
)
def test_conv_transpose_2d_scatters_its_input_through_the_kernel(
    stride, padding, output_padding, dilation, bias, rng
):
    """Every argument changes where a contribution lands, so each is exercised against the scatter the
    layer stands for. The per-axis case is the one that catches an axis order swapped somewhere. The
    bias is added after the scatter, so a layer that folded it in before would add it once per
    contributing tap instead of once per output position."""
    X_np = rng.normal(size=(2, 5, 6, 3)).astype(floatX)
    X = pt.tensor("X", shape=(None, 5, 6, 3), dtype=floatX)
    layer = ConvTranspose2D(
        "transpose",
        in_channels=3,
        out_channels=4,
        kernel_size=3,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        bias=bias,
    )
    layer.W.set_value(rng.normal(size=layer.W.get_value().shape).astype(floatX))
    expected_bias = 0.0
    if bias:
        expected_bias = rng.normal(size=(4,)).astype(floatX)
        layer.b.set_value(expected_bias)

    expected = scatter_through_kernel(
        X_np,
        layer.W.get_value(),
        layer.stride,
        [before for before, _ in layer.padding],
        layer.output_padding,
        layer.dilation,
    )
    np.testing.assert_allclose(
        pytensor.function([X], layer(X))(X_np), expected + expected_bias, atol=ATOL
    )


def test_conv_transpose_1d_scatters_its_input_through_the_kernel(rng):
    """The rank-1 layer shares `_ConvTransposeNd`, so this checks the spatial count reaches every
    argument rather than repeating the rank-2 coverage."""
    X_np = rng.normal(size=(2, 7, 3)).astype(floatX)
    X = pt.tensor("X", shape=(None, 7, 3), dtype=floatX)
    layer = ConvTranspose1D(
        "transpose",
        in_channels=3,
        out_channels=5,
        kernel_size=4,
        stride=2,
        padding=1,
        output_padding=1,
        bias=False,
    )
    layer.W.set_value(rng.normal(size=layer.W.get_value().shape).astype(floatX))

    expected = scatter_through_kernel(
        X_np,
        layer.W.get_value(),
        layer.stride,
        [before for before, _ in layer.padding],
        layer.output_padding,
        layer.dilation,
    )
    np.testing.assert_allclose(pytensor.function([X], layer(X))(X_np), expected, atol=ATOL)


def test_conv_transpose_inverts_a_convolutions_shape():
    """The operation exists to undo what a convolution did to a shape, so a transpose built with the
    same geometry has to return the size the convolution consumed."""
    X = pt.tensor("X", shape=(None, 9, 9, 3), dtype=floatX)
    forward = Conv2D("down", in_channels=3, out_channels=4, kernel_size=3, stride=2)
    backward = ConvTranspose2D("up", in_channels=4, out_channels=3, kernel_size=3, stride=2)

    reduced = forward(X)
    assert reduced.type.shape[1:3] == (4, 4)
    assert backward(reduced).type.shape[1:3] == (9, 9)


@pytest.mark.parametrize("output_padding", [2, -1], ids=["equals_stride", "negative"])
def test_output_padding_must_stay_below_the_stride(output_padding):
    """A stride of `s` maps `s` input sizes onto one output size, so an `output_padding` of `s` names a
    size the stride never produced, and a negative one names nothing at all. Both are caught at
    construction rather than as a shape error later."""
    with pytest.raises(ValueError, match="less than that stride"):
        ConvTranspose2D(
            "t",
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            stride=2,
            output_padding=output_padding,
        )


def test_padding_does_not_take_the_forward_layers_keywords():
    """`padding="same"` reads as a request the layer cannot honour: here padding removes output rather
    than adding input, so there is no size for "same" to preserve."""
    with pytest.raises(ValueError, match="removes elements from its output"):
        ConvTranspose2D("t", in_channels=1, out_channels=1, kernel_size=3, padding="same")


def test_the_gradient_crosses_the_output_padding_and_the_crop(rng):
    """The pullback has to undo the leftover row `output_padding` adds and the rows the crop removes,
    and no forward comparison sees either of those through the gradient."""
    layer = ConvTranspose2D(
        "transpose",
        in_channels=2,
        out_channels=3,
        kernel_size=3,
        stride=2,
        padding=1,
        output_padding=1,
        bias=False,
    )
    X_np = rng.normal(size=(2, 4, 3, 2)).astype(floatX)
    W_np = rng.normal(size=layer.W.get_value().shape).astype(floatX)

    def transpose(X, W):
        return graph_replace(layer(X), {layer.W: W})

    verify_grad(transpose, [X_np, W_np], rng=np.random.default_rng(0))


def test_cropping_more_than_the_output_holds_is_rejected():
    """Padding here removes output rather than adding input, so enough of it leaves a zero-length axis.
    That reaches the op and comes back empty rather than failing, which is the shape of mistake worth
    catching where the input's length is known."""
    X = pt.tensor("X", shape=(2, 2, 2, 3), dtype=floatX)
    layer = ConvTranspose2D("t", in_channels=3, out_channels=4, kernel_size=3, padding=2)

    with pytest.raises(ValueError, match="leaving nothing"):
        layer(X)


def test_a_crop_that_only_a_short_input_cannot_afford_is_allowed():
    """The same crop that empties a length-1 input is right for a longer one, so the check has to read
    the input rather than the arguments alone -- rejecting this at construction would be wrong."""
    X = pt.tensor("X", shape=(2, 2, 2, 3), dtype=floatX)
    layer = ConvTranspose2D("t", in_channels=3, out_channels=4, kernel_size=3, stride=2, padding=2)

    assert layer(X).type.shape[1:3] == (1, 1)


def test_a_symbolic_spatial_extent_skips_the_crop_check(rng):
    """Only a statically known length can be checked, so a symbolic one is let through rather than
    guessed at -- and then has to compute the scatter a declared length would."""
    X = pt.tensor("X", shape=(None, None, None, 3), dtype=floatX)
    layer = ConvTranspose2D(
        "t", in_channels=3, out_channels=4, kernel_size=3, padding=2, bias=False
    )
    layer.W.set_value(rng.normal(size=layer.W.get_value().shape).astype(floatX))
    X_np = rng.normal(size=(2, 5, 5, 3)).astype(floatX)

    unchecked = pytensor.function([X], layer(X))(X_np)
    expected = scatter_through_kernel(
        X_np,
        layer.W.get_value(),
        layer.stride,
        [before for before, _ in layer.padding],
        layer.output_padding,
        layer.dilation,
    )
    assert unchecked.shape == (2, 3, 3, 4)
    np.testing.assert_allclose(unchecked, expected, atol=ATOL)
