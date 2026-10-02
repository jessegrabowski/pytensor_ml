import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor_ml.layers import (
    ConstantPad1D,
    ReflectionPad1D,
    ReflectionPad2D,
    ReplicationPad1D,
    ReplicationPad2D,
    ZeroPad2D,
)

floatX = pytensor.config.floatX


@pytest.fixture
def rng():
    return np.random.default_rng(sum(map(ord, "padding")))


@pytest.mark.parametrize(
    "layer, pad_width, numpy_mode, numpy_kwargs",
    [
        (
            ZeroPad2D(padding=((1, 2), (3, 4))),
            ((1, 2), (3, 4)),
            "constant",
            {"constant_values": 0.0},
        ),
        (ReflectionPad2D(padding=((1, 2), (2, 1))), ((1, 2), (2, 1)), "reflect", {}),
        (ReplicationPad2D(padding=((2, 1), (1, 2))), ((2, 1), (1, 2)), "edge", {}),
    ],
    ids=["zero", "reflection", "replication"],
)
def test_each_mode_pads_the_spatial_axes_like_numpy(
    layer, pad_width, numpy_mode, numpy_kwargs, rng
):
    """numpy is the independent implementation here -- comparing against `pt.pad` would only restate
    what the layer already calls. The widths are spelled out rather than read back off the layer, so a
    resolver that swapped the axes or reversed an end would move both sides together and pass. Every
    case is asymmetric for the same reason."""
    X_np = rng.normal(size=(2, 6, 7, 3)).astype(floatX)
    X = pt.tensor("X", shape=(None, 6, 7, 3), dtype=floatX)

    expected = np.pad(X_np, [(0, 0), *pad_width, (0, 0)], mode=numpy_mode, **numpy_kwargs)
    np.testing.assert_allclose(pytensor.function([X], layer(X))(X_np), expected)


@pytest.mark.parametrize(
    "layer, numpy_mode, numpy_kwargs",
    [
        (ConstantPad1D(padding=(2, 4), value=3.5), "constant", {"constant_values": 3.5}),
        (ReplicationPad1D(padding=(2, 4)), "edge", {}),
    ],
    ids=["constant", "replication"],
)
def test_one_spatial_axis_reads_a_bare_pair_as_its_two_ends(layer, numpy_mode, numpy_kwargs, rng):
    """Over a single axis `(2, 4)` can only mean its two ends, and that is how torch reads it too --
    the per-axis reading would be a length mismatch."""
    X_np = rng.normal(size=(2, 9, 3)).astype(floatX)
    X = pt.tensor("X", shape=(None, 9, 3), dtype=floatX)

    expected = np.pad(X_np, [(0, 0), (2, 4), (0, 0)], mode=numpy_mode, **numpy_kwargs)
    np.testing.assert_allclose(pytensor.function([X], layer(X))(X_np), expected)


def test_padding_wider_than_the_axis_keeps_reflecting(rng):
    """Torch refuses a reflection wider than what it has to mirror; numpy folds back and forth, and so
    do we. Pinned because it is a deliberate difference from the framework these layers are named
    after."""
    X_np = np.arange(4, dtype=floatX).reshape(1, 4, 1)
    X = pt.tensor("X", shape=(1, 4, 1), dtype=floatX)

    padded = pytensor.function([X], ReflectionPad1D(padding=5)(X))(X_np)
    np.testing.assert_allclose(padded, np.pad(X_np, [(0, 0), (5, 5), (0, 0)], mode="reflect"))


def test_an_input_of_the_wrong_rank_is_rejected():
    """The layers are channels-last over a fixed number of spatial axes, so a channels-first image or
    a batch of vectors is a mistake worth naming rather than padding the wrong axes."""
    with pytest.raises(ValueError, match="4-dimensional input"):
        ZeroPad2D(padding=1)(pt.tensor("X", shape=(2, 8, 3), dtype=floatX))


@pytest.mark.parametrize(
    "padding, message",
    [
        (-1, "cannot be negative"),
        (((1, 2), (3, -4)), "cannot be negative"),
        ((1, 2, 3), "one amount per axis"),
        (((1, 2, 3), (1, 1)), r"is a \(before, after\) pair"),
    ],
    ids=["negative_scalar", "negative_in_pair", "wrong_count", "triple_for_one_axis"],
)
def test_padding_amounts_are_validated(padding, message):
    """Each of these describes something the layer cannot do, and each would otherwise surface as a
    confusing shape further downstream."""
    with pytest.raises(ValueError, match=message):
        ZeroPad2D(padding=padding)


def test_one_amount_per_axis_pads_both_of_its_sides():
    """Over two axes a flat pair is one amount per axis, so `(1, 2)` puts 1 on the top and bottom and
    2 on the left and right. The 1-D layers read the same pair as one axis's two ends, and the two
    readings must not leak into each other."""
    assert ZeroPad2D(padding=(1, 2)).padding == ((1, 1), (2, 2))
