import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor_ml.layers import Upsample1D, Upsample2D

floatX = pytensor.config.floatX
ATOL = 1e-6 if floatX == "float64" else 1e-4


@pytest.fixture
def rng():
    return np.random.default_rng(sum(map(ord, "pytensor_ml upsample")))


def test_nearest_repeats_each_element(rng):
    X = pt.tensor("X", shape=(None, 3, 4, 2))
    out = Upsample2D(scale_factor=(2, 3))(X)

    X_np = rng.normal(size=(2, 3, 4, 2)).astype(floatX)
    expected = np.repeat(np.repeat(X_np, 2, axis=1), 3, axis=2)

    np.testing.assert_allclose(out.eval({X: X_np}), expected, rtol=1e-6, atol=ATOL)


def test_nearest_indexes_by_floor_of_the_ratio():
    """Stretching 5 to 7 does not repeat evenly, and which elements get the extra copy is the whole
    convention: output j reads input floor(j * 5 / 7)."""
    X = pt.tensor("X", shape=(None, 5, 1))
    out = Upsample1D(size=7)(X)

    X_np = np.arange(5, dtype=floatX).reshape(1, 5, 1)

    np.testing.assert_allclose(out.eval({X: X_np}).ravel(), [0, 0, 1, 2, 2, 3, 4])


HALF_PIXEL_INPUT = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=floatX).reshape(1, 2, 2, 1)
HALF_PIXEL_DOUBLED = [
    [1.00, 1.25, 1.75, 2.00],
    [1.50, 1.75, 2.25, 2.50],
    [2.50, 2.75, 3.25, 3.50],
    [3.00, 3.25, 3.75, 4.00],
]


def test_bilinear_half_pixel_convention():
    """With align_corners=False each element covers a unit interval and output centers map to input
    centers, so a 2x2 doubled leaves the corners unchanged and the outermost samples land outside
    the input and clamp. This is torch's default and the one every diffusion decoder uses."""
    X = pt.tensor("X", shape=(1, 2, 2, 1))
    out = Upsample2D(scale_factor=2, mode="bilinear")(X)

    np.testing.assert_allclose(
        out.eval({X: HALF_PIXEL_INPUT})[0, :, :, 0], HALF_PIXEL_DOUBLED, rtol=1e-6, atol=ATOL
    )


def test_bilinear_corner_aligned_convention():
    """With align_corners=True the outermost outputs are pinned to the outermost inputs and the rest
    are spread evenly between them, which puts the samples a half pixel away from where the default
    convention puts them."""
    X = pt.tensor("X", shape=(1, 2, 2, 1))
    out = Upsample2D(scale_factor=2, mode="bilinear", align_corners=True)(X)

    X_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=floatX).reshape(1, 2, 2, 1)
    third = 1.0 / 3.0
    expected = [
        [1.0, 1 + third, 1 + 2 * third, 2.0],
        [1 + 2 * third, 2.0, 2 + third, 2 + 2 * third],
        [2 + third, 2 + 2 * third, 3.0, 3 + third],
        [3.0, 3 + third, 3 + 2 * third, 4.0],
    ]

    np.testing.assert_allclose(out.eval({X: X_np})[0, :, :, 0], expected, rtol=1e-6, atol=ATOL)


def test_linear_interpolates_along_a_sequence():
    X = pt.tensor("X", shape=(1, 3, 1))
    out = Upsample1D(scale_factor=2, mode="linear")(X)

    X_np = np.array([1.0, 2.0, 3.0], dtype=floatX).reshape(1, 3, 1)
    expected = [1.0, 1.25, 1.75, 2.25, 2.75, 3.0]

    np.testing.assert_allclose(out.eval({X: X_np}).ravel(), expected, rtol=1e-6, atol=ATOL)


def half_pixel_interpolation(X, size):
    """Separable linear interpolation where output centers map to input centers. `np.interp` holds the
    end values beyond the outermost centers, which is the clamp the half-pixel convention needs."""
    out = X
    for axis, out_extent in enumerate(size, start=1):
        in_extent = out.shape[axis]
        source = (np.arange(out_extent) + 0.5) * in_extent / out_extent - 0.5
        out = np.apply_along_axis(
            lambda values: np.interp(source, np.arange(in_extent), values), axis, out
        )
    return out


def test_explicit_size_resamples_to_that_extent(rng):
    """A size that is not a uniform factor of the input, shrinking one axis and growing the other, so
    nothing else covers these extents. The axes are given height-first, which a swap turns into the
    wrong shape rather than wrong values."""
    X = pt.tensor("X", shape=(None, 3, 4, 2))
    out = Upsample2D(size=(2, 9), mode="bilinear")(X)
    X_np = rng.normal(size=(2, 3, 4, 2)).astype(floatX)

    assert out.type.shape == (None, 2, 9, 2)
    np.testing.assert_allclose(
        out.eval({X: X_np}), half_pixel_interpolation(X_np, (2, 9)), rtol=1e-6, atol=ATOL
    )


def test_a_single_element_output_is_well_defined():
    """Spreading over a closed interval divides by one less than the output extent, which is zero
    when the output is a single element."""
    X = pt.tensor("X", shape=(1, 3, 4, 1))
    out = Upsample2D(size=1, mode="bilinear", align_corners=True)(X)

    X_np = np.arange(1, 13, dtype=floatX).reshape(1, 3, 4, 1)

    np.testing.assert_allclose(out.eval({X: X_np}).ravel(), [1.0], rtol=1e-6, atol=ATOL)


def test_a_known_extent_survives_the_gather():
    """The resampled extent is an ordinary index gather, which reports a static extent only when the
    index vector has one. A downstream pool checks its window against these."""
    X = pt.tensor("X", shape=(None, 3, 4, 2))

    assert Upsample2D(scale_factor=2)(X).type.shape == (None, 6, 8, 2)


def test_an_unknown_extent_still_resamples():
    """The extents go symbolic when the input's are unknown, which must change the graph and not the
    answer. A square input cannot tell the two spatial axes apart, so a non-square one is fed to the
    same function to pin which extent each axis doubles."""
    X = pt.tensor("X", shape=(None, None, None, None))
    out = Upsample2D(scale_factor=2, mode="bilinear")(X)
    resample = pytensor.function([X], out)

    assert out.type.shape == (None, None, None, None)
    np.testing.assert_allclose(
        resample(HALF_PIXEL_INPUT)[0, :, :, 0], HALF_PIXEL_DOUBLED, rtol=1e-6, atol=ATOL
    )
    assert resample(np.zeros((1, 3, 4, 2), dtype=floatX)).shape == (1, 6, 8, 2)


@pytest.mark.parametrize(
    "floatx, dtype",
    [("float32", "float32"), ("float64", "float32"), ("float64", "float16")],
    ids=["at_floatx", "float32_under_float64", "float16_under_float64"],
)
def test_bilinear_keeps_the_input_dtype(floatx, dtype):
    """The interpolation weights are built from integer extents, and building them at floatX, or
    dividing the extents as integers, would carry a narrower input up to float64."""
    with pytensor.config.change_flags(floatX=floatx):
        X = pt.tensor("X", shape=(None, 3, 4, 2), dtype=dtype)

        assert Upsample2D(scale_factor=2, mode="bilinear")(X).dtype == dtype


# Every way of asking for a resampling that means nothing. `mode` is spelled for its rank, so the
# name that works on one class is wrong on the other rather than a harmless alias.
REJECTED_CONFIGURATIONS = [
    (Upsample2D, {}, "exactly one of scale_factor and size"),
    (Upsample2D, {"scale_factor": 2, "size": 4}, "exactly one of scale_factor and size"),
    (Upsample2D, {"scale_factor": 2, "mode": "linear"}, "interpolates either 'nearest'"),
    (Upsample1D, {"scale_factor": 2, "mode": "bilinear"}, "interpolates either 'nearest'"),
    (Upsample2D, {"scale_factor": 2, "align_corners": True}, "no corners to align"),
    (Upsample2D, {"scale_factor": 0}, "needs a positive"),
    (Upsample2D, {"size": (4, -1)}, "needs a positive"),
]


@pytest.mark.parametrize(
    "layer, kwargs, message",
    REJECTED_CONFIGURATIONS,
    ids=[
        "neither",
        "both",
        "2d_linear",
        "1d_bilinear",
        "corners_without_interpolation",
        "zero_factor",
        "negative_size",
    ],
)
def test_a_meaningless_configuration_raises(layer, kwargs, message):
    with pytest.raises(ValueError, match=message):
        layer(**kwargs)


def test_wrong_rank_raises():
    with pytest.raises(ValueError, match="4-dimensional"):
        Upsample2D(scale_factor=2)(pt.tensor("X", shape=(None, 4, 2)))


def test_a_factor_in_the_name_slot_raises():
    with pytest.raises(TypeError, match=r"Upsample2D\(scale_factor=2\)"):
        Upsample2D(2)
