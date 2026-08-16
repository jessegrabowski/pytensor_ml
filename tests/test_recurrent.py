import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor_ml.activations import Activation, ReLU, Tanh
from pytensor_ml.layers import RNN, Input, Linear
from pytensor_ml.loss import SquaredError
from pytensor_ml.model import Model
from pytensor_ml.optim import adam
from pytensor_ml.params import trainable
from pytensor_ml.pytensorf import collect_trainable_params
from pytensor_ml.state import OneInitializer, ZeroInitializer

floatX = pytensor.config.floatX

# The reference loop below sums in a different order than the graph, so the gap tracks the precision.
ATOL = 1e-6 if floatX == "float64" else 1e-5


@pytest.fixture
def rng():
    return np.random.default_rng(sum(map(ord, "pytensor_ml recurrent")))


def unrolled(X_np, W_ih, b, W_hh, phi, h0=None):
    """The recurrence written as a python loop, one step at a time, as the reference to check against."""
    h = np.zeros((*X_np.shape[:-2], W_hh.shape[0]), dtype=floatX) if h0 is None else h0
    states = []
    for t in range(X_np.shape[-2]):
        h = phi(X_np[..., t, :] @ W_ih + b + h @ W_hh)
        states.append(h)
    return np.stack(states, axis=-2)


def draw_parameters(layer, rng):
    """Set every parameter to a fresh draw and hand the values back for the reference to use."""
    W_ih = rng.normal(size=(layer.n_in, layer.n_hidden)).astype(floatX)
    b = rng.normal(size=(layer.n_hidden,)).astype(floatX)
    W_hh = rng.normal(size=(layer.n_hidden, layer.n_hidden)).astype(floatX)
    layer.W_ih.set_value(W_ih)
    layer.b.set_value(b)
    layer.W_hh.set_value(W_hh)
    return W_ih, b, W_hh


@pytest.mark.parametrize(
    "activation, reference",
    [(Tanh(), np.tanh), (ReLU(), lambda x: np.maximum(x, 0.0))],
    ids=["tanh", "relu"],
)
def test_matches_a_step_by_step_reference(activation, reference, rng):
    X = pt.tensor("X", shape=(None, None, 4))
    layer = RNN("rnn", n_in=4, n_hidden=3, activation=activation)
    out = layer(X)
    assert out.type.shape == (None, None, 3)

    W_ih, b, W_hh = draw_parameters(layer, rng)
    X_np = rng.normal(size=(5, 7, 4)).astype(floatX)

    np.testing.assert_allclose(
        out.eval({X: X_np}), unrolled(X_np, W_ih, b, W_hh, reference), atol=ATOL
    )


def test_starts_from_a_given_state(rng):
    """The state a caller hands in has to reach the first step, not just sit in the graph -- a zeros
    default that quietly ignored it would agree with the reference on every other test here."""
    X = pt.tensor("X", shape=(None, None, 4))
    h0 = pt.tensor("h0", shape=(None, 3))
    layer = RNN("rnn", n_in=4, n_hidden=3)
    out = layer(X, h0)

    W_ih, b, W_hh = draw_parameters(layer, rng)
    X_np = rng.normal(size=(5, 7, 4)).astype(floatX)
    h0_np = rng.normal(size=(5, 3)).astype(floatX)

    np.testing.assert_allclose(
        out.eval({X: X_np, h0: h0_np}),
        unrolled(X_np, W_ih, b, W_hh, np.tanh, h0=h0_np),
        atol=ATOL,
    )


def test_the_recurrent_weight_first_acts_on_the_second_step(rng):
    """The recurrence is real and runs forward in time. Starting from a zero state, the first output does
    not touch the recurrent weight at all and the second does, which is what separates a scan from an
    input projection applied position by position."""
    X = pt.tensor("X", shape=(None, None, 4))
    layer = RNN("rnn", n_in=4, n_hidden=3)
    out = layer(X)
    draw_parameters(layer, rng)
    X_np = rng.normal(size=(5, 7, 4)).astype(floatX)

    first, second = (
        pytensor.grad(out[..., step, :].sum(), layer.W_hh).eval({X: X_np}) for step in (0, 1)
    )

    np.testing.assert_allclose(first, np.zeros((3, 3)), atol=ATOL)
    assert np.abs(second).max() > 1e-3


def test_every_parameter_is_reachable_through_the_scan(rng):
    """The recurrent weight enters the graph as a non-sequence of the scan rather than as a plain input, so
    a collector that stopped at the scan node would train the projection and leave the recurrence frozen."""
    X = pt.tensor("X", shape=(None, None, 4))
    layer = RNN("rnn", n_in=4, n_hidden=3)
    out = layer(X)

    assert set(collect_trainable_params(out)) == {
        layer.W_ih,
        layer.b,
        layer.W_hh,
    }


def test_an_activation_brings_its_own_parameters_into_the_recurrence():
    """The step closes over whatever the activation holds, and scan lifts it in. A strict scan would reject
    a parameterized activation instead, telling the caller to add it to an input list they do not have."""

    class PReLU(Activation):
        def __init__(self):
            self.slope = trainable(
                np.asarray(0.25, dtype=floatX), "prelu_slope", initializer=OneInitializer()
            )

        def __call__(self, x):
            return pt.switch(x > 0, x, self.slope * x)

    activation = PReLU()
    layer = RNN("rnn", n_in=4, n_hidden=3, activation=activation)
    out = layer(pt.tensor("X", shape=(None, None, 4)))

    assert activation.slope in collect_trainable_params(out)


def test_trains_end_to_end(rng):
    """Gradients survive the round trip through the scan and the training machinery moves the parameters."""
    X = Input("X", shape=(None, 6, 4))
    y = Linear("head", 5, 1)(RNN("rnn", n_in=4, n_hidden=5)(X)[..., -1, :])
    model = Model(X, y).initialize(seed=1)
    step = model.compile_train(adam(learning_rate=0.05), SquaredError(), ndim_out=2)

    X_np = rng.normal(size=(32, 6, 4)).astype(floatX)
    y_np = X_np.sum(axis=(1, 2))[:, None].astype(floatX)

    losses = [float(step(X_np, y_np)) for _ in range(50)]
    assert losses[-1] < losses[0] / 5


def test_the_recurrent_weight_is_drawn_orthogonal_by_default():
    """Applied once per step, so its singular values compound: at one they leave the state alone however
    long the sequence, and spread around one they explode the gradient along some directions while
    vanishing it along others. The input weight keeps the usual fan-scaled draw, checked structurally as
    well as by spread -- on a square matrix the two draws have the same entry standard deviation, so
    spread alone would not notice it picking up the recurrent default."""
    layer = RNN("rnn", n_in=16, n_hidden=64)

    W_hh = layer.W_hh.get_value()
    np.testing.assert_allclose(W_hh.T @ W_hh, np.eye(64), atol=ATOL)

    W_ih = layer.W_ih.get_value()
    assert np.abs(W_ih @ W_ih.T - np.eye(16)).max() > 0.1
    assert W_ih.std() == pytest.approx(np.sqrt(2.0 / 80), rel=0.1)


@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no_bias"])
def test_the_bias_is_optional(bias, rng):
    """Dropping the bias has to drop the parameter as well as the term. Leaving an unused one behind would
    hand the optimizer moment state to carry for a weight that never moves, and nothing else here builds
    the layer without it."""
    X = pt.tensor("X", shape=(None, None, 4))
    layer = RNN("rnn", n_in=4, n_hidden=3, bias=bias)
    out = layer(X)

    W_ih = rng.normal(size=(4, 3)).astype(floatX)
    W_hh = rng.normal(size=(3, 3)).astype(floatX)
    b = rng.normal(size=(3,)).astype(floatX) if bias else np.zeros(3, dtype=floatX)
    layer.W_ih.set_value(W_ih)
    layer.W_hh.set_value(W_hh)
    if bias:
        layer.b.set_value(b)
    X_np = rng.normal(size=(5, 7, 4)).astype(floatX)

    assert set(collect_trainable_params(out)) == (
        {layer.W_ih, layer.W_hh, layer.b} if bias else {layer.W_ih, layer.W_hh}
    )
    np.testing.assert_allclose(
        out.eval({X: X_np}), unrolled(X_np, W_ih, b, W_hh, np.tanh), atol=ATOL
    )


def test_the_recurrent_weight_takes_its_own_initializer():
    """The recurrent draw has a keyword of its own, and using it must not disturb the input projection,
    which shares the layer's other two."""
    layer = RNN("rnn", n_in=4, n_hidden=3, recurrent_initializer=ZeroInitializer())

    np.testing.assert_array_equal(layer.W_hh.get_value(), np.zeros((3, 3)))
    assert np.abs(layer.W_ih.get_value()).max() > 0.0


@pytest.mark.parametrize(
    "batch_shape", [(), (5,), (2, 5)], ids=["unbatched", "one_axis", "two_axes"]
)
def test_recurs_over_any_number_of_batch_axes(batch_shape, rng):
    """Time is the second-to-last axis, as it is for every other layer here. Taking the batch axis to be
    the leading one instead would give the right answer for a single batch axis and quietly transpose a
    stacked one -- and refuse a bare sequence, which needs no batch axis at all."""
    X = pt.tensor("X", shape=(*(None for _ in batch_shape), None, 4))
    layer = RNN("rnn", n_in=4, n_hidden=3)
    out = layer(X)

    W_ih, b, W_hh = draw_parameters(layer, rng)
    X_np = rng.normal(size=(*batch_shape, 7, 4)).astype(floatX)

    result = out.eval({X: X_np})
    assert result.shape == (*batch_shape, 7, 3)
    np.testing.assert_allclose(result, unrolled(X_np, W_ih, b, W_hh, np.tanh), atol=ATOL)


def test_the_state_takes_the_dtype_the_step_produces():
    """A float32 network fed a float64 sequence. The step promotes, so the state has to promote with it;
    a state pinned to floatX leaves scan comparing float32 against the float64 its inner function returns
    and refusing the graph. Nothing else here catches it, because every other test runs at one dtype."""
    with pytensor.config.change_flags(floatX="float32"):
        layer = RNN("rnn", n_in=4, n_hidden=3)
        X = pt.tensor("X", shape=(None, None, 4), dtype="float64")
        out = layer(X)

        assert out.dtype == "float64"
        assert out.eval({X: np.zeros((2, 5, 4), dtype="float64")}).dtype == np.dtype("float64")


def test_rejects_an_input_with_no_time_axis():
    layer = RNN("rnn", n_in=4, n_hidden=3)

    with pytest.raises(ValueError, match="no time axis to recur over"):
        layer(pt.tensor("X", shape=(4,)))


def test_rejects_an_initial_state_that_does_not_match_the_batch_axes():
    """The state carries one value per batch element, so its rank is fixed by the input's. Scan would
    otherwise broadcast a mismatched state into the recurrence and return a silently wrong shape."""
    layer = RNN("rnn", n_in=4, n_hidden=3)

    with pytest.raises(ValueError, match="needs a 2-dimensional state; got a 1-dimensional one"):
        layer(pt.tensor("X", shape=(None, None, 4)), pt.tensor("h0", shape=(3,)))
