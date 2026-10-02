import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor_ml.activations import Activation, ReLU
from pytensor_ml.layers import (
    GRU,
    LSTM,
    RNN,
    Bidirectional,
    Recurrent,
    RecurrentCell,
)
from pytensor_ml.params import trainable
from pytensor_ml.pytensorf import collect_trainable_params
from pytensor_ml.state import OneInitializer
from tests.conftest import constant

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
    W_ih = rng.normal(size=(layer.cell.n_in, layer.cell.n_hidden)).astype(floatX)
    b = rng.normal(size=(layer.cell.n_hidden,)).astype(floatX)
    W_hh = rng.normal(size=(layer.cell.n_hidden, layer.cell.n_hidden)).astype(floatX)
    layer.cell.W_ih.set_value(W_ih)
    layer.cell.b.set_value(b)
    layer.cell.W_hh.set_value(W_hh)
    return W_ih, b, W_hh


def test_matches_a_step_by_step_reference(rng):
    X = pt.tensor("X", shape=(None, None, 4))
    layer = RNN("rnn", n_in=4, n_hidden=3)
    out = layer(X)
    assert out.type.shape == (None, None, 3)

    W_ih, b, W_hh = draw_parameters(layer, rng)
    X_np = rng.normal(size=(5, 7, 4)).astype(floatX)

    np.testing.assert_allclose(
        out.eval({X: X_np}), unrolled(X_np, W_ih, b, W_hh, np.tanh), atol=ATOL
    )


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


@pytest.mark.parametrize(
    "layer_type, n_gates", [(RNN, 1), (GRU, 3), (LSTM, 4)], ids=["rnn", "gru", "lstm"]
)
def test_the_recurrent_weight_is_drawn_orthogonal_by_default(layer_type, n_gates):
    """Applied once per step, so its singular values compound: at one they leave the state alone however
    long the sequence, and spread around one they explode the gradient along some directions while
    vanishing it along others. One draw covers every gate, as in keras, so the check is on the whole
    wide matrix rather than on each gate's block. The input weight keeps the usual fan-scaled draw,
    checked structurally as well as by spread -- on a square matrix the two draws have the same entry
    standard deviation, so spread alone would not notice it picking up the recurrent default."""
    layer = layer_type("layer", n_in=16, n_hidden=32)

    W_hh = layer.cell.W_hh.get_value()
    assert W_hh.shape == (32, n_gates * 32)
    np.testing.assert_allclose(W_hh @ W_hh.T, np.eye(32), atol=ATOL)

    W_ih = layer.cell.W_ih.get_value()
    assert np.abs(W_ih @ W_ih.T - np.eye(16)).max() > 0.1
    # Both fans count the stacked axis, so the spread would be wrong if the draw saw one gate's shape.
    assert W_ih.std() == pytest.approx(np.sqrt(2.0 / (16 + n_gates * 32)), rel=0.1)


def test_the_bias_is_optional(rng):
    """Dropping the bias has to drop the parameter as well as the term. Leaving an unused one behind would
    hand the optimizer moment state to carry for a weight that never moves, and nothing else here builds
    the layer without it."""
    X = pt.tensor("X", shape=(None, None, 4))
    layer = RNN("rnn", n_in=4, n_hidden=3, bias=False)
    out = layer(X)

    W_ih = rng.normal(size=(4, 3)).astype(floatX)
    W_hh = rng.normal(size=(3, 3)).astype(floatX)
    layer.cell.W_ih.set_value(W_ih)
    layer.cell.W_hh.set_value(W_hh)
    X_np = rng.normal(size=(5, 7, 4)).astype(floatX)

    assert set(collect_trainable_params(out)) == {layer.cell.W_ih, layer.cell.W_hh}
    np.testing.assert_allclose(
        out.eval({X: X_np}),
        unrolled(X_np, W_ih, np.zeros(3, dtype=floatX), W_hh, np.tanh),
        atol=ATOL,
    )


@pytest.mark.parametrize(
    "layer_type, biases",
    [(RNN, ["b"]), (GRU, ["b", "c"]), (LSTM, ["b"])],
    ids=["rnn", "gru", "lstm"],
)
def test_every_recurrent_layer_forwards_its_initializers_to_the_cell(layer_type, biases):
    """Three keyword-only arguments reach the cell through the flat constructor, and one dropped on the
    floor, or handed to the wrong parameter, leaves a parameter silently at some other draw. A distinct
    constant per keyword says which one reached which parameter. A GRU's two biases share the one
    keyword."""
    layer = layer_type(
        "layer",
        n_in=4,
        n_hidden=3,
        weight_initializer=constant(value=1.0),
        recurrent_initializer=constant(value=2.0),
        bias_initializer=constant(value=3.0),
    )
    cell = layer.cell

    assert np.all(cell.W_ih.get_value() == 1.0)
    assert np.all(cell.W_hh.get_value() == 2.0)
    for bias in biases:
        assert np.all(getattr(cell, bias).get_value() == 3.0)


@pytest.mark.parametrize("batch_shape", [(), (2, 5)], ids=["unbatched", "two_axes"])
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


@pytest.mark.parametrize("layer_type", [RNN, GRU, LSTM], ids=["rnn", "gru", "lstm"])
@pytest.mark.parametrize(
    "parameter_dtype, input_dtype",
    [("float32", "float64"), ("float64", "float32")],
    ids=["wider_input", "wider_parameters"],
)
def test_the_state_takes_the_dtype_the_step_produces(layer_type, parameter_dtype, input_dtype):
    """The step promotes the input against the parameters, so the zero state has to promote the same
    way. A state pinned to floatX fails the wider input, and one built from the input's dtype alone
    fails the wider parameters: either leaves scan comparing a float32 state against the float64 its
    inner function returns, and it refuses the graph while building it. Every other test here runs at
    one dtype."""
    with pytensor.config.change_flags(floatX=parameter_dtype):
        layer = layer_type("layer", n_in=4, n_hidden=3)
        out = layer(pt.tensor("X", shape=(None, None, 4), dtype=input_dtype))

    assert out.dtype == "float64"


class TwoStateCell(RecurrentCell):
    """A cell carrying more than one tensor, as an LSTM does, for checking how a starting state is
    matched against several."""

    def __init__(self, n_hidden):
        self.n_hidden = n_hidden

    def step(self, x_t, running_sum, count):
        return running_sum + x_t, count + 1.0

    def initial_state(self, X):
        zeros = pt.zeros((*X.shape[:-2], self.n_hidden), dtype=X.dtype)
        return zeros, pt.zeros_like(zeros)


def test_a_rejected_state_names_which_one_of_several_is_wrong():
    """The message carries a position because a cell may carry many states, and a caller staring at two
    identically shaped arguments needs to know which one is wrong. A hardcoded index would read as 0 here."""
    X = pt.tensor("X", shape=(None, None, 3))
    good, bad = pt.matrix("good"), pt.vector("bad")

    with pytest.raises(
        ValueError, match="needs a 2-dimensional state at position 1; got a 1-dimensional one"
    ):
        Recurrent(TwoStateCell(3), name="two_state")(X, [good, bad])


def test_rejects_a_starting_state_the_cell_does_not_carry():
    """A cell's state count is part of its contract, and scan would otherwise report the mismatch from
    inside the inner function, where the message names nothing the caller wrote."""
    X = pt.tensor("X", shape=(None, None, 3))

    with pytest.raises(ValueError, match="carries 2 state tensor\\(s\\), but got 1"):
        Recurrent(TwoStateCell(3), name="two_state")(X, pt.matrix("only_one"))


def test_the_rnn_names_its_cell_parameters_after_the_layer():
    """The flat constructor hands its name to the cell it builds, so the parameters it owns are named
    for the layer a caller wrote rather than for a cell they never saw."""
    layer = RNN("rnn", n_in=4, n_hidden=3)

    assert [p.name for p in (layer.cell.W_ih, layer.cell.b, layer.cell.W_hh)] == [
        "rnn_W_ih",
        "rnn_b",
        "rnn_W_hh",
    ]


def test_rejects_an_input_with_no_time_axis():
    layer = RNN("rnn", n_in=4, n_hidden=3)

    with pytest.raises(ValueError, match="no time axis to recur over"):
        layer(pt.tensor("X", shape=(4,)))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def relu(x):
    return np.maximum(x, 0.0)


class HardSigmoid(Activation):
    def __call__(self, x):
        return pt.clip(x * 0.2 + 0.5, 0.0, 1.0)


def hard_sigmoid(x):
    return np.clip(x * 0.2 + 0.5, 0.0, 1.0)


def unrolled_gru(X_np, W_ih, b, W_hh, c, phi, gate=sigmoid):
    """The gated recurrence written as a python loop, as the reference to check the scan against."""
    n_hidden = W_hh.shape[0]
    h = np.zeros((*X_np.shape[:-2], n_hidden), dtype=floatX)
    states = []
    for t in range(X_np.shape[-2]):
        from_input = X_np[..., t, :] @ W_ih + b
        from_state = h @ W_hh
        reset = gate(from_input[..., :n_hidden] + from_state[..., :n_hidden])
        update = gate(
            from_input[..., n_hidden : 2 * n_hidden] + from_state[..., n_hidden : 2 * n_hidden]
        )
        candidate = phi(
            from_input[..., 2 * n_hidden :] + reset * (from_state[..., 2 * n_hidden :] + c)
        )
        h = (1 - update) * candidate + update * h
        states.append(h)
    return np.stack(states, axis=-2)


def draw_gru_parameters(layer, rng):
    """Set every parameter to a fresh draw and hand the values back for the reference to use."""
    n_in, n_hidden = layer.cell.n_in, layer.cell.n_hidden
    W_ih = rng.normal(size=(n_in, 3 * n_hidden)).astype(floatX)
    W_hh = rng.normal(size=(n_hidden, 3 * n_hidden)).astype(floatX)
    b = rng.normal(size=(3 * n_hidden,)).astype(floatX)
    c = rng.normal(size=(n_hidden,)).astype(floatX)
    layer.cell.W_ih.set_value(W_ih)
    layer.cell.W_hh.set_value(W_hh)
    layer.cell.b.set_value(b)
    layer.cell.c.set_value(c)
    return W_ih, b, W_hh, c


def test_the_gru_matches_a_step_by_step_reference(rng):
    X = pt.tensor("X", shape=(None, None, 4))
    layer = GRU("gru", n_in=4, n_hidden=3)
    out = layer(X)
    assert out.type.shape == (None, None, 3)

    W_ih, b, W_hh, c = draw_gru_parameters(layer, rng)
    X_np = rng.normal(size=(5, 7, 4)).astype(floatX)

    np.testing.assert_allclose(
        out.eval({X: X_np}), unrolled_gru(X_np, W_ih, b, W_hh, c, np.tanh), atol=ATOL
    )


def test_the_gru_gate_slices_do_not_cross(rng):
    """Three gates read three slices of one projection, and swapping two of them still produces a
    plausible sequence. Driving each gate to its own extreme in turn pins which slice is which: the
    reference loop alone would agree with any consistent misordering of the parameter layout."""
    X = pt.tensor("X", shape=(None, None, 4))
    h0 = pt.tensor("h0", shape=(None, 3))
    layer = GRU("gru", n_in=4, n_hidden=3)
    out = layer(X, h0)

    W_in = rng.normal(size=(4, 3)).astype(floatX)
    layer.cell.W_ih.set_value(np.concatenate([np.zeros((4, 6), dtype=floatX), W_in], axis=1))
    layer.cell.W_hh.set_value(np.zeros((3, 9), dtype=floatX))
    layer.cell.c.set_value(np.zeros(3, dtype=floatX))
    X_np = rng.normal(size=(5, 7, 4)).astype(floatX)
    h0_np = rng.normal(size=(5, 3)).astype(floatX)

    # An update gate held open carries the starting state to the end untouched, whatever the input does.
    layer.cell.b.set_value(np.array([0, 0, 0, 20, 20, 20, 0, 0, 0], dtype=floatX))
    held = out.eval({X: X_np, h0: h0_np})
    np.testing.assert_allclose(held, np.broadcast_to(h0_np[:, None, :], (5, 7, 3)), atol=1e-6)

    # A reset gate held shut cuts the state out of the candidate, and with the update gate shut too the
    # step keeps nothing at all: the layer becomes a memoryless projection.
    layer.cell.b.set_value(np.array([-20, -20, -20, -20, -20, -20, 0, 0, 0], dtype=floatX))
    forgotten = out.eval({X: X_np, h0: h0_np})
    np.testing.assert_allclose(forgotten, np.tanh(X_np @ W_in), atol=1e-6)


def test_the_gru_biases_are_optional(rng):
    """Dropping the bias drops both parameters as well as both terms; an unused one left behind would
    hand the optimizer moment state to carry for a weight that never moves."""
    X = pt.tensor("X", shape=(None, None, 4))
    layer = GRU("gru", n_in=4, n_hidden=3, bias=False)
    out = layer(X)

    W_ih = rng.normal(size=(4, 9)).astype(floatX)
    W_hh = rng.normal(size=(3, 9)).astype(floatX)
    layer.cell.W_ih.set_value(W_ih)
    layer.cell.W_hh.set_value(W_hh)
    b = np.zeros(9, dtype=floatX)
    c = np.zeros(3, dtype=floatX)
    X_np = rng.normal(size=(5, 7, 4)).astype(floatX)

    assert set(collect_trainable_params(out)) == {layer.cell.W_ih, layer.cell.W_hh}
    np.testing.assert_allclose(
        out.eval({X: X_np}), unrolled_gru(X_np, W_ih, b, W_hh, c, np.tanh), atol=ATOL
    )


def test_the_gru_gates_take_their_own_activation(rng):
    """The gates and the candidate have separate keywords, and each has to reach its own place: one
    leaking into the other, or a hardcoded default in either, disagrees with the reference. A hard
    sigmoid clipped at the same endpoints is the substitution a reader would actually make, and it
    disagrees with the logistic everywhere except the two points where they cross."""
    X = pt.tensor("X", shape=(None, None, 4))
    layer = GRU("gru", n_in=4, n_hidden=3, activation=ReLU(), gate_activation=HardSigmoid())
    out = layer(X)

    W_ih, b, W_hh, c = draw_gru_parameters(layer, rng)
    X_np = rng.normal(size=(5, 7, 4)).astype(floatX)
    hard = out.eval({X: X_np})

    np.testing.assert_allclose(
        hard, unrolled_gru(X_np, W_ih, b, W_hh, c, relu, gate=hard_sigmoid), atol=ATOL
    )
    # The same candidate behind logistic gates is a different function, not a rescaling.
    logistic = unrolled_gru(X_np, W_ih, b, W_hh, c, relu)
    assert np.abs(hard - logistic).max() > 0.01


def test_the_gru_recurs_over_any_number_of_batch_axes(rng):
    """Every gate is a slice of the last axis, and the candidate's bias broadcasts against whatever
    batch axes precede it. A bare sequence has none at all."""
    X = pt.tensor("X", shape=(None, 4))
    layer = GRU("gru", n_in=4, n_hidden=3)
    out = layer(X)

    W_ih, b, W_hh, c = draw_gru_parameters(layer, rng)
    X_np = rng.normal(size=(7, 4)).astype(floatX)

    np.testing.assert_allclose(
        out.eval({X: X_np}), unrolled_gru(X_np, W_ih, b, W_hh, c, np.tanh), atol=ATOL
    )


def unrolled_lstm(X_np, W_ih, b, W_hh, phi, gate=sigmoid):
    """The gated recurrence written as a python loop, as the reference to check the scan against."""
    n_hidden = W_hh.shape[0]
    h = np.zeros((*X_np.shape[:-2], n_hidden), dtype=floatX)
    c = np.zeros_like(h)
    states = []
    for t in range(X_np.shape[-2]):
        projected = X_np[..., t, :] @ W_ih + h @ W_hh + b
        pre_in, pre_forget, pre_candidate, pre_out = (
            projected[..., i * n_hidden : (i + 1) * n_hidden] for i in range(4)
        )
        c = gate(pre_forget) * c + gate(pre_in) * phi(pre_candidate)
        h = gate(pre_out) * phi(c)
        states.append(h)
    return np.stack(states, axis=-2)


def draw_lstm_parameters(layer, rng):
    """Set every parameter to a fresh draw and hand the values back for the reference to use."""
    n_in, n_hidden = layer.cell.n_in, layer.cell.n_hidden
    W_ih = rng.normal(size=(n_in, 4 * n_hidden)).astype(floatX)
    W_hh = rng.normal(size=(n_hidden, 4 * n_hidden)).astype(floatX)
    b = rng.normal(size=(4 * n_hidden,)).astype(floatX)
    layer.cell.W_ih.set_value(W_ih)
    layer.cell.W_hh.set_value(W_hh)
    layer.cell.b.set_value(b)
    return W_ih, b, W_hh


def test_the_lstm_matches_a_step_by_step_reference(rng):
    X = pt.tensor("X", shape=(None, None, 4))
    layer = LSTM("lstm", n_in=4, n_hidden=3)
    out = layer(X)
    assert out.type.shape == (None, None, 3)

    W_ih, b, W_hh = draw_lstm_parameters(layer, rng)
    X_np = rng.normal(size=(5, 7, 4)).astype(floatX)

    np.testing.assert_allclose(
        out.eval({X: X_np}), unrolled_lstm(X_np, W_ih, b, W_hh, np.tanh), atol=ATOL
    )


def test_the_lstm_gate_slices_do_not_cross(rng):
    """Four gates read four slices of one projection, and swapping two still produces a plausible
    sequence. Driving each to its own extreme pins which slice is which; the reference loop alone would
    agree with any consistent misordering of the parameter layout."""
    X = pt.tensor("X", shape=(None, None, 4))
    layer = LSTM("lstm", n_in=4, n_hidden=3)
    out = layer(X)

    layer.cell.W_ih.set_value(np.zeros((4, 12), dtype=floatX))
    layer.cell.W_hh.set_value(np.zeros((3, 12), dtype=floatX))
    X_np = rng.normal(size=(5, 7, 4)).astype(floatX)
    shut, opened = -20.0, 20.0

    def biases(gate_in, gate_forget, candidate, gate_out):
        return np.repeat([gate_in, gate_forget, candidate, gate_out], 3).astype(floatX)

    # The input gate open onto a saturated candidate writes tanh(20) into the memory, and the output
    # gate open exposes tanh of that. The forget gate is irrelevant while the memory starts at zero.
    layer.cell.b.set_value(biases(opened, shut, opened, opened))
    np.testing.assert_allclose(
        out.eval({X: X_np}), np.full((5, 7, 3), np.tanh(np.tanh(20.0)), dtype=floatX), atol=1e-6
    )

    # The output gate shut hides that same memory, so nothing reaches h however full the memory is.
    layer.cell.b.set_value(biases(opened, shut, opened, shut))
    np.testing.assert_allclose(out.eval({X: X_np}), np.zeros((5, 7, 3)), atol=1e-6)

    # The input gate shut writes nothing, so an open output gate exposes an empty memory.
    layer.cell.b.set_value(biases(shut, shut, opened, opened))
    np.testing.assert_allclose(out.eval({X: X_np}), np.zeros((5, 7, 3)), atol=1e-6)


def test_the_lstm_bias_is_optional(rng):
    """Dropping the bias drops the parameter as well as the term; an unused one left behind would hand
    the optimizer moment state to carry for a weight that never moves."""
    X = pt.tensor("X", shape=(None, None, 4))
    layer = LSTM("lstm", n_in=4, n_hidden=3, bias=False)
    out = layer(X)

    W_ih = rng.normal(size=(4, 12)).astype(floatX)
    W_hh = rng.normal(size=(3, 12)).astype(floatX)
    layer.cell.W_ih.set_value(W_ih)
    layer.cell.W_hh.set_value(W_hh)
    b = np.zeros(12, dtype=floatX)
    X_np = rng.normal(size=(5, 7, 4)).astype(floatX)

    assert set(collect_trainable_params(out)) == {layer.cell.W_ih, layer.cell.W_hh}
    np.testing.assert_allclose(
        out.eval({X: X_np}), unrolled_lstm(X_np, W_ih, b, W_hh, np.tanh), atol=ATOL
    )


def test_the_lstm_gates_take_their_own_activation(rng):
    """The gates and the candidate have separate keywords, and each has to reach its own place. A relu
    ``activation`` also pins it being applied twice: once to the candidate and again to the memory on
    the way out. At tanh, hardcoding either one would still agree."""
    X = pt.tensor("X", shape=(None, None, 4))
    layer = LSTM("lstm", n_in=4, n_hidden=3, activation=ReLU(), gate_activation=HardSigmoid())
    out = layer(X)

    W_ih, b, W_hh = draw_lstm_parameters(layer, rng)
    X_np = rng.normal(size=(5, 7, 4)).astype(floatX)
    hard = out.eval({X: X_np})

    np.testing.assert_allclose(
        hard, unrolled_lstm(X_np, W_ih, b, W_hh, relu, gate=hard_sigmoid), atol=ATOL
    )
    logistic = unrolled_lstm(X_np, W_ih, b, W_hh, relu)
    assert np.abs(hard - logistic).max() > 0.01


def test_the_lstm_recurs_over_any_number_of_batch_axes(rng):
    """Both carried states take their batch axes from the input, and the memory has to keep them across
    the step that combines it with the gates."""
    X = pt.tensor("X", shape=(None, 4))
    layer = LSTM("lstm", n_in=4, n_hidden=3)
    out = layer(X)

    W_ih, b, W_hh = draw_lstm_parameters(layer, rng)
    X_np = rng.normal(size=(7, 4)).astype(floatX)

    np.testing.assert_allclose(
        out.eval({X: X_np}), unrolled_lstm(X_np, W_ih, b, W_hh, np.tanh), atol=ATOL
    )


def test_the_lstm_memory_carries_gradient_across_a_long_sequence(rng):
    """What the memory is for. With the forget gate open it reaches the last step untouched by any
    weight, so the gradient back to the starting memory survives fifty steps; with the gate shut the
    same path is cut and the gradient is gone. An Elman state, multiplied by a weight every step,
    has no setting that does the first."""
    X = pt.tensor("X", shape=(None, None, 4))
    h0 = pt.tensor("h0", shape=(None, 3))
    c0 = pt.tensor("c0", shape=(None, 3))
    layer = LSTM("lstm", n_in=4, n_hidden=3)
    out = layer(X, [h0, c0])
    sensitivity = pt.grad(out[..., -1, :].sum(), c0)

    layer.cell.W_ih.set_value(np.zeros((4, 12), dtype=floatX))
    layer.cell.W_hh.set_value(np.zeros((3, 12), dtype=floatX))
    X_np = rng.normal(size=(5, 50, 4)).astype(floatX)
    h0_np = np.zeros((5, 3), dtype=floatX)
    c0_np = rng.normal(size=(5, 3)).astype(floatX)

    # Input gate shut so nothing is written, output gate open so the memory reaches h.
    layer.cell.b.set_value(np.repeat([-20.0, 20.0, 0.0, 20.0], 3).astype(floatX))
    remembered = sensitivity.eval({X: X_np, h0: h0_np, c0: c0_np})
    # d tanh(c_0) / d c_0, undiminished by the fifty steps in between.
    np.testing.assert_allclose(remembered, 1.0 - np.tanh(c0_np) ** 2, atol=1e-5)

    layer.cell.b.set_value(np.repeat([-20.0, -20.0, 0.0, 20.0], 3).astype(floatX))
    forgotten = sensitivity.eval({X: X_np, h0: h0_np, c0: c0_np})
    assert np.abs(forgotten).max() < 1e-6


def test_a_reversed_layer_reads_the_sequence_from_the_end(rng):
    """Running backward has to be exactly running forward over the flipped sequence, step for step.
    Anything that merely reordered the output would agree with a forward pass on a palindrome and on
    nothing else, so the input here is drawn."""
    X = pt.tensor("X", shape=(None, None, 4))
    backward = RNN("rnn", n_in=4, n_hidden=3, reverse=True)

    W_ih, b, W_hh = draw_parameters(backward, rng)
    X_np = rng.normal(size=(5, 7, 4)).astype(floatX)

    read_backward = backward(X).eval({X: X_np})

    # The reference reads the flipped sequence forward, then puts the answers back where they came from.
    on_flipped = unrolled(X_np[..., ::-1, :], W_ih, b, W_hh, np.tanh)
    np.testing.assert_allclose(read_backward, on_flipped[..., ::-1, :], atol=ATOL)


@pytest.mark.parametrize("layer_type", [RNN, GRU, LSTM], ids=["rnn", "gru", "lstm"])
def test_every_recurrent_layer_takes_a_direction(layer_type):
    """``reverse`` lives on the loop, not on the cell, so each flat constructor has to forward it. One
    that dropped it would quietly run forward. What the loop does with it is the reversed-layer test's
    business."""
    assert layer_type("layer", n_in=4, n_hidden=3, reverse=True).reverse


def test_bidirectional_owns_the_direction_of_both_layers(rng):
    """A caller who builds both halves the same way, or reverses the wrong one, still gets one pass in
    each direction -- and the layers they handed over keep the direction they were built with, so using
    one on its own afterwards is unaffected."""
    X = pt.tensor("X", shape=(None, None, 4))
    forward = GRU("fwd", n_in=4, n_hidden=3, reverse=True)
    backward = GRU("bwd", n_in=4, n_hidden=5)
    both = Bidirectional(forward, backward)(X)
    assert both.type.shape == (None, None, 8)

    X_np = rng.normal(size=(5, 7, 4)).astype(floatX)
    evaluated = both.eval({X: X_np})

    def parameters(layer):
        cell = layer.cell
        return cell.W_ih.get_value(), cell.b.get_value(), cell.W_hh.get_value(), cell.c.get_value()

    np.testing.assert_allclose(
        evaluated[..., :3], unrolled_gru(X_np, *parameters(forward), np.tanh), atol=ATOL
    )
    np.testing.assert_allclose(
        evaluated[..., 3:],
        unrolled_gru(X_np[..., ::-1, :], *parameters(backward), np.tanh)[..., ::-1, :],
        atol=ATOL,
    )
    assert forward.reverse and not backward.reverse


def test_bidirectional_rejects_one_layer_used_twice():
    """One layer in both slots runs, but with a single set of parameters shared between the directions,
    which is the one thing the two-layer signature exists to prevent."""
    layer = GRU("gru", n_in=4, n_hidden=3)
    with pytest.raises(ValueError, match="its own parameters"):
        Bidirectional(layer, layer)


def pad_to(sequences, padded_length):
    """Stack ragged sequences into a rectangle, with the mask that says where each one ends."""
    batch = len(sequences)
    padded = np.zeros((batch, padded_length, sequences[0].shape[-1]), dtype=floatX)
    mask = np.zeros((batch, padded_length), dtype=bool)
    for row, sequence in enumerate(sequences):
        padded[row, : len(sequence)] = sequence
        mask[row, : len(sequence)] = True
    return padded, mask


def test_a_mask_lets_one_batch_hold_sequences_of_different_lengths(rng):
    """The case padding exists for: every row a different length, run as one rectangle. Each row has to
    match what it would have given on its own."""
    X = pt.tensor("X", shape=(None, None, 4))
    mask = pt.tensor("mask", shape=(None, None), dtype=bool)
    layer = GRU("gru", n_in=4, n_hidden=3, reverse=True)
    parameters = draw_gru_parameters(layer, rng)

    lengths = [2, 5, 3]
    sequences = [rng.normal(size=(length, 4)).astype(floatX) for length in lengths]
    padded, mask_np = pad_to(sequences, padded_length=max(lengths))
    together = layer(X, mask=mask).eval({X: padded, mask: mask_np})

    for row, (sequence, length) in enumerate(zip(sequences, lengths)):
        alone = unrolled_gru(sequence[::-1], *parameters, np.tanh)[::-1]
        np.testing.assert_allclose(together[row, :length], alone, atol=ATOL)


def test_a_mask_holds_every_state_a_cell_carries(rng):
    """An LSTM's memory is masked alongside its output, or a padded step would go on writing to the
    memory that the output gate reads at the next real step."""
    X = pt.tensor("X", shape=(None, None, 4))
    mask = pt.tensor("mask", shape=(None, None), dtype=bool)
    layer = LSTM("lstm", n_in=4, n_hidden=3, reverse=True)
    parameters = draw_lstm_parameters(layer, rng)

    real = rng.normal(size=(4, 4)).astype(floatX)
    padded, mask_np = pad_to([real], padded_length=9)

    np.testing.assert_allclose(
        layer(X, mask=mask).eval({X: padded, mask: mask_np})[0, :4],
        unrolled_lstm(real[::-1], *parameters, np.tanh)[::-1],
        atol=ATOL,
    )


def test_a_mask_freezes_the_final_state_of_a_padded_forward_pass(rng):
    """A forward pass is causal, so padding leaves the real positions already correct and only the
    final state drifts on. A masked step emits the state the step before it left, which holds that
    final state flat across the padding and makes ``out[..., -1, :]`` true without a per-row gather."""
    X = pt.tensor("X", shape=(None, None, 4))
    mask = pt.tensor("mask", shape=(None, None), dtype=bool)
    layer = RNN("rnn", n_in=4, n_hidden=3)
    W_ih, b, W_hh = draw_parameters(layer, rng)

    real = rng.normal(size=(3, 4)).astype(floatX)
    padded, mask_np = pad_to([real], padded_length=7)
    out = layer(X, mask=mask).eval({X: padded, mask: mask_np})

    last_real = unrolled(real, W_ih, b, W_hh, np.tanh)[-1]
    np.testing.assert_allclose(out[0, -1], last_real, atol=ATOL)
    np.testing.assert_allclose(out[0, 3:], np.broadcast_to(last_real, (4, 3)), atol=ATOL)


def test_bidirectional_reads_the_mask_in_both_directions(rng):
    """The wrapper's backward half is the one that needs the mask to read the real steps right, and the
    forward half needs it to hold its final state across the padding. Both read it from the one
    argument."""
    X = pt.tensor("X", shape=(None, None, 4))
    mask = pt.tensor("mask", shape=(None, None), dtype=bool)
    forward = GRU("fwd", n_in=4, n_hidden=3)
    backward = GRU("bwd", n_in=4, n_hidden=5)
    layer = Bidirectional(forward, backward)
    # Drawn, not left at their defaults: a zero bias makes the zero state a fixed point, so the
    # padding would not move the state and the mask would have nothing to undo.
    forward_parameters = draw_gru_parameters(forward, rng)
    backward_parameters = draw_gru_parameters(backward, rng)

    real = rng.normal(size=(3, 4)).astype(floatX)
    padded, mask_np = pad_to([real], padded_length=8)
    together = layer(X, mask=mask).eval({X: padded, mask: mask_np})

    alone_forward = unrolled_gru(real, *forward_parameters, np.tanh)
    alone_backward = unrolled_gru(real[::-1], *backward_parameters, np.tanh)[::-1]
    np.testing.assert_allclose(together[0, :3, :3], alone_forward, atol=ATOL)
    np.testing.assert_allclose(together[0, :3, 3:], alone_backward, atol=ATOL)
    # A forward pass is right on the real steps with or without the mask. Only the padding after them
    # shows whether the mask reached this half.
    np.testing.assert_allclose(together[0, -1, :3], alone_forward[-1], atol=ATOL)


def test_rejects_a_mask_that_does_not_match_the_batch_axes():
    """A mask shaped like the input, feature axis and all, is the natural mistake; scan would take it
    as a sequence and fail somewhere inside the loop."""
    X = pt.tensor("X", shape=(None, None, 4))
    layer = RNN("rnn", n_in=4, n_hidden=3)

    with pytest.raises(ValueError, match="needs a 2-dimensional mask; got a 3-dimensional one"):
        layer(X, mask=pt.tensor("mask", shape=(None, None, 4), dtype=bool))


def test_rejects_a_mask_whose_time_axis_is_shorter_than_the_input(rng):
    """Scan takes its step count from the shortest sequence it is handed, so a mask one step short runs
    the whole recurrence one step short -- leaving ``out[..., -1, :]`` an early state rather than the
    last one, which is the failure the mask is there to prevent."""
    X = pt.tensor("X", shape=(None, None, 4))
    mask = pt.tensor("mask", shape=(None, None), dtype=bool)
    layer = RNN("rnn", n_in=4, n_hidden=3)
    out = layer(X, mask=mask)
    X_np = rng.normal(size=(2, 6, 4)).astype(floatX)

    assert out.eval({X: X_np, mask: np.ones((2, 6), dtype=bool)}).shape[-2] == 6
    with pytest.raises(AssertionError, match="has shape 5, expected 6"):
        out.eval({X: X_np, mask: np.ones((2, 5), dtype=bool)})


def test_non_finite_padding_leaves_the_gradient_finite(rng):
    """Padding is only a placeholder, so a batch padded with NaN has to train exactly like one padded
    with zeros. The masked step's discarded branch still meets the padding on the backward pass, where
    a zero cotangent times the step's derivative at NaN is NaN."""
    X = pt.tensor("X", shape=(None, None, 4))
    mask = pt.tensor("mask", shape=(None, None), dtype=bool)
    layer = RNN("rnn", n_in=4, n_hidden=3, reverse=True)
    draw_parameters(layer, rng)
    parameters = [layer.cell.W_ih, layer.cell.W_hh, layer.cell.b]
    gradients = pytensor.function([X, mask], pt.grad(layer(X, mask=mask).sum(), parameters))

    zero_padded, mask_np = pad_to([rng.normal(size=(3, 4)).astype(floatX)], padded_length=6)
    nan_padded = np.where(mask_np[..., None], zero_padded, np.nan).astype(floatX)

    for nan_gradient, zero_gradient in zip(
        gradients(nan_padded, mask_np), gradients(zero_padded, mask_np)
    ):
        np.testing.assert_array_equal(nan_gradient, zero_gradient)


class MatrixMemoryCell(RecurrentCell):
    """A cell whose state carries two feature axes, as a matrix-memory recurrence does. It sums the
    step's input into every entry, so a step that ran shows up everywhere in the state."""

    def __init__(self, rows, cols):
        self.rows, self.cols = rows, cols

    def step(self, x_t, *state):
        (memory,) = state
        return (memory + x_t.sum(axis=-1)[..., None, None],)

    def initial_state(self, X):
        return (pt.zeros((*X.shape[:-2], self.rows, self.cols), dtype=X.dtype),)


def test_a_mask_holds_a_state_of_any_rank(rng):
    """The mask names batch elements and the state adds however many feature axes the cell wants, so
    the two are lined up by the state's rank rather than by assuming exactly one feature axis."""
    X = pt.tensor("X", shape=(None, None, 4))
    mask = pt.tensor("mask", shape=(None, None), dtype=bool)
    layer = Recurrent(MatrixMemoryCell(2, 3), name="matrix")

    real = rng.normal(size=(3, 4)).astype(floatX)
    padded, mask_np = pad_to([real], padded_length=6)
    evaluated = layer(X, mask=mask).eval({X: padded, mask: mask_np})

    # Every entry holds the running sum of each step's input, frozen once the real steps run out.
    running = np.cumsum(real.sum(axis=-1))
    held = np.concatenate([running, np.full(3, running[-1])])
    assert evaluated.shape == (1, 6, 2, 3)
    np.testing.assert_allclose(
        evaluated, np.broadcast_to(held[None, :, None, None], (1, 6, 2, 3)), atol=ATOL
    )
