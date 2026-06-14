import json

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor.graph.traversal import ancestors
from pytensor.tensor.random.op import RandomVariable

from pytensor_ml.activations import GELU, LeakyReLU, ReLU, Sigmoid, Softmax, SoftPlus, Tanh
from pytensor_ml.json_serialize import (
    deserialize_graph,
    op_from_json,
    prop_to_json,
    serialize_graph,
)
from pytensor_ml.layers import BatchNorm2D, Concatenate, Dropout, Linear, Sequential, Squeeze
from pytensor_ml.params import collect_shared_variables, collect_trainable_params

ALL_ACTIVATIONS = [
    ReLU(),
    LeakyReLU(),
    Tanh(),
    Sigmoid(),
    SoftPlus(),
    Softmax(),
    GELU(approximate=False),
    GELU(approximate=True),
]


def assert_outputs_roundtrip(data_inputs, outputs, data_values):
    """Serialize the graph of ``outputs`` to JSON and back, and check the rebuilt graph computes the same."""
    output_list = outputs if isinstance(outputs, list) else [outputs]
    shared = collect_shared_variables(output_list)
    blob = json.dumps(serialize_graph([*data_inputs, *shared], output_list))
    rebuilt_inputs, rebuilt_outputs = deserialize_graph(json.loads(blob))

    original = pytensor.function(data_inputs, output_list)  # shared inputs captured
    restored = pytensor.function(rebuilt_inputs, rebuilt_outputs)  # every input explicit
    feed = [*data_values, *(s.get_value() for s in shared)]
    for got, want in zip(restored(*feed), original(*data_values)):
        np.testing.assert_allclose(got, want, rtol=1e-6)


def initialized_network(*layers, seed=0):
    rng = np.random.default_rng(seed)
    X = pt.matrix("X")
    output = Sequential(*layers)(X)
    for parameter in collect_trainable_params(output):
        parameter.set_value(rng.normal(size=parameter.get_value().shape))
    return X, output


def _activation_id(activation):
    if isinstance(activation, GELU) and activation.approximate:
        return "GELU_tanh"
    return type(activation).__name__


@pytest.mark.parametrize("activation", ALL_ACTIVATIONS, ids=_activation_id)
def test_every_activation_roundtrips(activation):
    X, output = initialized_network(Linear("fc1", 4, 8), activation, Linear("fc2", 8, 4))
    assert_outputs_roundtrip([X], output, [np.random.default_rng(1).normal(size=(5, 4))])


@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no_bias"])
def test_linear_bias_variants_roundtrip(bias):
    X, output = initialized_network(Linear("fc", 4, 3, bias=bias))
    assert_outputs_roundtrip([X], output, [np.random.default_rng(1).normal(size=(5, 4))])


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"affine": False}, {"track_running_stats": False}],
    ids=["default", "no_affine", "no_running_stats"],
)
def test_batchnorm_variants_roundtrip(kwargs):
    X, output = initialized_network(Linear("fc", 4, 4), BatchNorm2D("bn", n_in=4, **kwargs))
    assert_outputs_roundtrip([X], output, [np.random.default_rng(1).normal(size=(8, 4))])


def test_squeeze_roundtrips():
    X = pt.matrix("X")
    assert_outputs_roundtrip(
        [X], Squeeze(X[:, :1], axis=1), [np.random.default_rng(0).normal(size=(5, 4))]
    )


def test_concatenate_roundtrips():
    X = pt.matrix("X")
    assert_outputs_roundtrip(
        [X], Concatenate([X, X], axis=1), [np.random.default_rng(0).normal(size=(5, 4))]
    )


def test_multi_output_network_roundtrips():
    X = pt.matrix("X")
    output = [Linear("head_a", 4, 2)(X), Linear("head_b", 4, 3)(X)]
    for parameter in collect_trainable_params(output):
        parameter.set_value(np.random.default_rng(0).normal(size=parameter.get_value().shape))
    assert_outputs_roundtrip([X], output, [np.random.default_rng(1).normal(size=(5, 4))])


def test_dropout_graph_with_rng_roundtrips():
    # Dropout introduces an RNG (RandomGeneratorType) shared variable and a bernoulli RandomVariable; the
    # bernoulli must survive reconstruction, not collapse to an identity pass-through.
    X = pt.matrix("X")
    output = Dropout(p=0.5, random_state=0)(X)
    blob = json.dumps(serialize_graph([X, *collect_shared_variables(output)], [output]))
    _, rebuilt_outputs = deserialize_graph(json.loads(blob))
    assert any(
        node.owner and isinstance(node.owner.op, RandomVariable)
        for node in ancestors(rebuilt_outputs)
    )


def test_scan_recurrent_loop_roundtrips():
    rng = np.random.default_rng(0)
    sequence = pt.matrix("sequence")
    W_rec = pytensor.shared(rng.normal(size=(3, 3)), name="W_rec")

    def step(x_t, hidden):
        return pt.tanh(x_t + hidden @ W_rec)

    hidden_seq = pytensor.scan(
        step, sequences=sequence, outputs_info=pt.zeros(3), return_updates=False
    )
    assert_outputs_roundtrip([sequence], hidden_seq[-1], [rng.normal(size=(6, 3))])


def test_unregistered_scalar_op_raises_loudly():
    with pytest.raises(NotImplementedError, match="Unregistered scalar op"):
        op_from_json({"family": "scalar", "type": "NoSuchScalarOp"})


def test_non_json_native_prop_raises_loudly():
    # The "no silent drop" guarantee: a prop the codec can't represent must raise, not vanish.
    with pytest.raises(TypeError, match="Unserializable op prop"):
        prop_to_json(object())
