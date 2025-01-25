import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor.graph.basic import explicit_graph_inputs

from pytensor_ml.layers import Dropout, Linear, Sequential

floatX = pytensor.config.floatX


def test_linear_layer():
    X = pt.tensor("X", shape=(None, 6))
    linear = Linear(name="Linear_1", n_in=6, n_out=3)
    out = linear(X)

    X_in, W, b = out.owner.inputs
    [X_out] = out.owner.outputs

    assert out.owner.op.name == "Linear_1[(?,6) -> (?,3)]"

    assert W.name == "Linear_1_W"
    assert b.name == "Linear_1_b"
    assert X_out.name == "Linear_1_output"

    X_np = np.random.normal(size=(10, 6)).astype(floatX)
    W_np = np.random.normal(size=(6, 3)).astype(floatX)
    b_np = np.random.normal(size=(3,)).astype(floatX)

    res = out.eval({X: X_np, W: W_np, b: b_np})
    np.testing.assert_allclose(res, X_np @ W_np + b_np)


def test_sequential():
    mlp = Sequential(
        Linear(name="Linear_1", n_in=6, n_out=3), Linear(name="Linear_2", n_in=3, n_out=1)
    )

    X = pt.tensor("X", shape=(None, 6))
    out = mlp(X)
    assert out.type.shape == (None, 1)

    X_np = np.random.normal(size=(10, 6)).astype(floatX)
    W1_np = np.random.normal(size=(6, 3)).astype(floatX)
    b1_np = np.random.normal(size=(3,)).astype(floatX)
    W2_np = np.random.normal(size=(3, 1)).astype(floatX)
    b2_np = np.random.normal(size=(1,)).astype(floatX)

    f = pytensor.function(list(explicit_graph_inputs(out)), out)
    res = f(X_np, W1_np, b1_np, W2_np, b2_np)

    np.testing.assert_allclose(res, (X_np @ W1_np + b1_np) @ W2_np + b2_np)


def test_dropout():
    X = pt.tensor("X", shape=(None, 6))
    dropout = Dropout(name="Dropout_1", p=1.0)
    out = dropout(X)

    X_np = np.random.normal(size=(10, 6)).astype(floatX)

    res = out.eval({X: X_np})
    np.testing.assert_allclose(res, np.zeros_like(X_np))


def test_invalid_dropout_p_raises():
    with pytest.raises(
        ValueError, match="Dropout probability has to be between 0 and 1, but got -0.1"
    ):
        Dropout(name=None, p=-0.1)

    with pytest.raises(
        ValueError, match="Dropout probability has to be between 0 and 1, but got 1.1"
    ):
        Dropout(name=None, p=1.1)
