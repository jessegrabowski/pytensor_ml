import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor_ml.layers import BatchNorm2D, Dropout, Linear, Sequential
from pytensor_ml.pytensorf import rewrite_for_prediction

floatX = pytensor.config.floatX


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng()


@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no_bias"])
def test_linear_layer(bias, rng):
    X = pt.tensor("X", shape=(None, 6))
    linear = Linear(name="Linear_1", n_in=6, n_out=3, bias=bias)
    out = linear(X)

    X_in, *weights = out.owner.inputs
    [X_out] = out.owner.outputs

    assert out.owner.op.name == "Linear_1[(?,6) -> (?,3)]"

    expected_names = ["Linear_1_W", "Linear_1_b"] if bias else ["Linear_1_W"]
    assert [w.name for w in weights] == expected_names

    assert X_out.name == "Linear_1_output"

    X_np = rng.normal(size=(10, 6)).astype(floatX)
    W_np = rng.normal(size=(6, 3)).astype(floatX)
    b_np = rng.normal(size=(3,)).astype(floatX)

    linear.W.set_value(W_np)
    if bias:
        linear.b.set_value(b_np)

    res = out.eval({X: X_np})
    expected = X_np @ W_np + b_np if bias else X_np @ W_np
    np.testing.assert_allclose(res, expected)


def test_sequential(rng):
    linear1 = Linear(name="Linear_1", n_in=6, n_out=3)
    linear2 = Linear(name="Linear_2", n_in=3, n_out=1)
    mlp = Sequential(linear1, linear2)

    X = pt.tensor("X", shape=(None, 6))
    out = mlp(X)
    assert out.type.shape == (None, 1)

    X_np = rng.normal(size=(10, 6)).astype(floatX)
    W1_np = rng.normal(size=(6, 3)).astype(floatX)
    b1_np = rng.normal(size=(3,)).astype(floatX)
    W2_np = rng.normal(size=(3, 1)).astype(floatX)
    b2_np = rng.normal(size=(1,)).astype(floatX)

    # Set SharedVariable values directly
    linear1.W.set_value(W1_np)
    linear1.b.set_value(b1_np)
    linear2.W.set_value(W2_np)
    linear2.b.set_value(b2_np)

    f = pytensor.function([X], out)
    res = f(X_np)

    np.testing.assert_allclose(res, (X_np @ W1_np + b1_np) @ W2_np + b2_np)


def test_dropout(rng):
    X = pt.tensor("X", shape=(None, 6))
    dropout = Dropout(name="Dropout_1", p=1.0)
    out = dropout(X)

    X_np = rng.normal(size=(10, 6)).astype(floatX)

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


@pytest.mark.parametrize("n_in", [6, None], ids=["specified", "lazy"])
def test_batch_norm_2d_forward(n_in, rng):
    X = pt.tensor("X", shape=(None, 6))
    batch_norm = BatchNorm2D(name="BatchNorm_1", n_in=n_in)
    assert batch_norm.name == "BatchNorm_1"
    out = batch_norm(X)

    X_np = rng.normal(size=(10, 6)).astype(floatX)
    gamma_np = rng.normal(size=(6,)).astype(floatX) ** 2
    beta_np = rng.normal(size=(6,)).astype(dtype=floatX) ** 2

    mean_np = np.mean(X_np, axis=0)
    var_np = np.var(X_np, axis=0)

    batch_norm.scale.set_value(gamma_np)
    batch_norm.loc.set_value(beta_np)
    batch_norm.running_mean.set_value(np.zeros_like(mean_np))
    batch_norm.running_var.set_value(np.zeros_like(var_np))

    res = out.eval({X: X_np})
    expected = (X_np - mean_np) / np.sqrt(var_np + batch_norm.epsilon)
    expected = expected * gamma_np + beta_np

    np.testing.assert_allclose(res, expected, rtol=1e-5)


def test_batch_norm_2d_learns_population_stats():
    X = pt.tensor("X", shape=(None, 32))
    batch_norm = BatchNorm2D(name="BatchNorm_1", n_in=32, momentum=0.05, epsilon=1e-8)
    X_normalized = batch_norm(X)

    _, new_running_mean, new_running_var = X_normalized.owner.outputs

    loss = pt.square(X_normalized - X).mean()
    d_loss = pt.grad(loss, [batch_norm.loc, batch_norm.scale])

    learning_rate = 1e-1
    updates = {
        batch_norm.loc: batch_norm.loc - learning_rate * d_loss[0],
        batch_norm.scale: batch_norm.scale - learning_rate * d_loss[1],
        batch_norm.running_mean: new_running_mean,
        batch_norm.running_var: new_running_var,
    }

    f = pytensor.function([X], [X_normalized, loss], updates=updates)

    batch_norm.loc.set_value(np.zeros(32, dtype=batch_norm.loc.type.dtype))
    batch_norm.scale.set_value(np.ones(32, dtype=batch_norm.scale.type.dtype))
    batch_norm.running_mean.set_value(np.zeros(32, dtype=batch_norm.running_mean.type.dtype))
    batch_norm.running_var.set_value(np.ones(32, dtype=batch_norm.running_var.type.dtype))

    for t in range(500):
        data = np.random.normal(loc=3.2, scale=6.2, size=(100, 32)).astype(X.type.dtype)

        loc_val = batch_norm.loc.get_value()
        scale_val = batch_norm.scale.get_value()

        X_norm_val, loss_val = f(data)

        np.testing.assert_allclose(
            X_norm_val,
            (data - data.mean(axis=0)) / np.sqrt(data.var(axis=0) + 1e-8) * scale_val + loc_val,
            rtol=1e-4,
            atol=1e-6,
        )

    loc_val = batch_norm.loc.get_value()
    scale_val = batch_norm.scale.get_value()
    running_mean_val = batch_norm.running_mean.get_value()
    running_var_val = batch_norm.running_var.get_value()

    np.testing.assert_allclose(loc_val, 3.2, rtol=1e-1, atol=1e-1)
    np.testing.assert_allclose(scale_val, 6.2, rtol=1e-1, atol=1e-1)
    np.testing.assert_allclose(running_mean_val, 3.2, rtol=1e-1, atol=1e-1)
    np.testing.assert_allclose(np.sqrt(running_var_val), 6.2, rtol=1e-1, atol=1e-1)

    # Check that after rewrite, the population statistics are used
    X_normalized_pred = rewrite_for_prediction(X_normalized)
    f_pred = pytensor.function([X], X_normalized_pred)
    data = np.random.normal(loc=3.2, scale=6.2, size=(100, 32)).astype(X.type.dtype)

    np.testing.assert_allclose(
        f_pred(data),
        (data - running_mean_val) / np.sqrt(running_var_val + 1e-8) * scale_val + loc_val,
        rtol=1e-6,
        atol=1e-6,
    )
