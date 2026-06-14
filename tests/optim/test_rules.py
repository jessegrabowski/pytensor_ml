import numpy as np
import pytensor.tensor as pt
import pytest

from pytensor_ml.optim import (
    adadelta,
    adadelta_updates,
    adagrad,
    adagrad_updates,
    adam,
    adam_updates,
    adamw,
    sgd,
    sgd_updates,
)
from pytensor_ml.params import trainable
from pytensor_ml.pytensorf import function


@pytest.mark.parametrize(
    "rule",
    [
        sgd(learning_rate=1e-2),
        sgd(learning_rate=1e-2, momentum=0.9),
        sgd(learning_rate=1e-2, momentum=0.9, nesterov=True),
        adam(learning_rate=1e-2),
        adamw(learning_rate=1e-2, weight_decay=1e-2),
        adagrad(learning_rate=1e-1),
        adadelta(learning_rate=1.0),
    ],
    ids=["sgd", "sgd_momentum", "sgd_nesterov", "adam", "adamw", "adagrad", "adadelta"],
)
def test_rule_reduces_loss(run_training, rule):
    history = run_training(rule, n_steps=100)
    assert history[-1] < history[0]


def test_adam_matches_numpy_reference():
    """Adam updates match a step-by-step reference numpy implementation."""
    start = np.array([1.0, -2.0, 3.0])
    p = trainable(start.copy(), name="w")
    loss = 0.5 * (p**2).sum()  # gradient is exactly p
    fn = function([], loss, updates=adam_updates(loss, [p], learning_rate=0.1))

    lr, b1, b2, eps = 0.1, 0.9, 0.999, 1e-8
    ref = start.copy()
    m = np.zeros(3)
    v = np.zeros(3)
    for t in range(1, 11):
        g = ref.copy()  # grad evaluated at the pre-step parameter
        fn()
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g**2
        ref = ref - lr * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + eps)
        np.testing.assert_allclose(p.get_value(), ref, rtol=1e-6, atol=1e-8)


def test_adam_updates_keyed_by_object_with_named_state():
    """State is discovered by object identity; names exist only for serialization."""
    p = trainable(np.zeros(3), name="w")
    loss = (p**2).sum()
    updates = adam_updates(loss, [p])

    assert p in updates  # the exact param object is a key, not a renamed copy
    state_names = {key.name for key in updates if key is not p}
    assert state_names == {"adam/step_count", "w/adam/first_moment", "w/adam/second_moment"}


def test_adagrad_matches_numpy_reference():
    start = np.array([1.0, -2.0])
    p = trainable(start.copy(), name="w")
    loss = 0.5 * (p**2).sum()  # gradient is exactly p
    fn = function([], loss, updates=adagrad_updates(loss, [p], learning_rate=0.1))

    lr, eps = 0.1, 1e-8
    ref = start.copy()
    sum_squared = np.zeros(2)
    for _ in range(5):
        g = ref.copy()  # gradient evaluated at the pre-step parameter
        fn()
        sum_squared += g**2
        ref = ref - lr * g / np.sqrt(sum_squared + eps)
        np.testing.assert_allclose(p.get_value(), ref, rtol=1e-6)


def test_adadelta_matches_numpy_reference():
    start = np.array([1.0, -2.0])
    p = trainable(start.copy(), name="w")
    loss = 0.5 * (p**2).sum()  # gradient is exactly p
    fn = function([], loss, updates=adadelta_updates(loss, [p], learning_rate=1.0, rho=0.9))

    lr, rho, eps = 1.0, 0.9, 1e-8
    ref = start.copy()
    avg_squared_grad = np.zeros(2)
    avg_squared_update = np.zeros(2)
    for _ in range(5):
        g = ref.copy()  # gradient evaluated at the pre-step parameter
        fn()
        avg_squared_grad = rho * avg_squared_grad + (1 - rho) * g**2
        update = np.sqrt(avg_squared_update + eps) / np.sqrt(avg_squared_grad + eps) * g
        avg_squared_update = rho * avg_squared_update + (1 - rho) * update**2
        ref = ref - lr * update
        np.testing.assert_allclose(p.get_value(), ref, rtol=1e-6)


def test_precomputed_gradients_accepted():
    p = trainable(np.ones(2), name="w")
    gradients = [pt.constant(np.array([0.5, -0.5]))]
    updates = sgd(learning_rate=1.0)(gradients, [p])
    np.testing.assert_allclose(function([], updates[p])(), [0.5, 1.5])


def test_get_gradients_rejects_count_mismatch():
    weight = trainable(np.ones(2), name="w")
    bias = trainable(np.ones(2), name="b")
    one_gradient = [pt.constant(np.ones(2))]
    with pytest.raises(ValueError, match="1 gradients for 2 parameters"):
        sgd_updates(one_gradient, [weight, bias])
