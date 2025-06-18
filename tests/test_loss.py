import numpy as np
import pytest
from pytensor.graph.basic import explicit_graph_inputs

from scipy.special import softmax
from sklearn.metrics import log_loss

from pytensor_ml.loss import CrossEntropy, Reductions


def generate_categorical_data(expect_logits: bool):
    rng = np.random.default_rng()
    n_classes = rng.integers(2, 10)
    y_true = rng.integers(0, n_classes, size=(100,))
    y_true_onehot = np.eye(n_classes)[y_true]
    y_pred = (
        rng.random((100, n_classes))
        if expect_logits
        else rng.dirichlet(np.ones(n_classes), size=(100,))
    )

    return y_true, y_true_onehot, y_pred


@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("expect_logits", [True, False])
def test_cross_entropy_onehot_vs_labels(reduction: Reductions, expect_logits):
    y_true, y_true_onehot, y_pred = generate_categorical_data(expect_logits)

    loss_fn = CrossEntropy(
        expect_logits=expect_logits, expect_onehot_labels=False, reduction=reduction
    )
    loss_onehot_fn = CrossEntropy(
        expect_logits=expect_logits, expect_onehot_labels=True, reduction=reduction
    )

    loss = loss_fn(y_pred)
    loss_onehot = loss_onehot_fn(y_pred)

    loss_value = loss.eval({'y_true':y_true, 'y_pred': y_pred})
    loss_value_onehot = loss_onehot.eval({'y_true':y_true_onehot, 'y_pred': y_pred})

    np.testing.assert_allclose(loss_value, loss_value_onehot)


@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("expect_logits", [True, False])
@pytest.mark.parametrize("expect_onehot_labels", [True, False])
def test_cross_entropy(reduction: Reductions, expect_logits, expect_onehot_labels):
    loss_fn = CrossEntropy(
        reduction=reduction, expect_logits=expect_logits, expect_onehot_labels=expect_onehot_labels
    )

    y_true, y_true_onehot, y_pred = generate_categorical_data(expect_logits)
    loss = loss_fn(y_pred)
    loss_value = loss.eval({'y_true': y_true_onehot if expect_onehot_labels else y_true,
                            'y_pred': y_pred})

    if expect_logits:
        y_pred = softmax(y_pred, axis=-1)

    sklearn_loss = log_loss(y_true, y_pred, normalize=reduction == "mean")
    np.testing.assert_allclose(loss_value, sklearn_loss)
