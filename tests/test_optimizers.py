import pytensor
import pytest

from sklearn.datasets import load_digits, make_regression
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from pytensor_ml.activations import LeakyReLU
from pytensor_ml.layers import Linear, Sequential
from pytensor_ml.loss import CrossEntropy, SquaredError
from pytensor_ml.model import Model
from pytensor_ml.optimizers import SGD, Adadelta, ADAGrad, Adam, AdamW
from pytensor_ml.util import DataLoader


@pytest.fixture
def classification_data():
    X, y = load_digits(return_X_y=True)
    y_onehot = OneHotEncoder().fit_transform(y[:, None]).toarray()
    X_normed = MinMaxScaler().fit_transform(X)
    return X_normed, y_onehot


@pytest.fixture
def regression_data():
    X, y = make_regression(n_samples=1000, n_features=64, noise=10)
    X_normed = StandardScaler().fit_transform(X)
    y_normed = StandardScaler().fit_transform(y[:, None])

    return X_normed, y_normed


@pytest.fixture()
def classification_model(classification_data):
    X_in = pytensor.tensor.tensor("X_in", shape=(None, 64))
    prediction_network = Sequential(
        Linear("Linear_1", n_in=64, n_out=256),
        LeakyReLU(),
        Linear("Linear_2", n_in=256, n_out=128),
        LeakyReLU(),
        Linear("Logits", n_in=128, n_out=10),
    )

    y_hat = prediction_network(X_in)
    model = Model(X_in, y_hat)
    model.initialize()

    return model


@pytest.fixture()
def regression_model(regression_data):
    X_in = pytensor.tensor.tensor("X_in", shape=(None, 64))
    prediction_network = Sequential(
        Linear("Linear_1", n_in=64, n_out=256),
        LeakyReLU(),
        Linear("Linear_2", n_in=256, n_out=128),
        LeakyReLU(),
        Linear("Logits", n_in=128, n_out=1),
    )

    y_hat = prediction_network(X_in)
    model = Model(X_in, y_hat)
    model.initialize()

    return model


@pytest.fixture()
def classification_loss_fn():
    return CrossEntropy(expect_onehot_labels=True, expect_logits=True, reduction="mean")


@pytest.fixture()
def regression_loss_fn():
    return SquaredError(reduction="mean")


def training_loop(dataloader, optimizer, n_epochs: int = 100):
    loss_history = []

    for _ in range(n_epochs):
        loss = optimizer.step(*dataloader())
        loss_history.append(loss)

    return loss_history


@pytest.mark.parametrize(
    "model, loss_fn, data",
    [
        ("classification_model", "classification_loss_fn", "classification_data"),
        ("regression_model", "regression_loss_fn", "regression_data"),
    ],
    ids=["classification", "regression"],
)
def test_sgd(model, loss_fn, data, request):
    model, loss_fn, data = map(request.getfixturevalue, [model, loss_fn, data])
    optim = SGD(model, loss_fn, ndim_out=2, learning_rate=1e-3)
    dataloader = DataLoader(*data, batch_size=512)

    loss_history = training_loop(dataloader, optim, n_epochs=100)
    assert loss_history[0] > loss_history[-1]


@pytest.mark.parametrize(
    "model, loss_fn, data",
    [
        ("classification_model", "classification_loss_fn", "classification_data"),
        ("regression_model", "regression_loss_fn", "regression_data"),
    ],
    ids=["classification", "regression"],
)
def test_adagrad(model, loss_fn, data, request):
    model, loss_fn, data = map(request.getfixturevalue, [model, loss_fn, data])
    optim = ADAGrad(model, loss_fn, ndim_out=2, learning_rate=1e-3)
    dataloader = DataLoader(*data, batch_size=512)

    loss_history = training_loop(dataloader, optim, n_epochs=100)
    assert loss_history[0] > loss_history[-1]


@pytest.mark.parametrize(
    "model, loss_fn, data",
    [
        ("classification_model", "classification_loss_fn", "classification_data"),
        ("regression_model", "regression_loss_fn", "regression_data"),
    ],
    ids=["classification", "regression"],
)
def test_adam(model, loss_fn, data, request):
    model, loss_fn, data = map(request.getfixturevalue, [model, loss_fn, data])

    optim = Adam(model, loss_fn, ndim_out=2, learning_rate=1e-3)
    dataloader = DataLoader(*data, batch_size=512)

    loss_history = training_loop(dataloader, optim, n_epochs=10)
    assert loss_history[0] > loss_history[-1]


@pytest.mark.parametrize(
    "model, loss_fn, data",
    [
        ("classification_model", "classification_loss_fn", "classification_data"),
        ("regression_model", "regression_loss_fn", "regression_data"),
    ],
    ids=["classification", "regression"],
)
def test_adadelta(model, loss_fn, data, request):
    model, loss_fn, data = map(request.getfixturevalue, [model, loss_fn, data])
    optim = Adadelta(model, loss_fn, ndim_out=2, learning_rate=1.0)
    dataloader = DataLoader(*data, batch_size=512)

    loss_history = training_loop(dataloader, optim, n_epochs=100)
    assert loss_history[0] > loss_history[-1]


@pytest.mark.parametrize(
    "model, loss_fn, data",
    [
        ("classification_model", "classification_loss_fn", "classification_data"),
        ("regression_model", "regression_loss_fn", "regression_data"),
    ],
    ids=["classification", "regression"],
)
def test_adamw(model, loss_fn, data, request):
    model, loss_fn, data = map(request.getfixturevalue, [model, loss_fn, data])

    optim = AdamW(model, loss_fn, ndim_out=2, learning_rate=1e-3)
    dataloader = DataLoader(*data, batch_size=512)

    loss_history = training_loop(dataloader, optim, n_epochs=10)
    assert loss_history[0] > loss_history[-1]
