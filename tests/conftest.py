import pytensor.tensor as pt
import pytest

from pytensor_ml.activations import ReLU
from pytensor_ml.layers import BatchNorm2D, Dropout, Linear, Sequential


@pytest.fixture
def simple_network():
    X = pt.tensor("X", shape=(None, 64))
    network = Sequential(
        Linear("fc1", 64, 32),
        ReLU(),
        Linear("fc2", 32, 10),
    )
    y = network(X)
    return X, y


@pytest.fixture
def network_with_batchnorm():
    X = pt.tensor("X", shape=(None, 64))
    network = Sequential(
        Linear("fc1", 64, 32),
        BatchNorm2D("bn1", n_in=32),
        ReLU(),
        Linear("fc2", 32, 10),
    )
    y = network(X)
    return X, y


@pytest.fixture
def network_with_dropout():
    X = pt.tensor("X", shape=(None, 64))
    network = Sequential(
        Linear("fc1", 64, 32),
        Dropout(p=0.5),
        ReLU(),
        Linear("fc2", 32, 10),
    )
    y = network(X)
    return X, y
