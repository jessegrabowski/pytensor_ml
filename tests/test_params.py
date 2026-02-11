import pytensor.tensor as pt
import pytest

from pytensor_ml.activations import ReLU
from pytensor_ml.layers import BatchNorm2D, Dropout, Linear, Sequential
from pytensor_ml.params import (
    collect_data_inputs,
    collect_graph_inputs,
    collect_non_trainable_updates,
    collect_trainable_params,
)


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


class TestCollectGraphInputs:
    def test_simple_network(self, simple_network):
        X, y = simple_network
        inputs = collect_graph_inputs(y)
        # X + 2 weights + 2 biases = 5 inputs
        assert len(inputs) == 5
        assert X in inputs

    def test_accepts_single_variable(self, simple_network):
        X, y = simple_network
        inputs_single = collect_graph_inputs(y)
        inputs_list = collect_graph_inputs([y])
        assert inputs_single == inputs_list


class TestCollectTrainableParams:
    def test_simple_network_excludes_data_input(self, simple_network):
        X, y = simple_network
        params = collect_trainable_params(y, exclude={X})
        # 2 weights + 2 biases = 4 params
        assert len(params) == 4
        assert X not in params

    def test_batchnorm_excludes_running_stats(self, network_with_batchnorm):
        X, y = network_with_batchnorm
        params = collect_trainable_params(y, exclude={X})
        # fc1: W, b; bn1: loc, scale; fc2: W, b = 6 params
        # running_mean and running_var should be excluded
        assert len(params) == 6
        param_names = {p.name for p in params}
        assert "bn1_running_mean" not in param_names
        assert "bn1_running_var" not in param_names

    def test_without_exclude_includes_data_input(self, simple_network):
        X, y = simple_network
        params = collect_trainable_params(y)
        assert X in params


class TestCollectNonTrainableUpdates:
    def test_simple_network_no_updates(self, simple_network):
        _, y = simple_network
        updates = collect_non_trainable_updates(y)
        assert updates == {}

    def test_batchnorm_has_running_stat_updates(self, network_with_batchnorm):
        _, y = network_with_batchnorm
        updates = collect_non_trainable_updates(y)
        assert len(updates) == 2
        old_names = {v.name for v in updates.keys()}
        assert "bn1_running_mean" in old_names
        assert "bn1_running_var" in old_names

    def test_dropout_no_updates(self, network_with_dropout):
        _, y = network_with_dropout
        updates = collect_non_trainable_updates(y)
        assert updates == {}


class TestCollectDataInputs:
    def test_simple_network(self, simple_network):
        X, y = simple_network
        data_inputs = collect_data_inputs(y)
        assert data_inputs == [X]

    def test_batchnorm_network(self, network_with_batchnorm):
        X, y = network_with_batchnorm
        data_inputs = collect_data_inputs(y)
        assert data_inputs == [X]

    def test_with_explicit_params(self, simple_network):
        X, y = simple_network
        params = collect_trainable_params(y, exclude={X})
        data_inputs = collect_data_inputs(y, params=params)
        assert data_inputs == [X]
