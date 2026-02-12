import numpy as np

from pytensor_ml.params import (
    collect_non_trainable_params,
    collect_trainable_params,
    trainable,
)
from pytensor_ml.state import OptimizerState, initialize_params


class TestInitializeParams:
    def test_xavier_normal(self, simple_network):
        X, y = simple_network
        params = collect_trainable_params(y)
        values = initialize_params(params, scheme="xavier_normal", rng=np.random.default_rng(42))

        assert len(values) == len(params)
        for param, val in zip(params, values):
            assert val.shape == param.get_value().shape
            assert str(val.dtype) == str(param.get_value().dtype)

    def test_zeros(self, simple_network):
        X, y = simple_network
        params = collect_trainable_params(y)
        values = initialize_params(params, scheme="zeros")

        for val in values:
            np.testing.assert_array_equal(val, 0)

    def test_reproducible_with_seed(self, simple_network):
        X, y = simple_network
        params = collect_trainable_params(y)

        values1 = initialize_params(params, rng=np.random.default_rng(123))
        values2 = initialize_params(params, rng=np.random.default_rng(123))

        for v1, v2 in zip(values1, values2):
            np.testing.assert_array_equal(v1, v2)


class TestOptimizerState:
    def test_initialize_simple_network(self, simple_network):
        X, y = simple_network
        params = collect_trainable_params(y)

        state = OptimizerState(params)
        state.initialize(seed=42)

        assert len(state.param_values) == len(params)
        for param, val in zip(params, state.param_values):
            assert val.shape == param.get_value().shape

    def test_initialize_with_optimizer_params(self, simple_network):
        X, y = simple_network
        params = collect_trainable_params(y)
        # Simulate Adam's m and v buffers
        optimizer_params = [
            trainable(np.zeros_like(p.get_value()), name=f"{p.name}_m") for p in params
        ] + [trainable(np.zeros_like(p.get_value()), name=f"{p.name}_v") for p in params]

        state = OptimizerState(params, optimizer_params=optimizer_params)
        state.initialize(seed=42)

        assert len(state.optimizer_values) == 2 * len(params)
        # Optimizer state should be zeros by default
        for val in state.optimizer_values:
            np.testing.assert_array_equal(val, 0)

    def test_initialize_with_non_trainable(self, network_with_batchnorm):
        X, y = network_with_batchnorm
        params = collect_trainable_params(y)
        non_trainable = collect_non_trainable_params(y)

        state = OptimizerState(params, non_trainable_params=non_trainable)
        state.initialize(seed=42)

        assert len(state.non_trainable_values) == 2  # running_mean, running_var

    def test_update_values(self, simple_network):
        X, y = simple_network
        params = collect_trainable_params(y)

        state = OptimizerState(params)
        state.initialize(seed=42)

        new_values = [np.ones_like(v) for v in state.param_values]
        state.param_values = new_values

        for current in state.param_values:
            np.testing.assert_array_equal(current, 1)

    def test_method_chaining(self, simple_network):
        X, y = simple_network
        params = collect_trainable_params(y)

        state = OptimizerState(params).initialize(seed=42)
        assert state.param_values is not None


class TestStateCheckpointing:
    def test_state_dict_roundtrip(self, simple_network):
        X, y = simple_network
        params = collect_trainable_params(y)

        state = OptimizerState(params)
        state.initialize(seed=42)

        saved = state.state_dict()
        original_values = [v.copy() for v in state.param_values]

        # Modify state
        state.param_values = [np.zeros_like(v) for v in state.param_values]

        # Restore
        state.load_state_dict(saved)

        for orig, restored in zip(original_values, state.param_values):
            np.testing.assert_array_equal(orig, restored)

    def test_save_load_file(self, simple_network, tmp_path):
        X, y = simple_network
        params = collect_trainable_params(y)

        state = OptimizerState(params)
        state.initialize(seed=42)
        original_values = [v.copy() for v in state.param_values]

        path = tmp_path / "checkpoint.npz"
        state.save(str(path))

        # Modify state
        state.param_values = [np.zeros_like(v) for v in state.param_values]

        # Restore from file
        state.load(str(path))

        for orig, restored in zip(original_values, state.param_values):
            np.testing.assert_array_equal(orig, restored)
