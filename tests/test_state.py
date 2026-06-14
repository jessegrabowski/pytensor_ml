import numpy as np

from pytensor_ml.params import collect_trainable_params
from pytensor_ml.state import initialize_params


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
