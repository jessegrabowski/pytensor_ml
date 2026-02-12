import numpy as np
import pytensor.tensor as pt

from pytensor import config

from pytensor_ml.layers import BatchNorm2D, Linear, Sequential
from pytensor_ml.model import Model


class TestModelPredict:
    def test_simple_network(self):
        X = pt.tensor("X", shape=(None, 6))
        mlp = Sequential(Linear("fc1", n_in=6, n_out=3), Linear("fc2", n_in=3, n_out=1))
        y = mlp(X)

        model = Model(X, y)
        model.initialize(seed=42)

        X_test = np.random.randn(10, 6).astype(config.floatX)
        result = model.predict(X_test)

        assert result.shape == (10, 1)
        assert result.dtype == config.floatX

    def test_with_batchnorm(self):
        X = pt.tensor("X", shape=(None, 6))
        network = Sequential(
            Linear("fc1", n_in=6, n_out=3),
            BatchNorm2D("bn1", n_in=3),
            Linear("fc2", n_in=3, n_out=1),
        )
        y = network(X)

        model = Model(X, y)
        model.initialize(seed=42)

        X_test = np.random.randn(10, 6).astype(config.floatX)
        result = model.predict(X_test)

        assert result.shape == (10, 1)

    def test_predict_uses_running_stats(self):
        """Verify that predict uses running stats (via rewrite) not batch stats."""
        X = pt.tensor("X", shape=(None, 4))
        fc1 = Linear("fc1", n_in=4, n_out=4)
        bn = BatchNorm2D("bn1", n_in=4)
        network = Sequential(fc1, bn)
        y = network(X)

        model = Model(X, y)
        model.initialize(seed=42)

        # Set specific running stats
        bn.running_mean.set_value(np.array([1.0, 2.0, 3.0, 4.0], dtype=config.floatX))
        bn.running_var.set_value(np.array([1.0, 1.0, 1.0, 1.0], dtype=config.floatX))

        # Predict with two different batches - should give same normalization
        # if using running stats (batch stats would differ)
        X1 = np.random.randn(5, 4).astype(config.floatX)
        X2 = np.random.randn(20, 4).astype(config.floatX)

        # Get FC output before normalization for both batches
        fc_weight = fc1.W.get_value()
        fc_bias = fc1.b.get_value()
        fc_out1 = X1 @ fc_weight + fc_bias
        fc_out2 = X2 @ fc_weight + fc_bias

        # Expected: normalized using running stats
        scale = bn.scale.get_value()
        loc = bn.loc.get_value()
        running_mean = bn.running_mean.get_value()
        running_var = bn.running_var.get_value()

        expected1 = (fc_out1 - running_mean) / np.sqrt(running_var + bn.epsilon) * scale + loc
        expected2 = (fc_out2 - running_mean) / np.sqrt(running_var + bn.epsilon) * scale + loc

        result1 = model.predict(X1)
        result2 = model.predict(X2)

        np.testing.assert_allclose(result1, expected1, rtol=1e-5)
        np.testing.assert_allclose(result2, expected2, rtol=1e-5)

    def test_predict_caches_function(self):
        """Verify that predict function is compiled once and reused."""
        X = pt.tensor("X", shape=(None, 4))
        y = Linear("fc1", n_in=4, n_out=2)(X)

        model = Model(X, y)
        model.initialize(seed=42)

        assert model._predict_fn is None

        X_test = np.random.randn(5, 4).astype(config.floatX)
        model.predict(X_test)

        assert model._predict_fn is not None
        fn_id = id(model._predict_fn)

        # Second call should reuse same function
        model.predict(X_test)
        assert id(model._predict_fn) == fn_id
