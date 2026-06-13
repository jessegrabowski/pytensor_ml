import numpy as np
import pytensor.tensor as pt

from pytensor_ml.optim import clip_by_global_norm, clip_by_value
from pytensor_ml.params import trainable
from pytensor_ml.pytensorf import function


def test_clip_by_global_norm_rescales_oversized_step():
    p = trainable(np.zeros(3), name="w")
    step = pt.constant(np.array([3.0, 4.0, 0.0]))  # global norm 5
    out = clip_by_global_norm(1.0)({p: p + step}, [p])
    # clip = 1 / (5 + eps); new p = clip * step -> unit norm in the same direction
    np.testing.assert_allclose(function([], out[p])(), [0.6, 0.8, 0.0], rtol=1e-6)


def test_clip_by_global_norm_leaves_small_step():
    p = trainable(np.zeros(2), name="w")
    step = pt.constant(np.array([0.3, 0.4]))  # norm 0.5 < 1.0
    out = clip_by_global_norm(1.0)({p: p + step}, [p])
    np.testing.assert_allclose(function([], out[p])(), [0.3, 0.4], rtol=1e-6)


def test_clip_by_global_norm_uses_joint_norm_across_params():
    weight = trainable(np.zeros(2), name="w")
    bias = trainable(np.zeros(1), name="b")
    # Joint norm sqrt(3^2 + 4^2) = 5, so every step is scaled by 1/5; a per-parameter
    # norm would instead scale the two steps by 1/3 and 1/4.
    updates = {
        weight: weight + pt.constant(np.array([3.0, 0.0])),
        bias: bias + pt.constant(np.array([4.0])),
    }
    out = clip_by_global_norm(1.0)(updates, [weight, bias])
    new_weight, new_bias = function([], [out[weight], out[bias]])()
    np.testing.assert_allclose(new_weight, [0.6, 0.0], rtol=1e-6)
    np.testing.assert_allclose(new_bias, [0.8], rtol=1e-6)


def test_clip_by_value_clamps_elementwise():
    p = trainable(np.zeros(3), name="w")
    step = pt.constant(np.array([5.0, -5.0, 0.5]))
    out = clip_by_value(-1.0, 1.0)({p: p + step}, [p])
    np.testing.assert_allclose(function([], out[p])(), [1.0, -1.0, 0.5])
