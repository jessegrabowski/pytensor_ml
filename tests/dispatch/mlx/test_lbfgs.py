import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

pytest.importorskip("mlx.core")

from pytensor_ml.optim.lbfgs import LBFGSDirection
from tests.dispatch.mlx.test_basic import compare_mlx_and_py
from tests.optim.test_lbfgs import ring_stacks, two_loop_direction

floatX = pytensor.config.floatX


@pytest.mark.parametrize("n_pairs, count", [(2, 2), (4, 6)], ids=["not_yet_wrapped", "wrapped"])
def test_direction_matches_py(n_pairs, count):
    rng = np.random.default_rng(sum(map(ord, "MLX LBFGS")))
    shapes = [(3, 2), (4,)]
    size = sum(int(np.prod(shape)) for shape in shapes)
    memory_size, gamma = 4, 0.7
    gradient = rng.normal(size=size).astype(floatX)
    pairs = []
    for _ in range(n_pairs):
        s = rng.normal(size=size).astype(floatX)
        pairs.append((s, rng.normal(size=size).astype(floatX) + 0.5 * s))
    S, Y = ring_stacks(pairs, memory_size, count, shapes)
    splits = np.cumsum([int(np.prod(shape)) for shape in shapes])[:-1]
    gradient_pieces = [
        piece.reshape(shape) for piece, shape in zip(np.split(gradient, splits), shapes)
    ]

    gradients = [pt.tensor(f"g{i}", shape=shape) for i, shape in enumerate(shapes)]
    S_in = [pt.tensor(f"S{i}", shape=(memory_size, *shape)) for i, shape in enumerate(shapes)]
    Y_in = [pt.tensor(f"Y{i}", shape=(memory_size, *shape)) for i, shape in enumerate(shapes)]
    op = LBFGSDirection(n_parameters=2, memory_size=memory_size)
    outputs = op(count, gamma, *gradients, *S_in, *Y_in, return_list=True)

    _, got = compare_mlx_and_py(
        [*gradients, *S_in, *Y_in],
        outputs,
        [*gradient_pieces, *S, *Y],
        assert_fn=lambda got, want: np.testing.assert_allclose(got, want, rtol=1e-4),
    )
    want = two_loop_direction(gamma, gradient, pairs)
    np.testing.assert_allclose(
        np.concatenate([np.asarray(d).ravel() for d in got]), want, rtol=1e-4
    )


def test_a_single_parameter_returns_one_array():
    g = np.arange(5, dtype=floatX)
    S = np.zeros((3, 5), dtype=floatX)
    Y = np.zeros((3, 5), dtype=floatX)
    g_in = pt.tensor("g", shape=(5,))
    S_in = pt.tensor("S", shape=(3, 5))
    Y_in = pt.tensor("Y", shape=(3, 5))

    d = LBFGSDirection(n_parameters=1, memory_size=3)(0, 0.5, g_in, S_in, Y_in)

    compare_mlx_and_py([g_in, S_in, Y_in], d, [g, S, Y])
