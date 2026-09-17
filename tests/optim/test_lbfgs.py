import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor_ml.optim.lbfgs import LBFGSDirection
from pytensor_ml.pytensorf import function

floatX = pytensor.config.floatX
RTOL = 1e-6 if floatX == "float64" else 1e-4


def two_loop_direction(gamma, gradient, pairs):
    """Nocedal and Wright 7.4 on one flat vector, over ``(s, y)`` pairs given oldest first."""
    q = gradient.astype(np.float64)
    alphas = []
    for s, y in reversed(pairs):
        rho = 1.0 / (y @ s)
        alphas.append(rho * (s @ q))
        q = q - alphas[-1] * y
    r = gamma * q
    for (s, y), alpha in zip(pairs, reversed(alphas)):
        rho = 1.0 / (y @ s)
        beta = rho * (y @ r)
        r = r + (alpha - beta) * s
    return r


def ring_stacks(pairs, memory_size, count, shapes):
    """Lay chronological flat pairs into per-parameter ring stacks, newest at ``(count - 1) % memory_size``."""
    S = [np.zeros((memory_size, *shape), dtype=floatX) for shape in shapes]
    Y = [np.zeros((memory_size, *shape), dtype=floatX) for shape in shapes]
    splits = np.cumsum([int(np.prod(shape)) for shape in shapes])[:-1]
    for age, (s, y) in enumerate(reversed(pairs)):
        slot = (count - 1 - age) % memory_size
        for stack, piece in zip(S, np.split(s, splits)):
            stack[slot] = piece.reshape(stack.shape[1:])
        for stack, piece in zip(Y, np.split(y, splits)):
            stack[slot] = piece.reshape(stack.shape[1:])
    return S, Y


@pytest.mark.parametrize("n_pairs, count", [(2, 2), (4, 6)], ids=["not_yet_wrapped", "wrapped"])
def test_direction_matches_the_two_loop_recursion_over_a_ring(n_pairs, count):
    # The reference sees a flat vector and a chronological list, so it shares no ring or reshape
    # arithmetic with the op. Two parameters of different rank exercise the cross-parameter dot products.
    # Before the ring wraps its empty slots lead the order; after, the newest pair sits mid-ring.
    rng = np.random.default_rng(0)
    shapes = [(3, 2), (4,)]
    size = sum(int(np.prod(shape)) for shape in shapes)
    memory_size, gamma = 4, 0.7
    gradient = rng.normal(size=size).astype(floatX)
    pairs = []
    for _ in range(n_pairs):
        s = rng.normal(size=size).astype(floatX)
        pairs.append((s, rng.normal(size=size).astype(floatX) + 0.5 * s))  # keeps y . s > 0
    S, Y = ring_stacks(pairs, memory_size, count, shapes)

    op = LBFGSDirection(n_parameters=2, memory_size=memory_size)
    gradients = [pt.tensor(f"g{i}", shape=shape) for i, shape in enumerate(shapes)]
    S_in = [pt.tensor(f"S{i}", shape=(memory_size, *shape)) for i, shape in enumerate(shapes)]
    Y_in = [pt.tensor(f"Y{i}", shape=(memory_size, *shape)) for i, shape in enumerate(shapes)]
    direction = function(
        [*gradients, *S_in, *Y_in], op(count, gamma, *gradients, *S_in, *Y_in, return_list=True)
    )

    splits = np.cumsum([int(np.prod(shape)) for shape in shapes])[:-1]
    gradient_pieces = [
        piece.reshape(shape) for piece, shape in zip(np.split(gradient, splits), shapes)
    ]
    got = np.concatenate([d.ravel() for d in direction(*gradient_pieces, *S, *Y)])
    np.testing.assert_allclose(got, two_loop_direction(gamma, gradient, pairs), rtol=RTOL)


def test_an_empty_memory_scales_the_gradient():
    # Built at float32 whatever floatX is, so the Python-float gamma has a narrower dtype to upcast.
    rng = np.random.default_rng(1)
    g = rng.normal(size=5).astype("float32")
    S = np.zeros((3, 5), dtype="float32")
    Y = np.zeros((3, 5), dtype="float32")

    d = LBFGSDirection(n_parameters=1, memory_size=3)(0, 0.25, g, S, Y)

    assert d.dtype == "float32"
    np.testing.assert_allclose(d.eval(), 0.25 * g, rtol=1e-6)


@pytest.mark.parametrize(
    "props, tensors, message",
    [
        ({"n_parameters": 0, "memory_size": 3}, (), "n_parameters must be at least 1"),
        (
            {"n_parameters": 1, "memory_size": 0},
            (np.ones(2), np.ones((0, 2)), np.ones((0, 2))),
            "memory_size must be at least 1",
        ),
        (
            {"n_parameters": 1, "memory_size": 3},
            (np.ones(2), np.ones((3, 2)), np.ones((3, 2)), np.ones((3, 2))),
            "takes 3 tensors",
        ),
        (
            {"n_parameters": 1, "memory_size": 3},
            (np.ones(2), np.ones((4, 2)), np.ones((3, 2))),
            "memory_size=3",
        ),
        (
            {"n_parameters": 1, "memory_size": 3},
            (np.ones(2), np.ones((3, 2, 1)), np.ones((3, 2))),
            "memory_size=3",
        ),
        (
            {"n_parameters": 1, "memory_size": 3},
            (np.ones(2, dtype="float32"), np.ones((3, 2)), np.ones((3, 2))),
            "dtype",
        ),
    ],
    ids=["no_parameters", "no_memory", "extra_tensor", "wrong_slots", "wrong_rank", "wrong_dtype"],
)
def test_malformed_inputs_are_refused_at_build_time(props, tensors, message):
    with pytest.raises(ValueError, match=message):
        LBFGSDirection(**props)(0, 1.0, *tensors)
