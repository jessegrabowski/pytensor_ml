import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

pytest.importorskip("jax")

from pytensor_ml.optim.line_search import search_along, zoom_line_search
from tests.dispatch.jax.test_basic import jax_mode
from tests.optim.test_line_search import OBJECTIVES, PINNED, ROSENBROCK_START, D, X


@pytest.mark.parametrize("case, expected", PINNED.values(), ids=PINNED.keys())
def test_the_search_takes_the_pinned_step_on_jax(case, expected):
    objective, x0, direction_of, max_steps = case
    loss = OBJECTIVES[objective](X)
    [gradient] = pt.grad(loss, [X])
    result = search_along(loss, [X], [gradient], [D], zoom_line_search(max_steps))
    run = pytensor.function([X, D], list(result), mode=jax_mode)
    direction = direction_of(pytensor.function([X], gradient)(x0))

    step_size, failed, evaluations = run(x0, direction)

    np.testing.assert_allclose(float(step_size), expected[0], rtol=1e-9)
    assert (bool(failed), int(evaluations)) == expected[1:]


# Each row is (objective, start, direction from the gradient, search arguments, first trial, and the
# factor the gradients handed to the search are scaled by).
EDGES = {
    "stops_at_the_cap": (
        "quadratic",
        np.zeros(2),
        lambda g: np.ones(2),
        {"max_learning_rate": 0.1},
        1.0,
        1.0,
    ),
    "fails_back_onto_the_cap": (
        "quadratic",
        np.zeros(2),
        lambda g: np.ones(2),
        {"max_steps": 1, "max_learning_rate": 5.9},
        5.9,
        1.0,
    ),
    "uses_the_gradients_it_is_given": ("rosenbrock", ROSENBROCK_START, lambda g: -g, {}, 1.0, 0.5),
    "starts_from_a_nan": ("log_barrier", np.array([1.5]), lambda g: np.ones(1), {}, 1.0, 1.0),
    "starts_from_a_trial_other_than_one": (
        "rosenbrock",
        ROSENBROCK_START,
        lambda g: -g,
        {},
        3.0,
        1.0,
    ),
}


@pytest.mark.parametrize(
    "objective, x0, direction_of, arguments, guess, gradient_scale",
    EDGES.values(),
    ids=EDGES.keys(),
)
def test_the_search_on_jax_matches_the_default_backend(
    objective, x0, direction_of, arguments, guess, gradient_scale
):
    loss = OBJECTIVES[objective](X)
    [gradient] = pt.grad(loss, [X])
    result = search_along(
        loss, [X], [gradient_scale * gradient], [D], zoom_line_search(**arguments), guess=guess
    )
    direction = direction_of(pytensor.function([X], gradient)(x0))

    on_jax = pytensor.function([X, D], list(result), mode=jax_mode)(x0, direction)
    on_default = pytensor.function([X, D], list(result))(x0, direction)

    np.testing.assert_allclose(float(on_jax[0]), float(on_default[0]), rtol=1e-9)
    assert (bool(on_jax[1]), int(on_jax[2])) == (bool(on_default[1]), int(on_default[2]))
