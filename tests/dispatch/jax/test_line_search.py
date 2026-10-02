import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

pytest.importorskip("jax")
pytest.importorskip("optax")

from pytensor_ml.optim.line_search import search_along, zoom_line_search
from tests.dispatch.jax.test_basic import jax_mode
from tests.optim.test_line_search import OBJECTIVES, PINNED, D, X


@pytest.mark.parametrize("case, expected", PINNED.values(), ids=PINNED.keys())
def test_the_search_runs_through_optax_and_takes_the_same_step(case, expected):
    """JAX cannot run the `scan` the search loops in, so it runs optax's search instead. The pinned
    values are optax's own, and the default backends' port matches them, so all three agree."""
    objective, x0, direction_of, max_steps = case
    loss = OBJECTIVES[objective](X)
    [gradient] = pt.grad(loss, [X])
    result = search_along(loss, [X], [gradient], [D], zoom_line_search(max_steps))
    run = pytensor.function([X, D], list(result), mode=jax_mode)
    direction = direction_of(pytensor.function([X], gradient)(x0))

    step_size, failed, evaluations = run(x0, direction)

    np.testing.assert_allclose(float(step_size), expected[0], rtol=1e-9)
    assert (bool(failed), int(evaluations)) == expected[1:]


@pytest.mark.parametrize("guess", [0.5, 3.0], ids=["shorter", "longer"])
def test_a_first_trial_other_than_one_matches_the_default_backend(guess):
    """optax always starts at a step of one, so the direction is rescaled by the first trial on the way
    in and the step on the way out. A wrong rescaling, of the step or of the tolerances measured in
    steps, would land on a different point from the port's."""
    loss = OBJECTIVES["rosenbrock"](X)
    [gradient] = pt.grad(loss, [X])
    result = search_along(loss, [X], [gradient], [D], zoom_line_search(), guess=guess)
    x0 = np.array([-1.2, 1.0])
    direction = -pytensor.function([X], gradient)(x0)

    on_jax = pytensor.function([X, D], list(result), mode=jax_mode)(x0, direction)
    on_default = pytensor.function([X, D], list(result))(x0, direction)

    np.testing.assert_allclose(float(on_jax[0]), float(on_default[0]), rtol=1e-9)
    assert int(on_jax[2]) == int(on_default[2])


def test_a_search_stopped_at_its_largest_step_has_not_failed():
    """Capped at 0.1, the quadratic's search never brackets an acceptable step and stops at the cap,
    which counts as done. optax reports no failure flag, and its last trial misses the curvature
    condition there, so the flag read off that trial has to make the same exception."""
    loss = OBJECTIVES["quadratic"](X)
    [gradient] = pt.grad(loss, [X])
    result = search_along(loss, [X], [gradient], [D], zoom_line_search(max_learning_rate=0.1))
    x0, direction = np.zeros(2), np.ones(2)

    on_jax = pytensor.function([X, D], list(result), mode=jax_mode)(x0, direction)
    on_default = pytensor.function([X, D], list(result))(x0, direction)

    assert [np.asarray(value).item() for value in on_jax] == [0.1, False, 1]
    assert [np.asarray(value).item() for value in on_default] == [0.1, False, 1]
