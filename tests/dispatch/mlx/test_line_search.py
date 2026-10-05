import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

pytest.importorskip("mlx.core")

from pytensor.compile.mode import Mode
from pytensor.link.mlx.linker import MLXLinker

from pytensor_ml.optim.line_search import (
    STEP,
    LineSearchOp,
    find_piece,
    search_along,
    zoom_line_search,
)
from tests.dispatch.mlx.test_basic import mx
from tests.optim.test_line_search import OBJECTIVES, PINNED

# Single precision throughout, which is where the search can run as one Metal kernel per trial; the
# barrier's 0.9 would otherwise promote its graph to float64
SINGLE_PRECISION = {
    **OBJECTIVES,
    "log_barrier": lambda x: pt.sum((x - np.float32(0.9)) ** 2 - pt.log(1 - x)),
}

# The pinned cases, plus the options the pinned cases leave at their defaults, each as (case, search
# arguments, first trial)
CASES = {
    **{name: (case, {}, 1.0) for name, (case, _) in PINNED.items()},
    "capped": (PINNED["grows_until_it_brackets"][0], {"max_learning_rate": 0.1}, 1.0),
    "armijo_only": (PINNED["zooms_into_a_bracket"][0], {"approx_dec_rtol": None}, 1.0),
    "first_trial_not_one": (PINNED["zooms_into_a_bracket"][0], {}, 3.0),
}

DEVICES = [
    pytest.param(
        mx.gpu,
        id="kernel",
        marks=pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal"),
    ),
    pytest.param(mx.cpu, id="step_graph"),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("use_compile", [True, False], ids=["compiled", "eager"])
@pytest.mark.parametrize("case, arguments, guess", CASES.values(), ids=CASES.keys())
def test_the_search_takes_the_default_backends_step(case, arguments, guess, use_compile, device):
    """mlx runs every trial the search allows and holds the state once one is accepted, so it lands on
    the same step, after the same number of trials, as the `scan` that stops there."""
    objective, x0, direction_of, max_steps = case
    # Static shapes, as parameters have: a dynamic one adds broadcast checks mlx would drop with a warning
    X = pt.vector("x", dtype="float32", shape=x0.shape)
    D = pt.vector("d", dtype="float32", shape=x0.shape)
    loss = SINGLE_PRECISION[objective](X)
    assert loss.dtype == "float32"
    [gradient] = pt.grad(loss, [X])
    search = zoom_line_search(max_steps, **arguments)
    result = search_along(loss, [X], [gradient], [D], search, guess=guess)
    x0 = x0.astype("float32")
    direction = direction_of(pytensor.function([X], gradient)(x0)).astype("float32")
    mode = Mode(linker=MLXLinker(use_compile=use_compile), optimizer="fast_run")

    with mx.stream(device):
        on_mlx = pytensor.function([X, D], list(result), mode=mode)(x0, direction)
    on_default = pytensor.function([X, D], list(result))(x0, direction)

    np.testing.assert_allclose(float(on_mlx[0]), float(on_default[0]), rtol=1e-4)
    assert (bool(on_mlx[1]), int(on_mlx[2])) == (bool(on_default[1]), int(on_default[2]))


def test_a_double_precision_search_takes_the_default_backends_step():
    """Declared in float64, the search keeps to the step graph, which mlx runs at the precision of the
    device."""
    (objective, x0, direction_of, max_steps), _ = PINNED["zooms_into_a_bracket"]
    X = pt.vector("x", dtype="float64", shape=x0.shape)
    D = pt.vector("d", dtype="float64", shape=x0.shape)
    loss = OBJECTIVES[objective](X)
    [gradient] = pt.grad(loss, [X])
    result = search_along(loss, [X], [gradient], [D], zoom_line_search(max_steps))
    direction = direction_of(pytensor.function([X], gradient)(x0))
    mode = Mode(linker=MLXLinker(), optimizer="fast_run")

    on_mlx = pytensor.function([X, D], list(result), mode=mode)(x0, direction)
    on_default = pytensor.function([X, D], list(result))(x0, direction)

    np.testing.assert_allclose(float(on_mlx[0]), float(on_default[0]), rtol=1e-4)
    assert (bool(on_mlx[1]), int(on_mlx[2])) == (bool(on_default[1]), int(on_default[2]))


def test_the_dispatch_runs_the_pieces_the_compile_mode_rewrote():
    """The pieces are found in the graph the linker rewrote, so they follow the mode the function was
    compiled with, rather than being rewritten again under a default of the dispatch's choosing."""
    X = pt.vector("x", dtype="float32", shape=(2,))
    D = pt.vector("d", dtype="float32", shape=(2,))
    loss = pt.sum((X - 3.0) ** 2) / 2
    result = search_along(loss, [X], [pt.grad(loss, X)], [D], zoom_line_search())
    mode = Mode(linker=MLXLinker(), optimizer="fast_run")

    def step_nodes_under(mode):
        compiled = pytensor.function([X, D], list(result), mode=mode)
        [search] = [
            node.op
            for node in compiled.maker.fgraph.apply_nodes
            if isinstance(node.op, LineSearchOp)
        ]
        return len(find_piece(search, STEP).fgraph.apply_nodes)

    assert step_nodes_under(mode) < step_nodes_under(mode.excluding("fusion"))
