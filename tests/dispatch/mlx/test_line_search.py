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
from tests.optim.test_line_search import OBJECTIVES, PINNED


@pytest.mark.parametrize("use_compile", [True, False], ids=["compiled", "eager"])
@pytest.mark.parametrize(
    "name", ["accepts_the_guess", "zooms_into_a_bracket", "backs_out_of_a_nan_region"]
)
def test_the_search_takes_the_default_backends_step(name, use_compile):
    """mlx runs every trial the search allows and holds the state once one is accepted, so it lands on
    the same step, after the same number of trials, as the `scan` that stops there."""
    (objective, x0, direction_of, max_steps), _ = PINNED[name]
    # Static shapes, as parameters have: a dynamic one adds broadcast checks mlx would drop with a warning
    X = pt.vector("x", dtype="float32", shape=x0.shape)
    D = pt.vector("d", dtype="float32", shape=x0.shape)
    # Single precision throughout, which is where the search runs as one Metal kernel per trial; the
    # barrier's 0.9 would otherwise promote its graph to float64
    objective_of = {
        **OBJECTIVES,
        "log_barrier": lambda x: pt.sum((x - np.float32(0.9)) ** 2 - pt.log(1 - x)),
    }
    loss = objective_of[objective](X)
    assert loss.dtype == "float32"
    [gradient] = pt.grad(loss, [X])
    result = search_along(loss, [X], [gradient], [D], zoom_line_search(max_steps))
    x0 = x0.astype("float32")
    direction = direction_of(pytensor.function([X], gradient)(x0)).astype("float32")
    mode = Mode(linker=MLXLinker(use_compile=use_compile), optimizer="fast_run")

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
