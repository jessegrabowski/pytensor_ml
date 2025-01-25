from typing import cast

import pytensor

from pymc.pytensorf import SeedSequenceSeed, collect_default_updates, reseed_rngs
from pytensor import Mode
from pytensor.compile import Function, SharedVariable, get_mode
from pytensor.graph import rewrite_graph


def atleast_list(x):
    if not isinstance(x, list | tuple):
        return [x]
    return x


# Alias compile to function to match vanilla pytensor
def function(
    inputs,
    outputs,
    random_seed: SeedSequenceSeed = None,
    mode=None,
    include_prediction_rewrites: bool = False,
    **kwargs,
) -> Function:
    """
    Compile a Pytensor function, including specialized rewrites.

    Parameters
    ----------
    inputs: list of TensorVariables, optional
        Inputs of the compiled PyTensor function
    outputs: list of TensorVariables, optional
        Outputs of the compiled PyTensor function
    random_seed: int, array-like of int or SeedSequence, optional
        Seed used to override any RandomState/Generator shared variables in the graph.
        If not specified, the value of original shared variables will still be overwritten.
    include_prediction_rewrites: bool, optional
        If True, include additional rewrites specific to making predicitons
    mode: optional
        PyTensor mode used to compile the function

    Returns
    -------
    pytensor_function: Function
        Compiled function
    """
    rng_updates = collect_default_updates(
        inputs=[inp.variable if isinstance(inp, pytensor.In) else inp for inp in inputs],
        outputs=[
            out.variable if isinstance(out, pytensor.Out) else out for out in atleast_list(outputs)
        ],
    )

    if rng_updates:
        rngs = cast(list[SharedVariable], list(rng_updates))
        reseed_rngs(rngs, random_seed)

    rewrites = ["random_make_inplace"]
    if include_prediction_rewrites:
        rewrites.append("prediction")

    mode = get_mode(mode)
    opt_qry = mode.provided_optimizer.including(*rewrites)
    mode = Mode(linker=mode.linker, optimizer=opt_qry)
    pytensor_function = pytensor.function(
        inputs,
        outputs,
        updates={**rng_updates, **kwargs.pop("updates", {})},
        mode=mode,
        **kwargs,
    )
    return pytensor_function


def rewrite_pregrad(graph):
    """Apply simplifying or stabilizing rewrites to graph that are safe to use pre-grad."""
    # TODO: inline_layers is a fix for https://github.com/pymc-devs/pymc/issues/7657
    return rewrite_graph(graph, include=("canonicalize", "stabilize", "inline_layers"))


__all__ = ["function"]
