from typing import cast

import pytensor

from pymc.pytensorf import SeedSequenceSeed, collect_default_updates, reseed_rngs
from pytensor import Mode
from pytensor.compile import Function, SharedVariable, get_mode
from pytensor.compile.builders import OpFromGraph
from pytensor.graph import FunctionGraph, RewriteDatabaseQuery, rewrite_graph
from pytensor.tensor.variable import Variable


class LayerOp(OpFromGraph):
    # This can be removed once https://github.com/pymc-devs/pytensor/issues/1114 is fixed

    __props__ = ()

    def __init__(self, *args, **kwargs):
        prop_kwargs = {key: value for key, value in kwargs.items() if key in self.__props__}
        kwargs = {key: value for key, value in kwargs.items() if key not in self.__props__}

        for key, value in prop_kwargs.items():
            setattr(self, key, value)

        super().__init__(*args, **kwargs)

    def update_map(self) -> dict[int, int]:
        """Return a mapping of output indexes to input indexes"""
        return {}


def atleast_list(x):
    if not isinstance(x, list | tuple):
        return [x]
    return x


def function(
    inputs,
    outputs,
    random_seed: SeedSequenceSeed = None,
    mode=None,
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

    mode = get_mode(mode)
    opt_qry = mode.provided_optimizer.including("random_make_inplace")
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
    return rewrite_graph(graph, include=("canonicalize", "stabilize"))


def rewrite_for_prediction(graph):
    """Apply rewrites to specialize a graph for forward passes (e.g. removing Dropout layers)"""
    from pytensor_ml.rewriting.layers import predict_db

    if isinstance(graph, FunctionGraph):
        fgraph = graph
    else:
        outputs = [graph] if isinstance(graph, Variable) else graph
        fgraph = FunctionGraph(outputs=outputs, clone=True, copy_inputs=False)

    rewriter = predict_db.query(RewriteDatabaseQuery(include=["basic"]))
    rewriter.rewrite(fgraph)

    if isinstance(graph, FunctionGraph):
        return fgraph
    if isinstance(graph, Variable):
        return fgraph.outputs[0]

    return fgraph.outputs


__all__ = ["function"]
