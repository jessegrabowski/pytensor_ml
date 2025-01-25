from typing import cast

from pytensor.compile import optdb
from pytensor.graph.basic import Apply
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import in2out, node_rewriter
from pytensor.tensor.rewriting.ofg import inline_ofg_node
from pytensor.tensor.variable import Variable

from pytensor_ml.layers import DropoutLayer, LinearLayer


@node_rewriter([LinearLayer, DropoutLayer])
def inline_layers(fgraph: FunctionGraph, node: Apply) -> list[Variable] | None:
    """Inline einsums that are already optimized.

    This allows the inner garph to be optimized with the rest of the graph, now that we got ordering right.
    """

    return cast(list[Variable], inline_ofg_node(node))


optdb.register("inline_linear_layer", in2out(inline_layers), "inline_layers", position=100)


#
@node_rewriter([DropoutLayer])
def remove_dropout_for_prediction(fgraph: FunctionGraph, node: Apply) -> list[Variable] | None:
    """
    Set dropout probability to zero for all dropout layers.

    Parameters
    ----------
    fgraph: FunctionGraph
        Graph being rewritten
    node: Node
        Node being rewritten

    Returns
    -------
    X: Variable
        The input to the dropout layer, removing the dropout from the graph
    """
    X, rng = node.inputs
    return [X, None]


optdb.register(
    "remove_dropout_for_prediction",
    in2out(remove_dropout_for_prediction),
    "prediction",
    position=80,
)
