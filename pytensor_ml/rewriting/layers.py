from pytensor.graph.basic import Apply
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.graph.rewriting.db import EquilibriumDB
from pytensor.tensor.variable import Variable

from pytensor_ml.layers import DropoutLayer

predict_db = EquilibriumDB()


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
    return [X]


predict_db.register(
    "remove_dropout_for_prediction",
    remove_dropout_for_prediction,
    "basic",
)
