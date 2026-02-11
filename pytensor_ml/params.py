from collections.abc import Sequence

from pytensor.compile.sharedvalue import SharedVariable
from pytensor.graph import graph_inputs
from pytensor.graph.basic import Constant, Variable
from pytensor.graph.traversal import ancestors
from pytensor.tensor import TensorVariable

from pytensor_ml.pytensorf import LayerOp


def collect_graph_inputs(
    outputs: TensorVariable | Sequence[TensorVariable],
) -> list[TensorVariable]:
    """
    Collect all non-constant, non-shared graph inputs.

    Parameters
    ----------
    outputs : TensorVariable or Sequence of TensorVariable
        One or more graph outputs to trace back from.

    Returns
    -------
    graph_inputs : list of TensorVariable
        All graph inputs that are not Constants or SharedVariables.
    """
    if isinstance(outputs, Variable):
        outputs = [outputs]

    return list(
        filter(
            lambda var: not isinstance(var, Constant | SharedVariable), graph_inputs(list(outputs))
        )
    )


def collect_trainable_params(
    outputs: TensorVariable | Sequence[TensorVariable],
    exclude: set[Variable] | None = None,
) -> list[TensorVariable]:
    """
    Extract trainable parameters from a computation graph.

    Parameters
    ----------
    outputs : TensorVariable or Sequence of TensorVariable
        One or more graph outputs to trace back from.
    exclude : Set of Variable, optional
        Set of variables to exclude from the result (e.g., data inputs).

    Returns
    -------
    trainable_params : list of TensorVariable
        Trainable parameters (graph inputs that are not in exclude and not declared as non-trainable by
        LayerOp.update_map).
    """
    if isinstance(outputs, Variable):
        outputs = [outputs]

    exclude = set(exclude) if exclude else set()

    for ancestor in ancestors(list(outputs)):
        node = ancestor.owner
        if node is not None and isinstance(node.op, LayerOp):
            for input_idx in node.op.update_map().values():
                exclude.add(node.inputs[input_idx])

    return [var for var in collect_graph_inputs(outputs) if var not in exclude]


def collect_non_trainable_updates(
    outputs: TensorVariable | Sequence[TensorVariable],
) -> dict[TensorVariable, TensorVariable]:
    """
    Extract non-trainable update pairs from LayerOps.

    These are state variables that need to be updated during training but are not subject to gradient-based
    optimization (e.g., running mean/variance in batch normalization).

    Parameters
    ----------
    outputs
        One or more graph outputs to trace back from.

    Returns
    -------
    non_trainable_updates : dict
        Mapping from old variable to new variable for all updates declared by LayerOp.update_map().
    """
    if isinstance(outputs, Variable):
        outputs = [outputs]

    updates = {}
    for ancestor in ancestors(list(outputs)):
        node = ancestor.owner
        if node is not None and isinstance(node.op, LayerOp):
            for output_idx, input_idx in node.op.update_map().items():
                old_value = node.inputs[input_idx]
                new_value = node.outputs[output_idx]
                updates[old_value] = new_value

    return updates


def collect_data_inputs(
    outputs: TensorVariable | Sequence[TensorVariable],
    params: Sequence[TensorVariable] | None = None,
) -> list[TensorVariable]:
    """
    Extract data inputs from a graph (inputs that are not parameters).

    This function identifies which graph inputs are "data" (like X, y_true) versus "parameters" (learnable weights). It
    uses a heuristic: inputs that flow into LayerOps at trainable positions are parameters, others are data.

    Parameters
    ----------
    outputs : TensorVariable or Sequence of TensorVariable
        One or more graph outputs to trace back from.
    params : Sequence of TensorVariable, optional
        Known parameters to exclude. If None, will be inferred by identifying inputs that feed into LayerOp weight
        positions.

    Returns
    -------
    data_inputs : list of TensorVariable
        Graph inputs that are not trainable parameters or non-trainable state.
    """
    if isinstance(outputs, Variable):
        outputs = [outputs]

    all_inputs = collect_graph_inputs(outputs)

    if params is None:
        params = _infer_params_from_layer_ops(outputs, all_inputs)

    non_trainable = set(collect_non_trainable_updates(outputs).keys())
    param_set = set(params)

    return [var for var in all_inputs if var not in param_set and var not in non_trainable]


def _infer_params_from_layer_ops(
    outputs: Sequence[TensorVariable],
    all_inputs: list[TensorVariable],
) -> list[TensorVariable]:
    """
    Infer which inputs are parameters by tracing which inputs flow into LayerOps.

    An input is considered a parameter if it flows into a LayerOp at a position that is not a data input (position 0
    is typically data, others are weights).

    Parameters
    ----------
    outputs : Sequence of TensorVariable
        Graph outputs to trace back from.
    all_inputs : list of TensorVariable
        All graph inputs to consider.

    Returns
    -------
    param_list : list of TensorVariable
        Inputs that are inferred to be parameters based on their connection to LayerOps.
    """
    data_inputs: set[Variable] = set()
    non_trainable: set[Variable] = set()

    for ancestor in ancestors(list(outputs)):
        node = ancestor.owner
        if node is not None and isinstance(node.op, LayerOp):
            # First input to a LayerOp is typically data
            first_input = node.inputs[0]
            if first_input in all_inputs:
                data_inputs.add(first_input)

            # Inputs declared in update_map are non-trainable state
            for input_idx in node.op.update_map().values():
                non_trainable.add(node.inputs[input_idx])

    # Parameters are inputs that are neither data nor non-trainable
    return [var for var in all_inputs if var not in data_inputs and var not in non_trainable]
