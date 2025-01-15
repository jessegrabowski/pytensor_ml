import pytensor

from pymc.model.fgraph import ModelNamed, fgraph_from_model, model_from_fgraph
from pymc.pytensorf import toposort_replace


def shared_vars_to_explicit_inputs(model):
    """
    Replace shared variables in a model with explicit inputs. Typically, these will be pm.Data containers.

    Parameters
    ----------
    model: pm.Model
        PyMC model to be transformed

    Returns
    -------
    pm.Model
        New model with shared variables replaced by
    """

    fg, memo = fgraph_from_model(model)
    replacements = {}
    for output in fg.outputs:
        if isinstance(output.owner.op, ModelNamed):
            var = output.owner.inputs[0]
            replacements[var] = var.type(name=var.name)
    toposort_replace(fg, replacements=tuple(replacements.items()))
    new_model = model_from_fgraph(fg)

    assert all(
        not isinstance(x, pytensor.compile.SharedVariable | pytensor.graph.Constant)
        for x in new_model.data_vars
    )

    return new_model
