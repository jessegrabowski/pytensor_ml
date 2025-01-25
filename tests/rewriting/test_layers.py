import pytensor.tensor as pt
import pytest

from pytensor.graph.basic import explicit_graph_inputs
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.utils import rewrite_graph

from pytensor_ml.layers import Dropout, DropoutLayer, Linear, LinearLayer, Sequential


@pytest.fixture()
def feature_extractor_and_rng():
    d1 = Dropout("Dropout_1", p=0.5)
    d2 = Dropout("Dropout_2", p=0.5)
    feature_extractor = Sequential(
        Linear("Layer_1", n_in=6, n_out=3), d1, Linear("Layer_2", n_in=3, n_out=1), d2
    )

    # These won't be found by explicit_graph_inputs, but we need them as inputs to the fgraph later, so carry them
    # along
    rngs = [d1.rng, d2.rng]

    return feature_extractor, rngs


def test_remove_dropout(feature_extractor_and_rng):
    feature_extractor, rngs = feature_extractor_and_rng

    X = pt.tensor("X", shape=(None, 6))
    latent = feature_extractor(X)

    fg = FunctionGraph(inputs=list(explicit_graph_inputs(latent)) + rngs, outputs=[latent])

    assert len([node.op for node in fg.apply_nodes if isinstance(node.op, DropoutLayer)]) == 2
    fg = rewrite_graph(fg, include=("prediction",))

    assert len([node.op for node in fg.apply_nodes if isinstance(node.op, DropoutLayer)]) == 0


def test_inline_layers(feature_extractor_and_rng):
    feature_extractor, rngs = feature_extractor_and_rng

    X = pt.tensor("X", shape=(None, 6))
    latent = feature_extractor(X)

    fg = FunctionGraph(inputs=list(explicit_graph_inputs(latent)) + rngs, outputs=[latent])

    assert len([node.op for node in fg.apply_nodes if isinstance(node.op, DropoutLayer)]) == 2
    assert len([node.op for node in fg.apply_nodes if isinstance(node.op, LinearLayer)]) == 2

    fg = rewrite_graph(fg, include=("inline_layers",))

    assert len([node.op for node in fg.apply_nodes if isinstance(node.op, DropoutLayer)]) == 0
    assert len([node.op for node in fg.apply_nodes if isinstance(node.op, LinearLayer)]) == 0
