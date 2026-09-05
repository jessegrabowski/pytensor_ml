import pytensor.tensor as pt

from pytensor.graph.basic import Variable
from pytensor.tensor.variable import TensorVariable

from pytensor_ml.activations import GELU, Activation, QuickGELU, ReLU
from pytensor_ml.layers import Embedding, LayerNorm, Linear, TransformerBlock
from pytensor_ml.models.keys import KeyMap, channels_last
from pytensor_ml.models.registry import register_builder

_ACTIVATIONS: dict[str, Activation] = {
    "quick_gelu": QuickGELU(),
    "gelu": GELU(approximate=False),
    "gelu_new": GELU(approximate=True),
    "relu": ReLU(),
}


def _activation(hidden_act: str) -> Activation:
    activation = _ACTIVATIONS.get(hidden_act)
    if activation is None:
        raise ValueError(
            f"CLIP's hidden_act is {hidden_act!r}, which has no activation here. Known: "
            f"{sorted(_ACTIVATIONS)}."
        )
    return activation


@register_builder("CLIPTextModel")
def build_clip_text_model(config: dict, keys: KeyMap) -> tuple[list[Variable], list[Variable]]:
    """
    Build a CLIP text encoder and record where each parameter loads from.

    Parameters
    ----------
    config : dict
        Parsed ``config.json`` of a ``CLIPTextModel``.
    keys : KeyMap
        Filled with the checkpoint key of every parameter built.

    Returns
    -------
    data_inputs : list of Variable
        The ``input_ids`` placeholder, of shape ``(batch, sequence)``.
    outputs : list of Variable
        The final hidden state, then one entry per layer. SDXL conditions on the second-to-last
        layer's output rather than the final one, so the per-layer states are outputs rather than
        internals.
    """
    ids, final, hidden_states = _build_text_transformer(config, keys)
    return [ids], [final, *hidden_states]


@register_builder("CLIPTextModelWithProjection")
def build_clip_text_model_with_projection(
    config: dict, keys: KeyMap
) -> tuple[list[Variable], list[Variable]]:
    """
    Build a CLIP text encoder with the projection head SDXL's pooled conditioning comes from.

    Parameters
    ----------
    config : dict
        Parsed ``config.json`` of a ``CLIPTextModelWithProjection``.
    keys : KeyMap
        Filled with the checkpoint key of every parameter built.

    Returns
    -------
    data_inputs : list of Variable
        The ``input_ids`` placeholder, of shape ``(batch, sequence)``.
    outputs : list of Variable
        The projected pooled embedding, the final hidden state, then one entry per layer. As for
        :func:`build_clip_text_model`, the second-to-last entry is the layer SDXL conditions on.
    """
    ids, final, hidden_states = _build_text_transformer(config, keys)

    projection = Linear(
        "text_projection",
        n_in=config["hidden_size"],
        n_out=config["projection_dim"],
        bias=False,
    )
    keys.bind(projection.W, "text_projection.weight", transform=channels_last)

    pooled = projection(_pool(ids, final, config))
    pooled.name = "text_embeds"
    return [ids], [pooled, final, *hidden_states]


def _pool(ids: TensorVariable, final: TensorVariable, config: dict) -> TensorVariable:
    """
    Take each sequence's hidden state at its end-of-stream token.

    CLIP configs written before huggingface/transformers#24773 carry ``eos_token_id: 2`` whatever the
    tokenizer actually uses, so the position is found by taking the largest id rather than by matching
    that value. SDXL's checkpoints are of that vintage.
    """
    if config.get("eos_token_id", 2) == 2:
        position = ids.argmax(axis=-1)
    else:
        position = pt.eq(ids, config["eos_token_id"]).astype("int8").argmax(axis=-1)
    return final[pt.arange(ids.shape[0]), position]


def _build_text_transformer(
    config: dict, keys: KeyMap
) -> tuple[TensorVariable, TensorVariable, list[TensorVariable]]:
    """Everything the two CLIP text architectures share, up to the final layer norm."""
    width = config["hidden_size"]
    n_layers = config["num_hidden_layers"]
    epsilon = config["layer_norm_eps"]

    mlp_ratio, remainder = divmod(config["intermediate_size"], width)
    if remainder:
        raise ValueError(
            f"CLIP's intermediate_size ({config['intermediate_size']}) is not a whole multiple of "
            f"hidden_size ({width}), which TransformerBlock cannot express."
        )

    ids = pt.matrix("input_ids", dtype="int32")

    token_embedding = Embedding(
        "token_embedding", n_embeddings=config["vocab_size"], n_features=width
    )
    position_embedding = Embedding(
        "position_embedding",
        n_embeddings=config["max_position_embeddings"],
        n_features=width,
    )
    blocks = [
        TransformerBlock(
            f"layer_{i}",
            d_model=width,
            n_head=config["num_attention_heads"],
            mlp_ratio=mlp_ratio,
            activation=_activation(config["hidden_act"]),
            is_causal=True,
            epsilon=epsilon,
        )
        for i in range(n_layers)
    ]
    final_layer_norm = LayerNorm("final_layer_norm", n_in=width, epsilon=epsilon)

    with keys.scope("text_model"):
        with keys.scope("embeddings"):
            keys.bind(token_embedding.W, "token_embedding.weight")
            keys.bind(position_embedding.W, "position_embedding.weight")
        _bind_layer_norm(keys, final_layer_norm, "final_layer_norm")
        with keys.scope("encoder", "layers"):
            for i, block in enumerate(blocks):
                with keys.scope(str(i)):
                    _bind_layer(keys, block)

    # Positions come from the input's length rather than the checkpoint's position_ids buffer, which
    # is arange(n) and is surplus for that reason.
    hidden = token_embedding(ids) + position_embedding(pt.arange(ids.shape[1]))

    hidden_states = []
    for block in blocks:
        hidden = block(hidden)
        hidden_states.append(hidden)

    final = final_layer_norm(hidden)
    final.name = "last_hidden_state"
    return ids, final, hidden_states


def _bind_layer_norm(keys: KeyMap, norm: LayerNorm, *parts: str) -> None:
    # CLIP's norms are affine, so scale and loc are built; the layer types them as optional because a
    # norm without an affine transform owns neither.
    assert norm.scale is not None and norm.loc is not None
    keys.bind(norm.scale, *parts, "weight")
    keys.bind(norm.loc, *parts, "bias")


def _bind_layer(keys: KeyMap, block: TransformerBlock) -> None:
    _bind_layer_norm(keys, block.norm1, "layer_norm1")
    _bind_layer_norm(keys, block.norm2, "layer_norm2")

    projections = [
        ("q_proj", block.attn.q_proj),
        ("k_proj", block.attn.k_proj),
        ("v_proj", block.attn.v_proj),
        ("out_proj", block.attn.out_proj),
    ]
    for name, projection in projections:
        keys.bind(projection.W, "self_attn", name, "weight", transform=channels_last)
        keys.bind(projection.b, "self_attn", name, "bias")

    for name, linear in [("fc1", block.ff.fc_in), ("fc2", block.ff.fc_out)]:
        keys.bind(linear.W, "mlp", name, "weight", transform=channels_last)
        keys.bind(linear.b, "mlp", name, "bias")
