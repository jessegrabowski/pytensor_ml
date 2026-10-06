import jax
import jax.numpy as jnp

from pytensor.link.jax.dispatch import jax_funcify

from pytensor_ml.layers.attention import AttentionLayer


@jax_funcify.register(AttentionLayer)
def jax_funcify_AttentionLayer(op, node=None, **kwargs):
    """Dispatch the attention marker to ``jax.nn.dot_product_attention`` (XLA/cuDNN flash kernel)."""
    is_causal = op.is_causal
    scale = op.scale

    def attention(q, k, v, mask=None):
        # jax's is_causal aligns the triangle top-left, so a query block shorter than its keys, as when
        # decoding from a cache, would see only the first keys. The layer aligns it bottom-right, so
        # every query sees the whole prefix before it; build that triangle here when the lengths differ.
        q_len, kv_len = q.shape[-2], k.shape[-2]
        causal, causal_mask = is_causal, None
        if is_causal and q_len != kv_len:
            rows = jnp.arange(q_len)[:, None]
            cols = jnp.arange(kv_len)[None, :]
            causal, causal_mask = False, (cols <= rows + (kv_len - q_len))[None, None]
        # jax expects (batch, seq, head, dim); our convention is (batch, head, seq, dim). The additive
        # mask is (batch, head, seq, seq) in both, so only q/k/v are transposed. jax combines the
        # additive bias with whichever triangle applies, its own or the boolean one built above.
        q = jnp.swapaxes(q, -3, -2)
        k = jnp.swapaxes(k, -3, -2)
        v = jnp.swapaxes(v, -3, -2)
        out = jax.nn.dot_product_attention(
            q, k, v, bias=mask, mask=causal_mask, scale=scale, is_causal=causal
        )
        return jnp.swapaxes(out, -3, -2)

    return attention
