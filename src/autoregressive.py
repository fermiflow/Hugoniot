"""
    Autoregressive model of momenta occupation with fixed particle number.

    The present implementation is based on a Transformer composed of causal self-attention layers.

    Adapted from https://github.com/deepmind/dm-haiku/blob/main/examples/transformer/model.py
"""

import jax
import jax.numpy as jnp
import haiku as hk

from typing import Optional

class CausalSelfAttention(hk.MultiHeadAttention):
    """Self attention with a causal mask applied."""
    
    def __call__(self,
                 query: jnp.ndarray,
                 key: Optional[jnp.ndarray] = None,
                 value: Optional[jnp.ndarray] = None,
                ) -> jnp.ndarray:

        key = key if key is not None else query
        value = value if value is not None else query

        seq_len = query.shape[0]
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))[None, ...]

        return super().__call__(query, key, value, mask)

class DenseBlock(hk.Module):
    """A 2-layer MLP which widens then narrows the input."""

    def __init__(self,
                 hidden_size: int,
                 init_scale: float,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.hidden_size = hidden_size
        self.init_scale = init_scale

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        size = x.shape[-1]
        initializer = hk.initializers.VarianceScaling(self.init_scale)
        x = hk.Linear(self.hidden_size, w_init=initializer)(x)
        x = jnp.tanh(x)
        return hk.Linear(size, w_init=initializer)(x)

class Transformer(hk.Module):

    def __init__(self,
                 output_size: int,
                 num_layers: int, model_size: int, num_heads: int,
                 hidden_size: int,
                 remat: bool = False,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.output_size = output_size
        self.remat = remat

        self.num_layers, self.model_size, self.num_heads = \
                num_layers, model_size, num_heads
        if model_size % num_heads != 0:
            raise ValueError("Model_size of the transformer must be divisible "
                    "by the number of heads. Got model_size=%d and num_heads=%d."
                    % (model_size, num_heads))
        self.key_size = model_size // num_heads

        self.hidden_size = hidden_size

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        init_scale = 0.02 / self.num_layers

        x = hk.Linear(self.model_size,
                      w_init=hk.initializers.VarianceScaling(init_scale, "fan_out"),
                      name="embedding_mlp")(x)
        x = jnp.tanh(x)

        def block(x, init_scale, i):
            x_attn = CausalSelfAttention(self.num_heads,
                                            self.key_size,
                                            init_scale,
                                            name=f"layer{i}_attn")(x)
            x = x + x_attn
            x_dense = DenseBlock(self.hidden_size, init_scale, name=f"layer{i}_mlp")(x)
            x = x + x_dense
            return x
        
        for i in range(self.num_layers):
            x = block(x, init_scale, i)

        if self.remat:
            block = hk.remat(block, static_argnums=(1,2))

        x = jnp.tanh(x)
        x = hk.Linear(self.output_size,
                      w_init=hk.initializers.VarianceScaling(init_scale),
                      name="output_mlp")(x)

        x1init = hk.initializers.TruncatedNormal(jnp.sqrt(init_scale / self.output_size))
        x1hat = hk.get_parameter("x1hat", shape=(self.output_size,), init=x1init, dtype=jnp.float64)
        x = jnp.vstack((x1hat, x[:-1]))

        return x

if __name__ == "__main__":
    from jax.config import config
    config.update("jax_enable_x64", True) 

    num_states = 147
    nlayers = 2
    modelsize = 16
    nheads = 4
    nhidden = 32

    def forward_fn(state):
        model = Transformer(num_states, nlayers, modelsize, nheads, nhidden)
        return model(state)
    
    van = hk.transform(forward_fn)

    dim, n = 3, 14
    Emax = 10
    from orbitals import sp_orbitals
    sp_indices, Es = sp_orbitals(dim, Emax)
    sp_indices, Es = jnp.array(sp_indices), jnp.array(Es)
    sp_indices, Es = sp_indices[::-1], Es[::-1]
    state_idx_dummy = sp_indices[-n:].astype(jnp.float64)

    key = jax.random.PRNGKey(42)
    params_van = van.init(key, state_idx_dummy)

    from jax.flatten_util import ravel_pytree
    raveled_params_van, _ = ravel_pytree(params_van)
    print("#parameters in the autoregressive model: %d" % raveled_params_van.size)

    van()
