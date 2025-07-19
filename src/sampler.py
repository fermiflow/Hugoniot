import jax
import jax.numpy as jnp
from functools import partial

from src.condition import make_cond_prob

def make_mask(m, n, state_idx):
    """
        Make mask for state_idx with shape (n,) in [0, num_states). (not vmaped yet)
        The possible state marked as 1 and the forbiden state marked as 0.
        The mask is used to make sure get the right state_idx:
            1. Arrange from small to large.
            2. Range [0, num_states).
            3. Not repeat.
    """
    mask = jnp.tril(jnp.ones((n, m)), k=m-n)
    idx_lb = jnp.concatenate( (jnp.array([-1]), state_idx[:-1]) )
    mask = jnp.where(jnp.arange(m) > idx_lb[:, None], mask, 0.)
    return mask

def make_autoregressive_sampler(network, n, num_states, beta):

    def _logits(params, state_idx):
        """
            Given occupation state indices `state_idx` of the electrons, compute
        the masked logits for the family of autoregressive conditional probabilities.

        Relevant dimensions: (before vmapped)

        INPUT:
            state_idx: (n,), with elements being integers in [0, num_states).
        OUTPUT:
            masked_logits: (n, num_states)
        """
        
        logits = network.apply(params, None, state_idx.astype(jnp.float64).reshape(n, 1))
        mask_up = make_mask(num_states, n//2, state_idx[:n//2])
        mask_dn = make_mask(num_states, n//2, state_idx[n//2:])
        mask = jnp.concatenate([mask_up, mask_dn])
        masked_logits = jnp.where(mask, logits, -jnp.inf)
        return masked_logits

    def sampler(params, key, batch, bands):
        """
            Sample the state, return state_idx with shape (batch, n) in [0, num_states).
            Input:
                params: the parameters of the autoregressive model.
                key: the random key.
                batch: int, batchsize.
                bands: array of shape (batch, n_mo), the band structure of the system.
            Output:
                state_idx: array of shape (batch, n), the sampled state indices.
        """
        state_idx = jnp.zeros((batch, n), dtype=jnp.int32)
        _, single_cond_logp_fn_novmap = make_cond_prob(beta, n//2)
        single_cond_logp_fn = jax.vmap(single_cond_logp_fn_novmap, (0, None, 0), 0)
        for i in range(n):
            key, subkey = jax.random.split(key)
            # logits.shape: (batch, n, num_states)
            logits = jax.vmap(_logits, (None, 0), 0)(params, state_idx)
            cond_logits = single_cond_logp_fn(state_idx[:, i-1], n//2-i%(n//2), bands)
            state = jax.random.categorical(subkey, logits[:, i, :]+cond_logits, axis=-1)
            state_idx = state_idx.at[:, i].set(state)
        return state_idx

    def log_prob(params, sample, bands):
        """
            Return the log(probability) of "sample", which is state_idx with shape 
            (n,) in [0, num_states) (not vmaped yet), from autoregressive model.
            Input:
                params: the parameters of the autoregressive model.
                sample: array of shape (n,), the state indices (state_idx in main).
                bands: array of shape (n_mo,), the band structure of the system.
            Output:
                logp: scalar, the log(probability) of the sample.
        """
        cond_logp_fn, _ = make_cond_prob(beta, n//2)
        cond_logits_up = cond_logp_fn(sample[:n//2], bands) # (n//2, num_states)
        cond_logits_dn = cond_logp_fn(sample[n//2:], bands) # (n//2, num_states)
        cond_logits = jnp.vstack((cond_logits_up, cond_logits_dn)) # (n, num_states)
        logits = _logits(params, sample) + cond_logits
        logp = jax.nn.log_softmax(logits, axis=-1)
        logp = logp[jnp.arange(n), sample].sum()
        return logp

    return sampler, log_prob

"""
    Classical score function: params, sample -> score
    This function can be useful for stochastic reconfiguration, the second-order
optimization algorithm based on classical Fisher information matrix.

Relevant dimension: (after vmapped)

INPUT:
    params: a pytree sample: (batch, n), with elements being integers in [0, num_states).
OUTPUT:
    a pytree of the same structure as `params`, in which each leaf node has
an additional leading batch dimension.
"""
make_classical_score = lambda log_prob: jax.vmap(jax.grad(log_prob), (None, 0), 0)
make_classical_score_van = lambda log_prob: jax.vmap(jax.grad(log_prob), (None, 0, 0), 0)

if __name__ == "__main__":
    from jax.config import config
    config.update("jax_enable_x64", True)

    n, m = 4, 10
    state_idx = jnp.array([1, 4, 5, 7])
    mask = make_mask(m, n, state_idx)
    """
        For this example, the resulting mask is illustrated as follows:

                possible state indices
        0   1   2   3   4   5   6   7   8   9
        -------------------------------------
     1  *   *   *   *   *   *   *   0   0   0   1hat
     2  0   0   *   *   *   *   *   *   0   0   2hat(1)
     3  0   0   0   0   0   *   *   *   *   0   3hat(1, 2)
     4  0   0   0   0   0   0   *   *   *   *   4hat(1, 2, 3)

        The symbols "*" and "0" stand for allowed and prohibited states, respectively.
    """
    print("mask:\n", mask)
