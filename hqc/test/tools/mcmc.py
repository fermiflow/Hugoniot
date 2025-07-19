import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnums=(0, 3))
def mcmc(logp_fn, x, key, mc_steps, mc_width):
    """
        MCMC sampling of x from logp_fn.
        x has shape (batch, n, dim).
    """
    def step(i, state):
        x, logp, key, num_accepts = state
        key, key_proposal, key_accept = jax.random.split(key, 3)
        
        x_proposal = x + mc_width * jax.random.normal(key_proposal, x.shape)
        logp_proposal = logp_fn(x_proposal)

        ratio = jnp.exp((logp_proposal - logp))
        accept = jax.random.uniform(key_accept, ratio.shape) < ratio

        x_new = jnp.where(accept[:, None, None], x_proposal, x)
        logp_new = jnp.where(accept, logp_proposal, logp)
        num_accepts += accept.sum()

        return x_new, logp_new, key, num_accepts

    logp_init = logp_fn(x)
    x, logp, key, num_accepts = jax.lax.fori_loop(0, mc_steps, step, (x, logp_init, key, 0.))
    batchsize = x.shape[0]
    accept_rate = num_accepts / (mc_steps * batchsize) 
    return x, accept_rate

@partial(jax.jit, static_argnums=0)
def mcmc_ebes(logp_fn, x_init, key, mc_steps, mc_width,
              logp_init=None):
    """
        Markov Chain Monte Carlo (MCMC) sampling algorithm
        with electron-by-electron sampling (EBES).

    INPUT:
        logp_fn: callable that evaluate log-probability of a batch of configuration x.
            The signature is logp_fn(x), where x has shape (..., n, dim).
        x_init: initial value of x, with shape (..., n, dim).
        key: initial PRNG key.
        mc_steps: total number of Monte Carlo steps.
        mc_width: size of the Monte Carlo proposal.

        logp_init: initial logp (...,)

    OUTPUT:
        x: resulting batch samples, with the same shape as `x_init`.
    """

    def single_step(ii, state):
        x, logp, key, accept_rate = state
        key, key_proposal, key_accept = jax.random.split(key, 3)

        batchshape = x.shape[:-2]
        dim = x.shape[-1]
        x_move = jax.random.normal(key_proposal, (*batchshape, dim))
        x_proposal = x.at[..., ii, :].add(mc_width * x_move)
        logp_proposal = logp_fn(x_proposal)  # batchshape

        ratio = jnp.exp((logp_proposal - logp))
        accept = jax.random.uniform(key_accept, ratio.shape) < ratio
        accept_rate += accept.mean()

        x_new = jnp.where(accept[..., None, None], x_proposal, x)
        logp_new = jnp.where(accept, logp_proposal, logp)

        return x_new, logp_new, key, accept_rate

    def step(i, state):
        x, logp, key, accept_rate = state

        n = x.shape[-2]
        x_new, logp_new, key, accept_rate = jax.lax.fori_loop(0, n, single_step,
                                                                           (x, logp, key, accept_rate))
        return x_new, logp_new, key, accept_rate 

    if logp_init is None:
        logp_init = logp_fn(x_init)

    x, logp, key, accept_rate = jax.lax.fori_loop(0, mc_steps, step, (x_init, logp_init, key, 0.))
    n = x.shape[-2]
    accept_rate /= mc_steps * n
    return x, accept_rate