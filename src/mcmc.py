import jax
import jax.numpy as jnp
from functools import partial

from src.utils import clipped_force_fn

@partial(jax.jit, static_argnums=0)
def mcmc(logp_fn, x_init, key, mc_steps, mc_width=0.02):
    """
        Markov Chain Monte Carlo sampling algorithm.

    INPUT:
        logp_fn: callable that evaluate log-probability of a batch of configuration x.
            The signature is logp_fn(x), where x has shape (batch, n, dim).
        x_init: initial value of x, with shape (batch, n, dim).
        key: initial PRNG key.
        mc_steps: total number of Monte Carlo steps.
        mc_width: size of the Monte Carlo proposal.

    OUTPUT:
        x: resulting batch samples, with the same shape as `x_init`.
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
    
    logp_init = logp_fn(x_init)

    x, logp, key, num_accepts = jax.lax.fori_loop(0, mc_steps, step, (x_init, logp_init, key, 0.))
    batch = x.shape[0]
    accept_rate = jax.lax.pmean(num_accepts / (mc_steps * batch), axis_name="p")
    return x, accept_rate

@partial(jax.jit, static_argnums=(0, 1))
def mala(logp_fn, force_fn, x_init, key, mc_steps, mc_width,
         logp_init=None, force_init=None):
    """
        Metropolis-adjusted Langevin algorithm (MALA).

    INPUT:
        logp_fn: callable that evaluate log-probability of a batch of configuration x.
            The signature is logp_fn(x), where x has shape (..., n, dim).
        force_fn: grad(logp_fn)
        x_init: initial value of x, with shape (..., n, dim).
        key: initial PRNG key.
        mc_steps: total number of Monte Carlo steps.
        mc_width: size of the Monte Carlo proposal.

        logp_init: initial logp (...,)
        force_init: initial force (..., n, dim)

    OUTPUT:
        x: resulting batch samples, with the same shape as `x_init`.
    """

    def step(i, state):
        x, logp, force, key, accept_rate = state
        key, key_proposal, key_accept = jax.random.split(key, 3)
        
        x_proposal = x + 0.5 * force * mc_width**2 + mc_width * jax.random.normal(key_proposal, x.shape)
        logp_proposal = logp_fn(x_proposal)
        force_proposal = clipped_force_fn(force_fn, 1/mc_width**2, x_proposal)

        diff = jnp.sum(0.5*(force + force_proposal)*((x - x_proposal) + mc_width**2/4*(force - force_proposal)), axis=(-1,-2))
        
        ratio = jnp.exp(diff + (logp_proposal - logp))
        accept = jax.random.uniform(key_accept, ratio.shape) < ratio

        x_new = jnp.where(accept[..., None, None], x_proposal, x)
        logp_new = jnp.where(accept, logp_proposal, logp)
        force_new = jnp.where(accept[..., None, None], force_proposal, force)
        accept_rate += accept.mean()
        return x_new, logp_new, force_new, key, accept_rate
    
    if logp_init is None:
        logp_init = logp_fn(x_init)
    if force_init is None:
        force_init = clipped_force_fn(force_fn, 1/mc_width**2, x_init)

    x, logp, force, key, accept_rate = jax.lax.fori_loop(0, mc_steps, step, (x_init, logp_init, force_init, key, 0.))
    accept_rate /= mc_steps

    return x, logp, force, key, accept_rate

def adjust_mc_width(mc_width, ar, mc_alg):
    if mc_alg == "mala":
        # https://en.wikipedia.org/wiki/Metropolis-adjusted_Langevin_algorithm
        if ar > 0.6: mc_width = mc_width * 1.1
        if ar < 0.5: mc_width = mc_width / 1.1
    elif mc_alg == "hmc":
        if ar > 0.95: mc_width = mc_width * 1.1
        if ar < 0.8: mc_width = mc_width / 1.1
    elif mc_alg == "mcmc":
        # https://github.com/google-deepmind/ferminet/blob/main/ferminet/train.py#L731
        if ar > 0.55: mc_width = mc_width * 1.1
        if ar < 0.5: mc_width = mc_width / 1.1
    else:
        raise NotImplementedError
    return mc_width
