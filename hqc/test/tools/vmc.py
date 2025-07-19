import jax
import jax.numpy as jnp

from tools.mcmc import mcmc, mcmc_ebes

def sample_x_mcmc(key, xp, xe, logpsi2, mo_coeff, mc_steps, mc_width, L):
    """
        Sample electron coordinates from the ground state wavefunction.
    """
    key, key_mcmc = jax.random.split(key)
    logpsi2_mcmc_novmap = lambda x: logpsi2(x, xp, mo_coeff)
    logpsi2_mcmc = jax.vmap(logpsi2_mcmc_novmap)
    xe, acc = mcmc(logpsi2_mcmc, xe, key_mcmc, mc_steps, mc_width)
    xe -= L * jnp.floor(xe/L)
    return key, xe, acc