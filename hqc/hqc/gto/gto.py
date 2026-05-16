"""
GTO (Gaussian Type Orbital) evaluation for non-periodic systems.

This module handles evaluation of atomic orbitals at specific positions,
used for wavefunction reconstruction and QMC applications.
"""

import jax
import jax.numpy as jnp

const = (2 / jnp.pi)**0.75
coeff_sto3g = jnp.array([[3.42525091, 0.15432897],
                        [0.62391373, 0.53532814],
                        [0.16885540, 0.44463454]])
coeff_sto6g = jnp.array([[35.52322122, 0.00916359628],
                        [6.513143725, 0.04936149294],
                        [1.822142904, 0.16853830490],
                        [0.625955266, 0.37056279970],
                        [0.243076747, 0.41649152980],
                        [0.100112428, 0.13033408410]])

def make_ao(basis):
    """
    Make GTO atomic orbital evaluation function.

    Args:
        basis: str, GTO basis name (e.g., 'sto3g', 'sto6g')

    Returns:
        eval_gto: function that evaluates GTO orbitals at electron positions
    """
    if basis == 'sto3g':
        coeff = coeff_sto3g
    elif basis == 'sto6g':
        coeff = coeff_sto6g

    @jax.remat
    def eval_gto(xp, xe):
        """
        Evaluate GTO orbitals at electron position.

        Args:
            xp: array of shape (n, dim), proton positions
            xe: array of shape (dim,), single electron position

        Returns:
            gto: GTO orbital values at xe, shape (n_ao,)
        """
        r = jnp.sum(jnp.square(xe[None, :] - xp), axis=1)  # (n_p,)
        gto = const * jnp.einsum('i,i,i...->...', coeff[:, 1], jnp.power(coeff[:, 0], 0.75),
                jnp.exp(-jnp.einsum('i,...->i...', coeff[:, 0], r))).reshape(-1)  # (n_p,)
        return gto

    return eval_gto
