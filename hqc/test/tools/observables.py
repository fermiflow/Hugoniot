import jax
import jax.numpy as jnp
from functools import partial

from hqc.pbc.potential import kpoints, Madelung, potential_energy

@partial(jax.vmap, in_axes=(None, 0, None, None, None, None), out_axes=(0, 0, 0, 0, 0))
def observables(xp, xe, mo_coeff, n, rs, logpsi_grad_laplacian):
    L = (4/3*jnp.pi*n)**(1/3)
    
    Gmax = 15
    kappa = 10
    G = kpoints(3, Gmax)
    Vconst = n * rs/L * Madelung(3, kappa, G)

    vpp, vep, vee = potential_energy(jnp.array([jnp.concatenate([xp, xe], axis=0)]), kappa, G, L, rs)
    vpp += Vconst
    vee += Vconst

    grad, laplacian = logpsi_grad_laplacian(xe, xp, mo_coeff)
    kinetic = -laplacian - (grad**2).sum(axis=(-2, -1))

    Eloc = kinetic + vep + vee

    return Eloc.real, kinetic.real, vpp, vep, vee