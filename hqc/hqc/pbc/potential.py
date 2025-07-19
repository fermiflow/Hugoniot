import jax
import numpy as np
import jax.numpy as jnp
from functools import partial

def kpoints(dim, Gmax):
    """
        Compute all the integer k-mesh indices (n_1, ..., n_dim) in spatial
    dimention `dim`, whose length do not exceed `Gmax`.
    """
    n = np.arange(-Gmax, Gmax+1)
    nis = np.meshgrid(*( [n]*dim ))
    G = np.array([ni.flatten() for ni in nis]).T
    G2 = (G**2).sum(axis=-1)
    G = G[(G2<=Gmax**2) * (G2>0)]
    return jnp.array(G)

def Madelung(dim, kappa, G):
    """
        The Madelung constant of a simple cubic lattice of lattice constant L=1
    in spatial dimension `dim`, namely the electrostatic potential experienced by
    the unit charge at a lattice site.
    """
    Gnorm = jnp.linalg.norm(G, axis=-1)

    if dim == 3:
        g_k = jnp.exp(-jnp.pi**2 * Gnorm**2 / kappa**2) / (jnp.pi * Gnorm**2)
        g_0 = -jnp.pi / kappa**2
    elif dim == 2:
        g_k = jax.scipy.special.erfc(jnp.pi * Gnorm / kappa) / Gnorm
        g_0 = -2 * jnp.sqrt(jnp.pi) / kappa

    return g_k.sum() + g_0 - 2*kappa/jnp.sqrt(jnp.pi)

def psi(rij, kappa, G, forloop=True):
    """
        The electron coordinate-dependent part 1/2 sum_{i}sum_{j neq i} psi(r_i, r_j)
    of the electrostatic energy (per cell) for a periodic system of lattice constant L=1.
        NOTE: to account for the Madelung part `Vconst` returned by the function
    `Madelung`, add the term 0.5*n*Vconst.
    """
    dim = rij.shape[0]

    # Only the nearest neighbor is taken into account in the present implementation of real-space summation.
    dij = jnp.linalg.norm(rij, axis=-1)
    V_shortrange = (jax.scipy.special.erfc(kappa * dij) / dij)

    Gnorm = jnp.linalg.norm(G, axis=-1)

    if dim == 3:
        g_k = jnp.exp(-jnp.pi**2 * Gnorm**2 / kappa**2) / (jnp.pi * Gnorm**2)
        g_0 = -jnp.pi / kappa**2
    elif dim == 2:
        g_k = jax.scipy.special.erfc(jnp.pi * Gnorm / kappa) / Gnorm
        g_0 = -2 * jnp.sqrt(jnp.pi) / kappa

    if forloop:
        def _body_fun(i, val):
            return val + g_k[i] * jnp.cos(2*jnp.pi * jnp.sum(G[i]*rij))  
        V_longrange = jax.lax.fori_loop(0, G.shape[0], _body_fun, 0.0) + g_0
    
    else:
        V_longrange = ( g_k * jnp.cos(2*jnp.pi * jnp.sum(G*rij, axis=-1)) ).sum() \
                    + g_0 
     
    potential = V_shortrange + V_longrange
    return potential

@partial(jax.vmap, in_axes=(0, None, None, None, None), out_axes=0)
def potential_energy(x, kappa, G, L, rs):
    """
        Potential energy for a periodic box of size L, only the nontrivial
    coordinate-dependent part. Unit: Ry/rs^2.
        To account for the Madelung part `Vconst` returned by the function `Madelung`,
    add the term n*rs/L*Vconst. See also the docstring for function `psi`.

    INPUTS: 
        x: (n, dim) proton + electron coordinates
    """
    print("x.shape:", x.shape)
    n, dim = x.shape

    x -= L * jnp.floor(x/L)
    i, j = jnp.triu_indices(n, k=1)
    rij = ( (x[:, None, :] - x)[i, j] )/L
    rij -= jnp.rint(rij)
    
    Z = jnp.concatenate([jnp.ones(n//2), -jnp.ones(n//2)])

    #Zij = (Z[:, None] * Z)[i,j]
    # return 2*rs/L * jnp.sum( Zij * jax.vmap(psi, (0, None, None), 0)(rij, kappa, G) )

    total_charge = (Z[:, None]+Z )[i, j]

    v = jax.vmap(psi, (0, None, None), 0)(rij, kappa, G)

    v_pp = jnp.sum(jnp.where(total_charge==2, v, jnp.zeros_like(v)))
    v_ep = -jnp.sum(jnp.where(total_charge==0, v, jnp.zeros_like(v)))
    v_ee = jnp.sum(jnp.where(total_charge==-2, v, jnp.zeros_like(v)))

    return 2*rs/L*v_pp , 2*rs/L * v_ep , 2*rs/L*v_ee

def potential_energy_pp(xp, L, rs, kappa=10, Gmax=15):
    """
        Potential energy for a periodic box of size L, only the nontrivial
    coordinate-dependent part. Unit: Ry/rs^2.
        To account for the Madelung part `Vconst` returned by the function `Madelung`,
    add the term n*rs/L*Vconst. See also the docstring for function `psi`.

    Inputs: 
        xp: (n, dim) proton coordinates
        L: float, side length of unit cell, unit: rs.
        rs: float, Wigner-Seitz radius
        kappa: float, screening parameter
        Gmax: int, cutoff of G-vectors
    Returns:
        v_pp: float, potential energy of protons, unit: Ry.
    """
    n, dim = xp.shape
    xp -= L * jnp.floor(xp/L)
    i, j = jnp.triu_indices(n, k=1)
    rij = ( (xp[:, None, :] - xp)[i, j] )/L
    rij -= jnp.rint(rij)

    G = kpoints(3, Gmax)
    v = jax.vmap(psi, (0, None, None), 0)(rij, kappa, G)
    v_pp = jnp.sum(v)
    Vconst = n * Madelung(3, kappa, G)/L/rs

    return 2*v_pp/L/rs + Vconst

if __name__ == '__main__':
    n = 14
    rs = 1.31
    L = (4/3*jnp.pi*n)**(1/3)
    
    Gmax = 15
    kappa = 10
    G = kpoints(3, Gmax)
    Vconst = n * rs/L * Madelung(3, kappa, G)

    key = jax.random.PRNGKey(42)
    key_p, key_e = jax.random.split(key)
    xp = jax.random.uniform(key_p, (n, 3), minval=0., maxval=L)
    xe = jax.random.uniform(key_e, (n, 3), minval=0., maxval=L)
    print("xp.shape:", xp.shape)
    print("xe.shape:", xe.shape)

    x = jnp.array([jnp.concatenate([xp, xe], axis=0)])
    print("x.shape:", x.shape)

    vpp, vep, vee = potential_energy(x, kappa, G, L, rs)
    vpp += Vconst
    vee += Vconst
    vpp = vpp/rs**2
    vep = vep/rs**2
    vee = vee/rs**2
    print("potential_energy:", vpp, vep, vee)

    vpp = potential_energy_pp(xp, L, rs)
    print("potential_energy_pp:", vpp)
    