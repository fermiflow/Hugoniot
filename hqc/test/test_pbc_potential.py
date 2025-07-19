import jax
import numpy as np
import jax.numpy as jnp
from pyscf.pbc import gto, scf
jax.config.update("jax_enable_x64", True)

from hqc.basis.parse import load_as_str
from hqc.pbc.potential import potential_energy_pp

def pyscf_vpp(xp, L, rs):
    """
        Pyscf Hartree Fock solver for hydrogen.
    INPUT:
        n: number of protons.
        L: side length of unit cell, unit: rs.
        rs: average atomic spacing, unit: Bohr
        sigma: smearing width, unit: Hartree.
        xp: array of shape (n, dim), position of protons.
    OUTPUT:
        vpp: float, proton-proton potential energy.
    """
    n = xp.shape[0]
    basis = 'gth-szv'
    Ry = 2
    xp *= rs
    cell = gto.Cell()
    cell.unit = 'B'
    cell.a = np.eye(3) * L * rs
    cell.atom = []
    for ie in range(n):
        cell.atom.append(['H', tuple(xp[ie])])
    cell.spin = 0
    cell.basis = {'H':gto.parse(load_as_str('H', basis), optimize=True)}
    cell.symmetry = False
    cell.build()

    kmf = scf.hf.RHF(cell)

    return kmf.energy_nuc() * Ry

def test_pbc_potential():
    n = 14
    rs = 1.31
    L = (4/3*jnp.pi*n)**(1/3)

    key = jax.random.PRNGKey(42)
    xp = jax.random.uniform(key, (n, 3), minval=0., maxval=L)

    vpp_pyscf = pyscf_vpp(xp, L, rs)
    print("vpp_pyscf:", vpp_pyscf)
    vpp = potential_energy_pp(xp, L, rs)
    print("vpp:", vpp)
    assert jnp.isclose(vpp, vpp_pyscf, atol=1e-6)
