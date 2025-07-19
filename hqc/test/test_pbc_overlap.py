import jax
import numpy as np
import jax.numpy as jnp
from pyscf.pbc import gto, scf
from hqc.pbc.overlap import make_overlap
from hqc.basis.parse import load_as_str
jax.config.update("jax_enable_x64", True)

def pyscf_overlap(n, L, rs, xp, basis, kpt):
    """
        Pyscf overlap matrix.
    INPUT:
        n: number of protons.
        L: side length of unit cell, unit: rs.
        rs: average atomic spacing, unit: Bohr
        xp: array of shape (n, dim), position of protons.
        basis: gto basis name, eg:'gth-szv', 'gth-tzv2p', 'gth-qzv3p'.
        kpt: k-point, array of shape (3,).
    OUTPUT:
        overlap: overlap matrix, real or complex array of shape (n_ao, n_ao).
    """
    xp *= rs
    cell = gto.Cell()
    cell.unit = 'B'
    cell.a = np.eye(3) * L * rs
    cell.atom = []
    for ie in range(n):
        cell.atom.append(['H', tuple(xp[ie])])
    cell.spin = 0
    cell.basis = {'H':gto.parse(load_as_str('H', basis), optimize=True)}
    cell.build()

    kpts = [kpt.tolist()]
    kmf = scf.hf.RHF(cell)
    overlap = kmf.get_ovlp(kpt=kpts)

    return overlap

def test_overlap_gamma():
    """
        Test the hqc.pbc.overlap.make_overlap function at gamma point.
    """
    n, dim = 8, 3
    rs = 1.5
    rcut = 24
    cell = jnp.eye(3)
    basis_set = ['gth-szv', 'gth-dzv', 'gth-dzvp']
    L = (4/3*jnp.pi*n)**(1/3)*rs
    kpt = jnp.array([0, 0, 0])

    key = jax.random.PRNGKey(43)
    xp = jax.random.uniform(key, (n, dim), minval=0., maxval=L)

    print(print("\n============= begin test ============="))
    print("n:", n)
    print("rs:", rs)
    print("rcut:", rcut)
    print("basis_set:", basis_set)
    print("L:", L)
    print("xp:\n", xp)

    for basis in basis_set:
        
        print("\n==========", basis, "==========")
        # PBC pyscf benchmark
        overlap_func = make_overlap(n, L, rs, basis, rcut)
        overlap = overlap_func(xp)
        # print("overlap:\n", overlap)

        overlap_pyscf = pyscf_overlap(n, L, rs, xp, basis, kpt)
        # print("overlap_pyscf:\n", overlap_pyscf)
        assert np.allclose(overlap, overlap_pyscf, atol=1e-5)
        print("same as pyscf")

def test_overlap_kpt():
    """
        Test the hqc.pbc.overlap.make_overlap function at random kpt.
    """
    n, dim = 8, 3
    rs = 1.5
    rcut = 24
    cell = jnp.eye(3)
    basis_set = ['gth-szv', 'gth-dzv', 'gth-dzvp']
    L = (4/3*jnp.pi*n)**(1/3)*rs

    key = jax.random.PRNGKey(43)
    key_p, key_kpt = jax.random.split(key)
    xp = jax.random.uniform(key_p, (n, dim), minval=0., maxval=L)
    kpt = jax.random.uniform(key_kpt, (3,), minval=-jnp.pi/L, maxval=jnp.pi/L)

    print(print("\n============= begin test ============="))
    print("n:", n)
    print("rs:", rs)
    print("rcut:", rcut)
    print("basis_set:", basis_set)
    print("L:", L)
    print("xp:\n", xp)
    print("kpt:\n", kpt)

    for basis in basis_set:
        print("\n==========", basis, "==========")
        # PBC pyscf benchmark
        overlap_func = make_overlap(n, L, rs, basis, rcut, gamma=False)
        overlap = overlap_func(xp, kpt)
        # print("overlap:\n", overlap)

        overlap_pyscf = pyscf_overlap(n, L, rs, xp, basis, kpt)
        # print("overlap_pyscf:\n", overlap_pyscf)
        assert np.allclose(overlap, overlap_pyscf, atol=1e-5)
        print("same as pyscf")
    