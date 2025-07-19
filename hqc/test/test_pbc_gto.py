import jax
import numpy as np
import jax.numpy as jnp
from pyscf.pbc import gto
from hqc.pbc.gto import make_pbc_gto
jax.config.update("jax_enable_x64", True)

def pyscf_eval_gto(L, xp, xe, basis, kpt):

    cell = gto.Cell()
    cell.unit = 'B'
    for ip in range(xp.shape[0]):
        cell.atom.append(['H', tuple(xp[ip])])
    cell.spin = 0
    cell.basis = basis
    cell.a = np.eye(3) * L
    cell.build()
    kpts = [kpt.tolist()]
    gto_value = cell.pbc_eval_ao("GTOval_sph", xe, kpts=kpts)[0]
    return gto_value

def test_pbc_gto_gamma():
    """
        Test the hqc.pbc.gto function at gamma point.
    """
    n, dim = 8, 3
    rs = 1.5
    rcut = 24
    cell = jnp.eye(3)
    basis_set = ['gth-szv', 'gth-dzv', 'gth-dzvp']
    L = (4/3*jnp.pi*n)**(1/3)*rs
    kpt = jnp.array([0, 0, 0])

    key = jax.random.PRNGKey(43)
    key_p, key_e = jax.random.split(key)
    xp = jax.random.uniform(key_p, (n, dim), minval=0., maxval=L)
    xe = jax.random.uniform(key_e, (n, dim), minval=0., maxval=L)

    print(print("\n============= begin test ============="))
    print("n:", n)
    print("rs:", rs)
    print("rcut:", rcut)
    print("basis_set:", basis_set)
    print("L:", L)
    print("xp:\n", xp)
    print("xe:\n", xe)

    for basis in basis_set:
        
        print("\n==========", basis, "==========")
        # PBC pyscf benchmark
        eval_gto_novmap = make_pbc_gto(basis, L, rcut)
        eval_gto = jax.vmap(eval_gto_novmap, (None, 0), 0)
        gto = eval_gto(xp, xe)
        print("gto:\n", gto)

        gto_pyscf = pyscf_eval_gto(L, np.array(xp), np.array(xe), basis, kpt)
        print("gto_pyscf:\n", gto_pyscf)
        assert np.allclose(gto, gto_pyscf, atol=1e-5)
        print("same as pyscf")

        # PBC test
        image = np.random.randint(-2, 3, size=(n, dim)).dot(cell.T)*L
        gto_pbc = eval_gto(xp, xe+image)
        print("gto_pbc:\n", gto_pbc)
        assert np.allclose(gto, gto_pbc, atol=1e-3)
        print("pbc test passed")

        # jit test
        jax.jit(eval_gto)(xp, xe)

        # xe vmap test
        xe2 = jnp.concatenate([xe, xe]).reshape(2, n, dim)
        jax.vmap(eval_gto, (None, 0), 0)(xp, xe2)
        
        # xp vmap test
        xp2 = jnp.concatenate([xp, xp]).reshape(2, n, dim)
        jax.vmap(eval_gto, (0, None), 0)(xp2, xe)
        print("vmap test passed")

def test_pbc_gto_kpt():
    """
        Test the hqc.pbc.gto function at random kpt.
    """
    n, dim = 8, 3
    rs = 1.5
    rcut = 24
    cell = jnp.eye(3)
    basis_set = ['gth-szv', 'gth-dzv', 'gth-dzvp']
    L = (4/3*jnp.pi*n)**(1/3)*rs

    key = jax.random.PRNGKey(43)
    key_p, key_e, key_kpt = jax.random.split(key, 3)
    xp = jax.random.uniform(key_p, (n, dim), minval=0., maxval=L)
    xe = jax.random.uniform(key_e, (n, dim), minval=0., maxval=L)
    kpt = jax.random.uniform(key_kpt, (3,), minval=-jnp.pi/L, maxval=jnp.pi/L)

    print(print("\n============= begin test ============="))
    print("n:", n)
    print("rs:", rs)
    print("rcut:", rcut)
    print("basis_set:", basis_set)
    print("L:", L)
    print("xp:\n", xp)
    print("xe:\n", xe)
    print("kpt:\n", kpt)

    for basis in basis_set:
        
        print("\n==========", basis, "==========")
        # PBC pyscf benchmark
        eval_gto_novmap = make_pbc_gto(basis, L, rcut, gamma=False)
        eval_gto = jax.vmap(eval_gto_novmap, (None, 0, None), 0)
        gto = eval_gto(xp, xe, kpt)
        print("gto:\n", gto)

        gto_pyscf = pyscf_eval_gto(L, np.array(xp), np.array(xe), basis, kpt)
        print("gto_pyscf:\n", gto_pyscf)
        assert np.allclose(gto, gto_pyscf, atol=1e-5)
        print("same as pyscf")

        # jit test
        jax.jit(eval_gto)(xp, xe, kpt)

        # xe vmap test
        xe2 = jnp.concatenate([xe, xe]).reshape(2, n, dim)
        jax.vmap(eval_gto, (None, 0, None), 0)(xp, xe2, kpt)
        
        # xp vmap test
        xp2 = jnp.concatenate([xp, xp]).reshape(2, n, dim)
        jax.vmap(eval_gto, (0, None, None), 0)(xp2, xe, kpt)
        print("vmap test passed")
