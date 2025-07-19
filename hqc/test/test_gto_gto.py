from config import *
from pyscf import gto
from hqc.gto.ao import make_ao

def pyscf_eval_ao(xp, xe, basis):
    n = xp.shape[0]
    mol = gto.Mole()
    mol.unit = 'B'
    for i in range(n):
        mol.atom.append(['H', tuple(xp[i])])
    mol.spin = 0
    mol.basis = basis
    mol.build()
    ao_value = mol.eval_ao("GTOval_sph", xe)
    return ao_value

def test_pbc_ao():

	n, dim = 14, 3
	rs = 1.25
	L = (4/3*jnp.pi*n)**(1/3)*rs
	key = jax.random.PRNGKey(42)
	xp = jax.random.uniform(key, (n, dim), minval=0., maxval=L)
	key = jax.random.PRNGKey(43)
	xe = jax.random.uniform(key, (n, dim), minval=0., maxval=L)

	basis_set = ['sto3g', 'sto6g']
	for basis in basis_set:

		# pyscf benchmark
		pyscf_ao = pyscf_eval_ao(xp, xe, basis)
		eval_ao = jax.vmap(make_ao(basis), (None, 0), 0)
		ao = eval_ao(xp, xe)
		assert np.allclose(pyscf_ao, ao)

		# jit, vmap, test
		jax.jit(eval_ao)(xp, xe)
		xe2 = jnp.concatenate([xe, xe]).reshape(2, n, dim)
		jax.vmap(eval_ao, (None, 0), 0)(xp, xe2)
		xp2 = jnp.concatenate([xp, xp]).reshape(2, n, dim)
		jax.vmap(eval_ao, (0, None), 1)(xp2, xe)
		