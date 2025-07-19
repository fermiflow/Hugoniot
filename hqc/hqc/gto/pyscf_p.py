import jax
import jax.numpy as jnp
from pyscf import gto

# #BASIS SET: (4s,1p) -> [2s,1p]
# H    S
#      12.98511484             0.0310449        
#       1.95908134             0.2173262        
#       0.44403649             0.7516287        
# H    S
#       0.11282695             1.0000000        
# H    P
#       0.80305095             1.0000000 

const = (2 / jnp.pi)**0.75
dzp_s1 = jnp.array([[12.98511484, 0.0310449],   
                    [1.95908134, 0.2173262],
                    [0.44403649, 0.7516287]])
dzp_s2 = jnp.array([0.11282695, 1.0000000])
dzp_p1 = jnp.array([0.80305095, 1.0000000]) 

L = 3
n, dim = 4, 3
key = jax.random.PRNGKey(42)
xe = jax.random.uniform(key, (n, dim), minval=-L, maxval=L)

# pyscf dzvp
mol = gto.Mole()
mol.atom = '''H 0 0 0'''
mol.basis =  'dzp'
mol.spin = 1
mol.build()
pyscf_dzp = mol.eval_gto("GTOval_sph", xe)

# my dzp
def eval_dzp(xe):
    r2 = jnp.sum(jnp.square(xe))
    phi1 = const * jnp.einsum('i,i,i->', dzp_s1[:,1], dzp_s1[:,0]**0.75, jnp.exp(-dzp_s1[:,0] * r2))
    phi2 = const * dzp_s2[1] * dzp_s2[0]**0.75 * jnp.exp(-dzp_s2[0] * r2)
    phi3 = xe[0] *  dzp_p1[1] * jnp.exp(-dzp_p1[0] * r2)
    phi4 = xe[1] *  dzp_p1[1] * jnp.exp(-dzp_p1[0] * r2)
    phi5 = xe[2] *  dzp_p1[1] * jnp.exp(-dzp_p1[0] * r2)
    return jnp.array([phi1, phi2, phi3, phi4, phi5])

my_dzp = jax.vmap(eval_dzp, 0, 0)(xe)

print("pyscf dzp:\n", pyscf_dzp)
print("my dzp:\n", my_dzp)
print("ratio:\n", pyscf_dzp/my_dzp)
assert jnp.allclose(pyscf_dzp, my_dzp)

