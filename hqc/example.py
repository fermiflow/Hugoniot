import jax
import jax.numpy as jnp

from hqc.pbc.lcao import make_lcao
from hqc.pbc.slater import make_slater

n = 8
rs = 2.0
L = (4/3*jnp.pi*n)**(1/3)
basis = "gth-dzv"
grid_length = 0.12
dft = True
xc = "lda,vwn"
smearing = True
smearing_method = "fermi" 
smearing_temperature = 10000
beta = 157888.088922572/smearing_temperature # inverse temperature in unit of 1/Ry
smearing_sigma = 1/beta/2 # temperature in Hartree unit
search_method = "newton"
groundstate = True

# make lcao and slater functions
lcao = make_lcao(n, L, rs, basis, grid_length=grid_length, dft=dft, xc=xc, 
                 smearing=smearing, smearing_method=smearing_method, 
                 smearing_sigma=smearing_sigma, search_method=search_method)
slater = make_slater(n, L, rs, basis, groundstate=groundstate)

# initialize proton and electron positions
key = jax.random.PRNGKey(42)
key_p, key_e = jax.random.split(key) 
xp = jax.random.uniform(key_p, (n, 3), minval=0, maxval=L)
xe = jax.random.uniform(key_e, (n, 3), minval=0, maxval=L)

# run solver
mo_coeff, bands = lcao(xp)
slater_up, slater_dn = slater(xp, xe, mo_coeff)
print("mo_coeff:\n", mo_coeff)
print("bands:\n", bands)
print("slater_up:\n", slater_up)
print("slater_dn:\n", slater_dn)
