'''
This is a patch file for vmap jax_xc.
If you want to use the jax_xc here, please ignore this, since the jax_xc in this directory has been patched.
If you want to download jax_xc to this directory yourself, please run this script to patch jax_xc.

For vmap and jit, the xc functionals in "jax_xc" needs to be changed like this.
Take the "jax_xc/impl/lda_x.py" for example, the invoke function is

def invoke(
    p: NamedTuple, rho: Callable, r: jnp.ndarray, mo: Optional[Callable] = None,
    deorbitalize: Optional[float] = None,
):
    args = rho_to_arguments(p, rho, r, mo, deorbitalize)
    ret = pol(p, *args) if p.nspin == 2 else unpol(p, *args)
    dens = args[0] if p.nspin == 1 else sum(args[0])
    ret = float(dens >= p.dens_threshold) * ret
    return ret

Change the second to the last line from "ret = float(dens >= p.dens_threshold) * ret" to "ret = jnp.float64(dens >= p.dens_threshold) * ret"
'''

import os
import sys
import fileinput

dir_path = 'jax_xc/impl'
files = os.listdir(dir_path)
py_files = [f for f in files if f.endswith('.py')]

for file in py_files:
    filename = dir_path+'/'+file
    for line in fileinput.input(filename, inplace=True):
        line = line.replace('ret = float(dens', 'ret = jnp.float64(dens')
        sys.stdout.write(line)
