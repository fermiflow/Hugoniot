# hqc

Quantum chemistry calculations in Hydrogen system.

Returns Hartree fock or DFT energy of PBC or isolated Hydrogen system, unit: Rydberg.

## Install

clone and cd to this directory.

use "pip install -e ." to install hydrogenqc.
```bash
pip install -e .
```

use "pip uninstall hqc" to uninstall.
```bash
pip unintall hqc
```

## Import

use "import hqc" directly to import this package anywhere.
```python
import hqc
```

## Example
Compare the DFT results with Pyscf

```python
from hqc.pbc.lcao import make_lcao
from hqc.pbc.pyscf import pyscf_dft
import jax
import jax.numpy as jnp

rs = 1.25
n, dim = 8, 3
basis = 'gth-dzv'
T = 10000 # K
beta = 157888.088922572/T # inverse temperature in unit of 1/Ry
sigma = 1/beta/2 # temperature in Hartree unit

L = (4/3*jnp.pi*n)**(1/3)
key = jax.random.PRNGKey(42)
xp = jax.random.uniform(key, (n, dim), minval=0., maxval=L)

lcao = make_lcao(n, L, rs, basis, dft=True, smearing=True, smearing_sigma=sigma)
mo_coeff, bands = lcao(xp)
print("================= solver =================")
# print("mo_coeff:\n", mo_coeff)
print("bands:\n", bands)

mo_coeff_dft, energy_dft = pyscf_dft(n, L, rs, sigma, xp, basis, xc='lda,vwn', smearing=True, smearing_method='fermi')
print("================= pyscf =================")
# print("mo_coeff:\n", mo_coeff)
print("bands:\n", bands)                                          
```
or use "python example.py" to run a specific kind of test for hydrogen.
```bash
python example.py
```

## Requirements

        jax

Yes, we only need `jax`.  
`hqc` has better performance on GPU.  
If you want to run test, you need to install `pyscf`.

## Basis
The basis file is in the path `hqc.basis`.

`hqc.basis.parse` can find any `basis_name.dat` file in that path automatically, where `basis_name` is the name of your basis when you load it, for example, "gth-dzv" for `gth-dzv.dat`. 

`basis_name.dat` is in **CP2K** format, take `hqc.basis.gth-raw.gth-dzv.dat` as an example:
```
#BASIS SET
H DZV-GTH
  1
  1  0  0  4  2
        8.3744350009  -0.0283380461   0.0000000000
        1.8058681460  -0.1333810052   0.0000000000
        0.4852528328  -0.3995676063   0.0000000000
        0.1658236932  -0.5531027541   1.0000000000
#
```

## Functions
`hqc.pbc.gto`: Evaluate gto orbital.

`hqc.pbc.lcao`: HF or DFT solver.

`hqc.pbc.pes`: HF or DFT potential energy surface (PES).

`hqc.pbc.overlap`: Calculate basis overlap.

`hqc.pbc.slater`: Calculate slater matrix for LCAO orbitals.

## Release note
>### hqc 0.1.11
>Update the total energy in `hqc.pbc.pes` 'dev' mode returns from Eelec to Etot (add Vpp in E).

>### hqc 0.1.10
>Add `hqc.pbc.solver` to return more information of HF/DFT solver, including entropy.
>Add eval_entropy in `hqc.pbc.solver`.

>### hqc 0.1.9
>Add `hqc.pbc.pes` to calculate *potential energy surface (PES)*.  
>Add `hqc.pbc.potential` to calculate vpp.

>### hqc 0.1.8
>**[*Interface change*]**  
>Add *k-point* support in `hqc.pbc.slater`.

>### hqc 0.1.7
>Input and output type check.  

>### hqc 0.1.6
>Simplify *exchange_correlation_fn* in the scf loop. 

>### hqc 0.1.5
>Simplify structure, add `hqc.pbc.scf`.

>### hqc 0.1.4 
>**[*Interface change*]**  
>Add *k-point* support for DFT method.

>### hqc 0.1.3 
>**[*Interface change*]**  
>Add *k-point* support for HF method.

>### hqc 0.1.2 
>**[*Interface change*]**  
>Add *E* in the returns of `lcao` function.

>### hqc 0.1.1
>Add `hqc.pbc.slater`.

>### hqc 0.1.0 
>Rename `hqc.pbc.ao` to `hqc.pbc.gto`.  
>Rename `hqc.pbc.mo` to `hqc.pbc.lcao`.

>### hqc 0.0.2
>Update structure, ready to use `pip install -e .`

>### hqc 0.0.1
>Init.
