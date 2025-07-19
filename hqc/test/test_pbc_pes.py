import jax
import numpy as np
import jax.numpy as jnp
from pyscf.pbc import gto, scf, dft
jax.config.update("jax_enable_x64", True)

from hqc.pbc.pes import make_pes
from hqc.basis.parse import load_as_str

from config import *
from test_pyscf import pyscf_solver

# Global test variables
n, dim = 4, 3
rs = 1.5
L = (4/3*jnp.pi*n)**(1/3)
basis_set = ['gth-dzv', 'gth-dzvp']
basis_set = ['gth-dzv']
xc = 'lda,vwn'
rcut = 24
grid_length = 0.12
smearing_sigma = 0.1
key = jax.random.PRNGKey(42)
key_p, key_kpt = jax.random.split(key)
xp = jax.random.uniform(key_p, (n, 3), minval=0., maxval=L)
kpt = jax.random.uniform(key_kpt, (3,), minval=-jnp.pi/L/rs, maxval=jnp.pi/L/rs)

def pes_test(dft, smearing, gamma):
 
    print(f"{YELLOW}\n============= test info ============={RESET}")
    if dft and gamma:
        print(f"{BLUE}test:{RESET} dft gamma")
    elif dft and not gamma:
        print(f"{BLUE}test:{RESET} dft kpt")
    elif not dft and gamma:
        print(f"{BLUE}test:{RESET} hf gamma")
    else:
        print(f"{BLUE}test:{RESET} hf kpt")
    if gamma:
        kpoint = jnp.array([0., 0., 0.])
    else:
        kpoint = kpt
    print(f"{BLUE}n:{RESET}", n)
    print(f"{BLUE}rs:{RESET}", rs)
    print(f"{BLUE}L:{RESET}", L)
    print(f"{BLUE}kpt:{RESET}", kpoint)
    print(f"{BLUE}basis_set:{RESET}", basis_set)
    print(f"{BLUE}rcut:{RESET}", rcut)
    print(f"{BLUE}grid_length:{RESET}", grid_length)
    print(f"{BLUE}smearing:{RESET}", smearing)
    if smearing:
        print(f"{BLUE}smearing sigma:{RESET}", smearing_sigma)
    print(f"{BLUE}xp:\n{RESET}", xp)

    for basis in basis_set:
        print(f"{YELLOW}\n-----", basis, f"-----{RESET}")

        pyscf_data = pyscf_solver(n, L, rs, xp, kpoint, basis, ifdft=dft, xc=xc, smearing=smearing,
                                  smearing_method='fermi', smearing_sigma=smearing_sigma)

        pes = make_pes(n, L, rs, basis, grid_length=grid_length, dft=dft, 
                       smearing=smearing, smearing_sigma=smearing_sigma, gamma=gamma, mode='dev')
        if gamma:
            E, Ki, Vep, Vee, Vpp, Se, converged = pes(xp)
        else:
            E, Ki, Vep, Vee, Vpp, Se, converged = pes(xp, kpoint)

        print(f"{BLUE}converged:{RESET}", converged)
        assert converged
        print(f"{GREEN}solver converged{RESET}")

        print(f"{BLUE}Etot:{RESET}", E)
        print(f"{BLUE}Etot_pyscf:{RESET}", pyscf_data['Etot'])
        assert np.allclose(E, pyscf_data['Etot'], atol=1e-3)
        print(f"{GREEN}same E{RESET}")

        print(f"{BLUE}Ecore:{RESET}", Ki+Vep)
        print(f"{BLUE}Ecore_pyscf:{RESET}", pyscf_data["Ecore"])
        assert np.allclose(Ki+Vep, pyscf_data["Ecore"], atol=1e-3)
        print(f"{GREEN}same Ecore{RESET}")

        print(f"{BLUE}Vee:{RESET}", Vee)
        print(f"{BLUE}Vee_pyscf:{RESET}", pyscf_data["Vee"])
        assert np.allclose(Vee, pyscf_data["Vee"], atol=1e-3)
        print(f"{GREEN}same Vee{RESET}")

        print(f"{BLUE}Vpp:{RESET}", Vpp)
        print(f"{BLUE}Vpp_pyscf:{RESET}", pyscf_data["Vpp"])
        assert np.allclose(Vpp, pyscf_data["Vpp"], atol=1e-3)
        print(f"{GREEN}same Vpp{RESET}")

        print(f"{BLUE}Se:{RESET}", Se)
        print(f"{BLUE}Se_pyscf:{RESET}", pyscf_data["Se"])
        assert np.allclose(Se, pyscf_data["Se"], atol=1e-3)
        print(f"{GREEN}same Se{RESET}")

def test_pes_hf_gamma():
    dft = False
    gamma = True
    smearing = False
    pes_test(dft, smearing, gamma)

def test_pes_hf_kpt():
    dft = False
    gamma = False
    smearing = False
    pes_test(dft, smearing, gamma)

def test_pes_dft_gamma():
    dft = True
    gamma = True
    smearing = True
    pes_test(dft, smearing, gamma)

def test_pes_dft_kpt():
    dft = True
    gamma = False
    smearing = True
    pes_test(dft, smearing, gamma)

def test_pes_hf_gamma_smearing():
    dft = False
    gamma = True
    smearing = True
    pes_test(dft, smearing, gamma)

def test_pes_hf_kpt_smearing():
    dft = False
    gamma = False
    smearing = True
    pes_test(dft, smearing, gamma)

def test_pes_dft_gamma_smearing():
    dft = True
    gamma = True
    smearing = True
    pes_test(dft, smearing, gamma)

def test_pes_dft_kpt_smearing():
    dft = True
    gamma = False
    smearing = True
    pes_test(dft, smearing, gamma)
