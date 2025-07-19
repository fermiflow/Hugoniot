import jax
import numpy as np
import jax.numpy as jnp
from pyscf.pbc import gto, dft, scf
jax.config.update("jax_enable_x64", True)

from hqc.pbc.solver import make_solver
from hqc.basis.parse import load_as_str

from config import *
from test_pyscf import pyscf_solver

# Global test variables
n, dim = 4, 3
rs = 1.5
L = (4/3*jnp.pi*n)**(1/3)
basis_set = ['gth-dzv', 'gth-dzvp']
xc = 'lda,vwn'
rcut = 24
grid_length = 0.12
smearing_sigma = 0.1
key = jax.random.PRNGKey(42)
key_p, key_kpt = jax.random.split(key)
xp = jax.random.uniform(key_p, (n, 3), minval=0., maxval=L)
kpt = jax.random.uniform(key_kpt, (3,), minval=-jnp.pi/L/rs, maxval=jnp.pi/L/rs)

def lcao_test(dft, diis, smearing, gamma):
 
    print(f"\n{YELLOW}============= test info ============={RESET}")
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
    print(f"{BLUE}DIIS:{RESET}", diis)
    print(f"{BLUE}smearing:{RESET}", smearing)
    print(f"{BLUE}smearing sigma:{RESET}", smearing_sigma)
    print(f"{BLUE}xp:\n{RESET}", xp)

    for basis in basis_set:
        print(f"{YELLOW}\n-----", basis, f"-----{RESET}")

        # PBC energy test
        pyscf_data = pyscf_solver(n, L, rs, xp, kpoint, basis, ifdft=dft, xc=xc, smearing=smearing,
                                  smearing_method='fermi', smearing_sigma=smearing_sigma)

        solver = make_solver(n, L, rs, basis, grid_length=grid_length, diis=diis, dft=dft, 
                             smearing=smearing, smearing_sigma=smearing_sigma, gamma=gamma)
        if gamma:
            mo_coeff, dm, bands, E, Ki, Vep, Vee, Se, converged = solver(xp)
        else:
            mo_coeff, dm, bands, E, Ki, Vep, Vee, Se, converged = solver(xp, kpoint)

        print(f"{BLUE}converged:{RESET}", converged)
        assert converged
        print(f"{GREEN}solver converged{RESET}")

        mo_coeff = mo_coeff @ jnp.diag(jnp.sign(mo_coeff[0]).conjugate())
        mo_coeff_pyscf = pyscf_data["mo_coeff"] @ jnp.diag(jnp.sign(pyscf_data["mo_coeff"][0]).conjugate())
        # print("mo_coeff:\n", mo_coeff)
        # print("mo_coeff_pyscf:\n", mo_coeff_pyscf)
        print(f"{BLUE}max diff between mo_coeff and pyscf_mo_coeff{RESET}:", jnp.max(mo_coeff-mo_coeff_pyscf))
        assert np.allclose(mo_coeff, mo_coeff_pyscf, atol=1e-2)
        print(f"{GREEN}same mo_coeff{RESET}")

        print(f"{BLUE}max diff between dm and pyscf_dm:{RESET}", jnp.max(dm-pyscf_data["dm"]))
        assert np.allclose(dm, pyscf_data["dm"], atol=1e-2)
        print(f"{GREEN}same dm{RESET}")

        print(f"{BLUE}bands:\n{RESET}", bands)
        print(f"{BLUE}bands_pyscf:\n{RESET}", pyscf_data["bands"])
        assert np.allclose(bands, pyscf_data["bands"], atol=1e-3)
        print(f"{GREEN}same bands{RESET}")

        print(f"{BLUE}E:{RESET}", E)
        print(f"{BLUE}E_pyscf:{RESET}", pyscf_data["Eelec"])
        assert np.allclose(E, pyscf_data["Eelec"], atol=1e-3)
        print(f"{GREEN}same E{RESET}")

        print(f"{BLUE}Ecore:{RESET}", Ki+Vep)
        print(f"{BLUE}Ecore_pyscf:{RESET}", pyscf_data["Ecore"])
        assert np.allclose(Ki+Vep, pyscf_data["Ecore"], atol=1e-3)
        print(f"{GREEN}same Ecore{RESET}")

        print(f"{BLUE}Vee:{RESET}", Vee)
        print(f"{BLUE}Vee_pyscf:{RESET}", pyscf_data["Vee"])
        assert np.allclose(Vee, pyscf_data["Vee"], atol=1e-3)
        print(f"{GREEN}same Vee{RESET}")

        print(f"{BLUE}Se:{RESET}", Se)
        print(f"{BLUE}Se_pyscf:{RESET}", pyscf_data["Se"])
        assert np.allclose(Se, pyscf_data["Se"], atol=1e-3)
        print(f"{GREEN}same Se{RESET}")

def test_hf_gamma_diis():
    dft = False
    gamma = True
    diis = True
    smearing = False
    lcao_test(dft, diis, smearing, gamma)

def test_hf_kpt_diis():
    dft = False
    gamma = False
    diis = True
    smearing = False
    lcao_test(dft, diis, smearing, gamma)

def test_dft_gamma_diis():
    dft = True
    gamma = True
    diis = True
    smearing = False
    lcao_test(dft, diis, smearing, gamma)

def test_dft_kpt_diis():
    dft = True
    gamma = False
    diis = True
    smearing = False
    lcao_test(dft, diis, smearing, gamma)

def test_hf_gamma_fp():
    dft = False
    gamma = True
    diis = False
    smearing = False
    lcao_test(dft, diis, smearing, gamma)

def test_hf_kpt_fp():
    dft = False
    gamma = False
    diis = False
    smearing = False
    lcao_test(dft, diis, smearing, gamma)

def test_dft_gamma_fp():
    dft = True
    gamma = True
    diis = False
    smearing = False
    lcao_test(dft, diis, smearing, gamma)

def test_dft_kpt_fp():
    dft = True
    gamma = False
    diis = False
    smearing = False
    lcao_test(dft, diis, smearing, gamma)

def test_hf_gamma_diis_smearing():
    dft = False
    gamma = True
    diis = True
    smearing = True
    lcao_test(dft, diis, smearing, gamma)

def test_hf_kpt_diis_smearing():
    dft = False
    gamma = False
    diis = True
    smearing = True
    lcao_test(dft, diis, smearing, gamma)

def test_dft_gamma_diis_smearing():
    dft = True
    gamma = True
    diis = True
    smearing = True
    lcao_test(dft, diis, smearing, gamma)

def test_dft_kpt_diis_smearing():
    dft = True
    gamma = False
    diis = True
    smearing = True
    lcao_test(dft, diis, smearing, gamma)
