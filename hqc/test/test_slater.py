import jax
import time
import functools
import numpy as np
import jax.numpy as jnp
from functools import partial
from typing import Sequence, Optional
jax.config.update("jax_enable_x64", True)

from hqc.pbc.lcao import make_lcao
from hqc.pbc.slater import make_slater

from tools.vmc import sample_x_mcmc
from tools.observables import observables
from tools.logpsi import make_logpsi_hf, make_logpsi2, make_logpsi_grad_laplacian
from tools.logpsi import make_logpsi_hf_kpt, make_logpsi2_kpt, make_logpsi_grad_laplacian_kpt


def vmc_slater_hf(xp, rs, basis, kpt, rcut, grid_length, smearing, sigma, gamma, 
                  batchsize, mc_steps, mc_width, therm_steps, sample_steps):
    """
        This is a vmc test for HF slater determinant wavefunction.
        Input:
            xp: jnp.array, atom positions, shape (n, 3)
            rs: float, Wigner-Seitz radius
            basis: str, basis set
            kpt: jnp.array, k-point, shape (3,), kpt will only be used if gamma is False
            rcut: float, cutoff radius
            grid_length: float, grid length
            smearing: bool, if smearing
            sigma: float, smearing parameter
            gamma: bool, gamma point
            batchsize: int, batchsize
            mc_steps: int, mc steps
            mc_width: float, mc width
            therm_steps: int, vmc thermalization steps
            sample_steps: int, vmc sample teps
    """
    n = xp.shape[0]
    L = (4/3*jnp.pi*n)**(1/3)
    if gamma:
        kpoint = jnp.array([0., 0., 0.])
    else:
        kpoint = kpt
   
    print("------- system information -------")
    print(f"n: {n}")
    print(f"L: {L}")
    print(f"rs: {rs} (Bohr)")
    print(f"xp: \n{xp}")
    print(f"hf basis: {basis}")
    print(f"rcut: {rcut}")
    print(f"grid_length: {grid_length}")
    print(f"smearing: {smearing}")
    print(f"sigma: {sigma}")
    print(f"gamma: {gamma}")
    print(f"batchsize: {batchsize}")
    print(f"mc_steps: {mc_steps}")
    print(f"mc_width: {mc_width}")

    print("------- HF and MC results -------")

    lcao = make_lcao(n, L, rs, basis, rcut=rcut, grid_length=grid_length, dft=False, 
                     smearing=smearing, smearing_sigma=sigma, gamma=gamma, mode='debug')
    if gamma:
        mo_coeff, bands, e_hf, k_hf, vep_hf, vee_hf = lcao(xp)
    else:
        mo_coeff, bands, e_hf, k_hf, vep_hf, vee_hf = lcao(xp, kpoint)

    print("\nHF results:")
    print("mo_coeff:\n", mo_coeff)
    print("bands:\n", bands)
    print("e_hf per atom (k+vep+vee in Ry):", e_hf/n)
    print("k_hf per atom (Ry):", k_hf/n)
    print("vep_hf per atom (Ry):", vep_hf/n)
    print("vee_hf per atom (Ry):", vee_hf/n)

    hf_orbitals = make_slater(n, L, rs, basis=basis, rcut=rcut, groundstate=True, gamma=gamma)

    if gamma:
        logpsi = make_logpsi_hf(hf_orbitals)
        logpsi2 = make_logpsi2(logpsi)
        logpsi_grad_laplacian = make_logpsi_grad_laplacian(logpsi)
    else:
        logpsi_kpt = make_logpsi_hf_kpt(hf_orbitals)
        logpsi2_kpt = make_logpsi2_kpt(logpsi_kpt)
        logpsi_grad_laplacian_kpt = make_logpsi_grad_laplacian_kpt(logpsi_kpt)
        logpsi2 = lambda x, s, mo_coeff: logpsi2_kpt(x, s, mo_coeff, kpoint)
        logpsi_grad_laplacian = lambda x, s, mo_coeff: logpsi_grad_laplacian_kpt(x, s, mo_coeff, kpoint)

    key = jax.random.PRNGKey(42)
    xe = jax.random.uniform(key, (batchsize, n, 3), minval=0., maxval=L)

    for step in range(therm_steps):
        key, xe, acc = sample_x_mcmc(key, xp, xe, logpsi2, mo_coeff, mc_steps, mc_width, L)
        print("therm step:", step, "acc:", acc)

    e_mean = np.zeros(sample_steps)
    e_err = np.zeros(sample_steps)
    k_mean = np.zeros(sample_steps)
    k_err = np.zeros(sample_steps)
    vep_mean = np.zeros(sample_steps)
    vep_err = np.zeros(sample_steps)
    vee_mean = np.zeros(sample_steps)
    vee_err = np.zeros(sample_steps)
    vpp_mean = np.zeros(sample_steps)
    vpp_err = np.zeros(sample_steps)
    acc = np.zeros(sample_steps)

    for step in range(sample_steps):
        key, xe, acc[step] = sample_x_mcmc(key, xp, xe, logpsi2, mo_coeff, mc_steps, mc_width, L)
        e, k, vpp, vep, vee = observables(xp, xe, mo_coeff, n, rs, logpsi_grad_laplacian)

        e_mean[step] = e.mean()/rs**2/n 
        e_err[step] = e.std()/jnp.sqrt(batchsize)/rs**2/n

        k_mean[step] = k.mean()/rs**2/n 
        k_err[step] = k.std()/jnp.sqrt(batchsize)/rs**2/n 

        vep_mean[step] = vep.mean()/rs**2/n 
        vep_err[step] = vep.std()/jnp.sqrt(batchsize)/rs**2/n 

        vee_mean[step] = vee.mean()/rs**2/n 
        vee_err[step] = vee.std()/jnp.sqrt(batchsize)/rs**2/n 

        vpp_mean[step] = vpp.mean()/rs**2/n
        vpp_err[step] = vpp.std()/jnp.sqrt(batchsize)/rs**2/n 

        print ("steps, e, k, vep, vee, vpp, acc", 
                      step, 
                      e_mean[step], "+/-", e_err[step], 
                      k_mean[step], "+/-", k_err[step], 
                      vep_mean[step], "+/-", vep_err[step], 
                      vee_mean[step], "+/-", vee_err[step], 
                      vpp_mean[step], "+/-", vpp_err[step], 
                      acc[step])

    e_mean_tot = e_mean.mean()
    e_err_tot = e_mean.std()/jnp.sqrt(sample_steps)
    k_mean_tot = k_mean.mean()
    k_err_tot = k_mean.std()/jnp.sqrt(sample_steps)
    vep_mean_tot = vep_mean.mean()
    vep_err_tot = vep_mean.std()/jnp.sqrt(sample_steps)
    vee_mean_tot = vee_mean.mean()
    vee_err_tot = vee_mean.std()/jnp.sqrt(sample_steps)
    vpp_mean_tot = vpp_mean.mean()
    vpp_err_tot = vpp_mean.std()/jnp.sqrt(sample_steps)
    
    print("\nVMC results:")
    print("e_tot per atom (Ry):", e_mean_tot, "+/-", e_err_tot)
    print("k_tot per atom (Ry):", k_mean_tot, "+/-", k_err_tot)
    print("vep_tot per atom (Ry):", vep_mean_tot, "+/-", vep_err_tot)
    print("vee_tot per atom (Ry):", vee_mean_tot, "+/-", vee_err_tot)
    print("vpp_tot per atom (Ry):", vpp_mean_tot, "+/-", vpp_err_tot)


def test_slater_hf():
    pass

