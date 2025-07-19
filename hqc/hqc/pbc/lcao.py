import jax
import numpy as np
import jax.numpy as jnp
from functools import partial
from typing import Callable, Tuple
from jax import vmap, grad, jit, jacfwd

from hqc.pbc.solver import make_solver


def make_lcao(n: int, L: float, rs: float, basis: str,
              rcut: float = 24, 
              tol: float = 1e-7, 
              max_cycle: int = 100, 
              grid_length: float = 0.12,
              diis: bool = True, 
              diis_space: int = 8, 
              diis_start_cycle: int = 1, 
              diis_damp: float = 0.,
              use_jit: bool = True,
              dft: bool = False, 
              xc: str = 'lda,vwn',
              smearing: bool = False,
              smearing_method: str = 'fermi',
              smearing_sigma: float = 0.,
              search_method: str = 'newton', 
              search_cycle: int = 100, 
              search_tol: float= 1e-7,
              gamma: bool = True) -> Callable:
    """
        Make PBC Hartree Fock function.
        INPUT:
            n: int, number of hydrogen atoms in unit cell.
            L: float, side length of unit cell, unit: rs.
            rs: float, unit: Bohr
            basis: gto basis name, eg:'gth-szv'.
            tol: the tolerance for convergence.
            max_cycle: the maximum number of iterations.
            grid_length: the grid length for real space grid, unit: Bohr.
            diis: if True, use DIIS.
            diis_space: the number of vectors in DIIS space.
            diis_start_cycle: the cycle to start DIIS.
            diis_damp: the damping factor for DIIS.
            use_jit: if True, use jit.
            dft: if True, use DFT, if False, use HF.
            xc: exchange-correlation functional.
            smearing: if True, use smearing.
            smearing_method: 'fermi' or 'gauss'.
            smearing_sigma: smearing width, unit: Hartree.
            search_method: 'bisect' or 'newton'.
            search_cycle: the maximum number of iterations for search.
            search_tol: the tolerance for searching mu.
            gamma: bool, if True, return lcao(xp) for gamma point only, 
                         else, return lcao(xp, kpt) for a single k-point.
        OUTPUT:
            lcao: lcao function.
    """
    solver = make_solver(n, L, rs, basis, rcut=rcut, tol=tol, max_cycle=max_cycle, 
                         grid_length=grid_length, diis=diis, diis_space=diis_space, 
                         diis_start_cycle=diis_start_cycle, diis_damp=diis_damp,
                         use_jit=use_jit, dft=dft, xc=xc, smearing=smearing, 
                         smearing_method=smearing_method, smearing_sigma=smearing_sigma,
                         search_method=search_method, search_cycle=search_cycle, 
                         search_tol=search_tol, gamma=gamma)

    def lcao_gamma(xp: np.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
        """
            LCAO for gamma point.
            INPUT:
                xp: array of shape (n, dim), position of protons in rs unit.
                    Warining: xp * rs is in Bohr unit, xp is in rs unit.
            OUTPUT:
                mo_coeff: array of shape (n_ao, n_mo), molecular orbital coefficients.
                bands: array of shape (n_mo,), orbital energies, Unit: Rydberg.
                E: float, total energy of the electrons, Note that vpp is not include in E, Unit: Rydberg.
        """
        mo_coeff, _, bands, E, _, _, _, _, _ = solver(xp)
        return mo_coeff, bands, E

    def lcao_kpt(xp: np.ndarray, kpt: np.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
        """
            LCAO for k-point.
            INPUT:
                xp: array of shape (n, dim), position of protons in rs unit.
                    Warining: xp * rs is in Bohr unit, xp is in rs unit.
                kpt: array of shape (3,), k-point. (Unit: 1/Bohr)
                    1BZ: (-pi/L/rs, pi/L/rs)
            OUTPUT:
                mo_coeff: array of shape (n_ao, n_mo), molecular orbital coefficients.
                bands: array of shape (n_mo,), orbital energies, Unit: Rydberg.
                E: float, total energy of the electrons, Note that vpp is not include in E, Unit: Rydberg.
        """
        mo_coeff, _, bands, E, _, _, _, _, _ = solver(xp, kpt)
        return mo_coeff, bands, E

    if gamma:
        lcao = lcao_gamma
    else:
        lcao = lcao_kpt

    if use_jit:
        return jax.jit(lcao)
    else:
        return lcao