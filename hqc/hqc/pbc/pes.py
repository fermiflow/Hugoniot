import jax
import numpy as np
import jax.numpy as jnp
from functools import partial
from typing import Callable, Tuple

from hqc.pbc.solver import make_solver
from hqc.pbc.potential import potential_energy_pp

def make_pes(n: int, L: float, rs: float, basis: str,
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
             gamma: bool = True,
             Gmax: int = 15,
             kappa: float = 10,
             mode: str = 'default') -> Callable:
    """
        Make Potential Energy Surface (PES) function for a periodic box.
            E = k + vep + vee + vpp, Unit: Ry.
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
            gamma: bool, if True, return pes(xp) for gamma point only, 
                         else, return pes(xp, kpt) for a single k-point.
            Gmax: int, the cutoff of G-vectors.
            kappa: float, the screening parameter.
            mode: str, 'default' or 'dev'.
                    in 'default' mode, pes(xp) returns the total energy.
                    in 'dev' mode, pes(xp) returns the total energy and its components. 
        OUTPUT:
            pes: pes function.
                Inputs: xp: (n, dim) proton coordinates.
                        kpt: (3,) k-point coordinates, if gamma=False.
                Outputs: pes: float, total energy e = k+vep+vee+vpp, unit: Ry.
    """

    solver = make_solver(n, L, rs, basis, rcut=rcut, tol=tol, max_cycle=max_cycle, 
                         grid_length=grid_length, diis=diis, diis_space=diis_space, 
                         diis_start_cycle=diis_start_cycle, diis_damp=diis_damp,
                         use_jit=use_jit, dft=dft, xc=xc, smearing=smearing, 
                         smearing_method=smearing_method, smearing_sigma=smearing_sigma,
                         search_method=search_method, search_cycle=search_cycle, 
                         search_tol=search_tol, gamma=gamma)

    def pes_gamma(xp: np.ndarray) -> float:
        """
            Calculate the total energy for a given proton configuration at gamma point.
            INPUT:
                xp: (n, dim) proton coordinates.
            OUTPUT:
                e: float, total energy e = k+vep+vee+vpp, unit: Ry.
        """
        return solver(xp)[3] + potential_energy_pp(xp, L, rs, kappa=kappa, Gmax=Gmax)

    def pes_kpt(xp: np.ndarray, kpt: np.ndarray) -> float:
        """
            Calculate the total energy for a given proton configuration at a single k-point.
            INPUT:
                xp: (n, dim) proton coordinates.
                kpt: (3,) k-point coordinates.
            OUTPUT:
                e: float, total energy e = k+vep+vee+vpp, unit: Ry.
        """
        return solver(xp, kpt)[3] + potential_energy_pp(xp, L, rs, kappa=kappa, Gmax=Gmax)
    
    def pes_dev_gamma(xp: np.ndarray) -> Tuple[float, float, float, float, float, float, bool]:
        """
            Calculate the total energy and its components for a given proton configuration at gamma point.
            INPUT:
                xp: (n, dim) proton coordinates.
            OUTPUT:
                E: float, total energy, unit: Ry.
                Ki: float, electron kinetic energy, unit: Ry.
                Vep: float, electron-proton interaction energy, unit: Ry.
                Vee: float, electron-electron interaction energy, unit: Ry.
                Vpp: float, proton-proton interaction energy, unit: Ry.
                Se: float, electron entropy.
                converged: bool, if True, the calculation is converged.
        """
        _, _, _, E, Ki, Vep, Vee, Se, converged = solver(xp)
        Vpp = potential_energy_pp(xp, L, rs, kappa=kappa, Gmax=Gmax)
        return E+Vpp, Ki, Vep, Vee, Vpp, Se, converged

    def pes_dev_kpt(xp: np.ndarray, kpt: np.ndarray) -> Tuple[float, float, float, float, float, float, bool]:
        """
            Calculate the total energy and its components for a given proton configuration at a single k-point.
            INPUT:
                xp: (n, dim) proton coordinates.
                kpt: (3,) k-point coordinates.
            OUTPUT:
                E: float, total energy, unit: Ry.
                Ki: float, electron kinetic energy, unit: Ry.
                Vep: float, electron-proton interaction energy, unit: Ry.
                Vee: float, electron-electron interaction energy, unit: Ry.
                Vpp: float, proton-proton interaction energy, unit: Ry.
                Se: float, electron entropy.
                converged: bool, if True, the calculation is converged.
        """
        _, _, _, E, Ki, Vep, Vee, Se, converged = solver(xp, kpt)
        Vpp = potential_energy_pp(xp, L, rs, kappa=kappa, Gmax=Gmax)
        return E+Vpp, Ki, Vep, Vee, Vpp, Se, converged

    if gamma:
        if mode == 'dev':
            pes = pes_dev_gamma
        elif mode == 'default':
            pes = pes_gamma
        else:
            raise ValueError(f"Invalid mode: {mode}.")
    else:
        if mode == 'dev':
            pes = pes_dev_kpt
        elif mode == 'default':
            pes = pes_kpt
        else:
            raise ValueError(f"Invalid mode: {mode}.")

    if use_jit:
        return jax.jit(pes)
    else:
        return pes
    
