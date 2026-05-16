"""
High-level solver interface for non-periodic GTO calculations.

Provides a make_solver interface for Hartree-Fock calculations on isolated molecules.
Optimized for repeated calls with different atom positions but fixed atom types and electron count.
"""

import jax
import jax.numpy as jnp
from typing import Callable, Dict, Any
from hqc.gto.integral import prepare_basis_data, build_integral_matrices_vec
from hqc.gto.scf import make_scf


def make_solver(atom_charges,
                n_electrons: int,
                basis: str = 'gth-szv',
                tol: float = 1e-7,
                max_cycle: int = 100,
                diis: bool = True,
                diis_space: int = 8,
                diis_start_cycle: int = 1,
                diis_damp: float = 0.0,
                use_jit: bool = True) -> Callable:
    """
    Make Hartree-Fock solver function for isolated molecular systems.

    This function performs all preprocessing that doesn't depend on atom positions:
    - Basis set loading and normalization
    - Occupation number determination
    - SCF solver setup

    The returned solver can be called repeatedly with different atom_positions,
    making it efficient for geometry optimization, molecular dynamics, etc.

    Args:
        atom_charges: array of shape (n_atoms,), nuclear charges (fixed)
        n_electrons: int, number of electrons (fixed)
        basis: str, basis set name (e.g., 'gth-szv', 'sto-3g')
        tol: float, SCF convergence tolerance
        max_cycle: int, maximum number of SCF iterations
        diis: bool, whether to use DIIS acceleration
        diis_space: int, DIIS subspace size
        diis_start_cycle: int, cycle to start DIIS
        diis_damp: float, DIIS damping factor
        use_jit: bool, whether to JIT compile the solver

    Returns:
        hf: Hartree-Fock solver function with signature:
            hf(atom_positions) -> Dict[str, Any]

            Args:
                atom_positions: array of shape (n_atoms, 3), atomic positions in Bohr

            Returns a dictionary containing:
                - 'energy': total energy (electronic + nuclear repulsion)
                - 'energy_elec': electronic energy
                - 'energy_nuc': nuclear repulsion energy
                - 'mo_coeff': molecular orbital coefficients, shape (n_ao, n_mo)
                - 'mo_energy': orbital energies in Hartree, shape (n_mo,)
                - 'converged': bool, whether SCF converged
    """
    atom_charges = jnp.asarray(atom_charges)
    n_atoms = len(atom_charges)

    # Precompute basis set data (doesn't depend on positions)
    basis_data = prepare_basis_data(atom_charges, basis)
    n_ao = n_atoms * basis_data['n_ao_per_atom']

    # Precompute occupation numbers (doesn't depend on positions)
    n_alpha = (n_electrons + 1) // 2
    n_beta = n_electrons // 2

    if n_alpha == n_beta:
        # Closed-shell
        occ_numbers = jnp.zeros(n_ao)
        occ_numbers = occ_numbers.at[:n_alpha].set(2.0)
    else:
        # Open-shell (ROHF-like)
        occ_numbers = jnp.zeros(n_ao)
        occ_numbers = occ_numbers.at[:n_beta].set(2.0)
        occ_numbers = occ_numbers.at[n_beta:n_alpha].set(1.0)

    # Create SCF solver (preprocessing)
    scf = make_scf(diis=diis, diis_space=diis_space, diis_start_cycle=diis_start_cycle,
                   diis_damp=diis_damp, tol=tol, max_cycle=max_cycle)

    def hf(atom_positions):
        """
        Perform Hartree-Fock calculation for given atomic positions.

        Args:
            atom_positions: array of shape (n_atoms, 3), atomic positions in Bohr

        Returns:
            Dict[str, Any]: Dictionary containing energy, MO coefficients, etc.
        """
        atom_positions = jnp.asarray(atom_positions)

        # Build integral matrices (depends on positions)
        S, T, V, eri = build_integral_matrices_vec(atom_positions, basis_data)

        # Orthogonalization matrix: S^{-1/2}
        w_s, u_s = jnp.linalg.eigh(S)
        v_ovlp = jnp.dot(u_s, jnp.diag(w_s**(-0.5)))

        # Core Hamiltonian
        Hcore = T + V

        # Initial guess: diagonalize Hcore
        F1_init = jnp.einsum('pq,qr,rs->ps', v_ovlp.T, Hcore, v_ovlp)
        w1_init, c1_init = jnp.linalg.eigh(F1_init)
        mo_coeff_init = jnp.dot(v_ovlp, c1_init)

        # Initial density matrix
        def density_matrix(mo_coeff, w1):
            """Compute density matrix with given occupation numbers."""
            dm_mo = jnp.diag(occ_numbers)
            dm = jnp.einsum('ab,bc,dc->ad', mo_coeff, dm_mo, mo_coeff)
            return dm

        dm_init = density_matrix(mo_coeff_init, w1_init)

        # Hartree (Coulomb) function
        def hartree_fn(dm):
            """Compute Hartree matrix J."""
            J = jnp.einsum('rs,prsq->pq', dm, eri)
            return J

        # Exchange function (for Hartree-Fock)
        def exchange_fn(dm):
            """Compute exchange energy and matrix."""
            K = jnp.einsum('rs,pqsr->pq', dm, eri)
            Vx = -0.5 * K
            Ex = 0.5 * jnp.einsum('pq,qp', Vx, dm).real
            return Ex, Vx

        # DIIS error vector function
        def errvec_sdf_fn(dm, F):
            """Compute DIIS error vector: S@dm@F - F@dm@S."""
            return S @ dm @ F - F @ dm @ S

        # Run SCF
        mo_coeff, mo_energy, E_elec, converged = scf(
            v_ovlp, Hcore, dm_init,
            hartree_fn, exchange_fn,
            density_matrix, errvec_sdf_fn
        )

        # Nuclear repulsion energy
        E_nuc = 0.0
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                r_ij = jnp.linalg.norm(atom_positions[i] - atom_positions[j])
                E_nuc += atom_charges[i] * atom_charges[j] / r_ij

        E_total = E_elec + E_nuc

        return {
            'energy': E_total,
            'energy_elec': E_elec,
            'energy_nuc': E_nuc,
            'mo_coeff': mo_coeff,
            'mo_energy': mo_energy,
            'converged': converged
        }

    # Return JIT-compiled or regular function
    return jax.jit(hf) if use_jit else hf
