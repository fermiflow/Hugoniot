"""
Test SCF solver against PySCF reference calculations.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from pyscf import gto, scf
from hqc.gto.solver import make_solver

jax.config.update("jax_enable_x64", True)


class TestSCF:
    """Test SCF solver against PySCF."""

    def test_h_atom(self):
        """Test single H atom (no electron-electron interaction)."""
        # Single H atom
        atom_positions = jnp.array([[0.0, 0.0, 0.0]])
        atom_charges = jnp.array([1.0])
        n_electrons = 1

        # Create solver
        hf = make_solver(atom_charges, n_electrons, basis='gth-szv', diis=False, max_cycle=50, use_jit=False)
        result = hf(atom_positions)

        print(f"\nH atom:")
        print(f"  Converged: {result['converged']}")
        print(f"  E_total: {result['energy']:.6f} Ha")
        print(f"  E_elec: {result['energy_elec']:.6f} Ha")
        print(f"  E_nuc: {result['energy_nuc']:.6f} Ha")
        print(f"  MO energy: {result['mo_energy'][0]:.6f} Ha")

        # For single H atom, should converge
        assert result['converged']
        # Energy should be negative
        assert result['energy'] < 0
        # No nuclear repulsion
        assert jnp.abs(result['energy_nuc']) < 1e-10

    def test_h2_molecule(self):
        """Test H2 molecule and compare with PySCF."""
        # H2 molecule at equilibrium distance (1.4 Bohr)
        atom_positions = jnp.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
        atom_charges = jnp.array([1.0, 1.0])
        n_electrons = 2

        # HQC calculation
        hf = make_solver(atom_charges, n_electrons, basis='gth-szv', diis=True, max_cycle=50, use_jit=False)
        result = hf(atom_positions)

        print(f"\nH2 molecule (HQC):")
        print(f"  Converged: {result['converged']}")
        print(f"  E_total: {result['energy']:.6f} Ha")
        print(f"  E_elec: {result['energy_elec']:.6f} Ha")
        print(f"  E_nuc: {result['energy_nuc']:.6f} Ha")
        print(f"  MO energies: {result['mo_energy'][:2]}")

        # PySCF reference calculation
        mol = gto.M(atom='H 0 0 0; H 1.4 0 0', basis='gth-szv', unit='B')
        mf = scf.RHF(mol)
        mf.verbose = 0
        mf.kernel()

        print(f"\nH2 molecule (PySCF):")
        print(f"  Converged: {mf.converged}")
        print(f"  E_total: {mf.e_tot:.6f} Ha")
        print(f"  E_nuc: {mf.energy_nuc():.6f} Ha")
        print(f"  MO energies: {mf.mo_energy[:2]}")

        # Check convergence
        assert result['converged']
        assert mf.converged

        # Compare energies (should be close)
        print(f"\nEnergy difference: {abs(result['energy'] - mf.e_tot):.8f} Ha")
        assert jnp.abs(result['energy'] - mf.e_tot) < 1e-5, \
            f"Energy mismatch: HQC={result['energy']:.6f}, PySCF={mf.e_tot:.6f}"

        # Compare nuclear repulsion
        assert jnp.abs(result['energy_nuc'] - mf.energy_nuc()) < 1e-10

    def test_diis_convergence(self):
        """Test that DIIS converges for H2."""
        atom_positions = jnp.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
        atom_charges = jnp.array([1.0, 1.0])
        n_electrons = 2

        # DIIS iteration should converge
        hf = make_solver(atom_charges, n_electrons, basis='gth-szv', diis=True, max_cycle=100, use_jit=False)
        result = hf(atom_positions)

        print(f"\nDIIS: E={result['energy']:.6f}, converged={result['converged']}")

        # DIIS should converge
        assert result['converged']
        # Energy should be reasonable
        assert result['energy'] < -1.0  # H2 binding energy


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
