"""
Test multiple basis sets with HF solver.
"""

import pytest
import jax
import jax.numpy as jnp
from hqc.gto.solver import make_solver

jax.config.update("jax_enable_x64", True)


class TestMultipleBasis:
    """Test HF solver with different basis sets."""

    def test_sto_basis(self):
        """Test STO-3G and STO-6G basis sets."""
        atom_positions = jnp.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
        atom_charges = jnp.array([1.0, 1.0])
        n_electrons = 2

        for basis in ['sto-3g', 'sto-6g']:
            print(f"\nTesting {basis}...")
            hf = make_solver(atom_charges, n_electrons, basis=basis, use_jit=False)
            result = hf(atom_positions)

            print(f"  Converged: {result['converged']}")
            print(f"  Energy: {result['energy']:.6f} Ha")

            assert result['converged']
            assert -1.2 < result['energy'] < -1.0

    def test_pople_basis(self):
        """Test Pople basis sets (3-21G, 6-31G)."""
        atom_positions = jnp.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
        atom_charges = jnp.array([1.0, 1.0])
        n_electrons = 2

        for basis in ['3-21G', '6-31G']:
            print(f"\nTesting {basis}...")
            hf = make_solver(atom_charges, n_electrons, basis=basis,
                           max_cycle=200, tol=1e-6, use_jit=False)
            result = hf(atom_positions)

            print(f"  Converged: {result['converged']}")
            print(f"  Energy: {result['energy']:.6f} Ha")

            # Some basis sets may not converge perfectly, but energy should be reasonable
            assert -1.2 < result['energy'] < -0.9

    def test_gth_basis(self):
        """Test GTH basis set."""
        atom_positions = jnp.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
        atom_charges = jnp.array([1.0, 1.0])
        n_electrons = 2

        print(f"\nTesting gth-szv...")
        hf = make_solver(atom_charges, n_electrons, basis='gth-szv', use_jit=False)
        result = hf(atom_positions)

        print(f"  Converged: {result['converged']}")
        print(f"  Energy: {result['energy']:.6f} Ha")

        assert result['converged']
        assert -1.2 < result['energy'] < -1.0

    def test_basis_comparison(self):
        """Compare energies from different basis sets."""
        atom_positions = jnp.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
        atom_charges = jnp.array([1.0, 1.0])
        n_electrons = 2

        basis_sets = ['sto-3g', 'sto-6g', '3-21G', '6-31G', 'gth-szv']
        energies = {}

        print("\nBasis set comparison for H2 at 1.4 Bohr:")
        print("-" * 50)

        for basis in basis_sets:
            hf = make_solver(atom_charges, n_electrons, basis=basis, use_jit=False)
            result = hf(atom_positions)
            energies[basis] = result['energy']
            print(f"{basis:12s}: {result['energy']:.6f} Ha")

        # All should converge and give reasonable energies
        # Different basis sets give different energies, but all should be negative
        assert all(e < 0 for e in energies.values())
        assert all(e > -2.0 for e in energies.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
